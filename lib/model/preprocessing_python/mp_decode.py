"""Multiprocess PyAV video decode.

This module is deliberately lightweight — it imports only ``av`` and ``numpy``
(no torch / no server packages).  On Windows ``spawn`` the child processes
import *this* module to locate the worker function, so keeping it torch-free
keeps process startup fast (~tens of ms instead of seconds).

The strategy: split the timeline into contiguous chunks and decode each chunk
in its own process.  Because each process has its own GIL, the decoders run in
true parallel AND don't contend with the parent's asyncio inference pipeline —
the thing that throttles in-process (threaded) PyAV decode.  Only the small,
already-downscaled frames cross the process boundary.
"""

import math
import multiprocessing as mp
from typing import Optional

# Frames per dense clip handed to the parent. Purely a transport size.
DENSE_CLIP_FRAMES = 100


def _decode_chunk_worker(path, start_idx, count, frame_interval, max_long_edge,
                         use_timestamps, out_queue, options=None):
    """Decode one contiguous chunk and push (out_idx, HWC uint8 ndarray) tuples.

    Runs in its own process.  Emits the frame nearest each target timestamp
    (first frame whose time is >= target), giving true per-interval sampling.

    ``options`` (all optional):

    * ``skip_nonref``: let the decoder drop frames nothing else references.
      The frames it returns are unchanged, but a target can land on the next
      decodable frame instead of its own.
    * ``dense_sizes``: ``[(width, height), ...]``. Every frame of this chunk's
      time range is also resized to each size and pushed as
      ``("__dense__", segment, size_index, [n, H, W, 3] uint8)``, followed by
      ``("__segment__", segment, frame_count)``. The chunk owns the frames with
      ``segment_start <= time < segment_end``, so chunks partition the video.
    * ``dense_crop``: ``(x, y, width, height)`` applied before the dense resize.
    * ``segment``, ``segment_end``: this chunk's index and end time (``None``
      for the last chunk).
    """
    options = options or {}
    dense_sizes = list(options.get("dense_sizes") or [])
    try:
        import av

        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if options.get("skip_nonref") and not dense_sizes:
                stream.codec_context.skip_frame = "NONREF"
            time_base = stream.time_base
            average_rate = float(stream.average_rate) if stream.average_rate else 30.0
            tol = 0.5 / average_rate if average_rate > 0 else 0.02

            start_t = start_idx * frame_interval
            container.seek(int(start_t / time_base), stream=stream)

            dense = _DenseEmitter(options, start_t, stream, out_queue) if dense_sizes else None
            produced = 0
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                ft = float(frame.pts * time_base)
                if dense is not None and not dense.finished:
                    dense.add(frame, ft)
                if produced < count:
                    target = (start_idx + produced) * frame_interval
                    if ft + tol >= target:
                        if max_long_edge > 0 and max(frame.width, frame.height) > max_long_edge:
                            scale = max_long_edge / max(frame.width, frame.height)
                            tw = max(2, round(frame.width * scale))
                            th = max(2, round(frame.height * scale))
                            arr = frame.reformat(width=tw, height=th, format="rgb24").to_ndarray()
                        else:
                            arr = frame.to_ndarray(format="rgb24")
                        out_idx = target if use_timestamps else (start_idx + produced)
                        out_queue.put((out_idx, arr))
                        produced += 1
                if produced >= count and (dense is None or dense.finished):
                    break
            if dense is not None:
                dense.close()
        finally:
            container.close()
    except Exception as exc:  # surface to the parent as a sentinel payload
        try:
            out_queue.put(("__error__", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        out_queue.put(None)  # done sentinel


class _DenseEmitter:
    """Resizes every frame of one segment for the every-frame consumers."""

    def __init__(self, options, start_t, stream, out_queue):
        from av.video.reformatter import Interpolation, VideoReformatter

        self._interpolation = Interpolation.BICUBIC
        # One scaler for the whole segment. ``frame.reformat`` builds a new one
        # for every frame, which costs about as much as the scale itself.
        self._reformatter = VideoReformatter()
        self._sizes = [(int(w), int(h)) for w, h in options["dense_sizes"]]
        self._crop = options.get("dense_crop")
        self._segment = int(options.get("segment", 0))
        self._start = start_t if self._segment > 0 else None
        self._end = options.get("segment_end")
        self._first_pts = stream.start_time
        self._out = out_queue
        self._clips = [[] for _ in self._sizes]
        self._seen_any = False
        self.count = 0
        self.finished = False

    def add(self, frame, ft):
        if not self._seen_any:
            self._seen_any = True
            # The seek must land at or before the segment's first frame, or the
            # frames in between would silently be missing from the video.
            if (self._start is not None and ft > self._start + 1e-9
                    and not (self._first_pts is not None and frame.pts <= self._first_pts)):
                raise RuntimeError(
                    f"seek for segment {self._segment} landed at {ft:.3f}s, "
                    f"after its start {self._start:.3f}s"
                )
        if self._start is not None and ft < self._start - 1e-9:
            return  # belongs to the previous segment
        if self._end is not None and ft >= self._end - 1e-9:
            self.finished = True
            return
        if self._crop:
            import av
            import numpy as np

            # PyAV cannot crop, so round-trip the frame through RGB.
            x, y, width, height = self._crop
            region = frame.to_ndarray(format="rgb24")[y:y + height, x:x + width]
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(region), format="rgb24")
        for index, (width, height) in enumerate(self._sizes):
            self._clips[index].append(
                self._reformatter.reformat(frame, width=width, height=height, format="rgb24",
                                           interpolation=self._interpolation).to_ndarray()
            )
        self.count += 1
        if len(self._clips[0]) >= DENSE_CLIP_FRAMES:
            self._flush()

    def _flush(self):
        import numpy as np

        for index, clip in enumerate(self._clips):
            if clip:
                self._out.put(("__dense__", self._segment, index, np.stack(clip)))
                self._clips[index] = []

    def close(self):
        self._flush()
        self.finished = True
        self._out.put(("__segment__", self._segment, self.count))


def probe_duration_seconds(path) -> float:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        if container.duration:
            return container.duration / av.time_base
    return 0.0


def iter_parallel_frames(path, frame_interval, max_long_edge, decode_workers,
                         use_timestamps, queue_depth: int = 64, *, skip_nonref: bool = False,
                         dense_sizes=None, dense_crop=None, on_dense=None, on_segment=None,
                         emit_interval: bool = True):
    """Yield (out_idx, HWC uint8 ndarray) from parallel decode worker processes.

    Frames arrive out of timestamp order; each carries its target time/index.

    With ``dense_sizes`` every frame is also delivered, resized to each size,
    through ``on_dense(segment, size_index, frames)``; ``on_segment(segment,
    count)`` reports each segment's frame count once it is complete. Segments
    are the workers' time ranges, numbered in video order. ``emit_interval``
    False decodes for the dense consumers only and yields nothing.
    """
    duration = probe_duration_seconds(path)
    if duration <= 0 or not frame_interval or frame_interval <= 0:
        if dense_sizes:
            raise RuntimeError(f"cannot decode every frame of '{path}': its duration is unknown")
        return

    n_targets = int(duration / frame_interval) + 1
    n_workers = max(1, min(int(decode_workers), n_targets))
    per = math.ceil(n_targets / n_workers)

    ctx = mp.get_context("spawn")
    out_queue: mp.Queue = ctx.Queue(maxsize=queue_depth)

    chunks = [(i * per, min(per, n_targets - i * per)) for i in range(n_workers)]
    chunks = [(start_idx, count) for start_idx, count in chunks if count > 0]

    procs = []
    for segment, (start_idx, count) in enumerate(chunks):
        options = {"skip_nonref": bool(skip_nonref)}
        if dense_sizes:
            last = segment == len(chunks) - 1
            options.update(
                dense_sizes=[tuple(size) for size in dense_sizes],
                dense_crop=tuple(dense_crop) if dense_crop else None,
                segment=segment,
                segment_end=None if last else (start_idx + count) * frame_interval,
            )
        proc = ctx.Process(
            target=_decode_chunk_worker,
            args=(str(path), start_idx, count if emit_interval else 0, frame_interval,
                  int(max_long_edge or 0), bool(use_timestamps), out_queue, options),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    finished = 0
    error = None
    try:
        while finished < len(procs):
            item = out_queue.get()
            if item is None:
                finished += 1
                continue
            if isinstance(item, tuple) and isinstance(item[0], str):
                if item[0] == "__error__":
                    error = RuntimeError(f"parallel decode worker failed: {item[1]}")
                    break
                if item[0] == "__dense__":
                    if on_dense is not None:
                        on_dense(item[1], item[2], item[3])
                    continue
                if item[0] == "__segment__":
                    if on_segment is not None:
                        on_segment(item[1], item[2])
                    continue
            yield item
        if error is not None:
            raise error
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=2.0)
