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

import itertools
import math
import multiprocessing as mp
from typing import Optional

# Frames per dense clip handed to the parent. Purely a transport size.
DENSE_CLIP_FRAMES = 100

# Packets the decoder may reject in one chunk before the chunk fails. Only a
# chunk that feeds the every-frame consumers gets past rejected packets at all
# (see _DenseEmitter); more than this means the file is unreadable, not damaged.
MAX_REJECTED_PACKETS = 32

# Frames at either end of a video that no decoder can produce are left out of
# the every-frame delivery when they span at most this long: at the start,
# frames of a cut file that reference a keyframe before its first; at the end,
# an incomplete last frame. Leaving them out shifts no other frame, and every
# player skips them too. A longer gap is a cut or damaged file and fails it.
EDGE_GAP_SECONDS = 0.5

# Frames decoded to check that a video's frames come out in display order.
ORDER_PROBE_FRAMES = 48


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
      A segment that cannot deliver every one of its frames pushes
      ``("__dense_failed__", segment, message)`` instead of its count, and the
      chunk goes on decoding for its interval frames.
    * ``dense_crop``: ``(x, y, width, height)`` applied before the dense resize.
    * ``segment``, ``segment_end``: this chunk's index and end time (``None``
      for the last chunk). ``segment_end`` is ``None`` and ``segment`` 0 for a
      chunk that delivers every frame of the video on its own.
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
            dense = _DenseEmitter(options, start_t, stream, out_queue) if dense_sizes else None
            exact = dense is not None and dense.needs_exact_start
            produced = 0
            for frame in _decode_from(container, stream, start_t, exact, dense):
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


def _decode_from(container, stream, start_t, exact, tally=None):
    """Seek to ``start_t`` and return the decoded frames from there.

    A seek lands on the keyframe whose *decode* time is at or before the
    target, and that keyframe can still be displayed after it: with B-frames a
    keyframe is displayed later than it is decoded, and in an open GOP the
    frames displayed just before it reference the previous GOP, so the decoder
    drops them after the seek. With ``exact``, a seek whose first frame is
    after ``start_t``, or that yields no frame at all, is retried from further
    back until it is not. That costs decoding from an earlier keyframe, and
    only for the seeks that land late.

    ``tally`` sees every packet read and every packet the decoder rejects; see
    :func:`_decoded_frames`. With one, a seek is also retried when a packet
    displayed from ``start_t`` on was read but its frame did not come out
    before the first frame from ``start_t``: a seek that lands mid-GOP, as in
    MPEG-TS, can decode a frame before ``start_t`` and still lose the ones
    after it.
    """
    time_base = stream.time_base
    target = start_t
    backoff = 1.0
    while True:
        container.seek(int(target / time_base), stream=stream)
        if tally is not None:
            tally.restart()
        frames = _decoded_frames(container, stream, tally)
        if not exact or target <= 0:
            return frames
        first = next(frames, None)
        # No frame at all is a seek that landed too late as well: formats whose
        # index is not keyframe-accurate, such as MPEG-TS, can land mid-GOP
        # after the last keyframe, and the decoder then drops everything to EOF.
        if first is not None and not _starts_after(first, float(first.pts * time_base), start_t, stream):
            buffered = [first]
            if tally is not None:
                while float(buffered[-1].pts * time_base) < start_t - 1e-9:
                    following = next(frames, None)
                    if following is None:
                        break
                    buffered.append(following)
            if tally is None or not tally.missed_before(buffered[-1].pts):
                return itertools.chain(buffered, frames)
        target = max(0.0, start_t - backoff)
        backoff *= 2


def _decoded_frames(container, stream, tally):
    """The frames of ``stream`` with a timestamp, in display order.

    Without a ``tally`` this is ``container.decode``: a packet the decoder
    rejects raises. With one, each packet is counted through
    ``tally.packet(packet)``, a rejected packet goes to
    ``tally.rejected(packet, exc)`` instead of ending the decode, and a frame
    without a timestamp, which is dropped either way, is reported through
    ``tally.untimed_frame()``.
    """
    import av

    for packet in container.demux(stream):
        if tally is not None:
            tally.packet(packet)
        try:
            frames = packet.decode()
        except av.error.InvalidDataError as exc:
            if tally is None:
                raise
            tally.rejected(packet, exc)
            continue
        for frame in frames:
            if frame.pts is not None:
                yield frame
            elif tally is not None:
                tally.untimed_frame()


def _starts_after(frame, ft, start_t, stream):
    """Whether decoding from ``frame`` misses frames displayed from ``start_t`` on.

    The stream's own first frame misses nothing, wherever it is displayed.
    """
    first_pts = stream.start_time
    return ft > start_t + 1e-9 and not (first_pts is not None and frame.pts <= first_pts)


class _DenseEmitter:
    """Resizes every frame of one segment for the every-frame consumers.

    The consumers number frames by counting them, so a segment is delivered
    complete or not at all. The emitter fails the segment when the decoder
    rejects a packet, or when a packet displayed within the segment has no
    frame with its timestamp. A damaged frame does not always make the
    decoder raise: the frames that reference it can be dropped without an
    error. Timestamps rather than counts are compared because some formats
    store packets that never become frames of their own, such as VP8's hidden
    alt-ref frames, which share the timestamp of the frame after them.

    Frames must come out in display order for the segments to split the video
    correctly. :func:`iter_parallel_frames` checks that before it splits the
    video, and the emitter fails a segment whose frames go backwards anyway.
    """

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
        # False when the frames' timestamps do not follow display order, which
        # only one segment covering the whole video can cope with.
        self._ordered = bool(options.get("ordered", True))
        # The video's first decodable frame: packets displayed shortly before
        # it belong to frames no decoder can produce. See EDGE_GAP_SECONDS.
        self._first_pts = options.get("first_pts")
        self._stream = stream
        self._out = out_queue
        self._clips = [[] for _ in self._sizes]
        self._seen_any = False
        self._packet_pts = set()
        self._frame_pts = set()
        self._last_ft = None
        self._rejected = 0
        self.failed = False
        self.count = 0
        self.finished = False

    @property
    def needs_exact_start(self):
        """Whether decoding must start at or before this segment's first frame."""
        return self._start is not None

    def _in_segment(self, t):
        if self._start is not None and t < self._start - 1e-9:
            return False
        return self._end is None or t < self._end - 1e-9

    def restart(self):
        """Forget the packets of a seek that is about to be retried."""
        self._packet_pts.clear()

    def missed_before(self, pts):
        """Whether a packet of this segment displayed before ``pts`` was read."""
        return any(packet_pts < pts for packet_pts in self._packet_pts)

    def packet(self, packet):
        if self.finished or packet.size == 0 or packet.is_discard:
            return  # a flush packet, or one the decoder drops by design
        # A packet without a timestamp cannot be placed in a segment. Such
        # packets are the undecodable leading frames of a cut MKV, for example;
        # a frame that does come out without a timestamp fails the segment in
        # untimed_frame instead.
        if packet.pts is None or self._in_leading_gap(packet.pts):
            return
        if self._in_segment(float(packet.pts * self._stream.time_base)):
            self._packet_pts.add(packet.pts)

    def untimed_frame(self):
        if not self.finished:
            self._fail("a frame has no timestamp, so it cannot be placed in the video")

    def _in_leading_gap(self, pts):
        return (self._first_pts is not None and pts < self._first_pts
                and float((self._first_pts - pts) * self._stream.time_base) <= EDGE_GAP_SECONDS)

    def rejected(self, packet, exc):
        self._rejected += 1
        if self._rejected > MAX_REJECTED_PACKETS:
            raise exc
        if not self.finished:
            at = "" if packet.pts is None else f" at or before {float(packet.pts * self._stream.time_base):.3f}s"
            self._fail(f"the decoder rejected a packet{at}: {type(exc).__name__}: {exc}")

    def _fail(self, reason):
        self.failed = True
        self.finished = True
        self._clips = [[] for _ in self._sizes]
        self._out.put(("__dense_failed__", self._segment, f"segment {self._segment}: {reason}"))

    def add(self, frame, ft):
        if not self._seen_any:
            self._seen_any = True
            # _decode_from starts at or before the segment's first frame. Check
            # anyway: frames missing here would silently shift the whole video.
            if self._start is not None and _starts_after(frame, ft, self._start, self._stream):
                self._fail(f"the seek landed at {ft:.3f}s, after the segment start {self._start:.3f}s")
                return
        if self._start is not None and ft < self._start - 1e-9:
            return  # belongs to the previous segment
        if self._ordered and self._last_ft is not None and ft < self._last_ft - 1e-9:
            self._fail(f"the frame at {ft:.3f}s came out after the frame at {self._last_ft:.3f}s")
            return
        self._last_ft = ft
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
        self._frame_pts.add(frame.pts)
        self.count += 1
        if len(self._clips[0]) >= DENSE_CLIP_FRAMES:
            self._flush()

    def _is_trailing_gap(self, missing):
        """Whether ``missing`` are only frames after the video's last decoded
        one, spanning at most ``EDGE_GAP_SECONDS``."""
        if self._end is not None or not self._frame_pts:
            return False
        last = max(self._frame_pts)
        return missing[0] > last and float((missing[-1] - last) * self._stream.time_base) <= EDGE_GAP_SECONDS

    def _flush(self):
        import numpy as np

        for index, clip in enumerate(self._clips):
            if clip:
                self._out.put(("__dense__", self._segment, index, np.stack(clip)))
                self._clips[index] = []

    def close(self):
        if self.failed:
            return
        missing = sorted(self._packet_pts - self._frame_pts)
        if missing and not self._is_trailing_gap(missing):
            time_base = self._stream.time_base
            self._fail(f"{len(missing)} frames between {float(missing[0] * time_base):.3f}s and "
                       f"{float(missing[-1] * time_base):.3f}s did not decode")
            return
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


def probe_frame_order(path, frames: int = ORDER_PROBE_FRAMES):
    """``(ordered, first_pts)`` from decoding the first ``frames`` frames.

    ``ordered`` is whether they come out of the decoder with increasing
    timestamps; not so for AVI with B-frames, whose timestamps follow the
    packets' decode order. ``first_pts`` is the earliest timestamp among them,
    the video's first decodable frame, or ``None``. A video that cannot be
    decoded that far counts as ordered: the decode itself reports what is
    wrong with it.
    """
    import av

    seen = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                seen.append(frame.pts)
                if len(seen) >= frames:
                    break
    except av.error.FFmpegError:
        pass
    ordered = all(earlier <= later for earlier, later in zip(seen, seen[1:]))
    return ordered, (min(seen) if seen else None)


def iter_parallel_frames(path, frame_interval, max_long_edge, decode_workers,
                         use_timestamps, queue_depth: int = 64, *, skip_nonref: bool = False,
                         dense_sizes=None, dense_crop=None, on_dense=None, on_segment=None,
                         on_dense_failed=None, emit_interval: bool = True):
    """Yield (out_idx, HWC uint8 ndarray) from parallel decode worker processes.

    Frames arrive out of timestamp order; each carries its target time/index.

    With ``dense_sizes`` every frame is also delivered, resized to each size,
    through ``on_dense(segment, size_index, frames)``; ``on_segment(segment,
    count)`` reports each segment's frame count once it is complete. Segments
    are the workers' time ranges, numbered in video order. ``emit_interval``
    False decodes for the dense consumers only and yields nothing.

    A segment that cannot deliver all of its frames is reported through
    ``on_dense_failed(segment, message)`` and the interval frames keep coming;
    without that callback it fails the decode. With ``emit_interval`` False the
    decode then stops, since nothing is left to decode for.
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

    # The segments split the video by timestamp, which is only right when the
    # frames come out in display order. When they do not, such as AVI with
    # B-frames, one chunk delivers every frame and the rest only sample.
    split_dense, first_pts = probe_frame_order(path) if dense_sizes else (False, None)

    procs = []
    for segment, (start_idx, count) in enumerate(chunks):
        options = {"skip_nonref": bool(skip_nonref)}
        if dense_sizes and (split_dense or segment == 0):
            last = segment == len(chunks) - 1 or not split_dense
            options.update(
                dense_sizes=[tuple(size) for size in dense_sizes],
                dense_crop=tuple(dense_crop) if dense_crop else None,
                segment=segment,
                segment_end=None if last else (start_idx + count) * frame_interval,
                ordered=split_dense,
                first_pts=first_pts,
            )
        elif not emit_interval:
            continue  # nothing to decode for
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
                if item[0] == "__dense_failed__":
                    if on_dense_failed is None:
                        error = RuntimeError(f"parallel decode worker failed: {item[2]}")
                        break
                    on_dense_failed(item[1], item[2])
                    if not emit_interval:
                        break
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
