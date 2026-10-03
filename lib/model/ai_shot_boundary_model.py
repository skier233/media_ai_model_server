"""Asset-scope temporal segmentation (shot-boundary detection).

Unlike every other AI model in this server, a shot-boundary model cannot work
from frames sampled at the request's interval: it needs dense, *adjacent*
frames across the whole asset, at its own resolution. It therefore runs at
``asset`` scope and sets ``requires_every_frame``. It does not decode the
video itself: the shared video preprocessor decodes every frame once for the
request and hands them over as a ``DenseFrameStream`` (see
lib/model/dense_frames.py).

The artifact contract (see docs/asset-scope-models.md) is a self-contained ``.pt2``/``.pt`` module::

    forward(clip: [1, T, 3, H, W]) -> (shot_logits, intra_logits, inter_logits)

with ``shot_logits[..., :-1]`` a per-query distribution over the clip-relative
end frame, and the other two heads per-query class distributions whose last
class is the "no object" slot.  The int -> label-name maps come from a
``<model_file_name>.labels.json`` sidecar next to the artifact, mirroring how
tagging models read ``<model_file_name>.tags.txt``. The clip geometry ``H``/``W``
comes from the model yaml or, when it leaves it out, from the same sidecar, and
must agree when both give it; ``T`` is the sidecar's ``window_frames`` unless the
yaml's ``window_frames`` overrides it. The clip is always ImageNet-normalised.

Everything outside the artifact — windowing, the greedy query walk, context
pruning, cross-window merging and boundary normalization — is generic and
model-independent.
"""

import asyncio
import json
import queue
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from lib.model.ai_model import AIModel
from lib.model.dense_frames import DenseFrameStream, DenseStreamError, dense_stream_for
from lib.model.preprocessing_python.ffmpeg_pipe import probe_video, vr_crop_rect
from lib.model.preprocessing_python.image_preprocessing import (
    _validate_local_video_source,
    preprocess_video_mp_pyav,
)
from lib.pipeline.preprocess_spec import PreprocessSpec, apply_spec_batch

# NORMALIZATION_PRESETS key of the ImageNet mean/std the artifacts are trained with.
IMAGENET_NORMALIZATION = 0


class AIShotBoundaryModel(AIModel):
    """Runs a shot-boundary artifact over a whole video and emits segments."""

    # Consumes the video path and decodes every frame itself, so it runs only
    # in the asset-scope stage and only when a request names it. See
    # lib/model/whole_asset.py.
    whole_asset_model = True

    # Tells the shared video preprocessor to decode every frame of the video
    # when this model runs, and to deliver them at ``dense_frame_size()``.
    requires_every_frame = True

    MODES = ("default", "clean_shot")

    def __init__(self, configValues):
        super().__init__(configValues, keep_on_device=False)

        # One video at a time: a single request already saturates the model,
        # and batching whole videos would multiply peak memory for no gain.
        self.max_batch_size = 1
        self.instance_count = 1
        self.fill_to_batch = False

        self.window_frames = int(configValues.get("window_frames", 0) or 0)
        self.context_frames = int(configValues.get("context_frames", 0) or 0)
        self.mode = str(configValues.get("mode", "default") or "default").lower()
        self.general_intra_label = str(configValues.get("general_intra_label", "general"))
        self.new_start_inter_label = str(configValues.get("new_start_inter_label", "new_start"))
        self.preprocess_config = configValues.get("preprocess_config", None)

        # Decoding dominates runtime on long, high-resolution sources. See
        # ffmpeg_pipe._ladder for the measured backend comparison.
        decode = configValues.get("decode", None) or {}
        self.decode_backend = str(decode.get("backend", "auto"))
        self.decode_quality = str(decode.get("quality", "exact"))
        self.decode_device_index = decode.get("device_index", None)

        self.labels: Optional[dict] = None
        self._spec: Optional[PreprocessSpec] = None
        self._intra_names: Dict[int, str] = {}
        self._inter_names: Dict[int, str] = {}
        self._general_intra_index: Optional[int] = None
        self._new_start_inter_index: Optional[int] = None

    @property
    def _name(self) -> str:
        """Display name: the config yaml's stem once the manager attaches it, else the artifact."""
        return getattr(self, "config_name", None) or self.model_file_name

    def dense_frame_size(self) -> tuple:
        """``(width, height)`` every frame is resized to for this model."""
        if self._spec is None:
            self._load_labels()
        return self._spec.width, self._spec.height

    # ── lifecycle ────────────────────────────────────────────────────────

    async def load(self):
        # Everything that can be checked without the artifact is checked
        # first, so a misconfigured model fails before anything is loaded.
        if self.model_license_name is not None:
            # The licensed runner's raw multi-output call takes no precision
            # argument, and this model must run in float32 (see _run_window).
            raise ValueError(
                f"{self._name}: shot-boundary artifacts must be unencrypted .pt2 or .pt files"
            )
        categories = self.model_category if isinstance(self.model_category, list) else [self.model_category]
        categories = [str(category).strip() for category in categories if str(category or "").strip()]
        if len(categories) != 1:
            # The result is emitted under the model's one output, its category.
            raise ValueError(
                f"{self._name}: a shot-boundary model needs exactly one model_category, "
                f"not {categories or 'none'}"
            )
        if self.labels is None:
            self._load_labels()
        await super().load()
        self.logger.info(
            f"Shot-boundary model ready: window={self.window_frames} frames, "
            f"clip={self._spec.width}x{self._spec.height}, context={self.context_frames}, mode={self.mode}"
        )

    def _load_labels(self):
        """Read and check the label sidecar, and store what it gives only if
        every check passes."""
        if self.mode not in self.MODES:
            raise ValueError(f"{self._name}: mode must be one of {', '.join(self.MODES)}, not '{self.mode}'")
        sidecar = Path(f"./models/{self.model_file_name}.labels.json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"Shot-boundary label sidecar not found: {sidecar}. "
                f"It ships alongside the {self.model_file_name} artifact."
            )
        with open(sidecar, "r", encoding="utf-8") as handle:
            labels = json.load(handle)

        intra_names = {int(k): str(v) for k, v in (labels.get("intra") or {}).items()}
        inter_names = {int(k): str(v) for k, v in (labels.get("inter") or {}).items()}
        intra_index = labels.get("intra_name_to_index") or {}
        inter_index = labels.get("inter_name_to_index") or {}
        general_intra_index = (
            int(intra_index[self.general_intra_label]) if self.general_intra_label in intra_index else None
        )
        if self.new_start_inter_label not in inter_index:
            # Without it a shot that continues across a window seam is split in two.
            raise ValueError(
                f"{self._name}: inter label '{self.new_start_inter_label}' is not in {sidecar} "
                f"(set new_start_inter_label in the model yaml)"
            )
        new_start_inter_index = int(inter_index[self.new_start_inter_label])

        # The sidecar carries the artifact's trained window; the yaml may override.
        window_frames = self.window_frames if self.window_frames > 0 else int(labels.get("window_frames", 0) or 0)
        if window_frames <= 0:
            raise ValueError(
                f"{self._name}: window_frames is unknown "
                f"(set it in the model yaml or in {sidecar})"
            )
        if self.mode == "clean_shot" and general_intra_index is None:
            raise ValueError(
                f"{self._name}: mode=clean_shot needs intra label "
                f"'{self.general_intra_label}' in {sidecar}"
            )
        spec = self._clip_spec(labels, sidecar)

        self.labels = labels
        self._intra_names = intra_names
        self._inter_names = inter_names
        self._general_intra_index = general_intra_index
        self._new_start_inter_index = new_start_inter_index
        self.window_frames = window_frames
        self._spec = spec

    def _clip_spec(self, labels: dict, sidecar: Path) -> PreprocessSpec:
        """The clip geometry and normalisation the artifact was trained with.

        The geometry comes from the yaml's ``preprocess_config`` or, when that
        leaves it out, from the sidecar the exporter wrote; when both give it
        they must agree. Nothing falls back to a default size: frames at the
        wrong geometry still produce plausible shots, just wrong ones.
        """
        config = self.preprocess_config if isinstance(self.preprocess_config, dict) else {}

        def size(width, height, where):
            if width is None and height is None:
                return None
            if width is None or height is None or int(width) <= 0 or int(height) <= 0:
                raise ValueError(
                    f"{self._name}: {where} must give both a positive width and height, not {width}x{height}"
                )
            return int(width), int(height)

        from_yaml = size(config.get("width"), config.get("height"), "preprocess_config")
        from_sidecar = size(labels.get("frame_width"), labels.get("frame_height"), f"{sidecar} (frame_width/frame_height)")
        if from_yaml and from_sidecar and from_yaml != from_sidecar:
            raise ValueError(
                f"{self._name}: preprocess_config gives {from_yaml[0]}x{from_yaml[1]} but {sidecar} "
                f"gives {from_sidecar[0]}x{from_sidecar[1]}; the artifact was trained at one geometry"
            )
        width, height = from_yaml or from_sidecar or (0, 0)
        if not width:
            raise ValueError(
                f"{self._name}: the clip geometry is unknown "
                f"(set preprocess_config width/height, or frame_width/frame_height in {sidecar})"
            )
        normalization = config.get("normalization", IMAGENET_NORMALIZATION)
        if normalization != IMAGENET_NORMALIZATION:
            raise ValueError(
                f"{self._name}: shot-boundary clips are ImageNet-normalised "
                f"(normalization: {IMAGENET_NORMALIZATION}), not normalization: {normalization}"
            )
        return PreprocessSpec(
            width=width,
            height=height,
            normalization=IMAGENET_NORMALIZATION,
            device=str(config.get("device", "cpu")),
            # Inference runs in float32; see _run_window.
            precision="float32",
        )

    # ── worker ───────────────────────────────────────────────────────────

    async def worker_function(self, data):
        loop = asyncio.get_running_loop()
        for item in data:
            root = getattr(item.item_future, "root_future", item.item_future)
            if getattr(root, "future", None) is not None and root.future.done():
                # The request already finished (it timed out or was cancelled),
                # so nobody would read a result that takes minutes to produce.
                self.logger.info(f"{self._name}: request already finished; skipping its analysis")
                continue
            try:
                video_path = _validate_local_video_source(item.item_future[item.input_names[0]])
                vr_video = False
                if len(item.input_names) > 1:
                    raw_vr = item.item_future[item.input_names[1]]
                    vr_video = bool(raw_vr) if raw_vr is not None else False

                stream = self._shared_stream(root)
                try:
                    # Inference takes minutes, not milliseconds. Running it
                    # inline would block the event loop and stall every other
                    # model — including the tagging pass over this same video.
                    arguments = (video_path, vr_video) if stream is None else (video_path, vr_video, stream)
                    result = await loop.run_in_executor(None, self._analyze, *arguments)
                finally:
                    if stream is not None:
                        # Whatever happened, the shared decode must not keep
                        # waiting for this model to read.
                        stream.abandon()
                await item.item_future.set_data(item.output_names[0], result)
            except Exception as exception:  # noqa: BLE001 - reported per item
                # The item's future is the request's own. Failing it would throw
                # away the frame results of a request that also asked for
                # tagging, so the asset stage reports the error instead.
                self.logger.error(f"{self._name}: shot-boundary analysis failed: {exception}", exc_info=True)
                await item.item_future.set_data(item.output_names[0], exception)

    def _shared_stream(self, root):
        """This request's every-frame stream from the shared video preprocessor,
        or None when the request did not come through a pipeline (the model is
        then being driven directly, and ``_analyze`` decodes for itself through
        the same shared decode)."""
        pipeline = root["pipeline"] if hasattr(root, "__getitem__") else None
        if pipeline is None:
            return None
        preprocessor = pipeline.get_first_video_preprocessor()
        if preprocessor is None or self not in getattr(preprocessor, "dense_consumers", []):
            raise RuntimeError(
                f"{self._name} needs every frame, but this pipeline has no shared video "
                f"preprocessor to supply them (it needs a dynamic_video_ai stage)"
            )
        width, height = self.dense_frame_size()
        stream = dense_stream_for(root, self._name, width, height)
        stream.attach()
        return stream

    # ── inference ────────────────────────────────────────────────────────

    def _analyze(self, video_path, vr_video: bool, stream=None) -> dict:
        window = self.window_frames
        context = self.context_frames
        stride = window - 2 * context
        if stride <= 0:
            raise ValueError(
                f"context_frames={context} is too large for window_frames={window} "
                f"(stride would be {stride})"
            )

        spec = self._spec
        info = probe_video(video_path)
        duration = float(info.duration or 0.0)
        fps = float(info.fps or 30.0)

        # A VR frame keeps one eye. The crop is taken before the scale and
        # selects the same source pixels as vr_permute does on a decoded frame.
        frame_source = stream if stream is not None else make_video_frame_source(
            video_path,
            decode_size=(spec.width, spec.height),
            crop=vr_crop_rect(info.width, info.height) if vr_video else None,
        )
        analyze_started = time.perf_counter()
        inference_seconds = 0.0
        decode_wait_seconds = 0.0

        # The shared decode runs over contiguous segments of the video in
        # parallel, so frames arrive in order within a segment while segments
        # interleave. Each segment is windowed on its own, exactly as a whole
        # video is: OmniShotCut's reference implementation prepends `context`
        # black frames, then walks windows of `window` frames at `stride`, and
        # the last window is padded. Frames stay uint8 at clip geometry until
        # a window is complete, so memory is under one window per segment
        # regardless of video length or source resolution.
        blank = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        segments: Dict[int, "_Segment"] = {}

        def score(segment: "_Segment", clip, num_pad):
            nonlocal inference_seconds
            ranges, intra, inter = [], [], []
            inference_seconds += self._run_window(
                clip, num_pad, segment.window_index * stride, ranges, intra, inter)
            if num_pad > 0:
                # A padded window can place a shot's end inside its padding.
                # At the end of the video the partition trims that; inside it,
                # those frames belong to the next segment.
                kept = [index for index, frame_range in enumerate(ranges) if frame_range[0] < segment.frames]
                ranges = [[ranges[index][0], min(ranges[index][1], segment.frames)] for index in kept]
                intra = [intra[index] for index in kept]
                inter = [inter[index] for index in kept]
            segment.windows.append((ranges, intra, inter))
            del segment.buffer[:stride]
            segment.window_index += 1

        frames_iterator = iter(frame_source)
        while True:
            waited = time.perf_counter()
            item = next(frames_iterator, None)
            decode_wait_seconds += time.perf_counter() - waited
            if item is None:
                break
            segment = segments.get(item[0])
            if segment is None:
                segment = segments[item[0]] = _Segment([blank] * context)
            for frame in item[1]:
                segment.buffer.append(frame)
                segment.frames += 1
                while len(segment.buffer) >= window:
                    score(segment, segment.buffer[:window], 0)

        # Tail of each segment: pad what is left out to full windows until
        # every real frame has been scored. `len(buffer) > context` is exactly
        # "real frames remain unscored", since buffer holds padded[base:] and
        # padding adds `context` frames. With context frames one padded window
        # is not always enough: its trailing context can hold real frames it
        # does not score.
        for index in sorted(segments):
            segment = segments[index]
            while len(segment.buffer) > context:
                num_pad = window - len(segment.buffer)
                clip = list(segment.buffer)
                if num_pad > 0:
                    clip.extend([blank] * num_pad)
                score(segment, clip[:window], max(0, num_pad))

        real_frames = sum(segment.frames for segment in segments.values())
        if real_frames == 0:
            # An empty result would look like a video with no shots rather
            # than a failed decode.
            raise DenseStreamError(f"the {frame_source.backend_name} decode produced no frames")
        counts = getattr(frame_source, "segment_counts", None)
        if counts is not None:
            received = [segments[index].frames if index in segments else 0 for index in range(len(counts))]
            if received != list(counts):
                # Like a short read: a truncated decode yields results that
                # look entirely plausible and are wrong.
                raise DenseStreamError(
                    f"expected {list(counts)} frames per segment from the decode, received {received}"
                )

        # Global frame numbers follow from the segments' frame counts, so the
        # windows are merged only now, in video order. A segment seam is a
        # window seam like any other: a shot that continues across it is joined.
        ranges_full: List[List[int]] = []
        intra_full: List[int] = []
        inter_full: List[int] = []
        segment_offset = 0
        for index in sorted(segments):
            segment = segments[index]
            for ranges, intra, inter in segment.windows:
                merge_ranges(
                    ranges_full, intra_full, inter_full, ranges, intra, inter, segment_offset,
                    new_start_inter_index=self._new_start_inter_index,
                )
            segment_offset += segment.frames

        total_seconds = time.perf_counter() - analyze_started
        self.logger.info(
            f"Shot boundaries for {real_frames} frames in {total_seconds:.1f}s "
            f"({real_frames / total_seconds:.0f} fps): inference {inference_seconds:.1f}s, "
            f"waiting on decode {decode_wait_seconds:.1f}s, backend {frame_source.backend_name}"
            if total_seconds > 0 else
            f"Shot boundaries for {real_frames} frames (backend {frame_source.backend_name})"
        )

        if info.fps_assumed and duration > 0 and real_frames:
            # The container declares no frame rate, so 30 fps was only a guess;
            # the decoded frame count over the duration is a measurement.
            fps = real_frames / duration
        if duration <= 0 and fps > 0 and real_frames:
            duration = real_frames / fps
        if real_frames and duration < (real_frames - 1) / fps:
            # Seconds in the partition end at the duration, so shots starting
            # past it are absorbed into the last one before it. A duration a
            # little longer than the frames is common and harmless: the last
            # shot is stretched, and duration_mismatch_seconds reports it.
            self.logger.warning(
                f"{self._name}: {real_frames} frames at {fps:.3f} fps last "
                f"{real_frames / fps:.3f}s, but the duration is only {duration:.3f}s"
            )

        return assemble_result(
            model=self._name,
            model_version=self.model_version,
            mode=self.mode,
            fps=fps,
            duration=duration,
            frame_count=real_frames,
            window_frames=window,
            context_frames=context,
            decode_backend=frame_source.backend_name,
            ranges=ranges_full,
            shot_types=[self._intra_names.get(x, f"Unknown_{x}") for x in intra_full],
            transitions=[self._inter_names.get(x, f"Unknown_{x}") for x in inter_full],
            # clean_shot keeps only ordinary shots in `shots`. The partition in
            # `boundaries` always covers every frame, transitions included.
            in_shots=[
                self.mode != "clean_shot" or label == self._general_intra_index
                for label in intra_full
            ],
            timings={
                # Decode and inference overlap: this is the time scoring sat idle
                # waiting for frames, not the time decoding took.
                "decode_wait_seconds": round(decode_wait_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
                "analyze_seconds": round(total_seconds, 3),
            },
        )

    def _run_window(self, frames: List[np.ndarray], num_pad_frames: int, offset: int,
                    ranges_full: List[List[int]], intra_full: List[int],
                    inter_full: List[int]) -> float:
        """Run one clip of HxWx3 uint8 frames starting at real frame ``offset``,
        merge its predictions, and return seconds spent."""
        started = time.perf_counter()
        # One uint8 stack and one float conversion per window. Converting each
        # frame as it arrived meant thousands of tiny parallel torch ops, which
        # kept torch's CPU thread pool spinning on every core ffmpeg needed.
        # The values are bit-identical to that per-frame path.
        clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous().float()
        clip = apply_spec_batch(clip, self._spec).unsqueeze(0)
        shot_logits, intra_logits, inter_logits = self.model.run_raw_multi_output(clip, use_half=False)

        # [..., :-1] drops the trailing "no object" slot on every head.
        range_idx = torch.from_numpy(shot_logits).softmax(-1)[0, :, :-1].argmax(dim=-1)
        intra_idx = torch.from_numpy(intra_logits).softmax(-1)[0, :, :-1].argmax(dim=-1)
        inter_idx = torch.from_numpy(inter_logits).softmax(-1)[0, :, :-1].argmax(dim=-1)

        ranges, intra, inter = [], [], []
        start_frame_idx = 0
        for query in range(len(intra_idx)):
            intra_label = int(intra_idx[query])
            inter_label = int(inter_idx[query])
            end_frame_idx = int(range_idx[query])
            if start_frame_idx >= end_frame_idx:
                continue
            ranges.append([start_frame_idx, end_frame_idx])
            intra.append(intra_label)
            inter.append(inter_label)
            start_frame_idx = end_frame_idx
            if end_frame_idx >= self.window_frames - num_pad_frames:
                break

        ranges, intra, inter = prune_non_context_ranges(
            ranges, intra, inter, self.window_frames, self.context_frames
        )
        merge_ranges(
            ranges_full, intra_full, inter_full, ranges, intra, inter, offset,
            new_start_inter_index=self._new_start_inter_index,
        )
        return time.perf_counter() - started


# ── generic helpers (no model-specific knowledge) ────────────────────────


class _Segment:
    """Windowing state for one contiguous run of frames."""

    def __init__(self, buffer):
        self.buffer: List[np.ndarray] = list(buffer)
        self.frames = 0
        # Window k of the segment starts at its frame k * stride.
        self.window_index = 0
        # One (ranges, intra, inter) per scored window, in segment frame numbers.
        self.windows: List[tuple] = []


def make_video_frame_source(video_path, *, decode_size, crop=None, decode_workers=None, **_ignored):
    """Every frame of ``video_path`` at ``decode_size``, without a pipeline.

    Runs the shared preprocessor's decode on a background thread and returns
    its ``DenseFrameStream``. A request never comes this way: there the
    preprocessor is already decoding the video and supplies the stream.
    """
    stream = DenseFrameStream(decode_size[0], decode_size[1])
    stream.attach()

    def produce():
        try:
            for _frame in preprocess_video_mp_pyav(
                video_path, 1.0, 0, False, "cpu", False, norm_config=-1,
                decode_workers=decode_workers, dense_sinks=[(tuple(decode_size), [stream])],
                dense_crop=crop, emit_interval=False,
            ):
                pass
        except BaseException:  # noqa: BLE001 - already handed to the stream
            pass

    threading.Thread(target=produce, name="shot-boundary-decode", daemon=True).start()
    return stream


class _Failure:
    def __init__(self, exception: BaseException):
        self.exception = exception


_END = object()


class Prefetch:
    """Iterate ``iterable`` on a background thread, up to ``maxsize`` items ahead.

    Items arrive in order and an exception raised by the source is re-raised
    to the consumer. ``close()`` -- also reached by leaving a ``with`` block --
    stops the thread and closes the source, which is what reaps an ffmpeg
    process that is still running. It is safe before the first item and
    after the last.
    """

    def __init__(self, iterable: Iterable, maxsize: int):
        self._items: queue.Queue = queue.Queue(maxsize=max(1, maxsize))
        self._stop = threading.Event()
        self._done = False
        self._thread = threading.Thread(
            target=self._produce, args=(iterable,), name="shot-boundary-decode", daemon=True)
        self._thread.start()

    def _offer(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._items.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self, iterable):
        iterator = iter(iterable)
        try:
            for item in iterator:
                if not self._offer(item):
                    return
            self._offer(_END)
        except BaseException as exception:  # noqa: BLE001 - handed to the consumer
            self._offer(_Failure(exception))
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        item = self._items.get()
        if item is _END:
            self._done = True
            raise StopIteration
        if isinstance(item, _Failure):
            self.close()
            raise item.exception
        return item

    def close(self, timeout: float = 10.0) -> None:
        self._done = True
        self._stop.set()
        self._thread.join(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def prune_non_context_ranges(ranges, intra_labels, inter_labels, window_frames, context_frames):
    """Drop predictions that fall inside a window's context padding and re-base
    the survivors onto the window's own coordinate system."""
    new_ranges, new_intra, new_inter = [], [], []
    for shot_idx in range(len(ranges)):
        start_frame_idx, end_frame_idx = ranges[shot_idx]
        if end_frame_idx <= context_frames:
            continue
        if start_frame_idx >= window_frames - context_frames:
            break
        aligned_start = max(start_frame_idx, context_frames) - context_frames
        aligned_end = min(end_frame_idx, window_frames - context_frames) - context_frames
        new_ranges.append([aligned_start, aligned_end])
        new_intra.append(intra_labels[shot_idx])
        new_inter.append(inter_labels[shot_idx])
    return new_ranges, new_intra, new_inter


def merge_ranges(ranges_full, intra_full, inter_full, ranges, intra_labels, inter_labels,
                 offset, new_start_inter_index=None):
    """Append one window's predictions to the whole-video result, joining a shot
    that continues across the window seam.

    ``offset`` is the real frame the window starts at (``window_index * stride``),
    not the end of the previous shot: a window whose query walk stopped short of
    its end then leaves a correctly placed gap instead of shifting every later
    shot earlier.
    """
    if (
        len(intra_full) != 0
        and len(intra_labels) != 0
        and intra_full[-1] == intra_labels[0]
        and new_start_inter_index is not None
        and inter_labels[0] == new_start_inter_index
    ):
        ranges_full[-1][-1] = offset + ranges[0][-1]
        ranges = ranges[1:]
        intra_labels = intra_labels[1:]
        inter_labels = inter_labels[1:]

    for idx in range(len(ranges)):
        start_frame_idx, end_frame_idx = ranges[idx]
        ranges_full.append([offset + start_frame_idx, offset + end_frame_idx])
        intra_full.append(intra_labels[idx])
        inter_full.append(inter_labels[idx])

    return ranges_full, intra_full, inter_full


# Bumped whenever the shape of the `shot_boundaries` node changes in a way a
# consumer must know about. Version 1 only ever came from pre-release builds:
# it had no frames, and its `transition_after` held each shot's own inter
# label, which describes the cut *into* the shot.
SCHEMA_VERSION = 2


def assemble_result(
    *,
    model: str,
    model_version,
    mode: str,
    fps: float,
    duration: float,
    frame_count: int,
    window_frames: int,
    context_frames: int,
    decode_backend: str,
    ranges: List[List[int]],
    shot_types: List[str],
    transitions: List[str],
    in_shots: List[bool],
    timings: Dict[str, float],
) -> dict:
    """Build the ``analysis.other.shot_boundaries`` node from the model's shots.

    ``ranges`` are the model's half-open frame ranges in decode order, with
    ``shot_types`` (intra labels) and ``transitions`` (inter labels) aligned to
    them. ``in_shots`` selects which of them are listed in ``shots``; the
    partition in ``boundaries`` is always built from all of them.
    """
    boundaries, label_counts = build_partition(
        ranges=ranges,
        shot_types=shot_types,
        transitions=transitions,
        frame_count=frame_count,
        fps=fps,
        duration=duration,
    )
    shots = []
    for index, frame_range in enumerate(ranges):
        if index >= len(in_shots) or not in_shots[index]:
            continue
        start = max(0, min(frame_count, int(frame_range[0])))
        end = max(0, min(frame_count, int(frame_range[1])))
        if end <= start:
            continue
        shots.append({
            "start_frame": start,
            "end_frame": end,
            "intra": _label_at(shot_types, index),
            "inter": _label_at(transitions, index),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "model_version": None if model_version is None else str(model_version),
        "mode": mode,
        "fps": fps,
        "duration_seconds": round(duration, 3),
        "source_frame_count": frame_count,
        # duration_seconds minus the decoded frames' own length. Seconds in the
        # partition follow the duration, so a large value explains a stretched
        # last shot (positive) or absorbed tail frames (negative).
        "duration_mismatch_seconds": (
            round(duration - frame_count / fps, 3) if fps > 0 and frame_count > 0 else 0.0
        ),
        "window_frames": window_frames,
        "context_frames": context_frames,
        "decode_backend": decode_backend,
        **timings,
        "boundaries": boundaries,
        "label_counts": label_counts,
        # The model's own shots, frame-accurate. In clean_shot mode only
        # ordinary shots are listed, so this is not a partition.
        "shots": shots,
    }


def build_partition(ranges, shot_types, transitions, frame_count: int, fps: float, duration: float):
    """Turn the model's frame ranges into a contiguous, gapless partition of the asset.

    Consumers store this directly, so it is exact:

    * Frames are authoritative. Each element covers the half-open decoded-frame
      range ``[start_frame, end_frame)``; the first starts at 0, each starts
      where the previous ended, and the last ends at ``frame_count``.
    * ``start_seconds``/``end_seconds`` are ``frame / fps`` rounded to
      milliseconds; the first is 0.0 and the last is the asset duration. An
      element too short to survive the rounding is absorbed by its predecessor.
    * ``shot_type`` is the model's intra label and ``transition_in`` its inter
      label, which describes how the element is entered from the one before
      it. The first element has no incoming cut, so its ``transition_in`` is
      null.
    * Frames no model shot covers become ``filled`` elements with null labels.
      An overlap is trimmed off the later shot.

    Ranges are half-open: the greedy query walk starts each shot at the
    previous shot's end, and the final end equals the total frame count.
    """
    frame_count = int(frame_count or 0)
    fps = float(fps or 0.0)
    duration = float(duration or 0.0)
    if frame_count <= 0 or fps <= 0 or duration <= 0:
        return [], {"shot_type": {}, "transition_in": {}}

    items = []
    for index, frame_range in enumerate(ranges):
        if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
            continue
        start = max(0, min(frame_count, int(frame_range[0])))
        end = max(0, min(frame_count, int(frame_range[1])))
        if end > start:
            items.append((start, end, _label_at(shot_types, index), _label_at(transitions, index)))
    items.sort(key=lambda item: (item[0], item[1]))

    elements = []
    cursor = 0
    for start, end, shot_type, transition in items:
        if end <= cursor:
            continue
        if start > cursor:
            elements.append((cursor, start, None, None, True))
        elements.append((max(start, cursor), end, shot_type, transition, False))
        cursor = end
    if cursor < frame_count:
        elements.append((cursor, frame_count, None, None, True))

    last_second = round(duration, 3)

    def to_seconds(frame: int) -> float:
        if frame >= frame_count:
            return last_second
        return min(last_second, round(frame / fps, 3))

    boundaries: List[dict] = []
    for start, end, shot_type, transition, filled in elements:
        start_seconds = boundaries[-1]["end_seconds"] if boundaries else 0.0
        end_seconds = to_seconds(end)
        if end_seconds <= start_seconds:
            # Shorter than a millisecond once rounded (or past a duration
            # shorter than the decoded frames): its frames join the
            # predecessor, or the successor when there is none yet.
            if boundaries:
                boundaries[-1]["end_frame"] = end
            continue
        boundaries.append({
            "start_frame": start if boundaries else 0,
            "end_frame": end,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "shot_type": shot_type,
            "transition_in": transition,
            "filled": filled,
        })
    if boundaries:
        boundaries[0]["transition_in"] = None

    counts = {
        "shot_type": dict(Counter(b["shot_type"] for b in boundaries if b["shot_type"] is not None)),
        "transition_in": dict(Counter(b["transition_in"] for b in boundaries if b["transition_in"] is not None)),
    }
    return boundaries, counts


def _label_at(labels, index: int) -> Optional[str]:
    if index < len(labels) and labels[index] is not None:
        return str(labels[index])
    return None
