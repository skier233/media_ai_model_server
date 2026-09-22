"""Asset-scope temporal segmentation (shot-boundary detection).

Unlike every other AI model in this server, a shot-boundary model does not
consume preprocessed frames handed to it by the shared frame preprocessor: it
needs dense, *adjacent* frames across the whole asset, at its own resolution.
It therefore runs at ``asset`` scope, receives the video path directly, and
drives its own decode.

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
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch

from lib.model.ai_model import AIModel
from lib.model.preprocessing_python.ffmpeg_pipe import (
    make_video_frame_source,
    probe_video,
)
from lib.model.preprocessing_python.image_preprocessing import _validate_local_video_source, vr_permute
from lib.pipeline.preprocess_spec import PreprocessSpec, apply_spec_batch

# NORMALIZATION_PRESETS key of the ImageNet mean/std the artifacts are trained with.
IMAGENET_NORMALIZATION = 0


class AIShotBoundaryModel(AIModel):
    """Runs a shot-boundary artifact over a whole video and emits segments."""

    # Consumes the video path and decodes every frame itself, so it runs only
    # in the asset-scope stage and only when a request names it. See
    # lib/model/whole_asset.py.
    whole_asset_model = True

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

                # Decode + inference take minutes, not milliseconds. Running them
                # inline would block the event loop and stall every other model —
                # including the tagging pass over this same video.
                result = await loop.run_in_executor(None, self._analyze, video_path, vr_video)
                await item.item_future.set_data(item.output_names[0], result)
            except Exception as exception:  # noqa: BLE001 - reported per item
                # The item's future is the request's own. Failing it would throw
                # away the frame results of a request that also asked for
                # tagging, so the asset stage reports the error instead.
                self.logger.error(f"{self._name}: shot-boundary analysis failed: {exception}", exc_info=True)
                await item.item_future.set_data(item.output_names[0], exception)

    # ── inference ────────────────────────────────────────────────────────

    def _analyze(self, video_path, vr_video: bool) -> dict:
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

        ranges_full: List[List[int]] = []
        intra_full: List[int] = []
        inter_full: List[int] = []

        # Rolling window over the padded frame stream. OmniShotCut's reference
        # implementation prepends `context` black frames to the whole video,
        # then walks windows of `window` frames at `stride`. Holding only one
        # window at a time keeps peak memory at window*C*H*W regardless of
        # video length; with vr_video, H*W is one eye at source resolution.
        # Padding must match the decoded frames, which with vr_video are not at
        # the clip geometry until a window is resized, so it is shaped after
        # the first frame, and the leading context is added with it.
        blank: Optional[torch.Tensor] = None
        buffer: List[torch.Tensor] = []
        real_frames = 0
        # Window k starts at real frame k * stride.
        window_index = 0

        # vr_permute crops AFTER decode, so cropping a frame already squashed to
        # the clip geometry is not the same as cropping then resizing. Decode at
        # native size and let the CPU path crop when VR is in play.
        decode_size = None if vr_video else (spec.width, spec.height)
        frame_source = make_video_frame_source(
            video_path,
            decode_size=decode_size,
            frame_step=1,              # every frame
            backend=self.decode_backend,
            quality=self.decode_quality,
            device_index=self.decode_device_index,
            info=info,
        )
        analyze_started = time.perf_counter()
        inference_seconds = 0.0

        for _index, frame_np in frame_source:
            frame = self._to_frame_tensor(frame_np, vr_video)
            if blank is None:
                blank = torch.zeros_like(frame)
                buffer = [blank.clone() for _ in range(context)]
            buffer.append(frame)
            real_frames += 1
            while len(buffer) >= window:
                inference_seconds += self._run_window(
                    buffer[:window], 0, window_index * stride, ranges_full, intra_full, inter_full)
                del buffer[:stride]
                window_index += 1

        # Tail: pad what is left out to full windows until every real frame has
        # been scored. `len(buffer) > context` is exactly "real frames remain
        # unscored", since buffer holds padded[base:] and padding adds `context`
        # frames. With context frames one padded window is not always enough:
        # its trailing context can hold real frames it does not score.
        while blank is not None and len(buffer) > context:
            num_pad = window - len(buffer)
            clip = list(buffer)
            if num_pad > 0:
                clip.extend(blank.clone() for _ in range(num_pad))
            inference_seconds += self._run_window(
                clip[:window], max(0, num_pad), window_index * stride, ranges_full, intra_full, inter_full)
            del buffer[:stride]
            window_index += 1

        # Decode and inference interleave, so time the one we can bound exactly
        # and attribute the remainder to decoding.
        total_seconds = time.perf_counter() - analyze_started
        decode_seconds = max(0.0, total_seconds - inference_seconds)
        self.logger.info(
            f"Shot boundaries for {real_frames} frames: "
            f"decode {decode_seconds:.1f}s ({real_frames / decode_seconds:.0f} fps) / "
            f"inference {inference_seconds:.1f}s / backend {frame_source.backend_name}"
            if decode_seconds > 0 else
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
                "decode_seconds": round(decode_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
            },
        )

    def _to_frame_tensor(self, frame_np, vr_video: bool) -> torch.Tensor:
        """HxWx3 uint8 -> CHW float32 in [0,255], matching apply_spec_batch's input."""
        tensor = torch.from_numpy(frame_np.copy())
        if vr_video:
            tensor = vr_permute(tensor)
        return tensor.permute(2, 0, 1).float()

    def _run_window(self, frames: List[torch.Tensor], num_pad_frames: int, offset: int,
                    ranges_full: List[List[int]], intra_full: List[int],
                    inter_full: List[int]) -> float:
        """Run one clip starting at real frame ``offset``, merge its predictions,
        and return seconds spent."""
        started = time.perf_counter()
        clip = apply_spec_batch(torch.stack(frames), self._spec).unsqueeze(0)
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
