"""Dense video decoding through an ffmpeg raw pipe.

Models that consume *every* frame of a whole asset (temporal segmentation, for
example) are decode-bound rather than compute-bound: a 4K source costs far more
to decode than to run a small network over.

This module runs one ffmpeg that decodes and rescales the video and writes raw
RGB frames at the model's small geometry to a pipe, which is several times
faster than decoding frame by frame through PyAV. Software decoding is the
default. NVDEC, which decodes and rescales on the GPU so that only the small
frames cross PCIe, is used only when a model asks for it: it does not scale
with host cores, and its frames differ slightly from the software path.

It is deliberately self-contained: no torch, no dependency on the shared frame
preprocessor, and it yields plain numpy frames. Callers convert to tensors
however they already do.

Two properties are load-bearing and easy to get wrong:

* **Frame count must match a software decode exactly.** Consumers map frame
  index to time as ``index / fps``, so a dropped or duplicated frame shifts
  every timestamp after it. Hence ``-fps_mode passthrough`` and a ``select``
  frame modulo — never the ``fps`` filter, which resamples to a wall-clock rate.
* **Scaling must not alias.** ``scale_cuda`` uses a fixed 4x4 kernel with no
  ratio-adaptive widening, unlike swscale's bicubic. Reducing 3840x2160 to
  128x96 in one GPU step samples ~4 of every 30x22 source pixels. So we halve on
  the GPU until within 2x of the target and take the last step with swscale,
  which is what a reference software pipeline would have produced.
"""

import logging
import os
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np

logger = logging.getLogger("logger")

# NVDEC cannot decode these; fall back to software rather than fail.
_UNSUPPORTED_NVDEC_PIXEL_FORMATS = {
    "yuv444p", "yuv444p10le", "yuv444p12le", "yuvj444p",
    "yuv420p12le", "yuv422p12le", "gbrp", "gbrp10le",
}

# Codecs NVDEC has a hardware decoder for. Anything else (WMV2/3, MS-MPEG4v3,
# ProRes, ...) decodes in software regardless, so there is no point paying for
# a smoke test to discover that.
_NVDEC_CODECS = {
    "h264", "avc", "hevc", "h265", "mpeg1video", "mpeg2video",
    "mpeg4", "vc1", "vp8", "vp9", "av1",
}

_BACKEND_FFMPEG_CUDA = "ffmpeg_cuda"
_BACKEND_FFMPEG_CPU = "ffmpeg_cpu"
_BACKEND_AV = "av"


class DecodeSetupError(RuntimeError):
    """A decode backend is unusable for this source. Try the next rung."""


class DecodeStreamError(RuntimeError):
    """A decode failed part-way through. Never treat as a short video."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    pix_fmt: str
    codec: str
    fps: float
    duration: float
    nb_frames: int
    field_order: str
    color_range: Optional[str]
    color_matrix: Optional[str]
    # True when the container declares no frame rate and `fps` is a guess.
    fps_assumed: bool = False


@dataclass(frozen=True)
class FFmpegFeatures:
    path: str
    version: str
    has_cuda: bool
    has_scale_cuda: bool
    has_fps_mode: bool


@dataclass(frozen=True)
class ScalePlan:
    """How to get from the source geometry to an exact target size.

    ``gpu_steps`` are aspect-preserving halvings applied with ``scale_cuda``;
    ``target`` is the exact output size, reached by one last, small,
    aspect-breaking step with swscale (or on the GPU when there is no halving).
    ``crop`` is an ``(x, y, width, height)`` region of the source taken before
    any scaling, e.g. one eye of a VR frame; the plan is made from its size.
    """

    target: Optional[tuple]
    gpu_steps: tuple = ()
    crop: Optional[tuple] = None

    @property
    def is_noop(self) -> bool:
        return self.target is None


# ── environment ──────────────────────────────────────────────────────────

_features_cache: Optional[FFmpegFeatures] = None


def find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg, preferring an explicit override then a conda prefix."""
    override = os.environ.get("MEDIA_AI_FFMPEG")
    if override and os.path.isfile(override):
        return override
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "bin", "ffmpeg")
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("ffmpeg")


def ffmpeg_features(refresh: bool = False) -> Optional[FFmpegFeatures]:
    """Probe the ffmpeg binary once and cache what it can do."""
    global _features_cache
    if _features_cache is not None and not refresh:
        return _features_cache

    path = find_ffmpeg()
    if not path:
        return None
    try:
        version_out = subprocess.run(
            [path, "-hide_banner", "-version"], capture_output=True, text=True, timeout=20
        ).stdout
        hwaccels = subprocess.run(
            [path, "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=20
        ).stdout
        filters = subprocess.run(
            [path, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError) as exception:
        logger.warning(f"Could not probe ffmpeg at {path}: {exception}")
        return None

    version = version_out.splitlines()[0] if version_out else ""
    major = _major_version(version)
    _features_cache = FFmpegFeatures(
        path=path,
        version=version,
        has_cuda="cuda" in hwaccels.lower(),
        has_scale_cuda="scale_cuda" in filters,
        # -vsync is deprecated from ffmpeg 5 and warns loudly on 8.
        has_fps_mode=major >= 5,
    )
    return _features_cache


def _major_version(version_line: str) -> int:
    for token in version_line.split():
        if token.startswith("n"):
            token = token[1:]
        head = token.split(".")[0]
        if head.isdigit():
            return int(head)
    return 0


def probe_video(video_path) -> VideoInfo:
    """Read geometry, rate and colour tags without decoding."""
    import av

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise DecodeSetupError(f"No video stream in {video_path}")
        stream = container.streams.video[0]
        ctx = stream.codec_context
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if fps <= 0:
            fps = float(stream.guessed_rate) if stream.guessed_rate else 0.0
        fps_assumed = fps <= 0
        if fps_assumed:
            fps = 30.0

        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / av.time_base
        else:
            duration = (stream.frames / fps) if (stream.frames and fps) else 0.0

        return VideoInfo(
            width=int(ctx.width or 0),
            height=int(ctx.height or 0),
            pix_fmt=str(getattr(ctx, "pix_fmt", "") or ""),
            # The codec, not the decoder PyAV picked for it: AV1 opens with
            # libdav1d, and NVDEC eligibility is decided by codec.
            codec=_codec_name(ctx),
            fps=fps,
            duration=duration,
            nb_frames=int(stream.frames or 0),
            field_order=_enum_name(getattr(ctx, "field_order", None), _FIELD_ORDER_NAMES) or "unknown",
            color_range=_enum_name(getattr(stream, "color_range", None), _COLOR_RANGE_NAMES),
            color_matrix=_enum_name(getattr(stream, "colorspace", None), _COLOR_SPACE_NAMES),
            fps_assumed=fps_assumed,
        )


def _codec_name(ctx) -> str:
    codec = getattr(ctx, "codec", None)
    canonical = getattr(codec, "canonical_name", None) if codec is not None else None
    return str(canonical or getattr(ctx, "name", None) or "")


# PyAV exposes these as raw ffmpeg enum integers (and, in other versions, as
# objects with a .name). Normalise both to ffmpeg's string names.
_FIELD_ORDER_NAMES = {0: "unknown", 1: "progressive", 2: "tt", 3: "bb", 4: "tb", 5: "bt"}
_COLOR_RANGE_NAMES = {0: None, 1: "tv", 2: "pc"}
_COLOR_SPACE_NAMES = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 9: "bt2020nc"}


def _enum_name(value, names: dict):
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name:
        return str(name).lower()
    if isinstance(value, int):
        return names.get(value)
    text = str(value).strip().lower()
    if text.isdigit():
        return names.get(int(text))
    return text or None


def nvdec_supports(info: VideoInfo) -> tuple:
    """Whether NVDEC can be expected to decode this source."""
    if info.codec and info.codec.lower() not in _NVDEC_CODECS:
        return False, f"{info.codec} has no NVDEC decoder"
    if info.pix_fmt in _UNSUPPORTED_NVDEC_PIXEL_FORMATS:
        return False, f"pixel format {info.pix_fmt} is not supported by NVDEC"
    if "12le" in info.pix_fmt or "12be" in info.pix_fmt:
        return False, f"12-bit pixel format {info.pix_fmt} is not supported by NVDEC"
    # Interlaced content needs deinterlacing, which changes the frame count.
    if info.field_order and info.field_order not in ("progressive", "unknown"):
        return False, f"interlaced source ({info.field_order}) would change the frame count"
    if not info.width or not info.height:
        return False, "source geometry is unknown"
    return True, ""


# ── planning ─────────────────────────────────────────────────────────────

def _even_up(value: int) -> int:
    """Round up to the next even number; ffmpeg filters dislike odd dimensions."""
    return (value + 1) & ~1


def plan_scale(src_width: int, src_height: int, decode_size, quality: str = "exact",
               crop: Optional[tuple] = None) -> ScalePlan:
    """Plan the descent from the source geometry to an exact target size.

    ``quality="exact"`` halves on the GPU until within 2x of the target so the
    final swscale step matches a pure software pipeline. ``quality="fast"``
    goes straight there in one GPU step, which is quicker and aliases.
    With ``crop``, the cropped region is what gets scaled.
    """
    crop = tuple(int(value) for value in crop) if crop else None
    if crop:
        src_width, src_height = crop[2], crop[3]
    if not decode_size:
        return ScalePlan(target=None, crop=crop)

    target_width, target_height = int(decode_size[0]), int(decode_size[1])
    if target_width <= 0 or target_height <= 0:
        return ScalePlan(target=None, crop=crop)
    if (src_width, src_height) == (target_width, target_height):
        return ScalePlan(target=None, crop=crop)

    if quality == "fast":
        return ScalePlan(target=(target_width, target_height), crop=crop)

    steps = []
    width, height = src_width, src_height
    # Halve while either axis is still more than 2x the target. Halving keeps
    # the aspect ratio, so scale_cuda's fixed kernel stays adequate; the final
    # aspect-breaking step is left to swscale.
    while width > 2 * target_width or height > 2 * target_height:
        # Round the half UP to an even number. Rounding down can push a step
        # past 2x (270 -> 134 is 2.014x), which is exactly what the pyramid
        # exists to avoid.
        width = max(2, _even_up((width + 1) // 2))
        height = max(2, _even_up((height + 1) // 2))
        steps.append((width, height))
        if width <= target_width and height <= target_height:
            break

    return ScalePlan(target=(target_width, target_height), gpu_steps=tuple(steps), crop=crop)


def vr_crop_rect(width: int, height: int) -> tuple:
    """The ``(x, y, width, height)`` region ``vr_permute`` keeps of a VR frame.

    A 180-degree side-by-side frame (wider than 1.5:1) keeps its right half; a
    360-degree frame keeps the top half's centre 50%. Integer arithmetic
    matches ``vr_permute``'s slicing exactly, so doing the crop in ffmpeg
    selects the same source pixels as cropping a decoded frame in Python.
    """
    if width / height > 1.5:
        return (width // 2, 0, width - width // 2, height)
    return (width // 4, 0, 3 * width // 4 - width // 4, height // 2)


def build_filter_chain(plan: ScalePlan, frame_step: int, hw: bool) -> str:
    """Build the -vf chain. Pure: no I/O, unit-tested directly."""
    parts = []
    if frame_step > 1:
        # A frame modulo, never `fps=`: the latter resamples to a wall-clock
        # rate and so changes the frame count on VFR sources.
        parts.append(f"select='not(mod(n\\,{frame_step}))'")

    if plan.crop:
        if hw:
            # Frames are still CUDA surfaces here; the caller refuses NVDEC
            # for cropped sources rather than download them at full size.
            raise ValueError("a cropped source is decoded in software")
        x, y, width, height = plan.crop
        # exact=1: without it a 4:2:0 source has odd offsets rounded down,
        # which would select different pixels than vr_permute does.
        parts.append(f"crop={width}:{height}:{x}:{y}:exact=1")

    if hw:
        for index, (width, height) in enumerate(plan.gpu_steps):
            is_last = index == len(plan.gpu_steps) - 1
            step = f"scale_cuda={width}:{height}:interp_algo={'bicubic' if is_last else 'bilinear'}"
            if is_last:
                # Convert on the GPU so 10-bit sources (p010le) survive the
                # download, which `hwdownload,format=nv12` alone cannot do.
                step += ":format=nv12"
            parts.append(step)
        if not plan.gpu_steps and not plan.is_noop:
            width, height = plan.target
            parts.append(f"scale_cuda={width}:{height}:interp_algo=bicubic:format=nv12")
        parts.append("hwdownload")
        parts.append("format=nv12")
        if plan.gpu_steps and not plan.is_noop:
            parts.append(_cpu_scale(plan.target))
    elif not plan.is_noop:
        parts.append(_cpu_scale(plan.target))

    # No trailing `format=rgb24`: the output's `-pix_fmt rgb24` performs the
    # YUV->RGB conversion as part of the same swscale operation that rescales,
    # which is what PyAV's `reformat` and OmniShotCut's reference implementation
    # both do. Converting as a separate filter measurably diverges from them
    # (mean |delta| 8.8 vs 0.9 on saturated synthetic content).
    return ",".join(parts)


def _cpu_scale(target) -> str:
    width, height = target
    return f"scale={width}:{height}:flags=bicubic"


def build_command(
    ffmpeg: str,
    video_path,
    filter_chain: str,
    hw: bool,
    has_fps_mode: bool,
    device_index: Optional[int] = None,
    frames: Optional[int] = None,
) -> list:
    """Build the full ffmpeg argv. Pure: no I/O, unit-tested directly."""
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if hw:
        command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        if device_index is not None:
            command += ["-hwaccel_device", str(device_index)]
    # Decode the stored frames as they are, like PyAV does: autorotation would
    # hand a rotated clip to the model, and rotate a VR frame before its crop.
    command += ["-noautorotate", "-i", str(video_path)]
    # passthrough keeps ffmpeg from duplicating or dropping frames to hit a rate.
    command += ["-fps_mode", "passthrough"] if has_fps_mode else ["-vsync", "passthrough"]
    command += ["-map", "0:v:0", "-an", "-sn", "-dn"]
    if filter_chain:
        command += ["-vf", filter_chain]
    if frames:
        command += ["-frames:v", str(frames)]
    command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return command


# ── execution ────────────────────────────────────────────────────────────

def _drain_stderr(stream, sink: deque) -> None:
    try:
        for line in iter(stream.readline, b""):
            sink.append(line.decode("utf-8", "replace").rstrip())
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def smoke_test(command: Sequence[str], width: int, height: int, timeout: float = 60.0) -> None:
    """Decode two frames with the real command before committing to it.

    Catches what a capability probe cannot: NVDEC session exhaustion, driver
    mismatches, an unexpected output geometry, a corrupt header. Costs a
    fraction of a second instead of a whole pass.
    """
    try:
        result = subprocess.run(list(command), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exception:
        raise DecodeSetupError(f"ffmpeg smoke test timed out after {timeout}s") from exception
    except OSError as exception:
        raise DecodeSetupError(f"ffmpeg could not be started: {exception}") from exception

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise DecodeSetupError(
            f"ffmpeg smoke test failed (exit {result.returncode}): {detail[-1] if detail else 'no output'}"
        )

    frame_bytes = width * height * 3
    if frame_bytes <= 0 or len(result.stdout) == 0:
        raise DecodeSetupError("ffmpeg smoke test produced no frames")
    if len(result.stdout) % frame_bytes != 0:
        raise DecodeSetupError(
            f"ffmpeg smoke test produced {len(result.stdout)} bytes, "
            f"not a multiple of the expected {frame_bytes} bytes/frame"
        )


def iter_rgb_frames(command: Sequence[str], width: int, height: int) -> Iterator[np.ndarray]:
    """Yield HxWx3 uint8 frames from one ffmpeg process.

    A short read that is not a clean EOF is an error, never a short video: a
    truncated decode produces results that look entirely plausible and are wrong.
    """
    frame_bytes = width * height * 3
    stderr_tail: deque = deque(maxlen=200)
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # A full stderr pipe deadlocks ffmpeg just as surely as a full data pipe.
        bufsize=0,
    )
    stderr_thread = threading.Thread(
        target=_drain_stderr, args=(process.stderr, stderr_tail), daemon=True
    )
    stderr_thread.start()

    try:
        while True:
            buffer = bytearray(frame_bytes)
            filled = 0
            while filled < frame_bytes:
                chunk = process.stdout.readinto(memoryview(buffer)[filled:])
                if not chunk:
                    break
                filled += chunk
            if filled == 0:
                break
            if filled < frame_bytes:
                raise DecodeStreamError(
                    f"ffmpeg produced a partial frame ({filled}/{frame_bytes} bytes); "
                    f"output truncated: {' | '.join(list(stderr_tail)[-3:]) or 'no stderr'}"
                )
            yield np.frombuffer(bytes(buffer), dtype=np.uint8).reshape(height, width, 3)

        returncode = process.wait(timeout=30)
        if returncode != 0:
            raise DecodeStreamError(
                f"ffmpeg exited {returncode}: {' | '.join(list(stderr_tail)[-3:]) or 'no stderr'}"
            )
    finally:
        try:
            if process.stdout:
                process.stdout.close()
        except OSError:
            pass
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg did not exit after kill()")
        stderr_thread.join(timeout=2)


def _iter_pyav_frames(video_path, plan: ScalePlan, frame_step: int) -> Iterator[np.ndarray]:
    """Software fallback: PyAV decode with one swscale reformat per frame."""
    import av
    from av.video.reformatter import Interpolation

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(video=0)):
            if frame_step > 1 and (index % frame_step) != 0:
                continue
            if plan.crop:
                # PyAV cannot crop, so round-trip one frame through RGB. Only
                # the fallback pays this, and only one frame is held at a time.
                x, y, width, height = plan.crop
                region = frame.to_ndarray(format="rgb24")[y:y + height, x:x + width]
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(region), format="rgb24")
            if plan.is_noop:
                yield frame.to_ndarray(format="rgb24")
            else:
                width, height = plan.target
                yield frame.reformat(
                    width=width, height=height, format="rgb24",
                    interpolation=Interpolation.BICUBIC,
                ).to_ndarray()


# ── public entry point ───────────────────────────────────────────────────

class FrameSource:
    """An opened decode. Iterating yields ``(index, HxWx3 uint8)``.

    Deliberately NOT a generator function: setup failures must surface from the
    factory so the fallback ladder can act on them. A generator's body does not
    run until the first ``next()``, by which point the caller has already
    committed to a backend.
    """

    def __init__(self, backend_name: str, info: VideoInfo, frames_fn, width: int, height: int):
        self.backend_name = backend_name
        self.info = info
        self.width = width
        self.height = height
        self._frames_fn = frames_fn
        self.frames_emitted = 0

    def __iter__(self) -> Iterator[tuple]:
        for index, frame in enumerate(self._frames_fn()):
            self.frames_emitted = index + 1
            yield index, frame
        if self.frames_emitted == 0:
            # Like a short read: an empty result would look like a video with
            # no shots rather than a failed decode.
            raise DecodeStreamError(f"the {self.backend_name} decode produced no frames")


def make_video_frame_source(
    video_path,
    *,
    decode_size=None,
    frame_step: int = 1,
    backend: str = "auto",
    quality: str = "exact",
    device_index: Optional[int] = None,
    info: Optional[VideoInfo] = None,
    crop: Optional[tuple] = None,
) -> FrameSource:
    """Open the fastest usable decode for this source.

    Tries the backends of ``backend``'s ladder in order (see ``_ladder``; NVDEC
    only when asked for). Every check runs before a single frame is produced,
    so a failure here is recoverable.
    ``crop`` (x, y, width, height) is applied to the source before scaling.
    """
    info = info or probe_video(video_path)
    plan = plan_scale(info.width, info.height, decode_size, quality, crop=crop)
    if not plan.is_noop:
        width, height = plan.target
    elif plan.crop:
        width, height = plan.crop[2], plan.crop[3]
    else:
        width, height = info.width, info.height

    for rung in _ladder(backend):
        try:
            return _open(rung, video_path, info, plan, frame_step, width, height, device_index)
        except DecodeSetupError as exception:
            logger.info(f"Decode backend {rung} unavailable for this source: {exception}")

    raise DecodeSetupError(f"No usable decode backend for {video_path}")


def _ladder(backend: str) -> list:
    """Backends to try, in order.

    ``auto`` deliberately does NOT include NVDEC. Measured on a 32-core host
    against a 4K60 H.264 source at 128x96:

        ffmpeg_cpu   417 fps   max|delta| 3 vs PyAV  (mean 0.0000)
        ffmpeg_cuda  275 fps   max|delta| 40 vs PyAV (mean 0.19)
        av           108 fps   (the previous behaviour)

    NVDEC is capped by the card's decoder engine, so it does not scale with the
    host: pinning the decode to 2/4/8/16/32 cores measured 247/248/248/247/247
    fps for ``ffmpeg_cuda`` against 91/165/260/313/324 for ``ffmpeg_cpu``. They
    cross near 8 cores, so NVDEC wins only on a small or already-busy host. It
    also differs numerically -- ``scale_cuda`` has a fixed kernel, enough to move
    a borderline shot boundary. Software ffmpeg is indistinguishable from the
    PyAV path the model was validated against, so it is the default and NVDEC
    stays opt-in.
    """
    backend = (backend or "auto").strip().lower()
    if backend == "auto":
        return [_BACKEND_FFMPEG_CPU, _BACKEND_AV]
    if backend in (_BACKEND_FFMPEG_CUDA, _BACKEND_FFMPEG_CPU):
        # An explicit choice still falls back rather than failing the request.
        return [backend, _BACKEND_FFMPEG_CPU, _BACKEND_AV] if backend == _BACKEND_FFMPEG_CUDA \
            else [backend, _BACKEND_AV]
    return [_BACKEND_AV]


def _open(rung, video_path, info, plan, frame_step, width, height, device_index) -> FrameSource:
    if rung == _BACKEND_AV:
        return FrameSource(
            _BACKEND_AV, info,
            lambda: _iter_pyav_frames(video_path, plan, frame_step),
            width, height,
        )

    features = ffmpeg_features()
    if features is None:
        raise DecodeSetupError("ffmpeg was not found")

    hw = rung == _BACKEND_FFMPEG_CUDA
    if hw:
        if plan.crop:
            raise DecodeSetupError("cropped sources are decoded in software")
        if not features.has_cuda:
            raise DecodeSetupError("this ffmpeg was built without CUDA support")
        if not features.has_scale_cuda:
            raise DecodeSetupError("this ffmpeg has no scale_cuda filter")
        supported, reason = nvdec_supports(info)
        if not supported:
            raise DecodeSetupError(reason)

    chain = build_filter_chain(plan, frame_step, hw)
    command = build_command(
        features.path, video_path, chain, hw, features.has_fps_mode, device_index
    )
    smoke_test(
        build_command(
            features.path, video_path, chain, hw, features.has_fps_mode, device_index, frames=2
        ),
        width, height,
    )
    return FrameSource(rung, info, lambda: iter_rgb_frames(command, width, height), width, height)
