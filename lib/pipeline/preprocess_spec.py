"""PreprocessSpec — a resolution/normalization/device descriptor for pipeline preprocessing.

Each AI model in a pipeline declares what preprocessed tensor it needs
(resolution, normalization, device, precision).  The preprocessor collects
all unique specs and produces one tensor per spec from a single disk/video
read.  Models are wired to consume the output matching their spec's key.
"""

from dataclasses import dataclass, replace
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F


# Normalization presets: id → (mean, std) for per-channel [0,1]-scaled tensors.
NORMALIZATION_PRESETS = {
    0: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),             # ImageNet
    1: ([0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),        
}


@dataclass(frozen=True)
class PreprocessSpec:
    """Immutable descriptor for one preprocessed tensor a pipeline needs.

    The preprocessor produces one output per unique spec.  Models are wired
    to consume the output whose key matches their spec.
    """

    width: int = 0            # Target width.  0 = preserve original aspect.
    height: int = 0           # Target height. 0 = preserve original aspect.
    normalization: int = -1   # -1 = raw [0,255]; 0 = ImageNet; 1 = CLIP.
    # Default "cpu": preprocessed frames are resized/normalized on CPU and stay
    # in system RAM until the model worker moves them to GPU at inference time
    # (every worker / PythonModel.run_model already does .to(device)). This keeps
    # the in-flight preprocessed-frame backlog OUT of VRAM — the main driver of
    # the gradual GPU-memory climb on long videos — and benchmarks ~12-15%
    # faster than GPU preprocessing (it avoids copying full-res raw frames over
    # PCIe and offloads work from the bottleneck GPU). Set "gpu" per model via
    # preprocess_config.device only if a model genuinely needs GPU-side prep.
    device: str = "cpu"       # "cpu" or "gpu".
    half_precision: bool = False
    max_long_edge: int = 0    # 0 = no cap; >0 = cap longest edge (preserving aspect).
    center_crop: bool = False # True = resize shortest edge then center-crop (CLIP-style).
    interpolation: str = "bilinear"  # "bilinear" or "bicubic".
    precision: str = ""       # "float16", "bfloat16", "float32", or "" (auto from half_precision).

    @property
    def effective_dtype(self) -> torch.dtype:
        """Resolved dtype: explicit precision field wins, else half_precision flag."""
        if self.precision == "bfloat16":
            return torch.bfloat16
        if self.precision == "float16" or (not self.precision and self.half_precision):
            return torch.float16
        return torch.float32

    @property
    def key(self) -> str:
        """Deterministic, human-readable pipeline-data key."""
        size = f"{self.width}x{self.height}" if (self.width or self.height) else "native"
        norm = "raw" if self.normalization == -1 else f"n{self.normalization}"
        dtype = self.effective_dtype
        prec_map = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}
        prec = prec_map.get(dtype, "f32")
        cap = f"_max{self.max_long_edge}" if self.max_long_edge > 0 else ""
        crop = "_ccrop" if self.center_crop else ""
        interp = f"_{self.interpolation}" if self.interpolation != "bilinear" else ""
        return f"prep__{size}_{norm}_{self.device}_{prec}{cap}{crop}{interp}"

    @property
    def effective_resolution(self) -> int:
        """Approximate pixel dimension for sorting (highest-first).

        0 means 'original' and sorts as largest.
        """
        fixed = max(self.width, self.height)
        cap = self.max_long_edge
        if fixed > 0:
            return fixed
        if cap > 0:
            return cap
        return 999_999  # native = treat as largest

    @property
    def _geometry_key(self) -> tuple:
        """All fields except precision — for detecting specs that can share a tensor."""
        return (self.width, self.height, self.normalization, self.device,
                self.max_long_edge, self.center_crop, self.interpolation)

    def as_float32(self) -> "PreprocessSpec":
        """Return a copy promoted to float32.  No-op if already float32."""
        if self.effective_dtype == torch.float32:
            return self
        return replace(self, half_precision=False, precision="float32")

    # ── Factory helpers ────────────────────────────────────────────────

    @staticmethod
    def for_model(model_inner) -> "PreprocessSpec":
        """Auto-derive a spec from a model's config properties.

        Checks for an explicit ``preprocess_config`` dict first (fully
        customizable), then falls back to role-based heuristics.
        """
        pc = getattr(model_inner, "preprocess_config", None)
        if pc and isinstance(pc, dict):
            size = pc.get("image_size", getattr(model_inner, "model_image_size", 512) or 512)
            return PreprocessSpec(
                width=pc.get("width", size),
                height=pc.get("height", size),
                normalization=pc.get("normalization", -1),
                device=pc.get("device", "cpu"),
                half_precision=pc.get("half_precision", False),
                max_long_edge=pc.get("max_long_edge", 0),
                center_crop=pc.get("center_crop", False),
                interpolation=pc.get("interpolation", "bilinear"),
                precision=pc.get("precision", ""),
            )

        face_role = getattr(model_inner, "face_model_role", None)
        if face_role is not None:
            return PreprocessSpec.for_detector(model_inner)

        capabilities = set(getattr(model_inner, "model_capabilities", None) or [])
        if "detection" in capabilities:
            return PreprocessSpec.for_detector(model_inner)

        return PreprocessSpec.for_tagging_model(model_inner)

    @staticmethod
    def for_tagging_model(model_inner) -> "PreprocessSpec":
        """Spec for a standard image-tagging model."""
        size = getattr(model_inner, "model_image_size", 512) or 512
        norm = getattr(model_inner, "normalization_config", None)
        if norm is None or norm < 0:
            norm = 1  # default CLIP
        precision = getattr(model_inner, "model_precision", None) or ""
        return PreprocessSpec(
            width=size, height=size,
            normalization=norm, device="cpu",
            half_precision=not precision,  # default half unless explicit precision
            max_long_edge=0,
            precision=precision,
        )

    @staticmethod
    def for_detector(model_inner) -> "PreprocessSpec":
        """Spec for a detection model (face detector, etc.)."""
        size = getattr(model_inner, "model_image_size", 640) or 640
        return PreprocessSpec(
            width=0, height=0,
            normalization=-1, device="cpu",
            half_precision=False, max_long_edge=size,
        )

    @staticmethod
    def for_region_source(max_long_edge: Optional[int] = None) -> "PreprocessSpec":
        """Spec for a full-resolution region-source tensor (face/region cropping).

        Stays on CPU to avoid GPU memory pressure from large tensors.
        """
        if max_long_edge is None:
            try:
                max_long_edge = int(os.environ.get("REGION_SOURCE_MAX_LONG_EDGE", "1920"))
            except (ValueError, TypeError):
                max_long_edge = 1920
        return PreprocessSpec(
            width=0, height=0,
            normalization=-1, device="cpu",
            half_precision=False, max_long_edge=max_long_edge,
        )


# ── Decode sizing ─────────────────────────────────────────────────────

def vr_view_size(width: int, height: int) -> tuple:
    """Size of the region ``vr_permute`` keeps from a ``width`` x ``height`` frame."""
    if width / height > 1.5:
        return width - width // 2, height
    return 3 * width // 4 - width // 4, height // 2


def decode_long_edge_for_specs(specs, source_width: int, source_height: int, vr_video: bool = False) -> int:
    """Longest edge a video may be downscaled to at decode time, or 0 for no cap.

    The decoded frame is the source every spec is resized from, so it has to
    keep enough pixels for each of them on *both* axes. A spec's largest side
    alone is not enough: a 512x512 squash of a 16:9 source needs 512 rows, and
    capping the long edge at 512 leaves 288 that are then stretched back up.

    A VR frame is cropped to one view after decoding, so the specs are sized
    against that view and the cap is applied to the whole frame.
    """
    if not specs or source_width <= 0 or source_height <= 0:
        return 0
    view_width, view_height = (
        vr_view_size(source_width, source_height) if vr_video else (source_width, source_height)
    )
    if view_width <= 0 or view_height <= 0:
        return 0

    scale = 0.0
    for spec in specs:
        if spec.width > 0 and spec.height > 0:
            if spec.center_crop:
                needed = min(spec.width, spec.height) / min(view_width, view_height)
            else:
                needed = max(spec.width / view_width, spec.height / view_height)
        elif spec.max_long_edge > 0:
            needed = spec.max_long_edge / max(view_width, view_height)
        else:
            return 0  # a native-resolution spec needs the full frame
        scale = max(scale, needed)

    if scale >= 1.0:
        return 0
    return math.ceil(max(source_width, source_height) * scale)


# ── Tensor processing ─────────────────────────────────────────────────

def apply_spec(source: torch.Tensor, spec: PreprocessSpec) -> torch.Tensor:
    """Apply a *PreprocessSpec* to a raw ``float32`` ``[0, 255]`` CHW tensor.

    The source tensor is never modified in-place.  Returns a new tensor on
    the spec's target device with the requested resolution, normalization,
    and precision.
    """
    t = source
    h, w = t.shape[-2], t.shape[-1]
    resized = False
    interp_mode = "bicubic" if spec.interpolation == "bicubic" else "bilinear"

    # 1. Resize -----------------------------------------------------------
    if spec.width > 0 and spec.height > 0:
        if spec.center_crop:
            # CLIP-style: resize shortest edge, then center-crop.
            short_edge = min(h, w)
            target_short = min(spec.width, spec.height)
            if short_edge != target_short or h == w:
                scale = target_short / short_edge
                new_h = max(1, round(h * scale))
                new_w = max(1, round(w * scale))
                t = F.interpolate(
                    t.unsqueeze(0), size=(new_h, new_w),
                    mode=interp_mode, align_corners=False,
                ).squeeze(0)
            # Center-crop to target size.
            ch, cw = t.shape[-2], t.shape[-1]
            top = (ch - spec.height) // 2
            left = (cw - spec.width) // 2
            t = t[:, top : top + spec.height, left : left + spec.width]
            resized = True
        elif h != spec.height or w != spec.width:
            t = F.interpolate(
                t.unsqueeze(0), size=(spec.height, spec.width),
                mode=interp_mode, align_corners=False,
            ).squeeze(0)
            resized = True
    elif spec.max_long_edge > 0:
        long_edge = max(h, w)
        if long_edge > spec.max_long_edge:
            scale = spec.max_long_edge / long_edge
            new_h = max(1, round(h * scale))
            new_w = max(1, round(w * scale))
            t = F.interpolate(
                t.unsqueeze(0), size=(new_h, new_w),
                mode=interp_mode, align_corners=False,
            ).squeeze(0)
            resized = True

    # Ensure we never alias the source tensor.
    if not resized:
        t = t.clone()

    # 2. Device -----------------------------------------------------------
    target_device = (
        torch.device("cpu") if spec.device == "cpu"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if t.device != target_device:
        t = t.to(target_device)

    # 3. Normalize (after resize, on target device) -----------------------
    if spec.normalization >= 0:
        mean_list, std_list = NORMALIZATION_PRESETS[spec.normalization]
        t = t / 255.0
        mean = torch.tensor(mean_list, device=t.device, dtype=t.dtype).view(-1, 1, 1)
        std = torch.tensor(std_list, device=t.device, dtype=t.dtype).view(-1, 1, 1)
        t = (t - mean) / std

    # 4. Precision --------------------------------------------------------
    if spec.half_precision or spec.precision:
        t = t.to(spec.effective_dtype)

    return t


def apply_spec_batch(source: torch.Tensor, spec: PreprocessSpec) -> torch.Tensor:
    """Batched form of :func:`apply_spec` for an ``[N, C, H, W]`` source.

    Produces an ``[N, C, Hs, Ws]`` tensor on the spec's target device.  For GPU
    specs the batch is moved to the device *before* resizing, so a single H2D
    copy serves the whole batch and the resize/normalize run as a handful of
    batched GPU kernels — instead of one CPU resize + H2D per frame.  This keeps
    the per-frame work off the Python GIL so video decode and inference overlap.

    All frames in a batch must share the same H, W (true for one video).
    """
    t = source
    if t.dim() != 4:
        raise ValueError(f"apply_spec_batch expects [N, C, H, W]; got {tuple(t.shape)}")

    target_device = (
        torch.device("cpu") if spec.device == "cpu"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Move the (pre-resize, hence smaller) batch to the target device up front.
    if t.device != target_device:
        t = t.to(target_device, non_blocking=(target_device.type == "cuda"))

    h, w = t.shape[-2], t.shape[-1]
    resized = False
    interp_mode = "bicubic" if spec.interpolation == "bicubic" else "bilinear"

    # 1. Resize -----------------------------------------------------------
    if spec.width > 0 and spec.height > 0:
        if spec.center_crop:
            short_edge = min(h, w)
            target_short = min(spec.width, spec.height)
            if short_edge != target_short or h == w:
                scale = target_short / short_edge
                new_h = max(1, round(h * scale))
                new_w = max(1, round(w * scale))
                t = F.interpolate(t, size=(new_h, new_w), mode=interp_mode, align_corners=False)
            ch, cw = t.shape[-2], t.shape[-1]
            top = (ch - spec.height) // 2
            left = (cw - spec.width) // 2
            t = t[:, :, top : top + spec.height, left : left + spec.width]
            resized = True
        elif h != spec.height or w != spec.width:
            t = F.interpolate(t, size=(spec.height, spec.width), mode=interp_mode, align_corners=False)
            resized = True
    elif spec.max_long_edge > 0:
        long_edge = max(h, w)
        if long_edge > spec.max_long_edge:
            scale = spec.max_long_edge / long_edge
            new_h = max(1, round(h * scale))
            new_w = max(1, round(w * scale))
            t = F.interpolate(t, size=(new_h, new_w), mode=interp_mode, align_corners=False)
            resized = True

    if not resized:
        t = t.clone()

    # 2. Normalize --------------------------------------------------------
    if spec.normalization >= 0:
        mean_list, std_list = NORMALIZATION_PRESETS[spec.normalization]
        t = t / 255.0
        mean = torch.tensor(mean_list, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std_list, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
        t = (t - mean) / std

    # 3. Precision --------------------------------------------------------
    if spec.half_precision or spec.precision:
        t = t.to(spec.effective_dtype)

    return t
