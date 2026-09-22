"""Models that analyse a whole video and drive their own decode.

``asset`` is also the target scope of whole-image and whole-audio analysis, so
a model's supported scopes cannot say whether it needs the video path rather
than frames from the shared preprocessor. The model type can: only these types
run in the asset-scope stage of a video pipeline, and they run only when a
request names them.
"""

# Model yaml ``type`` values whose models decode the whole video themselves.
WHOLE_ASSET_MODEL_TYPES = frozenset({"shot_boundary"})


def is_whole_asset_model(model) -> bool:
    """True for a model that consumes a whole video and decodes it itself.

    Accepts a model config dict (judged by its ``type``), a model instance
    (judged by its ``whole_asset_model`` class attribute), or a ModelProcessor
    or ModelWrapper around one.
    """
    if isinstance(model, dict):
        return str(model.get("type") or "").strip().lower() in WHOLE_ASSET_MODEL_TYPES
    candidate = model
    for _ in range(3):
        if candidate is None:
            return False
        if getattr(candidate, "whole_asset_model", False) is True:
            return True
        candidate = getattr(candidate, "model", None)
    return False
