from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict

from lib.model.postprocessing import tag_models

class ImagePathList(BaseModel):
    paths: List[str]
    pipeline_name: str = None
    threshold: float = None
    return_confidence: bool = None

class VideoPathList(BaseModel):
    path: str
    returnTimestamps: bool = True
    pipeline_name: str = None
    frame_interval: float = None
    threshold: float = None
    return_confidence: bool = None
    vr_video: bool = False
    existing_json_data: Any = None

class VideoRequestV3(BaseModel):
    path: str
    categories_to_skip: List[str] = None
    frame_interval: float = None
    threshold: float = None
    return_confidence: bool = None
    vr_video: bool = False

class ImageRequestV3(BaseModel):
    paths: List[str]
    threshold: float = None
    return_confidence: bool = None


class AnalyzeWantV4(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    capability: str | None = None
    capabilities: List[str] | None = None
    category: str | None = None
    categories: List[str] | None = None
    model_category: str | None = None
    scope: str | None = None
    scopes: List[str] | None = None
    models: List[str] | None = None
    from_detection: str | None = None


class ImageRequestV4(BaseModel):
    paths: List[str]
    pipeline_name: str | None = None
    threshold: float = None
    return_confidence: bool = True
    categories_to_skip: List[str] | None = None
    want: List[AnalyzeWantV4] | None = None
    load_policy: str = "use_loaded"


class VideoRequestV4(BaseModel):
    path: str
    pipeline_name: str | None = None
    frame_interval: float = None
    threshold: float = None
    return_confidence: bool = True
    vr_video: bool = False
    categories_to_skip: List[str] | None = None
    want: List[AnalyzeWantV4] | None = None
    load_policy: str = "use_loaded"


class AudioRequestV4(BaseModel):
    paths: List[str]
    pipeline_name: str | None = None
    threshold: float = None
    want: List[AnalyzeWantV4] | None = None
    load_policy: str = "use_loaded"


class ModelSelectionRequestV4(BaseModel):
    models: List[str]


class TextEncodeRequestV4(BaseModel):
    text: str
    kind_family: str


class CustomPipelineRequestV4(BaseModel):
    pipeline_name: str
    display_name: str | None = None
    media_kind: str
    capability_id: str | None = None
    description: str | None = None
    capability_ids: List[str] | None = None
    claim_ids: List[str] | None = None
    use_all_full_image_models: bool = False
    full_image_models: List[str] | None = None
    detector_models: List[str] | None = None
    region_models: Dict[str, List[str]] | None = None
    use_all_audio_models: bool = False
    audio_models: List[str] | None = None
    use_all_asset_models: bool = True
    asset_models: List[str] | None = None

class OptimizeMarkerSettings(BaseModel):
    existing_json_data: Any = None
    desired_timespan_data: Any = None

class VideoResult(BaseModel):
    result: Any

class ImageResult(BaseModel):
    result: Any
    models: List[Any] | None = None

class AIModelInfo(BaseModel):
    name: str
    config_name: str | None = None
    identifier: int | None = None
    version: float | None = None
    categories: List[str] | None = None
    type: str
    capabilities: List[str] | None = None
    supported_scopes: List[str] | None = None

class AudioRequest(BaseModel):
    paths: List[str]
    threshold: float = None