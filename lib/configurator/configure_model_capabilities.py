import copy
import os
import yaml


model_capabilities_yaml_path = "./config/model_capabilities.yaml"


_DEFAULT_FACE_REGION_FLOW = {
    "detector_models": [
        {
            "capabilities": ["detection"],
            "categories": ["face_detections"],
        },
    ],
    "region_models": [
        {
            "detector": {
                "capabilities": ["detection"],
                "categories": ["face_detections"],
            },
            "models": [
                {
                    "capabilities": ["embedding"],
                    "categories": ["face_embeddings"],
                    "scopes": ["region"],
                },
            ],
        },
    ],
}


DEFAULT_MODEL_CAPABILITIES_CONFIG = {
    "image_pipeline_dynamic_v4": {
        "full_image_models": "ALL",
        **copy.deepcopy(_DEFAULT_FACE_REGION_FLOW),
    },
    "video_pipeline_dynamic_v4": {
        "full_image_models": "ALL",
        **copy.deepcopy(_DEFAULT_FACE_REGION_FLOW),
        "asset_models": "ALL",
    },
    "audio_pipeline_v4": {
        "audio_models": "ALL",
    },
}


def ensure_model_capabilities_config():
    if os.path.exists(model_capabilities_yaml_path):
        return

    os.makedirs(os.path.dirname(model_capabilities_yaml_path), exist_ok=True)
    with open(model_capabilities_yaml_path, "w", encoding="utf-8") as handle:
        yaml.dump(DEFAULT_MODEL_CAPABILITIES_CONFIG, handle, default_flow_style=False, sort_keys=False)


def load_model_capabilities_config():
    ensure_model_capabilities_config()
    with open(model_capabilities_yaml_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or copy.deepcopy(DEFAULT_MODEL_CAPABILITIES_CONFIG)


def save_model_capabilities_config(config):
    os.makedirs(os.path.dirname(model_capabilities_yaml_path), exist_ok=True)
    with open(model_capabilities_yaml_path, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
