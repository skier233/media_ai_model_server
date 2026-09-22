"""Migration to pipeline schema 4.1: the asset-scope stage.

v4.1 adds a ``dynamic_asset_ai`` stage to video pipelines, feeding an extra
``assetResults`` input to ``video_result_postprocessor_v4``.  The shipped
pipeline yaml is updated in place by an install, but custom pipelines
registered through ``POST /v4/pipelines/custom`` were written to disk by an
older server and would otherwise silently never produce asset-scope output.
Their ``model_capabilities.yaml`` entries gain ``asset_models: ALL`` so the new
stage's selection is explicit.
"""

import glob
import os

import yaml

CAPABILITIES_PATH = "./config/model_capabilities.yaml"

ASSET_STAGE = {
    "name": "dynamic_asset_ai",
    "inputs": ["video_path", "vr_video", "skipped_categories", "requested_model_names"],
    "outputs": ["assetResults"],
}


def migrate_to_4_1():
    for pipeline_path in sorted(glob.glob("./config/pipelines/*.yaml")):
        try:
            with open(pipeline_path, "r") as handle:
                config = yaml.safe_load(handle)
        except Exception:
            continue
        if not isinstance(config, dict):
            continue

        models = config.get("models")
        if not isinstance(models, list):
            continue

        postprocessor = next(
            (m for m in models
             if isinstance(m, dict) and m.get("name") == "video_result_postprocessor_v4"),
            None,
        )
        if postprocessor is None:
            continue

        inputs = list(postprocessor.get("inputs") or [])
        has_stage = any(
            isinstance(m, dict) and m.get("name") == "dynamic_asset_ai" for m in models
        )
        if has_stage and "assetResults" in inputs:
            continue

        if not has_stage:
            models.insert(models.index(postprocessor), dict(ASSET_STAGE))
        if "assetResults" not in inputs:
            inputs.append("assetResults")
            postprocessor["inputs"] = inputs

        config["models"] = models
        config["version"] = 4.1

        with open(pipeline_path, "w") as handle:
            yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
        _select_all_asset_models(os.path.splitext(os.path.basename(pipeline_path))[0])
        print(f"Migrated {os.path.basename(pipeline_path)} to pipeline schema 4.1")


def _select_all_asset_models(pipeline_name):
    """Give an existing capabilities entry for ``pipeline_name`` an explicit asset stage."""
    try:
        with open(CAPABILITIES_PATH, "r", encoding="utf-8") as handle:
            capabilities = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return
    entry = capabilities.get(pipeline_name) if isinstance(capabilities, dict) else None
    if not isinstance(entry, dict) or "asset_models" in entry:
        return
    entry["asset_models"] = "ALL"
    with open(CAPABILITIES_PATH, "w", encoding="utf-8") as handle:
        yaml.dump(capabilities, handle, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    migrate_to_4_1()
