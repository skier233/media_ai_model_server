"""Custom pipelines written to disk by an older server must gain the asset stage."""
import os

import yaml

from lib.migrations.migration_v41 import migrate_to_4_1


V40_VIDEO = {
    "inputs": ["video_path", "return_timestamps", "time_interval", "threshold",
               "return_confidence", "vr_video", "skipped_categories", "requested_model_names"],
    "output": "results",
    "short_name": "custom_video",
    "version": 4.0,
    "models": [
        {"name": "dynamic_video_ai", "inputs": ["video_path"], "outputs": ["childrenResults"]},
        {"name": "video_result_postprocessor_v4",
         "inputs": ["childrenResults", "video_path", "time_interval"], "outputs": ["results"]},
    ],
}


def _run_in(tmp_path, monkeypatch, pipelines):
    monkeypatch.chdir(tmp_path)
    os.makedirs("config/pipelines")
    for name, config in pipelines.items():
        with open(f"config/pipelines/{name}.yaml", "w") as handle:
            yaml.dump(config, handle, sort_keys=False)
    migrate_to_4_1()
    out = {}
    for name in pipelines:
        with open(f"config/pipelines/{name}.yaml") as handle:
            out[name] = yaml.safe_load(handle)
    return out


def test_adds_stage_and_postprocessor_input(tmp_path, monkeypatch):
    got = _run_in(tmp_path, monkeypatch, {"custom_video": V40_VIDEO})["custom_video"]
    names = [m["name"] for m in got["models"]]
    assert names == ["dynamic_video_ai", "dynamic_asset_ai", "video_result_postprocessor_v4"]
    post = got["models"][-1]
    assert post["inputs"] == ["childrenResults", "video_path", "time_interval", "assetResults"]
    assert got["version"] == 4.1


def test_is_idempotent(tmp_path, monkeypatch):
    once = _run_in(tmp_path, monkeypatch, {"custom_video": V40_VIDEO})["custom_video"]
    migrate_to_4_1()
    with open("config/pipelines/custom_video.yaml") as handle:
        twice = yaml.safe_load(handle)
    assert once == twice


def test_leaves_non_video_pipelines_alone(tmp_path, monkeypatch):
    audio = {
        "inputs": ["audio_path"], "output": "result", "short_name": "a", "version": 4.0,
        "models": [{"name": "audio_result_postprocessor_v4", "inputs": ["x"], "outputs": ["result"]}],
    }
    got = _run_in(tmp_path, monkeypatch, {"custom_audio": audio})["custom_audio"]
    assert got == audio


def test_existing_capabilities_entries_gain_an_explicit_asset_stage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("config/pipelines")
    with open("config/model_capabilities.yaml", "w") as handle:
        yaml.dump({"custom_video": {"full_image_models": "ALL"},
                   "other": {"full_image_models": "ALL"}}, handle, sort_keys=False)
    with open("config/pipelines/custom_video.yaml", "w") as handle:
        yaml.dump(V40_VIDEO, handle, sort_keys=False)
    migrate_to_4_1()
    with open("config/model_capabilities.yaml") as handle:
        capabilities = yaml.safe_load(handle)
    assert capabilities["custom_video"] == {"full_image_models": "ALL", "asset_models": "ALL"}
    assert capabilities["other"] == {"full_image_models": "ALL"}


def test_a_missing_capabilities_file_is_not_created(tmp_path, monkeypatch):
    _run_in(tmp_path, monkeypatch, {"custom_video": V40_VIDEO})
    assert not os.path.exists("config/model_capabilities.yaml")
