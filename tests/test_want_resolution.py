"""`want` must be able to select an asset-scope model on a video request.

/v4/analyze/video passes default_scope="frame". Before the asset stage existed
that was harmless, because every selectable model was frame-scoped. Applying it
to an explicit capability ask would now silently match nothing and 400.
"""
import types

import pytest

from lib.server import v4_service

CATALOG_FUNCTION = v4_service.get_model_catalog


CATALOG = [
    {"config_name": "tagger", "capabilities": ["tagging"],
     "categories": ["actions"], "supported_scopes": ["frame"]},
    {"config_name": "face_detector", "capabilities": ["detection"],
     "categories": ["face_detections"], "supported_scopes": ["frame"]},
    {"config_name": "boundaries", "capabilities": ["temporal_segmentation"],
     "categories": ["shot_boundaries"], "supported_scopes": ["asset"], "whole_asset": True},
    # `asset` is also the scope of whole-image analysis.
    {"config_name": "image_tagger", "capabilities": ["tagging"],
     "categories": ["image_tags"], "supported_scopes": ["asset"]},
]


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    monkeypatch.setattr(v4_service, "get_model_catalog", lambda _sm: CATALOG)


def want(**kwargs):
    fields = dict(models=None, capability=None, capabilities=None,
                  category=None, categories=None, model_category=None,
                  scope=None, scopes=None)
    fields.update(kwargs)
    return types.SimpleNamespace(**fields)


def resolve(items, default_scope="frame"):
    return v4_service.resolve_want_model_names(None, items, default_scope)


def test_capability_ask_reaches_asset_scope_despite_frame_default():
    assert resolve([want(capability="temporal_segmentation")]) == ["boundaries"]


def test_category_ask_reaches_asset_scope_despite_frame_default():
    assert resolve([want(category="shot_boundaries")]) == ["boundaries"]


def test_explicit_scope_still_honoured():
    assert resolve([want(capability="temporal_segmentation", scope="asset")]) == ["boundaries"]
    with pytest.raises(ValueError):
        resolve([want(capability="temporal_segmentation", scope="frame")])


def test_bare_want_item_still_uses_the_default_scope():
    # No capability/category: the default scope is the only filter, and it must
    # still exclude asset-scope models from a plain frame request.
    assert resolve([want()]) == ["tagger", "face_detector"]


def test_a_video_tagging_ask_pulls_in_neither_asset_nor_image_models():
    assert resolve([want(capability="tagging")]) == ["tagger"]


def test_explicit_model_names_bypass_scope_filtering():
    assert resolve([want(models=["boundaries"])]) == ["boundaries"]


def test_combined_request_selects_both_stages():
    got = resolve([want(capability="tagging"), want(capability="temporal_segmentation")])
    assert got == ["tagger", "boundaries"]


def test_explicit_model_list_excludes_the_asset_model():
    # A client that names every model it wants never selects an installed
    # asset-scope model by accident.
    assert resolve([want(models=["tagger"])]) == ["tagger"]


def test_image_requests_keep_their_default_scope():
    assert resolve([want(capability="tagging")], default_scope="asset") == ["image_tagger"]
    with pytest.raises(ValueError):
        resolve([want(capability="detection")], default_scope="asset")


@pytest.mark.parametrize("item", [
    want(),                                      # the default asset scope alone
    want(scope="asset"),                         # an explicit asset scope
])
def test_image_and_audio_scope_asks_never_match_a_whole_video_model(item):
    assert resolve([item], default_scope="asset") == ["image_tagger"]


def test_a_whole_video_capability_is_not_available_to_image_requests():
    with pytest.raises(ValueError):
        resolve([want(capability="temporal_segmentation")], default_scope="asset")


def test_a_shot_boundary_artifact_needs_its_label_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "shots.pt2").write_bytes(b"")
    config = {"yaml_file_name": "shots", "model_file_name": "shots", "type": "shot_boundary"}
    available, path = v4_service._model_artifact_status(config)
    assert (available, path) == (False, "models/shots.labels.json")
    (tmp_path / "models" / "shots.labels.json").write_text("{}")
    assert v4_service._model_artifact_status(config)[0] is True
    # Other models never needed one.
    assert v4_service._model_artifact_status(dict(config, type="python_model"))[0] is True


def test_an_encrypted_shot_model_is_unavailable(monkeypatch):
    config = {"yaml_file_name": "shots", "model_file_name": "shots", "type": "shot_boundary",
              "model_category": ["shot_boundaries"], "model_license_name": "licenseV1.1"}
    monkeypatch.setattr(v4_service, "load_active_ai_models", lambda: [])
    monkeypatch.setattr(v4_service, "load_available_ai_models", lambda: [config])
    monkeypatch.setattr(v4_service, "_model_artifact_status", lambda _config: (True, "models/shots.pt2.enc"))
    server_manager = types.SimpleNamespace(pipeline_manager=types.SimpleNamespace(pipelines={}))
    [entry] = CATALOG_FUNCTION(server_manager)
    assert entry["incompatible"] is True
    assert "must be unencrypted" in entry["incompatibility_reason"]

    monkeypatch.setattr(v4_service, "_get_available_models_by_name", lambda: {"shots": config})
    with pytest.raises(ValueError, match="must be unencrypted"):
        v4_service._validate_models_can_activate(["shots"], [])
