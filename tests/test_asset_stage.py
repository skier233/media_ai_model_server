"""The asset-scope stage: collection, response shaping, and decode skipping."""
import asyncio
import types

import pytest

from lib.async_lib.async_processing import ItemFuture, QueueItem, Skip
from lib.config.model_capabilities import ModelCapabilitiesConfig
from lib.model.python_functions import asset_result_collector, _apply_asset_results
from lib.model.video_preprocessor import VideoPreprocessorModel


async def _future(data):
    async def noop(item_future, key):
        return None
    return await ItemFuture.create(None, dict(data), noop)


# ── asset_result_collector ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collector_completes_with_no_asset_models():
    """The common deployment installs no asset model; the stage must still finish."""
    future = await _future({"video_path": "/tmp/a.mp4"})
    await asset_result_collector([QueueItem(future, ["video_path"], ["assetResults"])])
    result = future["assetResults"]
    assert result is not None
    assert result["_outputs"] == []
    # The trigger input must not leak into the result.
    assert "video_path" not in result


@pytest.mark.asyncio
async def test_collector_gathers_results_and_excludes_trigger():
    future = await _future({"video_path": "/tmp/a.mp4", "shot_boundaries": {"boundaries": []}})
    await asset_result_collector(
        [QueueItem(future, ["video_path", "shot_boundaries"], ["assetResults"])]
    )
    result = future["assetResults"]
    assert result["shot_boundaries"] == {"boundaries": []}
    assert "video_path" not in result
    assert result["_outputs"] == [{"model_step": "shot_boundaries", "status": "ok"}]


@pytest.mark.asyncio
async def test_collector_records_skips_and_errors():
    future = await _future({
        "video_path": "/tmp/a.mp4",
        "skipped_thing": Skip(),
        "broken_thing": RuntimeError("boom"),
    })
    await asset_result_collector(
        [QueueItem(future, ["video_path", "skipped_thing", "broken_thing"], ["assetResults"])]
    )
    result = future["assetResults"]
    assert "skipped_thing" not in result and "broken_thing" not in result
    statuses = {o["model_step"]: o["status"] for o in result["_outputs"]}
    assert statuses == {"skipped_thing": "skipped", "broken_thing": "error"}
    assert result["_errors"][0]["model_step"] == "broken_thing"


@pytest.mark.asyncio
async def test_collector_names_an_exception_that_has_no_message():
    future = await _future({"video_path": "/tmp/a.mp4", "shot_boundaries": MemoryError()})
    await asset_result_collector([QueueItem(future, ["video_path", "shot_boundaries"], ["assetResults"])])
    assert future["assetResults"]["_errors"] == [{"model_step": "shot_boundaries", "error": "MemoryError"}]


@pytest.mark.asyncio
async def test_collector_reports_why_a_model_was_skipped():
    future = await _future({"video_path": "/tmp/a.mp4", "shot_boundaries": Skip("not_named")})
    await asset_result_collector([QueueItem(future, ["video_path", "shot_boundaries"], ["assetResults"])])
    assert future["assetResults"]["_outputs"] == [
        {"model_step": "shot_boundaries", "status": "skipped", "reason": "not_named"}
    ]


# ── response shaping ─────────────────────────────────────────────────────

def test_temporal_segmentation_lands_in_analysis_other():
    """The shape must match the image pipeline's asset-level `analysis`, which is
    what clients already know how to parse."""
    payload = {"frames": []}
    _apply_asset_results(
        payload,
        {"shot_boundaries": {"boundaries": [1]}, "_outputs": [{"model_step": "x", "status": "ok"}]},
        {"shot_boundaries": "temporal_segmentation"},
    )
    assert payload["analysis"] == {"other": {"shot_boundaries": {"boundaries": [1]}}}
    assert payload["asset_outputs"]


def test_known_capabilities_bucket_under_analysis_capabilities():
    payload = {"frames": []}
    _apply_asset_results(payload, {"whole_clip_vibe": [0.1]}, {"whole_clip_vibe": "embedding"})
    assert payload["analysis"] == {"capabilities": {"embedding": {"whole_clip_vibe": [0.1]}}}


def test_absent_stage_adds_nothing():
    payload = {"frames": []}
    _apply_asset_results(payload, None, {})
    assert payload == {"frames": []}


def test_stage_that_ran_with_no_models_adds_nothing():
    payload = {"frames": []}
    _apply_asset_results(payload, {"_outputs": []}, {})
    assert "analysis" not in payload


def test_skipped_model_is_not_emitted():
    payload = {"frames": []}
    _apply_asset_results(payload, {"shot_boundaries": Skip()}, {"shot_boundaries": "temporal_segmentation"})
    assert "analysis" not in payload


# ── decode skipping ──────────────────────────────────────────────────────

def _preprocessor(downstream):
    model = VideoPreprocessorModel({"type": "video_preprocessor", "frame_interval": 2})
    model.downstream_model_names = downstream
    return model


class _Future:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, key):
        return self.data.get(key)


@pytest.mark.parametrize("downstream,requested,expected", [
    (None, ["anything"], False),            # not wired by the dynamic manager
    (set(), None, True),                    # no frame-scope model active at all
    ({"tagger"}, None, False),              # no explicit request -> run everything
    ({"tagger"}, ["tagger"], False),        # requested model uses these frames
    ({"tagger"}, ["shot_boundaries"], True),  # only asset-scope models requested
    ({"tagger"}, [], False),                # empty request list means "no filter"
])
def test_should_skip_decode(downstream, requested, expected):
    model = _preprocessor(downstream)
    future = _Future({"requested_model_names": requested})
    inputs = ["video_path", "return_timestamps", "time_interval", "threshold",
              "return_confidence", "vr_video", "skipped_categories", "requested_model_names"]
    assert model._should_skip_decode(future, inputs) is expected


# ── model_capabilities asset stage ───────────────────────────────────────

def _capabilities_config(config):
    import logging
    return ModelCapabilitiesConfig(config, active_model_library=[], logger=logging.getLogger("test"))


def _model(name, capabilities, scopes, whole_asset=False):
    inner = types.SimpleNamespace(
        config_name=name, model_file_name=name,
        model_capabilities=capabilities, supported_target_scopes=scopes,
        model_type="ShotBoundary" if whole_asset else "Tagging",
        whole_asset_model=whole_asset,
    )
    return types.SimpleNamespace(model=inner, config_name=name)


def _boundaries():
    return _model("boundaries", ["temporal_segmentation"], ["asset"], whole_asset=True)


def test_asset_stage_all_selects_only_whole_asset_models():
    config = _capabilities_config({"p": {"asset_models": "ALL"}})
    models = [
        _boundaries(),
        _model("tagger", ["tagging"], ["frame"]),
        _model("embedder", ["embedding"], ["frame", "region"]),
        # `asset` is also the scope of whole-image analysis.
        _model("image_tagger", ["tagging"], ["asset"]),
    ]
    assert config.resolve_model_names_for_stage("p", "asset", models) == ["boundaries"]


def test_asset_stage_absent_means_every_whole_asset_model():
    """An entry written before the asset stage existed must not leave it
    unconstrained, or every frame model would be wired into it."""
    config = _capabilities_config({"p": {"full_image_models": "ALL"}})
    models = [_boundaries(), _model("tagger", ["tagging"], ["asset", "frame", "region"])]
    assert config.resolve_model_names_for_stage("p", "asset", models) == ["boundaries"]


def test_asset_stage_all_honours_exclude():
    config = _capabilities_config({"p": {"asset_models": {"mode": "ALL", "exclude": ["boundaries"]}}})
    other = _model("other_cuts", ["temporal_segmentation"], ["asset"], whole_asset=True)
    assert config.resolve_model_names_for_stage("p", "asset", [_boundaries(), other]) == ["other_cuts"]


def _pruned(config, active):
    import logging

    capabilities = ModelCapabilitiesConfig(config, active_model_library=active, logger=logging.getLogger("test"))
    capabilities.prune_unavailable_models()
    capabilities.validate()
    return capabilities


def test_asset_models_drop_inactive_models_at_startup():
    config = {"p": {"full_image_models": "ALL", "asset_models": ["boundaries", "never_installed"]}}
    capabilities = _pruned(config, ["boundaries", "tagger"])
    assert capabilities.config["p"]["asset_models"] == ["boundaries"]


def test_unloading_the_listed_asset_model_leaves_an_empty_stage_not_all():
    """The listed models were chosen; falling back to ALL would start others,
    and keeping the name would fail validation and the whole pipeline."""
    config = {"p": {"full_image_models": "ALL", "asset_models": ["boundaries"]}}
    before = _pruned(config, ["boundaries", "tagger"])
    assert before.config["p"]["asset_models"] == ["boundaries"]
    after = _pruned({"p": dict(config["p"])}, ["tagger"])        # boundaries unloaded
    assert after.config["p"]["asset_models"] == []
    other = _model("other_cuts", ["temporal_segmentation"], ["asset"], whole_asset=True)
    assert after.resolve_model_names_for_stage("p", "asset", [other]) == []


def _shipped_video_entry(asset_models):
    import copy

    from lib.configurator.configure_model_capabilities import DEFAULT_MODEL_CAPABILITIES_CONFIG

    entry = copy.deepcopy(DEFAULT_MODEL_CAPABILITIES_CONFIG["video_pipeline_dynamic_v4"])
    entry["asset_models"] = asset_models
    return {"video_pipeline_dynamic_v4": entry}


def test_asset_models_are_pruned_in_the_shipped_video_entry():
    """Its detector and region models are selectors, which exempt the entry
    from the name pruning; the asset_models list must be pruned regardless."""
    active = ["face_detector", "face_embedder"]
    unloaded = _pruned(_shipped_video_entry(["boundaries"]), active)
    assert unloaded.config["video_pipeline_dynamic_v4"]["asset_models"] == []
    loaded = _pruned(_shipped_video_entry(["boundaries"]), active + ["boundaries"])
    assert loaded.config["video_pipeline_dynamic_v4"]["asset_models"] == ["boundaries"]


def test_asset_models_pruning_keeps_selector_items():
    selector = {"categories": ["shot_boundaries"]}
    capabilities = _pruned(_shipped_video_entry(["gone", selector]), ["face_detector"])
    assert capabilities.config["video_pipeline_dynamic_v4"]["asset_models"] == [selector]


def test_asset_stage_accepts_an_explicit_list():
    config = _capabilities_config({"p": {"asset_models": ["boundaries"]}})
    models = [_boundaries()]
    assert config.resolve_model_names_for_stage("p", "asset", models) == ["boundaries"]


# ── stage isolation ──────────────────────────────────────────────────────

def _manager_with(models, capabilities=None):
    """A DynamicAIManager with model selection stubbed onto a fixed model set."""
    import logging
    from lib.pipeline.dynamic_ai_manager import DynamicAIManager

    manager = DynamicAIManager.__new__(DynamicAIManager)
    manager.models = models
    manager.models_by_config_name = {m.config_name: m for m in models}
    manager.loaded = True
    manager.logger = logging.getLogger("test")
    manager.model_capabilities = _capabilities_config(capabilities or {})
    return manager


def test_asset_only_model_is_never_selected_for_a_frame_stage():
    """A pipeline with no stage constraints (the v3 pipelines) must not wire a
    whole-asset model in as a per-frame model."""
    boundaries = _boundaries()
    tagger = _model("tagger", ["tagging"], ["frame"])
    manager = _manager_with([boundaries, tagger])

    frame_stage = manager._select_models(mode="video")
    assert [m.config_name for m in frame_stage] == ["tagger"]

    asset_stage = manager._select_models(mode="asset", required_scope="asset")
    assert [m.config_name for m in asset_stage] == ["boundaries"]


def test_a_model_supporting_both_scopes_still_serves_the_frame_stage():
    both = _model("both", ["tagging"], ["asset", "frame"])
    manager = _manager_with([both])
    assert [m.config_name for m in manager._select_models(mode="video")] == ["both"]


def test_frame_models_never_reach_the_asset_stage():
    """Not by default scopes, not through a missing asset_models key, and not
    through an asset_models list that names one by mistake."""
    boundaries = _boundaries()
    tagger = _model("tagger", ["tagging"], ["asset", "frame", "region"])
    for capabilities in ({}, {"p": {"full_image_models": "ALL"}}, {"p": {"asset_models": ["tagger", "boundaries"]}}):
        manager = _manager_with([tagger, boundaries], capabilities)
        selected = manager._select_models(mode="asset", required_scope="asset", pipeline_name="p")
        assert [m.config_name for m in selected] == ["boundaries"], capabilities


def test_an_image_model_with_asset_scope_still_serves_its_stage():
    """`asset` is the scope of whole-image analysis too; such a model is not a
    whole-video model and keeps its place in the image stage."""
    image_tagger = _model("image_tagger", ["tagging"], ["asset"])
    manager = _manager_with([image_tagger, _boundaries()])
    assert [m.config_name for m in manager._select_models(mode="image")] == ["image_tagger"]
    assert [m.config_name for m in manager._select_models(mode="asset", required_scope="asset")] == ["boundaries"]


def test_a_whole_asset_model_is_never_wired_as_a_detector():
    detector = _model("face_detector", ["detection"], ["frame"])
    manager = _manager_with(
        [detector, _boundaries()],
        {"p": {"detector_models": ["face_detector", "boundaries"]}},
    )
    assert manager._frame_detector_names("p") == ["face_detector"]


def test_an_audio_model_with_asset_scope_still_serves_its_stage():
    """A whole-audio embedder declares `asset` scope; it is not a whole-video model."""
    embedder = _model("audio_embedder", ["embedding"], ["asset"])
    embedder.model.model_type = "AudioEmbedding"
    manager = _manager_with([embedder, _boundaries()], {"audio_p": {"audio_models": "ALL"}})
    selected = manager._select_models(mode="audio", pipeline_name="audio_p")
    assert [m.config_name for m in selected] == ["audio_embedder"]


def test_an_audio_stage_without_audio_models_is_still_an_error():
    """Only a video stage may be empty because the pipeline has whole-video
    models to run."""
    manager = _manager_with(
        [_boundaries(), _model("tagger", ["tagging"], ["asset", "frame"])],
        {"audio_p": {"audio_models": "ALL"}},
    )
    with pytest.raises(ValueError, match="No active AI models matched"):
        manager.get_dynamic_models(mode="audio", inputs=["audio_path"], outputs=["audio_results"],
                                   pipeline_name="audio_p")


# ── whole-asset models run only when named ───────────────────────────────

def _processor(whole_asset, scopes=("asset",), name="boundaries", max_queue_size=None):
    from lib.async_lib.async_processing import ModelProcessor
    from lib.model.ai_model import AIModel

    model = AIModel({
        "model_file_name": name,
        "model_category": ["shot_boundaries"],
        "supported_target_scopes": list(scopes),
        "max_batch_size": 1,
        "instance_count": 1,
        "max_queue_size": max_queue_size,
    })
    model.config_name = name
    # AIModel drops max_queue_size on a CPU-only host; the test needs it as given.
    model.max_queue_size = max_queue_size
    if whole_asset:
        model.whole_asset_model = True
    return ModelProcessor(model)


async def _queue_item(requested, path="/tmp/a.mp4"):
    future = await _future({"video_path": path, "requested_model_names": requested})
    return future, QueueItem(future, ["video_path", "requested_model_names"], ["shot_boundaries"])


@pytest.mark.asyncio
@pytest.mark.parametrize("whole_asset,scopes,requested,reason", [
    (True, ["asset"], None, "not_named"),        # legacy endpoints and /v4 without `want`
    (True, ["asset"], [], "not_named"),
    (True, ["asset"], ["boundaries"], None),      # asked for by name
    (True, ["asset"], ["tagger"], "not_requested"),
    (False, ["frame"], None, None),               # no list still means "run everything"
    (False, ["asset"], None, None),               # whole-image and whole-audio models too
    (False, ["asset", "frame"], None, None),
])
async def test_whole_asset_models_run_only_when_named(whole_asset, scopes, requested, reason):
    processor = _processor(whole_asset, scopes)
    future, item = await _queue_item(requested)
    await processor.add_to_queue(item)
    if reason is None:
        assert processor.queue.qsize() == 1
        assert future["shot_boundaries"] is None
    else:
        assert processor.queue.qsize() == 0
        assert isinstance(future["shot_boundaries"], Skip)
        assert future["shot_boundaries"].reason == reason


@pytest.mark.asyncio
async def test_a_stage_without_requested_names_is_not_gated():
    """Only a request that carries the names, even as None, can be judged."""
    processor = _processor(True)
    future = await _future({"video_path": "/tmp/a.mp4"})
    await processor.add_to_queue(QueueItem(future, ["video_path"], ["shot_boundaries"]))
    assert processor.queue.qsize() == 1
    assert future["shot_boundaries"] is None


@pytest.mark.asyncio
async def test_a_skipped_request_never_waits_behind_a_busy_whole_asset_model():
    """The skip is decided when the item arrives, so a request that does not
    need the model is not queued behind analyses that take minutes, and a full
    queue cannot block it."""
    processor = _processor(True, max_queue_size=1)
    _busy, named = await _queue_item(["boundaries"], "/tmp/busy.mp4")
    await processor.add_to_queue(named)
    assert processor.queue.full()   # the single worker is busy with it

    for requested, reason in ((None, "not_named"), (["tagger"], "not_requested")):
        future, item = await _queue_item(requested, "/tmp/other.mp4")
        await asyncio.wait_for(processor.add_to_queue(item), timeout=1.0)
        assert future["shot_boundaries"].reason == reason
    assert processor.queue.qsize() == 1


def test_the_generated_default_selects_whole_asset_models_explicitly():
    from lib.configurator.configure_model_capabilities import DEFAULT_MODEL_CAPABILITIES_CONFIG

    assert DEFAULT_MODEL_CAPABILITIES_CONFIG["video_pipeline_dynamic_v4"]["asset_models"] == "ALL"
