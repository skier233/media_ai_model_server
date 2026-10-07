"""`_analyze` end to end with a scripted model: windowing, the tail, the frame
rate and failures.

No artifact, no GPU and no video: the frame source and the model are fakes, and
the model's logits are chosen so its greedy query walk yields scripted shots.
"""
import logging

import numpy as np
import pytest

import lib.model.ai_shot_boundary_model as sbm
from lib.async_lib.async_processing import ItemFuture, QueueItem
from lib.model.preprocessing_python.ffmpeg_pipe import VideoInfo
from lib.model.python_functions import _apply_asset_results, asset_result_collector
from lib.pipeline.preprocess_spec import PreprocessSpec

WIDTH, HEIGHT = 16, 12


class ScriptedModel:
    """Each call is one window; ``ends`` lists the window-relative end frame
    each query predicts. Every shot is General and entered by a hard cut."""

    def __init__(self, window, ends_per_window, queries=6):
        self.window = window
        self.ends_per_window = [list(ends) for ends in ends_per_window]
        self.queries = queries
        self.calls = 0

    def run_raw_multi_output(self, clip, use_half=True):
        assert tuple(clip.shape[:2]) == (1, self.window)
        ends = self.ends_per_window[min(self.calls, len(self.ends_per_window) - 1)]
        self.calls += 1
        shot = np.zeros((1, self.queries, self.window + 2), dtype=np.float32)
        for query in range(self.queries):
            # End frame 0 never starts a shot, so unscripted queries are skipped.
            shot[0, query, ends[query] if query < len(ends) else 0] = 10.0
        # The trailing class of each head is the "no object" slot.
        intra = np.zeros((1, self.queries, 3), dtype=np.float32)
        intra[..., 0] = 10.0   # General
        inter = np.zeros((1, self.queries, 3), dtype=np.float32)
        inter[..., 1] = 10.0   # Hard_Cut
        return shot, intra, inter


class FakeSource:
    backend_name = "fake"

    def __init__(self, count):
        self.count = count

    def __iter__(self):
        blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        return ((index, blank) for index in range(self.count))


def make_model(monkeypatch, *, window, context, ends_per_window, frames, fps=25.0,
               duration=None, fps_assumed=False):
    model = sbm.AIShotBoundaryModel.__new__(sbm.AIShotBoundaryModel)
    model.logger = logging.getLogger("test")
    model.config_name = model.model_file_name = "scripted_shots"
    model.model_version = 1.0
    model.mode = "default"
    model.window_frames = window
    model.context_frames = context
    model.decode_backend, model.decode_quality, model.decode_device_index = "auto", "exact", None
    model._spec = PreprocessSpec(width=WIDTH, height=HEIGHT, normalization=0)
    model._intra_names = {0: "General", 1: "Dissolve"}
    model._inter_names = {0: "New_Start", 1: "Hard_Cut"}
    model._general_intra_index = 0
    model._new_start_inter_index = 0
    model.model = ScriptedModel(window, ends_per_window)
    info = VideoInfo(WIDTH, HEIGHT, "yuv420p", "h264", fps,
                     frames / fps if duration is None else duration, frames,
                     "progressive", None, None, fps_assumed=fps_assumed)
    monkeypatch.setattr(sbm, "probe_video", lambda _path: info)
    monkeypatch.setattr(sbm, "make_video_frame_source", lambda *_args, **_kwargs: FakeSource(frames))
    return model


def spans(result):
    return [(b["start_frame"], b["end_frame"], b["filled"]) for b in result["boundaries"]]


def test_every_frame_is_scored_with_context_frames(monkeypatch):
    """With context frames one padded tail window leaves real frames in its
    trailing context; the tail keeps going until none are left."""
    model = make_model(monkeypatch, window=100, context=10, ends_per_window=[[100]], frames=165)
    result = model._analyze("/fake.mp4", False)
    assert spans(result) == [(0, 80, False), (80, 160, False), (160, 165, False)]
    assert model.model.calls == 3


def test_without_context_one_padded_tail_window_is_enough(monkeypatch):
    model = make_model(monkeypatch, window=100, context=0, ends_per_window=[[100]], frames=150)
    result = model._analyze("/fake.mp4", False)
    assert spans(result) == [(0, 100, False), (100, 150, False)]
    assert model.model.calls == 2


def test_a_window_that_stops_short_leaves_a_placed_gap(monkeypatch):
    """Each window is placed at its own first frame, so a walk that stops short
    leaves a filled gap rather than shifting every later shot earlier."""
    model = make_model(monkeypatch, window=100, context=0, ends_per_window=[[90], [50, 100]], frames=200)
    result = model._analyze("/fake.mp4", False)
    assert spans(result) == [(0, 90, False), (90, 100, True), (100, 150, False), (150, 200, False)]


class SizedSource:
    backend_name = "fake"

    def __init__(self, count, size):
        self.count, self.size = count, size

    def __iter__(self):
        width, height = self.size
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        return ((index, blank) for index in range(self.count))


def test_a_vr_video_is_padded_like_its_frames(monkeypatch):
    """Whatever size the decoded VR frames have, the padding added to the last
    window must have the same, or the window cannot be stacked."""
    model = make_model(monkeypatch, window=100, context=0, ends_per_window=[[100]], frames=150)
    native = (64, 32)   # a 2:1 side-by-side source
    info = VideoInfo(*native, "yuv420p", "h264", 25.0, 6.0, 150, "progressive", None, None)
    monkeypatch.setattr(sbm, "probe_video", lambda _path: info)

    def make_source(_path, *, decode_size=None, crop=None, **_kwargs):
        # The size a real decode yields: the requested size, else the crop,
        # else the source's own.
        size = decode_size or (crop[2:] if crop else native)
        return SizedSource(150, size)

    monkeypatch.setattr(sbm, "make_video_frame_source", make_source)
    result = model._analyze("/fake.mp4", True)
    assert spans(result) == [(0, 100, False), (100, 150, False)]


def test_frame_rate_is_measured_when_the_container_declares_none(monkeypatch):
    model = make_model(monkeypatch, window=100, context=0, ends_per_window=[[50, 100]], frames=200,
                       fps=30.0, duration=8.0, fps_assumed=True)
    result = model._analyze("/fake.mp4", False)
    assert result["fps"] == 25.0
    assert result["duration_mismatch_seconds"] == 0.0
    assert [b["end_seconds"] for b in result["boundaries"]] == [2.0, 4.0, 6.0, 8.0]


def test_a_duration_shorter_than_the_frames_is_reported(monkeypatch, caplog):
    """200 frames at 25 fps last 8.0 s, but the container says 5.0 s."""
    model = make_model(monkeypatch, window=100, context=0, ends_per_window=[[50, 100]], frames=200,
                       fps=25.0, duration=5.0)
    with caplog.at_level(logging.WARNING, logger="test"):
        result = model._analyze("/fake.mp4", False)
    assert "the duration is only 5.000s" in caplog.text
    assert result["duration_mismatch_seconds"] == -3.0
    # Seconds end at the duration, so the shot starting at 6.0 s has no room
    # and is absorbed into the one before it; frames still cover everything.
    assert [(b["start_frame"], b["end_frame"], b["end_seconds"]) for b in result["boundaries"]] == [
        (0, 50, 2.0), (50, 100, 4.0), (100, 200, 5.0),
    ]


def test_a_slightly_longer_duration_stretches_the_last_shot_quietly(monkeypatch, caplog):
    """Containers often run a little past the last frame; that is not an error."""
    model = make_model(monkeypatch, window=100, context=0, ends_per_window=[[50, 100]], frames=200,
                       fps=25.0, duration=8.12)
    with caplog.at_level(logging.WARNING, logger="test"):
        result = model._analyze("/fake.mp4", False)
    assert caplog.text == ""
    assert result["duration_mismatch_seconds"] == 0.12
    assert result["boundaries"][-1]["end_seconds"] == 8.12


async def _noop(_future, _key):
    return None


@pytest.mark.asyncio
async def test_a_failed_analysis_is_reported_without_failing_the_request(monkeypatch, tmp_path, caplog):
    model = make_model(monkeypatch, window=100, context=0, ends_per_window=[[100]], frames=100)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")

    def fail(_path, _vr):
        raise RuntimeError("decode ended early")

    monkeypatch.setattr(model, "_analyze", fail)
    future = await ItemFuture.create(None, {"video_path": str(video), "vr_video": False}, _noop)
    with caplog.at_level(logging.ERROR, logger="test"):
        await model.worker_function([QueueItem(future, ["video_path", "vr_video"], ["shot_boundaries"])])
    assert "Traceback" in caplog.text   # the log keeps the stack, not just the message

    assert not future.future.done(), "the request itself must not fail"
    assert isinstance(future["shot_boundaries"], RuntimeError)

    await asset_result_collector([QueueItem(future, ["video_path", "shot_boundaries"], ["assetResults"])])
    payload = {"frames": [{"frame_index": 0, "actions": ["kept"]}]}
    _apply_asset_results(payload, future["assetResults"], {"shot_boundaries": "temporal_segmentation"})
    assert payload["frames"] == [{"frame_index": 0, "actions": ["kept"]}]
    assert "analysis" not in payload
    assert payload["asset_errors"] == [{"model_step": "shot_boundaries", "error": "decode ended early"}]


# ── load-time validation ─────────────────────────────────────────────────

def _labels_model(tmp_path, monkeypatch, labels, **config):
    import json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "shots.labels.json").write_text(json.dumps(labels))
    return sbm.AIShotBoundaryModel({"model_file_name": "shots", "model_category": ["shot_boundaries"], **config})


LABELS = {
    "window_frames": 100,
    "frame_width": 128,
    "frame_height": 96,
    "intra": {"0": "General", "1": "Dissolve"},
    "inter": {"0": "New_Start", "1": "Hard_Cut"},
    "intra_name_to_index": {"general": 0, "dissolve": 1},
    "inter_name_to_index": {"new_start": 0, "hard_cut": 1},
}


def test_labels_load(tmp_path, monkeypatch):
    model = _labels_model(tmp_path, monkeypatch, LABELS, mode="clean_shot")
    model._load_labels()
    assert (model.window_frames, model._general_intra_index, model._new_start_inter_index) == (100, 0, 0)


def test_an_unknown_mode_is_rejected(tmp_path, monkeypatch):
    model = _labels_model(tmp_path, monkeypatch, LABELS, mode="clean-shot")
    with pytest.raises(ValueError, match="mode must be one of default, clean_shot"):
        model._load_labels()


def test_a_missing_new_start_label_is_rejected(tmp_path, monkeypatch):
    """Without it, every shot continuing across a window seam would be split."""
    labels = dict(LABELS, inter_name_to_index={"hard_cut": 1})
    model = _labels_model(tmp_path, monkeypatch, labels)
    with pytest.raises(ValueError, match="'new_start' is not in"):
        model._load_labels()


@pytest.mark.asyncio
async def test_an_encrypted_artifact_is_rejected_before_loading(tmp_path, monkeypatch):
    model = _labels_model(tmp_path, monkeypatch, LABELS, model_license_name="licenseV1.1")
    with pytest.raises(ValueError, match="must be unencrypted"):
        await model.load()


@pytest.mark.parametrize("categories", [["shot_boundaries", "cuts"], [], None])
def test_a_shot_model_needs_exactly_one_category(tmp_path, monkeypatch, categories):
    """The result is emitted under the model's one output, its category."""
    import asyncio

    model = _labels_model(tmp_path, monkeypatch, LABELS, model_category=categories)
    with pytest.raises(ValueError, match="exactly one model_category"):
        asyncio.run(model.load())


def test_invalid_labels_leave_the_model_untouched(tmp_path, monkeypatch):
    labels = dict(LABELS, inter_name_to_index={"hard_cut": 1})
    model = _labels_model(tmp_path, monkeypatch, labels)
    with pytest.raises(ValueError):
        model._load_labels()
    assert model.labels is None and model._spec is None and model._intra_names == {}


@pytest.mark.asyncio
async def test_the_sidecar_is_checked_before_the_artifact_is_loaded(tmp_path, monkeypatch):
    from lib.model.ai_model import AIModel

    loaded = []

    async def artifact_load(self):
        loaded.append(self)

    monkeypatch.setattr(AIModel, "load", artifact_load)
    model = _labels_model(tmp_path, monkeypatch, dict(LABELS, inter_name_to_index={}))
    with pytest.raises(ValueError, match="'new_start' is not in"):
        await model.load()
    assert loaded == []

    (tmp_path / "good").mkdir()
    good = _labels_model(tmp_path / "good", monkeypatch, LABELS)
    await good.load()
    assert loaded == [good]
    assert (good._spec.width, good._spec.height, good._spec.normalization) == (128, 96, 0)


# ── clip geometry and normalisation ──────────────────────────────────────

def _spec_model(tmp_path, monkeypatch, sidecar_size, preprocess_config):
    labels = {key: value for key, value in LABELS.items() if key not in ("frame_width", "frame_height")}
    if sidecar_size:
        labels.update(frame_width=sidecar_size[0], frame_height=sidecar_size[1])
    config = {} if preprocess_config is None else {"preprocess_config": preprocess_config}
    return _labels_model(tmp_path, monkeypatch, labels, **config)


@pytest.mark.parametrize("sidecar_size,preprocess_config", [
    ((128, 96), None),                                        # from the sidecar alone
    (None, {"width": 128, "height": 96, "normalization": 0}), # from the yaml alone
    ((128, 96), {"width": 128, "height": 96}),                # both, agreeing
])
def test_clip_geometry_comes_from_the_yaml_or_the_sidecar(tmp_path, monkeypatch, sidecar_size, preprocess_config):
    model = _spec_model(tmp_path, monkeypatch, sidecar_size, preprocess_config)
    model._load_labels()
    spec = model._spec
    assert (spec.width, spec.height, spec.normalization, spec.precision) == (128, 96, 0, "float32")


@pytest.mark.parametrize("sidecar_size,preprocess_config,message", [
    ((128, 96), {"width": 160, "height": 96}, "trained at one geometry"),
    (None, None, "clip geometry is unknown"),
    (None, {"normalization": 0}, "clip geometry is unknown"),   # never a silent 512x512
    (None, {"width": 128}, "both a positive width and height"),
    ((128, 96), {"normalization": 1}, "ImageNet-normalised"),
])
def test_bad_clip_settings_are_rejected(tmp_path, monkeypatch, sidecar_size, preprocess_config, message):
    model = _spec_model(tmp_path, monkeypatch, sidecar_size, preprocess_config)
    with pytest.raises(ValueError, match=message):
        model._load_labels()


# ── the worker ───────────────────────────────────────────────────────────

def _worker_model(monkeypatch):
    model = make_model(monkeypatch, window=100, context=0, ends_per_window=[[100]], frames=100)
    analysed = []

    def analyze(path, vr_video):
        analysed.append(path)
        return {"boundaries": []}

    monkeypatch.setattr(model, "_analyze", analyze)
    return model, analysed


@pytest.mark.asyncio
async def test_a_request_that_already_finished_is_not_analysed(monkeypatch, tmp_path):
    """A timed-out request's future is cancelled; minutes of decode would be wasted."""
    model, analysed = _worker_model(monkeypatch)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    future = await ItemFuture.create(None, {"video_path": str(video), "vr_video": False}, _noop)
    future.future.cancel()
    await model.worker_function([QueueItem(future, ["video_path", "vr_video"], ["shot_boundaries"])])
    assert analysed == []


@pytest.mark.asyncio
async def test_the_video_path_is_normalised_like_the_frame_stage(monkeypatch, tmp_path):
    model, analysed = _worker_model(monkeypatch)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    future = await ItemFuture.create(None, {"video_path": f"  {video}  ", "vr_video": False}, _noop)
    await model.worker_function([QueueItem(future, ["video_path", "vr_video"], ["shot_boundaries"])])
    assert analysed == [str(video)]

    missing = await ItemFuture.create(None, {"video_path": str(tmp_path / "gone.mp4"), "vr_video": False}, _noop)
    await model.worker_function([QueueItem(missing, ["video_path", "vr_video"], ["shot_boundaries"])])
    assert isinstance(missing["shot_boundaries"], FileNotFoundError)
    assert analysed == [str(video)]
