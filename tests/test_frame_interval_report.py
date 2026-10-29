"""Video results report the frame interval the video was actually sampled at.

Everything here runs without a GPU.
"""
import shutil
import subprocess
import types

import pytest

from lib.async_lib.async_processing import ItemFuture, QueueItem
from lib.model import python_functions
from lib.model.video_preprocessor import EFFECTIVE_FRAME_INTERVAL_KEY, VideoPreprocessorModel

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

CONFIG_FRAME_INTERVAL = 2

V4_INPUTS = ["video_path", "return_timestamps", "time_interval", "threshold",
             "return_confidence", "vr_video", "skipped_categories", "requested_model_names"]


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Ten seconds at 30 fps with a keyframe every half second."""
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x90:rate=30", "-t", "10",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "15",
         str(path)],
        check=True,
    )
    return path


async def _future(data):
    async def noop(item_future, key):
        return None
    return await ItemFuture.create(None, dict(data), noop)


def _pipeline():
    return types.SimpleNamespace(get_ai_models_info=lambda: [])


async def _preprocess(path, requested_interval):
    """Run the video preprocessor on ``path``; return the future and frame times."""
    model = VideoPreprocessorModel({
        "type": "video_preprocessor",
        "frame_interval": CONFIG_FRAME_INTERVAL,
    })
    future = await _future({
        "video_path": str(path),
        "return_timestamps": True,
        "time_interval": requested_interval,
        "threshold": None,
        "return_confidence": None,
        "vr_video": False,
        "skipped_categories": None,
        "requested_model_names": None,
    })
    outputs = ["children", "frame_index", "dynamic_threshold", "dynamic_return_confidence",
               "dynamic_skipped_categories", "dynamic_requested_model_names"]
    await model.worker_function([QueueItem(future, V4_INPUTS, outputs)])
    children = future["children"]
    assert children, "the preprocessor produced no frames"
    return future, [child["frame_index"] for child in children]


async def _v4_result(future, frame_times):
    await future.set_data("pipeline", _pipeline())
    await future.set_data("childrenResults", [{"frame_index": t} for t in frame_times])
    inputs = ["childrenResults", "video_path", "time_interval", "assetResults"]
    await python_functions.video_result_postprocessor_v4([QueueItem(future, inputs, ["results"])])
    return future["results"]


# ── end to end: decode, then the v4 response ─────────────────────────────

@needs_ffmpeg
@pytest.mark.asyncio
@pytest.mark.parametrize("requested,expected", [
    (None, CONFIG_FRAME_INTERVAL),  # no interval requested: the configured one is used
    (0, CONFIG_FRAME_INTERVAL),     # 0 also falls back to the configured interval
    (1, 1),                         # a requested interval is used as is
])
async def test_v4_reports_the_interval_frames_were_sampled_at(clip, requested, expected):
    future, frame_times = await _preprocess(clip, requested)
    assert future[EFFECTIVE_FRAME_INTERVAL_KEY] == expected

    result = await _v4_result(future, frame_times)

    assert result["frame_interval_seconds"] == float(expected)
    times = [frame["time_seconds"] for frame in result["frames"]]
    gaps = {round(b - a, 3) for a, b in zip(times, times[1:])}
    assert gaps == {float(expected)}


# ── each result builder reads the recorded interval ──────────────────────

def _capture(monkeypatch, cls):
    """Record the server result a builder hands to ``cls.from_server_result``."""
    captured = {}
    original = cls.from_server_result

    def from_server_result(server_result, *args, **kwargs):
        captured.update(server_result)
        return original(server_result, *args, **kwargs)

    monkeypatch.setattr(cls, "from_server_result", from_server_result)
    return captured


async def _builder_future(clip, requested, effective):
    data = {
        "childrenResults": [],
        "video_path": str(clip),
        "time_interval": requested,
        "threshold": 0.5,  # the v1 builder requires one
        "existing_video_data": None,
        "skipped_categories": None,
        "assetResults": None,
        "pipeline": _pipeline(),
    }
    if effective is not None:
        data[EFFECTIVE_FRAME_INTERVAL_KEY] = effective
    return await _future(data)


async def _reported(monkeypatch, clip, builder, requested, effective):
    future = await _builder_future(clip, requested, effective)
    if builder == "v1":
        captured = _capture(monkeypatch, python_functions.AIVideoResult)
        inputs = ["childrenResults", "video_path", "time_interval", "threshold", "existing_video_data"]
        await python_functions.video_result_postprocessor([QueueItem(future, inputs, ["results"])])
        return captured["frame_interval"]
    if builder == "v3":
        captured = _capture(monkeypatch, python_functions.AIVideoResultV3)
        inputs = ["childrenResults", "video_path", "time_interval", "threshold", "skipped_categories"]
        await python_functions.video_result_postprocessor_v3([QueueItem(future, inputs, ["results"])])
        return captured["frame_interval"]
    inputs = ["childrenResults", "video_path", "time_interval", "assetResults"]
    await python_functions.video_result_postprocessor_v4([QueueItem(future, inputs, ["results"])])
    return future["results"]["frame_interval_seconds"]


@needs_ffmpeg
@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["v1", "v3", "v4"])
async def test_unrequested_interval_reports_the_effective_one(monkeypatch, clip, builder):
    assert await _reported(monkeypatch, clip, builder, None, CONFIG_FRAME_INTERVAL) == float(CONFIG_FRAME_INTERVAL)


@needs_ffmpeg
@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["v1", "v3", "v4"])
async def test_requested_interval_is_reported(monkeypatch, clip, builder):
    assert await _reported(monkeypatch, clip, builder, 1, 1) == 1.0
