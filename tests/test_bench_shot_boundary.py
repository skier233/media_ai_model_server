"""Smoke tests for scripts/bench_shot_boundary.py's model levels.

The benchmark drives the production model through its private seams, so a
change to them breaks it without breaking the server. These run the model
levels with a stub artifact: no GPU and no real model.
"""
import importlib.util
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import lib.model.ai_shot_boundary_model as sbm
from lib.model.preprocessing_python import ffmpeg_pipe as fp
from lib.pipeline.preprocess_spec import PreprocessSpec

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_shot_boundary.py"
WINDOW = 20


def _load_bench():
    # The benchmark is Linux only: it imports `resource` and reads /proc.
    pytest.importorskip("resource")
    spec = importlib.util.spec_from_file_location("bench_shot_boundary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubArtifact:
    """Every query ends at the window's last frame: one shot per window."""

    def __call__(self, clip):
        queries = 4
        shot = np.zeros((1, queries, WINDOW + 2), dtype=np.float32)
        shot[0, :, WINDOW] = 10.0
        intra = np.zeros((1, queries, 3), dtype=np.float32)
        intra[..., 0] = 10.0
        inter = np.zeros((1, queries, 3), dtype=np.float32)
        inter[..., 1] = 10.0
        return shot, intra, inter


class StubRunner:
    """Stands in for PythonModel: run_raw_multi_output calls the module, as it does."""

    device = torch.device("cpu")

    def __init__(self):
        self.model = StubArtifact()

    def run_raw_multi_output(self, clip, use_half=True):
        return self.model(clip)


def _stub_model():
    model = sbm.AIShotBoundaryModel.__new__(sbm.AIShotBoundaryModel)
    model.logger = logging.getLogger("test")
    model.config_name = model.model_file_name = "stub_shots"
    model.model_version = 1.0
    model.mode = "default"
    model.window_frames = WINDOW
    model.context_frames = 0
    model.decode_backend, model.decode_quality, model.decode_device_index = "auto", "exact", None
    model._spec = PreprocessSpec(width=32, height=24, normalization=0, precision="float32")
    model._intra_names = {0: "General", 1: "Dissolve"}
    model._inter_names = {0: "New_Start", 1: "Hard_Cut"}
    model._general_intra_index = 0
    model._new_start_inter_index = 0
    model.model = StubRunner()
    return model


def _context(bench, model):
    features = fp.ffmpeg_features()
    if features is None:
        pytest.skip("ffmpeg not installed")
    config = {"model_file_name": "stub_shots", "decode": {"backend": "auto"},
              "preprocess_config": {"width": 32, "height": 24, "normalization": 0}}
    return bench.Context(config, torch, sbm, model, features, 0.3, [])


def test_inference_microbench_runs_the_current_model():
    bench = _load_bench()
    result = bench.inference_microbench(_context(bench, _stub_model()), windows=3, warmup=1)
    assert result["windows"] == 3 and result["window_frames"] == WINDOW
    assert result["ms_per_window"] > 0 and result["model_call_ms"] >= 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_measure_model_analyses_a_real_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30",
         "-t", "2", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(clip)],
        check=True,
    )
    bench = _load_bench()
    model = _stub_model()
    result = bench.measure_model(_context(bench, model), {"excerpt": {"file": str(clip)}}, "L4")
    detail = result["detail"]
    assert result["frames"] == 60
    assert detail["backend"] == "ffmpeg_cpu"
    assert detail["windows"] == 3 and detail["boundaries"] == 3
    assert detail["first_frame_s"] is not None and len(detail["boundaries_sha1"]) == 12
    # Instrumentation is undone: the model's own methods are back.
    assert "_run_window" not in vars(model) and "run_raw_multi_output" not in vars(model.model)


def test_clip_size_falls_back_to_the_sidecar(tmp_path, monkeypatch):
    bench = _load_bench()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "stub_shots.labels.json").write_text('{"frame_width": 128, "frame_height": 96}')
    assert bench.clip_size({"model_file_name": "stub_shots"}) == (128, 96)
    assert bench.clip_size({"model_file_name": "stub_shots",
                            "preprocess_config": {"width": 64, "height": 48}}) == (64, 48)
    with pytest.raises(SystemExit, match="no clip geometry"):
        bench.clip_size({"model_file_name": "missing"})
