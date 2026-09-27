"""Decode-pipeline tests. Everything here runs without a GPU."""

import shutil
import subprocess

import numpy as np
import pytest

from lib.model.preprocessing_python import ffmpeg_pipe as fp

UHD = fp.VideoInfo(
    width=3840, height=2160, pix_fmt="yuv420p", codec="h264", fps=59.94,
    duration=2663.0, nb_frames=159618, field_order="progressive",
    color_range="tv", color_matrix="bt709",
)


def _chain(info=UHD, target=(128, 96), hw=True, quality="exact", frame_step=1):
    plan = fp.plan_scale(info.width, info.height, target, quality)
    return fp.build_filter_chain(plan, frame_step, hw)


# ── scale planning ───────────────────────────────────────────────────────

def test_pyramid_never_exceeds_two_times_per_step():
    """scale_cuda has a fixed kernel with no ratio-adaptive widening, so a big
    single-step reduction aliases. Every step must stay within 2x."""
    plan = fp.plan_scale(3840, 2160, (128, 96))
    sizes = [(3840, 2160), *plan.gpu_steps]
    for (prev_w, prev_h), (next_w, next_h) in zip(sizes, sizes[1:]):
        assert prev_w / next_w <= 2.0 + 1e-9
        assert prev_h / next_h <= 2.0 + 1e-9
    last_w, last_h = plan.gpu_steps[-1]
    assert last_w / 128 <= 2.0 and last_h / 96 <= 2.0


def test_pyramid_preserves_aspect_ratio():
    plan = fp.plan_scale(3840, 2160, (128, 96))
    for width, height in plan.gpu_steps:
        assert abs((width / height) - (3840 / 2160)) < 0.02


def test_no_plan_when_already_at_target_or_unset():
    assert fp.plan_scale(128, 96, (128, 96)).is_noop
    assert fp.plan_scale(3840, 2160, None).is_noop


def test_fast_quality_skips_the_pyramid():
    assert fp.plan_scale(3840, 2160, (128, 96), "fast").gpu_steps == ()


# ── filter chain ─────────────────────────────────────────────────────────

def test_gpu_scaling_happens_before_the_download():
    """The whole point: shrink on the GPU so tiny frames cross PCIe."""
    chain = _chain()
    assert chain.index("scale_cuda") < chain.index("hwdownload")


def test_final_gpu_step_converts_pixel_format():
    """10-bit sources arrive as p010le; `hwdownload,format=nv12` alone fails."""
    steps = [s for s in _chain().split(",") if s.startswith("scale_cuda")]
    assert "format=nv12" in steps[-1]
    assert all("format=nv12" not in step for step in steps[:-1])


def test_chain_uses_literal_integers_not_expressions():
    assert "if(gt(" not in _chain()


def test_frame_step_uses_select_not_the_fps_filter():
    """`fps=` resamples to a wall-clock rate and changes the frame count."""
    chain = _chain(frame_step=4)
    assert "select=" in chain
    assert "fps=" not in chain
    assert chain.index("select=") < chain.index("scale_cuda")


def test_conversion_to_rgb_is_left_to_the_output_pixel_format():
    """A separate `format=rgb24` filter converts in a second swscale pass and
    measurably diverges from PyAV's single reformat."""
    assert "format=rgb24" not in _chain()
    assert "format=rgb24" not in _chain(hw=False)


def test_cpu_chain_scales_once_without_gpu_filters():
    chain = _chain(hw=False)
    assert "scale_cuda" not in chain and "hwdownload" not in chain
    assert chain == "scale=128:96:flags=bicubic"


# ── command ──────────────────────────────────────────────────────────────

def test_command_pins_frame_timing_and_never_resamples():
    command = fp.build_command("ffmpeg", "in.mp4", _chain(), True, True)
    assert "-fps_mode" in command and command[command.index("-fps_mode") + 1] == "passthrough"
    assert "-r" not in command and "-vsync" not in command


def test_command_falls_back_to_vsync_on_older_ffmpeg():
    command = fp.build_command("ffmpeg", "in.mp4", _chain(), True, False)
    assert "-vsync" in command and "-fps_mode" not in command


def test_hardware_flags_only_present_for_the_cuda_backend():
    hw = fp.build_command("ffmpeg", "in.mp4", _chain(), True, True, device_index=1)
    assert "-hwaccel" in hw and "cuda" in hw
    assert hw[hw.index("-hwaccel_device") + 1] == "1"
    assert "-hwaccel" not in fp.build_command("ffmpeg", "in.mp4", _chain(hw=False), False, True)


def test_raw_output_is_rgb24_on_stdout():
    command = fp.build_command("ffmpeg", "in.mp4", _chain(), True, True)
    assert command[-3:] == ["-pix_fmt", "rgb24", "pipe:1"]
    assert "-f" in command and command[command.index("-f") + 1] == "rawvideo"


@pytest.mark.parametrize("hw", [True, False])
def test_command_decodes_frames_as_stored(hw):
    """PyAV ignores a display rotation; so must ffmpeg, or the two backends
    hand the model different frames and a VR crop lands on a rotated frame."""
    command = fp.build_command("ffmpeg", "in.mp4", _chain(hw=hw), hw, True)
    assert command[command.index("-i") - 1] == "-noautorotate"


# ── capability matrix ────────────────────────────────────────────────────

@pytest.mark.parametrize("pix_fmt,field_order,supported", [
    ("yuv420p", "progressive", True),
    ("yuv420p10le", "progressive", True),
    ("yuv444p", "progressive", False),        # NVDEC cannot decode 4:4:4
    ("yuv420p12le", "progressive", False),    # nor 12-bit
    ("yuv420p", "tt", False),                 # deinterlacing would change the count
])
def test_nvdec_capability_matrix(pix_fmt, field_order, supported):
    info = fp.VideoInfo(1920, 1080, pix_fmt, "h264", 30.0, 4.0, 120,
                        field_order, "tv", "bt709")
    assert fp.nvdec_supports(info)[0] is supported


@pytest.mark.parametrize("codec,supported", [
    ("h264", True), ("hevc", True), ("av1", True), ("vc1", True), ("mpeg4", True),
    # NVDEC has no decoder for these, so trying is a wasted smoke test.
    ("wmv3", False), ("wmv2", False), ("msmpeg4v3", False), ("prores", False),
])
def test_nvdec_codec_allowlist(codec, supported):
    info = fp.VideoInfo(1920, 1080, "yuv420p", codec, 30.0, 4.0, 120,
                        "progressive", "tv", "bt709")
    assert fp.nvdec_supports(info)[0] is supported


def test_unknown_geometry_is_rejected():
    info = fp.VideoInfo(0, 0, "yuv420p", "h264", 30.0, 4.0, 120,
                        "progressive", None, None)
    assert fp.nvdec_supports(info)[0] is False


# ── enum normalisation ───────────────────────────────────────────────────

def test_pyav_integer_enums_are_named():
    """PyAV 18 returns raw ffmpeg enum integers, not objects with .name."""
    assert fp._enum_name(1, fp._FIELD_ORDER_NAMES) == "progressive"
    assert fp._enum_name(1, fp._COLOR_RANGE_NAMES) == "tv"
    assert fp._enum_name(1, fp._COLOR_SPACE_NAMES) == "bt709"
    assert fp._enum_name(None, fp._FIELD_ORDER_NAMES) is None


# ── ladder ───────────────────────────────────────────────────────────────

def test_auto_does_not_silently_choose_nvdec():
    """NVDEC is slower here and its frames differ, so it stays opt-in."""
    assert fp._BACKEND_FFMPEG_CUDA not in fp._ladder("auto")
    assert fp._ladder("auto")[0] == fp._BACKEND_FFMPEG_CPU


def test_every_backend_falls_back_rather_than_failing():
    assert fp._ladder("ffmpeg_cuda")[-1] == fp._BACKEND_AV
    assert fp._ladder("ffmpeg_cpu")[-1] == fp._BACKEND_AV


def test_setup_failure_raises_from_the_factory_not_the_first_frame(tmp_path, monkeypatch):
    """A generator body would not run until the first next(), by which point the
    caller has already committed to a backend."""
    monkeypatch.setattr(fp, "ffmpeg_features", lambda *a, **k: None)
    missing = tmp_path / "missing.mp4"
    missing.write_bytes(b"not a video")
    with pytest.raises(Exception):
        fp.make_video_frame_source(missing, decode_size=(128, 96), backend="ffmpeg_cpu")


def test_smoke_test_rejects_a_partial_frame(monkeypatch):
    monkeypatch.setattr(fp.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout=b"\x00" * 100, stderr=b""))
    with pytest.raises(fp.DecodeSetupError, match="multiple"):
        fp.smoke_test(["ffmpeg"], 128, 96)


def test_smoke_test_rejects_an_empty_decode(monkeypatch):
    monkeypatch.setattr(fp.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout=b"", stderr=b""))
    with pytest.raises(fp.DecodeSetupError, match="no frames"):
        fp.smoke_test(["ffmpeg"], 128, 96)


def test_smoke_test_reports_a_nonzero_exit(monkeypatch):
    monkeypatch.setattr(fp.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 1, stdout=b"", stderr=b"Impossible to convert\n"))
    with pytest.raises(fp.DecodeSetupError, match="exit 1"):
        fp.smoke_test(["ffmpeg"], 128, 96)


# ── end to end, software only ────────────────────────────────────────────

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "src.mp4"
    # Tag the colour properties. Real sources carry them, and without them the
    # two backends pick different swscale defaults for the YUV->RGB conversion.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=640x360:rate=30", "-t", "2",
         "-pix_fmt", "yuv420p", "-c:v", "libx264",
         "-color_primaries", "bt709", "-color_trc", "bt709",
         "-colorspace", "bt709", "-color_range", "tv", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_ffmpeg_and_pyav_agree_on_frame_count_and_pixels(clip):
    """Frame count must match exactly: consumers map index to time as index/fps.

    Pixels must also agree closely, because the model's shot predictions were
    validated against the PyAV path. Measured on a real tagged 4K source the
    largest difference is 3/255; the tolerance here is deliberately tight so a
    change in either path's scaling or colour handling fails loudly.
    """
    frames = {}
    for backend in ("av", "ffmpeg_cpu"):
        source = fp.make_video_frame_source(clip, decode_size=(128, 96), backend=backend)
        assert source.backend_name == backend
        frames[backend] = np.stack([frame for _index, frame in source])

    assert frames["av"].shape == frames["ffmpeg_cpu"].shape == (60, 96, 128, 3)
    delta = np.abs(frames["av"].astype(np.int16) - frames["ffmpeg_cpu"].astype(np.int16))
    # The mean bound is the meaningful one: a colour-matrix or range mismatch
    # shifts every pixel and blows it immediately. The max bound is loose because
    # testsrc2 is deliberately harsh - saturated primaries and hard edges at a 5x
    # reduction. On real tagged 4K footage the measured figures are max 3,
    # mean 0.0000.
    assert delta.mean() < 2.0, f"systematic divergence: mean delta {delta.mean():.3f}"
    assert delta.max() <= 32, f"decode paths diverged: max delta {delta.max()}"


@needs_ffmpeg
def test_a_rotated_clip_decodes_the_same_on_both_backends(clip, tmp_path):
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(clip),
         "-c", "copy", str(rotated)],
        check=True,
    )
    frames = {}
    for backend in ("av", "ffmpeg_cpu"):
        source = fp.make_video_frame_source(rotated, decode_size=(128, 96), backend=backend)
        frames[backend] = np.stack([frame for _index, frame in source])
    delta = np.abs(frames["av"].astype(np.int16) - frames["ffmpeg_cpu"].astype(np.int16))
    # Autorotation turns this into a comparison of different pictures (mean
    # delta above 100); decoded as stored, the paths agree as on any clip.
    assert delta.mean() < 2.0, f"backends diverged on a rotated clip: mean delta {delta.mean():.3f}"


def test_probe_flags_a_frame_rate_it_had_to_assume(monkeypatch):
    import types

    import av

    ctx = types.SimpleNamespace(name="h264", codec=None, width=640, height=360,
                                pix_fmt="yuv420p", field_order=None)
    stream = types.SimpleNamespace(
        codec_context=ctx, average_rate=None, guessed_rate=None, duration=None, time_base=None,
        frames=0, color_range=None, colorspace=None,
    )

    class Container:
        streams = types.SimpleNamespace(video=[stream])
        duration = 8 * av.time_base

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(av, "open", lambda _path: Container())
    info = fp.probe_video("/tmp/no-rate.mkv")
    assert info.fps_assumed is True and info.fps == 30.0


@needs_ffmpeg
def test_probe_reads_the_rate_of_a_real_clip(clip):
    info = fp.probe_video(clip)
    assert info.fps == 30.0 and info.fps_assumed is False
    assert info.codec == "h264"


def test_a_decode_that_yields_no_frames_is_an_error():
    """Like a short read: an empty result would pass for a video with no shots."""
    empty = fp.FrameSource("ffmpeg_cpu", UHD, lambda: iter(()), 128, 96)
    with pytest.raises(fp.DecodeStreamError, match="produced no frames"):
        list(empty)

    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    one = fp.FrameSource("ffmpeg_cpu", UHD, lambda: iter([frame]), 128, 96)
    assert [index for index, _frame in one] == [0]


@needs_ffmpeg
def test_abandoning_the_iterator_reaps_ffmpeg(clip):
    features = fp.ffmpeg_features()
    plan = fp.plan_scale(640, 360, (128, 96))
    command = fp.build_command(
        features.path, clip, fp.build_filter_chain(plan, 1, False), False, features.has_fps_mode)
    iterator = fp.iter_rgb_frames(command, 128, 96)
    for _ in zip(range(3), iterator):
        pass
    iterator.close()  # must not leave a zombie or block


# ── VR crop ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("width,height", [
    (5760, 2880), (5244, 2622), (7680, 3840), (1921, 960),   # 180-degree side by side
    (4096, 4096), (2880, 2880), (3841, 3840), (1080, 1080),  # 360-degree
])
def test_vr_crop_rect_selects_exactly_what_vr_permute_keeps(width, height):
    """Cropping in ffmpeg must pick the same source pixels as the Python crop
    it replaces, including for odd dimensions."""
    import torch
    from lib.model.preprocessing_python.image_preprocessing import vr_permute

    rows = np.arange(height, dtype=np.int64)[:, None] * 100_000
    frame = torch.from_numpy(np.broadcast_to(rows + np.arange(width)[None, :], (height, width)).copy())
    kept = vr_permute(frame)
    x, y, crop_width, crop_height = fp.vr_crop_rect(width, height)
    assert (crop_height, crop_width) == tuple(kept.shape)
    assert int(kept[0, 0]) == y * 100_000 + x


def test_crop_happens_before_scaling_and_is_exact():
    plan = fp.plan_scale(5760, 2880, (128, 96), crop=fp.vr_crop_rect(5760, 2880))
    chain = fp.build_filter_chain(plan, 1, False)
    assert chain.startswith("crop=2880:2880:2880:0:exact=1,")
    assert chain.index("crop=") < chain.index("scale=")


def test_scale_plan_is_made_from_the_cropped_size():
    plan = fp.plan_scale(5760, 2880, (2880, 2880), crop=(2880, 0, 2880, 2880))
    assert plan.is_noop and plan.crop == (2880, 0, 2880, 2880)
    assert fp.build_filter_chain(plan, 1, False) == "crop=2880:2880:2880:0:exact=1"


def test_a_cropped_source_is_never_given_to_nvdec():
    plan = fp.plan_scale(5760, 2880, (128, 96), crop=fp.vr_crop_rect(5760, 2880))
    with pytest.raises(ValueError):
        fp.build_filter_chain(plan, 1, True)
    assert fp._ladder("ffmpeg_cuda")[1:] == ["ffmpeg_cpu", "av"]  # so it falls through


@needs_ffmpeg
def test_ffmpeg_crop_matches_cropping_a_decoded_frame(clip):
    """No scaling: the only difference allowed is chroma interpolation at the
    crop edge, since ffmpeg crops before converting 4:2:0 to RGB."""
    import torch
    from lib.model.preprocessing_python.image_preprocessing import vr_permute

    full = fp.make_video_frame_source(clip, decode_size=None, backend="ffmpeg_cpu")
    reference = np.stack([vr_permute(torch.from_numpy(frame.copy())).numpy() for _i, frame in full])
    rect = fp.vr_crop_rect(640, 360)
    cropped = fp.make_video_frame_source(clip, decode_size=(rect[2], rect[3]), backend="ffmpeg_cpu", crop=rect)
    assert (cropped.width, cropped.height) == (rect[2], rect[3])
    frames = np.stack([frame for _i, frame in cropped])
    assert frames.shape == reference.shape
    delta = np.abs(frames.astype(np.int16) - reference.astype(np.int16))
    assert delta[:, :, 2:].max() <= 2, "pixels away from the crop edge must match"


@needs_ffmpeg
@pytest.mark.parametrize("backend", ["ffmpeg_cpu", "av"])
def test_cropped_decode_scales_to_the_target_on_every_backend(clip, backend):
    source = fp.make_video_frame_source(
        clip, decode_size=(128, 96), backend=backend, crop=fp.vr_crop_rect(640, 360))
    assert source.backend_name == backend
    frames = [frame for _i, frame in source]
    assert len(frames) == 60 and frames[0].shape == (96, 128, 3)

