"""Parallel PyAV decode tests. Everything here runs without a GPU."""

import shutil
import subprocess

import numpy as np
import pytest

from lib.model.preprocessing_python import mp_decode

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

DENSE_SIZE = (32, 18)


@pytest.fixture(scope="module")
def open_gop_clip(tmp_path_factory):
    """Open-GOP HEVC at 29.97 fps with a keyframe every 60 frames.

    Keyframes are displayed at 2.002s and 4.004s but decoded before 2.0s and
    4.0s, and the frames displayed just before each keyframe reference the
    previous GOP. A seek to 4.0s therefore lands on the 4.004s keyframe, which
    is how real HEVC camera and encoder output looks.
    """
    path = tmp_path_factory.mktemp("clip") / "open_gop.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x90:rate=30000/1001", "-t", "6",
         "-pix_fmt", "yuv420p", "-c:v", "libx265",
         "-x265-params", "log-level=error:keyint=60:min-keyint=60:scenecut=0:open-gop=1:bframes=4",
         str(path)],
        check=True,
    )
    return path


def _dense_decode(path, frame_interval, workers):
    segments = {}
    counts = {}

    def on_dense(segment, _size_index, frames):
        segments.setdefault(segment, []).append(frames)

    for _ in mp_decode.iter_parallel_frames(
            path, frame_interval, 0, workers, True, dense_sizes=[DENSE_SIZE],
            on_dense=on_dense, on_segment=counts.__setitem__, emit_interval=False):
        pass
    order = sorted(counts)
    frames = np.concatenate([clip for segment in order for clip in segments.get(segment, [])])
    return [counts[segment] for segment in order], frames


@needs_ffmpeg
def test_segments_starting_just_before_a_keyframe_keep_every_frame(open_gop_clip):
    """A 0.5s interval over four workers puts segment starts at 2.0s and 4.0s,
    just before keyframes displayed at 2.002s and 4.004s. Seeking there skips
    the frames displayed between the segment start and the keyframe, so each
    worker must start from an earlier keyframe instead."""
    counts, frames = _dense_decode(open_gop_clip, 0.5, 4)
    _, reference = _dense_decode(open_gop_clip, 0.5, 1)

    assert counts == [60, 60, 60, 0]
    assert len(reference) == 180
    np.testing.assert_array_equal(frames, reference)


@pytest.fixture(scope="module")
def single_keyframe_ts(tmp_path_factory):
    """MPEG-TS with one keyframe in 10 seconds. A TS seek is not
    keyframe-accurate, so a segment starting late in the file lands after the
    only keyframe and decodes nothing at all."""
    path = tmp_path_factory.mktemp("ts") / "single_keyframe.ts"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x90:rate=25", "-t", "10",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "250", "-keyint_min", "250",
         "-sc_threshold", "0", "-f", "mpegts", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_a_seek_that_decodes_nothing_starts_further_back(single_keyframe_ts):
    counts, frames = _dense_decode(single_keyframe_ts, 0.5, 5)
    _, reference = _dense_decode(single_keyframe_ts, 0.5, 1)

    assert len(counts) == 5 and all(counts)
    assert len(reference) == 250
    np.testing.assert_array_equal(frames, reference)
