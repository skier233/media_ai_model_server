"""Cover art is a video stream, but must never be analyzed as the movie."""

from types import SimpleNamespace
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
import pytest

from lib.model.preprocessing_python import ffmpeg_pipe, mp_decode
from lib.model.preprocessing_python.video_stream import select_video_stream


@pytest.mark.parametrize("image_flag", [av.stream.Disposition.attached_pic,
                                       av.stream.Disposition.timed_thumbnails])
def test_skips_images_before_the_video(image_flag):
    image = SimpleNamespace(disposition=image_flag)
    first = SimpleNamespace(disposition=0)
    second = SimpleNamespace(disposition=av.stream.Disposition.default)
    container = SimpleNamespace(streams=SimpleNamespace(video=[image, first, second]))
    assert select_video_stream(container) is first


@pytest.mark.parametrize("dispositions", [[], [av.stream.Disposition.attached_pic]])
def test_an_image_alone_is_not_a_movie(dispositions):
    container = SimpleNamespace(streams=SimpleNamespace(
        video=[SimpleNamespace(disposition=flag) for flag in dispositions]))
    with pytest.raises(ValueError, match="No video stream other than"):
        select_video_stream(container)


@pytest.fixture(scope="module")
def covered_clip(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    folder = tmp_path_factory.mktemp("cover")
    movie, cover, path = folder / "movie.mp4", folder / "cover.jpg", folder / "covered.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x90:rate=25", "-t", "4", "-c:v", "libx264",
         "-g", "25", "-bf", "0", str(movie)], check=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "color=c=red:size=48x32", "-frames:v", "1", str(cover)], check=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(movie), "-i", str(cover),
         "-map", "0:v", "-map", "1:v", "-c", "copy", "-disposition:v:1", "attached_pic",
         str(path)], check=True)
    with av.open(str(path)) as container:
        assert len(container.streams.video) == 2
        assert any(s.disposition & av.stream.Disposition.attached_pic for s in container.streams.video)
    return path


@pytest.fixture
def cover_first(monkeypatch, covered_clip):
    """Expose real cover/video streams in ASF's cover-first order.

    MP4 stores cover art last. Reorder only the stream list, leaving actual
    PyAV streams, packets, decoding and seeking intact for these regressions.
    """
    original_open = av.open

    class Container:
        def __init__(self, *args, **kwargs):
            self.inner = original_open(*args, **kwargs)
            self.streams = SimpleNamespace(video=sorted(
                self.inner.streams.video,
                key=lambda s: not bool(s.disposition & av.stream.Disposition.attached_pic)))

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.inner.close()

    monkeypatch.setattr(av, "open", Container)
    return str(covered_clip)


@pytest.mark.parametrize("dense", [False, True])
def test_chunk_decodes_the_movie_with_a_cover_first(cover_first, dense):
    messages = []
    options = {"decode_threads": 1, "skip_nonref": True}
    if dense:
        options["dense_sizes"] = [(32, 18)]
    mp_decode._decode_chunk_worker(
        cover_first, 0, 8, 0.5, 64, True,
        SimpleNamespace(put=messages.append), options)
    samples = [item for item in messages if item is not None and not isinstance(item[0], str)]
    assert [target for target, _frame in samples] == [i * 0.5 for i in range(8)]
    assert all(frame.shape == (36, 64, 3) for _target, frame in samples)
    assert not any(item and item[0] in ("__error__", "__dense_failed__") for item in messages)
    if dense:
        assert ("__segment__", 0, 100) in messages


def test_probes_read_movie_metadata_with_a_cover_first(cover_first):
    from lib.model.preprocessing_python import image_preprocessing as images

    assert mp_decode.probe_duration_seconds(cover_first) == 4
    assert mp_decode.probe_frame_order(cover_first) == (True, 0)
    assert images.get_video_duration_av(cover_first) == 4
    assert images.probe_video_dimensions(cover_first) == (160, 90)
    assert images.probe_keyframe_interval_seconds(cover_first) == 1
    info = ffmpeg_pipe.probe_video(cover_first)
    assert (info.codec, info.width, info.height, info.fps) == ("h264", 160, 90, 25)


@pytest.mark.parametrize("backend", ["av", "av_seek"])
def test_other_pyav_backends_decode_movie_with_a_cover_first(cover_first, backend):
    from lib.model.preprocessing_python import image_preprocessing as images

    preprocess = getattr(images, f"preprocess_video_{backend}")
    frames = list(preprocess(cover_first, frame_interval=1, img_size=32,
                             device="cpu", use_half_precision=False, use_timestamps=True))
    assert [target for target, _frame in frames] == [0, 1, 2, 3]


def test_ffmpeg_and_its_pyav_fallback_decode_the_movie(cover_first):
    for backend in ("av", "ffmpeg_cpu"):
        decoded = []
        for path in (cover_first, Path(cover_first).with_name("movie.mp4")):
            source = ffmpeg_pipe.make_video_frame_source(
                path, decode_size=(32, 18), backend=backend)
            decoded.append(np.stack([frame for _index, frame in source]))
        assert decoded[0].shape == (100, 18, 32, 3)
        np.testing.assert_array_equal(*decoded)


def test_ffmpeg_mapping_excludes_cover_art():
    command = ffmpeg_pipe.build_command("ffmpeg", "covered.wmv", "", False, True)
    assert command[command.index("-map") + 1] == "0:V:0"
