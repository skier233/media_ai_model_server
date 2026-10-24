"""Parallel PyAV decode tests. Everything here runs without a GPU."""

import itertools
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


@pytest.fixture(scope="module")
def long_gop_mp4(tmp_path_factory):
    """MP4 with one keyframe in 10 seconds, like the long GOPs of AV1 and
    streaming encodes. Every segment's seek lands on the keyframe at 0, seconds
    before the segment starts."""
    path = tmp_path_factory.mktemp("long_gop") / "long_gop.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x90:rate=30000/1001", "-t", "10",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "300", "-keyint_min", "300",
         "-sc_threshold", "0", "-bf", "2", str(path)],
        check=True,
    )
    return path


def _shared_frames(path, workers):
    """Interval frames by target, segment counts and every frame, from one decode."""
    segments = {}
    counts = {}

    def on_dense(segment, _size_index, frames):
        segments.setdefault(segment, []).append(frames)

    interval = dict(mp_decode.iter_parallel_frames(
        path, 0.5, 0, workers, True, dense_sizes=[DENSE_SIZE], on_dense=on_dense,
        on_segment=counts.__setitem__))
    order = sorted(counts)
    frames = np.concatenate([clip for segment in order for clip in segments.get(segment, [])])
    return interval, [counts[segment] for segment in order], frames


@needs_ffmpeg
def test_a_seek_far_before_the_segment_still_delivers_its_frames(long_gop_mp4):
    interval, counts, frames = _shared_frames(long_gop_mp4, 4)
    reference_interval, _, reference = _shared_frames(long_gop_mp4, 1)

    assert len(counts) == 4 and all(counts)
    assert len(reference) == 300
    np.testing.assert_array_equal(frames, reference)
    assert sorted(interval) == sorted(reference_interval) and len(interval) == 20
    for target, frame in reference_interval.items():
        np.testing.assert_array_equal(interval[target], frame, err_msg=f"target {target}")


def test_frames_decoded_before_the_segment_are_not_kept():
    """Only the first frame, for the seek check, and the frames within the
    tolerance of the segment start, which can still be sampled for it, are kept
    of the frames decoded between the keyframe and the segment start."""
    import gc
    import weakref
    from fractions import Fraction

    alive = weakref.WeakSet()

    class _Decoded:
        def __init__(self, pts):
            self.pts = pts
            alive.add(self)

    class _GopPacket(_Packet):
        def decode(self):
            return [_Decoded(self.pts)]

    class _Container:
        def __init__(self):
            self.seeks = []

        def seek(self, position, stream):
            self.seeks.append(position)

        def demux(self, _stream):
            return (_GopPacket(pts) for pts in range(100))  # one keyframe at 0

    class _Tally:
        def restart(self):
            pass

        def packet(self, _packet):
            pass

        def missed_before(self, _pts):
            return False

    class _TenthStream(_Stream):
        time_base = Fraction(1, 10)

    container = _Container()
    frames = mp_decode._decode_from(container, _TenthStream(), 6.0, True, _Tally(), tolerance=0.15)
    gc.collect()

    assert container.seeks == [60]
    assert sorted(frame.pts for frame in alive) == [0, 59, 60]
    assert [frame.pts for frame in itertools.islice(frames, 4)] == [0, 59, 60, 61]


@pytest.fixture(scope="module")
def damaged_clip(tmp_path_factory):
    """H.264 whose frame at 1.4s has a broken NAL length, which the decoder
    rejects like the damage in real downloads. Also returns the clean source."""
    import av

    folder = tmp_path_factory.mktemp("damaged")
    clean = folder / "clean.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x90:rate=30", "-t", "4",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "30", "-bf", "2", str(clean)],
        check=True,
    )
    with av.open(str(clean)) as container:
        stream = container.streams.video[0]
        packets = [(abs(float(packet.pts * stream.time_base) - 1.4), packet.pos)
                   for packet in container.demux(stream) if packet.size and not packet.is_keyframe]
    damaged = folder / "damaged.mp4"
    data = bytearray(clean.read_bytes())
    position = min(packets)[1]
    data[position:position + 4] = b"\x7f\xff\xff\xff"
    damaged.write_bytes(bytes(data))
    return damaged, clean


def _shared_decode(path, frame_interval):
    """Interval targets and dense failures from the shared decode."""
    failures = []
    targets = sorted(target for target, _frame in mp_decode.iter_parallel_frames(
        path, frame_interval, 0, 4, True, dense_sizes=[DENSE_SIZE], on_dense=lambda *_: None,
        on_dense_failed=lambda segment, message: failures.append(message)))
    return targets, failures


@needs_ffmpeg
def test_a_damaged_frame_fails_only_the_every_frame_stream(damaged_clip):
    """The every-frame consumers number frames by counting them, so a frame the
    decoder rejects fails their stream. The interval frames do not depend on
    every frame and keep coming, so the other analyses still run."""
    damaged, clean = damaged_clip
    targets, failures = _shared_decode(damaged, 0.5)
    clean_targets, clean_failures = _shared_decode(clean, 0.5)

    assert clean_failures == []
    # A segment can start decoding from a keyframe in the segment before it,
    # so more than one segment may decode the damaged frame.
    assert failures and all("the decoder rejected a packet" in failure for failure in failures)
    assert targets == clean_targets


@needs_ffmpeg
def test_a_damaged_frame_does_not_fail_an_interval_only_decode(damaged_clip, caplog):
    """Without every-frame consumers a rejected packet costs only the frames
    that depend on it: every interval target still gets a frame, and the
    skipped packet is logged."""
    damaged, clean = damaged_clip
    targets = sorted(target for target, _frame in mp_decode.iter_parallel_frames(damaged, 0.5, 0, 4, True))
    clean_targets = sorted(target for target, _frame in mp_decode.iter_parallel_frames(clean, 0.5, 0, 4, True))

    assert targets == clean_targets
    assert "damaged packet" in caplog.text


@pytest.fixture(scope="module")
def badly_damaged_clips(damaged_clip, tmp_path_factory):
    """Preserve MP4 metadata; corrupt over 32 packets and over 10% of the
    stream, with a clean keyframe after the damaged stretch for recovery."""
    import av

    _damaged, clean = damaged_clip
    folder = tmp_path_factory.mktemp("burst_damage")
    burst, unreadable = folder / "burst.mp4", folder / "unreadable.mp4"
    burst_data = bytearray(clean.read_bytes())
    unreadable_data = bytearray(burst_data)
    broken = 0
    with av.open(str(clean)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.size <= 4 or packet.pts is None:
                continue
            at = packet.pos
            unreadable_data[at:at + 4] = b"\x7f\xff\xff\xff"
            if 1.0 <= float(packet.pts * stream.time_base) < 2.5:
                burst_data[at:at + 4] = b"\x7f\xff\xff\xff"
                broken += 1
    assert broken > 32
    burst.write_bytes(burst_data)
    unreadable.write_bytes(unreadable_data)
    return burst, unreadable, clean


@needs_ffmpeg
@pytest.mark.parametrize("workers", [1, 4])
def test_many_packet_errors_do_not_discard_recovered_samples(badly_damaged_clips, workers, caplog):
    burst, _unreadable, clean = badly_damaged_clips
    targets = sorted(target for target, _ in mp_decode.iter_parallel_frames(burst, 0.5, 0, workers, True))
    expected = sorted(target for target, _ in mp_decode.iter_parallel_frames(clean, 0.5, 0, workers, True))
    assert targets == expected == [index * 0.5 for index in range(8)]
    assert "damaged packet" in caplog.text and "produced" in caplog.text


@needs_ffmpeg
def test_many_packet_errors_fail_only_the_dense_branch(badly_damaged_clips):
    burst, _unreadable, clean = badly_damaged_clips
    targets, failures = _shared_decode(burst, 0.5)
    expected, _ = _shared_decode(clean, 0.5)
    assert targets == expected
    assert failures and all("the decoder rejected a packet" in message for message in failures)


@needs_ffmpeg
@pytest.mark.parametrize("workers", [1, 4])
def test_a_video_without_any_decodable_samples_still_fails(badly_damaged_clips, workers):
    _burst, unreadable, _clean = badly_damaged_clips
    with pytest.raises(RuntimeError, match="no interval frames could be decoded"):
        list(mp_decode.iter_parallel_frames(unreadable, 0.5, 0, workers, True))


def test_an_empty_damaged_chunk_does_not_discard_samples_from_other_chunks(monkeypatch, caplog):
    import queue

    class Context:
        def Queue(self, maxsize):
            self.queue = queue.Queue()
            return self.queue

        def Process(self, target, args, daemon):
            context = self
            start = args[1]
            class Process:
                def start(self):
                    if start == 0:
                        context.queue.put(("__rejected__", 0, 70, "InvalidDataError", 100, 0))
                    else:
                        context.queue.put((3.0, "decodable frame"))
                    context.queue.put(None)

                def is_alive(self):
                    return False

                def join(self, timeout):
                    pass
            return Process()

    monkeypatch.setattr(mp_decode.mp, "get_context", lambda _: Context())
    monkeypatch.setattr(mp_decode, "probe_duration_seconds", lambda _: 4)
    assert list(mp_decode.iter_parallel_frames("synthetic.mp4", 1, 0, 2, True)) == [(3.0, "decodable frame")]
    assert "produced 0 interval sample(s)" in caplog.text


def test_unrelated_decode_errors_are_not_treated_as_damaged_packets():
    class BrokenPacket(_Packet):
        def decode(self):
            raise OSError("disk read failed")
    class Container:
        def demux(self, _stream):
            return [BrokenPacket(0)]
    with pytest.raises(OSError, match="disk read failed"):
        list(mp_decode._decoded_frames(Container(), _Stream(), None, mp_decode._Rejects(_Stream())))


@needs_ffmpeg
def test_the_dense_stream_fails_and_the_interval_frames_continue(damaged_clip):
    import threading

    from lib.model.dense_frames import DenseFrameStream, DenseStreamError
    from lib.model.preprocessing_python.image_preprocessing import preprocess_video_mp_pyav

    damaged, _clean = damaged_clip
    stream = DenseFrameStream(*DENSE_SIZE)
    raised = []

    def consume():
        try:
            for _ in stream:
                pass
        except DenseStreamError as exc:
            raised.append(exc)
        finally:
            stream.abandon()

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    frames = list(preprocess_video_mp_pyav(
        str(damaged), 0.5, 0, False, "cpu", True, norm_config=-1, decode_workers=4,
        dense_sinks=[(DENSE_SIZE, [stream])]))
    reader.join(timeout=10)

    assert len(frames) == 8
    assert len(raised) == 1 and "cannot decode every frame" in str(raised[0])
    assert stream.segment_counts is None


@pytest.fixture(scope="module")
def vp8_alt_ref_clip(tmp_path_factory):
    """Two-pass VP8 with alt-ref frames: each hidden alt-ref is a packet of its
    own that never becomes a frame, and shares the next frame's timestamp."""
    folder = tmp_path_factory.mktemp("vp8")
    path = folder / "alt_ref.webm"
    for number, output in ((1, ["-f", "null", "-"]), (2, [str(path)])):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc2=size=160x90:rate=30", "-t", "4",
             "-c:v", "libvpx", "-b:v", "200k", "-auto-alt-ref", "1", "-lag-in-frames", "16",
             "-pass", str(number), "-passlogfile", str(folder / "pass"), *output],
            check=True,
        )
    return path


@needs_ffmpeg
def test_packets_that_never_become_frames_do_not_fail_a_segment(vp8_alt_ref_clip):
    counts, frames = _dense_decode(vp8_alt_ref_clip, 0.5, 4)
    _, reference = _dense_decode(vp8_alt_ref_clip, 0.5, 1)

    assert sum(counts) == len(reference) == 120
    np.testing.assert_array_equal(frames, reference)


@pytest.fixture(scope="module")
def b_frame_avi(tmp_path_factory):
    """AVI with B-frames: its timestamps follow decode order, so the frames come
    out of the decoder with timestamps that go backwards."""
    path = tmp_path_factory.mktemp("avi") / "b_frames.avi"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x90:rate=30", "-t", "4",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-bf", "2", "-g", "30", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_frames_out_of_display_order_are_delivered_as_one_segment(b_frame_avi):
    assert mp_decode.probe_frame_order(b_frame_avi)[0] is False

    counts, frames = _dense_decode(b_frame_avi, 0.5, 4)
    _, reference = _dense_decode(b_frame_avi, 0.5, 1)

    assert counts == [120]
    np.testing.assert_array_equal(frames, reference)


@needs_ffmpeg
def test_a_damaged_frame_ends_a_decode_for_the_every_frame_stream_only(damaged_clip):
    """A request for shot boundaries alone samples no interval frames, so a
    failed every-frame stream ends the decode instead of decoding on for
    nothing."""
    damaged, _clean = damaged_clip
    failures = []
    frames = list(mp_decode.iter_parallel_frames(
        damaged, 0.5, 0, 4, True, dense_sizes=[DENSE_SIZE], on_dense=lambda *_: None,
        on_dense_failed=lambda segment, message: failures.append(message), emit_interval=False))

    assert frames == []
    assert len(failures) == 1 and "the decoder rejected a packet" in failures[0]


class _Stream:
    time_base = 1
    start_time = 0


class _Packet:
    def __init__(self, pts, size=1):
        self.pts = pts
        self.size = size
        self.is_discard = False


class _Frame:
    def __init__(self, pts):
        self.pts = pts


def _emitter(segment=1, start=2, end=4, **options):
    out = []

    class _Queue:
        put = out.append

    options = {"dense_sizes": [DENSE_SIZE], "segment": segment, "segment_end": end, **options}
    return mp_decode._DenseEmitter(options, start, _Stream(), _Queue()), out


def _frameless(emitter, pts):
    """Record a frame as added without resizing a real one."""
    emitter._frame_pts.add(pts)
    emitter.count += 1


def test_a_packet_of_the_segment_without_a_frame_fails_it():
    """The decoder can drop a damaged frame's dependants without an error, so a
    segment is checked against its packets, not only for rejections."""
    emitter, out = _emitter()
    for pts in (1, 2, 3, 4):  # 1 and 4 belong to the neighbouring segments
        emitter.packet(_Packet(pts))
    _frameless(emitter, 2)
    emitter.close()

    assert out == [("__dense_failed__", 1, "segment 1: 1 frames between 3.000s and 3.000s did not decode")]


def test_frames_shortly_before_the_first_decodable_one_are_left_out():
    """A cut video's first frames can reference a keyframe that is not in the
    file. No decoder produces them, so a short such gap is not a failure."""
    emitter, out = _emitter(segment=0, start=0, end=None, first_pts=1)
    emitter.packet(_Packet(0.75))
    emitter.packet(_Packet(1))
    _frameless(emitter, 1)
    emitter.close()

    assert out == [("__segment__", 0, 1)]

    emitter, out = _emitter(segment=0, start=0, end=None, first_pts=2)
    emitter.packet(_Packet(1))  # a whole second before it: a damaged or cut file
    emitter.packet(_Packet(2))
    _frameless(emitter, 2)
    emitter.close()

    assert out[0][0] == "__dense_failed__"


def test_an_incomplete_last_frame_is_left_out():
    """A file whose last frame cannot be decoded loses only that frame: no
    other frame moves, so the segment still succeeds. A longer missing tail
    fails it."""
    emitter, out = _emitter(start=2, end=None)
    for pts in (2, 3, 3.25):
        emitter.packet(_Packet(pts))
    _frameless(emitter, 2)
    _frameless(emitter, 3)
    emitter.close()

    assert out == [("__segment__", 1, 2)]

    emitter, out = _emitter(start=2, end=None)
    for pts in (2, 3, 4):
        emitter.packet(_Packet(pts))
    _frameless(emitter, 2)
    emitter.close()

    assert out[0][0] == "__dense_failed__"


def test_a_frame_without_a_timestamp_fails_the_segment():
    emitter, out = _emitter()

    def decode():
        return [_Frame(None)]

    packet = _Packet(3)
    packet.decode = decode

    class _Container:
        def demux(self, _stream):
            return [packet]

    assert list(mp_decode._decoded_frames(_Container(), _Stream(), emitter)) == []
    assert out == [("__dense_failed__", 1,
                    "segment 1: a frame has no timestamp, so it cannot be placed in the video")]


def test_repeated_packet_errors_fail_the_dense_stream_once_without_aborting_sampling():
    emitter, _out = _emitter()
    error = ValueError("rejected")
    for _ in range(100):
        emitter.rejected(_Packet(3), error)
    assert emitter.failed
    assert len(_out) == 1 and _out[0][0] == "__dense_failed__"


@pytest.fixture(scope="module")
def indexed_flv(tmp_path_factory):
    """FLV with a keyframe index whose video starts at 0.04s, like files from
    old Flash video sites. Its demuxer refuses a seek to 0."""
    path = tmp_path_factory.mktemp("flv") / "indexed.flv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x90:rate=25", "-t", "6",
         "-pix_fmt", "yuv420p", "-c:v", "flv1", "-g", "25", "-output_ts_offset", "0.04",
         "-flvflags", "add_keyframe_index", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_a_video_starting_after_zero_seeks_to_its_start(indexed_flv):
    targets = list(mp_decode.iter_parallel_frames(indexed_flv, 0.5, 0, 4, True))
    counts, frames = _dense_decode(indexed_flv, 0.5, 4)
    _, reference = _dense_decode(indexed_flv, 0.5, 1)

    assert len(targets) == 13
    assert sum(counts) == len(reference) == 150
    np.testing.assert_array_equal(frames, reference)


def test_mpeg4_part_2_decodes_on_one_thread_for_every_frame_consumers():
    assert mp_decode._threading("mpeg4", True, 4) == ("SLICE", 1)
    assert mp_decode._threading("mpeg4", True, None) == ("SLICE", 1)
    assert mp_decode._threading("mpeg4", False, 4) == ("AUTO", 4)
    assert mp_decode._threading("h264", True, 4) == ("AUTO", 4)
    assert mp_decode._threading("h264", True, None) == ("AUTO", None)


def test_the_workers_split_the_cpus_between_their_decoders():
    assert mp_decode.threads_per_worker(8, cpus=32) == 4
    assert mp_decode.threads_per_worker(8, cpus=12) == 1
    assert mp_decode.threads_per_worker(8, cpus=4) == 1
    assert mp_decode.threads_per_worker(3, cpus=16) == 5
    assert mp_decode.threads_per_worker(1, cpus=64) == mp_decode.MAX_DECODE_THREADS
    assert mp_decode.threads_per_worker(0, cpus=2) == 2


def _launched_options(monkeypatch, path, workers, **kwargs):
    """The options each worker of a decode is started with. Nothing decodes."""
    import queue

    launched = []

    class _Context:
        def Queue(self, maxsize):
            self.queue = queue.Queue()
            return self.queue

        def Process(self, target, args, daemon):
            launched.append(args[-1])
            context = self

            class _Process:
                def start(self):
                    context.queue.put(None)

                def is_alive(self):
                    return False

                def join(self, timeout):
                    pass

            return _Process()

    monkeypatch.setattr(mp_decode.mp, "get_context", lambda _method: _Context())
    monkeypatch.setattr(mp_decode, "_available_cpus", lambda: 16)
    list(mp_decode.iter_parallel_frames(path, 0.5, 0, workers, True, **kwargs))
    return launched


@needs_ffmpeg
def test_each_worker_is_given_its_share_of_the_cpus(monkeypatch, open_gop_clip, b_frame_avi):
    options = _launched_options(monkeypatch, open_gop_clip, 4)
    assert [option["decode_threads"] for option in options] == [4] * 4

    options = _launched_options(monkeypatch, open_gop_clip, 4, decode_threads_per_worker=3)
    assert [option["decode_threads"] for option in options] == [3] * 4

    # Only the workers that are started share the CPUs: here one worker
    # delivers every frame and the others would only have sampled.
    options = _launched_options(monkeypatch, b_frame_avi, 4, dense_sizes=[DENSE_SIZE],
                                on_dense=lambda *_: None, emit_interval=False)
    assert [option["decode_threads"] for option in options] == [8]


def test_a_refused_seek_back_to_zero_falls_back_to_the_stream_start():
    class _Container:
        def __init__(self):
            self.seeks = []

        def seek(self, position, stream):
            self.seeks.append(position)
            if position == 0:
                raise PermissionError(1, "Operation not permitted")

    class _Started(_Stream):
        start_time = 40

    container = _Container()
    mp_decode._seek(container, _Started(), 0, fresh=True)
    assert container.seeks == []
    mp_decode._seek(container, _Started(), 0, fresh=False)
    assert container.seeks == [0, 40]
