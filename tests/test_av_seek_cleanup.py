"""The seek worker owns libav until its last read has actually finished."""
import threading
from types import SimpleNamespace

import av
import pytest

from lib.model.preprocessing_python import image_preprocessing as images


class Container:
    def __init__(self, block_second_read=False, fail=False):
        self.streams = SimpleNamespace(video=[SimpleNamespace(duration=3, time_base=1, average_rate=1, disposition=0)])
        self.reading = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.block_second_read = block_second_read
        self.fail = fail
        self.seeks = 0
        self.closing_thread = None
        self.converted = 0
        self.queue_filled = threading.Event()

    def seek(self, *_args, **_kwargs):
        self.seeks += 1
        if self.block_second_read and self.seeks == 2:
            self.reading.set()
            self.release.wait(10)
        if self.fail:
            raise RuntimeError("decode failed")

    def decode(self, stream):
        assert stream is self.streams.video[0]
        assert not self.closed.is_set(), "read after close"
        yield SimpleNamespace(to_ndarray=self.to_array)

    def to_array(self, **_kwargs):
        self.converted += 1
        if self.converted >= 18:
            self.queue_filled.set()
        return object()

    def close(self):
        self.closing_thread = threading.current_thread()
        self.closed.set()


def source(monkeypatch, container):
    monkeypatch.setattr(av, "open", lambda *_args: container)
    monkeypatch.setattr(images, "_validate_local_video_source", lambda value: value)
    monkeypatch.setattr(images, "_prepare_frame", lambda value, *_args: value)
    return images.preprocess_video_av_seek("synthetic.mp4", frame_interval=1, device="cpu")


def test_early_close_does_not_close_a_container_still_being_read(monkeypatch):
    container = Container(block_second_read=True)
    frames = source(monkeypatch, container)
    next(frames)
    assert container.reading.wait(2)
    try:
        frames.close()  # the worker stays blocked past the two-second join
        assert not container.closed.is_set()
    finally:
        container.release.set()
    assert container.closed.wait(2)
    assert container.closing_thread is not threading.current_thread()


def test_normal_completion_closes_from_the_decoder_thread(monkeypatch):
    container = Container()
    assert len(list(source(monkeypatch, container))) == 3
    assert container.closed.is_set()
    assert container.closing_thread is not threading.current_thread()


def test_decode_failure_closes_and_reaches_the_consumer(monkeypatch):
    container = Container(fail=True)
    with pytest.raises(RuntimeError, match="decode failed"):
        list(source(monkeypatch, container))
    assert container.closed.is_set()


def test_early_close_stops_a_producer_blocked_on_a_full_queue(monkeypatch):
    container = Container()
    container.streams.video[0].duration = 10_000
    frames = source(monkeypatch, container)
    next(frames)
    assert container.queue_filled.wait(2)
    frames.close()
    assert container.closed.wait(2)
    assert container.seeks < 30
    assert container.closing_thread is not threading.current_thread()
