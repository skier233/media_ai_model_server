"""The decode-ahead thread between ffmpeg and the shot-boundary model."""

import threading
import time

import pytest

from lib.model.ai_shot_boundary_model import Prefetch


class Source:
    """An iterator that records whether it was closed and from which thread."""

    def __init__(self, count, fail_at=None, delay=0.0):
        self.count, self.fail_at, self.delay = count, fail_at, delay
        self.produced = 0
        self.closed = threading.Event()

    def __iter__(self):
        return self._generate()

    def _generate(self):
        try:
            for index in range(self.count):
                if index == self.fail_at:
                    raise RuntimeError("decode failed")
                if self.delay:
                    time.sleep(self.delay)
                self.produced += 1
                yield index
        finally:
            self.closed.set()


def test_items_arrive_in_order_and_the_source_is_closed():
    source = Source(500)
    with Prefetch(source, maxsize=8) as items:
        assert list(items) == list(range(500))
    assert source.closed.wait(2)


def test_a_source_error_reaches_the_consumer():
    source = Source(100, fail_at=40)
    with Prefetch(source, maxsize=8) as items:
        received = []
        with pytest.raises(RuntimeError, match="decode failed"):
            for item in items:
                received.append(item)
    assert received == list(range(40))
    assert source.closed.wait(2)


def test_it_reads_ahead_but_no_further_than_maxsize():
    source = Source(1000)
    items = Prefetch(source, maxsize=10)
    time.sleep(0.3)
    # The queue holds maxsize, plus one item waiting to be offered.
    assert 10 <= source.produced <= 11
    items.close()
    assert source.closed.wait(2)


@pytest.mark.parametrize("consumed", [0, 5])
def test_closing_early_stops_the_thread_and_closes_the_source(consumed):
    """This is what reaps ffmpeg when analysis fails part-way, or before it starts."""
    source = Source(10_000, delay=0.001)
    items = Prefetch(source, maxsize=4)
    for _ in range(consumed):
        next(items)
    items.close()
    assert source.closed.wait(2)
    assert source.produced < 10_000
    with pytest.raises(StopIteration):
        next(items)
