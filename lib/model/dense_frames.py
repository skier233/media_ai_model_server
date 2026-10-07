"""Every-frame delivery from the shared video preprocessor.

Most models score frames sampled at the request's ``frame_interval``. A model
that sets ``requires_every_frame`` needs all of them, at its own (small) clip
geometry. It does not decode the video itself: when a request runs such a
model, the shared preprocessor decodes every frame once, resizes every frame
only for that model, and resizes the interval frames for everyone else.

The decode runs in parallel over contiguous time segments, so frames arrive
in order *within* a segment while segments interleave. A ``DenseFrameStream``
carries them from the preprocessor's thread to the model's:

    for segment, frames in stream:      # frames: [n, H, W, 3] uint8
        ...
    stream.segment_counts               # frames per segment, in video order

Global frame numbers follow from the per-segment counts, which are only known
once every segment before it has finished.
"""

import queue
import threading
import time
from typing import Dict, List, Optional

BACKEND_NAME = "shared_pyav"

_END = object()


class DenseStreamError(RuntimeError):
    """The shared decode could not deliver every frame."""


class DenseFrameStream:
    def __init__(self, width: int, height: int, maxsize: int = 32, attach_timeout: float = 120.0):
        self.width = int(width)
        self.height = int(height)
        self.backend_name = BACKEND_NAME
        self.segment_counts: Optional[List[int]] = None
        self._items: queue.Queue = queue.Queue(maxsize=max(1, maxsize))
        self._abandoned = threading.Event()
        self._attached = threading.Event()
        self._attach_timeout = attach_timeout
        self._created = time.monotonic()

    # ── producer side (the preprocessor's decode thread) ──────────────────

    def put(self, segment: int, frames) -> bool:
        """Hand over consecutive frames of one segment. False once nobody is reading."""
        return self._offer((segment, frames))

    def finish(self, segment_counts: List[int]) -> None:
        self.segment_counts = [int(count) for count in segment_counts]
        self._offer(_END)

    def fail(self, exception: BaseException) -> None:
        self._offer(exception)

    def _offer(self, item) -> bool:
        while not self._abandoned.is_set():
            try:
                self._items.put(item, timeout=0.2)
                return True
            except queue.Full:
                if (not self._attached.is_set()
                        and time.monotonic() - self._created > self._attach_timeout):
                    # The consumer never started. Stop feeding it rather than
                    # stalling the frames every other model is waiting for.
                    self._abandoned.set()
        return False

    # ── consumer side (the model) ─────────────────────────────────────────

    def attach(self) -> None:
        self._attached.set()

    def abandon(self) -> None:
        """Stop reading. The producer's pending and future puts return at once."""
        self._abandoned.set()

    def __iter__(self):
        self._attached.set()
        while True:
            try:
                item = self._items.get(timeout=0.5)
            except queue.Empty:
                if self._abandoned.is_set():
                    raise DenseStreamError("the every-frame stream was abandoned before it finished")
                continue
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise DenseStreamError(f"the shared decode failed: {item}") from item
            yield item


def dense_streams(root_future) -> Dict[str, DenseFrameStream]:
    """The every-frame streams of one request, keyed by consumer model name."""
    streams = getattr(root_future, "_dense_streams", None)
    if streams is None:
        streams = {}
        setattr(root_future, "_dense_streams", streams)
    return streams


def dense_stream_for(root_future, name: str, width: int, height: int) -> DenseFrameStream:
    """Get or create ``name``'s stream. The preprocessor and the model each call
    this for the same request, in either order."""
    streams = dense_streams(root_future)
    stream = streams.get(name)
    if stream is None:
        stream = DenseFrameStream(width, height)
        streams[name] = stream
    return stream
