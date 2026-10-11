"""Optional observers may inspect control cycles but cannot abort robot cleanup."""
from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class RuntimeObserver(Protocol):
    def on_tracking(self, sample, tracking) -> None: ...
    def on_ready(self) -> None: ...
    def on_step(self, observation, sent_action, status, sample) -> None: ...
    def on_stop(self) -> None: ...


class GuardedObserver:
    """Disable a failed observer while preserving the runtime's shutdown policy.

    Hooks must be short, non-blocking operations. In particular, on_stop only
    signals the observer; its owner joins IO workers after the runtime exits.
    """

    def __init__(self, observer: RuntimeObserver):
        self.observer = observer
        self._failed = threading.Event()

    def _call(self, method, *args):
        if self._failed.is_set():
            return
        try:
            getattr(self.observer, method)(*args)
        except Exception:
            self._failed.set()
            logger.exception("Runtime observer failed in %s; disabling observer, arm control continues", method)
            try:
                self.observer.on_stop()
            except Exception:
                logger.exception("Failed to stop runtime observer")

    def on_tracking(self, sample, tracking):
        self._call("on_tracking", sample, tracking)

    def on_ready(self):
        self._call("on_ready")

    def on_step(self, observation, sent_action, status, sample):
        self._call("on_step", observation, sent_action, status, sample)

    def on_stop(self):
        self._call("on_stop")
