"""Left-menu rising edges from raw Tracking payloads, separate from motion buttons."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class MenuEvent:
    received_ns: int
    epoch: int


class LeftMenuButton:
    def __init__(self, field="menuButton", timeout_s=.2, debounce_s=.25):
        self.field = field
        self.timeout_ns = int(timeout_s * 1e9)
        self.debounce_ns = int(debounce_s * 1e9)
        self._lock = threading.Lock()
        self._events = deque(maxlen=8)
        self._epoch = None
        self._last_received_ns = 0
        self._last_tracking_ns = 0
        self._last_edge_ns = -self.debounce_ns
        self._released = False
        self.available = False
        self.left_fields = ()
        self.overflow = False

    def feed(self, sample, tracking):
        now = sample.received_monotonic_ns
        controllers = tracking.get("Controller", {})
        left = controllers.get("left", {}) if isinstance(controllers, Mapping) else {}
        value = left.get(self.field) if isinstance(left, Mapping) else None
        # An absent/invalid field must not masquerade as a released button.
        if type(value) is bool:
            pressed = value
        elif type(value) in (int, float) and value in (0, 1):
            pressed = bool(value)
        elif isinstance(value, str) and value.lower() in {"true", "false", "0", "1"}:
            pressed = value.lower() in {"true", "1"}
        else:
            pressed = None
        with self._lock:
            self.left_fields = tuple(left) if isinstance(left, Mapping) else ()
            self.available = pressed is not None
            restarted = (sample.stream_epoch != self._epoch or now - self._last_received_ns > self.timeout_ns)
            if restarted or pressed is None:
                self._released = False
                self._events.clear()
                self._last_tracking_ns = 0
            self._epoch = sample.stream_epoch
            self._last_received_ns = now
            timestamp = sample.tracking_timestamp_ns
            if timestamp and timestamp <= self._last_tracking_ns:
                return
            self._last_tracking_ns = timestamp
            if pressed is False:
                self._released = True
            elif pressed is True:
                if self._released and now - self._last_edge_ns >= self.debounce_ns:
                    if len(self._events) == self._events.maxlen:
                        self.overflow = True
                    else:
                        self._events.append(MenuEvent(now, sample.stream_epoch))
                    self._last_edge_ns = now
                self._released = False

    def drain(self, now_ns, epoch=None):
        with self._lock:
            events = list(self._events)
            self._events.clear()
            # Callback events can arrive just after the control cycle's timestamp.
            # Keep these for the next cycle rather than silently losing a click.
            self._events.extend(event for event in events if event.received_ns > now_ns
                                and (epoch is None or event.epoch == epoch))
        return [event for event in events
                if 0 <= now_ns - event.received_ns <= self.timeout_ns
                and (epoch is None or event.epoch == epoch)]
