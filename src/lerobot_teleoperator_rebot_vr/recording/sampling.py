"""Downsample complete control cycles without using variable send latency as a clock."""
from __future__ import annotations

import math


class CycleSampler:
    def __init__(self, control_fps: float, recording_fps: int):
        ratio = control_fps / recording_fps
        if not math.isfinite(ratio) or ratio < 1 or abs(ratio - round(ratio)) > 1e-8:
            raise ValueError("control fps must be an integer multiple of recording fps")
        self.every = round(ratio)
        self.period_ns = round(1e9 / recording_fps)
        self.reset()

    def reset(self):
        self._remaining = 0
        self._last_ns = None
        self._last_cycle_ns = None

    def due(self, now_ns: int) -> bool:
        if self._last_cycle_ns is not None and now_ns <= self._last_cycle_ns:
            raise RuntimeError("control timestamp did not advance; episode rejected")
        self._last_cycle_ns = now_ns
        # Check even the skipped cycles so a stopped/overloaded control loop
        # cannot silently compress a gap into the nominal video frame rate.
        if self._last_ns is not None and now_ns - self._last_ns >= 2 * self.period_ns:
            raise RuntimeError("recording deadline missed; not compressing a gap into constant-fps data")
        if self._remaining:
            self._remaining -= 1
            return False
        self._remaining = self.every - 1
        self._last_ns = now_ns
        return True
