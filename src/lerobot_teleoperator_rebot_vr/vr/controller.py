"""PICO 4 pose source for the XRoboToolkit V1 TCP protocol."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from ..config_rebot_vr import RebotVRConfig
from .models import VRFrame
from .tracking import ControllerSample
from .xr_v1 import V1TrackingSource


logger = logging.getLogger(__name__)


class VRController(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def is_tracking(self) -> bool: ...

    def connect(self) -> None: ...

    def get_action(self) -> dict[str, Any]: ...

    def latest_sample(self) -> ControllerSample | None: ...

    def disconnect(self) -> None: ...


def _safe_vr_action(frame: VRFrame | None = None) -> dict[str, Any]:
    if frame is None:
        return {
            "grip_pos": np.zeros(3, dtype=float),
            "grip_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            "squeeze": 0.0,
            "trigger": 0.0,
            "is_tracking": False,
            "received_monotonic_ns": time.monotonic_ns(),
            "tracking_timestamp_ns": 0,
            "stream_epoch": 0,
            "side": "right",
            "primary_button": False,
            "secondary_button": False,
            "status": None,
            "head_pos": None,
            "head_quat": None,
        }
    return {
        "grip_pos": frame.grip_pos.copy(),
        "grip_quat": frame.grip_quat.copy(),
        "squeeze": frame.squeeze,
        "trigger": frame.trigger,
        "is_tracking": frame.is_tracking,
        "received_monotonic_ns": frame.received_monotonic_ns,
        "tracking_timestamp_ns": frame.tracking_timestamp_ns,
        "stream_epoch": frame.stream_epoch,
        "side": frame.side,
        "primary_button": frame.primary_button,
        "secondary_button": frame.secondary_button,
        "status": frame.status,
        "head_pos": None if frame.head_pos is None else frame.head_pos.copy(),
        "head_quat": None if frame.head_quat is None else frame.head_quat.copy(),
    }


def _transform_pose(
    position: np.ndarray, quaternion: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotation = transform[:3, :3]
    transformed_position = rotation @ position + transform[:3, 3]
    source_rotation = Rotation.from_quat(quaternion).as_matrix()
    transformed_rotation = rotation @ source_rotation @ rotation.T
    return transformed_position, Rotation.from_matrix(transformed_rotation).as_quat()


class XRoboToolkitV1Controller:
    """VRController adapter over the validated latest-only V1 source."""

    def __init__(self, config: RebotVRConfig, *, on_tracking=None) -> None:
        self.config = config
        self._on_tracking = on_tracking
        self._source = V1TrackingSource(
            host=config.ws_host,
            port=config.ws_port,
            side=config.hand_side,
            on_status=logger.info,
            on_sample=self._on_sample,
        )
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._latest: VRFrame | None = None

    @property
    def is_connected(self) -> bool:
        return self._running.is_set() or self._source.running

    @property
    def is_tracking(self) -> bool:
        sample = self.latest_sample()
        return bool(sample is not None and self._is_fresh(sample))

    def connect(self) -> None:
        if self.is_connected:
            raise RuntimeError("XRoboToolkitV1Controller is already connected")
        self._source.start()
        self._running.set()

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("XRoboToolkitV1Controller is not connected")
        sample = self._source.latest_sample()
        if sample is None:
            return _safe_vr_action()
        with self._lock:
            frame = self._latest
        if frame is None or not self._is_fresh(frame):
            return _safe_vr_action()
        return _safe_vr_action(frame)

    def disconnect(self) -> None:
        self._running.clear()
        self._source.stop()
        with self._lock:
            self._latest = None

    def feed_bytes(self, data: bytes, *, received_monotonic_ns: int | None = None) -> None:
        self._source.feed_bytes(data, received_monotonic_ns=received_monotonic_ns)

    def latest_sample(self) -> ControllerSample | None:
        return self._source.latest_sample()

    def stats(self):
        return self._source.stats()

    def _on_sample(
        self, sample: ControllerSample, _tracking: Mapping[str, object]
    ) -> None:
        frame = self._tracking_to_frame(sample)
        with self._lock:
            self._latest = frame
        if self._on_tracking is not None:
            self._on_tracking(sample, _tracking)

    def _tracking_to_frame(self, sample: ControllerSample) -> VRFrame:
        transform = np.asarray(self.config.base_T_anchor, dtype=float)
        position, quaternion = _transform_pose(
            sample.position, sample.quaternion_xyzw, transform
        )
        return VRFrame(
            grip_pos=position,
            grip_quat=quaternion,
            squeeze=sample.grip,
            trigger=sample.trigger,
            received_monotonic_ns=sample.received_monotonic_ns,
            tracking_timestamp_ns=sample.tracking_timestamp_ns,
            stream_epoch=sample.stream_epoch,
            side=sample.side,
            primary_button=sample.primary_button,
            secondary_button=sample.secondary_button,
            status=sample.status,
        )

    def _is_fresh(self, frame: VRFrame | ControllerSample) -> bool:
        age_ns = max(0, time.monotonic_ns() - frame.received_monotonic_ns)
        return age_ns <= int(self.config.stale_timeout * 1e9)


def make_vr_controller(config: RebotVRConfig, *, on_tracking=None) -> VRController:
    return XRoboToolkitV1Controller(config, on_tracking=on_tracking)


__all__ = [
    "VRController",
    "XRoboToolkitV1Controller",
    "make_vr_controller",
]
