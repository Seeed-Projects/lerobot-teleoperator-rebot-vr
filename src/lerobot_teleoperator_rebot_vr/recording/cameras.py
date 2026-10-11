"""Background USB capture and bounded, timestamped RGB snapshots."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time

import numpy as np

from .config import CameraConfig


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    captured_ns: int  # Host monotonic time after read/decode, not exposure time.


class USBCamera:
    def __init__(self, config: CameraConfig, *, capture_factory=None):
        self.config = config
        self._capture_factory = capture_factory
        self._capture = None
        self._frames = deque(maxlen=8)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._error = None

    def open(self):
        import cv2
        factory = self._capture_factory or cv2.VideoCapture
        cfg = self.config
        self._capture = factory(str(cfg.device_path), cv2.CAP_V4L2)
        try:
            if not self._capture.isOpened():
                raise RuntimeError(f"cannot open USB camera {cfg.device_path}")
            self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            self._capture.set(cv2.CAP_PROP_FPS, cfg.fps)
            if not self._capture.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size):
                raise RuntimeError(f"camera {cfg.device_path}: could not set buffer_size={cfg.buffer_size}")
            if cfg.auto_exposure_priority is not None:
                from .v4l2 import set_auto_exposure_priority
                set_auto_exposure_priority(cfg.device_path, cfg.auto_exposure_priority)
            self._thread = threading.Thread(target=self._run, name=f"record-{cfg.device_path.name}", daemon=True)
            self._thread.start()
            if not self._ready.wait(5.):
                raise TimeoutError(f"no frames from {cfg.device_path} within 5 s")
            if self._error:
                raise RuntimeError(self._error)
        except BaseException:
            self.close()
            raise

    def _run(self):
        import cv2
        try:
            while not self._stop.is_set():
                ok, bgr = self._capture.read()
                if not ok:
                    raise RuntimeError(f"USB camera read failed: {self.config.device_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                captured_ns = time.monotonic_ns()
                expected = (self.config.height, self.config.width, 3)
                if rgb.shape != expected or rgb.dtype != np.uint8:
                    raise RuntimeError(f"camera returned {rgb.shape}/{rgb.dtype}; expected {expected}/uint8")
                rgb.flags.writeable = False
                with self._lock:
                    self._frames.append(CameraFrame(rgb, captured_ns))
                self._ready.set()
        except Exception as exc:
            self._error = str(exc)
            self._ready.set()
        finally:
            self._capture.release()

    def latest_before(self, reference_ns):
        if self._error:
            raise RuntimeError(self._error)
        with self._lock:
            for frame in reversed(self._frames):
                if frame.captured_ns <= reference_ns:
                    return frame
        raise RuntimeError("no camera frame preceding robot observation")

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.)
            if self._thread.is_alive():
                # Never release VideoCapture concurrently with a blocked native read.
                raise RuntimeError(f"camera read did not stop: {self.config.device_path}")
        elif self._capture is not None:
            self._capture.release()
        self._capture = None


class CameraRig:
    def __init__(self, config, *, camera_factory=USBCamera):
        self.config = config
        self.cameras = {name: camera_factory(cfg) for name, cfg in config.cameras.items()}
        self._opened = []

    def open(self):
        try:
            for camera in self.cameras.values():
                camera.open()
                self._opened.append(camera)
        except BaseException:
            self.close()
            raise

    def snapshot(self, reference_ns):
        frames = {name: camera.latest_before(reference_ns) for name, camera in self.cameras.items()}
        times = [f.captured_ns for f in frames.values()]
        if any(not 0 <= reference_ns - t <= self.config.max_camera_age_ms * 1e6 for t in times):
            raise RuntimeError("camera frame stale; episode rejected")
        if max(times) - min(times) > self.config.max_camera_skew_ms * 1e6:
            raise RuntimeError("main/wrist camera time skew exceeded; episode rejected")
        return frames

    def close(self):
        errors = []
        for camera in reversed(self._opened):
            try:
                camera.close()
            except Exception as exc:
                errors.append(str(exc))
        self._opened.clear()
        if errors:
            raise RuntimeError("; ".join(errors))
