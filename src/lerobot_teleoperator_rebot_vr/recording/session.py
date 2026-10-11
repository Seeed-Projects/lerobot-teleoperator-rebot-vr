"""Episode state machine and bounded asynchronous persistence.

The arm thread only validates small arrays, peeks camera buffers and enqueues.
Camera/disk failures reject the current episode, never alter motor commands.
"""
from __future__ import annotations

from datetime import datetime
import logging
from queue import Queue
import threading
import time

import numpy as np

from ..constants import JOINT_NAMES
from .buttons import LeftMenuButton
from .cameras import CameraRig
from .config import dataset_name
from .dataset import CAMERA_KEYS, LocalDataset, wall_time_iso
from .sampling import CycleSampler

logger = logging.getLogger(__name__)


class RecordingSession:
    def __init__(self, config, runtime_args, *, rig=None, writer_factory=LocalDataset):
        self.config, self.runtime_args = config, runtime_args
        self.session_ns = time.monotonic_ns()
        self.root = config.output_dir / dataset_name(config.task, "RS", datetime.now().astimezone())
        self.buttons = LeftMenuButton(config.menu_field, runtime_args.stale_timeout)
        self.rig = rig or CameraRig(config)
        self._writer_factory = writer_factory
        self._writer = None
        self._thread = None
        self._state_lock = threading.RLock()
        self._queue = Queue(maxsize=config.queue_frames + 3)  # Reserve room for END/CLOSE.
        self.state = "IDLE"
        self.error = None
        self._count = 0
        self._first_ns = None
        self._sampler = CycleSampler(runtime_args.fps, config.fps)
        self._last_camera_ns = {}
        self._last_epoch = None
        self._warned_menu = False
        self._ready = False

    def open(self):
        try:
            self.rig.open()
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self._writer = self._writer_factory(self.config, self.root, self.runtime_args, self.session_ns)
            # Imports/codec preflight may take time; verify both streams are
            # still producing fresh, aligned frames before enabling motors.
            self.rig.snapshot(time.monotonic_ns())
            self._thread = threading.Thread(target=self._write_loop, name="lerobot-record-writer", daemon=True)
            self._thread.start()
        except BaseException:
            if self._writer is not None and (self._thread is None or not self._thread.is_alive()):
                try:
                    self._writer.close()
                except Exception:
                    logger.exception("Dataset cleanup failed during recorder startup")
            try:
                self.rig.close()
            except Exception:
                logger.exception("Camera cleanup failed during recorder startup")
            raise
        print(f"[record] Local dataset: {self.root}\n"
              "[record] Left Menu: start / stop and save. Motion buttons unchanged.\n"
              "[record] Ctrl+C discards an unfinished episode; already saved episodes are retained.", flush=True)

    def on_tracking(self, sample, tracking):
        self.buttons.feed(sample, tracking)

    def on_ready(self):
        # Ignore menu clicks made during automatic startup. Require a release.
        self.buttons = LeftMenuButton(self.config.menu_field, self.runtime_args.stale_timeout)
        self._ready = True

    def on_step(self, observation, sent_action, status, sample):
        """Called after the arm command; does not block on camera or disk IO."""
        with self._state_lock:
            self._on_step(observation, sent_action, status, sample)

    def _on_step(self, observation, sent_action, status, sample):
        if not self._ready:
            return
        now_ns = status.command_send_finished_monotonic_ns
        epoch = None if sample is None else sample.stream_epoch
        if sample is not None and not self.buttons.available and not self._warned_menu:
            print(f"[record] Waiting for Controller.left.{self.config.menu_field}; "
                  f"received left fields: {self.buttons.left_fields}. Use --check-input to verify sender support.", flush=True)
            self._warned_menu = True
        events = self.buttons.drain(now_ns, epoch)
        if self.buttons.overflow:
            self._reject("left-menu event queue overflow")
            self.buttons.overflow = False
            events = []
        valid = (status.feedback_valid and status.tracking and sent_action is not None
                 and not status.feedback_abort_requested)
        # A/B return motions are valid commanded actions. They do not change
        # recording state; Left Menu still controls episode start and stop.
        # Stop applies to the already validated prefix. A fault in this same
        # cycle must not erase that prefix before the user's stop is handled.
        # No observation/action from the stop cycle is added to the episode.
        for _event in events:
            if self.state == "IDLE":
                if not valid:
                    reason, _ = self._control_rejection(status, sent_action, sample, now_ns, check_epoch=False)
                    print(f"[record] Start rejected: {reason}", flush=True)
                    continue
                self._start(epoch)
            elif self.state in {"STARTING", "RECORDING"}:
                self._finish(save=self._count > 0, reason="left_menu")
            else:
                print(f"[record] {self.state}: wait before starting another episode.", flush=True)
        # An epoch change cannot split a continuing demonstration across reconnects.
        if self.state in {"STARTING", "RECORDING"} and (
                not valid or (self._last_epoch is not None and epoch != self._last_epoch)):
            reason, diagnostic = self._control_rejection(status, sent_action, sample, now_ns)
            self._finish(save=False, reason=reason, diagnostic=diagnostic)
        if self.state != "RECORDING":
            return
        if self._first_ns is not None:
            if now_ns - self._first_ns >= self.config.max_episode_s * 1e9:
                self._finish(save=True, reason="max_episode_duration")
                return
        try:
            if not self._sampler.due(now_ns):
                return
            state = np.array([observation[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
            action = np.array([sent_action[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
            if not (np.isfinite(state).all() and np.isfinite(action).all()):
                raise ValueError("non-finite state or action")
            camera_frames = self.rig.snapshot(status.feedback_finished_monotonic_ns)
            for name, camera in camera_frames.items():
                if camera.captured_ns <= self._last_camera_ns.get(name, -1):
                    raise RuntimeError(f"{name} camera frame repeated or moved backwards; episode rejected")
            frame = {"observation.state": state, "action": action, "task": self.config.task,
                "observation.host_time": np.array([(status.feedback_finished_monotonic_ns-self.session_ns)*1e-9], dtype=np.float64),
                "action.host_time": np.array([(now_ns-self.session_ns)*1e-9], dtype=np.float64),
                "observation.camera_time": np.array([
                    (camera_frames[name].captured_ns-self.session_ns)*1e-9 for name in ("main", "wrist")], dtype=np.float64)}
            for name, key in CAMERA_KEYS.items():
                frame[key] = camera_frames[name].image
            if self._queue.qsize() >= self.config.queue_frames:
                raise RuntimeError("disk writer queue full; episode rejected instead of dropping frames")
            self._queue.put_nowait(("frame", frame))
            if self._first_ns is None:
                self._first_ns = now_ns
            self._last_camera_ns = {name: camera.captured_ns for name, camera in camera_frames.items()}
            self._count += 1
        except Exception as exc:
            self._reject(str(exc))

    def _start(self, epoch):
        self._count = 0
        self._first_ns = None
        self._sampler.reset()
        self._last_camera_ns = {}
        self._last_epoch = epoch
        self.state = "STARTING"
        self._queue.put_nowait(("begin", {"task": self.config.task, "robot_family": "RS",
            "started_at": wall_time_iso(), "start_monotonic_ns": time.monotonic_ns()}))
        print("[record] STARTING", flush=True)

    def _control_rejection(self, status, sent_action, sample, now_ns, *, check_epoch=True):
        """Preserve the exact failing gates; a combined message cannot diagnose a lost episode."""
        reasons = []
        epoch = None if sample is None else sample.stream_epoch
        received_ns = None if sample is None else getattr(sample, "received_monotonic_ns", None)
        age_ms = None if received_ns is None else (now_ns - received_ns) * 1e-6
        if not status.feedback_valid:
            detail = getattr(status, "feedback_fault_reason", "") or "unspecified"
            reasons.append(f"robot feedback invalid ({detail})")
        if not status.tracking:
            reasons.append("VR tracking unavailable" + (" (no sample)" if age_ms is None else
                           f" (sample_age_ms={age_ms:.1f}, stale_timeout_ms={self.runtime_args.stale_timeout*1000:.1f})"))
        if sent_action is None:
            reasons.append("no robot action sent")
        if status.feedback_abort_requested:
            reasons.append("feedback abort requested")
        if check_epoch and self._last_epoch is not None and epoch != self._last_epoch:
            reasons.append(f"VR stream changed: {self._last_epoch} -> {epoch}")
        state = getattr(status, "state", None)
        diagnostic = {
            "control_state": getattr(state, "value", state),
            "feedback_valid": status.feedback_valid,
            "feedback_fault_reason": getattr(status, "feedback_fault_reason", ""),
            "feedback_fault_count": getattr(status, "feedback_fault_count", None),
            "feedback_abort_requested": status.feedback_abort_requested,
            "tracking": status.tracking, "sample_age_ms": age_ms,
            "stale_timeout_ms": self.runtime_args.stale_timeout * 1000,
            "sample_status": None if sample is None else getattr(sample, "status", None),
            "episode_stream_epoch": self._last_epoch, "current_stream_epoch": epoch,
            "return_target": status.return_target, "action_sent": sent_action is not None,
            "primary_button": getattr(status, "primary_button", None),
            "secondary_button": getattr(status, "secondary_button", None),
        }
        return "; ".join(reasons), diagnostic

    def _finish(self, *, save, reason, diagnostic=None):
        if self.state not in {"STARTING", "RECORDING"}:
            return
        self.state = "SAVING" if save else "DISCARDING"
        self._queue.put_nowait(("end", {"save": save, "metadata": {
            "ended_at": wall_time_iso(), "frames": self._count, "reason": reason,
            "first_frame_monotonic_ns": self._first_ns,
            **({"rejection_diagnostic": diagnostic} if diagnostic is not None else {})}}))
        print(f"[record] {self.state}: {self._count} frames ({reason})", flush=True)

    def _reject(self, reason):
        if self.state in {"STARTING", "RECORDING"}:
            self._finish(save=False, reason=reason)

    def on_stop(self):
        with self._state_lock:
            self._ready = False
            self._reject("runtime stopped before left-menu save")

    def _write_loop(self):
        try:
            while True:
                kind, payload = self._queue.get()
                try:
                    if kind == "close":
                        return
                    if kind == "begin":
                        self._writer.begin(payload)
                        with self._state_lock:
                            if self.state == "STARTING":
                                self.state = "RECORDING"
                                print("[record] RECORDING", flush=True)
                    elif kind == "frame":
                        self._writer.add_frame(payload)
                    elif kind == "end":
                        self._writer.end(**payload)
                        print(f"[record] {'SAVED' if payload['save'] else 'DISCARDED'}; "
                              f"saved episodes={self._writer.saved}", flush=True)
                        with self._state_lock:
                            self.state = "IDLE"
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            with self._state_lock:
                self.error = str(exc)
                self.state = "ERROR"
            logger.exception("Recording disabled after writer failure; arm control continues")
            try:
                self._writer.fail(self.error)
            except Exception:
                logger.exception("Could not persist recording error")
        finally:
            try:
                self._writer.close()
            except Exception as exc:
                with self._state_lock:
                    self.error = self.error or str(exc)
                    self.state = "ERROR"
                logger.exception("Could not finalize recording dataset")

    def close(self):
        self.on_stop()
        try:
            if self._thread is not None and self._thread.is_alive():
                self._queue.put(("close", None), timeout=5.)
                while self._thread.is_alive():
                    self._thread.join(timeout=10.)
                    if self._thread.is_alive():
                        print("[record] Waiting for dataset finalization...", flush=True)
        finally:
            self.rig.close()
        if self.error:
            raise RuntimeError(f"recording writer failed: {self.error}; inspect {self.root}")
