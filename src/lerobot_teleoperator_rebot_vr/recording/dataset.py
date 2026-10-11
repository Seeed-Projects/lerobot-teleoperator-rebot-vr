"""Native LeRobot v3 storage; only the recording writer thread calls this module."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from ..constants import JOINT_NAMES


CAMERA_KEYS = {"main": "observation.images.main", "wrist": "observation.images.wrist"}


def features(config):
    result = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (7,), "names": list(JOINT_NAMES)},
        "observation.host_time": {"dtype": "float64", "shape": (1,), "names": ["seconds_since_session_start"]},
        "action.host_time": {"dtype": "float64", "shape": (1,), "names": ["seconds_since_session_start"]},
        "observation.camera_time": {"dtype": "float64", "shape": (2,), "names": ["main", "wrist"]},
    }
    for name, cfg in config.cameras.items():
        result[CAMERA_KEYS[name]] = {"dtype": "video", "shape": (cfg.height, cfg.width, 3),
                                   "names": ["height", "width", "channels"]}
    return result


def wall_time_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class LocalDataset:
    def __init__(self, config, root: Path, runtime_args, session_ns):
        # Import and codec validation precede any motor connection.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.configs import RGBEncoderConfig
        import av
        av.codec.Codec("libx264", "w")
        if not hasattr(LeRobotDataset, "resume"):
            raise RuntimeError("Recording requires LeRobot with create/resume/finalize support (0.6.2+)")
        self._class = LeRobotDataset
        self._encoder = RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=23, preset="veryfast")
        self.config, self.root = config, root
        self.repo_id = "local/" + root.name.encode("ascii", "ignore").decode().strip("_")
        self.robot_type = runtime_args.robot_model
        self.saved = 0
        self._dataset = None
        self._finalized = False
        self._manifest = {"schema_version": 1, "task": config.task,
            "robot_family": "RS", "robot_type": self.robot_type, "robot_id": runtime_args.robot_id,
            "created_at": wall_time_iso(), "session_monotonic_ns": session_ns,
            "state_units": "degrees", "action_units": "degrees",
            "action_semantics": "absolute joint position targets actually sent, after all clamps",
            "joint_order": list(JOINT_NAMES), "camera_color": "RGB",
            "video_encoding": "H.264, yuv420p, CRF 23, veryfast; bounded streaming queues",
            "camera_timestamps": "host monotonic read/decode completion; no hardware exposure synchronization",
            "time_features": "seconds relative to session_monotonic_ns; LeRobot timestamp is frame_index/fps",
            "recording_config": config.as_dict(), "teleoperation_config": vars(runtime_args),
            "episodes": []}
        self._open_new()
        self._write_manifest()

    def _open_new(self):
        self._dataset = self._class.create(repo_id=self.repo_id, root=self.root,
            fps=self.config.fps, features=features(self.config), robot_type=self.robot_type,
            use_videos=True, rgb_encoder=self._encoder, image_writer_threads=0,
            metadata_buffer_size=1, encoder_threads=1, streaming_encoding=True,
            encoder_queue_maxsize=self.config.queue_frames)
        self._validate_encoder()

    def _validate_encoder(self):
        # LeRobot 0.6.x currently exposes dropped-frame diagnostics only on its
        # writer's encoder. Keep this version-specific check inside the adapter:
        # native streaming otherwise warns and saves videos with missing frames.
        encoder = self._dataset.writer._streaming_encoder
        if encoder is None or not hasattr(encoder, "_dropped_frames"):
            raise RuntimeError("LeRobot streaming encoder lacks dropped-frame diagnostics")

    def _check_dropped_frames(self):
        dropped = self._dataset.writer._streaming_encoder._dropped_frames
        if any(dropped.values()):
            raise RuntimeError(f"video encoder dropped frames; refusing to save misaligned data: {dropped}")

    def begin(self, metadata):
        if self._finalized:
            # Reopen only local metadata after a save. No Hub upload/download requested.
            self._dataset = self._class.resume(repo_id=self.repo_id, root=self.root,
                rgb_encoder=self._encoder, image_writer_threads=0, encoder_threads=1,
                streaming_encoding=True, encoder_queue_maxsize=self.config.queue_frames)
            self._finalized = False
            self._validate_encoder()
        self._current = dict(metadata)
        self._manifest["active_episode"] = self._current
        self._write_manifest()

    def add_frame(self, frame):
        self._dataset.add_frame(frame)
        self._check_dropped_frames()

    def end(self, *, save, metadata):
        entry = {**self._current, **metadata}
        if save:
            self._check_dropped_frames()
            self._dataset.save_episode(parallel_encoding=False)
            # Close Parquet footers before reporting SAVED. Completed episodes
            # remain readable even if a later recording or control run fails.
            self._dataset.finalize()
            self._finalized = True
            entry["episode_index"] = self.saved
            self.saved += 1
            entry["status"] = "saved"
        else:
            self._dataset.clear_episode_buffer()
            entry["status"] = "discarded"
        self._manifest["episodes"].append(entry)
        self._manifest.pop("active_episode", None)
        self._write_manifest()

    def fail(self, message):
        self._manifest["error"] = str(message)
        if "active_episode" in self._manifest:
            self._manifest["active_episode"].update(status="error", ended_at=wall_time_iso())
        self._write_manifest()

    def _write_manifest(self):
        path = self.root / "meta" / "recording.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
        temp.replace(path)

    def close(self):
        if self._dataset is not None and not self._finalized:
            self._dataset.clear_episode_buffer()
            self._dataset.finalize()
            self._finalized = True
