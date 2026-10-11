"""Recording configuration, validation and local dataset naming."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import math
import re

import yaml


@dataclass(frozen=True)
class CameraConfig:
    device: int | str
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"
    buffer_size: int = 1
    # V4L2: False keeps automatic exposure from varying the frame rate.
    # None leaves the camera's existing policy untouched.
    auto_exposure_priority: bool | None = None

    def __post_init__(self):
        if isinstance(self.device, bool) or not isinstance(self.device, (int, str)):
            raise ValueError("camera device must be an index or /dev path")
        if isinstance(self.device, int) and self.device < 0:
            raise ValueError("camera index must be non-negative")
        if isinstance(self.device, str) and not self.device.startswith("/dev/"):
            raise ValueError("USB camera paths must start with /dev/")
        for key in ("width", "height", "fps"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"camera {key} must be a positive integer")
        if self.width % 2 or self.height % 2:
            raise ValueError("video width and height must be even")
        if len(self.fourcc) != 4:
            raise ValueError("camera fourcc must contain four characters")
        if type(self.buffer_size) is not int or not 1 <= self.buffer_size <= 16:
            raise ValueError("camera buffer_size must be an integer in 1..16")
        if self.auto_exposure_priority is not None and type(self.auto_exposure_priority) is not bool:
            raise ValueError("camera auto_exposure_priority must be true, false or null")

    @property
    def device_path(self):
        return Path(f"/dev/video{self.device}" if isinstance(self.device, int) else self.device).resolve()


@dataclass(frozen=True)
class RecordingConfig:
    output_dir: Path
    cameras: dict[str, CameraConfig]
    task: str = ""
    fps: int = 25
    queue_frames: int = 16
    max_camera_age_ms: float = 100.
    max_camera_skew_ms: float = 50.
    max_episode_s: float = 300.
    menu_field: str = "menuButton"
    teleoperation: dict = field(default_factory=dict)

    def __post_init__(self):
        if set(self.cameras) != {"main", "wrist"}:
            raise ValueError("configure exactly two USB cameras: main and wrist")
        if len({c.device_path for c in self.cameras.values()}) != 2:
            raise ValueError("main and wrist must use different camera devices")
        if type(self.fps) is not int or not 1 <= self.fps <= 60:
            raise ValueError("recording fps must be an integer in 1..60")
        if any(c.fps < self.fps for c in self.cameras.values()):
            raise ValueError("camera fps must be at least recording fps")
        if type(self.queue_frames) is not int or not 2 <= self.queue_frames <= 120:
            raise ValueError("queue_frames must be in 2..120")
        for key in ("max_camera_age_ms", "max_camera_skew_ms", "max_episode_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not isinstance(self.menu_field, str) or not self.menu_field.strip():
            raise ValueError("menu_field must be the left controller's menu field name")
        if not isinstance(self.task, str):
            raise ValueError("task must be a string")
        allowed = {"control_config", "robot_port", "robot_id", "hand"}
        if not isinstance(self.teleoperation, dict) or set(self.teleoperation) - allowed:
            raise ValueError(f"teleoperation supports only {sorted(allowed)}")
        if self.teleoperation.get("hand", "right") not in {"left", "right"}:
            raise ValueError("teleoperation.hand must be left or right")

    def as_dict(self):
        return asdict(self)


def load_config(path: Path, *, task: str | None = None) -> RecordingConfig:
    path = path.expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("recording config must be a mapping")
    data = dict(data)
    data["cameras"] = {name: CameraConfig(**value) for name, value in data.get("cameras", {}).items()}
    output = Path(data.get("output_dir", "../recordings")).expanduser()
    data["output_dir"] = (path.parent / output).resolve()
    teleop = dict(data.get("teleoperation", {}))
    profile = Path(teleop.get("control_config", "rs_mit_split.yaml")).expanduser()
    teleop["control_config"] = str((path.parent / profile).resolve())
    data["teleoperation"] = teleop
    if task is not None:
        data["task"] = task.strip()
    return RecordingConfig(**data)


def dataset_name(task: str, robot_type: str, now: datetime) -> str:
    if not task.strip():
        raise ValueError("provide a task description with --task")
    slug = re.sub(r"[^\w\-]+", "_", task.strip(), flags=re.UNICODE).strip("_")[:64] or "task"
    return f"{robot_type}_{slug}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
