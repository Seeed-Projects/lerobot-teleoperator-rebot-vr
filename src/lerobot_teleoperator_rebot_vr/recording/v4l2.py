"""Explicit Linux camera controls not exposed by OpenCV's capture properties."""
from __future__ import annotations

import fcntl
import os
import struct


VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903


def set_auto_exposure_priority(device, enabled: bool) -> None:
    """Set and verify the UVC permission to lower FPS for automatic exposure.

    This does not disable automatic exposure or choose an exposure duration.
    A requested but unsupported/ignored control fails before recording starts.
    """
    fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    try:
        control = bytearray(struct.pack("Ii", V4L2_CID_EXPOSURE_AUTO_PRIORITY, int(enabled)))
        fcntl.ioctl(fd, VIDIOC_S_CTRL, control)
        readback = bytearray(struct.pack("Ii", V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0))
        fcntl.ioctl(fd, VIDIOC_G_CTRL, readback)
        actual = struct.unpack_from("i", readback, 4)[0]
        if actual != int(enabled):
            raise RuntimeError(f"camera {device}: auto_exposure_priority requested={int(enabled)}, readback={actual}")
    except OSError as exc:
        raise RuntimeError(f"camera {device}: cannot configure auto_exposure_priority: {exc}") from exc
    finally:
        os.close(fd)
