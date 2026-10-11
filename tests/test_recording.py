"""Recording contracts: synthetic VR, cameras and writers, never robot hardware."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot_teleoperator_rebot_vr.constants import JOINT_NAMES
from lerobot_teleoperator_rebot_vr.recording.buttons import LeftMenuButton
from lerobot_teleoperator_rebot_vr.recording.cameras import CameraFrame, CameraRig, USBCamera
from lerobot_teleoperator_rebot_vr.recording.config import CameraConfig, RecordingConfig, dataset_name, load_config
from lerobot_teleoperator_rebot_vr.recording.session import RecordingSession
from lerobot_teleoperator_rebot_vr.runtime.observer import GuardedObserver


def cfg(tmp_path, **kwargs):
    return RecordingConfig(tmp_path, {"main": CameraConfig(0, 64, 48), "wrist": CameraConfig(2, 64, 48)},
                           task="拿起红色积木", **kwargs)


def sample(ms, epoch=0, stamp=None):
    return SimpleNamespace(received_monotonic_ns=int(ms*1e6), stream_epoch=epoch,
                           tracking_timestamp_ns=int(ms*1e6) if stamp is None else stamp)


def feed(button, ms, value, **kwargs):
    button.feed(sample(ms, **kwargs), {"Controller": {"left": {"menuButton": value},
                                                   "right": {"menuButton": True}}})


def test_left_menu_requires_release_uses_rising_edges_and_preserves_other_buttons():
    button = LeftMenuButton()
    feed(button, 100, True)
    assert not button.drain(100_000_000)
    feed(button, 120, False)
    feed(button, 140, True)
    feed(button, 160, True)
    assert len(button.drain(160_000_000, 0)) == 1
    feed(button, 180, False)
    feed(button, 200, True)  # debounce
    assert not button.drain(200_000_000)
    for ms in range(220, 420, 20):
        feed(button, ms, False)
    feed(button, 420, True)
    assert len(button.drain(420_000_000)) == 1
    button.feed(sample(440), {"Controller": {"left": {"primaryButton": True, "secondaryButton": True}}})
    assert not button.available
    assert not button.drain(440_000_000)


@pytest.mark.parametrize("value", [None, "pressed", 2, {}, []])
def test_invalid_menu_value_is_not_a_release(value):
    button = LeftMenuButton()
    feed(button, 100, value)
    feed(button, 120, True)
    assert not button.drain(120_000_000)


@pytest.mark.parametrize("case", ["gap", "epoch", "duplicate", "stale", "wrong_epoch"])
def test_menu_drops_stale_reconnected_or_duplicate_edges(case):
    button = LeftMenuButton()
    feed(button, 100, False)
    if case == "gap":
        feed(button, 400, True)
        assert not button.drain(400_000_000)
    elif case == "epoch":
        feed(button, 120, True, epoch=1)
        assert not button.drain(120_000_000)
    elif case == "duplicate":
        feed(button, 120, True, stamp=100_000_000)
        assert not button.drain(120_000_000)
    else:
        feed(button, 120, True)
        assert not button.drain(400_000_000 if case == "stale" else 120_000_000,
                                epoch=1 if case == "wrong_epoch" else 0)


def test_menu_overflow_is_explicit():
    button = LeftMenuButton(debounce_s=0)
    for ms in range(100, 460, 40):
        feed(button, ms, False)
        feed(button, ms+20, True)
    assert button.overflow
    assert len(button.drain(460_000_000)) <= 8


def test_real_tracking_adapter_delivers_menu_without_repurposing_abxy():
    import json
    from lerobot_teleoperator_rebot_vr.config_rebot_vr import RebotVRConfig
    from lerobot_teleoperator_rebot_vr.vr.controller import make_vr_controller
    from lerobot_teleoperator_rebot_vr.vr.xr_v1 import PacketParser, CMD_FUNCTION
    button = LeftMenuButton()
    controller = make_vr_controller(RebotVRConfig(hand_side="right"), on_tracking=button.feed)
    for ms, pressed in [(100, False), (120, True)]:
        tracking = {"timeStampNs": ms, "Controller": {
            "left": {"menuButton": pressed, "primaryButton": False, "secondaryButton": True},
            "right": {"pose": "0,0,0,0,0,0,1", "grip": .8, "trigger": .7,
                      "primaryButton": True, "secondaryButton": True, "status": 1}}}
        controller._source.feed_bytes(PacketParser.pack(CMD_FUNCTION, json.dumps({
            "functionName": "Tracking", "value": tracking})), received_monotonic_ns=ms*1_000_000)
    assert len(button.drain(120_000_000)) == 1
    frame = controller.latest_sample()
    assert frame.primary_button and frame.secondary_button
    assert frame.grip == .8 and frame.trigger == .7


def test_config_paths_identity_and_validation(tmp_path):
    config = load_config(Path('config/record_rs.yaml'), task="摆放积木")
    assert config.output_dir.name == "recordings"
    assert config.task == "摆放积木"
    assert config.teleoperation["robot_id"] == "rebot_b601_rs_left"
    assert Path(config.teleoperation["control_config"]).is_absolute()
    assert dataset_name("摆放 / 积木", "RS", datetime(2026, 10, 10, 12, 0)) == "RS_摆放_积木_20261010_120000_000000"
    with pytest.raises(ValueError, match="different"):
        RecordingConfig(tmp_path, {"main": CameraConfig(0), "wrist": CameraConfig('/dev/video0')})
    with pytest.raises(ValueError, match="even"):
        CameraConfig(0, width=63)
    with pytest.raises(ValueError, match="integer"):
        cfg(tmp_path, fps=25.0)


class Capture:
    def __init__(self, *args, bad_shape=False):
        self.released = False
        self.bad_shape = bad_shape
    def isOpened(self): return True
    def set(self, *args): return True
    def read(self):
        time.sleep(.002)
        return True, np.full((46 if self.bad_shape else 48, 64, 3), [10, 20, 30], dtype=np.uint8)
    def release(self): self.released = True


def test_usb_capture_rgb_timestamp_and_cleanup():
    capture = Capture()
    camera = USBCamera(CameraConfig(0, 64, 48), capture_factory=lambda *args: capture)
    try:
        camera.open()
        frame = camera.latest_before(time.monotonic_ns())
        assert frame.image[0, 0].tolist() == [30, 20, 10]
        assert not frame.image.flags.writeable
    finally:
        camera.close()
    assert capture.released


def test_usb_capture_wrong_resolution_fails_before_runtime():
    capture = Capture(bad_shape=True)
    camera = USBCamera(CameraConfig(0, 64, 48), capture_factory=lambda *args: capture)
    with pytest.raises(RuntimeError, match="expected"):
        camera.open()
    assert capture.released


@pytest.mark.parametrize("times,reason", [([800, 990], "stale"), ([920, 990], "skew"), ([1001, 990], "stale")])
def test_camera_rig_rejects_stale_future_and_skew(tmp_path, times, reason):
    rig = CameraRig(cfg(tmp_path))
    rig.cameras = {name: SimpleNamespace(latest_before=lambda now, t=t: CameraFrame(None, t*1_000_000))
                   for name, t in zip(("main", "wrist"), times)}
    with pytest.raises(RuntimeError, match=reason):
        rig.snapshot(1_000_000_000)


class Rig:
    def __init__(self): self.closed = False
    def open(self): pass
    def snapshot(self, ns):
        return {name: CameraFrame(np.zeros((48, 64, 3), np.uint8), ns-1_000_000)
                for name in ("main", "wrist")}
    def close(self): self.closed = True


class Writer:
    def __init__(self, *args):
        self.frames, self.ends = [], []
        self.saved = 0
        self.closed = False
    def begin(self, metadata): self.frames = []
    def add_frame(self, frame): self.frames.append(frame)
    def end(self, *, save, metadata):
        self.ends.append((save, metadata, list(self.frames)))
        self.saved += int(save)
    def fail(self, error): self.error = error
    def close(self): self.closed = True


def wait_for(predicate):
    until = time.monotonic()+3
    while not predicate():
        if time.monotonic() > until:
            raise AssertionError('worker did not complete')
        time.sleep(.001)


@pytest.fixture
def session(tmp_path):
    recorder = RecordingSession(cfg(tmp_path), SimpleNamespace(stale_timeout=.2, fps=50), rig=Rig(), writer_factory=Writer)
    recorder.open()
    recorder.on_ready()
    yield recorder
    recorder.close()


def step(recorder, ms, *, valid=True, epoch=0, return_target=""):
    observation = {f"{name}.pos": i for i, name in enumerate(JOINT_NAMES)}
    sent = {key: val+.125 for key, val in observation.items()}
    status = SimpleNamespace(command_send_finished_monotonic_ns=int(ms*1e6),
        feedback_finished_monotonic_ns=int(ms*1e6)-1_000_000, feedback_valid=valid,
        tracking=valid, return_target=return_target, feedback_abort_requested=False)
    recorder.on_step(observation, sent, status, sample(ms, epoch))


def start(recorder):
    feed(recorder.buttons, 1000, False)
    feed(recorder.buttons, 1020, True)
    step(recorder, 1020)
    wait_for(lambda: recorder.state == "RECORDING")
    step(recorder, 1040)


def test_episode_records_observation_and_actual_sent_action_at_25hz(session):
    start(session)
    for ms in range(1060, 1300, 20):
        feed(session.buttons, ms, False)
        step(session, ms)
    feed(session.buttons, 1300, True)
    step(session, 1300)
    wait_for(lambda: session.state == "IDLE")
    saved, metadata, frames = session._writer.ends[-1]
    assert saved and metadata["reason"] == "left_menu"
    assert len(frames) == 7
    assert metadata["frames"] == len(frames)
    np.testing.assert_array_equal(frames[0]["observation.state"], np.arange(7))
    np.testing.assert_array_equal(frames[0]["action"], np.arange(7)+.125)
    assert frames[0]["task"] == "拿起红色积木"
    times = [f["action.host_time"][0] for f in frames]
    assert np.diff(times) == pytest.approx([.04]*6)
    assert all(f["action.host_time"][0] > f["observation.host_time"][0] for f in frames)


@pytest.mark.parametrize("cause", ["tracking", "epoch", "camera", "gap", "exit"])
def test_invalid_or_unfinished_episode_is_discarded(session, cause):
    start(session)
    if cause == "camera":
        step(session, 1060)
        session.rig.snapshot = Mock(side_effect=RuntimeError("camera disconnected"))
    if cause == "exit":
        session.on_stop()
        step(session, 1080)  # return-to-zero after on_stop cannot add frames
    else:
        step(session, 1200 if cause == "gap" else 1080, valid=cause != "tracking",
             epoch=int(cause == "epoch"))
    wait_for(lambda: session.state == "IDLE")
    assert not session._writer.ends[-1][0]
    assert session._writer.saved == 0
    assert session._writer.ends[-1][1]["frames"] >= 1


@pytest.mark.parametrize('return_target', ['zero', 'home'])
@pytest.mark.parametrize('start_during_return', [False, True])
def test_return_keeps_recording_until_menu_save(session, return_target, start_during_return):
    feed(session.buttons, 1000, False)
    feed(session.buttons, 1020, True)
    step(session, 1020, return_target=return_target if start_during_return else '')
    wait_for(lambda: session.state == 'RECORDING')
    step(session, 1040)
    # The return starts before the stop click and remains latched after arrival.
    for ms in range(1060, 1500, 20):
        feed(session.buttons, ms, False)
        step(session, ms, return_target=return_target)
        assert session.state == 'RECORDING'
    assert not session._writer.ends
    feed(session.buttons, 1500, True)
    step(session, 1500, return_target=return_target)
    wait_for(lambda: session.state == 'IDLE')
    saved, metadata, frames = session._writer.ends[-1]
    assert saved and metadata['reason'] == 'left_menu'
    assert metadata['frames'] == len(frames) == 12
    assert np.diff([f['action.host_time'][0] for f in frames]) == pytest.approx([.04]*11)
    np.testing.assert_array_equal(frames[-1]['action'], np.arange(7)+.125)
    assert set(frames[-1]) >= {'observation.images.main', 'observation.images.wrist'}


def test_start_is_rejected_without_tracking_and_startup_click_is_ignored(session):
    feed(session.buttons, 1000, False)
    feed(session.buttons, 1020, True)
    step(session, 1020, valid=False)
    assert session.state == "IDLE"
    assert not session._writer.ends
    session.on_ready()
    feed(session.buttons, 1300, True)
    step(session, 1300)
    assert session.state == "IDLE"


def test_writer_overflow_discards_whole_episode_without_blocking_arm(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowWriter(Writer):
        def add_frame(self, frame):
            entered.set()
            assert release.wait(3)
            super().add_frame(frame)
    recorder = RecordingSession(cfg(tmp_path, queue_frames=2), SimpleNamespace(stale_timeout=.2, fps=50),
                                rig=Rig(), writer_factory=SlowWriter)
    recorder.open()
    recorder.on_ready()
    try:
        start(recorder)
        assert entered.wait(1)
        for ms in range(1060, 1220, 20):
            step(recorder, ms)
        assert recorder.state == "DISCARDING"  # no blocking on the deliberately stuck writer
        release.set()
        wait_for(lambda: recorder.state == "IDLE")
        assert not recorder._writer.ends[-1][0]
        assert 'queue full' in recorder._writer.ends[-1][1]['reason']
    finally:
        release.set()
        recorder.close()


def test_writer_error_disables_recording_and_preserves_failure(tmp_path):
    class FailingWriter(Writer):
        def add_frame(self, frame): raise OSError("disk full")
    recorder = RecordingSession(cfg(tmp_path), SimpleNamespace(stale_timeout=.2, fps=50),
                                rig=Rig(), writer_factory=FailingWriter)
    recorder.open()
    recorder.on_ready()
    start(recorder)
    wait_for(lambda: recorder.state == "ERROR")
    step(recorder, 1080)
    with pytest.raises(RuntimeError, match="disk full"):
        recorder.close()
    assert recorder.rig.closed
    assert recorder._writer.error == "disk full"


@pytest.mark.parametrize("hook", ["on_tracking", "on_ready", "on_step"])
def test_observer_exceptions_cannot_abort_control_or_cleanup(hook):
    observer = Mock()
    getattr(observer, hook).side_effect = RuntimeError("injected observer failure")
    guarded = GuardedObserver(observer)
    args = {"on_tracking": (None, {}), "on_ready": (), "on_step": ({}, {}, None, None)}
    getattr(guarded, hook)(*args[hook])
    guarded.on_step({}, {}, None, None)
    guarded.on_stop()
    observer.on_stop.assert_called_once()
    getattr(observer, hook).assert_called_once()


def test_menu_event_after_cycle_timestamp_is_retained():
    button = LeftMenuButton()
    feed(button, 100, False)
    feed(button, 120, True)
    assert not button.drain(119_000_000)
    assert len(button.drain(140_000_000)) == 1


def test_duration_limit_saves_and_held_button_cannot_start_again(tmp_path):
    recorder = RecordingSession(cfg(tmp_path, max_episode_s=.08), SimpleNamespace(stale_timeout=.2, fps=50),
                                rig=Rig(), writer_factory=Writer)
    recorder.open()
    recorder.on_ready()
    try:
        start(recorder)
        step(recorder, 1080)
        step(recorder, 1120)
        wait_for(lambda: recorder.state == 'IDLE')
        assert recorder._writer.ends[-1][0]
        assert recorder._writer.ends[-1][1]['reason'] == 'max_episode_duration'
        feed(recorder.buttons, 1140, True)
        step(recorder, 1140)
        assert recorder.state == 'IDLE'
    finally:
        recorder.close()


def test_quick_stop_during_writer_start_discards_empty_episode(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowBegin(Writer):
        def begin(self, metadata):
            entered.set()
            assert release.wait(3)
            super().begin(metadata)
    recorder = RecordingSession(cfg(tmp_path), SimpleNamespace(stale_timeout=.2, fps=50),
                                rig=Rig(), writer_factory=SlowBegin)
    recorder.open()
    recorder.on_ready()
    try:
        feed(recorder.buttons, 1000, False)
        feed(recorder.buttons, 1020, True)
        step(recorder, 1020)
        assert entered.wait(1)
        for ms in range(1040, 1320, 20):
            feed(recorder.buttons, ms, False)
            step(recorder, ms)
        feed(recorder.buttons, 1320, True)
        step(recorder, 1320)
        assert recorder.state == 'DISCARDING'
        release.set()
        wait_for(lambda: recorder.state == 'IDLE')
        assert recorder._writer.ends[-1][1]['frames'] == 0
        assert not recorder._writer.ends[-1][0]
    finally:
        release.set()
        recorder.close()


def test_cli_dry_run_and_input_check_do_not_construct_recorder(monkeypatch, capsys):
    from lerobot_teleoperator_rebot_vr.recording import cli
    monkeypatch.setattr('lerobot_teleoperator_rebot_vr.recording.session.RecordingSession',
                        Mock(side_effect=AssertionError('must not open hardware')))
    cli.main(['--dry-run', '--task', '搬积木'])
    assert '搬积木' in capsys.readouterr().out
    check = Mock()
    monkeypatch.setattr(cli, 'check_input', check)
    cli.main(['--check-input'])
    check.assert_called_once()
    with pytest.raises(SystemExit):
        cli.main([])  # no task, before calibration/camera/CAN access


@pytest.mark.parametrize('enabled', [False, True])
def test_camera_exposure_priority_control_writes_reads_and_closes(monkeypatch, enabled):
    import struct
    from lerobot_teleoperator_rebot_vr.recording import v4l2
    calls = []
    monkeypatch.setattr(v4l2.os, 'open', lambda *args: 123)
    closed = Mock()
    monkeypatch.setattr(v4l2.os, 'close', closed)
    def ioctl(fd, request, data):
        assert fd == 123
        assert struct.unpack_from('I', data)[0] == v4l2.V4L2_CID_EXPOSURE_AUTO_PRIORITY
        calls.append(request)
        if request == v4l2.VIDIOC_S_CTRL:
            assert struct.unpack_from('i', data, 4)[0] == int(enabled)
        else:
            struct.pack_into('i', data, 4, int(enabled))
    monkeypatch.setattr(v4l2.fcntl, 'ioctl', ioctl)
    v4l2.set_auto_exposure_priority('/dev/video6', enabled)
    assert calls == [v4l2.VIDIOC_S_CTRL, v4l2.VIDIOC_G_CTRL]
    closed.assert_called_once_with(123)


@pytest.mark.parametrize('failure', ['unsupported', 'ignored'])
def test_camera_exposure_priority_failure_is_not_silent(monkeypatch, failure):
    import struct
    from lerobot_teleoperator_rebot_vr.recording import v4l2
    monkeypatch.setattr(v4l2.os, 'open', lambda *args: 123)
    closed = Mock()
    monkeypatch.setattr(v4l2.os, 'close', closed)
    def ioctl(fd, request, data):
        if failure == 'unsupported':
            raise OSError('unsupported control')
        if request == v4l2.VIDIOC_G_CTRL:
            struct.pack_into('i', data, 4, 1)
    monkeypatch.setattr(v4l2.fcntl, 'ioctl', ioctl)
    with pytest.raises(RuntimeError, match='auto_exposure_priority'):
        v4l2.set_auto_exposure_priority('/dev/video6', False)
    closed.assert_called_once_with(123)


def test_camera_applies_explicit_exposure_policy_before_capture_and_cleans_up_on_failure(monkeypatch):
    from lerobot_teleoperator_rebot_vr.recording import v4l2
    capture = Capture()
    apply = Mock(side_effect=RuntimeError('unsupported control'))
    monkeypatch.setattr(v4l2, 'set_auto_exposure_priority', apply)
    camera = USBCamera(CameraConfig(6, 64, 48, auto_exposure_priority=False),
                       capture_factory=lambda *args: capture)
    with pytest.raises(RuntimeError, match='unsupported control'):
        camera.open()
    apply.assert_called_once_with(Path('/dev/video6'), False)
    assert camera._thread is None and capture.released
    with pytest.raises(ValueError, match='auto_exposure_priority'):
        CameraConfig(6, auto_exposure_priority='false')


def test_camera_uses_configured_driver_buffers(monkeypatch):
    import cv2
    capture = Capture()
    capture.set = Mock(return_value=True)
    camera = USBCamera(CameraConfig(6, 64, 48, buffer_size=4), capture_factory=lambda *args: capture)
    try:
        camera.open()
        capture.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 4)
    finally:
        camera.close()
    assert capture.released


def test_camera_rejects_unsupported_driver_buffer_request():
    import cv2
    capture = Capture()
    capture.set = lambda prop, value: prop != cv2.CAP_PROP_BUFFERSIZE
    camera = USBCamera(CameraConfig(6, 64, 48, buffer_size=4), capture_factory=lambda *args: capture)
    with pytest.raises(RuntimeError, match='buffer_size=4'):
        camera.open()
    assert camera._thread is None and capture.released


@pytest.mark.parametrize('value', [0, 17, True, '4'])
def test_camera_rejects_invalid_driver_buffer_count(value):
    with pytest.raises(ValueError, match='buffer_size'):
        CameraConfig(6, buffer_size=value)


@pytest.mark.parametrize('name', ['main', 'wrist'])
def test_repeated_camera_frame_cannot_be_saved_as_a_new_observation(session, name):
    start(session)
    old_ns = session._last_camera_ns[name]
    snapshot = session.rig.snapshot
    def repeat(ns):
        frames = snapshot(ns)
        frames[name] = CameraFrame(frames[name].image, old_ns)
        return frames
    session.rig.snapshot = repeat
    step(session, 1060)
    step(session, 1080)
    wait_for(lambda: session.state == 'IDLE')
    saved, metadata, frames = session._writer.ends[-1]
    assert not saved and len(frames) == 1
    assert name + ' camera frame repeated' in metadata['reason']


@pytest.mark.parametrize('fault,expected', [
    ('tracking', 'VR tracking unavailable'),
    ('feedback', 'robot feedback invalid (elbow feedback timeout)'),
    ('epoch', 'VR stream changed: 0 -> 1'),
    ('action', 'no robot action sent'),
    ('abort', 'feedback abort requested'),
])
def test_episode_rejection_records_the_specific_control_gate(session, fault, expected):
    start(session)
    status = SimpleNamespace(command_send_finished_monotonic_ns=1_300_000_000,
        feedback_finished_monotonic_ns=1_299_000_000,
        feedback_valid=fault != 'feedback', tracking=fault != 'tracking',
        return_target='zero',  # A real fault must still reject during a return.
        feedback_abort_requested=fault == 'abort',
        feedback_fault_reason='elbow feedback timeout' if fault == 'feedback' else '',
        feedback_fault_count=1 if fault == 'feedback' else 0,
        state=SimpleNamespace(value='stale' if fault == 'tracking' else 'idle'),
        primary_button=False, secondary_button=True)
    frame = sample(1000 if fault == 'tracking' else 1300, epoch=int(fault == 'epoch'))
    action = None if fault == 'action' else {f'{name}.pos': 0 for name in JOINT_NAMES}
    session.on_step({}, action, status, frame)
    wait_for(lambda: session.state == 'IDLE')
    saved, metadata, frames = session._writer.ends[-1]
    assert not saved
    assert metadata['reason'] == expected or metadata['reason'].startswith(expected + ' (')
    diag = metadata['rejection_diagnostic']
    assert diag['feedback_valid'] == (fault != 'feedback')
    assert diag['tracking'] == (fault != 'tracking')
    assert diag['current_stream_epoch'] == int(fault == 'epoch')
    if fault == 'tracking':
        assert diag['sample_age_ms'] == pytest.approx(300)
        assert diag['stale_timeout_ms'] == 200
    assert len(frames) >= 1


@pytest.mark.parametrize('fault', ['tracking', 'feedback', 'return', 'action', 'abort'])
def test_menu_stop_saves_only_valid_prefix_when_current_cycle_is_invalid(session, fault):
    start(session)
    for ms in range(1060, 1300, 20):
        feed(session.buttons, ms, False)
        step(session, ms)
    feed(session.buttons, 1300, True)
    status = SimpleNamespace(command_send_finished_monotonic_ns=1_300_000_000,
        feedback_finished_monotonic_ns=1_299_000_000,
        feedback_valid=fault != 'feedback', tracking=fault != 'tracking',
        return_target='zero' if fault == 'return' else '',
        feedback_abort_requested=fault == 'abort')
    # Invalid data here would fail frame construction if the stop cycle were sampled.
    session.on_step({}, None if fault == 'action' else {}, status, sample(1300))
    wait_for(lambda: session.state == 'IDLE')
    saved, metadata, frames = session._writer.ends[-1]
    assert saved and metadata['reason'] == 'left_menu'
    assert metadata['frames'] == len(frames) == 7
    assert frames[-1]['action.host_time'][0] == pytest.approx((1_280_000_000-session.session_ns)*1e-9)
    session.on_stop()
    assert session._writer.saved == 1
