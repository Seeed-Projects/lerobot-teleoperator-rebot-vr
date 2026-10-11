"""RS transport/runner contracts; no hardware is opened by these tests."""

from types import SimpleNamespace
import math
import threading
import time

import numpy as np
import pytest

from lerobot_teleoperator_rebot_vr.constants import JOINT_NAMES
from lerobot_teleoperator_rebot_vr.control.controller import FullBodyQPIKController
from lerobot_teleoperator_rebot_vr.control.dynamics import B601GravityCompensator
from lerobot_teleoperator_rebot_vr.control.mit import MITCommandDispatcher
from lerobot_teleoperator_rebot_vr.control.types import CartesianControlConfig
from lerobot_teleoperator_rebot_vr.hardware.profiles import RS
from lerobot_teleoperator_rebot_vr.hardware.rs import RSConnection, RSFollower, RSFollowerConfig
from lerobot_teleoperator_rebot_vr.ik.kinematics import B601Kinematics
from lerobot_teleoperator_rebot_vr.ik.split_solver import SplitIKSolver
from lerobot_teleoperator_rebot_vr.runtime.cli import build_parser, follower_relative_target, validate_args
from lerobot_teleoperator_rebot_vr.runtime.shutdown import ManagedFollower, ShutdownPolicy
from lerobot_teleoperator_rebot_vr.vr.models import VRFrame


Q = np.array([.1, .7, 1.1, .2, -.1, .3, -.192])


class Motor:
    def __init__(self, index, events):
        self.index, self.events = index, events
        self.pos = Q[index - 1]
        self.error = None
        self.fault = 0
        self.barrier = None

    def robstride_get_param_f32_host_id(self, register, host, timeout_ms):
        assert (register, host) == (0x7019, 0xFD)
        assert 0 < timeout_ms <= 1000
        if self.barrier:
            self.barrier.wait(timeout=2)
        self.events.append((self.index, "read"))
        if self.error:
            raise self.error
        return self.pos

    def ensure_mode(self, mode, timeout):
        assert int(mode) == 1
        self.events.append((self.index, "mode"))

    def enable(self):
        self.events.append((self.index, "enable"))

    def disable(self):
        self.events.append((self.index, "disable"))

    def send_mit(self, *values):
        self.events.append((self.index, "mit", values))

    def request_feedback(self):
        self.events.append((self.index, "request"))

    def get_state(self):
        return SimpleNamespace(status_code=self.fault, can_id=self.index,
                               arbitration_id=0x020000FD | (self.index << 8))

    def close(self):
        self.events.append((self.index, "close"))


class Bus:
    def __init__(self, channel):
        assert channel == "can0"
        self.events = []
        self.motors = []

    def add_robstride_motor(self, index, host, model):
        assert host == 0xFD
        assert model == ("rs-06" if index <= 3 else "rs-00")
        motor = Motor(index, self.events)
        self.motors.append(motor)
        return motor

    def poll_feedback_once(self):
        pass

    def close_bus(self):
        self.events.append((0, "close_bus"))

    def close(self):
        self.events.append((0, "close"))


def test_rs_selection_does_not_change_dm_defaults_and_rejects_wrong_profiles():
    parser = build_parser()
    dm = parser.parse_args([])
    assert (dm.robot_model, dm.motor_control_mode, dm.can_adapter) == ("b601_dm", "pos_vel", "damiao")
    rs = parser.parse_args(["--robot-model", "b601_rs"])
    validate_args(rs)
    assert (rs.motor_control_mode, rs.can_adapter, rs.robot_port) == ("mit", "socketcan", "can0")
    assert rs.move_to_initial and rs.return_to_zero_on_exit
    assert rs.initial_q == pytest.approx([0., .8, .8, 0., 0., 0.])
    assert not rs.rs_zero_confirmed and rs.gripper_enabled
    assert rs.disable_torque_on_disconnect
    assert rs.exit_zero_tolerance_deg == 2.0
    with pytest.raises(SystemExit):
        parser.parse_args(["--robot-model", "b601_rs", "--motor-control-mode", "pos_vel"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--robot-model", "b601_rs", "--control-config", "config/mit_split.yaml"])
    rs.can_adapter = "damiao"
    with pytest.raises(ValueError, match="SocketCAN"):
        validate_args(rs)


def test_rs_gripper_requires_measured_endpoints_and_accepts_either_direction():
    parser = build_parser()
    args = parser.parse_args(["--robot-model", "b601_rs", "--gripper-enabled"])
    args.gripper_open_deg = None
    with pytest.raises(ValueError, match="provided"):
        validate_args(args)
    for opened, closed in ((310.12, 0.), (30., -11.), (-80., -11.)):
        args.gripper_open_deg, args.gripper_closed_deg = opened, closed
        validate_args(args)
        CartesianControlConfig(robot_model="b601_rs", gripper_open_deg=opened,
                               gripper_closed_deg=closed)


def test_read_only_probe_batches_independent_reads_without_motion_or_mode_calls():
    connection = RSConnection(controller_factory=Bus)
    connection.open()
    bus = connection.bus
    barrier = threading.Barrier(7)
    for motor in connection.motors.values():
        motor.barrier = barrier
    try:
        results = connection.read_positions()
        assert [results[name][0] for name in JOINT_NAMES] == pytest.approx(Q)
        assert all(end >= start for _, start, end in results.values())
    finally:
        connection.close()
    assert not any(event[1] in ("enable", "disable", "mode", "mit") for event in bus.events)
    assert sum(e[1] == "close" for e in bus.events) == 8


def test_unconfirmed_zero_cannot_be_bypassed_by_no_calibrate(tmp_path):
    robot = RSFollower(RSFollowerConfig(calibration_dir=tmp_path), controller_factory=lambda _: pytest.fail("bus opened"))
    with pytest.raises(RuntimeError, match="unconfirmed"):
        robot.connect(calibrate=False)


@pytest.mark.parametrize("bad", [float("nan"), -1.])
def test_all_feedback_must_pass_before_any_axis_enables(bad):
    def factory(channel):
        bus = Bus(channel)
        original = bus.add_robstride_motor
        def add(index, host, model):
            motor = original(index, host, model)
            if index == 3:
                motor.pos = bad
            return motor
        bus.add_robstride_motor = add
        return bus
    robot = RSFollower(RSFollowerConfig(zero_confirmed=True), controller_factory=factory)
    try:
        with pytest.raises(RuntimeError):
            robot.connect()
        assert not any(e[1] in ("mode", "enable", "mit") for e in robot.bus.events)
    finally:
        robot.close()


def test_read_failure_invalidates_complete_snapshot_and_does_not_use_sdk_cache():
    robot = RSFollower(RSFollowerConfig(), controller_factory=Bus)
    robot.open()
    try:
        assert robot.get_observation()["elbow_flex.pos"] == pytest.approx(math.degrees(Q[2]))
        robot.motors["elbow_flex"].error = TimeoutError("missing reply")
        with pytest.raises(RuntimeError, match="elbow_flex"):
            robot.get_observation()
        assert not robot.feedback_intervals_ns
        assert not robot._last_positions_rad
    finally:
        robot.close()


def test_cached_motor_faults_and_excessive_read_span_reject_feedback():
    robot = RSFollower(RSFollowerConfig(), controller_factory=Bus)
    robot.open()
    try:
        robot.motors["wrist_yaw"].fault = 4
        with pytest.raises(RuntimeError, match="fault bits"):
            robot.get_observation()
        robot.motors["wrist_yaw"].fault = 0
        robot.config.max_feedback_span_s = 0
        with pytest.raises(RuntimeError, match="freshness budget"):
            robot.get_observation()
    finally:
        robot.close()


def test_rs_model_split_and_gravity_use_native_positive_shoulder_coordinates():
    model = B601Kinematics(RS.model_path())
    try:
        controller = FullBodyQPIKController(model, xr_to_base_rotation=np.eye(3),
            config=CartesianControlConfig(robot_model="b601_rs"))
        assert controller.home_q_rad[1:3] == pytest.approx([.8, .8])
        assert controller.lower_limit_rad[1:3] == pytest.approx([0., 0.])
        gravity = B601GravityCompensator(RS.model_path(dynamics=True))
        solver = SplitIKSolver(model, max_solve_time_ms=1000)
        rng = np.random.default_rng(5)
        for _ in range(12):
            q = rng.uniform(np.array(RS.lower_rad) + .2, np.array(RS.upper_rad) - .2)
            pos, rot = model.forward_kinematics(q)
            anchor, _ = model.wrist_anchor_pose(q)
            moved = q.copy()
            moved[3:] += .1
            assert model.wrist_anchor_pose(moved)[0] == pytest.approx(anchor, abs=1e-10)
            result = solver.solve(target_position=anchor, target_rotation=rot,
                q_actual=q, q_nominal=q, dq_previous=np.zeros(6), dt=.02,
                max_joint_speed=np.ones(6), max_joint_acceleration=np.ones(6))
            assert result.success
            assert np.max(np.abs(result.joint_velocity_rad_s)) < .001
            assert np.isfinite(gravity.gravity_torque(q)).all()
    finally:
        model.close()


def test_rs_disabled_gripper_ignores_trigger_and_zero_button():
    model = B601Kinematics(RS.model_path())
    try:
        config = CartesianControlConfig(robot_model="b601_rs", gripper_enabled=False,
                                        gripper_open_deg=None, gripper_closed_deg=None)
        controller = FullBodyQPIKController(model, xr_to_base_rotation=np.eye(3), config=config)
        observation = {f"{name}.pos": math.degrees(value) for name, value in zip(JOINT_NAMES, Q)}
        for i, (trigger, button) in enumerate(((0., False), (1., False), (0., True))):
            stamp = time.monotonic_ns()
            frame = VRFrame(grip_pos=np.zeros(3), grip_quat=[0, 0, 0, 1],
                            squeeze=0., trigger=trigger, secondary_button=button,
                            received_monotonic_ns=stamp)
            action, status = controller.update(frame, observation, .02, now_ns=stamp)
            assert status.feedback_valid
            assert action["gripper.pos"] == pytest.approx(math.degrees(Q[6]))
    finally:
        model.close()


@pytest.mark.parametrize("trigger", [0., 1.])
def test_rs_measured_gripper_travel_maps_trigger_without_grip(trigger):
    args = build_parser().parse_args(["--robot-model", "b601_rs"])
    validate_args(args)
    model = B601Kinematics(RS.model_path())
    robot = RSFollower(RSFollowerConfig(max_relative_target=follower_relative_target(
        args.max_relative_target_deg, args.wrist_relative_target_deg,
        args.gripper_relative_target_deg)), controller_factory=Bus)
    robot.open()
    try:
        robot.motors["gripper"].pos = math.radians(155.06)
        observation = robot.get_observation()
        controller = FullBodyQPIKController(model, xr_to_base_rotation=np.eye(3),
            config=CartesianControlConfig(robot_model="b601_rs", gripper_enabled=args.gripper_enabled,
                gripper_open_deg=args.gripper_open_deg, gripper_closed_deg=args.gripper_closed_deg,
                gripper_max_speed_deg_s=args.gripper_max_speed_deg_s,
                gripper_max_acceleration_deg_s2=args.gripper_max_acceleration_deg_s2,
                max_command_feedback_error_deg=args.max_relative_target_deg * .9,
                gripper_command_feedback_error_deg=args.gripper_relative_target_deg * .9))
        stamp = time.monotonic_ns()
        frame = VRFrame(grip_pos=np.zeros(3), grip_quat=[0, 0, 0, 1],
                        squeeze=0., trigger=trigger, received_monotonic_ns=stamp)
        action, status = controller.update(frame, observation, .02, now_ns=stamp)
        assert status.feedback_valid
        assert controller.gripper.goal_deg == pytest.approx(310.12 if trigger == 0 else 0.)
        command = action["gripper.pos"]
        assert 0. <= command <= 310.12
        assert (command > 155.06) if trigger == 0 else (command < 155.06)
        assert abs(command - 155.06) <= args.gripper_max_speed_deg_s * .02
        dispatcher = MITCommandDispatcher(robot, robot_model="b601_rs", kp=np.ones(6),
            kd=np.ones(6), torque_limit_nm=np.ones(6), arm_velocity_limit_rad_s=np.ones(6),
            gripper_velocity_limit_deg_s=args.gripper_max_speed_deg_s)
        dispatcher.set_observation(observation)
        dispatcher.stop_arm_velocity(immediate=True)  # Grip is released; Trigger still works.
        dispatcher.send_action(action, gripper_velocity_deg_s=status.gripper_velocity_deg_s)
        assert robot.bus.events[-1][0] == 7
        assert robot.bus.events[-1][2][0] == pytest.approx(math.radians(command))
        assert robot.bus.events[-1][2][1] == pytest.approx(math.radians(status.gripper_velocity_deg_s))
        assert status.gripper_velocity_deg_s != 0.
        # Reproduce a stationary gripper: the RS profile must build more effort
        # than the inherited 3-deg arm window, but remain bounded in both directions.
        for _ in range(100):
            stamp += 20_000_000
            frame = VRFrame(grip_pos=np.zeros(3), grip_quat=[0, 0, 0, 1],
                            squeeze=0., trigger=trigger, received_monotonic_ns=stamp)
            action, status = controller.update(frame, observation, .02, now_ns=stamp)
            previous = command
            sent = dispatcher.send_action(action, gripper_velocity_deg_s=status.gripper_velocity_deg_s)
            command = sent["gripper.pos"]
            assert abs(command - previous) <= args.gripper_max_speed_deg_s * .02 + 1e-6
            assert abs(command - 155.06) <= 10.8 + 1e-6
        assert command - 155.06 == pytest.approx(10.8 if trigger == 0 else -10.8)
        # Contact stops trajectory feedforward instead of increasing holding force.
        assert status.gripper_velocity_deg_s == pytest.approx(0.)
        assert robot.bus.events[-1][2][1] == pytest.approx(0.)
        # Dispatcher still independently clamps even a malformed large target.
        sent = dispatcher.send_action({**action, "gripper.pos": 310.12 if trigger == 0 else 0.},
                                      gripper_velocity_deg_s=240. if trigger == 0 else -240.)
        assert sent["gripper.pos"] - 155.06 == pytest.approx(12. if trigger == 0 else -12.)
        assert robot.bus.events[-1][2][1] == 0.
        action, status = controller.update(None, observation, .02, now_ns=stamp + 300_000_000)
        assert not status.tracking
        assert status.gripper_velocity_deg_s == 0.
    finally:
        model.close()
        robot.close()


@pytest.mark.parametrize("gripper_enabled", [False, True])
def test_rs_shared_runner_holds_pose_and_cleans_up_without_claiming_disabled(monkeypatch, tmp_path, caplog, gripper_enabled):
    from lerobot_teleoperator_rebot_vr.runtime import real
    from lerobot_teleoperator_rebot_vr.hardware import rs

    robots = []
    def create(config):
        robot = RSFollower(config, controller_factory=Bus)
        robots.append(robot)
        return robot
    monkeypatch.setattr(rs, "RSFollower", create)
    monkeypatch.setattr(real.signal, "signal", lambda *args: None)
    def sample():
        return (VRFrame(grip_pos=np.zeros(3), grip_quat=[0, 0, 0, 1], squeeze=0.,
                        trigger=1., received_monotonic_ns=time.monotonic_ns())
                if gripper_enabled else None)
    monkeypatch.setattr(real, "make_vr_controller", lambda config: SimpleNamespace(
        connect=lambda: None, disconnect=lambda: None, latest_sample=sample))
    from lerobot_teleoperator_rebot_vr.hardware.rs_calibration import RSCalibrationStore
    RSCalibrationStore(directory=tmp_path).save(dict.fromkeys(JOINT_NAMES, 0.), JOINT_NAMES)
    args = build_parser().parse_args(["--robot-model", "b601_rs", "--calibration-dir", str(tmp_path),
        "--no-move-to-initial", "--gripper-enabled" if gripper_enabled else "--no-gripper-enabled",
        "--duration", ".06", "--disable-attempts", "1", "--disable-interval-s", ".001",
        "--disable-feedback-wait-s", ".001", "--csv-log", str(tmp_path / "rs.csv")])
    validate_args(args)
    buses = []
    original = RSFollower.open
    def opened(self):
        original(self)
        if gripper_enabled:
            self.motors["gripper"].pos = math.radians(155.)
        buses.append(self.bus)
    monkeypatch.setattr(RSFollower, "open", opened)
    real._run_arm(args)
    events = buses[0].events
    assert sum(e[1] == "enable" for e in events) == 7
    assert sum(e[1] == "disable" for e in events) == 0
    commands = [e for e in events if e[1] == "mit" and e[2][2] > 0]
    assert commands
    for index, _, values in commands:
        if index == 7 and gripper_enabled:
            assert abs(math.degrees(values[0]) - 155.) <= 10.8 + 1e-6
            assert -math.radians(args.gripper_max_speed_deg_s) <= values[1] <= 0.
            continue
        assert values[0] == pytest.approx(Q[index-1])
        assert values[1] == 0.
    if gripper_enabled:
        assert any(index == 7 and values[1] < 0. for index, _, values in commands)
    assert robots[0]._reader is None
    assert robots[0].bus is None
    assert (tmp_path / "rs_shutdown.json").is_file()
    import json
    report = json.loads((tmp_path / "rs_shutdown.json").read_text())
    assert all(not motor["confirmed"] for motor in report["motors"])
    assert all("retention" in motor["reason"] for motor in report["motors"])
    assert "automatic torque-off suppressed" in caplog.text
    assert "cached_status=DISABLED" not in caplog.text
    import csv
    with (tmp_path / "rs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    samples = [row for row in rows if row["phase"] == "teleop"]
    assert samples
    assert all(row["gripper_enabled"] == str(gripper_enabled) for row in samples)
    assert all(row["gripper_command_sent"] == "True" for row in samples)
    assert all(row["vr_hand"] == "right" for row in samples)
    if gripper_enabled:
        assert any(float(row["mit_desired_velocity_gripper_deg_s"]) < 0. for row in samples)
    else:
        assert float(samples[0]["sent_gripper_deg"]) == pytest.approx(math.degrees(Q[6]))
        assert all(float(row["mit_desired_velocity_gripper_deg_s"]) == 0. for row in samples)
    assert float(samples[0]["mit_kp_gripper_nm_rad"]) == 5.


def test_rs_torque_ceiling_does_not_change_dm_ceiling():
    args = build_parser().parse_args(["--robot-model", "b601_rs"])
    args.mit_torque_limit_nm = list(RS.effort_nm)
    validate_args(args)
    args.robot_model = "b601_dm"
    args.gripper_open_deg, args.gripper_closed_deg = -180., 0.
    with pytest.raises(ValueError, match="torque limits"):
        validate_args(args)


def test_runtime_rejects_unconfirmed_zero_before_opening_vr_or_model(monkeypatch, tmp_path):
    from lerobot_teleoperator_rebot_vr.runtime import real
    monkeypatch.setattr(real, "B601Kinematics", lambda *a: pytest.fail("model opened"))
    monkeypatch.setattr(real, "make_vr_controller", lambda *a: pytest.fail("VR opened"))
    args = build_parser().parse_args(["--robot-model", "b601_rs", "--calibration-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="zeros are unconfirmed"):
        real._run_arm(args)


@pytest.mark.parametrize("operation", ["ensure_mode", "enable"])
def test_partial_rs_connect_failure_still_disables_and_closes_every_axis(monkeypatch, operation):
    original = getattr(Motor, operation)
    def failing(self, *args):
        if self.index == 4:
            raise RuntimeError("injected connection failure")
        return original(self, *args)
    monkeypatch.setattr(Motor, operation, failing)
    robot = RSFollower(RSFollowerConfig(zero_confirmed=True), controller_factory=Bus)
    managed = ManagedFollower(robot, policy=ShutdownPolicy(1, .001, .001))
    with pytest.raises(RuntimeError, match="injected"):
        managed.connect()
    bus = robot.bus
    assert managed.needs_cleanup
    report = managed.disconnect()
    assert sum(e[1] == "disable" for e in bus.events) == 7
    assert sum(e[1] == "close" for e in bus.events) == 8
    assert report.close_bus_completed
    assert robot._reader is None and robot.bus is None


def test_read_only_transport_cleanup_continues_after_close_failure(monkeypatch):
    connection = RSConnection(controller_factory=Bus)
    connection.open()
    bus = connection.bus
    def failing():
        raise RuntimeError("injected close failure")
    monkeypatch.setattr(bus.motors[0], "close", failing)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        connection.close()
    assert sum(e[1] == "close" for e in bus.events) == 7
    assert not any(e[1] in ("disable", "enable", "mit") for e in bus.events)
    assert connection._reader is None and connection.bus is None


def test_rs_dispatcher_selects_native_gravity_and_propagates_send_failure(monkeypatch):
    robot = RSFollower(RSFollowerConfig(), controller_factory=Bus)
    robot.open()
    try:
        dispatcher = MITCommandDispatcher(robot, robot_model="b601_rs",
            kp=np.ones(6), kd=np.ones(6), torque_limit_nm=np.ones(6),
            arm_velocity_limit_rad_s=np.ones(6))
        assert dispatcher.gravity.urdf_path == RS.model_path(dynamics=True)
        observation = dispatcher.get_observation()
        def failing(*args):
            raise RuntimeError("injected send failure")
        monkeypatch.setattr(robot.motors["elbow_flex"], "send_mit", failing)
        with pytest.raises(RuntimeError, match="injected send"):
            dispatcher.send_action(observation)
        assert dispatcher._last_sent_goal_deg is None
    finally:
        robot.close()


@pytest.mark.parametrize("axis", range(6))
@pytest.mark.parametrize("edge", ["lower", "upper"])
@pytest.mark.parametrize("excess_deg, accepted", [(.01, True), (.11, False)])
def test_rs_feedback_tolerance_accepts_noise_but_keeps_commands_inside_limits(axis, edge, excess_deg, accepted):
    limits = RS.lower_rad if edge == "lower" else RS.upper_rad
    measured = limits[axis] + math.radians(excess_deg) * (-1 if edge == "lower" else 1)
    def factory(channel):
        bus = Bus(channel)
        original = bus.add_robstride_motor
        def add(index, host, model):
            motor = original(index, host, model)
            if index == axis + 1:
                motor.pos = measured
            return motor
        bus.add_robstride_motor = add
        return bus
    robot = RSFollower(RSFollowerConfig(zero_confirmed=True), controller_factory=factory)
    model = B601Kinematics(RS.model_path())
    controller = FullBodyQPIKController(model, xr_to_base_rotation=np.eye(3),
        config=CartesianControlConfig(robot_model="b601_rs", gripper_enabled=False))
    try:
        if accepted:
            robot.connect()
        else:
            with pytest.raises(RuntimeError, match="feedback tolerance"):
                robot.connect()
            assert not any(e[1] in ("mode", "enable", "mit") for e in robot.bus.events)
        observation = robot.get_observation()
        action, status = controller.update(None, observation, .02)
        assert status.feedback_valid == accepted
        # Keep the original measured value, including its tiny limit excess.
        assert observation[f"{JOINT_NAMES[axis]}.pos"] == pytest.approx(math.degrees(measured))
        if accepted:
            assert action is not None
            dispatcher = MITCommandDispatcher(robot, robot_model="b601_rs",
                kp=np.ones(6), kd=np.ones(6), torque_limit_nm=np.ones(6),
                arm_velocity_limit_rad_s=np.ones(6))
            dispatcher.set_observation(observation)
            dispatcher.send_action(action)
            for index, kind, *payload in robot.bus.events:
                if kind == "mit" and index <= 6:
                    assert RS.lower_rad[index - 1] <= payload[0][0] <= RS.upper_rad[index - 1]
        else:
            assert "outside_limits:" in status.feedback_fault_reason
    finally:
        model.close()
        robot.close()
