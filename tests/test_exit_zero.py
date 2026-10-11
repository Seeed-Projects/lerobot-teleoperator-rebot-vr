from types import SimpleNamespace
from unittest.mock import Mock
import signal

import numpy as np
import pytest

from lerobot_teleoperator_rebot_vr.control.types import ARM_JOINT_NAMES
from lerobot_teleoperator_rebot_vr.runtime import real, safety, shutdown
from lerobot_teleoperator_rebot_vr.runtime.cli import build_parser, validate_args


def observation(q_deg, gripper=-37.0):
    return {**{f"{name}.pos": float(q) for name, q in zip(ARM_JOINT_NAMES, q_deg)}, "gripper.pos": gripper}


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FollowingRobot:
    def __init__(self, *, follow=True):
        self.q_deg = np.array([10.0, -20.0, -15.0, 5.0, 10.0, -5.0])
        self.follow = follow
        self.actions = []
        self.config = SimpleNamespace(pos_vel_velocity=[315.0] * 3 + [687.0] * 3 + [1200.0])
        self.velocity_limits = []
        self.reads = 0

    def get_observation(self):
        self.reads += 1
        return observation(self.q_deg, -37.0 + self.reads * 0.01)

    def send_action(self, action):
        self.actions.append(action.copy())
        self.velocity_limits.append(list(self.config.pos_vel_velocity))
        if self.follow:
            self.q_deg = np.array([action[f"{name}.pos"] for name in ARM_JOINT_NAMES])
        return action


@pytest.fixture
def zero_setup(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(safety.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(safety.time, "sleep", clock.sleep)
    args = build_parser().parse_args([])
    args.fps = 100.0
    return args, clock


def run_zero(robot, args, should_stop=lambda: False):
    return safety.move_to_zero_pose(
        robot, lower_limit_rad=np.full(6, -3.0), upper_limit_rad=np.full(6, 3.0),
        args=args, should_stop=should_stop,
    )


def test_zero_trajectory_reaches_feedback_and_holds_gripper_with_lower_speed(zero_setup):
    args, _ = zero_setup
    args.wrist_speed_rad_s = 0.25
    robot = FollowingRobot()
    original_velocity = robot.config.pos_vel_velocity
    original_q = robot.q_deg.copy()
    assert run_zero(robot, args)
    assert np.max(np.abs(robot.q_deg)) <= args.initial_move_tolerance_deg
    commands = np.array([[a[f"{name}.pos"] for name in ARM_JOINT_NAMES] for a in robot.actions])
    steps = np.diff(np.vstack([original_q, commands]), axis=0)
    assert np.max(np.abs(np.deg2rad(steps))) <= 0.25 / args.fps + 1e-9
    assert all(a["gripper.pos"] == pytest.approx(-36.99) for a in robot.actions)
    assert all(v[:6] == pytest.approx([np.rad2deg(0.25)] * 6) for v in robot.velocity_limits)
    assert all(v[6] == 1200.0 for v in robot.velocity_limits)
    assert robot.config.pos_vel_velocity is original_velocity
    assert args.max_joint_speed_rad_s == 3.0


@pytest.mark.parametrize("failure", ["stall", "timeout"])
def test_zero_stops_when_feedback_never_reaches_target(zero_setup, failure):
    args, _ = zero_setup
    args.initial_stall_timeout = 0.1 if failure == "stall" else 5.0
    args.initial_move_timeout = 0.1 if failure == "timeout" else 5.0
    robot = FollowingRobot(follow=False)
    original_velocity = robot.config.pos_vel_velocity
    with pytest.raises(RuntimeError, match="not following" if failure == "stall" else "timed out"):
        run_zero(robot, args)
    assert robot.config.pos_vel_velocity is original_velocity
    assert len(robot.actions) < 30


@pytest.mark.parametrize("bad", [{}, observation([0] * 6, float("nan")), observation([float("nan")] * 6)])
def test_zero_rejects_invalid_feedback_before_sending(zero_setup, bad):
    args, _ = zero_setup
    robot = FollowingRobot()
    robot.get_observation = lambda: bad
    with pytest.raises(RuntimeError, match="feedback is invalid"):
        run_zero(robot, args)
    assert not robot.actions


def test_second_stop_during_feedback_prevents_next_zero_command(zero_setup):
    args, _ = zero_setup
    robot = FollowingRobot()
    assert not run_zero(robot, args, should_stop=lambda: robot.reads > 0)
    assert not robot.actions


@pytest.fixture
def runner(monkeypatch):
    import lerobot.robots.rebot_b601_follower as follower_module

    events = []
    handlers = {}
    args = build_parser().parse_args(["--no-move-to-initial"])
    status = SimpleNamespace(feedback_valid=True, feedback_abort_requested=False,
                             velocity_diagnostics={})
    clock = Clock()
    monkeypatch.setattr(shutdown, "time", SimpleNamespace(
        monotonic=clock.monotonic, monotonic_ns=lambda: int(clock.now * 1e9), sleep=clock.sleep))

    class Robot:
        def __init__(self, config):
            self.config = config
            self.is_connected = False
            self.bus = None
            self.motors = {}
            self.cameras = {}

        def connect(self, **kwargs):
            self.is_connected = True
            self.bus = SimpleNamespace(
                poll_feedback_once=lambda: None,
                close_bus=lambda: events.append("close_bus"),
                close=lambda: events.append("disconnect"))
            for name, (send_id, recv_id) in self.config.motor_can_ids.items():
                state = SimpleNamespace(can_id=send_id, arbitration_id=recv_id, status_code=0)
                self.motors[name] = SimpleNamespace(
                    disable=lambda name=name: events.append("disable_" + name),
                    request_feedback=lambda: None,
                    get_state=lambda state=state: state,
                    close=lambda name=name: events.append("close_" + name))
            events.append("connect")

        def disconnect(self):
            raise AssertionError("LeRobot disconnect must not be called")

        def get_observation(self):
            return observation([0] * 6)

    robot_holder = []

    def make_robot(config):
        robot = Robot(config)
        robot_holder.append(robot)
        return robot

    arm = SimpleNamespace(
        home_q_rad=np.zeros(6), lower_limit_rad=np.full(6, -3.0), upper_limit_rad=np.full(6, 3.0),
        start=lambda: events.append("arm_start"), stop=lambda: events.append("arm_stop"),
    )

    def update(*args):
        handlers[signal.SIGINT](signal.SIGINT, None)
        return None, status

    arm.update = update
    vr = SimpleNamespace(connect=lambda: events.append("vr_connect"),
                         disconnect=lambda: events.append("vr_stop"), latest_sample=lambda: None)
    kinematics = SimpleNamespace(close=lambda: events.append("kinematics_close"))

    def zero(*args, **kwargs):
        events.append("zero")
        assert not kwargs["should_stop"]()
        return True

    zero_mock = Mock(side_effect=zero)
    monkeypatch.setattr(real, "_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(real.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    monkeypatch.setattr(real, "B601Kinematics", lambda *args: kinematics)
    monkeypatch.setattr(real, "FullBodyQPIKController", lambda *args, **kwargs: arm)
    monkeypatch.setattr(real, "make_vr_controller", lambda *args, **kwargs: vr)
    monkeypatch.setattr(follower_module, "RebotB601Follower", make_robot)
    monkeypatch.setattr(real, "_move_to_zero_pose", zero_mock)
    return SimpleNamespace(args=args, events=events, handlers=handlers, arm=arm, status=status,
                           zero=zero_mock, robots=robot_holder)


def test_normal_ctrl_c_stops_vr_then_returns_before_disconnect(runner):
    real.main()
    assert runner.events.index("arm_stop") < runner.events.index("zero")
    assert runner.events.index("vr_stop") < runner.events.index("zero") < runner.events.index("disconnect")
    assert runner.events.index("zero") < runner.events.index("disable_shoulder_pan")
    assert runner.events.count("disable_gripper") == 3
    runner.zero.assert_called_once()
    assert runner.robots[0].config.disable_torque_on_disconnect


@pytest.mark.parametrize("observer_failure", [False, True])
def test_recording_observer_stops_before_exit_zero_and_failure_preserves_shutdown(runner, observer_failure):
    def ready():
        if observer_failure:
            raise OSError("injected observer failure")
    observer = SimpleNamespace(on_tracking=lambda *args: None, on_ready=ready,
        on_step=lambda *args: None, on_stop=lambda: runner.events.append("record_stop"))
    real._run_arm(runner.args, observer=observer)
    assert runner.events.index("record_stop") < runner.events.index("zero")
    assert runner.events.index("zero") < runner.events.index("disable_shoulder_pan")
    runner.zero.assert_called_once()


def test_runtime_forwards_gripper_mit_gains(runner):
    runner.args.gripper_mit_kp = 11.0
    runner.args.gripper_mit_kd = 0.45

    real.main()

    config = runner.robots[0].config
    assert config.gripper_mit_kp == pytest.approx(11.0)
    assert config.gripper_mit_kd == pytest.approx(0.45)
    assert config.mit_kp[-1] == pytest.approx(11.0)
    assert config.mit_kd[-1] == pytest.approx(0.45)


@pytest.mark.parametrize("reason", ["feedback", "error", "sigterm", "double_sigint", "disabled", "duration"])
def test_non_normal_exits_do_not_return_to_zero(runner, reason):
    def update(*args):
        sig = signal.SIGTERM if reason == "sigterm" else signal.SIGINT
        runner.handlers[sig](sig, None)
        if reason == "double_sigint":
            runner.handlers[sig](sig, None)
        if reason == "error":
            raise RuntimeError("injected control failure")
        if reason == "feedback":
            runner.status.feedback_valid = False
        return None, runner.status

    runner.arm.update = update
    if reason == "disabled":
        runner.args.return_to_zero_on_exit = False
    if reason == "duration":
        runner.args.duration = 1e-9
    if reason == "error":
        with pytest.raises(RuntimeError, match="injected control failure"):
            real.main()
    else:
        real.main()
    runner.zero.assert_not_called()
    assert "disconnect" in runner.events


@pytest.mark.parametrize("failure", ["exception", "second_ctrl_c"])
def test_failed_or_interrupted_return_still_disconnects(runner, failure):
    def zero(*args, **kwargs):
        runner.events.append("zero")
        if failure == "exception":
            raise RuntimeError("invalid zero feedback")
        runner.handlers[signal.SIGINT](signal.SIGINT, None)
        assert kwargs["should_stop"]()
        return False

    runner.zero.side_effect = zero
    if failure == "exception":
        with pytest.raises(RuntimeError, match="invalid zero feedback"):
            real.main()
    else:
        real.main()
    assert runner.events.index("zero") < runner.events.index("disconnect")

    assert not any(event.startswith("disable_") for event in runner.events)
    assert not runner.robots[0].config.disable_torque_on_disconnect


def test_startup_failure_does_not_attempt_zero(runner, monkeypatch):
    runner.args.move_to_initial = True
    monkeypatch.setattr(real, "_move_to_initial_pose", Mock(side_effect=RuntimeError("startup stalled")))
    with pytest.raises(RuntimeError, match="startup stalled"):
        real.main()
    runner.zero.assert_not_called()
    assert "disconnect" in runner.events


def test_ctrl_c_during_startup_can_return_to_zero(runner, monkeypatch):
    runner.args.move_to_initial = True

    def initial(*args, **kwargs):
        runner.handlers[signal.SIGINT](signal.SIGINT, None)
        return False

    monkeypatch.setattr(real, "_move_to_initial_pose", initial)
    real.main()
    runner.zero.assert_called_once()
    assert runner.events.index("zero") < runner.events.index("disconnect")


def test_persistent_feedback_fault_keeps_hold_exit_even_with_ctrl_c(runner, monkeypatch):
    runner.status.feedback_valid = False
    runner.status.feedback_abort_requested = True
    runner.status.feedback_fault_count = 5
    runner.status.feedback_fault_reason = "invalid feedback"
    monkeypatch.setattr(real, "replace", lambda status, **kwargs: status)
    monkeypatch.setattr(real, "_status_line", lambda *args: "fault")
    settle = Mock(side_effect=lambda *args, **kwargs: runner.events.append("hold"))
    monkeypatch.setattr(real, "_settle_persistent_feedback_fault", settle)
    with pytest.raises(safety.PersistentFeedbackFault):
        real.main()
    runner.zero.assert_not_called()
    settle.assert_called_once()
    assert runner.events.index("hold") < runner.events.index("disconnect")
    assert not runner.robots[0].config.disable_torque_on_disconnect
    assert not any(event.startswith("disable_") for event in runner.events)


def test_mit_zero_clears_previous_velocity_target(runner, monkeypatch):
    class MITProxy:
        def __init__(self, robot, **kwargs):
            self.robot = robot

        def __getattr__(self, name):
            return getattr(self.robot, name)

        def stop_arm_velocity(self, *, immediate):
            assert immediate
            runner.events.append("mit_velocity_reset")

    monkeypatch.setattr(real, "MITCommandDispatcher", MITProxy)
    runner.args.motor_control_mode = "mit"
    real.main()
    assert runner.events.index("mit_velocity_reset") < runner.events.index("zero")
    assert runner.events.index("zero") < runner.events.index("disconnect")


def test_normal_ctrl_c_respects_explicit_torque_retention(runner):
    runner.args.disable_torque_on_disconnect = False
    real.main()
    runner.zero.assert_called_once()
    assert not runner.robots[0].config.disable_torque_on_disconnect
    assert not any(event.startswith("disable_") for event in runner.events)


def test_partial_connection_failure_is_cleaned_without_homing(runner, monkeypatch):
    original_connect = real.ManagedFollower.connect

    def broken_connect(self, **kwargs):
        original_connect(self, **kwargs)
        self.robot.is_connected = False
        raise RuntimeError("mode configuration failed")

    monkeypatch.setattr(real.ManagedFollower, "connect", broken_connect)
    with pytest.raises(RuntimeError, match="mode configuration failed"):
        real.main()
    runner.zero.assert_not_called()
    assert runner.events.count("disable_gripper") == 3
    assert runner.events.index("close_bus") < runner.events.index("disconnect")


@pytest.mark.parametrize("option", ["--exit-zero-speed-rad-s", "--exit-zero-acceleration-rad-s2", "--exit-zero-tolerance-deg", "--exit-zero-stall-timeout"])
@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_exit_motion_limits_must_be_finite_and_positive(option, value):
    with pytest.raises(ValueError):
        validate_args(build_parser().parse_args([option, value]))


def test_zero_67_degree_residual_is_recorded_and_not_hidden_by_startup_tolerance(zero_setup, tmp_path):
    import csv
    from lerobot_teleoperator_rebot_vr.diagnostics.logger import CSVLogger

    args, _ = zero_setup
    args.initial_move_tolerance_deg = 10.0
    args.exit_zero_tolerance_deg = 3.0
    robot = FollowingRobot(follow=False)
    robot.q_deg = np.array([0., 0., -6.7, 0., 0., 0.])
    rows = []
    with pytest.raises(RuntimeError, match="not following"):
        safety.move_to_zero_pose(
            robot, lower_limit_rad=np.full(6, -3.), upper_limit_rad=np.full(6, 3.),
            args=args, should_stop=lambda: False, write_row=rows.append, arm_id="right",
        )
    assert rows[0]["motion_result"] == "started"
    assert rows[-1]["motion_result"] == "failed"
    assert rows[-2]["row_kind"] == "sample"
    last = rows[-1]
    assert last["phase"] == "exit_zero" and last["arm_id"] == "right"
    assert last["motion_tolerance_deg"] == 3.0
    assert last["actual_elbow_flex_deg"] == pytest.approx(-6.7)
    assert last["sent_elbow_flex_deg"] == pytest.approx(0.)
    assert last["error_elbow_flex_deg"] == pytest.approx(6.7)
    assert last["feedback_freshness"] == "unknown_no_receive_timestamp"
    assert args.initial_move_tolerance_deg == 10.0
    log = CSVLogger(tmp_path / "motion.csv")
    for row in rows:
        log.write_row(row)
    log.close()
    with log.output_path.open() as stream:
        written = list(csv.DictReader(stream))
    assert written[-1]["motion_result"] == "failed"


def test_startup_records_completion_and_interruption(zero_setup):
    args, _ = zero_setup
    for stop in (False, True):
        rows = []
        reached = safety.move_to_initial_pose(
            FollowingRobot(), target_rad=np.zeros(6), lower_limit_rad=np.full(6, -3.),
            upper_limit_rad=np.full(6, 3.), args=args, should_stop=lambda: stop,
            write_row=rows.append, arm_id="left",
        )
        assert reached is not stop
        assert rows[-1]["phase"] == "startup"
        assert rows[-1]["motion_result"] == ("interrupted" if stop else "completed")


def test_rs_two_degree_residual_cannot_complete_zero(zero_setup):
    args, _ = zero_setup
    args.exit_zero_tolerance_deg = .5
    robot = FollowingRobot(follow=False)
    robot.q_deg = np.array([0., .6, 2.63, 0., 0., 0.])
    with pytest.raises(RuntimeError, match="not following"):
        run_zero(robot, args)


def test_zero_requires_continuous_stability_not_three_passes_through_tolerance():
    from lerobot_teleoperator_rebot_vr.control.startup import StartupPoseMover
    mover = StartupPoseMover(np.zeros(6), lower_limit_rad=np.full(6, -3.),
        upper_limit_rad=np.full(6, 3.), max_speed_rad_s=.5, max_acceleration_rad_s2=1.,
        tolerance_rad=np.deg2rad(.5), settle_time_s=.5, settle_speed_rad_s=np.deg2rad(1.))
    for i in range(100):
        assert not mover.update(np.deg2rad([(-1)**i * .2] * 6), .02).done
    for _ in range(20):
        assert not mover.update(np.zeros(6), .02).done
    for _ in range(10):
        status = mover.update(np.zeros(6), .02)
    assert status.done


def test_rs_slow_near_zero_progress_is_not_misclassified_as_stall(zero_setup):
    args, clock = zero_setup
    args.robot_model = "b601_rs"
    args.exit_zero_tolerance_deg = .5
    started = clock.now
    robot = FollowingRobot(follow=False)

    def slow_feedback():
        # Only 0.06 deg improvement in 5s: old 0.125-deg progress threshold
        # would abort at 0.54 deg, despite continuous convergence.
        q4 = max(.45, .6 - .012 * (clock.now - started))
        return observation([0., .2, .3, q4, 0., 0.])

    robot.get_observation = slow_feedback
    assert run_zero(robot, args)
    assert 8.8 <= clock.now - started <= 10.
    assert slow_feedback()["wrist_flex.pos"] <= .5


def test_rs_stalled_just_outside_tolerance_still_fails_with_precise_diagnostics(zero_setup):
    args, _ = zero_setup
    args.robot_model = "b601_rs"
    args.exit_zero_tolerance_deg = .5
    robot = FollowingRobot(follow=False)
    robot.q_deg = np.array([0., .2, .3, .504, 0., 0.])
    with pytest.raises(RuntimeError, match=r"wrist_flex=0\.5040deg.*tolerance=0\.5000deg"):
        run_zero(robot, args)


@pytest.mark.parametrize("outcome", ["recovers", "stalled", "total_timeout"])
def test_separate_zero_stall_window_keeps_arrival_and_total_deadline(zero_setup, outcome):
    args, clock = zero_setup
    args.robot_model = "b601_rs"
    args.exit_zero_tolerance_deg = .5
    args.exit_zero_stall_timeout = 10.
    args.initial_move_timeout = 3. if outcome == "total_timeout" else 30.
    robot = FollowingRobot(follow=False)
    started = clock.now

    def delayed_feedback():
        # Synthetic breakaway after 8 s: exercises timing, not a prediction
        # that the real wrist will overcome friction within this window.
        wrist = .4 if outcome == "recovers" and clock.now - started >= 8. else .76
        return observation([0., .2, .3, wrist, 0., 0.])

    robot.get_observation = delayed_feedback
    if outcome == "recovers":
        assert run_zero(robot, args)
        assert 8.5 <= clock.now - started < 9.
    else:
        match = "timed out" if outcome == "total_timeout" else "wrist_flex=0.7600deg"
        with pytest.raises(RuntimeError, match=match):
            run_zero(robot, args)
        expected = 3. if outcome == "total_timeout" else 10.
        assert expected <= clock.now - started < expected + .1
    assert args.initial_stall_timeout == 5.  # Startup and caller remain unchanged.
    assert args.exit_zero_tolerance_deg == .5


def test_rs_recovers_from_worst_joint_switch_and_transient_overshoot(zero_setup):
    args, clock = zero_setup
    args.robot_model = "b601_rs"
    args.exit_zero_tolerance_deg = .5
    started = clock.now
    robot = FollowingRobot(follow=False)

    def coupled_feedback():
        t = clock.now - started
        q3 = 1.7 if t < .7 else max(.3, 1.7 - (t - .7) * (1.4 / .3))
        q4 = min(.67, .02 + max(0., t - .9) * (.65 / .4))
        if t > 2.:
            q4 = max(.45, .67 - .035 * (t - 2.))
        return observation([0., .2, q3, q4, 0., 0.])

    robot.get_observation = coupled_feedback
    # Global error briefly reaches ~0.3 before q4 rises to 0.67, then recovers.
    # An all-time minimum would abort after 5s even though q4 is improving.
    assert run_zero(robot, args)
    assert 7.3 <= clock.now - started <= 8.


def test_recent_joint_progress_does_not_let_one_axis_mask_a_stalled_axis():
    monitor = safety.JointProgressWindow(np.deg2rad(.5), 5., np.deg2rad(.025))
    for i in range(51):
        stalled = monitor.update(np.deg2rad([0., 10. - i * .1, .6, 0., 0., 0.]), i * .1)
    assert stalled == [2]


def test_recent_joint_progress_rejects_sustained_divergence():
    monitor = safety.JointProgressWindow(np.deg2rad(.5), 5., np.deg2rad(.025))
    for i in range(51):
        stalled = monitor.update(np.deg2rad([0., 0., .6 + i * .01, 0., 0., 0.]), i * .1)
    assert stalled == [2]


def test_old_progress_expires_when_axis_stops_improving():
    monitor = safety.JointProgressWindow(np.deg2rad(.5), 5., np.deg2rad(.025))
    failures = []
    for i in range(101):
        t = i * .1
        stalled = monitor.update(np.deg2rad([0., 0., max(.7, 1. - t * .1), 0., 0., 0.]), t)
        if stalled:
            failures.append(t)
    assert failures
    assert 7. <= failures[0] <= 8.1
