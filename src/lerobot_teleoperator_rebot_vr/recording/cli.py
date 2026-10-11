"""RS single-arm VR recording entry point; ordinary teleoperation is unchanged."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import signal
import time

from .buttons import LeftMenuButton
from .config import load_config


def make_runtime_args(config):
    from ..runtime.cli import build_parser, validate_args
    argv = ["--robot-model", "b601_rs"]
    for key, value in config.teleoperation.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.robot_model != "b601_rs" or args.motor_control_mode != "mit":
        raise ValueError("First recording version supports RS single-arm MIT only")
    # Integer downsampling makes fixed-rate frame timing auditable.
    ratio = args.fps / config.fps
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-8:
        raise ValueError("teleoperation fps must be an integer multiple of recording fps (e.g. 50 / 25)")
    return args


def check_input(config, args):
    from ..config_rebot_vr import RebotVRConfig
    from ..vr.controller import make_vr_controller
    button = LeftMenuButton(config.menu_field, args.stale_timeout)
    controller = make_vr_controller(RebotVRConfig(hand_side=args.hand,
        ws_host=args.host, ws_port=args.port), on_tracking=button.feed)
    print(f"Input-only check; no cameras or CAN. Looking for Controller.left.{config.menu_field}.\n"
          "Release Left Menu, then press/release it. Ctrl+C exits.", flush=True)
    controller.connect()
    last_report = None
    try:
        while True:
            report = (button.available, button.left_fields)
            if report != last_report:
                print(f"Menu field available={button.available}; left fields={button.left_fields}", flush=True)
                last_report = report
            for _event in button.drain(time.monotonic_ns()):
                print("LEFT MENU PRESSED", flush=True)
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        controller.disconnect()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record local LeRobot datasets from one RS arm and two USB cameras")
    parser.add_argument("--config", type=Path, default=Path("config/record_rs.yaml"))
    parser.add_argument("--task", help="Task/content description, saved in the folder name and dataset")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without hardware or output creation")
    parser.add_argument("--check-input", action="store_true", help="Check Left Menu events without CAN or cameras")
    cli = parser.parse_args(argv)
    config = load_config(cli.config, task=cli.task)
    args = make_runtime_args(config)
    if cli.dry_run:
        print(json.dumps({"recording": asdict(config), "teleoperation": vars(args)},
                         ensure_ascii=False, indent=2, default=str))
        return
    if cli.check_input:
        check_input(config, args)
        return
    if not config.task.strip():
        parser.error("provide --task describing the demonstration, or set task in the recording YAML")
    # Verify calibration before opening cameras or creating dataset directories.
    from ..hardware.rs_calibration import RSCalibrationStore
    if not RSCalibrationStore(args.robot_id, args.calibration_dir).load():
        parser.error(f"missing RS calibration for {args.robot_id}; run lerobot-calibrate first")
    from ..runtime.real import ArmRuntime
    from .session import RecordingSession
    recorder = RecordingSession(config, args)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    recorder.open()  # Cameras and LeRobot/codec preflight before any motor enable.
    try:
        try:
            ArmRuntime(args, observer=recorder).run()
        except BaseException:
            # Preserve the primary robot exception if IO teardown also fails.
            try:
                recorder.close()
            except Exception:
                logging.getLogger(__name__).exception("Recording cleanup also failed")
            raise
        else:
            # Join disk IO only after the robot's shutdown procedure completes.
            recorder.close()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
