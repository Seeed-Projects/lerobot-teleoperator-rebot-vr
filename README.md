# reBot VR Teleoperation

**English** | [简体中文](README.zh-CN.md)

<h3 align="center">reBot VR Teleoperation with PICO 4</h3>

<p align="center">
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://github.com/huggingface/lerobot"><img alt="LeRobot" src="https://img.shields.io/badge/LeRobot-0.6.x-orange?style=for-the-badge"></a>
  <img alt="split IK" src="https://img.shields.io/badge/IK-split-green?style=for-the-badge">
  <a href="https://www.picoxr.com"><img alt="PICO 4" src="https://img.shields.io/badge/PICO-4-blueviolet?style=for-the-badge"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue?style=for-the-badge"></a>
</p>

> reBot VR Teleoperation brings VR control to reBot B601-RS and B601-DM robotic arms. PICO 4 controllers let operators intuitively control end-effector position, orientation, and gripper motion for single-arm tasks or coordinated two-arm operation. Built with LeRobot, the project provides calibration, motion control, and runtime logging for robot demonstrations and teleoperation data collection.

* * *

## Overview

- **VR teleoperation**: Map PICO 4 controller poses to robot motion in real time, with single- and dual-arm operation, Grip clutching, button-triggered returns, and gripper control.
- **Split IK**: A position QP for q1–q3 tracks the joint4 pivot, while a closed-form solution for q4–q6 follows wrist orientation. Wrist rotation does not interfere with shoulder position tracking.
- **Dual-arm coordination and safety**: RS supports MIT and DM supports MIT / POS_VEL, with synchronized startup, coordinated fault handling, joint-limit margins, braking constraints, fault holds, startup positioning, and return to zero on exit.
- **Diagnostics**: CSV logging, latency percentile summaries, and an offline analysis API for reviewing runs and tuning parameters.

* * *

## Demo

<p align="center">
  <img src="assest/dual-arm-block-grab.gif" width="600" alt="Dual-arm VR teleoperation demo">
</p>

## Installation

Requires Linux, Python 3.12+, LeRobot 0.6.x, and PICO 4. RS uses SocketCAN; DM uses a Damiao serial-to-CAN adapter.

**1. Install the LeRobot environment**

Run these commands inside the `lerobot` directory, replacing the path with your local checkout. If the environment already exists, activate it without recreating it.

```bash
cd /path/to/lerobot
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

**2. Install the teleoperation plugin**

Keep that environment active, then install from this project's root directory:

```bash
cd /path/to/lerobot-teleoperator-rebot-vr
uv pip install -e .
```

**3. Install the sender app on PICO 4**

Enable developer mode and USB debugging on PICO 4, connect it to the computer via USB, and authorize debugging in the headset. Run this command on the computer from this project's root directory to install the APK on the **PICO 4 headset**:

```bash
adb install -r assest/reBot.apk
```

## B601-RS

### Enable CAN Interfaces

Connect the USB-to-CAN adapters, then enable the interfaces at **1 Mbps**. A single arm needs only the first command; skip interfaces that are already enabled.

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

### Single Arm (MIT)

Bring up `can0` at 1 Mbps. Align the arm to the official zero pose, fully close the gripper, and support the arm before calibration. Use the same robot ID for calibration and teleoperation.

```bash
lerobot-calibrate \
  --robot.type=rebot_b601_rs_follower \
  --robot.port=can0 \
  --robot.id=rebot_b601_rs_vr

rebot-vr-teleoperate \
  --robot-model b601_rs \
  --robot-port can0 \
  --robot-id rebot_b601_rs_vr \
  --control-config config/rs_mit_split.yaml
```

### Dual Arms (MIT)

The [dual-arm profile](config/dual_rs_mit_split.yaml) uses left `can0` and right `can1`. Calibrate each arm separately:

```bash
lerobot-calibrate --robot.type=rebot_b601_rs_follower --robot.port=can0 --robot.id=rebot_b601_rs_left
lerobot-calibrate --robot.type=rebot_b601_rs_follower --robot.port=can1 --robot.id=rebot_b601_rs_right

# Validate configuration without connecting hardware
rebot-vr-teleoperate-dual --config config/dual_rs_mit_split.yaml --dry-run

# Low-speed startup/return-to-zero check; physically moves both arms
rebot-vr-teleoperate-dual --config config/dual_rs_mit_split.yaml --startup-zero-test

# Teleoperation
rebot-vr-teleoperate-dual --config config/dual_rs_mit_split.yaml
```

## B601-DM

### Single Arm

Adjust the serial port, calibrate the zero position, then start:

```bash
lerobot-calibrate \
  --robot.type=rebot_b601_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=rebot_b601_vr

rebot-vr-teleoperate \
  --robot-port /dev/ttyACM0 \
  --motor-control-mode mit \
  --control-config config/mit_split.yaml
```

For POS_VEL, use mode `pos_vel` and configuration `config/pos_vel_split.yaml`.

### Dual Arms

Set each `robot_port` in the [DM dual-arm profile](config/dual_mit_split.yaml). Calibrate each arm using its port and the ID `rebot_left` / `rebot_right`, then start:

```bash
rebot-vr-teleoperate-dual --config config/dual_mit_split.yaml
```

Use `--dry-run` to validate configuration or `--startup-zero-test` for a low-speed check.

## Controller Inputs

| Input | Behavior |
| --- | --- |
| Grip | Hold to follow; release to hold position. Fully release before reactivating after startup or tracking recovery |
| Trigger | Control the corresponding gripper independently |
| Right A / Left X | Return the corresponding arm to its initial pose |
| Right B / Left Y | Return the corresponding arm to zero and close its gripper |
| Left Menu | Recording program only: start / stop and save an episode |
| Ctrl+C | Return to zero on normal exit; press again to abort the return. This is not a hardware emergency stop |

## Data Recording

Supports **one RS arm and two USB cameras (main/wrist)**. Images, joint states, actions, and task instructions are saved in **LeRobot v3** format at 25 FPS by default.

Press **Left Menu** to start / stop and save; wait for `SAVED`. B returns to zero without interrupting recording. `Ctrl+C` discards an unsaved episode.

Output: `recordings/RS_task_datetime/`. `data/` contains joint states, actions, and timestamps (Parquet); `videos/` contains both camera streams (MP4); `meta/` contains tasks, robot details, recording times, and statistics.

Set camera paths, CAN port, and the calibrated robot ID in [config/record_rs.yaml](config/record_rs.yaml). With the LeRobot environment active, run from this project's root:

```bash
uv pip install -e '.[record]'
rebot-vr-record --config config/record_rs.yaml --task "Stack the green block on the blue block"
```
Example Dataset: https://huggingface.co/datasets/Wangzhaowu/rebot_b601_rs_single_arm_block_stacking


## Operating Notes

- Startup moves the arms automatically. Check physical zero positions, joint directions, and clearance before running.
- Dual arms require upright bases facing the same direction and independent CAN buses. Inter-arm collision detection is not provided. Tracking or feedback faults on either side stop both arms from following.
- Prepare reliable support before shutdown: disabling torque can let an arm fall, and torque retention does not protect against power loss.

## Logs and Documentation

RS / DM dual-arm logs are stored in `logs/dual_rs/` and `logs/dual/`. For single-arm logging, add `--csv-log logs/session.csv`.

Offline tests:

```bash
uv pip install -e '.[test]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

License: [Apache-2.0](LICENSE)
