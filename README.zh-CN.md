# reBot VR Teleoperation

[English](README.md) | **简体中文**

<h3 align="center">基于 PICO 4 的 reBot VR 遥操作系统</h3>

<p align="center">
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://github.com/huggingface/lerobot"><img alt="LeRobot" src="https://img.shields.io/badge/LeRobot-0.6.x-orange?style=for-the-badge"></a>
  <img alt="split IK" src="https://img.shields.io/badge/IK-split-green?style=for-the-badge">
  <a href="https://www.picoxr.com"><img alt="PICO 4" src="https://img.shields.io/badge/PICO-4-blueviolet?style=for-the-badge"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue?style=for-the-badge"></a>
</p>

> reBot VR Teleoperation 是面向 reBot 机械臂的 VR 遥操作项目，支持 B601-RS 和 B601-DM。通过 PICO 4 手柄，操作者可以直观控制机械臂的末端位置、姿态及夹爪开合，完成单臂操作或双臂协作。项目集成 LeRobot，提供标定、运动控制和运行数据记录功能，适用于机器人操作演示与遥操作数据采集。

* * *

## 项目简介

- **VR 遥操作**：PICO 4 手柄位姿实时映射末端，支持单/双臂操作，Grip 离合、按键回位、夹爪控制。
- **分离式 IK**：q1–q3 位置 QP 跟踪 joint4 轴心，q4–q6 闭式跟随腕姿，腕部转动不干扰肩部跟踪。
- **双臂协调与安全**：RS 支持 MIT、DM 支持 MIT / POS_VEL 模式，可实现同步启动、故障同停、限位内缩、制动预留、故障保持、启动/退出归零。
- **诊断**：用于事后复现与调参分析的 CSV 记录、延迟分位数汇总与离线分析的 API。

* * *

## 演示视频

<p align="center">
  <img src="assest/dual-arm-block-grab.gif" width="600" alt="双臂遥操作演示">
</p>

## 安装

需要 Linux、Python 3.12+、LeRobot 0.6.x 和 PICO 4。RS 使用 SocketCAN，DM 使用达妙串口转 CAN。

**1. 安装 LeRobot 环境**

以下命令在 `lerobot` 文件夹下执行，路径按实际位置替换。已有环境时只需激活，无须重新创建。

```bash
cd /path/to/lerobot
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

**2. 安装遥操插件**

保持上述环境已激活，进入本项目根目录安装：

```bash
cd /path/to/lerobot-teleoperator-rebot-vr
uv pip install -e .
```

**3. 在 PICO 4 上安装发送端应用**

开启 PICO 4 的开发者模式与 USB 调试，通过 USB 连接电脑并在头显中允许调试。
在电脑上的本项目根目录执行以下命令，将 APK 安装到 **PICO 4 头显**：

```bash
adb install -r assest/reBot.apk
```

## B601-RS 使用

### 启用 CAN 接口

连接 USB 转 CAN 设备后执行，波特率为 **1 Mbps**；单臂只需第一条，已启用的接口无需重复执行。

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

### 单臂（MIT）

启用 `can0`（1 Mbps），将机械臂摆至官方零位、夹爪完全闭合并可靠支撑，然后标定。标定 ID 须与启动命令一致。

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

### 双臂（MIT）

[双臂配置](config/dual_rs_mit_split.yaml)默认左臂 `can0`、右臂 `can1`，两侧分别标定：

```bash
lerobot-calibrate --robot.type=rebot_b601_rs_follower --robot.port=can0 --robot.id=rebot_b601_rs_left
lerobot-calibrate --robot.type=rebot_b601_rs_follower --robot.port=can1 --robot.id=rebot_b601_rs_right

# 配置检查，不连接硬件
rebot-vr-teleoperate-dual --config config/dual_rs_mit_split.yaml --dry-run

# 低速启动／回零检查，会实际移动两臂
rebot-vr-teleoperate-dual --config config/dual_rs_mit_split.yaml --startup-zero-test

# 正常遥操
rebot-vr-teleoperate-dual --config config/dual_rs_mit_split.yaml
```

## B601-DM 使用

### 单臂

按实际连接修改串口，完成零位标定后启动：

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

使用 POS_VEL 时，将模式改为 `pos_vel`，配置改为 `config/pos_vel_split.yaml`。

### 双臂

在 [DM 双臂配置](config/dual_mit_split.yaml)中填写各臂的 `robot_port`，分别使用对应端口及 `rebot_left` / `rebot_right` 完成标定，然后启动：

```bash
rebot-vr-teleoperate-dual --config config/dual_mit_split.yaml
```

同样支持 `--dry-run` 配置检查和 `--startup-zero-test` 低速验证。

## 手柄操作

| 操作 | 行为 |
| --- | --- |
| Grip | 按住跟随，松开保持；启动或跟踪恢复后先完全松开再按 |
| Trigger | 独立控制对应夹爪 |
| 右 A／左 X | 对应臂返回初始姿态 |
| 右 B／左 Y | 对应臂回零并闭合夹爪 |
| 左菜单键 | 仅在录制程序中：开始／结束并保存一段示范 |
| Ctrl+C | 正常运行时回零后退出；再次按下中止回零，不等同于硬件急停 |

## 数据录制

支持 **RS 单臂 + 主摄／腕摄两路 USB 摄像头**，以 **LeRobot v3** 格式保存图像、关节状态、动作及任务指令，默认 25 FPS。


**左菜单键**开始／结束并保存，看到 `SAVED` 即完成。B 键回零不会中断录制；`Ctrl+C` 会丢弃未保存片段。

数据保存在 `recordings/RS_任务名称_日期时间/`：`data/` 为关节状态、动作和时间戳（Parquet），`videos/` 为两路视频（MP4），`meta/` 为任务、机械臂信息、录制时间及统计数据。

先在 [config/record_rs.yaml](config/record_rs.yaml) 中设置相机路径、CAN 端口和已标定的机械臂 ID，再在已激活的 LeRobot 环境、本项目根目录执行：

```bash
uv pip install -e '.[record]'
rebot-vr-record --config config/record_rs.yaml --task "将绿色方块叠放到蓝色方块上"
```
示例数据集：https://huggingface.co/datasets/Wangzhaowu/rebot_b601_rs_single_arm_block_stacking


## 运行注意事项

- 启动会自动移动机械臂，运行前核对实体零位、关节方向及运动空间。
- 双臂要求竖直、同向安装且使用独立 CAN；不提供跨臂碰撞检测。任一侧跟踪或反馈异常时，双方停止跟随。
- 退出前准备可靠支撑：失能可能导致机械臂下坠，保留力矩也不能保证断电后保持。

## 日志与文档

RS / DM 双臂日志分别位于 `logs/dual_rs/`、`logs/dual/`；单臂用 `--csv-log logs/session.csv` 开启记录。

离线测试：

```bash
uv pip install -e '.[test]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```


许可证：[Apache-2.0](LICENSE)
