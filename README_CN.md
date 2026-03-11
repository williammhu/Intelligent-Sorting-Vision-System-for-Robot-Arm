# Freenove 机械臂的眼手标定

## 概述

本仓库提供了一套面向 Freenove 机械臂的、可直接落地的视觉到机器人工作流，并特别强调面向智能分拣场景的一体化视觉识别能力。

从核心定位上看，这个项目并不局限于通用目标检测。它将三项紧密耦合的能力整合到了同一条运行链路中：

- 基于 ArUco 或 AprilTag 标记的平面眼手标定，
- 用于物品身份识别的条码或二维码解码，以及
- 基于标定平面的平面尺寸估计。

这些能力随后又与机器人执行逻辑连接起来，使系统能够完成如下闭环：识别物品、估计其在工作平面上的物理尺寸、确定其对应的机器人空间目标位置，并为后续的智能分拣与搬运流程提供一体化支撑。

换句话说，这个代码库应当被理解为一套“具备条码感知与尺寸感知能力的机器人视觉识别栈”，而不只是一个简单的检测演示。

## 主要能力

- 使用安装在末端执行器上的打印标记，对相机到机器人的平面变换进行标定。
- 通过 TCP 直接驱动 Freenove 机械臂，而不依赖原厂桌面应用程序。
- 在共享的相机-机器人工作区中使用 YOLOv8 进行目标检测。
- 使用基于 ZXing 的处理流程，对检测到的目标执行条码或二维码解码。
- 通过已标定的单应矩阵，以毫米为单位估计目标在平面上的边界框尺寸。
- 将检测、条码解码与尺寸估计融合为单一视觉识别流程，用于后续智能分拣一体化集成。
- 执行可配置的抓取放置流程，包括运动参数、时序、队列同步和夹爪行为。
- 运行基于 Tkinter 的监控面板，以实时显示机器人状态、解码结果、目标坐标和尺寸反馈。
- 通过独立的辅助脚本测试机械臂运动、夹爪行为、相机、标记生成、单应矩阵输出和解码能力。

## 仓库结构

最重要的文件包括：

- [`Hand_Eye_Calibration.py`](./Hand_Eye_Calibration.py)：平面眼手标定与 follow 模式工作流。
- [`freenove_arm.py`](./freenove_arm.py)：基于 Freenove 原生 TCP 文本协议的高层机器人客户端。
- [`client.py`](./client.py)：机器人客户端使用的 socket 传输封装。
- [`command.py`](./command.py)：Freenove 协议中的命令标记定义。
- [`messageQueue.py`](./messageQueue.py)：socket 客户端使用的队列辅助模块。
- [`Integration.py`](./Integration.py)：基于标定像素到机器人映射的 YOLOv8 抓取放置流程。
- [`yolov8_test1.py`](./yolov8_test1.py)：本项目中最关键的融合识别流程，整合了检测、条码解码与平面尺寸估计。
- [`yolov8_ui_dashboard.py`](./yolov8_ui_dashboard.py)：带有实时解码与尺寸反馈的机器人监控 UI 面板。
- [`yolo_only.py`](./yolo_only.py)：不驱动机械臂运动的检测与条码解码工具。
- [`quick_move.py`](./quick_move.py)：最小化的机械臂运动测试工具。
- [`save_parms/`](./save_parms)：用于保存 `homography.npy` 等标定产物。
- [`test/`](./test)：包含标记生成、相机测试、夹爪测试、单应矩阵验证和解码实验等辅助脚本。

## 硬件与环境假设

本代码库默认假设你的实验环境与下述配置类似：

- 一台可通过 TCP 访问的 Freenove 机械臂，
- 一个 USB 相机或其他兼容 OpenCV 的视频源，
- 一个由相机与机械臂共同覆盖的平面工作区，
- 一个在标定过程中使用的打印版 ArUco 或 AprilTag 标记，
- 若干放置在已标定平面上的带条码或二维码的目标物品，以及
- 一个 Windows 开发环境。

若干脚本明确使用了面向 Windows 的默认设置，例如 `cv2.CAP_DSHOW`、本地磁盘路径以及桌面 OpenCV 窗口。虽然代码可以适配其他环境，但当前仓库显然是针对 Windows 做过调优的。

## Python 版本

请使用 Python 3.10 或更高版本。

仓库中使用了 `int | None` 这类现代类型语法，因此需要 Python 3.10+。

## 依赖项

使用以下命令安装核心依赖：

```bash
pip install numpy opencv-contrib-python ultralytics pillow
```

可选依赖：

- `zxing`，用于 [`yolo_only.py`](./yolo_only.py)、[`yolov8_test1.py`](./yolov8_test1.py) 和 [`yolov8_ui_dashboard.py`](./yolov8_ui_dashboard.py) 等脚本中的条码或二维码解码
- 如果你的 ZXing 封装依赖 Java，则还需要安装 Java 运行时

示例：

```bash
pip install zxing
```

## 标定流程

标定过程会估计图像像素坐标与机器人 XY 坐标之间的平面单应关系。这个平面变换既是机器人定位的几何基础，也是尺寸估计的几何基础。

### 第 1 步：启动机器人服务端

在运行 Python 客户端工作流之前，先启动 Freenove 机器人端的 TCP 服务。

### 第 2 步：安装标定标记

将打印好的标记稳定且可重复地固定在夹爪或工具头上。

### 第 3 步：运行标定

```bash
python Hand_Eye_Calibration.py --mode calibrate --host <robot-ip> --camera-id 0 --step
```

常用参数包括：

- `--camera-id`：选择 OpenCV 相机源
- `--cam-width`、`--cam-height`、`--cam-fps`：请求采集参数
- `--z-height`：定义标定平面的高度
- `--step`：在每一次标定运动前要求手动确认
- `--dry-run`：测试命令流程而不实际发送机器人运动
- `--verbose`：打印发送给机器人的每一条命令

标定成功后会写出：

- `save_parms/homography.npy`
- `save_parms/homography_inv.npy`

这些文件是下游识别与操作脚本运行所必需的。

## 运行主要流程

### 1. 眼手 follow 模式

完成标定后，同一个脚本可以将检测到的标记位置反向映射回机器人坐标：

```bash
python Hand_Eye_Calibration.py --mode follow --host <robot-ip> --camera-id 0
```

### 2. 最小化机械臂运动测试

在运行更复杂的流程之前，使用这个脚本验证连通性与基础运动：

```bash
python quick_move.py --host <robot-ip> --interactive --verbose
```

Dry-run 示例：

```bash
python quick_move.py --dry-run --verbose
```

### 3. 融合视觉识别流程

```bash
python yolov8_test1.py --host <robot-ip> --camera-id 0 --weights path/to/best.pt --parms-dir save_parms
```

这个流程最能体现本项目的核心思想。它会：

- 使用 YOLOv8 检测目标物品，
- 对候选检测目标执行条码或二维码解码，
- 通过已标定的单应矩阵估计目标在平面上的毫米级尺寸，
- 将目标中心映射为机器人坐标，以及
- 为自动抓取或智能分拣逻辑准备输入结果。

对于智能分拣应用而言，这种组合尤其重要：身份信息来自条码解码，几何信息来自尺寸估计，而执行能力来自已标定的机器人空间映射。

### 4. YOLOv8 抓取放置工作流

```bash
python Integration.py --host <robot-ip> --camera-id 0 --weights path/to/best.pt --parms-dir save_parms
```

该脚本会：

- 加载一个 YOLOv8 模型，
- 在相机画面中检测目标，
- 通过已标定的单应矩阵将选中的检测框中心转换为机器人坐标，以及
- 执行带有可配置时序、投放位置和夹爪参数的抓取放置流程。

### 5. Dashboard 工作流

```bash
python yolov8_ui_dashboard.py --host <robot-ip> --camera-id 0 --weights path/to/best.pt --parms-dir save_parms
```

该面板提供：

- 实时视频显示，
- 机器人有限状态机可视化，
- 最近事件日志，
- 解码文本反馈，
- 平面尺寸估计反馈，以及
- 稳定检测确认状态。

### 6. 仅检测工作流

```bash
python yolo_only.py --weights path/to/best.pt --source 0 --decode
```

当你希望在不驱动机器人运动的前提下验证检测与条码解码能力时，这个脚本非常有用。

## 为什么条码解码和尺寸估计很重要

在这个仓库中，条码解码和尺寸估计并不是次要附加功能，而是预期系统行为的核心组成部分。

- 条码解码为每一个检测到的目标提供机器可读的身份信息。
- 尺寸估计为目标在标定平面上的几何特征提供定量信息。
- 两者结合后，视觉系统就不再只是“看见一个目标”，而是能够同时回答目标在哪里、它是什么、以及它有多大。

正是这种组合，使得该项目适合用于一体化智能分拣工作流、类仓储搬运实验以及机器人辅助分类闭环任务。

## 重要配置说明

### YOLO 权重路径

部分脚本中仍然保留了与特定机器相关的默认权重路径，例如：

- `D:\yolo\runs\detect\train\weights\best.pt`

你应该通过 `--weights` 参数覆盖为你自己的模型路径。不要假设这些硬编码默认值在另一台机器上依然有效。

### 标定是平面的，而不是完整 3D

当前方法估计的是固定工作平面上的二维单应矩阵。当目标物体位于与标定时相同的平面上时，这种方法是合适的。它并不是一个通用的六自由度或完整体积式眼手标定流程。

### 尺寸估计依赖于标定平面

平面尺寸估计默认假设目标物体位于与标定时相同的工作平面上。如果目标的高度或姿态破坏了这个假设，那么物理尺寸估计的精度也会相应下降。

### 安全性

在发送真实的机器人运动之前：

- 确认机器人 IP 与 TCP 端口正确，
- 确认工作空间已清空，
- 先使用 `--dry-run` 做验证，
- 仔细检查 home、drop 和 pick 的高度参数，以及
- 在初次运行时保持保守的速度与时序设置。

## 测试与工具脚本

[`test/`](./test) 目录中包含的辅助脚本可用于：

- 相机验证，
- 标记生成与标记检测，
- 单应矩阵验证，
- 夹爪与夹爪舵机测试，
- 图像解码实验，以及
- 与条码相关的测试工具。

这些工具主要用于开发与排障，而不是面向最终用户的正式入口。

## 典型端到端流程

1. 启动机器人端的 Freenove TCP 服务。
2. 使用 [`quick_move.py`](./quick_move.py) 验证机器人连接，最好先在 `--dry-run` 模式下进行。
3. 打印并安装标定标记。
4. 以 `--mode calibrate` 运行 [`Hand_Eye_Calibration.py`](./Hand_Eye_Calibration.py)。
5. 确认 `save_parms/homography.npy` 已成功生成。
6. 使用 [`yolov8_test1.py`](./yolov8_test1.py)、[`yolo_only.py`](./yolo_only.py) 或 [`test/`](./test) 中的脚本验证检测、条码解码与平面尺寸估计能力。
7. 运行 [`Integration.py`](./Integration.py) 或 [`yolov8_ui_dashboard.py`](./yolov8_ui_dashboard.py) 来执行完整的机器人流程。
8. 将解码得到的身份信息、估计得到的尺寸信息以及机器人目标坐标整合进你所需的下游智能分拣策略。

## 当前范围

与其说这是一个打包完善的机器人框架，不如说这个仓库更适合被理解为一个面向应用研究或实验室集成的代码库。它的核心价值在于，将已标定的机器人控制与融合视觉识别结合起来，尤其是条码解码与尺寸估计，从而支撑一体化智能分拣实验。若要将其迁移到新的环境中，用户应当预期先检查并调整相机 ID、机器人 IP、运动范围、夹爪参数、模型路径以及条码可读性条件，然后再投入使用。
