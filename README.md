# Eye-Hand Calibration for the Freenove Robot Arm

## Overview

This repository contains a practical vision-to-robot workflow for a Freenove robotic arm. It combines:

- planar hand-eye calibration based on ArUco or AprilTag markers,
- a lightweight TCP client for the Freenove command protocol,
- YOLOv8-based visual detection pipelines,
- pick-and-place demos that map image coordinates into robot-space targets, and
- utility scripts for robot motion testing, camera validation, barcode decoding, and calibration verification.

The project is designed around a tabletop setup in which the camera and the robot operate over a shared planar workspace. Calibration produces a 2D homography that transforms image pixels into robot XY coordinates. That mapping is then reused by the downstream detection and manipulation scripts.

## Main Capabilities

- Calibrate a camera-to-robot planar transform using a printed marker attached to the end effector.
- Drive the Freenove arm directly over TCP without relying on the original vendor desktop application.
- Run object detection with YOLOv8 and convert detections into robot pick targets.
- Execute scripted pick-and-place sequences with configurable motion, timing, queue synchronization, and gripper behavior.
- Run a Tkinter-based monitoring dashboard for live vision and robot-state feedback.
- Test motion, gripper, camera, marker generation, and homography outputs through small standalone utilities.

## Repository Structure

The most important files are:

- [`Hand_Eye_Calibration.py`](./Hand_Eye_Calibration.py): planar hand-eye calibration and follow-mode workflow.
- [`freenove_arm.py`](./freenove_arm.py): high-level Freenove robot client built on the native TCP text protocol.
- [`client.py`](./client.py): socket transport wrapper used by the robot client.
- [`command.py`](./command.py): command token definitions for the Freenove protocol.
- [`messageQueue.py`](./messageQueue.py): queue helper used by the socket client.
- [`Integration.py`](./Integration.py): YOLOv8 pick-and-place pipeline for automated manipulation.
- [`yolov8_ui_dashboard.py`](./yolov8_ui_dashboard.py): UI dashboard for monitored detection and robot control.
- [`yolo_only.py`](./yolo_only.py): detection and barcode-decoding utility without robot motion.
- [`quick_move.py`](./quick_move.py): minimal robot motion test utility.
- [`save_parms/`](./save_parms): calibration artifacts such as `homography.npy`.
- [`test/`](./test): helper scripts for marker generation, camera tests, gripper tests, and calibration validation.

## Hardware and Environment Assumptions

This codebase assumes a setup similar to the following:

- a Freenove robot arm reachable over TCP,
- a USB camera or other OpenCV-compatible video source,
- a flat working plane shared by the camera and the robot,
- a printed ArUco or AprilTag marker used during calibration,
- a Windows development environment.

Several scripts explicitly use Windows-oriented defaults such as `cv2.CAP_DSHOW`, local drive paths, and desktop OpenCV windows. The code can be adapted for other environments, but the current repository is clearly tuned for Windows.

## Python Version

Use Python 3.10 or newer.

The repository uses modern type syntax such as `int | None`, which requires Python 3.10+.

## Dependencies

Install the core dependencies with:

```bash
pip install numpy opencv-contrib-python ultralytics pillow
```

Optional dependencies:

- `zxing` for barcode or QR decoding in scripts such as [`yolo_only.py`](./yolo_only.py) and [`yolov8_ui_dashboard.py`](./yolov8_ui_dashboard.py)
- a Java runtime if your ZXing wrapper requires it

Example:

```bash
pip install zxing
```

## Calibration Workflow

The calibration process estimates a planar homography between image pixels and robot XY coordinates.

### Step 1: Start the robot server

Bring up the Freenove robot-side TCP service before running the Python client workflows.

### Step 2: Attach the calibration marker

Attach the printed marker to the gripper or toolhead in a stable and repeatable position.

### Step 3: Run calibration

```bash
python Hand_Eye_Calibration.py --mode calibrate --host <robot-ip> --camera-id 0 --step
```

Typical useful options include:

- `--camera-id` to select the OpenCV camera source
- `--cam-width`, `--cam-height`, `--cam-fps` to request capture parameters
- `--z-height` to define the calibration plane
- `--step` to require manual confirmation before each calibration move
- `--dry-run` to test the command flow without sending robot motion
- `--verbose` to print every robot command

Successful calibration writes:

- `save_parms/homography.npy`
- `save_parms/homography_inv.npy`

These files are required by the downstream pick-and-place scripts.

## Running the Main Pipelines

### 1. Hand-eye follow mode

After calibration, the same script can map the detected marker position back into robot coordinates:

```bash
python Hand_Eye_Calibration.py --mode follow --host <robot-ip> --camera-id 0
```

### 2. Minimal robot motion test

Use this script to verify connectivity and basic motion before running more complex flows:

```bash
python quick_move.py --host <robot-ip> --interactive --verbose
```

Dry-run example:

```bash
python quick_move.py --dry-run --verbose
```

### 3. YOLOv8 pick-and-place workflow

```bash
python Integration.py --host <robot-ip> --camera-id 0 --weights path/to/best.pt --parms-dir save_parms
```

This script:

- loads a YOLOv8 model,
- detects objects in the camera frame,
- converts the selected detection center into robot coordinates through the calibrated homography,
- executes a pick-and-place sequence with configurable timing, drop positions, and gripper parameters.

### 4. Dashboard workflow

```bash
python yolov8_ui_dashboard.py --host <robot-ip> --camera-id 0 --weights path/to/best.pt --parms-dir save_parms
```

This dashboard provides:

- live video display,
- robot finite-state visibility,
- recent event logs,
- decoded text feedback,
- confirmation status for stable detections.

### 5. Detection-only workflow

```bash
python yolo_only.py --weights path/to/best.pt --source 0 --decode
```

This is useful when validating the vision pipeline independently from robot motion.

## Important Configuration Notes

### YOLO weight paths

Some scripts still contain machine-specific default weight paths such as:

- `D:\yolo\runs\detect\train\weights\best.pt`

You should override these with your own model path via `--weights`. Do not assume the hard-coded defaults are valid on another machine.

### Calibration is planar, not full 3D

The current method estimates a 2D homography for a fixed working plane. It is appropriate when objects lie on the same surface used during calibration. It is not a general 6D or full volumetric hand-eye calibration pipeline.

### Safety

Before sending real robot motion:

- verify the robot IP and TCP port,
- confirm the workspace is clear,
- test with `--dry-run` first,
- validate home, drop, and pick heights carefully,
- keep conservative speed and timing settings during initial runs.

## Test and Utility Scripts

The [`test/`](./test) directory includes helper scripts for:

- camera verification,
- marker generation and marker detection,
- homography validation,
- gripper and gripper-servo testing,
- image decoding experiments.

These utilities are intended for development and troubleshooting rather than as polished end-user entry points.

## Typical End-to-End Procedure

1. Start the robot-side Freenove TCP service.
2. Verify robot connectivity with [`quick_move.py`](./quick_move.py), preferably in `--dry-run` mode first.
3. Print and attach the calibration marker.
4. Run [`Hand_Eye_Calibration.py`](./Hand_Eye_Calibration.py) in `--mode calibrate`.
5. Confirm that `save_parms/homography.npy` has been generated.
6. Validate the camera and detection pipeline with [`yolo_only.py`](./yolo_only.py) or one of the scripts in [`test/`](./test).
7. Run [`Integration.py`](./Integration.py) or [`yolov8_ui_dashboard.py`](./yolov8_ui_dashboard.py) for full pick-and-place execution.

## Current Scope

This repository is best understood as an applied research or lab-integration codebase rather than a packaged robotics framework. Its strengths are directness, inspectability, and practical experimental control. Users adopting it for a new environment should expect to review camera IDs, robot IPs, motion limits, gripper parameters, and model paths before deployment.
