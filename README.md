# Eye-Hand Calibration for the Freenove Robot Arm

## Overview

This repository provides a practical vision-to-robot workflow for a Freenove robotic arm, with a particular emphasis on integrated visual recognition for intelligent sorting scenarios.

At its core, the project is not limited to generic object detection. It combines three tightly coupled capabilities into a single operational pipeline:

- planar hand-eye calibration based on ArUco or AprilTag markers,
- barcode or QR-code decoding for item identification, and
- planar size estimation for geometric characterization on the calibrated work surface.

These capabilities are then connected to robot execution logic so that the system can identify an item, estimate its physical footprint on the plane, determine its robot-space target position, and support downstream intelligent sorting and handling workflows in an integrated manner.

In other words, this codebase should be understood as a barcode-aware and size-aware visual recognition stack for robotic manipulation, rather than as a simple detection demo.

## Main Capabilities

- Calibrate a camera-to-robot planar transform using a printed marker attached to the end effector.
- Drive the Freenove arm directly over TCP without relying on the original vendor desktop application.
- Detect objects with YOLOv8 in a shared camera-robot workspace.
- Decode barcode or QR payloads from detected items using ZXing-based processing.
- Estimate planar bounding-box size in millimeters through the calibrated homography.
- Fuse detection, barcode decoding, and size estimation into a single visual recognition pipeline suitable for intelligent sorting integration.
- Execute scripted pick-and-place sequences with configurable motion, timing, queue synchronization, and gripper behavior.
- Run a Tkinter-based monitoring dashboard for live robot state, decoded identity, target coordinates, and size feedback.
- Test motion, gripper behavior, cameras, marker generation, homography outputs, and decoding utilities through standalone helper scripts.

## Repository Structure

The most important files are:

- [`Hand_Eye_Calibration.py`](./Hand_Eye_Calibration.py): planar hand-eye calibration and follow-mode workflow.
- [`freenove_arm.py`](./freenove_arm.py): high-level Freenove robot client built on the native TCP text protocol.
- [`client.py`](./client.py): socket transport wrapper used by the robot client.
- [`command.py`](./command.py): command token definitions for the Freenove protocol.
- [`messageQueue.py`](./messageQueue.py): queue helper used by the socket client.
- [`Integration.py`](./Integration.py): YOLOv8-based pick-and-place workflow using calibrated pixel-to-robot mapping.
- [`yolov8_test1.py`](./yolov8_test1.py): the key fused-recognition pipeline that combines detection, barcode decoding, and planar size estimation.
- [`yolov8_ui_dashboard.py`](./yolov8_ui_dashboard.py): UI dashboard for monitored robot control with live decode and size feedback.
- [`yolo_only.py`](./yolo_only.py): detection and barcode-decoding utility without robot motion.
- [`quick_move.py`](./quick_move.py): minimal robot motion test utility.
- [`save_parms/`](./save_parms): calibration artifacts such as `homography.npy`.
- [`test/`](./test): helper scripts for marker generation, camera tests, gripper tests, homography validation, and decoding experiments.

## Hardware and Environment Assumptions

This codebase assumes a setup similar to the following:

- a Freenove robot arm reachable over TCP,
- a USB camera or other OpenCV-compatible video source,
- a flat working plane shared by the camera and the robot,
- a printed ArUco or AprilTag marker used during calibration,
- barcoded or QR-coded target items located on the calibrated plane, and
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

- `zxing` for barcode or QR decoding in scripts such as [`yolo_only.py`](./yolo_only.py), [`yolov8_test1.py`](./yolov8_test1.py), and [`yolov8_ui_dashboard.py`](./yolov8_ui_dashboard.py)
- a Java runtime if your ZXing wrapper requires it

Example:

```bash
pip install zxing
```

## Calibration Workflow

The calibration process estimates a planar homography between image pixels and robot XY coordinates. That planar transform is the geometric foundation for both robot positioning and size estimation.

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

These files are required by the downstream recognition and manipulation scripts.

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

### 3. Fused visual recognition workflow

```bash
python yolov8_test1.py --host <robot-ip> --camera-id 0 --weights path/to/best.pt --parms-dir save_parms
```

This workflow is the clearest representation of the project’s central idea. It:

- detects items with YOLOv8,
- decodes barcode or QR content from candidate detections,
- estimates planar item size in millimeters from the calibrated homography,
- maps the item center into robot coordinates, and
- prepares the result for automated grasping or intelligent sorting logic.

For intelligent sorting applications, this combination is especially important: identity comes from barcode decoding, geometry comes from size estimation, and actuation comes from the calibrated robot mapping.

### 4. YOLOv8 pick-and-place workflow

```bash
python Integration.py --host <robot-ip> --camera-id 0 --weights path/to/best.pt --parms-dir save_parms
```

This script:

- loads a YOLOv8 model,
- detects objects in the camera frame,
- converts the selected detection center into robot coordinates through the calibrated homography, and
- executes a pick-and-place sequence with configurable timing, drop positions, and gripper parameters.

### 5. Dashboard workflow

```bash
python yolov8_ui_dashboard.py --host <robot-ip> --camera-id 0 --weights path/to/best.pt --parms-dir save_parms
```

This dashboard provides:

- live video display,
- robot finite-state visibility,
- recent event logs,
- decoded text feedback,
- planar size estimation feedback, and
- confirmation status for stable detections.

### 6. Detection-only workflow

```bash
python yolo_only.py --weights path/to/best.pt --source 0 --decode
```

This is useful when validating the detection and barcode-decoding stack independently from robot motion.

## Why Barcode Decoding and Size Estimation Matter

Barcode decoding and size estimation are not secondary add-ons in this repository. They are central to the intended system behavior.

- Barcode decoding provides machine-readable identity information for each detected item.
- Size estimation provides quantitative geometric information on the calibrated plane.
- Together, they turn a generic vision detector into a sorting-oriented recognition module that can distinguish not only where an item is, but also what it is and how large it is.

That combination is what makes the project suitable for integrated intelligent sorting workflows, warehouse-style handling experiments, and closed-loop robot-assisted classification tasks.

## Important Configuration Notes

### YOLO weight paths

Some scripts still contain machine-specific default weight paths such as:

- `D:\yolo\runs\detect\train\weights\best.pt`

You should override these with your own model path via `--weights`. Do not assume the hard-coded defaults are valid on another machine.

### Calibration is planar, not full 3D

The current method estimates a 2D homography for a fixed working plane. It is appropriate when objects lie on the same surface used during calibration. It is not a general 6D or full volumetric hand-eye calibration pipeline.

### Size estimation depends on the calibrated plane

Planar size estimation assumes that the target object lies on the same working plane used during calibration. If the object height or pose violates that assumption, the physical size estimate will degrade accordingly.

### Safety

Before sending real robot motion:

- verify the robot IP and TCP port,
- confirm the workspace is clear,
- test with `--dry-run` first,
- validate home, drop, and pick heights carefully, and
- keep conservative speed and timing settings during initial runs.

## Test and Utility Scripts

The [`test/`](./test) directory includes helper scripts for:

- camera verification,
- marker generation and marker detection,
- homography validation,
- gripper and gripper-servo testing,
- image decoding experiments, and
- barcode-related testing utilities.

These utilities are intended for development and troubleshooting rather than as polished end-user entry points.

## Typical End-to-End Procedure

1. Start the robot-side Freenove TCP service.
2. Verify robot connectivity with [`quick_move.py`](./quick_move.py), preferably in `--dry-run` mode first.
3. Print and attach the calibration marker.
4. Run [`Hand_Eye_Calibration.py`](./Hand_Eye_Calibration.py) in `--mode calibrate`.
5. Confirm that `save_parms/homography.npy` has been generated.
6. Validate detection, barcode decoding, and planar size estimation with [`yolov8_test1.py`](./yolov8_test1.py), [`yolo_only.py`](./yolo_only.py), or one of the scripts in [`test/`](./test).
7. Run [`Integration.py`](./Integration.py) or [`yolov8_ui_dashboard.py`](./yolov8_ui_dashboard.py) for full robot execution.
8. Integrate decoded identity, estimated size, and robot target coordinates into the downstream intelligent sorting policy of your choice.

## Current Scope

This repository is best understood as an applied research or lab-integration codebase rather than as a packaged robotics framework. Its core value lies in combining calibrated robot control with fused visual recognition, especially barcode decoding and size estimation, to support integrated intelligent sorting experiments. Users adopting it for a new environment should expect to review camera IDs, robot IPs, motion limits, gripper parameters, model paths, and barcode readability conditions before deployment.
