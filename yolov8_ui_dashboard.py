from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

ULTRALYTICS_CONFIG_DIR = os.path.join(os.getcwd(), ".ultralytics")
os.makedirs(ULTRALYTICS_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", ULTRALYTICS_CONFIG_DIR)

from ultralytics import YOLO

from Hand_Eye_Calibration import HomographyResult
from freenove_arm import FreenoveArmClient
from yolov8_test1 import decode_with_zxing, estimate_bbox_size_cm, pixel_to_robot


GRIPPER_SERVO_INDEX = 0
GRIPPER_OPEN_ANGLE = 70
GRIPPER_CLOSE_ANGLE = 10
VIDEO_DISPLAY_WIDTH = 1152
VIDEO_DISPLAY_HEIGHT = 648


@dataclass
class DashboardState:
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    latest_frame: Optional[np.ndarray] = None
    robot_state: str = "IDLE"
    robot_status: str = "Waiting for target"
    robot_status_kind: str = "idle"
    vision_status: str = "Starting vision pipeline"
    vision_status_kind: str = "busy"
    decoded_text: str = "N/A"
    size_text: str = "N/A"
    class_text: str = "N/A"
    target_text: str = "N/A"
    camera_text: str = "Initializing camera"
    confirm_text: str = "Waiting for stable detections"
    recent_events: Deque[str] = field(default_factory=lambda: deque(maxlen=8))

    def push_event(self, text: str) -> None:
        with self.lock:
            self.recent_events.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")

    def set_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            self.latest_frame = frame.copy()

    def set_robot_state(self, state: str, status: str, kind: str = "busy") -> None:
        with self.lock:
            self.robot_state = state
            self.robot_status = status
            self.robot_status_kind = kind

    def set_vision_status(self, status: str, kind: str = "idle") -> None:
        with self.lock:
            self.vision_status = status
            self.vision_status_kind = kind

    def set_detection_info(
        self,
        *,
        decoded_text: Optional[str] = None,
        size_text: Optional[str] = None,
        class_text: Optional[str] = None,
        target_text: Optional[str] = None,
        confirm_text: Optional[str] = None,
        camera_text: Optional[str] = None,
    ) -> None:
        with self.lock:
            if decoded_text is not None:
                self.decoded_text = decoded_text
            if size_text is not None:
                self.size_text = size_text
            if class_text is not None:
                self.class_text = class_text
            if target_text is not None:
                self.target_text = target_text
            if confirm_text is not None:
                self.confirm_text = confirm_text
            if camera_text is not None:
                self.camera_text = camera_text

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "latest_frame": None if self.latest_frame is None else self.latest_frame.copy(),
                "robot_state": self.robot_state,
                "robot_status": self.robot_status,
                "robot_status_kind": self.robot_status_kind,
                "vision_status": self.vision_status,
                "vision_status_kind": self.vision_status_kind,
                "decoded_text": self.decoded_text,
                "size_text": self.size_text,
                "class_text": self.class_text,
                "target_text": self.target_text,
                "camera_text": self.camera_text,
                "confirm_text": self.confirm_text,
                "recent_events": list(self.recent_events),
            }


def draw_status_chip(image: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    cv2.rectangle(image, (18, 18), (560, 58), (255, 255, 255), -1)
    cv2.rectangle(image, (18, 18), (560, 58), color, 2)
    cv2.putText(image, text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)


def update_robot_card(state: DashboardState, fsm_state: str, message: str, kind: str = "busy") -> None:
    state.set_robot_state(fsm_state, message, kind)
    state.push_event(f"FSM -> {fsm_state}: {message}")


def robot_worker_ui(
    args: argparse.Namespace,
    state: DashboardState,
    target_queue: "queue.Queue[Optional[Tuple[float, float, float, int]]]",
    busy_flag: threading.Event,
    done_flag: threading.Event,
    stop_event: threading.Event,
) -> None:
    try:
        with FreenoveArmClient(
            host=args.host,
            port=args.port,
            dry_run=args.dry_run,
            auto_enable=not args.skip_enable,
            verbose=args.verbose,
        ) as arm:
            queue_sync_enabled = False
            if args.queue_sync and not args.dry_run:
                try:
                    arm.set_action_feedback(True)
                    queue_sync_enabled = True
                    state.push_event("Queue feedback synchronization enabled")
                except Exception as exc:
                    state.push_event(f"Queue feedback unavailable: {exc}")

            def move_phase_done(start_t: float) -> bool:
                nonlocal queue_sync_enabled
                if (time.time() - start_t) < args.move_wait:
                    return False
                if not queue_sync_enabled:
                    return True
                try:
                    return arm.wait_action_queue_empty(
                        timeout=args.sync_timeout,
                        min_empty_time=args.queue_empty_stable,
                        poll_interval=tick,
                    )
                except Exception as exc:
                    queue_sync_enabled = False
                    state.push_event(f"Queue sync fallback: {exc}")
                    try:
                        arm.set_action_feedback(False)
                    except Exception:
                        pass
                    return True

            if args.home_first:
                update_robot_card(state, "HOME_INIT", "Returning to sensor point", "busy")
                arm.return_to_sensor_point(1)
                arm.wait(0.5)
            if args.ground_clearance is not None:
                update_robot_card(state, "CLEARANCE_SETUP", "Applying ground clearance", "busy")
                arm.set_ground_clearance(args.ground_clearance)
                arm.wait(0.1)

            home_point = (args.home_x, args.home_y, args.home_z)
            drop_point = (args.drop_x, args.drop_y, args.drop_z)
            drop_release_point = (args.drop_x, args.drop_y, args.drop_release_z)
            drop_post_open_lift_point = (args.drop_x, args.drop_y, args.post_open_lift_z)
            tick = 0.02
            command_sent = False
            start_time = 0.0
            current_target: Optional[Tuple[float, float, float, int]] = None
            fsm_state = "IDLE"
            state.set_robot_state("IDLE", "Waiting for target", "idle")

            while not stop_event.is_set():
                now = time.time()

                if fsm_state == "IDLE":
                    try:
                        item = target_queue.get(timeout=tick)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    current_target = item
                    busy_flag.set()
                    done_flag.clear()
                    fsm_state = "PRE_OPEN_BEFORE_TARGET"
                    command_sent = False
                    update_robot_card(state, fsm_state, "Preparing gripper for pickup")
                    continue

                if current_target is None:
                    fsm_state = "IDLE"
                    busy_flag.clear()
                    continue

                x, y, z, _cls_id = current_target

                if fsm_state == "PRE_OPEN_BEFORE_TARGET":
                    arm.sweep_servo(
                        GRIPPER_SERVO_INDEX,
                        GRIPPER_CLOSE_ANGLE,
                        GRIPPER_OPEN_ANGLE,
                        step_deg=args.gripper_step_deg,
                        step_delay_s=args.gripper_step_delay,
                    )
                    fsm_state = "MOVE_TO_TARGET"
                    update_robot_card(state, fsm_state, f"Moving to target X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
                    continue

                if fsm_state == "MOVE_TO_TARGET":
                    if not command_sent:
                        arm.move_to(x, y, z, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        fsm_state = "WAIT_TARGET"
                        command_sent = False
                        start_time = now
                        update_robot_card(state, fsm_state, "Settling at target")
                    arm.wait(tick)
                    continue

                if fsm_state == "WAIT_TARGET":
                    if now - start_time >= args.wait_at_target:
                        fsm_state = "GRIPPER_OPEN"
                        command_sent = False
                        update_robot_card(state, fsm_state, "Opening gripper")
                    arm.wait(tick)
                    continue

                if fsm_state == "GRIPPER_OPEN":
                    if not command_sent:
                        arm.sweep_servo(
                            GRIPPER_SERVO_INDEX,
                            GRIPPER_OPEN_ANGLE,
                            GRIPPER_OPEN_ANGLE,
                            step_deg=args.gripper_step_deg,
                            step_delay_s=args.gripper_step_delay,
                        )
                        start_time = time.time()
                        command_sent = True
                    if now - start_time >= args.gripper_wait:
                        fsm_state = "WAIT_OPEN"
                        command_sent = False
                        start_time = now
                        update_robot_card(state, fsm_state, "Holding open position")
                    arm.wait(tick)
                    continue

                if fsm_state == "WAIT_OPEN":
                    if now - start_time >= args.wait_after_open:
                        fsm_state = "GRIPPER_CLOSE"
                        command_sent = False
                        update_robot_card(state, fsm_state, "Closing gripper")
                    arm.wait(tick)
                    continue

                if fsm_state == "GRIPPER_CLOSE":
                    if not command_sent:
                        arm.sweep_servo(
                            GRIPPER_SERVO_INDEX,
                            GRIPPER_OPEN_ANGLE,
                            GRIPPER_CLOSE_ANGLE,
                            step_deg=args.gripper_step_deg,
                            step_delay_s=args.gripper_step_delay,
                        )
                        start_time = time.time()
                        command_sent = True
                    if now - start_time >= args.gripper_wait:
                        fsm_state = "WAIT_CLOSE"
                        command_sent = False
                        start_time = now
                        update_robot_card(state, fsm_state, "Confirming grip")
                    arm.wait(tick)
                    continue

                if fsm_state == "WAIT_CLOSE":
                    if now - start_time >= args.wait_after_close:
                        fsm_state = "LIFT_AFTER_GRIP"
                        command_sent = False
                        update_robot_card(state, fsm_state, "Lifting object from table")
                    arm.wait(tick)
                    continue

                if fsm_state == "LIFT_AFTER_GRIP":
                    if not command_sent:
                        arm.move_to(x, y, args.drop_z, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        fsm_state = "MOVE_TO_DROP"
                        command_sent = False
                        update_robot_card(state, fsm_state, "Moving above drop zone")
                    arm.wait(tick)
                    continue

                if fsm_state == "MOVE_TO_DROP":
                    if not command_sent:
                        arm.move_to(*drop_point, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        fsm_state = "MOVE_TO_DROP_RELEASE_Z"
                        command_sent = False
                        update_robot_card(state, fsm_state, "Descending to release height")
                    arm.wait(tick)
                    continue

                if fsm_state == "MOVE_TO_DROP_RELEASE_Z":
                    if not command_sent:
                        arm.move_to(*drop_release_point, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        fsm_state = "WAIT_DROP_RELEASE"
                        command_sent = False
                        start_time = now
                        update_robot_card(state, fsm_state, "Holding release position")
                    arm.wait(tick)
                    continue

                if fsm_state == "WAIT_DROP_RELEASE":
                    if now - start_time >= args.wait_at_drop:
                        fsm_state = "RELEASE_OPEN"
                        command_sent = False
                        update_robot_card(state, fsm_state, "Opening gripper to release")
                    arm.wait(tick)
                    continue

                if fsm_state == "RELEASE_OPEN":
                    if not command_sent:
                        arm.sweep_servo(
                            GRIPPER_SERVO_INDEX,
                            GRIPPER_CLOSE_ANGLE,
                            GRIPPER_OPEN_ANGLE,
                            step_deg=args.gripper_step_deg,
                            step_delay_s=args.gripper_step_delay,
                        )
                        start_time = time.time()
                        command_sent = True
                    if now - start_time >= args.gripper_wait:
                        fsm_state = "POST_RELEASE_LIFT"
                        command_sent = False
                        update_robot_card(state, fsm_state, "Lifting after release")
                    arm.wait(tick)
                    continue

                if fsm_state == "POST_RELEASE_LIFT":
                    if not command_sent:
                        arm.move_to(*drop_post_open_lift_point, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        fsm_state = "WAIT_POST_RELEASE_LIFT"
                        command_sent = False
                        start_time = now
                        update_robot_card(state, fsm_state, "Stabilizing before close")
                    arm.wait(tick)
                    continue

                if fsm_state == "WAIT_POST_RELEASE_LIFT":
                    if now - start_time >= args.post_open_wait:
                        fsm_state = "RELEASE_CLOSE"
                        command_sent = False
                        update_robot_card(state, fsm_state, "Closing gripper after release")
                    arm.wait(tick)
                    continue

                if fsm_state == "RELEASE_CLOSE":
                    if not command_sent:
                        arm.sweep_servo(
                            GRIPPER_SERVO_INDEX,
                            GRIPPER_OPEN_ANGLE,
                            GRIPPER_CLOSE_ANGLE,
                            step_deg=args.gripper_step_deg,
                            step_delay_s=args.gripper_step_delay,
                        )
                        start_time = time.time()
                        command_sent = True
                    if now - start_time >= args.gripper_wait:
                        fsm_state = "MOVE_HOME"
                        command_sent = False
                        update_robot_card(state, fsm_state, "Returning to home position")
                    arm.wait(tick)
                    continue

                if fsm_state == "MOVE_HOME":
                    if not command_sent:
                        arm.move_to(*home_point, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        fsm_state = "WAIT_HOME"
                        command_sent = False
                        start_time = now
                        update_robot_card(state, fsm_state, "Final home settle")
                    arm.wait(tick)
                    continue

                if fsm_state == "WAIT_HOME":
                    if now - start_time >= args.wait_after_close:
                        fsm_state = "IDLE"
                        current_target = None
                        busy_flag.clear()
                        done_flag.set()
                        state.set_robot_state("IDLE", "Waiting for next target", "ok")
                        state.push_event("Cycle completed")
                    arm.wait(tick)

            if queue_sync_enabled:
                try:
                    arm.set_action_feedback(False)
                except Exception:
                    pass
    except Exception as exc:
        state.set_robot_state("ERROR", f"Robot worker error: {exc}", "error")
        state.push_event(f"Robot worker error: {exc}")
        busy_flag.clear()
        done_flag.set()


def vision_worker(
    args: argparse.Namespace,
    state: DashboardState,
    target_queue: "queue.Queue[Optional[Tuple[float, float, float, int]]]",
    busy_flag: threading.Event,
    done_flag: threading.Event,
    stop_event: threading.Event,
) -> None:
    try:
        homography = HomographyResult.load(args.parms_dir).homography
    except FileNotFoundError:
        state.set_vision_status("Homography files not found", "error")
        state.push_event("Homography files not found in save_parms")
        return

    try:
        model = YOLO(args.weights)
    except Exception as exc:
        state.set_vision_status(f"Model load failed: {exc}", "error")
        state.push_event(f"Model load failed: {exc}")
        return

    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        state.set_vision_status("Could not open camera", "error")
        state.push_event(f"Camera open failed for index {args.camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    state.set_detection_info(camera_text=f"{actual_w} x {actual_h} @ {actual_fps:.1f} FPS")
    state.push_event(f"Camera ready: {actual_w}x{actual_h}@{actual_fps:.1f}")

    allowed = set(args.classes) if args.classes else None
    last_sent_time: Optional[float] = None
    last_sent_xy: Optional[Tuple[float, float]] = None
    last_sent_code: Optional[str] = None
    confirm_frames = 0
    confirm_cls_id: Optional[int] = None
    confirm_xy: Optional[Tuple[float, float]] = None
    confirm_code: Optional[str] = None
    confirm_size_mm: Optional[Tuple[float, float]] = None
    confirm_conf_threshold = max(0.7, float(args.min_pick_conf))

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                state.set_vision_status("Frame read failed", "error")
                state.push_event("Camera frame read failed")
                time.sleep(0.2)
                continue

            if done_flag.is_set():
                done_flag.clear()
                busy_flag.clear()

            if busy_flag.is_set():
                vis = frame.copy()
                draw_status_chip(vis, "Robot is executing the current cycle", (235, 99, 37))
                state.set_vision_status("Robot is busy", "busy")
                state.set_detection_info(confirm_text="Vision paused while robot is active")
                state.set_frame(vis)
                continue

            result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            vis = result.plot()

            status = "No target"
            status_kind = "ok"
            status_color = (15, 157, 88)
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                h, w = frame.shape[:2]
                decoded_candidates = []

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    cls_name = result.names[cls_id]
                    if allowed and cls_name not in allowed:
                        continue

                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (44, 166, 255), 2)

                    decoded_text: Optional[str] = None
                    size_w_mm: Optional[float] = None
                    size_h_mm: Optional[float] = None

                    x1c = max(0, min(x1, w - 1))
                    x2c = max(0, min(x2, w - 1))
                    y1c = max(0, min(y1, h - 1))
                    y2c = max(0, min(y2, h - 1))
                    if x2c > x1c and y2c > y1c:
                        roi = frame[y1c:y2c, x1c:x2c]
                        roi_big = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                        gray = cv2.cvtColor(roi_big, cv2.COLOR_BGR2GRAY)
                        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        decoded_text = decode_with_zxing(bw)
                        if decoded_text:
                            size_w_mm, size_h_mm = estimate_bbox_size_cm(xyxy, homography)

                    label = f"{cls_name} {conf:.2f}"
                    if decoded_text and size_w_mm is not None and size_h_mm is not None:
                        label = f"{label} | {decoded_text[:24]} | {size_w_mm:.1f}x{size_h_mm:.1f}mm"
                    cv2.putText(
                        vis,
                        label,
                        (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (44, 166, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    if decoded_text and size_w_mm is not None and size_h_mm is not None:
                        cx = float((xyxy[0] + xyxy[2]) / 2.0)
                        cy = float((xyxy[1] + xyxy[3]) / 2.0)
                        rx, ry = pixel_to_robot((cx, cy), homography)
                        decoded_candidates.append((conf, cls_id, cls_name, rx, ry, decoded_text, size_w_mm, size_h_mm))

                if decoded_candidates:
                    decoded_candidates.sort(key=lambda item: item[0], reverse=True)
                    conf, cls_id, cls_name, rx, ry, decoded_text, size_w_mm, size_h_mm = decoded_candidates[0]
                    state.set_detection_info(
                        decoded_text=decoded_text,
                        size_text=f"{size_w_mm:.1f} x {size_h_mm:.1f} mm",
                        class_text=f"{cls_name}  ({conf:.2f})",
                        target_text=f"X={rx:.1f} mm, Y={ry:.1f} mm, Z={args.pick_z:.1f} mm",
                    )

                    if conf >= confirm_conf_threshold:
                        now = time.time()
                        if (
                            confirm_frames > 0
                            and confirm_xy is not None
                            and confirm_cls_id is not None
                            and confirm_code is not None
                            and confirm_size_mm is not None
                        ):
                            dx = rx - confirm_xy[0]
                            dy = ry - confirm_xy[1]
                            dist = (dx * dx + dy * dy) ** 0.5
                            size_delta = max(
                                abs(size_w_mm - confirm_size_mm[0]),
                                abs(size_h_mm - confirm_size_mm[1]),
                            )
                            same_candidate = (
                                cls_id == confirm_cls_id
                                and decoded_text == confirm_code
                                and dist <= args.same_target_mm
                                and size_delta <= args.size_stable_mm
                            )
                        else:
                            same_candidate = False

                        confirm_frames = confirm_frames + 1 if same_candidate else 1
                        confirm_cls_id = cls_id
                        confirm_xy = (rx, ry)
                        confirm_code = decoded_text
                        confirm_size_mm = (size_w_mm, size_h_mm)

                        if confirm_frames < args.confirm_frames:
                            status = f"Confirming target {confirm_frames}/{args.confirm_frames}"
                            status_kind = "warn"
                            status_color = (33, 179, 217)
                            state.set_detection_info(confirm_text=f"Stability check: {confirm_frames}/{args.confirm_frames}")
                        else:
                            in_cooldown = last_sent_time is not None and (now - last_sent_time) < args.cooldown
                            same_recent = False
                            if last_sent_time is not None and last_sent_xy is not None:
                                dt = now - last_sent_time
                                dx = rx - last_sent_xy[0]
                                dy = ry - last_sent_xy[1]
                                dist = (dx * dx + dy * dy) ** 0.5
                                same_recent = (
                                    dt <= args.same_target_sec
                                    and dist <= args.same_target_mm
                                    and last_sent_code == decoded_text
                                )

                            if in_cooldown:
                                remaining_cd = args.cooldown - (now - last_sent_time)
                                status = f"Cooldown {max(0.0, remaining_cd):.1f}s"
                                status_kind = "warn"
                                status_color = (33, 179, 217)
                                state.set_detection_info(confirm_text="Waiting for cooldown to finish")
                            elif same_recent:
                                status = "Duplicate target skipped"
                                status_kind = "warn"
                                status_color = (33, 179, 217)
                                state.set_detection_info(confirm_text="Duplicate target filtered")
                            else:
                                try:
                                    target_queue.put_nowait((rx, ry, args.pick_z, cls_id))
                                    busy_flag.set()
                                    last_sent_time = now
                                    last_sent_xy = (rx, ry)
                                    last_sent_code = decoded_text
                                    status = "Target sent to robot"
                                    status_kind = "ok"
                                    status_color = (15, 157, 88)
                                    state.push_event(f"Target queued: {decoded_text} @ X={rx:.1f}, Y={ry:.1f}")
                                    state.set_detection_info(confirm_text="Target accepted")
                                except queue.Full:
                                    status = "Robot queue is full"
                                    status_kind = "warn"
                                    status_color = (33, 179, 217)
                                    state.set_detection_info(confirm_text="Queue is full, waiting")

                            confirm_frames = 0
                            confirm_cls_id = None
                            confirm_xy = None
                            confirm_code = None
                            confirm_size_mm = None
                    else:
                        confirm_frames = 0
                        confirm_cls_id = None
                        confirm_xy = None
                        confirm_code = None
                        confirm_size_mm = None
                        status = f"Low confidence {conf:.2f}"
                        status_kind = "warn"
                        status_color = (33, 179, 217)
                        state.set_detection_info(confirm_text="Confidence below pick threshold")
                else:
                    confirm_frames = 0
                    confirm_cls_id = None
                    confirm_xy = None
                    confirm_code = None
                    confirm_size_mm = None
                    status = "Barcode decode failed"
                    status_kind = "warn"
                    status_color = (33, 179, 217)
                    state.set_detection_info(confirm_text="Detection exists, but decode failed")
            else:
                confirm_frames = 0
                confirm_cls_id = None
                confirm_xy = None
                confirm_code = None
                confirm_size_mm = None
                state.set_detection_info(
                    decoded_text="N/A",
                    size_text="N/A",
                    class_text="N/A",
                    target_text="N/A",
                    confirm_text="No object candidate",
                )

            draw_status_chip(vis, status, status_color)
            state.set_vision_status(status, status_kind)
            state.set_frame(vis)
    finally:
        cap.release()


class YoloDashboardApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state = DashboardState()
        self.stop_event = threading.Event()
        self.busy_flag = threading.Event()
        self.done_flag = threading.Event()
        self.target_queue: "queue.Queue[Optional[Tuple[float, float, float, int]]]" = queue.Queue(maxsize=1)
        self.video_image: Optional[ImageTk.PhotoImage] = None

        self.root = tk.Tk()
        self.root.title("YOLO Pick Dashboard")
        self.root.geometry("1600x900")
        self.root.minsize(1480, 840)
        self.root.configure(bg="#EEF4F7")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()
        self._start_workers()
        self.root.after(50, self.refresh_ui)

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg="#10344A", height=78)
        header.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text="Vision-Guided Pick Dashboard", bg="#10344A", fg="#F8FAFC", font=("Segoe UI Semibold", 21)).grid(
            row=0, column=0, sticky="w", padx=18, pady=(12, 2)
        )
        tk.Label(
            header,
            text="Live camera stream, robot FSM, decode result, and size estimation",
            bg="#10344A",
            fg="#B8D5E3",
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 10))

        body = tk.Frame(self.root, bg="#EEF4F7")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0, minsize=VIDEO_DISPLAY_WIDTH)
        body.grid_columnconfigure(1, weight=1, minsize=300)

        left = tk.Frame(body, bg="#EEF4F7")
        left.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        left.grid_rowconfigure(0, weight=0, minsize=VIDEO_DISPLAY_HEIGHT + 28)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        right = tk.Frame(body, bg="#EEF4F7")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        self.video_card = tk.Frame(left, bg="#0E2230", bd=0, highlightthickness=0)
        self.video_card.grid(row=0, column=0, sticky="nw", pady=(0, 10))
        self.video_card.grid_rowconfigure(1, weight=1)
        self.video_card.grid_columnconfigure(0, weight=1)
        tk.Label(
            self.video_card,
            text="LIVE STREAM",
            bg="#0E2230",
            fg="#C7D8E2",
            font=("Segoe UI Semibold", 10),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=6, pady=(3, 4))
        self.video_label = tk.Label(self.video_card, bg="#0E2230", bd=0, width=VIDEO_DISPLAY_WIDTH, height=VIDEO_DISPLAY_HEIGHT)
        self.video_label.grid(row=1, column=0, sticky="nw")

        bottom_grid = tk.Frame(left, bg="#EEF4F7")
        bottom_grid.grid(row=1, column=0, sticky="nsew")
        bottom_grid.grid_columnconfigure(0, weight=1)
        bottom_grid.grid_columnconfigure(1, weight=1)

        self.events_card = self._create_card(bottom_grid)
        self.events_card.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.events_card.grid_rowconfigure(2, weight=1)
        self._card_title(self.events_card, "Recent Events", "Latest system activity")
        self.events_text = tk.Text(
            self.events_card,
            height=7,
            bg="#F8FBFD",
            fg="#2A3D4F",
            bd=0,
            highlightthickness=0,
            wrap="word",
            font=("Consolas", 9),
            padx=12,
            pady=10,
        )
        self.events_text.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 14))
        self.events_text.configure(state="disabled")

        self.camera_card = self._create_card(bottom_grid)
        self.camera_card.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self._card_title(self.camera_card, "Camera & Vision", "Runtime stream details")
        self.camera_value = self._metric(self.camera_card, "Camera Mode", "Initializing camera")
        self.vision_value = self._metric(self.camera_card, "Vision Status", "Starting vision pipeline")
        self.confirm_value = self._metric(self.camera_card, "Decision Gate", "Waiting for stable detections")

        self.fsm_card = self._create_card(right)
        self.fsm_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._card_title(self.fsm_card, "Robot FSM", "Current finite-state-machine position")
        self.state_badge = tk.Label(
            self.fsm_card,
            text="IDLE",
            bg="#DDEFF8",
            fg="#0C5672",
            font=("Segoe UI Semibold", 18),
            padx=18,
            pady=10,
        )
        self.state_badge.grid(row=2, column=0, sticky="w", padx=18, pady=(4, 8))
        self.robot_status_label = tk.Label(
            self.fsm_card,
            text="Waiting for target",
            bg="#FFFFFF",
            fg="#466071",
            justify="left",
            wraplength=280,
            font=("Segoe UI", 10),
        )
        self.robot_status_label.grid(row=3, column=0, sticky="w", padx=18, pady=(0, 14))

        self.decode_card = self._create_card(right)
        self.decode_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self._card_title(self.decode_card, "Decode Result", "ZXing-decoded content")
        self.decode_value = tk.Label(
            self.decode_card,
            text="N/A",
            bg="#FFFFFF",
            fg="#163246",
            justify="left",
            wraplength=280,
            font=("Consolas", 11),
        )
        self.decode_value.grid(row=2, column=0, sticky="w", padx=18, pady=(4, 14))

        self.metrics_card = self._create_card(right)
        self.metrics_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._card_title(self.metrics_card, "Measurement", "Class, confidence, size, and target pose")
        self.class_value = self._metric(self.metrics_card, "Object", "N/A")
        self.size_value = self._metric(self.metrics_card, "Estimated Size", "N/A")
        self.target_value = self._metric(self.metrics_card, "Robot Target", "N/A")

    def _create_card(self, parent: tk.Widget) -> tk.Frame:
        frame = tk.Frame(parent, bg="#FFFFFF", bd=0, highlightthickness=1, highlightbackground="#D7E3EA")
        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _card_title(self, parent: tk.Frame, title: str, subtitle: str) -> None:
        tk.Label(parent, text=title, bg="#FFFFFF", fg="#163246", font=("Segoe UI Semibold", 16)).grid(
            row=0, column=0, sticky="w", padx=18, pady=(16, 2)
        )
        tk.Label(parent, text=subtitle, bg="#FFFFFF", fg="#738596", font=("Segoe UI", 10)).grid(
            row=1, column=0, sticky="w", padx=18, pady=(0, 12)
        )

    def _metric(self, parent: tk.Frame, name: str, default_value: str) -> tk.Label:
        row = parent.grid_size()[1]
        block = tk.Frame(parent, bg="#F7FAFC", highlightthickness=1, highlightbackground="#E0EAF0")
        block.grid(row=row, column=0, sticky="ew", padx=18, pady=(0, 12))
        block.grid_columnconfigure(0, weight=1)
        tk.Label(block, text=name.upper(), bg="#F7FAFC", fg="#7A8B9C", font=("Segoe UI Semibold", 9)).grid(
            row=0, column=0, sticky="w", padx=14, pady=(10, 4)
        )
        value_label = tk.Label(
            block,
            text=default_value,
            bg="#F7FAFC",
            fg="#163246",
            justify="left",
            wraplength=280,
            font=("Segoe UI", 10),
        )
        value_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))
        return value_label

    def _start_workers(self) -> None:
        self.robot_thread = threading.Thread(
            target=robot_worker_ui,
            args=(self.args, self.state, self.target_queue, self.busy_flag, self.done_flag, self.stop_event),
            daemon=True,
        )
        self.vision_thread = threading.Thread(
            target=vision_worker,
            args=(self.args, self.state, self.target_queue, self.busy_flag, self.done_flag, self.stop_event),
            daemon=True,
        )
        self.robot_thread.start()
        self.vision_thread.start()

    def refresh_ui(self) -> None:
        snap = self.state.snapshot()
        self._update_video(snap["latest_frame"])
        self._set_text(self.decode_value, snap["decoded_text"])
        self._set_text(self.class_value, snap["class_text"])
        self._set_text(self.size_value, snap["size_text"])
        self._set_text(self.target_value, snap["target_text"])
        self._set_text(self.camera_value, snap["camera_text"])
        self._set_text(self.vision_value, snap["vision_status"])
        self._set_text(self.confirm_value, snap["confirm_text"])
        self._set_text(self.robot_status_label, snap["robot_status"])

        badge_bg = "#DDEFF8"
        badge_fg = "#0C5672"
        if snap["robot_status_kind"] == "ok":
            badge_bg = "#DCFCE7"
            badge_fg = "#166534"
        elif snap["robot_status_kind"] == "error":
            badge_bg = "#FEE2E2"
            badge_fg = "#B91C1C"
        elif snap["robot_status_kind"] == "busy":
            badge_bg = "#DBEAFE"
            badge_fg = "#1D4ED8"
        self.state_badge.configure(text=snap["robot_state"], bg=badge_bg, fg=badge_fg)

        self.events_text.configure(state="normal")
        self.events_text.delete("1.0", tk.END)
        for event in snap["recent_events"]:
            self.events_text.insert(tk.END, event + "\n")
        self.events_text.configure(state="disabled")

        if not self.stop_event.is_set():
            self.root.after(50, self.refresh_ui)

    def _update_video(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        width = VIDEO_DISPLAY_WIDTH
        height = VIDEO_DISPLAY_HEIGHT
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((width, height), Image.Resampling.BILINEAR)
        self.video_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.video_image)

    @staticmethod
    def _set_text(widget: tk.Widget, value: str) -> None:
        widget.configure(text=value)

    def on_close(self) -> None:
        self.stop_event.set()
        try:
            self.target_queue.put_nowait(None)
        except queue.Full:
            pass
        self.root.after(100, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()
        self.stop_event.set()
        try:
            self.target_queue.put_nowait(None)
        except queue.Full:
            pass
        self.robot_thread.join(timeout=2.0)
        self.vision_thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 dashboard with live stream, FSM status, and decode results")
    parser.add_argument("--weights", type=str, default=r"D:\yolo\runs\detect\train\weights\best.pt", help="YOLO model weights")
    parser.add_argument("--camera-id", type=int, default=2, help="OpenCV camera index")
    parser.add_argument("--cam-width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--cam-height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--cam-fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument("--conf", type=float, default=0.7, help="YOLO detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--min-pick-conf", type=float, default=0.7, help="Minimum confidence to send a pick command")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Required stable decoded frames before sending a target")
    parser.add_argument("--size-stable-mm", type=float, default=8.0, help="Max size delta between frames to count as stable")
    parser.add_argument("--classes", nargs="*", help="Optional class name filter")
    parser.add_argument("--parms-dir", type=str, default="save_parms", help="Directory containing homography.npy")
    parser.add_argument("--z-height", type=float, default=90.0, help="Table Z height for picking")
    parser.add_argument("--pick-z", type=float, default=None, help="Optional pick Z override")
    parser.add_argument("--speed", type=int, default=50, help="Robot move speed hint")
    parser.add_argument("--move-wait", type=float, default=3.0, help="Wait after each move_to before next action")
    parser.add_argument("--queue-sync", dest="queue_sync", action="store_true", default=True, help="Enable S12 queue feedback")
    parser.add_argument("--no-queue-sync", dest="queue_sync", action="store_false", help="Disable S12 queue feedback")
    parser.add_argument("--sync-timeout", type=float, default=2.0, help="Timeout waiting for queue feedback")
    parser.add_argument("--queue-empty-stable", type=float, default=0.05, help="Required stable empty-queue duration")
    parser.add_argument("--gripper-servo-index", type=int, default=GRIPPER_SERVO_INDEX, help="Gripper servo index")
    parser.add_argument("--gripper-open", type=int, default=GRIPPER_OPEN_ANGLE, help="Gripper open angle")
    parser.add_argument("--gripper-close", type=int, default=GRIPPER_CLOSE_ANGLE, help="Gripper close angle")
    parser.add_argument("--gripper-step-deg", type=float, default=5.0, help="Servo step angle for smooth motion")
    parser.add_argument("--gripper-step-delay", type=float, default=0.12, help="Delay between servo step commands")
    parser.add_argument("--gripper-wait", type=float, default=3.0, help="Wait after gripper command")
    parser.add_argument("--wait-at-target", type=float, default=3.0, help="Wait after arriving at target")
    parser.add_argument("--wait-after-open", type=float, default=3.0, help="Wait after gripper open")
    parser.add_argument("--wait-after-close", type=float, default=3.0, help="Wait after gripper close")
    parser.add_argument("--drop-x", type=float, default=100.0, help="Drop point X")
    parser.add_argument("--drop-y", type=float, default=150.0, help="Drop point Y")
    parser.add_argument("--drop-z", type=float, default=150.0, help="Drop transfer height Z")
    parser.add_argument("--drop-release-z", type=float, default=100.0, help="Drop release height Z")
    parser.add_argument("--post-open-lift-z", type=float, default=150.0, help="Z height after release-open")
    parser.add_argument("--post-open-wait", type=float, default=3.0, help="Wait after post-release lift")
    parser.add_argument("--wait-at-drop", type=float, default=3.0, help="Wait at drop release height")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Global cooldown after sending a target")
    parser.add_argument("--same-target-mm", type=float, default=20.0, help="XY threshold for same-target check")
    parser.add_argument("--same-target-sec", type=float, default=8.0, help="Time window for same-target check")
    parser.add_argument("--home-x", type=float, default=0.0, help="Home X")
    parser.add_argument("--home-y", type=float, default=200.0, help="Home Y")
    parser.add_argument("--home-z", type=float, default=90.0, help="Home Z")
    parser.add_argument("--calib-x", type=float, default=None, help="Deprecated alias for --home-x")
    parser.add_argument("--calib-y", type=float, default=None, help="Deprecated alias for --home-y")
    parser.add_argument("--calib-z", type=float, default=None, help="Deprecated alias for --home-z")
    parser.add_argument("--host", type=str, default="10.149.70.8", help="Robot IP address")
    parser.add_argument("--port", type=int, default=5000, help="Robot TCP port")
    parser.add_argument("--dry-run", action="store_true", help="Print commands instead of sending them")
    parser.add_argument("--skip-enable", action="store_true", help="Do not enable motors on connect")
    parser.add_argument("--verbose", action="store_true", help="Print every command sent to the arm")
    parser.add_argument("--home-first", dest="home_first", action="store_true", default=True, help="Send S10 F1 after enabling motors")
    parser.add_argument("--no-home-first", dest="home_first", action="store_false", help="Skip the initial home move")
    parser.add_argument("--ground-clearance", type=float, default=None, help="Optional ground clearance height")
    return parser.parse_args()


def main() -> None:
    global GRIPPER_SERVO_INDEX, GRIPPER_OPEN_ANGLE, GRIPPER_CLOSE_ANGLE
    args = parse_args()
    GRIPPER_SERVO_INDEX = int(args.gripper_servo_index)
    GRIPPER_OPEN_ANGLE = int(args.gripper_open)
    GRIPPER_CLOSE_ANGLE = int(args.gripper_close)
    args.pick_z = args.pick_z if args.pick_z is not None else args.z_height
    if args.calib_x is not None:
        args.home_x = args.calib_x
    if args.calib_y is not None:
        args.home_y = args.calib_y
    if args.calib_z is not None:
        args.home_z = args.calib_z

    YoloDashboardApp(args).run()


if __name__ == "__main__":
    main()
