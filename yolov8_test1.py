"""
YOLOv8 + hand-eye calibration (planar homography) pick-and-place demo.
两秒稳定检测后画框，zxing解码，随后机械臂移动
Workflow
--------
1) Run Hand_Eye_Calibration.py --mode calibrate (or calibration_test.py) to generate
   save_parms/homography.npy / homography_inv.npy.
2) Place objects on the same plane used for calibration.
3) Start this script. It will:
   - detect objects with YOLOv8,
   - map the pixel centre to robot XY via the homography,
   - send the target to the Freenove arm client.

The detection loop stays in the main thread (so OpenCV windows work on Windows);
robot motion runs in a background thread. 本版本采用循环取放：检测到目标即发送一次抓取，
抓取完成后继续搜索；按 q 退出。
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import zxing

from Hand_Eye_Calibration import HomographyResult
from freenove_arm import FreenoveArmClient


GRIPPER_SERVO_INDEX = 0
GRIPPER_OPEN_ANGLE = 70
GRIPPER_CLOSE_ANGLE = 10
ZXING_READER = zxing.BarCodeReader()


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def pixel_to_robot(pixel: Sequence[float], homography: np.ndarray) -> Tuple[float, float]:
    """Map (u, v) pixel to robot XY using a 3x3 homography."""
    pt = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    mapped = homography @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def estimate_bbox_size_cm(xyxy: Sequence[float], homography: np.ndarray) -> Tuple[float, float]:
    """
    Estimate planar object size (cm) from one detection box using homography.
    Assumes the target lies on the calibrated plane.
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    p_tl = pixel_to_robot((x1, y1), homography)
    p_tr = pixel_to_robot((x2, y1), homography)
    p_br = pixel_to_robot((x2, y2), homography)
    p_bl = pixel_to_robot((x1, y2), homography)

    def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    width_cm = 0.5 * (dist(p_tl, p_tr) + dist(p_bl, p_br))
    height_cm = 0.5 * (dist(p_tl, p_bl) + dist(p_tr, p_br))
    return width_cm, height_cm


def select_best_box(result, allowed: Optional[Set[str]]) -> Optional[int]:
    """返回最高置信度的检测（可选类别过滤）。"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = None
    best_conf = -1.0
    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        cls_name = result.names[cls_id]
        if allowed and cls_name not in allowed:
            continue
        if conf > best_conf:
            best_conf = conf
            best_idx = i
    return best_idx


def decode_with_zxing(image: np.ndarray) -> Optional[str]:
    """
    Decode barcode/QR with ZXing.
    The python binding expects a file path, so write a temporary PNG first.
    """
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(buf.tobytes())
            tmp_path = tmp.name
        result = ZXING_READER.decode(tmp_path, try_harder=True)
        if result is None:
            return None
        return getattr(result, "parsed", None) or getattr(result, "raw", None)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# --------------------------------------------------------------------------- #
# Robot worker
# --------------------------------------------------------------------------- #
def robot_worker(
    args: argparse.Namespace,
    target_queue: "queue.Queue",
    busy_flag: threading.Event,
    done_flag: threading.Event,
) -> None:
    """
    Finite-state machine worker: consumes one target at a time and executes
    the move/gripper sequence without long blocking sleeps.
    Each target is a tuple (x_mm, y_mm, z_mm, cls_id).
    """
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
                    print("[robot] queue feedback sync enabled (S12 K1)")
                except Exception as exc:
                    print(f"[robot] queue feedback unavailable, fallback to time wait: {exc}")
                    queue_sync_enabled = False

            def move_phase_done(start_t: float) -> bool:
                nonlocal queue_sync_enabled
                # Keep historical minimum wait and optionally ensure TCP queue backlog is drained.
                if (time.time() - start_t) < args.move_wait:
                    return False
                if not queue_sync_enabled:
                    return True
                try:
                    ok = arm.wait_action_queue_empty(
                        timeout=args.sync_timeout,
                        min_empty_time=args.queue_empty_stable,
                        poll_interval=tick,
                    )
                    if not ok:
                        print("[robot] queue sync timeout, fallback to time wait")
                        queue_sync_enabled = False
                        try:
                            arm.set_action_feedback(False)
                        except Exception:
                            pass
                        return True
                    return True
                except Exception as exc:
                    print(f"[robot] queue sync error, fallback to time wait: {exc}")
                    queue_sync_enabled = False
                    try:
                        arm.set_action_feedback(False)
                    except Exception:
                        pass
                    return True

            if args.home_first:
                arm.return_to_sensor_point(1)
                arm.wait(0.5)
            if args.ground_clearance is not None:
                arm.set_ground_clearance(args.ground_clearance)
                arm.wait(0.1)
            print("[robot] connected to arm")
            home_point = (args.home_x, args.home_y, args.home_z)
            drop_point = (args.drop_x, args.drop_y, args.drop_z)
            drop_release_point = (args.drop_x, args.drop_y, args.drop_release_z)
            drop_post_open_lift_point = (args.drop_x, args.drop_y, args.post_open_lift_z)
            state = "IDLE"
            command_sent = False
            start_time = 0.0
            current_target = None  # (x, y, z, cls_id)
            tick = 0.02  # 20ms tick

            while True:
                now = time.time()

                if state == "IDLE":
                    # blocking wait for next target
                    try:
                        item = target_queue.get(timeout=tick)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    current_target = item
                    busy_flag.set()
                    done_flag.clear()
                    state = "PRE_OPEN_BEFORE_TARGET"
                    command_sent = False
                    start_time = 0.0
                    continue

                x, y, z, cls_id = current_target  # type: ignore[misc]

                if state == "PRE_OPEN_BEFORE_TARGET":
                    print("[robot] step=pre_open_before_target (no wait)")
                    arm.sweep_servo(
                        GRIPPER_SERVO_INDEX,
                        GRIPPER_CLOSE_ANGLE,
                        GRIPPER_OPEN_ANGLE,
                        step_deg=args.gripper_step_deg,
                        step_delay_s=args.gripper_step_delay,
                    )
                    state = "MOVE_TO_TARGET"
                    command_sent = False
                    start_time = 0.0
                    continue

                if state == "MOVE_TO_TARGET":
                    if not command_sent:
                        print(f"[robot] step=move_to_target wait_at_target={args.wait_at_target:.1f}s")
                        arm.move_to(x, y, z, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        state = "WAIT_TARGET"
                        command_sent = False
                        start_time = now
                    arm.wait(tick)
                    continue

                if state == "WAIT_TARGET":
                    if now - start_time >= args.wait_at_target:
                        state = "GRIPPER_OPEN"
                        command_sent = False
                        start_time = 0.0
                    arm.wait(tick)
                    continue

                if state == "GRIPPER_OPEN":
                    if not command_sent:
                        print(f"[robot] step=gripper_open wait_after_open={args.wait_after_open:.1f}s")
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
                        state = "WAIT_OPEN"
                        start_time = now
                        command_sent = False
                    arm.wait(tick)
                    continue

                if state == "WAIT_OPEN":
                    if now - start_time >= args.wait_after_open:
                        state = "GRIPPER_CLOSE"
                        command_sent = False
                        start_time = 0.0
                    arm.wait(tick)
                    continue

                if state == "GRIPPER_CLOSE":
                    if not command_sent:
                        print(f"[robot] step=gripper_close wait_after_close={args.wait_after_close:.1f}s")
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
                        state = "WAIT_CLOSE"
                        start_time = now
                        command_sent = False
                    arm.wait(tick)
                    continue

                if state == "WAIT_CLOSE":
                    if now - start_time >= args.wait_after_close:
                        state = "LIFT_AFTER_GRIP"
                        command_sent = False
                        start_time = 0.0
                    arm.wait(tick)
                    continue

                if state == "LIFT_AFTER_GRIP":
                    if not command_sent:
                        print(f"[robot] step=lift_after_grip target_z={args.drop_z:.1f}")
                        arm.move_to(x, y, args.drop_z, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        state = "MOVE_TO_DROP"
                        start_time = now
                        command_sent = False
                    arm.wait(tick)
                    continue

                if state == "MOVE_TO_DROP":
                    if not command_sent:
                        print(f"[robot] step=move_to_drop_top drop_z={args.drop_z:.1f}")
                        arm.move_to(*drop_point, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        state = "MOVE_TO_DROP_RELEASE_Z"
                        start_time = now
                        command_sent = False
                    arm.wait(tick)
                    continue

                if state == "MOVE_TO_DROP_RELEASE_Z":
                    if not command_sent:
                        print(f"[robot] step=drop_descend release_z={args.drop_release_z:.1f}")
                        arm.move_to(*drop_release_point, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        state = "WAIT_DROP_RELEASE"
                        start_time = now
                        command_sent = False
                    arm.wait(tick)
                    continue

                if state == "WAIT_DROP_RELEASE":
                    if now - start_time >= args.wait_at_drop:
                        state = "RELEASE_OPEN"
                        command_sent = False
                        start_time = 0.0
                    arm.wait(tick)
                    continue

                if state == "RELEASE_OPEN":
                    if not command_sent:
                        print("[robot] step=release_open")
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
                        state = "POST_RELEASE_LIFT"
                        start_time = now
                        command_sent = False
                    arm.wait(tick)
                    continue

                if state == "POST_RELEASE_LIFT":
                    if not command_sent:
                        print(f"[robot] step=post_release_lift target_z={args.post_open_lift_z:.1f}")
                        arm.move_to(*drop_post_open_lift_point, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        state = "WAIT_POST_RELEASE_LIFT"
                        start_time = now
                        command_sent = False
                    arm.wait(tick)
                    continue

                if state == "WAIT_POST_RELEASE_LIFT":
                    if now - start_time >= args.post_open_wait:
                        state = "RELEASE_CLOSE"
                        command_sent = False
                        start_time = 0.0
                    arm.wait(tick)
                    continue

                if state == "RELEASE_CLOSE":
                    if not command_sent:
                        print("[robot] step=release_close")
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
                        state = "MOVE_HOME"
                        command_sent = False
                        start_time = 0.0
                    arm.wait(tick)
                    continue

                if state == "MOVE_HOME":
                    if not command_sent:
                        print("[robot] step=return_home")
                        arm.move_to(*home_point, speed=args.speed)
                        start_time = now
                        command_sent = True
                    if move_phase_done(start_time):
                        state = "WAIT_HOME"
                        start_time = now
                        command_sent = False
                    arm.wait(tick)
                    continue

                if state == "WAIT_HOME":
                    if now - start_time >= args.wait_after_close:
                        state = "IDLE"
                        command_sent = False
                        current_target = None
                        busy_flag.clear()
                        done_flag.set()
                    arm.wait(tick)
                    continue
            if queue_sync_enabled:
                try:
                    arm.set_action_feedback(False)
                except Exception as exc:
                    print(f"[robot] failed to disable queue feedback cleanly: {exc}")
    except Exception as exc:  # pragma: no cover - hardware path
        print(f"[robot] error: {exc}")
        busy_flag.clear()
        done_flag.set()


# --------------------------------------------------------------------------- #
# Detection loop (runs in main thread)
# --------------------------------------------------------------------------- #
def detection_loop(args: argparse.Namespace) -> None:
    try:
        homography = HomographyResult.load(args.parms_dir).homography
    except FileNotFoundError:
        raise SystemExit(
            f"Homography files not found in '{args.parms_dir}'. "
            "Please run Hand_Eye_Calibration.py --mode calibrate first."
        )

    model = YOLO(args.weights)
    # Match Hand_Eye_Calibration: explicitly use DirectShow backend for stability on Windows.
    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise SystemExit("Could not open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"[CAM] requested {args.cam_width}x{args.cam_height}@{args.cam_fps} -> "
        f"got {actual_w}x{actual_h}@{actual_fps:.1f}",
        flush=True,
    )

    allowed = set(args.classes) if args.classes else None
    target_queue: queue.Queue = queue.Queue(maxsize=8)
    busy_flag = threading.Event()
    done_flag = threading.Event()
    last_printed_code: Optional[str] = None
    last_printed_time: float = 0.0
    confirm_states: Dict[Tuple[int, str], Dict[str, Any]] = {}
    recent_sent_targets: List[Tuple[float, float, float, str]] = []
    confirm_conf_threshold = max(0.6, float(args.min_pick_conf))
    pending_required_decode_count = 0

    robot_thread = threading.Thread(
        target=robot_worker,
        args=(args, target_queue, busy_flag, done_flag),
        daemon=True,
    )
    robot_thread.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[vision] failed to read frame")
                break

            # 上一轮动作完成，清理标志，继续搜索
            if done_flag.is_set():
                done_flag.clear()
                busy_flag.clear()

            if busy_flag.is_set():
                status = "Robot running..."
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("yolo + hand-eye", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            result = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            vis = result.plot()

            status = "No target"
            status_color = (0, 255, 0)
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                h, w = frame.shape[:2]
                decoded_candidates = []
                now = time.time()
                detected_box_count = 0
                recent_sent_targets = [
                    item for item in recent_sent_targets if (now - item[0]) <= args.same_target_sec
                ]

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    cls_name = result.names[cls_id]
                    if allowed and cls_name not in allowed:
                        continue
                    detected_box_count += 1

                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    decoded_text: Optional[str] = None
                    size_w_cm: Optional[float] = None
                    size_h_cm: Optional[float] = None
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
                            size_w_cm, size_h_cm = estimate_bbox_size_cm(xyxy, homography)
                            now_print = time.time()
                            if decoded_text != last_printed_code or (now_print - last_printed_time) >= 1.0:
                                print(
                                    f"[ZXING] {decoded_text} | size={size_w_cm:.1f}x{size_h_cm:.1f}cm",
                                    flush=True,
                                )
                                last_printed_code = decoded_text
                                last_printed_time = now_print

                    box_label = f"{cls_name} {conf:.2f}"
                    if decoded_text and size_w_cm is not None and size_h_cm is not None:
                        box_label = f"{box_label} | {decoded_text[:32]} | {size_w_cm:.1f}x{size_h_cm:.1f}cm"
                    cv2.putText(
                        vis, box_label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                    )

                    if decoded_text and size_w_cm is not None and size_h_cm is not None:
                        cx = float((xyxy[0] + xyxy[2]) / 2.0)
                        cy = float((xyxy[1] + xyxy[3]) / 2.0)
                        rx, ry = pixel_to_robot((cx, cy), homography)
                        decoded_candidates.append((conf, cls_id, cls_name, rx, ry, decoded_text, size_w_cm, size_h_cm))

                if detected_box_count >= 2:
                    pending_required_decode_count = max(pending_required_decode_count, detected_box_count)
                elif detected_box_count == 0:
                    pending_required_decode_count = 0

                if decoded_candidates:
                    high_conf_candidates = [item for item in decoded_candidates if item[0] >= confirm_conf_threshold]

                    if high_conf_candidates:
                        seen_keys = set()
                        confirmed_candidates = []

                        for conf, cls_id, cls_name, rx, ry, decoded_text, size_w_cm, size_h_cm in high_conf_candidates:
                            key = (cls_id, decoded_text)
                            seen_keys.add(key)
                            prev_state = confirm_states.get(key)

                            same_candidate = False
                            if prev_state is not None:
                                prev_xy = prev_state["xy"]
                                prev_size = prev_state["size"]
                                dx = rx - prev_xy[0]
                                dy = ry - prev_xy[1]
                                dist = (dx * dx + dy * dy) ** 0.5
                                size_delta = max(
                                    abs(size_w_cm - prev_size[0]),
                                    abs(size_h_cm - prev_size[1]),
                                )
                                same_candidate = (
                                    dist <= args.same_target_mm and size_delta <= args.size_stable_cm
                                )

                            frames = int(prev_state["frames"]) + 1 if same_candidate and prev_state is not None else 1
                            confirm_states[key] = {
                                "frames": frames,
                                "xy": (rx, ry),
                                "size": (size_w_cm, size_h_cm),
                            }

                            if frames >= args.confirm_frames:
                                confirmed_candidates.append(
                                    (conf, cls_id, cls_name, rx, ry, decoded_text, size_w_cm, size_h_cm)
                                )

                        confirm_states = {key: confirm_states[key] for key in seen_keys}

                        if confirmed_candidates:
                            required_decode_count = pending_required_decode_count if pending_required_decode_count >= 2 else 1

                            if len(confirmed_candidates) < required_decode_count:
                                status = f"Waiting decode {len(confirmed_candidates)}/{required_decode_count}"
                                status_color = (0, 255, 255)
                                cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                                cv2.imshow("yolo + hand-eye", vis)
                                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                    break
                                continue

                            confirmed_candidates.sort(key=lambda item: (item[6] * item[7], item[0]), reverse=True)
                            sent_count = 0
                            send_limit = min(required_decode_count, len(confirmed_candidates))

                            for candidate in confirmed_candidates[:send_limit]:
                                (
                                    conf,
                                    cls_id,
                                    cls_name,
                                    rx,
                                    ry,
                                    decoded_text,
                                    size_w_cm,
                                    size_h_cm,
                                ) = candidate
                                same_recent = False
                                for sent_time, sent_x, sent_y, sent_code in recent_sent_targets:
                                    dx = rx - sent_x
                                    dy = ry - sent_y
                                    dist = (dx * dx + dy * dy) ** 0.5
                                    if (
                                        sent_code == decoded_text
                                        and (now - sent_time) <= args.same_target_sec
                                        and dist <= args.same_target_mm
                                    ):
                                        same_recent = True
                                        break

                                if not same_recent:
                                    try:
                                        target_queue.put_nowait((rx, ry, args.pick_z, cls_id))
                                        recent_sent_targets.append((now, rx, ry, decoded_text))
                                        confirm_states.pop((cls_id, decoded_text), None)
                                        sent_count += 1
                                    except queue.Full:
                                        status = "Queue full"
                                        status_color = (0, 255, 255)
                                        break

                            if sent_count > 0:
                                busy_flag.set()
                                status = f"Targets sent {sent_count}"
                                pending_required_decode_count = 0
                            else:
                                status = "Dedup skip"
                                status_color = (0, 255, 255)
                        else:
                            max_frames = max(int(confirm_states[key]["frames"]) for key in seen_keys)
                            if pending_required_decode_count >= 2:
                                status = f"Waiting decode 0/{pending_required_decode_count}"
                            else:
                                status = f"Confirming {len(seen_keys)} target(s) {max_frames}/{args.confirm_frames}"
                            status_color = (0, 255, 255)
                    else:
                        best_conf = max(item[0] for item in decoded_candidates)
                        if pending_required_decode_count >= 2:
                            status = f"Waiting decode 0/{pending_required_decode_count}"
                        else:
                            confirm_states.clear()
                            status = f"Low conf {best_conf:.2f}"
                        status_color = (0, 255, 255)
                else:
                    if pending_required_decode_count >= 2:
                        status = f"Waiting decode 0/{pending_required_decode_count}"
                    else:
                        confirm_states.clear()
                        status = "Barcode decode failed"
                    status_color = (0, 255, 255)
            else:
                confirm_states.clear()
                pending_required_decode_count = 0

            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.imshow("yolo + hand-eye", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        target_queue.put(None)
        robot_thread.join(timeout=5)
        cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine YOLOv8 detection with homography-based hand-eye calibration")
    parser.add_argument("--weights", type=str, default=r"D:\yolo\runs\detect\train\weights\best.pt", help="YOLO model weights")
    parser.add_argument("--camera-id", type=int, default=2, help="OpenCV camera index")
    parser.add_argument("--cam-width", type=int, default=1280, help="Requested camera width (pixels)")
    parser.add_argument("--cam-height", type=int, default=720, help="Requested camera height (pixels)")
    parser.add_argument("--cam-fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument("--conf", type=float, default=0.6, help="YOLO detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--min-pick-conf", type=float, default=0.6, help="Minimum confidence to send a pick command")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Required stable decoded frames before sending a target")
    parser.add_argument(
        "--size-stable-cm",
        dest="size_stable_cm",
        type=float,
        default=8.0,
        help="Max size delta (cm) between frames to count as stable",
    )
    parser.add_argument("--classes", nargs="*", help="Optional class name filter for picking")
    parser.add_argument("--parms-dir", type=str, default="save_parms", help="Directory containing homography.npy")
    parser.add_argument("--z-height", type=float, default=90.0, help="Table Z height for picking (mm)")
    parser.add_argument("--pick-z", type=float, default=None, help="Optional pick Z override; defaults to --z-height")
    parser.add_argument("--speed", type=int, default=50, help="Robot move speed hint")
    parser.add_argument("--move-wait", type=float, default=3.0, help="Wait after each move_to before next action (s)")
    parser.add_argument(
        "--queue-sync",
        dest="queue_sync",
        action="store_true",
        default=True,
        help="Enable S12 queue feedback synchronization for move transitions",
    )
    parser.add_argument(
        "--no-queue-sync",
        dest="queue_sync",
        action="store_false",
        help="Disable queue feedback sync and use pure time waits",
    )
    parser.add_argument(
        "--sync-timeout",
        type=float,
        default=2.0,
        help="Timeout for waiting queue feedback to report empty (s)",
    )
    parser.add_argument(
        "--queue-empty-stable",
        type=float,
        default=0.05,
        help="Required stable duration with queue length 0 before considering it empty (s)",
    )
    parser.add_argument("--gripper-servo-index", type=int, default=GRIPPER_SERVO_INDEX, help="Gripper servo index")
    parser.add_argument("--gripper-open", type=int, default=GRIPPER_OPEN_ANGLE, help="Gripper open angle")
    parser.add_argument("--gripper-close", type=int, default=GRIPPER_CLOSE_ANGLE, help="Gripper close angle")
    parser.add_argument("--gripper-step-deg", type=float, default=5.0, help="Servo angle step used for slow open/close motion")
    parser.add_argument("--gripper-step-delay", type=float, default=0.12, help="Delay between servo angle steps (s)")
    parser.add_argument("--gripper-wait", type=float, default=3.0, help="Wait after gripper open/close command (s)")
    parser.add_argument("--wait-at-target", type=float, default=3.0, help="Wait at target after arriving (s)")
    parser.add_argument("--wait-after-open", type=float, default=3.0, help="Wait after gripper open (s)")
    parser.add_argument("--wait-after-close", type=float, default=3.0, help="Wait after gripper close (s)")
    parser.add_argument("--drop-x", type=float, default=-100.0, help="Drop point X (mm)")
    parser.add_argument("--drop-y", type=float, default=150.0, help="Drop point Y (mm)")
    parser.add_argument("--drop-z", type=float, default=150.0, help="Drop transfer height Z (mm)")
    parser.add_argument("--drop-release-z", type=float, default=100.0, help="Drop release height Z (mm)")
    parser.add_argument("--post-open-lift-z", type=float, default=130.0, help="Z height after release-open before close (mm)")
    parser.add_argument("--post-open-wait", type=float, default=3.0, help="Wait time after post-release lift before close (s)")
    parser.add_argument("--wait-at-drop", type=float, default=3.0, help="Wait at drop release height before release (s)")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Global cooldown after sending a target (s)")
    parser.add_argument("--same-target-mm", type=float, default=20.0, help="Treat target as same if XY distance is within this threshold (mm)")
    parser.add_argument("--same-target-sec", type=float, default=8.0, help="Dedup time window for same-target check (s)")
    parser.add_argument("--home-x", type=float, default=0.0, help="Home X to return after move (mm)")
    parser.add_argument("--home-y", type=float, default=200.0, help="Home Y to return after move (mm)")
    parser.add_argument("--home-z", type=float, default=90.0, help="Home Z to return after move (mm)")
    parser.add_argument("--calib-x", type=float, default=None, help="(Deprecated) alias for --home-x")
    parser.add_argument("--calib-y", type=float, default=None, help="(Deprecated) alias for --home-y")
    parser.add_argument("--calib-z", type=float, default=None, help="(Deprecated) alias for --home-z")
    parser.add_argument("--host", type=str, default="10.149.70.8", help="Robot IP address")
    parser.add_argument("--port", type=int, default=5000, help="Robot TCP port")
    parser.add_argument("--dry-run", action="store_true", help="Print commands instead of sending to robot")
    parser.add_argument("--skip-enable", action="store_true", help="Do not send motor enable command on connect")
    parser.add_argument("--verbose", action="store_true", help="Print every command sent to the arm")
    parser.add_argument(
        "--home-first",
        dest="home_first",
        action="store_true",
        default=True,
        help="Send S10 F1 once after enabling motors (recommended)",
    )
    parser.add_argument(
        "--no-home-first",
        dest="home_first",
        action="store_false",
        help="Skip the homing move after enabling motors",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=None,
        help="Optional ground clearance height to set via S3 (mm)",
    )
    return parser.parse_args()


def main() -> None:
    global GRIPPER_SERVO_INDEX, GRIPPER_OPEN_ANGLE, GRIPPER_CLOSE_ANGLE
    args = parse_args()
    GRIPPER_SERVO_INDEX = int(args.gripper_servo_index)
    GRIPPER_OPEN_ANGLE = int(args.gripper_open)
    GRIPPER_CLOSE_ANGLE = int(args.gripper_close)
    # Backward-compatibility for pick Z and home aliases.
    args.pick_z = args.pick_z if args.pick_z is not None else args.z_height
    if args.calib_x is not None:
        args.home_x = args.calib_x
    if args.calib_y is not None:
        args.home_y = args.calib_y
    if args.calib_z is not None:
        args.home_z = args.calib_z
    detection_loop(args)


if __name__ == "__main__":
    main()
