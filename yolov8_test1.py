"""
YOLOv8 + hand-eye calibration (planar homography) pick-and-place demo.

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
import queue
import threading
import time
from typing import Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from Hand_Eye_Calibration import HomographyResult
from freenove_arm import FreenoveArmClient

GRIPPER_SERVO_INDEX = 4
GRIPPER_OPEN_ANGLE = 55
GRIPPER_CLOSE_ANGLE = 15


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def pixel_to_robot(pixel: Sequence[float], homography: np.ndarray) -> Tuple[float, float]:
    """Map (u, v) pixel to robot XY using a 3x3 homography."""
    pt = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    mapped = homography @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


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
                    state = "MOVE_TO_TARGET"
                    command_sent = False
                    start_time = 0.0
                    continue

                x, y, z, cls_id = current_target  # type: ignore[misc]

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
                        arm.set_servo(GRIPPER_SERVO_INDEX, GRIPPER_OPEN_ANGLE)
                        start_time = now
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
                        arm.set_servo(GRIPPER_SERVO_INDEX, GRIPPER_CLOSE_ANGLE)
                        start_time = now
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
                        arm.set_servo(GRIPPER_SERVO_INDEX, GRIPPER_OPEN_ANGLE)
                        start_time = now
                        command_sent = True
                    if now - start_time >= args.gripper_wait:
                        state = "WAIT_RELEASE"
                        start_time = now
                        command_sent = False
                    arm.wait(tick)
                    continue

                if state == "WAIT_RELEASE":
                    if now - start_time >= args.wait_after_open:
                        state = "RELEASE_CLOSE"
                        command_sent = False
                        start_time = 0.0
                    arm.wait(tick)
                    continue

                if state == "RELEASE_CLOSE":
                    if not command_sent:
                        print("[robot] step=release_close")
                        arm.set_servo(GRIPPER_SERVO_INDEX, GRIPPER_CLOSE_ANGLE)
                        start_time = now
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

    allowed = set(args.classes) if args.classes else None
    target_queue: queue.Queue = queue.Queue(maxsize=1)
    busy_flag = threading.Event()
    done_flag = threading.Event()
    last_sent_time: Optional[float] = None
    last_sent_xy: Optional[Tuple[float, float]] = None

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
                verbose=False,
            )[0]

            best_idx = select_best_box(result, allowed)
            status = "No target"

            if best_idx is not None:
                xyxy = result.boxes.xyxy[best_idx].cpu().numpy()
                cls_id = int(result.boxes.cls[best_idx])
                conf = float(result.boxes.conf[best_idx])
                cx = float((xyxy[0] + xyxy[2]) / 2.0)
                cy = float((xyxy[1] + xyxy[3]) / 2.0)
                rx, ry = pixel_to_robot((cx, cy), homography)

                # draw overlay
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"{result.names[cls_id]} {conf:.2f}",
                    (int(xyxy[0]), int(xyxy[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"robot XY=({rx:.1f},{ry:.1f})",
                    (int(xyxy[0]), int(xyxy[1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                if conf >= args.min_pick_conf and not busy_flag.is_set():
                    now = time.time()
                    in_cooldown = last_sent_time is not None and (now - last_sent_time) < args.cooldown
                    same_recent = False
                    if last_sent_time is not None and last_sent_xy is not None:
                        dt = now - last_sent_time
                        dx = rx - last_sent_xy[0]
                        dy = ry - last_sent_xy[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        same_recent = dt <= args.same_target_sec and dist <= args.same_target_mm

                    if in_cooldown:
                        remaining = args.cooldown - (now - last_sent_time)
                        status = f"Cooldown {max(0.0, remaining):.1f}s"
                        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.imshow("yolo + hand-eye", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            break
                        continue

                    if same_recent:
                        status = "Dedup skip"
                        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.imshow("yolo + hand-eye", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            break
                        continue

                    try:
                        target_queue.put_nowait((rx, ry, args.pick_z, cls_id))
                        busy_flag.set()
                        last_sent_time = now
                        last_sent_xy = (rx, ry)
                        status = "Target sent"
                    except queue.Full:
                        status = "Queue full"

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("yolo + hand-eye", frame)
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
    parser.add_argument("--conf", type=float, default=0.7, help="YOLO detection confidence threshold")
    parser.add_argument("--min-pick-conf", type=float, default=0.7, help="Minimum confidence to send a pick command")
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
    parser.add_argument("--gripper-wait", type=float, default=0.4, help="Wait after gripper open/close command (s)")
    parser.add_argument("--wait-at-target", type=float, default=3.0, help="Wait at target after arriving (s)")
    parser.add_argument("--wait-after-open", type=float, default=3.0, help="Wait after gripper open (s)")
    parser.add_argument("--wait-after-close", type=float, default=3.0, help="Wait after gripper close (s)")
    parser.add_argument("--drop-x", type=float, default=0.0, help="Drop point X (mm)")
    parser.add_argument("--drop-y", type=float, default=150.0, help="Drop point Y (mm)")
    parser.add_argument("--drop-z", type=float, default=120.0, help="Drop transfer height Z (mm)")
    parser.add_argument("--drop-release-z", type=float, default=90.0, help="Drop release height Z (mm)")
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
    parser.add_argument("--host", type=str, default="10.149.65.232", help="Robot IP address")
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
