"""
YOLOv8 + hand-eye calibration (planar homography) pick-and-place demo.
精简版，不包括解码和yolo画框

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
from typing import Optional, Sequence, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from Hand_Eye_Calibration import HomographyResult
from freenove_arm import FreenoveArmClient

ROBOT_HOST = "10.149.65.232"
ROBOT_PORT = 5000
YOLO_WEIGHTS = r"D:\yolo\runs\detect\train\weights\best.pt"
YOLO_CONF = 0.7
PARMS_DIR = "save_parms"
GRIPPER_SERVO_INDEX = 0
HOME_X = 0.0
HOME_Y = 200.0
HOME_Z = 90.0


def pixel_to_robot(pixel: Sequence[float], homography: np.ndarray) -> Tuple[float, float]:
    """Map (u, v) pixel to robot XY using a 3x3 homography."""
    pt = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    mapped = homography @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def select_best_box(result) -> Optional[int]:
    """返回最高置信度的检测（可选类别过滤）。"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = None
    best_conf = -1.0
    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        if conf > best_conf:
            best_conf = conf
            best_idx = i
    return best_idx



def robot_worker(
    args: argparse.Namespace,
    target_queue: "queue.Queue",
    busy_flag: threading.Event,
    done_flag: threading.Event,
) -> None:
    """
    Finite-state machine worker: consumes one target at a time and executes
    the move/gripper sequence without long blocking sleeps.
    Each target is a tuple (x_mm, y_mm, z_mm).
    """
    try:
        with FreenoveArmClient(
            host=ROBOT_HOST,
            port=ROBOT_PORT,
            dry_run=False,
            auto_enable=True,
            verbose=args.verbose,
        ) as arm:
            queue_sync_enabled = False
            try:
                arm.set_action_feedback(True)
                queue_sync_enabled = True
                print("[robot] queue feedback sync enabled (S12 K1)")
            except Exception as exc:
                print(f"[robot] queue feedback unavailable, fallback to time wait: {exc}")
                queue_sync_enabled = False

            def move_phase_done(start_t: float) -> bool:
                nonlocal queue_sync_enabled
                def disable_queue_sync(message: str) -> None:
                    nonlocal queue_sync_enabled
                    print(message)
                    queue_sync_enabled = False
                    try:
                        arm.set_action_feedback(False)
                    except Exception:
                        pass

                # Keep historical minimum wait and optionally ensure TCP queue backlog is drained.
                if (time.time() - start_t) < args.wait:
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
                        disable_queue_sync("[robot] queue sync timeout, fallback to time wait")
                        return True
                    return True
                except Exception as exc:
                    disable_queue_sync(f"[robot] queue sync error, fallback to time wait: {exc}")
                    return True

            arm.return_to_sensor_point(1)
            arm.wait(0.5)
            if args.ground_clearance is not None:
                arm.set_ground_clearance(args.ground_clearance)
                arm.wait(0.1)
            print("[robot] connected to arm")
            home_point = (HOME_X, HOME_Y, HOME_Z)
            drop_point = (args.drop_x, args.drop_y, args.drop_z)
            drop_release_point = (args.drop_x, args.drop_y, args.drop_release_z)
            gripper_servo_index = GRIPPER_SERVO_INDEX
            gripper_open_angle = int(args.gripper_open)
            gripper_close_angle = int(args.gripper_close)
            move_speed = 50
            state = "IDLE"
            command_sent = False
            start_time = 0.0
            current_target = None  # (x, y, z)
            tick = 0.02  # 20ms tick

            def transition(next_state: str, *, start_at: Optional[float] = None) -> None:
                nonlocal state, command_sent, start_time
                state = next_state
                command_sent = False
                if start_at is not None:
                    start_time = start_at

            def run_move_state(
                now_t: float,
                log_msg: str,
                target_xyz: Tuple[float, float, float],
                next_state: str,
            ) -> None:
                nonlocal command_sent, start_time
                if not command_sent:
                    print(log_msg)
                    arm.move_to(*target_xyz, speed=move_speed)
                    start_time = now_t
                    command_sent = True
                if move_phase_done(start_time):
                    transition(next_state, start_at=now_t)
                arm.wait(tick)

            def run_servo_state(
                now_t: float,
                log_msg: str,
                angle: int,
                next_state: str,
                next_start_at: float,
            ) -> None:
                nonlocal command_sent, start_time
                if not command_sent:
                    print(log_msg)
                    arm.set_servo(gripper_servo_index, angle)
                    start_time = now_t
                    command_sent = True
                if (now_t - start_time) >= args.gripper_wait:
                    transition(next_state, start_at=next_start_at)
                arm.wait(tick)

            def run_wait_state(now_t: float, wait_s: float, next_state: str, next_start_at: float) -> None:
                if (now_t - start_time) >= wait_s:
                    transition(next_state, start_at=next_start_at)
                arm.wait(tick)

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
                    transition("MOVE_TO_TARGET", start_at=0.0)
                    continue

                x, y, z = current_target  # type: ignore[misc]

                if state == "MOVE_TO_TARGET":
                    run_move_state(
                        now,
                        f"[robot] step=move_to_target wait={args.wait:.1f}s",
                        (x, y, z),
                        "WAIT_TARGET",
                    )
                    continue

                if state == "WAIT_TARGET":
                    run_wait_state(now, args.wait, "GRIPPER_OPEN", 0.0)
                    continue

                if state == "GRIPPER_OPEN":
                    run_servo_state(
                        now,
                        f"[robot] step=gripper_open wait={args.wait:.1f}s",
                        gripper_open_angle,
                        "WAIT_OPEN",
                        now,
                    )
                    continue

                if state == "WAIT_OPEN":
                    run_wait_state(now, args.wait, "GRIPPER_CLOSE", 0.0)
                    continue

                if state == "GRIPPER_CLOSE":
                    run_servo_state(
                        now,
                        f"[robot] step=gripper_close wait={args.wait:.1f}s",
                        gripper_close_angle,
                        "WAIT_CLOSE",
                        now,
                    )
                    continue

                if state == "WAIT_CLOSE":
                    run_wait_state(now, args.wait, "LIFT_AFTER_GRIP", 0.0)
                    continue

                if state == "LIFT_AFTER_GRIP":
                    run_move_state(now,
                        f"[robot] step=lift_after_grip target_z={args.drop_z:.1f}",
                        (x, y, args.drop_z),
                        "MOVE_TO_DROP",
                    )
                    continue

                if state == "MOVE_TO_DROP":
                    run_move_state(
                        now,
                        f"[robot] step=move_to_drop_top drop_z={args.drop_z:.1f}",
                        drop_point,
                        "MOVE_TO_DROP_RELEASE_Z",
                    )
                    continue

                if state == "MOVE_TO_DROP_RELEASE_Z":
                    run_move_state(
                        now,
                        f"[robot] step=drop_descend release_z={args.drop_release_z:.1f}",
                        drop_release_point,
                        "WAIT_DROP_RELEASE",
                    )
                    continue

                if state == "WAIT_DROP_RELEASE":
                    run_wait_state(now, args.wait, "RELEASE_OPEN", 0.0)
                    continue

                if state == "RELEASE_OPEN":
                    run_servo_state(now, "[robot] step=release_open", gripper_open_angle, "WAIT_RELEASE", now)
                    continue

                if state == "WAIT_RELEASE":
                    run_wait_state(now, args.wait, "RELEASE_CLOSE", 0.0)
                    continue

                if state == "RELEASE_CLOSE":
                    run_servo_state(now, "[robot] step=release_close", gripper_close_angle, "MOVE_HOME", 0.0)
                    continue

                if state == "MOVE_HOME":
                    run_move_state(now, "[robot] step=return_home", home_point, "WAIT_HOME")
                    continue

                if state == "WAIT_HOME":
                    if now - start_time >= args.wait:
                        transition("IDLE")
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

def detection_loop(args: argparse.Namespace) -> None:
    def render_status_and_check_quit(
        frame: np.ndarray,
        status: str,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> bool:
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("yolo + hand-eye", frame)
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def get_pick_skip_status(
        now_t: float,
        target_xy: Tuple[float, float],
        prev_time: Optional[float],
        prev_xy: Optional[Tuple[float, float]],
    ) -> Optional[str]:
        if prev_time is None:
            return None
        elapsed = now_t - prev_time
        if elapsed < args.cooldown:
            return f"Cooldown {max(0.0, args.cooldown - elapsed):.1f}s"
        if prev_xy is None:
            return None
        dx = target_xy[0] - prev_xy[0]
        dy = target_xy[1] - prev_xy[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if elapsed <= args.same_target_sec and dist <= args.same_target_mm:
            return "Dedup skip"
        return None

    try:
        homography = HomographyResult.load(PARMS_DIR).homography
    except FileNotFoundError:
        raise SystemExit(
            f"Homography files not found in '{PARMS_DIR}'. "
            "Please run Hand_Eye_Calibration.py --mode calibrate first."
        )

    model = YOLO(YOLO_WEIGHTS)
    # Match Hand_Eye_Calibration: explicitly use DirectShow backend for stability on Windows.
    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise SystemExit("Could not open camera")

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
                if render_status_and_check_quit(frame, status):
                    break
                continue

            result = model.predict(
                source=frame,
                conf=YOLO_CONF,
                verbose=False,
            )[0]

            best_idx = select_best_box(result)
            status = "No target"

            if best_idx is not None:
                xyxy = result.boxes.xyxy[best_idx].cpu().numpy()
                cx = float((xyxy[0] + xyxy[2]) / 2.0)
                cy = float((xyxy[1] + xyxy[3]) / 2.0)
                rx, ry = pixel_to_robot((cx, cy), homography)

                if not busy_flag.is_set():
                    now = time.time()
                    skip_status = get_pick_skip_status(now, (rx, ry), last_sent_time, last_sent_xy)
                    if skip_status is not None:
                        status = skip_status
                        if render_status_and_check_quit(frame, status, (0, 255, 255)):
                            break
                        continue

                    try:
                        target_queue.put_nowait((rx, ry, args.pick_z))
                        busy_flag.set()
                        last_sent_time = now
                        last_sent_xy = (rx, ry)
                        status = "Target sent"
                    except queue.Full:
                        status = "Queue full"

            if render_status_and_check_quit(frame, status):
                break
    finally:
        target_queue.put(None)
        robot_thread.join(timeout=5)
        cap.release()
        cv2.destroyAllWindows()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine YOLOv8 detection with homography-based hand-eye calibration")
    scalar_specs = [
        ("--camera-id", {"type": int, "default": 2}),
        ("--pick-z", {"type": float, "default": 90.0}),
        ("--wait", {"type": float, "default": 3.0}),
        ("--sync-timeout", {"type": float, "default": 2.0}),
        ("--queue-empty-stable", {"type": float, "default": 0.05},),
        ("--gripper-open", {"type": int, "default": 70}),
        ("--gripper-close", {"type": int, "default": 0}),
        ("--gripper-wait", {"type": float, "default": 3.0}),
        ("--drop-x", {"type": float, "default": 0.0}),
        ("--drop-y", {"type": float, "default": 150.0}),
        ("--drop-z", {"type": float, "default": 120.0}),
        ("--drop-release-z", {"type": float, "default": 90.0}),
        ("--cooldown", {"type": float, "default": 1.5}),
        ("--same-target-mm", {"type": float, "default": 20.0},),
        ("--same-target-sec", {"type": float, "default": 8.0}),
        ("--verbose", {"action": "store_true"}),
        ("--ground-clearance", {"type": float, "default": None}),
    ]
    for flag, kwargs in scalar_specs:
        parser.add_argument(flag, **kwargs)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detection_loop(args)


if __name__ == "__main__":
    main()
