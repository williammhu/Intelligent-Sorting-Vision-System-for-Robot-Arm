"""
YOLOv8 + planar hand–eye (homography) + ONE-SHOT robot move.

Goal:
- Detect a target once
- Convert pixel center -> robot XY using homography
- Send ONE move_to(target), wait HOLD seconds, then ONE move_to(home)
- During robot motion: DO NOT send any more move commands (prevents "jitter/chasing")
- Optionally: repeat for multiple picks (set --repeat)

Windows note:
- For USB cams on Windows, CAP_DSHOW is often more stable than MSMF.
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


# ----------------------------- helpers ------------------------------------ #
def pixel_to_robot(pixel: Sequence[float], homography: np.ndarray) -> Tuple[float, float]:
    pt = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    mapped = homography @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def select_best_box(result) -> Optional[int]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = None
    best_conf = -1.0
    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        cls_name = result.names[cls_id]
        if conf > best_conf:
            best_conf = conf
            best_idx = i
    return best_idx


# ---------------------------- robot thread -------------------------------- #
def robot_worker(
    args: argparse.Namespace,
    target_queue: "queue.Queue[Optional[Tuple[float, float, float, int]]]",
    busy_flag: threading.Event,
    done_flag: threading.Event,
    stop_flag: threading.Event,
) -> None:
    """
    Consumes exactly one target at a time:
      - move_to(target)
      - hold
      - move_to(home)
    Then sets done_flag and clears busy_flag.

    If --repeat > 1, the main thread will enqueue more targets later.
    """
    def log(msg: str) -> None:
        print(f"[robot] {msg}", flush=True)

    try:
        with FreenoveArmClient(
            host=args.host,
            port=args.port,
            dry_run=args.dry_run,
            auto_enable=not args.skip_enable,
        ) as arm:
            log("connected")

            if args.home_first:
                log("home_first: returning to sensor/home point")
                # If your client exposes a different name, adjust here:
                # arm.return_to_sensor_point(1)
                try:
                    arm.return_to_sensor_point(1)  # type: ignore[attr-defined]
                except Exception:
                    # fallback: just move to the specified home point
                    arm.move_to(args.home_x, args.home_y, args.home_z)
                time.sleep(args.settle)

            while not stop_flag.is_set():
                item = target_queue.get()
                if item is None:
                    break

                x, y, z, cls_id = item
                busy_flag.set()
                done_flag.clear()

                log(f"TARGET cls={cls_id} -> ({x:.1f}, {y:.1f}, {z:.1f})")
                log("move_to(target)  [1/2]")
                arm.move_to(x, y, z)
                time.sleep(args.hold_time)

                log("move_to(home)    [2/2]")
                arm.move_to(args.home_x, args.home_y, args.home_z)
                time.sleep(args.settle)

                busy_flag.clear()
                done_flag.set()
                log("done")

    except Exception as exc:
        print(f"[robot] ERROR: {exc}", flush=True)
        busy_flag.clear()
        done_flag.set()


# ---------------------------- vision loop --------------------------------- #
def open_camera(index: int) -> cv2.VideoCapture:
    # Align with yolov8_test.py: force DirectShow backend for Windows stability.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {index}")

    print(f"[vision] camera opened index={index}", flush=True)
    return cap


def detection_loop(args: argparse.Namespace) -> None:
    homography = HomographyResult.load(args.parms_dir).homography

    model = YOLO(args.weights)
    cap = open_camera(args.camera_id)

    target_queue: "queue.Queue[Optional[Tuple[float, float, float, int]]]" = queue.Queue(maxsize=1)
    busy_flag = threading.Event()
    done_flag = threading.Event()
    stop_flag = threading.Event()

    t = threading.Thread(
        target=robot_worker,
        args=(args, target_queue, busy_flag, done_flag, stop_flag),
        daemon=True,
    )
    t.start()

    # ONE-SHOT gating: require target stable for N frames before triggering
    stable_count = 0
    last_xy = None  # type: Optional[Tuple[float, float]]
    triggered = False
    picks_done = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[vision] failed to read frame", flush=True)
                break

            status = "SEARCH"
            overlay = frame

            # If robot is busy, DO NOT run "trigger" logic. You may still run YOLO for display,
            # but it is safer (and faster) to skip YOLO while busy.
            if busy_flag.is_set() or triggered:
                status = "ROBOT BUSY / WAIT"
            else:
                # Run YOLO only when we are allowed to trigger
                result = model.predict(source=frame, conf=args.conf, verbose=False)[0]
                overlay = result.plot()

                best_idx = select_best_box(result)

                if best_idx is None:
                    stable_count = 0
                    last_xy = None
                    status = "NO TARGET"
                else:
                    xyxy = result.boxes.xyxy[best_idx].cpu().numpy()
                    cls_id = int(result.boxes.cls[best_idx])
                    conf = float(result.boxes.conf[best_idx])

                    cx = float((xyxy[0] + xyxy[2]) / 2.0)
                    cy = float((xyxy[1] + xyxy[3]) / 2.0)

                    rx, ry = pixel_to_robot((cx, cy), homography)

                    # draw a small overlay of robot coords
                    cv2.putText(
                        overlay,
                        f"robot XY=({rx:.1f},{ry:.1f}) conf={conf:.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    # Stability gate: only trigger if (rx,ry) stays within a small mm window for N frames
                    if conf < args.min_pick_conf:
                        stable_count = 0
                        last_xy = None
                        status = f"HOLD (conf<{args.min_pick_conf})"
                    else:
                        if last_xy is None:
                            last_xy = (rx, ry)
                            stable_count = 1
                        else:
                            dx = rx - last_xy[0]
                            dy = ry - last_xy[1]
                            if (dx * dx + dy * dy) ** 0.5 <= args.stable_mm:
                                stable_count += 1
                            else:
                                stable_count = 1
                                last_xy = (rx, ry)

                        status = f"STABLE {stable_count}/{args.stable_frames}"

                        # Trigger exactly once when stable enough
                        if stable_count >= args.stable_frames:
                            try:
                                target_queue.put_nowait((rx, ry, args.z_height, cls_id))
                                triggered = True
                                busy_flag.set()  # immediately lock out any further triggers
                                status = "TRIGGERED (one-shot)"
                                print(f"[vision] TRIGGER -> ({rx:.1f},{ry:.1f},{args.z_height:.1f}) cls={cls_id}", flush=True)
                            except queue.Full:
                                status = "QUEUE FULL"

            # If we have triggered, wait for done_flag then allow next pick (if repeat)
            if triggered and done_flag.is_set():
                triggered = False
                busy_flag.clear()
                done_flag.clear()
                stable_count = 0
                last_xy = None
                picks_done += 1
                print(f"[vision] pick cycle done ({picks_done}/{args.repeat})", flush=True)

                if picks_done >= args.repeat:
                    status = "ALL DONE"
                    break
                else:
                    status = "READY FOR NEXT"

            cv2.putText(overlay, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("YOLO + homography (one-shot)", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_flag.set()
        try:
            target_queue.put_nowait(None)
        except queue.Full:
            pass
        t.join(timeout=5)
        cap.release()
        cv2.destroyAllWindows()


# ------------------------------- CLI -------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO + homography + one-shot robot move (no jitter)")
    p.add_argument("--weights", type=str, default=r"D:\yolo\runs\detect\train\weights\best.pt", help="YOLO model weights")
    p.add_argument("--parms-dir", type=str, default="save_parms", help="Directory containing homography.npy")

    p.add_argument("--camera-id", type=int, default=2, help="OpenCV camera index (DirectShow backend on Windows)")
    p.add_argument("--conf", type=float, default=0.7, help="YOLO detection confidence threshold")
    p.add_argument("--min-pick-conf", type=float, default=0.7, help="Minimum conf to allow a trigger")

    p.add_argument("--z-height", type=float, default=90.0, help="Fixed Z for target (mm)")
    p.add_argument("--hold-time", type=float, default=5.0, help="Hold seconds at target before returning home")
    p.add_argument("--settle", type=float, default=0.6, help="Extra settle time after moves (s)")

    p.add_argument("--home-x", type=float, default=0.0, help="Home/calibration X (mm)")
    p.add_argument("--home-y", type=float, default=200.0, help="Home/calibration Y (mm)")
    p.add_argument("--home-z", type=float, default=120.0, help="Home/calibration Z (mm)")
    p.add_argument("--home-first", action="store_true", help="Go home once after connecting")
    p.add_argument("--repeat", type=int, default=1, help="How many pick cycles before exit")

    # stability gate to prevent triggering on noisy detections
    p.add_argument("--stable-frames", type=int, default=1, help="Need N stable frames before triggering")
    p.add_argument("--stable-mm", type=float, default=5.0, help="Max XY movement (mm) between frames to count as stable")

    p.add_argument("--host", type=str, default="10.149.65.232", help="Robot IP")
    p.add_argument("--port", type=int, default=5000, help="Robot TCP port")
    p.add_argument("--dry-run", action="store_true", help="Print commands instead of sending")
    p.add_argument("--skip-enable", action="store_true", help="Do not send motor enable on connect")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detection_loop(args)


if __name__ == "__main__":
    main()
