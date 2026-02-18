"""
Planar hand-eye calibration script for the Freenove robot arm.

This script estimates a 2D homography between camera pixels and robot XY
coordinates using an ArUco/AprilTag marker attached to the gripper.
python Hand_Eye_Calibration.py --mode calibrate --step
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from freenove_arm import FreenoveArmClient

# Supported ArUco/AprilTag dictionaries by name.
SUPPORTED_ARUCO_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
]
ARUCO_DICT_NAMES: Dict[str, int] = {name: getattr(cv2.aruco, name) for name in SUPPORTED_ARUCO_NAMES}

DEFAULT_ARUCO_NAME = "DICT_5X5_100"
MARKER_LENGTH_METERS = 0.02


@dataclass
class HomographyResult:
    homography: np.ndarray
    inverse: np.ndarray

    def save(self, directory: str = "save_parms") -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "homography.npy"), self.homography)
        np.save(os.path.join(directory, "homography_inv.npy"), self.inverse)

    @classmethod
    def load(cls, directory: str = "save_parms") -> "HomographyResult":
        h_path = os.path.join(directory, "homography.npy")
        inv_path = os.path.join(directory, "homography_inv.npy")
        homography = np.load(h_path)
        inverse = np.load(inv_path)
        return cls(homography=homography, inverse=inverse)


class PlaneCalibrator:
    def __init__(self, dictionary_name: str = DEFAULT_ARUCO_NAME):
        if dictionary_name != "auto" and dictionary_name not in SUPPORTED_ARUCO_NAMES:
            raise ValueError(f"Unknown dictionary {dictionary_name}")

        self.cv2 = cv2
        self.aruco = cv2.aruco
        self.aruco_dicts = ARUCO_DICT_NAMES
        self.np = np

        self.parameters = self.aruco.DetectorParameters()
        target_names = SUPPORTED_ARUCO_NAMES if dictionary_name == "auto" else [dictionary_name]
        self.detectors: List[Tuple[str, object, object]] = []
        for name in target_names:
            marker_dict = self.aruco.getPredefinedDictionary(self.aruco_dicts[name])
            self.detectors.append((name, marker_dict, self.aruco.ArucoDetector(marker_dict, self.parameters)))

        self.active_dict: Optional[str] = None
        self.camera = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera index 2. Try index 0/1 and verify no other app is using it.")

    def read_frame(self) -> np.ndarray:
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def detect_marker(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)

        corners = ids = None
        for name, _marker_dict, detector in self.detectors:
            corners, ids, _rejected = detector.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                self.active_dict = name
                break

        if ids is None or len(ids) == 0:
            return frame, None

        self.aruco.drawDetectedMarkers(frame, corners, ids)
        height, width = frame.shape[:2]
        focal_length = max(width, height)
        camera_matrix = self.np.array(
            [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]],
            dtype=self.np.float32,
        )
        dist_coeffs = self.np.zeros((1, 5))
        rvecs, tvecs, _ = self.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs
        )
        for rvec, tvec in zip(rvecs, tvecs):
            self.cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_METERS)

        corner = corners[0][0]
        center = corner.mean(axis=0)
        self.cv2.circle(frame, tuple(center.astype(int)), 5, (0, 0, 255), -1)
        label = f"{self.active_dict or 'Aruco'} center"
        self.cv2.putText(
            frame,
            label,
            (int(center[0]) + 5, int(center[1]) - 5),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        return frame, center

    def collect_correspondences(
        self,
        robot_points: Sequence[Tuple[float, float, float]],
        robot: FreenoveArmClient,
        settle_time: float = 1.0,
        step_mode: bool = False,
        queue_wait_timeout: float = 8.0,
        capture_timeout: float = 6.0,
        stable_frames: int = 8,
        stable_threshold_px: float = 2.0,
        flush_frames: int = 4,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        np_module = self.np
        image_points: List[np.ndarray] = []
        robot_xy_points: List[np.ndarray] = []

        required_stable_frames = max(1, int(stable_frames))
        max_center_shift_px = max(0.0, float(stable_threshold_px))
        frame_flush_count = max(0, int(flush_frames))

        for idx, (x, y, z) in enumerate(robot_points):
            if step_mode:
                input(f"[{idx + 1}/{len(robot_points)}] Press Enter to move to ({x}, {y}, {z}) ...")

            print(f"Moving to calibration point {idx + 1}/{len(robot_points)}: ({x}, {y}, {z})")
            robot.move_to(x, y, z)

            queue_empty = robot.wait_action_queue_empty(timeout=max(0.1, queue_wait_timeout))
            if not queue_empty and not robot.dry_run:
                raise RuntimeError(
                    "Robot queue did not become empty before capture. "
                    "Increase --queue-timeout or verify S12 queue feedback is enabled on the server."
                )

            if settle_time > 0:
                robot.wait(settle_time)

            for _ in range(frame_flush_count):
                self.read_frame()

            stable_centers: List[np.ndarray] = []
            center: Optional[np.ndarray] = None
            deadline = time.monotonic() + max(0.1, capture_timeout)

            while True:
                frame = self.read_frame()
                vis, detected_center = self.detect_marker(frame)

                now = time.monotonic()
                remaining = max(0.0, deadline - now)
                status = (
                    f"Point {idx + 1}/{len(robot_points)} | "
                    f"stable {len(stable_centers)}/{required_stable_frames} | "
                    f"{remaining:.1f}s"
                )
                self.cv2.putText(vis, status, (10, 30), self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.cv2.imshow("calibration", vis)

                key = self.cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise RuntimeError("Calibration cancelled by user (pressed q)")

                if now >= deadline:
                    raise RuntimeError(
                        f"Timeout waiting for stable marker at point ({x}, {y}, {z}). "
                        "Increase --capture-timeout or improve marker visibility/stability."
                    )

                if detected_center is None:
                    stable_centers.clear()
                    continue

                current = np_module.array(detected_center, dtype=np_module.float32)
                if not stable_centers:
                    stable_centers.append(current)
                    continue

                shift_px = float(np_module.linalg.norm(current - stable_centers[-1]))
                if shift_px <= max_center_shift_px:
                    stable_centers.append(current)
                else:
                    stable_centers = [current]

                if len(stable_centers) >= required_stable_frames:
                    stable_slice = stable_centers[-required_stable_frames:]
                    center = np_module.mean(np_module.stack(stable_slice), axis=0).astype(np_module.float32)
                    break

            if center is None:
                raise RuntimeError("Internal error: marker center not set after stability check")

            image_points.append(center)
            robot_xy_points.append(np_module.array([x, y], dtype=np_module.float32))
            print(f"Captured image point {center} for robot XY ({x}, {y})")

        return image_points, robot_xy_points

    @staticmethod
    def compute_homography(image_points: Sequence[np.ndarray], robot_points: Sequence[np.ndarray]) -> HomographyResult:
        img_pts = np.array(image_points, dtype=np.float32)
        rob_pts = np.array(robot_points, dtype=np.float32)
        if img_pts.shape[0] < 4:
            raise ValueError("At least 4 points are required to compute a homography")

        H, mask = cv2.findHomography(img_pts, rob_pts, method=cv2.RANSAC)
        if H is None:
            raise RuntimeError("Homography estimation failed")
        inliers = int(mask.sum()) if mask is not None else img_pts.shape[0]
        print(f"Homography computed with {inliers}/{len(img_pts)} inliers")
        return HomographyResult(homography=H, inverse=np.linalg.inv(H))

    def pixel_to_robot(self, pixel: Sequence[float], homography: np.ndarray) -> Tuple[float, float]:
        pt = np.array([[pixel[0], pixel[1], 1.0]], dtype=np.float64).T
        mapped = homography @ pt
        mapped /= mapped[2]
        return float(mapped[0]), float(mapped[1])

    def follow_marker(
        self,
        robot: FreenoveArmClient,
        homography: HomographyResult,
        z_height: float,
        move_speed: int = 50,
    ) -> None:
        print("Starting follow mode. Press 'q' to exit.")
        while True:
            frame = self.read_frame()
            vis, center = self.detect_marker(frame)
            status = "Marker not found"

            if center is not None:
                x, y = self.pixel_to_robot(center, homography.homography)
                status = f"Target -> X: {x:.1f} mm, Y: {y:.1f} mm, Z: {z_height:.1f} mm"
                robot.move_to(x, y, z_height, speed=move_speed)
                self.cv2.circle(vis, (int(center[0]), int(center[1])), 10, (255, 0, 0), 2)
            self.cv2.putText(vis, status, (10, 30), self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.cv2.imshow("follow", vis)
            key = self.cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        self.camera.release()
        self.cv2.destroyAllWindows()


def default_calibration_points(z_height: float) -> List[Tuple[float, float, float]]:
    return [
        (0, 200, z_height),
        (100, 250, z_height),
        (100, 200, z_height),
        (0, 150, z_height),
        (-100, 200, z_height),
        (-100, 250, z_height),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planar hand-eye calibration for Freenove arm + USB camera")
    parser.add_argument("--mode", choices=["calibrate", "follow"], required=True, help="Workflow to run")
    parser.add_argument("--z-height", type=float, default=90.0, help="Fixed Z height used for calibration and following (mm)")
    parser.add_argument("--host", type=str, default="10.149.65.232", help="Robot IP address")
    parser.add_argument("--port", type=int, default=5000, help="Robot TCP port (matches client.py default)")
    parser.add_argument("--dry-run", action="store_true", help="Print robot commands instead of sending them")
    parser.add_argument("--skip-enable", action="store_true", help="Do not send the S8 motor enable command on connect")
    parser.add_argument("--speed", type=int, default=50, help="Robot move speed hint (not all firmware uses this)")
    parser.add_argument("--settle", type=float, default=5.0, help="Extra delay after robot reaches each calibration point (s)")
    parser.add_argument(
        "--queue-timeout",
        type=float,
        default=8.0,
        help="Timeout waiting for robot action queue to be empty before capture (s)",
    )
    parser.add_argument(
        "--capture-timeout",
        type=float,
        default=6.0,
        help="Timeout waiting for a stable marker at each point (s)",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=1,
        help="Required consecutive stable marker detections before capture",
    )
    parser.add_argument(
        "--stable-threshold-px",
        type=float,
        default=2.0,
        help="Max marker center shift between frames to count as stable (pixels)",
    )
    parser.add_argument(
        "--flush-frames",
        type=int,
        default=4,
        help="Number of camera frames to discard after each move before detection",
    )
    parser.add_argument("--step", action="store_true", help="Pause for Enter before each calibration move")
    parser.add_argument(
        "--home-first",
        dest="home_first",
        action="store_true",
        default=True,
        help="Send S10 F1 right after enabling motors (default: on; use --no-home-first to skip)",
    )
    parser.add_argument(
        "--no-home-first",
        dest="home_first",
        action="store_false",
        help="Skip the S10 F1 homing step",
    )
    parser.add_argument("--ground-clearance", type=float, help="Send S3 to set the ground clearance height (mm)")
    parser.add_argument("--verbose", action="store_true", help="Print every command sent to the arm")
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default=DEFAULT_ARUCO_NAME,
        choices=["auto"] + SUPPORTED_ARUCO_NAMES,
        help='Marker dictionary name, or "auto" to scan all supported ArUco/AprilTag sets.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrator = PlaneCalibrator(dictionary_name=args.aruco_dict)

    with FreenoveArmClient(
        host=args.host,
        port=args.port,
        dry_run=args.dry_run,
        auto_enable=not args.skip_enable,
        verbose=args.verbose,
    ) as arm:
        if args.home_first:
            arm.return_to_sensor_point(1)
            arm.wait(0.5)

        if args.ground_clearance is not None:
            arm.set_ground_clearance(args.ground_clearance)
            arm.wait(0.1)

        if args.mode == "calibrate":
            robot_points = default_calibration_points(args.z_height)
            arm.set_action_feedback(True)
            try:
                img_pts, rob_pts = calibrator.collect_correspondences(
                    robot_points,
                    arm,
                    settle_time=args.settle,
                    step_mode=args.step,
                    queue_wait_timeout=args.queue_timeout,
                    capture_timeout=args.capture_timeout,
                    stable_frames=args.stable_frames,
                    stable_threshold_px=args.stable_threshold_px,
                    flush_frames=args.flush_frames,
                )
                H = calibrator.compute_homography(img_pts, rob_pts)
                H.save()
                print("Homography saved to save_parms/homography.npy and save_parms/homography_inv.npy")
            finally:
                try:
                    arm.set_action_feedback(False)
                except Exception as exc:
                    print(f"[warn] failed to disable action feedback cleanly: {exc}")
                calibrator.camera.release()
                cv2.destroyAllWindows()
        elif args.mode == "follow":
            H = HomographyResult.load()
            calibrator.follow_marker(arm, H, z_height=args.z_height, move_speed=args.speed)


if __name__ == "__main__":
    main()

