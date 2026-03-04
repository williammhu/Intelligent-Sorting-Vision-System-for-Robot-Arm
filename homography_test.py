import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CAMERA_ID = 2
DEFAULT_CAM_WIDTH = 1280
DEFAULT_CAM_HEIGHT = 720
DEFAULT_CAM_FPS = 30
WINDOW_NAME = "homography test"
CLICKED_POINTS_COLOR = (0, 0, 255)
OVERLAY_POINTS_COLOR = (0, 255, 255)


def load_homography(parms_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    homography_path = Path(parms_dir) / "homography.npy"
    homography_inv_path = Path(parms_dir) / "homography_inv.npy"

    homography = np.load(homography_path).astype(np.float64)
    homography = homography / homography[2, 2]

    if homography_inv_path.exists():
        homography_inv = np.load(homography_inv_path).astype(np.float64)
    else:
        homography_inv = np.linalg.inv(homography)
        print(f"{homography_inv_path} not found, computed inverse from homography.npy")

    homography_inv = homography_inv / homography_inv[2, 2]
    return homography, homography_inv


def pixel_to_robot(u: float, v: float, homography: np.ndarray) -> Tuple[float, float]:
    point = np.array([u, v, 1.0], dtype=np.float64)
    mapped = homography @ point
    mapped = mapped / mapped[2]
    return float(mapped[0]), float(mapped[1])


def robot_to_pixel(x: float, y: float, homography_inv: np.ndarray) -> Tuple[float, float]:
    point = np.array([x, y, 1.0], dtype=np.float64)
    mapped = homography_inv @ point
    mapped = mapped / mapped[2]
    return float(mapped[0]), float(mapped[1])


def parse_point(text: str) -> Tuple[float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Point '{text}' must be in x,y format")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Point '{text}' contains non-numeric values") from exc


def parse_point_list(text: str) -> List[Tuple[float, float]]:
    stripped = text.strip()
    if not stripped:
        return []
    return [parse_point(item) for item in stripped.split(";") if item.strip()]


def ensure_parent_dir(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def open_camera(camera_id: int, cam_width: int, cam_height: int, cam_fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, cam_fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera opened: requested {cam_width}x{cam_height}@{cam_fps}, "
        f"actual {actual_width}x{actual_height}@{actual_fps:.1f}"
    )
    return cap


def draw_labeled_marker(
    frame: np.ndarray,
    x: float,
    y: float,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    px = int(round(x))
    py = int(round(y))
    cv2.drawMarker(
        frame,
        (px, py),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=22,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (px + 8, py - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def prompt_xy_for_click(pixel: Tuple[int, int], remaining: Optional[int]) -> Tuple[float, float]:
    hint = ""
    if remaining is not None:
        hint = f" ({remaining} auto XY remaining after this click)"
    while True:
        raw = input(f"Enter robot XY in mm for pixel ({pixel[0]}, {pixel[1]}){hint}: ").strip()
        try:
            return parse_point(raw)
        except argparse.ArgumentTypeError as exc:
            print(exc)


def append_csv_row(csv_path: Path, row: Sequence[object], header: Sequence[str]) -> None:
    ensure_parent_dir(csv_path)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
        handle.flush()


def run_overlay(args: argparse.Namespace) -> None:
    _, homography_inv = load_homography(args.parms_dir)
    points_xy = parse_point_list(args.points)
    if not points_xy:
        raise SystemExit("overlay requires --points \"X1,Y1;X2,Y2;...\"")

    projected: List[Tuple[float, float, str]] = []
    for idx, (x, y) in enumerate(points_xy, start=1):
        u, v = robot_to_pixel(x, y, homography_inv)
        projected.append((u, v, f"P{idx} ({x:.1f},{y:.1f})"))
        print(f"Robot({x:.1f},{y:.1f}) mm -> Pixel({u:.1f},{v:.1f})")

    cap = open_camera(args.camera_id, args.cam_width, args.cam_height, args.cam_fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            for u, v, label in projected:
                draw_labeled_marker(frame, u, v, label, OVERLAY_POINTS_COLOR)

            cv2.putText(
                frame,
                "overlay mode: press q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_collect(args: argparse.Namespace) -> None:
    auto_xy = parse_point_list(args.xy_list) if args.xy_list else []
    pending_clicks: Deque[Tuple[int, int]] = deque()
    collected_rows: List[Tuple[int, int, float, float, str]] = []
    csv_path = Path(args.csv)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_clicks.append((int(x), int(y)))
            print(f"Captured click at pixel ({x}, {y})")

    cap = open_camera(args.camera_id, args.cam_width, args.cam_height, args.cam_fps)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print("collect mode: left-click image points, press q to quit")
    if auto_xy:
        print(f"Using predefined XY list with {len(auto_xy)} entries")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            while pending_clicks:
                u, v = pending_clicks.popleft()
                if auto_xy:
                    x, y = auto_xy.pop(0)
                    print(f"Assigned predefined XY ({x:.1f}, {y:.1f}) mm to pixel ({u}, {v})")
                else:
                    x, y = prompt_xy_for_click((u, v), None)

                timestamp = datetime.now().isoformat(timespec="seconds")
                append_csv_row(
                    csv_path,
                    [u, v, x, y, timestamp],
                    ["u", "v", "X", "Y", "timestamp"],
                )
                collected_rows.append((u, v, x, y, timestamp))
                print(f"Saved: u={u}, v={v}, X={x:.3f}, Y={y:.3f}, timestamp={timestamp}")

            for idx, (u, v, x, y, _) in enumerate(collected_rows, start=1):
                draw_labeled_marker(frame, u, v, f"C{idx} ({x:.1f},{y:.1f})", CLICKED_POINTS_COLOR)

            status = f"collect mode: saved {len(collected_rows)} point(s), press q to quit"
            if auto_xy:
                status += f", {len(auto_xy)} auto XY left"
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def load_measurements(csv_path: Path) -> List[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")
    return rows


def save_error_plots(
    summary_rows: List[dict],
    output_dir: Path,
    hist_png: str,
    vector_png: str,
) -> None:
    x = np.array([float(row["X"]) for row in summary_rows], dtype=np.float64)
    y = np.array([float(row["Y"]) for row in summary_rows], dtype=np.float64)
    x_hat = np.array([float(row["Xhat"]) for row in summary_rows], dtype=np.float64)
    y_hat = np.array([float(row["Yhat"]) for row in summary_rows], dtype=np.float64)
    errors = np.array([float(row["error_mm"]) for row in summary_rows], dtype=np.float64)

    ensure_parent_dir(output_dir / hist_png)
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=min(20, max(5, len(errors))), color="#4C72B0", edgecolor="black")
    plt.xlabel("Mapping error (mm)")
    plt.ylabel("Count")
    plt.title("Pixel-to-Robot Mapping Error Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / hist_png, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.quiver(
        x,
        y,
        x_hat - x,
        y_hat - y,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#DD8452",
    )
    plt.scatter(x, y, color="#55A868", s=18, label="Ground truth")
    plt.scatter(x_hat, y_hat, color="#C44E52", s=18, label="Predicted")
    plt.xlabel("Robot X (mm)")
    plt.ylabel("Robot Y (mm)")
    plt.title("Mapping Error Vectors")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / vector_png, dpi=150)
    plt.close()


def run_evaluate(args: argparse.Namespace) -> None:
    homography, _ = load_homography(args.parms_dir)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    rows = load_measurements(csv_path)

    summary_rows: List[dict] = []
    errors: List[float] = []
    for row in rows:
        u = float(row["u"])
        v = float(row["v"])
        x = float(row["X"])
        y = float(row["Y"])
        x_hat, y_hat = pixel_to_robot(u, v, homography)
        dx = x_hat - x
        dy = y_hat - y
        error_mm = float(np.hypot(dx, dy))
        errors.append(error_mm)
        summary_rows.append(
            {
                "u": u,
                "v": v,
                "X": x,
                "Y": y,
                "Xhat": x_hat,
                "Yhat": y_hat,
                "dX": dx,
                "dY": dy,
                "error_mm": error_mm,
                "timestamp": row.get("timestamp", ""),
            }
        )

    errors_np = np.array(errors, dtype=np.float64)
    metrics = [
        ("count", float(len(errors_np))),
        ("mean_mm", float(np.mean(errors_np))),
        ("median_mm", float(np.median(errors_np))),
        ("std_mm", float(np.std(errors_np))),
        ("p95_mm", float(np.percentile(errors_np, 95))),
        ("max_mm", float(np.max(errors_np))),
    ]

    print("Mapping Error Summary")
    print("---------------------")
    for name, value in metrics:
        if name == "count":
            print(f"{name:<12} {int(value)}")
        else:
            print(f"{name:<12} {value:10.3f}")

    ensure_parent_dir(output_dir / "mapping_error_summary.csv")
    summary_csv = output_dir / "mapping_error_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["u", "v", "X", "Y", "Xhat", "Yhat", "dX", "dY", "error_mm", "timestamp"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    save_error_plots(
        summary_rows,
        output_dir,
        hist_png="mapping_error_histogram.png",
        vector_png="mapping_error_vectors.png",
    )
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved histogram : {output_dir / 'mapping_error_histogram.png'}")
    print(f"Saved vectors   : {output_dir / 'mapping_error_vectors.png'}")


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-id", type=int, default=DEFAULT_CAMERA_ID, help="OpenCV camera index")
    parser.add_argument(
        "--cam-width",
        type=int,
        default=DEFAULT_CAM_WIDTH,
        help="Requested camera width in pixels; calibration used 1280",
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=DEFAULT_CAM_HEIGHT,
        help="Requested camera height in pixels; calibration used 720",
    )
    parser.add_argument("--cam-fps", type=int, default=DEFAULT_CAM_FPS, help="Requested camera FPS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hand-eye homography overlay, collection, and evaluation utility")
    parser.add_argument("--parms-dir", type=str, default="save_parms", help="Directory containing homography files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    overlay_parser = subparsers.add_parser("overlay", help="Project robot-plane points into the live camera image")
    overlay_parser.add_argument("--points", type=str, required=True, help="Robot-plane points as X1,Y1;X2,Y2;...")
    add_camera_args(overlay_parser)
    overlay_parser.set_defaults(func=run_overlay)

    collect_parser = subparsers.add_parser("collect", help="Click image points and save pixel/robot correspondences")
    collect_parser.add_argument("--csv", type=str, default="save_parms/mapping_points.csv", help="CSV output path")
    collect_parser.add_argument(
        "--xy-list",
        type=str,
        default="",
        help="Optional predefined robot-plane list as X1,Y1;X2,Y2;... used in click order",
    )
    add_camera_args(collect_parser)
    collect_parser.set_defaults(func=run_collect)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate pixel-to-robot mapping accuracy from a CSV")
    evaluate_parser.add_argument("--csv", type=str, required=True, help="CSV created by collect mode")
    evaluate_parser.add_argument(
        "--output-dir",
        type=str,
        default="save_parms",
        help="Directory for mapping_error_summary.csv and plot PNG files",
    )
    evaluate_parser.set_defaults(func=run_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
