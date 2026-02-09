"""
Standalone gripper-servo test utility for Freenove arm.

Examples:
  python gripper_servo_test.py --dry-run --verbose
  python gripper_servo_test.py --mode open
  python gripper_servo_test.py --mode close --gripper-close 150
  python gripper_servo_test.py --mode toggle --cycles 3 --gripper-wait 0.5
  python gripper_servo_test.py --mode interactive
"""

from __future__ import annotations

import argparse

from freenove_arm import FreenoveArmClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test gripper servo (S9 I{index} A{angle}) using FreenoveArmClient")
    p.add_argument("--host", type=str, default="10.149.65.232", help="Robot IP address")
    p.add_argument("--port", type=int, default=5000, help="Robot TCP port")
    p.add_argument("--servo-index", type=int, default=1, help="Servo index (GPIO16 is typically index 1)")
    p.add_argument("--gripper-open", type=int, default=90.0, help="Open angle")
    p.add_argument("--gripper-close", type=int, default=10.0, help="Close angle (reset)")
    p.add_argument("--gripper-wait", type=float, default=0.4, help="Wait after each servo command (s)")
    p.add_argument("--ramp", action="store_true", help="Move servo gradually (slower open/close)")
    p.add_argument("--step-size", type=int, default=5, help="Angle increment per ramp step (deg)")
    p.add_argument("--step-wait", type=float, default=0.05, help="Wait per ramp step (s)")
    p.add_argument("--mode", choices=["open", "close", "toggle", "interactive"], default="toggle", help="Test mode")
    p.add_argument("--cycles", type=int, default=1, help="Number of open/close cycles for toggle mode")
    p.add_argument("--end-open", action="store_true", help="After toggle cycles, set gripper to open angle")
    p.add_argument("--skip-enable", action="store_true", help="Do not send motor enable command on connect")
    p.add_argument("--dry-run", action="store_true", help="Print commands instead of sending")
    p.add_argument("--verbose", action="store_true", help="Print every command sent")
    p.add_argument(
        "--home-first",
        dest="home_first",
        action="store_true",
        default=False,
        help="Send S10 F1 once after connecting",
    )
    p.add_argument(
        "--no-home-first",
        dest="home_first",
        action="store_false",
        help="Skip the homing move",
    )
    return p.parse_args()


def move_servo(arm: FreenoveArmClient, index: int, target: int, wait_s: float, tag: str, state) -> None:
    """Move servo, optionally in small steps to slow it down."""
    target = max(0, min(180, target))
    current = state["angle"]
    print(f"[gripper] {tag}: index={index} angle={current}->{target}")

    if not state["ramp"] or current is None or current == target:
        arm.set_servo(index, target)
        arm.wait(wait_s)
        state["angle"] = target
        return

    step = state["step_size"] if target > current else -state["step_size"]
    angle = current
    while (step > 0 and angle < target) or (step < 0 and angle > target):
        angle = angle + step
        if (step > 0 and angle > target) or (step < 0 and angle < target):
            angle = target
        arm.set_servo(index, int(angle))
        arm.wait(state["step_wait"])
    state["angle"] = target
    arm.wait(wait_s)


def run_interactive(arm: FreenoveArmClient, args: argparse.Namespace, state) -> None:
    print("Interactive mode commands: open | close | <angle 0-180> | q")
    while True:
        try:
            line = input("> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n[gripper] exit interactive")
            break

        if not line:
            continue
        if line in {"q", "quit", "exit"}:
            break
        if line == "open":
            move_servo(arm, args.servo_index, args.gripper_open, args.gripper_wait, "open", state)
            continue
        if line == "close":
            move_servo(arm, args.servo_index, args.gripper_close, args.gripper_wait, "close", state)
            continue

        try:
            angle = int(line)
        except ValueError:
            print("[gripper] invalid input")
            continue

        move_servo(arm, args.servo_index, angle, args.gripper_wait, "set", state)


def main() -> None:
    args = parse_args()
    with FreenoveArmClient(
        host=args.host,
        port=args.port,
        dry_run=args.dry_run,
        auto_enable=not args.skip_enable,
        verbose=args.verbose,
    ) as arm:
        print("[gripper] connected")
        state = {
            "angle": None,
            "ramp": args.ramp,
            "step_size": max(1, args.step_size),
            "step_wait": max(0.0, args.step_wait),
        }

        if args.home_first:
            print("[gripper] home_first")
            arm.return_to_sensor_point(1)
            arm.wait(0.5)

        # Reset then open to 90°, wait 3s for visual confirmation.
        # Startup sequence: close -> wait -> open -> wait -> close
        move_servo(arm, args.servo_index, args.gripper_close, args.gripper_wait, "startup_close", state)
        arm.wait(3.0)
        move_servo(arm, args.servo_index, args.gripper_open, args.gripper_wait, "startup_open_90", state)
        arm.wait(3.0)
        move_servo(arm, args.servo_index, args.gripper_close, args.gripper_wait, "startup_close_again", state)

        if args.mode == "open":
            send_angle(arm, args.servo_index, args.gripper_open, args.gripper_wait, "open")
            return

        if args.mode == "close":
            move_servo(arm, args.servo_index, args.gripper_close, args.gripper_wait, "close", state)
            return

        if args.mode == "interactive":
            run_interactive(arm, args, state)
            return

        cycles = max(1, args.cycles)
        for i in range(cycles):
            print(f"[gripper] cycle {i + 1}/{cycles}")
            move_servo(arm, args.servo_index, args.gripper_open, args.gripper_wait, "open", state)
            move_servo(arm, args.servo_index, args.gripper_close, args.gripper_wait, "close", state)

        if args.end_open:
            move_servo(arm, args.servo_index, args.gripper_open, args.gripper_wait, "end_open", state)

        print("[gripper] done")


if __name__ == "__main__":
    main()
