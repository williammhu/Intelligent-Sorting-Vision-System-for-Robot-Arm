"""
Freenove robot arm client that speaks the same TCP text protocol as the official app.

The Freenove desktop UI (see ``main.py``/``client.py``) sends simple ASCII
commands such as ``G0 X0 Y200 Z45`` to port 5000.  This wrapper reuses the
existing `Client` class so calibration scripts can drive the arm directly
without duplicating socket handling code.  When ``dry_run=True`` commands are
printed instead of sent, which is handy for debugging on a machine without the
arm attached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from client import Client
from command import Command


@dataclass
class FreenoveArmClient:
    host: str = "10.149.65.232"
    port: int = 5000
    dry_run: bool = False
    auto_enable: bool = True
    verbose: bool = False

    _client: Optional[Client] = field(default=None, init=False, repr=False)
    _cmd: Command = field(default_factory=Command, init=False, repr=False)

    def __enter__(self) -> "FreenoveArmClient":
        self.connect()
        if self.auto_enable:
            self.enable_motors(True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # Connection helpers -------------------------------------------------
    def connect(self) -> None:
        """Open the TCP connection unless running in dry-run mode."""

        if self.dry_run:
            return
        if self._client is None:
            self._client = Client()
            self._client.port = self.port
            if not self._client.connect(self.host):
                self._client = None
                raise RuntimeError(f"Could not connect to Freenove arm at {self.host}:{self.port}")

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            finally:
                self._client = None

    # Command primitives --------------------------------------------------
    def _send(self, text: str) -> None:
        """Send a raw command line, adding CRLF to match the app behaviour."""

        if self.verbose or self.dry_run:
            prefix = "[dry-run]" if self.dry_run else "[send]"
            print(f"{prefix} {text}")
        if self.dry_run:
            return

        if self._client is None or not self._client.connect_flag:
            self.connect()
        if self._client is None or not self._client.connect_flag:
            raise RuntimeError("Socket is not connected")
        self._client.send_messages(text + "\r\n")

    def enable_motors(self, enable: bool = True) -> None:
        """
        Mirror the UI \"Load Motor\" toggle.
        From the stock UI (see freenove_source_code/main.py): S8 E0 = load/enable, S8 E1 = relax.
        """

        state = "0" if enable else "1"
        cmd = f"{self._cmd.CUSTOM_ACTION}8 {self._cmd.ARM_ENABLE}{state}"
        self._send(cmd)

    def move_to(self, x: float, y: float, z: float, speed: int | None = None, dwell_ms: int | None = None) -> None:
        """
        Absolute move in millimetres.

        The native protocol encodes position as ``G0 X.. Y.. Z..``.  The official
        instructions pair this with ``G4 T<ms>`` to pause after the move; pass
        ``dwell_ms`` to emit the pause directly instead of sleeping on the
        client.
        """

        cmd = (
            f"{self._cmd.MOVE_ACTION}0 "
            f"{self._cmd.AXIS_X_ACTION}{x:.1f} "
            f"{self._cmd.AXIS_Y_ACTION}{y:.1f} "
            f"{self._cmd.AXIS_Z_ACTION}{z:.1f}"
        )
        self._send(cmd)
        if dwell_ms is not None:
            self._send(f"{self._cmd.MOVE_ACTION}4 {self._cmd.DELAY_T_ACTION}{dwell_ms}")

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def send_raw(self, text: str) -> None:
        """Send a raw protocol line, e.g. ``S9 I1 A90``."""

        self._send(text.strip())

    # Utility commands mirroring the official control table ----------------
    def return_to_sensor_point(self, index: int = 0) -> None:
        """Return the arm to its magnetic/sensor origin (``S10 F{index}``)."""

        self._send(f"{self._cmd.CUSTOM_ACTION}10 {self._cmd.ARM_SENSOR_POINT}{index}")

    def set_home(self, x: float, y: float, z: float) -> None:
        """Store a home location with ``S5 X.. Y.. Z..``."""

        cmd = (
            f"{self._cmd.CUSTOM_ACTION}5 "
            f"{self._cmd.AXIS_X_ACTION}{x:.1f} "
            f"{self._cmd.AXIS_Y_ACTION}{y:.1f} "
            f"{self._cmd.AXIS_Z_ACTION}{z:.1f}"
        )
        self._send(cmd)

    def set_ground_clearance(self, height_mm: float) -> None:
        """Set the ground clearance/padding height (``S3 O..``)."""

        self._send(f"{self._cmd.CUSTOM_ACTION}3 {self._cmd.GROUND_HEIGHT}{height_mm:.1f}")

    def set_clamp_length(self, length_mm: float) -> None:
        """Configure gripper length (``S4 L..``)."""

        self._send(f"{self._cmd.CUSTOM_ACTION}4 {self._cmd.CLAMP_LENGTH}{length_mm:.1f}")

    def set_frequency(self, frequency_hz: int) -> None:
        """Adjust pulse frequency with ``S6 Q..``."""

        self._send(f"{self._cmd.CUSTOM_ACTION}6 {self._cmd.ARM_FREQUENCY}{frequency_hz}")

    def set_microstep(self, resolution: int) -> None:
        """Configure microstep resolution via ``S7 W..``."""

        self._send(f"{self._cmd.CUSTOM_ACTION}7 {self._cmd.ARM_MSX}{resolution}")

    def set_servo(self, index: int, angle: float) -> None:
        """Set servo angle via ``S9 I{index} A{angle}``."""

        safe_index = max(0, min(4, int(index)))
        safe_angle = max(0, min(180, int(round(angle))))
        self._send(
            f"{self._cmd.CUSTOM_ACTION}9 "
            f"{self._cmd.ARM_SERVO_INDEX}{safe_index} "
            f"{self._cmd.ARM_SERVO_ANGLE}{safe_angle}"
        )

    def beep(self, frequency_hz: int) -> None:
        """Play a buzzer tone (``S2 D..``)."""

        self._send(f"{self._cmd.CUSTOM_ACTION}2 {self._cmd.BUZZER_ACTION}{frequency_hz}")

    def stop_buzzer(self) -> None:
        """Silence the buzzer (``S2 D0``)."""

        self._send(f"{self._cmd.CUSTOM_ACTION}2 {self._cmd.BUZZER_ACTION}0")
