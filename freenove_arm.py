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

import queue
import re
import threading
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
    _recv_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _recv_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _parse_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _workers_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _feedback_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _send_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _rx_chunks: "queue.Queue[str]" = field(default_factory=queue.Queue, init=False, repr=False)
    _rx_buffer: str = field(default="", init=False, repr=False)
    _queue_feedback_enabled: bool = field(default=False, init=False, repr=False)
    _queue_len_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _action_queue_len: Optional[int] = field(default=None, init=False, repr=False)
    _queue_len_updated_at: float = field(default=0.0, init=False, repr=False)
    _last_queue_action_send_at: float = field(default=0.0, init=False, repr=False)

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
        client = self._client
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass
        stopped = self._stop_receive_workers()
        if not stopped and self.verbose:
            print("[warn] feedback workers did not stop cleanly during close()")
        self._client = None

    # Command primitives --------------------------------------------------
    def _send(self, text: str) -> None:
        """Send a raw command line, adding CRLF to match the app behaviour."""

        if self.verbose or self.dry_run:
            prefix = "[dry-run]" if self.dry_run else "[send]"
            print(f"{prefix} {text}")
        if self.dry_run:
            return

        with self._send_lock:
            if self._client is None or not self._client.connect_flag:
                self.connect()
            if self._client is None or not self._client.connect_flag:
                raise RuntimeError("Socket is not connected")
            self._client.send_messages(text + "\r\n")
            if self._command_affects_action_queue(text):
                self._last_queue_action_send_at = time.monotonic()

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

    # Feedback helpers ----------------------------------------------------
    def set_action_feedback(self, enable: bool = True) -> None:
        """
        Enable/disable S12 queue-length feedback from the Raspberry Pi server.
        When enabled, parse incoming ``S12 K<n>`` messages and cache ``n``.
        """

        with self._feedback_lock:
            if self.dry_run:
                self._queue_feedback_enabled = False
                return

            if enable:
                if self._queue_feedback_enabled:
                    self._ensure_receive_workers()
                    return
                self._ensure_receive_workers()
                with self._queue_len_lock:
                    self._action_queue_len = None
                    self._queue_len_updated_at = 0.0
                self._send(f"{self._cmd.CUSTOM_ACTION}12 {self._cmd.ARM_QUERY}1")
                self._queue_feedback_enabled = True
                return

            if self._queue_feedback_enabled and self._client is not None and self._client.connect_flag:
                try:
                    self._send(f"{self._cmd.CUSTOM_ACTION}12 {self._cmd.ARM_QUERY}0")
                except Exception:
                    # The connection may already be down during shutdown paths.
                    pass
            self._queue_feedback_enabled = False
            stopped = self._stop_receive_workers()
            with self._queue_len_lock:
                self._action_queue_len = None
                self._queue_len_updated_at = 0.0
            if not stopped:
                raise RuntimeError("Failed to stop feedback receive workers cleanly")

    def get_action_queue_len(self) -> Optional[int]:
        with self._queue_len_lock:
            return self._action_queue_len

    def wait_action_queue_empty(
        self,
        timeout: float = 2.0,
        min_empty_time: float = 0.05,
        poll_interval: float = 0.02,
    ) -> bool:
        """
        Wait until the server-reported queue length is 0 for a short stable period.
        Queue data is considered valid only if at least one feedback update was
        received after the most recent action-queue command send time.
        Returns True on success and False on timeout/unavailable feedback.
        """

        if not self._queue_feedback_enabled:
            if self.verbose:
                print("[sync] action feedback is disabled")
            return False

        deadline = time.monotonic() + max(0.0, timeout)
        required_feedback_after = self._last_queue_action_send_at
        empty_since = None
        saw_fresh_feedback = False
        while time.monotonic() < deadline:
            with self._queue_len_lock:
                qlen = self._action_queue_len
                updated_at = self._queue_len_updated_at
            if qlen is not None and updated_at > required_feedback_after:
                saw_fresh_feedback = True
                if qlen == 0:
                    if empty_since is None:
                        empty_since = time.monotonic()
                    elif (time.monotonic() - empty_since) >= max(0.0, min_empty_time):
                        return True
                else:
                    empty_since = None
            time.sleep(max(0.0, poll_interval))

        if self.verbose:
            if not saw_fresh_feedback:
                print("[sync] timeout waiting for fresh S12 K feedback")
            else:
                print("[sync] timeout waiting for queue empty")
        return False

    def send_raw(self, text: str) -> None:
        """Send a raw protocol line, e.g. ``S9 I1 A90``."""

        self._send(text.strip())

    def _ensure_receive_workers(self) -> None:
        if self.dry_run:
            return
        if self._client is None or not self._client.connect_flag:
            self.connect()
        if self._client is None:
            raise RuntimeError("Socket is not connected")

        with self._workers_lock:
            recv_alive = self._recv_thread is not None and self._recv_thread.is_alive()
            parse_alive = self._parse_thread is not None and self._parse_thread.is_alive()
            if recv_alive and parse_alive:
                return
            if recv_alive or parse_alive:
                # Inconsistent worker state: refuse to start another reader on the same socket.
                raise RuntimeError("Feedback workers are in inconsistent state; stop them before restart")

            self._recv_stop.clear()
            self._clear_rx_chunks()
            self._rx_buffer = ""
            client = self._client
            self._recv_thread = threading.Thread(
                target=self._recv_loop,
                args=(client,),
                daemon=True,
                name="arm-rx",
            )
            self._parse_thread = threading.Thread(target=self._parse_feedback_loop, daemon=True, name="arm-rx-parse")
            self._recv_thread.start()
            self._parse_thread.start()

    def _stop_receive_workers(self) -> bool:
        with self._workers_lock:
            self._recv_stop.set()
            try:
                self._rx_chunks.put_nowait("")
            except Exception:
                pass
            for thread in (self._recv_thread, self._parse_thread):
                if thread is not None and thread.is_alive():
                    thread.join(timeout=2.0)
            recv_alive = self._recv_thread is not None and self._recv_thread.is_alive()
            parse_alive = self._parse_thread is not None and self._parse_thread.is_alive()
            if not recv_alive:
                self._recv_thread = None
            if not parse_alive:
                self._parse_thread = None
            self._queue_feedback_enabled = False
            return not (recv_alive or parse_alive)

    def _recv_loop(self, client: Client) -> None:
        try:
            client.receive_messages(stop_event=self._recv_stop, on_data=self._on_rx_chunk, timeout_s=0.2)
        except Exception as exc:
            if self.verbose:
                print(f"[recv] receive loop exited with error: {exc}")
        finally:
            self._recv_stop.set()
            try:
                self._rx_chunks.put_nowait("")
            except Exception:
                pass

    def _on_rx_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._rx_chunks.put(chunk)

    def _parse_feedback_loop(self) -> None:
        pattern = re.compile(r"^S12\s+K(-?\d+)$")
        while not self._recv_stop.is_set():
            try:
                payload = self._rx_chunks.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                continue
            self._rx_buffer += payload
            self._rx_buffer = self._rx_buffer.replace("\r\n", "\n").replace("\r", "\n")
            while "\n" in self._rx_buffer:
                line, self._rx_buffer = self._rx_buffer.split("\n", 1)
                text = line.strip()
                if not text:
                    continue
                if self.verbose:
                    print(f"[recv] {text}")
                match = pattern.match(text)
                if match:
                    with self._queue_len_lock:
                        self._action_queue_len = int(match.group(1))
                        self._queue_len_updated_at = time.monotonic()
        self._rx_buffer = ""

    def _clear_rx_chunks(self) -> None:
        while True:
            try:
                self._rx_chunks.get_nowait()
            except queue.Empty:
                break

    def _command_affects_action_queue(self, text: str) -> bool:
        head = text.strip().split(" ", 1)[0].upper()
        return head in {"G0", "G1", "G4", "S9", "S10", "S11"}

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
