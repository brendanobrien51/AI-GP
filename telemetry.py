"""
Lap Telemetry System
====================
Tracks per-gate timing, missed gates, and CV detection rates.
Outputs CSV for human analysis and JSON for optimizer feedback.
"""

import csv
import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GateRecord:
    """Record of one gate traversal attempt."""
    gate_index: int
    gate_name: str
    entry_time: float           # wall-clock time when APPROACH started
    exit_time: Optional[float] = None  # wall-clock time when EXIT completed
    missed: bool = False        # timed out without a clean exit
    retries: int = 0
    max_speed_achieved: float = 0.0
    cv_detection_rate: float = 0.0    # fraction of frames with valid detection


class LapTelemetry:
    """
    Collects per-lap and per-gate statistics for one racing lap.
    Designed for human analysis (CSV) and automated optimization (JSON).
    """

    def __init__(self, total_gates: int, csv_path: str = "cv_race_log.csv"):
        self._csv_path = csv_path
        self._total_gates = total_gates  # Fixed expected gate count (not attempted)
        self._lap_start: Optional[float] = None
        self._lap_end: Optional[float] = None
        self._gates: list[GateRecord] = []
        self._current_gate: Optional[GateRecord] = None
        self._frame_count: int = 0
        self._detection_count: int = 0

        self._fieldnames = [
            "gate_index", "gate_name", "split_time_s", "exit_time_s",
            "missed", "retries", "max_speed_m_s", "cv_detection_rate"
        ]

    def lap_start(self) -> None:
        """Call once at beginning of run."""
        self._lap_start = time.time()
        self._gates.clear()
        self._frame_count = 0
        self._detection_count = 0

    def gate_start(self, gate_index: int, gate_name: str) -> None:
        """Call when APPROACH phase begins for a gate."""
        self._current_gate = GateRecord(
            gate_index=gate_index,
            gate_name=gate_name,
            entry_time=time.time(),
            max_speed_achieved=0.0,
        )

    def gate_exit(self) -> None:
        """Call when EXIT phase completes (gate passed cleanly)."""
        if self._current_gate is None:
            return
        self._current_gate.exit_time = time.time()
        self._current_gate.missed = False
        self._update_detection_rate()
        self._gates.append(self._current_gate)
        self._current_gate = None

    def gate_missed(self) -> None:
        """Call on timeout or max-retries skip."""
        if self._current_gate is None:
            return
        self._current_gate.missed = True
        self._update_detection_rate()
        self._gates.append(self._current_gate)
        self._current_gate = None

    def update_frame(self, speed: float, detection_found: bool) -> None:
        """Call every control loop tick to update current gate stats."""
        if self._current_gate is None:
            return

        self._frame_count += 1
        if detection_found:
            self._detection_count += 1

        if speed > self._current_gate.max_speed_achieved:
            self._current_gate.max_speed_achieved = speed

    def _update_detection_rate(self) -> None:
        """Compute CV detection rate for current gate."""
        if self._current_gate is not None:
            if self._frame_count > 0:
                self._current_gate.cv_detection_rate = (
                    self._detection_count / self._frame_count
                )
            # Reset counters for next gate
            self._frame_count = 0
            self._detection_count = 0

    def lap_end(self) -> None:
        """Call when course complete."""
        self._lap_end = time.time()

    @property
    def lap_time(self) -> float:
        """Total elapsed lap time in seconds. Returns inf if not finished."""
        if self._lap_start is None or self._lap_end is None:
            return float("inf")
        return self._lap_end - self._lap_start

    @property
    def missed_gates(self) -> int:
        """Count of gates that were skipped (timeout or max retries)."""
        return sum(1 for g in self._gates if g.missed)

    @property
    def gates_completed(self) -> int:
        """Count of gates passed cleanly."""
        return len(self._gates) - self.missed_gates

    def save_csv(self) -> None:
        """Append a summary row per gate to the CSV file."""
        try:
            # Check if file exists to determine if we need to write header
            try:
                with open(self._csv_path, "r") as f:
                    existing = f.read()
                    need_header = len(existing) == 0
            except FileNotFoundError:
                need_header = True

            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)

                if need_header:
                    writer.writeheader()

                for gate in self._gates:
                    if gate.exit_time is not None:
                        split_time = gate.exit_time - gate.entry_time
                    else:
                        split_time = None

                    writer.writerow({
                        "gate_index": gate.gate_index,
                        "gate_name": gate.gate_name,
                        "split_time_s": split_time,
                        "exit_time_s": gate.exit_time,
                        "missed": "yes" if gate.missed else "no",
                        "retries": gate.retries,
                        "max_speed_m_s": f"{gate.max_speed_achieved:.2f}",
                        "cv_detection_rate": f"{gate.cv_detection_rate:.2%}",
                    })
        except Exception as e:
            print(f"Warning: Failed to save telemetry CSV: {e}")

    def summary_dict(self) -> dict:
        """Return a dict suitable for JSON serialization (optimizer feedback)."""
        # Calculate additional metrics for optimizer
        total_retries = sum(g.retries for g in self._gates)
        avg_cv_detection = 0.0
        if self._gates:
            valid_detections = [g.cv_detection_rate for g in self._gates if g.cv_detection_rate > 0]
            if valid_detections:
                avg_cv_detection = sum(valid_detections) / len(valid_detections)

        return {
            "lap_time": self.lap_time if not math.isinf(self.lap_time) else None,
            "missed_gates": self.missed_gates,
            "gates_completed": self.gates_completed,
            "total_gates": self._total_gates,  # Fixed: use expected gate count, not attempted
            "total_retries": total_retries,
            "avg_cv_detection_rate": avg_cv_detection,
        }

    def to_json(self, path: str) -> None:
        """Save summary to JSON file."""
        try:
            with open(path, "w") as f:
                json.dump(self.summary_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save telemetry JSON: {e}")


# Import math for summary_dict
import math
