"""Reading the raw CosmicWatch detector files.

Two traps in this format, neither of them deducible from the original code:

1. **The header lies about column order.** It declares
   `SiPM[mV] Temp[C] Pressure[Pa]`, but the data carry pressure (~93600 Pa)
   BEFORE temperature (~28 C). The original script compensated with a
   per-row magnitude heuristic (thresholds 1000 and 200), undocumented and
   leaving no trace. Here the decision is made once per file, against declared
   physical ranges, and is recorded in the quality report.

2. **Malformed lines really do occur.** `data/raw/015deg` is truncated at the
   top: it lost the first lines of the preamble and its first line is the
   fragment ` 118  | Hexa address:  0x76`. The original script ran
   `except Exception: continue` and dropped it silently. Here it is counted,
   its text is kept, and it is reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .provenance import portable_path

# The header the files declare. Kept verbatim so it can be contrasted, and so
# there is a record that it does not describe the data.
DECLARED_HEADER = (
    "Event TimeStamp[ms] ADC1 ADC2 SiPM[mV] Temp[C] Pressure[Pa] "
    "DeadTime[us] Coincident Name"
)

# Actual order of columns 6 and 7, verified against all eight files.
ACTUAL_HEADER = (
    "Event TimeStamp[ms] ADC1 ADC2 SiPM[mV] Pressure[Pa] Temp[C] "
    "DeadTime[us] Coincident Name"
)

N_COLUMNS = 10

# Admissible physical ranges, used both to decide the order of columns 6 and 7
# and to flag out-of-range readings. Sea-level pressure is 101325 Pa; this set
# sits near 93600 Pa, consistent with moderate altitude.
PRESSURE_RANGE_PA = (50_000.0, 110_000.0)
TEMPERATURE_RANGE_C = (-20.0, 60.0)

# Reading the BMP280 emits before settling. It appears in 8 rows spread over
# five of the eight files, always at the start of the run.
STARTUP_PRESSURE_PA = 73157.02
STARTUP_TEMPERATURE_C = 23.13
STARTUP_TOLERANCE = 0.01



@dataclass
class QualityReport:
    """What went wrong while reading a file. Reported, never discarded."""

    path: str
    total_lines: int = 0
    data_lines: int = 0
    rejected_lines: int = 0
    rejected_texts: list[str] = field(default_factory=list)
    header_present: bool = False
    columns_swapped: bool = False
    sensor_startup_rows: int = 0
    sensor_startup_indices: list[int] = field(default_factory=list)
    out_of_range_rows: int = 0

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "total_lines": self.total_lines,
            "data_lines": self.data_lines,
            "rejected_lines": self.rejected_lines,
            "rejected_texts": self.rejected_texts,
            "header_present": self.header_present,
            "columns_swapped": self.columns_swapped,
            "sensor_startup_rows": self.sensor_startup_rows,
            "sensor_startup_indices": self.sensor_startup_indices,
            "out_of_range_rows": self.out_of_range_rows,
        }


@dataclass
class Run:
    """One run: one file, one nominal angle."""

    angle_deg: int
    path: str
    event: np.ndarray
    timestamp_ms: np.ndarray
    adc1: np.ndarray
    adc2: np.ndarray
    sipm_mv: np.ndarray
    pressure_pa: np.ndarray
    temperature_c: np.ndarray
    deadtime_us: np.ndarray
    coincident: np.ndarray
    sensor_startup: np.ndarray
    quality: QualityReport

    def __len__(self) -> int:
        return len(self.timestamp_ms)

    @property
    def elapsed_s(self) -> np.ndarray:
        """Time since the first event, in seconds.

        The original script accumulated positive differences with `cumsum`,
        which is equivalent to this unless the clock runs backwards. It never
        does in any of the eight files (checked in 01_validation), and plain
        subtraction is the honest definition: a backwards clock is something
        you want to see, not absorb silently.
        """
        return (self.timestamp_ms - self.timestamp_ms[0]) / 1000.0

    @property
    def window_s(self) -> float:
        """Duration of the observed window [t1, tN], in seconds."""
        return float(self.elapsed_s[-1])

    @property
    def dead_time_s(self) -> float:
        """Accumulated dead time of the run, in seconds.

        The DeadTime[us] column was parsed by the original script and never
        used: the published rate was wall-clock, not live-time corrected.
        """
        return float(self.deadtime_us.sum()) / 1e6

    @property
    def live_time_s(self) -> float:
        return self.window_s - self.dead_time_s

    @property
    def dead_fraction(self) -> float:
        return self.dead_time_s / self.window_s


def parse_line(line: str) -> tuple:
    """Parse one data line. Raises ValueError if it is not one.

    Returns the ten columns in their literal file order, without yet deciding
    which of the two middle ones is pressure and which is temperature: that
    decision is made per file in `load_run`.
    """
    parts = re.split(r"\s+", line.strip())
    if len(parts) != N_COLUMNS:
        raise ValueError(f"expected {N_COLUMNS} columns, found {len(parts)}")
    return (
        int(parts[0]),
        int(parts[1]),
        int(parts[2]),
        int(parts[3]),
        float(parts[4]),
        float(parts[5]),
        float(parts[6]),
        float(parts[7]),
        int(parts[8]),
        parts[9],
    )


def _decide_column_order(col6: np.ndarray, col7: np.ndarray) -> bool:
    """Decide whether columns 6 and 7 arrive as (pressure, temperature).

    Returns True if they are swapped with respect to the declared header, i.e.
    if column 6 holds the pressure. The decision uses the median over the whole
    file rather than row by row: one anomalous row must not be able to change
    how the columns are interpreted.
    """
    m6, m7 = float(np.median(col6)), float(np.median(col7))
    six_is_pressure = (
        PRESSURE_RANGE_PA[0] <= m6 <= PRESSURE_RANGE_PA[1]
        and TEMPERATURE_RANGE_C[0] <= m7 <= TEMPERATURE_RANGE_C[1]
    )
    seven_is_pressure = (
        PRESSURE_RANGE_PA[0] <= m7 <= PRESSURE_RANGE_PA[1]
        and TEMPERATURE_RANGE_C[0] <= m6 <= TEMPERATURE_RANGE_C[1]
    )
    if six_is_pressure and not seven_is_pressure:
        return True
    if seven_is_pressure and not six_is_pressure:
        return False
    raise ValueError(
        "cannot decide which column is pressure and which is temperature: "
        f"medians col6 = {m6:.2f}, col7 = {m7:.2f}. Admissible ranges "
        f"pressure {PRESSURE_RANGE_PA} Pa, temperature {TEMPERATURE_RANGE_C} C."
    )


def load_run(path: str | Path, angle_deg: int | None = None) -> Run:
    """Read one raw file and return a Run with its quality report."""
    path = Path(path)
    if angle_deg is None:
        matched = re.match(r"^(\d+)deg$", path.name)
        if not matched:
            raise ValueError(
                f"cannot infer the angle from name {path.name!r}; "
                "expected NNNdeg (for example 045deg)"
            )
        angle_deg = int(matched.group(1))

    quality = QualityReport(path=portable_path(path))
    rows: list[tuple] = []

    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            quality.total_lines += 1
            bare = line.strip()
            if not bare:
                continue
            if bare.startswith("#"):
                if "TimeStamp" in bare and "Coincident" in bare:
                    quality.header_present = True
                continue
            try:
                rows.append(parse_line(line))
            except ValueError:
                quality.rejected_lines += 1
                if len(quality.rejected_texts) < 10:
                    quality.rejected_texts.append(bare)

    if not rows:
        raise ValueError(f"no readable data line in {path}")
    quality.data_lines = len(rows)

    event = np.array([r[0] for r in rows], dtype=np.int64)
    timestamp_ms = np.array([r[1] for r in rows], dtype=np.float64)
    adc1 = np.array([r[2] for r in rows], dtype=np.int64)
    adc2 = np.array([r[3] for r in rows], dtype=np.int64)
    sipm_mv = np.array([r[4] for r in rows], dtype=np.float64)
    col6 = np.array([r[5] for r in rows], dtype=np.float64)
    col7 = np.array([r[6] for r in rows], dtype=np.float64)
    deadtime_us = np.array([r[7] for r in rows], dtype=np.float64)
    coincident = np.array([r[8] for r in rows], dtype=np.int64)

    quality.columns_swapped = _decide_column_order(col6, col7)
    if quality.columns_swapped:
        pressure_pa, temperature_c = col6, col7
    else:
        pressure_pa, temperature_c = col7, col6

    startup = (
        np.abs(pressure_pa - STARTUP_PRESSURE_PA) < STARTUP_TOLERANCE
    ) & (np.abs(temperature_c - STARTUP_TEMPERATURE_C) < STARTUP_TOLERANCE)
    quality.sensor_startup_rows = int(startup.sum())
    quality.sensor_startup_indices = [int(i) for i in np.flatnonzero(startup)]

    out_of_range = (
        (pressure_pa < PRESSURE_RANGE_PA[0])
        | (pressure_pa > PRESSURE_RANGE_PA[1])
        | (temperature_c < TEMPERATURE_RANGE_C[0])
        | (temperature_c > TEMPERATURE_RANGE_C[1])
    )
    quality.out_of_range_rows = int(out_of_range.sum())

    return Run(
        angle_deg=angle_deg,
        path=portable_path(path),
        event=event,
        timestamp_ms=timestamp_ms,
        adc1=adc1,
        adc2=adc2,
        sipm_mv=sipm_mv,
        pressure_pa=pressure_pa,
        temperature_c=temperature_c,
        deadtime_us=deadtime_us,
        coincident=coincident,
        sensor_startup=startup,
        quality=quality,
    )


def load_all(directory: str | Path = "data/raw") -> dict[int, Run]:
    """Read the eight runs from a directory, keyed by nominal angle."""
    directory = Path(directory)
    runs = {}
    for path in sorted(directory.iterdir()):
        if re.match(r"^\d+deg$", path.name):
            run = load_run(path)
            runs[run.angle_deg] = run
    if not runs:
        raise ValueError(f"no NNNdeg files in {directory}")
    return runs
