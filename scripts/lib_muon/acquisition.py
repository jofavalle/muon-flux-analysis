"""Reading the long, multi-acquisition files of the 2026 altitude campaign.

`io.load_run` reads the 2025 angular runs: one short acquisition per file, one
nominal angle per file. The 2026 files break three of its assumptions, and the
2025 reader is left untouched on purpose (its products are published as _v1
and must keep reproducing), so they get a reader of their own:

1. **A file holds several acquisitions.** Every time the detector is unplugged
   and plugged back in it prints a new preamble and restarts both the event
   counter and the clock. `tacachico` holds three. A single monotonic clock
   across the file would give windows that are wrong or even negative. Here a
   new acquisition starts wherever the event counter goes back.
2. **A comment can be glued to a data row.** When the restart happens in the
   middle of a line, the first preamble line ends up after the last event:
   `... 0 COSMIC# Temperature 23.13C`. The nine numeric fields are complete,
   so the row is kept and the comment is cut and recorded.
3. **Rows can be incomplete or fused.** A row with fewer fields, or two rows
   fused into one line, is rejected, counted and its text kept. No field is
   ever invented.

The acceptance rule is: after cutting a glued comment, ten whitespace-separated
fields, the first nine numeric and the tenth starting with `COSMIC`. It is the
same rule the analysis presented at the International Cosmic Day 2026 used, so
the counts here are the counts quoted there.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from .io import (
    N_COLUMNS,
    STARTUP_PRESSURE_PA,
    STARTUP_TEMPERATURE_C,
    STARTUP_TOLERANCE,
    _decide_column_order,
)
from .provenance import portable_path

# Rejected-line texts kept per file. The count is always complete.
MAX_REJECTED_TEXTS = 20


@dataclass
class Acquisition:
    """One acquisition: from a detector start to the next restart."""

    index: int
    event: np.ndarray           # event counter as printed
    time_s: np.ndarray          # seconds since the detector's zero
    adc1: np.ndarray            # low-gain channel, ADC counts
    adc2: np.ndarray            # high-gain channel, ADC counts
    sipm_mv: np.ndarray         # computed by the firmware from ADC1 and ADC2
    pressure_pa: np.ndarray
    temperature_c: np.ndarray   # inside the detector case
    deadtime_s: np.ndarray      # dead time accumulated since the previous event
    coincident: np.ndarray
    line: np.ndarray            # line number in the file, from 1

    def __len__(self) -> int:
        return len(self.event)

    def select(self, mask: np.ndarray) -> "Acquisition":
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return Acquisition(**{k: (v if k == "index" else v[mask]) for k, v in values.items()})

    def window(self, t0_s: float, t1_s: float) -> "Acquisition":
        """Events with t0 <= t < t1, in seconds since the detector's zero."""
        return self.select((self.time_s >= t0_s) & (self.time_s < t1_s))

    def counter_jumps(self) -> list[tuple[int, int]]:
        """Pairs (before, after) where the counter does not advance by exactly one."""
        steps = np.diff(self.event)
        return [(int(self.event[i]), int(self.event[i + 1]))
                for i in np.flatnonzero(steps != 1)]


@dataclass
class AcquisitionFile:
    path: str
    sha256: str
    total_lines: int
    comment_lines: int = 0
    rejected_lines: int = 0
    rejected_texts: list[tuple[int, str]] = field(default_factory=list)
    glued_comment_lines: list[int] = field(default_factory=list)
    columns_swapped: bool = False
    sensor_startup_rows: int = 0
    acquisitions: list[Acquisition] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return sum(len(a) for a in self.acquisitions)

    def summary(self) -> dict:
        """Plain dictionary for the products: counts and positions, no arrays."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "total_lines": self.total_lines,
            "comment_lines": self.comment_lines,
            "data_rows": self.n_rows,
            "rejected_lines": self.rejected_lines,
            "rejected_texts": [[n, t] for n, t in self.rejected_texts],
            "glued_comment_lines": self.glued_comment_lines,
            "columns_swapped": self.columns_swapped,
            "sensor_startup_rows": self.sensor_startup_rows,
            "acquisitions": [
                {
                    "index": a.index,
                    "rows": len(a),
                    "first_event": int(a.event[0]),
                    "last_event": int(a.event[-1]),
                    "first_time_s": float(a.time_s[0]),
                    "last_time_s": float(a.time_s[-1]),
                    "first_line": int(a.line[0]),
                    "counter_jumps": [list(j) for j in a.counter_jumps()],
                }
                for a in self.acquisitions
            ],
        }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_acquisitions(path: str | Path) -> AcquisitionFile:
    """Read a raw file, split it into acquisitions and report what was rejected."""
    path = Path(path)
    report = AcquisitionFile(path=portable_path(path), sha256=sha256(path), total_lines=0)
    rows: list[list[float]] = []
    lines: list[int] = []

    with open(path, "r", encoding="ascii", errors="replace") as handle:
        for number, raw in enumerate(handle, 1):
            report.total_lines = number
            bare = raw.strip()
            if not bare:
                continue
            if bare.startswith("#"):
                report.comment_lines += 1
                continue
            glued = "#" in bare
            if glued:
                bare = bare[: bare.index("#")]
            parts = bare.split()
            if len(parts) == N_COLUMNS and parts[9].startswith("COSMIC"):
                try:
                    rows.append([float(x) for x in parts[:9]])
                except ValueError:
                    pass
                else:
                    lines.append(number)
                    if glued:
                        report.glued_comment_lines.append(number)
                    continue
            report.rejected_lines += 1
            if len(report.rejected_texts) < MAX_REJECTED_TEXTS:
                report.rejected_texts.append((number, raw.strip()[:80]))

    if not rows:
        raise ValueError(f"no readable data row in {path}")
    table = np.array(rows, dtype=np.float64)
    line_numbers = np.array(lines, dtype=np.int64)

    report.columns_swapped = _decide_column_order(table[:, 5], table[:, 6])
    i_pressure, i_temperature = (5, 6) if report.columns_swapped else (6, 5)
    startup = (
        (np.abs(table[:, i_pressure] - STARTUP_PRESSURE_PA) < STARTUP_TOLERANCE)
        & (np.abs(table[:, i_temperature] - STARTUP_TEMPERATURE_C) < STARTUP_TOLERANCE)
    )
    report.sensor_startup_rows = int(startup.sum())

    # A new acquisition starts where the event counter goes back.
    starts = np.flatnonzero(np.diff(table[:, 0]) < 0) + 1
    for index, rows_of in enumerate(np.split(np.arange(len(table)), starts)):
        block = table[rows_of]
        report.acquisitions.append(Acquisition(
            index=index,
            event=block[:, 0].astype(np.int64),
            time_s=block[:, 1] / 1e3,
            adc1=block[:, 2].astype(np.int64),
            adc2=block[:, 3].astype(np.int64),
            sipm_mv=block[:, 4],
            pressure_pa=block[:, i_pressure],
            temperature_c=block[:, i_temperature],
            deadtime_s=block[:, 7] / 1e6,
            coincident=block[:, 8].astype(np.int64),
            line=line_numbers[rows_of],
        ))
    return report
