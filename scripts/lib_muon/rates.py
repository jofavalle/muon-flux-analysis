"""Count rates corrected for live time.

Three corrections over the original script, all of which change the published
number:

1. **Live time.** The `DeadTime[us]` column was parsed and never used. The dead
   fraction is 9.09 % to 9.28 % across the eight files, so the published rate
   was systematically about 10 % low.
2. **Unbiased estimator.** For N events observed in the window [t1, tN] there
   are N-1 inter-event intervals, not N. The unbiased rate estimator is
   (N-1)/T, not N/T. The bias is 1/N, here 0.15 % to 0.22 %: small next to the
   statistics, but free to fix.
3. **No binning dependence.** The window is the full [t1, tN], not truncated to
   a multiple of the bin size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Run


@dataclass
class Rate:
    """Rate of one run, with everything needed to redo the computation."""

    angle_deg: int
    n_events: int
    n_intervals: int
    window_s: float
    dead_time_s: float
    live_time_s: float
    dead_fraction: float
    wall_rate: float
    live_rate: float
    live_rate_error: float

    def as_dict(self) -> dict:
        return {
            "angle_deg": self.angle_deg,
            "n_events": self.n_events,
            "n_intervals": self.n_intervals,
            "window_s": self.window_s,
            "dead_time_s": self.dead_time_s,
            "live_time_s": self.live_time_s,
            "dead_fraction": self.dead_fraction,
            "wall_rate_hz": self.wall_rate,
            "live_rate_hz": self.live_rate,
            "live_rate_error_hz": self.live_rate_error,
        }


def compute_rate(run: Run, coincident_only: bool = True) -> Rate:
    """Rate of one run, corrected for live time.

    `coincident_only` is kept for fidelity to the original analysis, but it is
    a no-op on this data: `Coincident == 1` for all 4760 events across the
    eight files. With no singles recorded, the accidental rate is not
    estimable from this set; see docs/dataset.md.
    """
    times = run.elapsed_s
    if coincident_only:
        times = times[run.coincident == 1]
    if len(times) < 2:
        raise ValueError(
            f"the {run.angle_deg} degree run has {len(times)} events: "
            "at least 2 are needed to define a rate"
        )

    n = len(times)
    window = float(times[-1] - times[0])
    dead = run.dead_time_s
    live = window - dead
    if live <= 0:
        raise ValueError(
            f"non-positive live time at {run.angle_deg} degrees: "
            f"window {window:.2f} s, dead time {dead:.2f} s"
        )

    n_intervals = n - 1
    live_rate = n_intervals / live
    # n_intervals is Poisson with mean R*T_live at fixed T_live, so the
    # relative error on the rate is 1/sqrt(n_intervals).
    error = live_rate / np.sqrt(n_intervals)

    return Rate(
        angle_deg=run.angle_deg,
        n_events=n,
        n_intervals=n_intervals,
        window_s=window,
        dead_time_s=dead,
        live_time_s=live,
        dead_fraction=dead / window,
        wall_rate=n_intervals / window,
        live_rate=live_rate,
        live_rate_error=error,
    )


def normalise(rates: dict[int, Rate], reference_angle: int = 0) -> dict[int, tuple[float, float]]:
    """Rates relative to the reference angle, with propagated errors.

    A warning the original script did not make: these ratios share a
    denominator, so they are NOT independent and a diagonal chi-square over
    them is ill-posed. They are useful for plotting and for comparing against
    the earlier analysis; inference is done in `models` on the raw counts,
    where the problem does not arise.
    """
    if reference_angle not in rates:
        raise ValueError(f"reference angle {reference_angle} missing from the rates")
    reference = rates[reference_angle]
    r0, e0 = reference.live_rate, reference.live_rate_error
    out = {}
    for angle, rate in rates.items():
        ratio = rate.live_rate / r0
        error = np.sqrt((rate.live_rate_error / r0) ** 2
                        + (rate.live_rate * e0 / r0**2) ** 2)
        out[angle] = (float(ratio), float(error))
    return out
