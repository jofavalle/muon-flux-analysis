"""Time binning, written once.

In the original script this logic appeared three times almost verbatim
(`plot_per_second`, `plot_time_vs_rate_all`, `compute_mean_rate`), so any fix
had to be applied three times. Worse: the third copy truncated the run duration
to a whole multiple of the bin size, which let a purely cosmetic parameter move
the published mean rate. Measured on the real data: the rate at 0 degrees goes
from 1.8118 1/s with 1 s bins to 1.8194 1/s with 60 s bins, and the ratio
I(90)/I(0) moves from 0.8079 to 0.8148. Here binning only serves to draw time
series and enters no physical number.
"""

from __future__ import annotations

import numpy as np


def complete_bins(
    times_s: np.ndarray, bin_s: float, start_s: float = 0.0,
    end_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count events in fixed-width bins, dropping the final partial bin.

    Returns (centres, counts, widths) with constant width equal to `bin_s`.
    The final partial bin is dropped because its apparent rate is inflated: a
    0.2 s bin holding one event reads 5 1/s. This affects only the time-series
    figures; physical rates are computed in `rates` over the full window,
    without binning.
    """
    if bin_s <= 0:
        raise ValueError(f"bin_s must be positive, got {bin_s}")
    if end_s is None:
        end_s = float(times_s[-1]) if len(times_s) else start_s

    inside = times_s[(times_s >= start_s) & (times_s <= end_s)]
    if len(inside) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    n_bins = int(np.floor((end_s - start_s) / bin_s))
    if n_bins < 1:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    edges = start_s + np.arange(n_bins + 1) * bin_s
    counts, _ = np.histogram(inside, bins=edges)
    centres = edges[:-1] + bin_s / 2.0
    widths = np.full(n_bins, bin_s, dtype=float)
    return centres, counts.astype(float), widths


def binned_rates(
    times_s: np.ndarray, bin_s: float, start_s: float = 0.0,
    end_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instantaneous rate time series with Poisson errors.

    Returns (centres, rate, error). The error of an empty bin is taken as
    1/width, the 68 % upper bound for a Poisson observing zero: returning zero
    would claim absolute certainty where there is no measurement.
    """
    centres, counts, widths = complete_bins(times_s, bin_s, start_s, end_s)
    if len(centres) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    rates = counts / widths
    errors = np.where(counts > 0, np.sqrt(counts), 1.0) / widths
    return centres, rates, errors


def moving_average(values: np.ndarray, window_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Centred moving average and local standard deviation.

    Replaces the pandas `rolling` of the original script. Edges use the
    truncated window available, as `min_periods=1` did.
    """
    window_bins = max(int(window_bins), 1)
    n = len(values)
    means = np.empty(n, dtype=float)
    deviations = np.empty(n, dtype=float)
    half = window_bins // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + window_bins - half)
        chunk = values[lo:hi]
        means[i] = float(np.mean(chunk))
        deviations[i] = float(np.std(chunk))
    return means, deviations
