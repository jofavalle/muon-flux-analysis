"""What a single scintillator slab should see when it is tilted.

The cos^2(theta) law describes the flux per unit solid angle. A narrow
telescope of two detectors in coincidence selects one direction and so follows
it, down to zero at the horizon. A single slab does not: it counts every muon
that crosses it from any direction, and the rate is the flux integrated
against the area the slab presents to each direction,

    R(alpha) = integral over the upper hemisphere of I(theta) A_proj(alpha, Omega) dOmega,

with I(theta) proportional to cos^n(theta). For a box of sides a x b x c the
projected area along a unit vector d is the sum over its three face pairs,
A_proj = ab |d.e3| + bc |d.e1| + ac |d.e2|, where e1, e2, e3 are the box axes.

Two closed forms for n = 2 serve as tests: a thin plate falls to exactly 1/2
when vertical, and a 5 x 5 x 1 cm box to exactly 2/3 (its edges keep
catching vertical muons). Even a pure muon sample seen by one slab never comes
close to the zero that cos^2 predicts at 90 degrees.
"""

from __future__ import annotations

import numpy as np

# Assumed scintillator size, the one of the CosmicWatch design. The actual
# slab was never measured (docs/dataset.md 4.1): this is an assumption.
DEFAULT_DIMS_CM = (5.0, 5.0, 1.0)


def _hemisphere_grid(n_theta: int):
    theta = (np.arange(n_theta) + 0.5) * (np.pi / 2) / n_theta
    phi = (np.arange(4 * n_theta) + 0.5) * (2 * np.pi) / (4 * n_theta)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    direction = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    solid_angle = np.sin(t) * (np.pi / 2 / n_theta) * (2 * np.pi / (4 * n_theta))
    return t, direction, solid_angle


def slab_rate(alpha_deg: float, dims_cm=DEFAULT_DIMS_CM, n: float = 2.0,
              n_theta: int = 400) -> float:
    """Rate, in arbitrary units, of a box tilted by alpha about one of its edges.

    alpha = 0 is the large face horizontal. The tilt is a rotation about the
    y axis, so the a x b face turns towards the vertical.
    """
    a, b, c = dims_cm
    theta, d, d_omega = _hemisphere_grid(n_theta)
    al = np.deg2rad(alpha_deg)
    e3 = np.array([np.sin(al), 0.0, np.cos(al)])     # normal of the a x b face
    e1 = np.array([np.cos(al), 0.0, -np.sin(al)])    # normal of the b x c face
    e2 = np.array([0.0, 1.0, 0.0])                   # normal of the a x c face
    area = (a * b * np.abs(np.tensordot(e3, d, 1))
            + b * c * np.abs(np.tensordot(e1, d, 1))
            + a * c * np.abs(np.tensordot(e2, d, 1)))
    return float(np.sum(np.cos(theta) ** n * area * d_omega))


def slab_response(alpha_deg, dims_cm=DEFAULT_DIMS_CM, n: float = 2.0,
                  n_theta: int = 400) -> np.ndarray:
    """g(alpha) = R(alpha) / R(0) for every angle given."""
    reference = slab_rate(0.0, dims_cm, n, n_theta)
    return np.array([slab_rate(x, dims_cm, n, n_theta) / reference
                     for x in np.atleast_1d(alpha_deg)])


def two_component(g: np.ndarray, muon_fraction: float) -> np.ndarray:
    """Relative rate when a fraction f of the triggers at 0 degrees are muons.

    The rest is taken as independent of the orientation (methodology 10.3):
    f g(alpha) + 1 - f.
    """
    return muon_fraction * np.asarray(g) + 1.0 - muon_fraction


def fit_muon_fraction(rates: np.ndarray, errors: np.ndarray, g: np.ndarray,
                      step: float = 0.001) -> dict:
    """Fit R0 (f g + 1 - f) by weighted least squares, f in [0, 1].

    For fixed f the model is linear in R0, so R0 is profiled out in closed form
    and f is scanned on a grid: no optimiser, no starting point, same answer on
    every machine. The interval is where chi2 rises by 1 over its minimum,
    clipped to [0, 1].
    """
    w = 1.0 / np.asarray(errors) ** 2
    r = np.asarray(rates)
    grid = np.arange(0.0, 1.0 + step / 2, step)
    chi2 = np.empty_like(grid)
    for i, f in enumerate(grid):
        h = two_component(g, f)
        r0 = np.sum(w * r * h) / np.sum(w * h * h)
        chi2[i] = np.sum(w * (r - r0 * h) ** 2)
    best = int(np.argmin(chi2))
    inside = grid[chi2 <= chi2[best] + 1.0]
    h = two_component(g, grid[best])
    return {
        "muon_fraction": float(grid[best]),
        "interval": [float(inside.min()), float(inside.max())],
        "scale": float(np.sum(w * r * h) / np.sum(w * h * h)),
        "chi2": float(chi2[best]),
        "dof": int(len(r) - 2),
    }
