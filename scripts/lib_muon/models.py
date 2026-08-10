"""Angular models and Poisson maximum-likelihood inference.

Why likelihood on raw counts rather than chi-square on ratios, which is what
the original script did:

- The seven ratios I(theta)/I(0) share a denominator. They are NOT
  independent, and the original diagonal chi-squares ignored that correlation
  entirely. On raw counts the problem does not exist.
- With 445 to 766 counts per point the Gaussian approximation is decent, but
  the Poisson likelihood is exact and costs nothing extra.
- The 0 degree point is degenerate after normalising: its residual is zero by
  construction, yet the original script still counted it in the degrees of
  freedom, understating the reduced chi-square.

**Clamping cos(90).** `np.cos(np.deg2rad(90))` is 6.123e-17, not zero. Raised
to a small exponent it gives order-unity values: 6.12e-17**0.01 = 0.688. That
is why the original grid search found its minimum pinned to the lower edge and
published n = 0.006. Here we work in cos(theta) space and clamp every
|cos| < COS_EPS to exact zero.

**Consequence of clamping, which is not a detail.** With cos(90) = 0 exactly,
any cos^n with n > 0 predicts exactly zero counts at 90 degrees. Since 446 were
observed there, the likelihood of pure cos^n is rigorously zero for every
n > 0: the model does not merely fit badly, it is impossible. The only reason
the original script could fit it at all was the 6.123e-17. To bound how much
modulation the data admit, one needs the signal-plus-background family.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

# Below this, a cosine is zero. 1e-12 sits far above the 6.123e-17 produced by
# cos(deg2rad(90)) and far below the cosine of any angle physically
# distinguishable from 90 degrees: cos(89.9 degrees) = 1.7e-3.
COS_EPS = 1e-12

# Nelder-Mead tolerances. `fatol` is absolute on -logL, which here is between
# 20 and 60: asking for 1e-12 would demand 1e-14 relative precision, at the
# edge of double precision, and the optimiser would exhaust its iterations with
# the parameter already well recovered. 1e-9 is ample for Delta(-2lnL) = 1 to
# be unambiguous.
NELDER_MEAD_OPTIONS = {
    "xatol": 1e-9,
    "fatol": 1e-9,
    "maxiter": 20000,
    "maxfev": 20000,
}


def clamped_cosine(angles_deg: np.ndarray, eps: float = COS_EPS) -> np.ndarray:
    """cos(theta) with near-zero values clamped to exact zero.

    This is the fix for the defect that motivated rebuilding the analysis.
    """
    cos_t = np.cos(np.deg2rad(np.asarray(angles_deg, dtype=float)))
    return np.where(np.abs(cos_t) < eps, 0.0, cos_t)


def cosn_shape(cos_t: np.ndarray, n: float) -> np.ndarray:
    """cos^n(theta), defined also at cos = 0 and for n = 0.

    With numpy's 0**0 = 1 convention, n = 0 gives the constant shape.
    """
    cos_t = np.asarray(cos_t, dtype=float)
    positive = cos_t > 0
    out = np.zeros_like(cos_t)
    out[positive] = cos_t[positive] ** n
    if n == 0:
        out[~positive] = 1.0
    return out


def poisson_nll(mu: np.ndarray, counts: np.ndarray) -> float:
    """Poisson -log L, including the log(k!) term.

    Returns infinity if any mu is zero where counts were observed: that model
    has rigorously zero probability, and saying so is the result.
    """
    mu = np.asarray(mu, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if np.any(mu < 0):
        return np.inf
    if np.any((mu == 0) & (counts > 0)):
        return np.inf
    safe = np.where(mu > 0, mu, 1.0)
    return float(np.sum(safe - counts * np.log(safe) + gammaln(counts + 1.0)))


@dataclass
class Fit:
    """Result of a fit, with what is needed to compare it against another."""

    name: str
    parameters: dict[str, float]
    nll: float
    n_parameters: int
    n_points: int
    mu: np.ndarray = field(repr=False)
    converged: bool = True

    @property
    def aic(self) -> float:
        return 2 * self.n_parameters + 2 * self.nll

    @property
    def bic(self) -> float:
        return self.n_parameters * np.log(self.n_points) + 2 * self.nll

    @property
    def dof(self) -> int:
        return self.n_points - self.n_parameters

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "nll": self.nll if np.isfinite(self.nll) else None,
            "nll_infinite": not np.isfinite(self.nll),
            "n_parameters": self.n_parameters,
            "n_points": self.n_points,
            "dof": self.dof,
            "aic": self.aic if np.isfinite(self.aic) else None,
            "bic": self.bic if np.isfinite(self.bic) else None,
            "converged": self.converged,
        }


def fit_constant(counts: np.ndarray, times: np.ndarray) -> Fit:
    """Single rate: mu_i = R * T_i. One parameter.

    The maximum-likelihood estimator is closed form, R = sum(N)/sum(T).
    """
    counts = np.asarray(counts, dtype=float)
    times = np.asarray(times, dtype=float)
    r = float(counts.sum() / times.sum())
    mu = r * times
    return Fit(
        name="constant",
        parameters={"R_hz": r},
        nll=poisson_nll(mu, counts),
        n_parameters=1,
        n_points=len(counts),
        mu=mu,
    )


def fit_pure_cosn(counts: np.ndarray, times: np.ndarray, cos_t: np.ndarray) -> Fit:
    """mu_i = R * T_i * cos^n(theta_i). Two parameters.

    With cos(90) clamped to zero this model has null likelihood as soon as some
    point has cos = 0 with observed counts, which is exactly the case here. It
    is fitted anyway to leave a reproducible record that the result is
    -logL = infinity, not merely an ugly number.
    """
    counts = np.asarray(counts, dtype=float)
    times = np.asarray(times, dtype=float)
    impossible = np.any((cos_t <= 0) & (counts > 0))

    def objective(p):
        log_r, n = p
        if not (0.0 <= n <= 10.0):
            return np.inf
        return poisson_nll(np.exp(log_r) * times * cosn_shape(cos_t, n), counts)

    if impossible:
        # Any n > 0 gives infinite -logL. The formal optimum is n = 0, which is
        # the constant model in disguise; it is reported as not converged.
        r = float(counts.sum() / times.sum())
        return Fit(
            name="pure cos^n",
            parameters={"R_hz": r, "n": 0.0},
            nll=np.inf,
            n_parameters=2,
            n_points=len(counts),
            mu=r * times * cosn_shape(cos_t, 0.0),
            converged=False,
        )

    r0 = float(counts.sum() / times.sum())
    result = optimize.minimize(
        objective, x0=[np.log(r0), 1.0], method="Nelder-Mead",
        options=NELDER_MEAD_OPTIONS,
    )
    log_r, n = result.x
    mu = np.exp(log_r) * times * cosn_shape(cos_t, n)
    return Fit(
        name="pure cos^n",
        parameters={"R_hz": float(np.exp(log_r)), "n": float(n)},
        nll=float(result.fun),
        n_parameters=2,
        n_points=len(counts),
        mu=mu,
        converged=bool(result.success),
    )


def fit_fixed_cosn(
    counts: np.ndarray, times: np.ndarray, cos_t: np.ndarray, n: float = 2.0
) -> Fit:
    """mu_i = R * T_i * cos^n with n fixed. One parameter.

    With n = 2 this is the physical hypothesis the repository promised.
    """
    counts = np.asarray(counts, dtype=float)
    times = np.asarray(times, dtype=float)
    shape = cosn_shape(cos_t, n)
    denominator = float(np.sum(times * shape))
    r = float(counts.sum() / denominator) if denominator > 0 else np.nan
    mu = r * times * shape
    nll = poisson_nll(mu, counts)
    return Fit(
        name=f"fixed cos^{n:g}",
        parameters={"R_hz": r, "n": float(n)},
        nll=nll,
        n_parameters=1,
        n_points=len(counts),
        mu=mu,
        converged=bool(np.isfinite(nll)),
    )


def fit_signal_background(
    counts: np.ndarray,
    times: np.ndarray,
    cos_t: np.ndarray,
    n: float | None = None,
) -> Fit:
    """mu_i = T_i * (S * cos^n(theta_i) + B).

    The only family that accommodates both a modulated component and the fact
    that counts are observed at 90 degrees. B collects in a single number what
    all three hypotheses of the analysis predict at 90 degrees: wide geometric
    acceptance, an accidental component, and an undocumented angular
    convention. The data cannot separate them, and the report does not pretend
    otherwise.

    With `n` given, S and B are fitted (two parameters). With `n = None`, n too.
    """
    counts = np.asarray(counts, dtype=float)
    times = np.asarray(times, dtype=float)
    free_n = n is None
    typical_rate = float(counts.sum() / times.sum())

    def mu_of(s, exponent, b):
        return times * (s * cosn_shape(cos_t, exponent) + b)

    def objective(p):
        if free_n:
            s, exponent, b = p[0], p[1], p[2]
        else:
            s, b = p[0], p[1]
            exponent = n
        if s < 0 or b <= 0 or exponent < 0 or exponent > 10:
            return np.inf
        return poisson_nll(mu_of(s, exponent, b), counts)

    best = None
    signal_seeds = (0.05, 0.2, 0.5)
    exponent_seeds = (0.5, 1.0, 2.0) if free_n else (n,)
    for s0 in signal_seeds:
        for n0 in exponent_seeds:
            x0 = [s0 * typical_rate, n0, (1 - s0) * typical_rate] if free_n else [
                s0 * typical_rate, (1 - s0) * typical_rate
            ]
            result = optimize.minimize(
                objective, x0=x0, method="Nelder-Mead",
                options=NELDER_MEAD_OPTIONS,
            )
            if best is None or result.fun < best.fun:
                best = result

    if free_n:
        s, exponent, b = best.x
    else:
        s, b = best.x
        exponent = float(n)

    parameters = {"S_hz": float(s), "B_hz": float(b), "n": float(exponent)}
    total = s + b
    parameters["modulated_fraction"] = float(s / total) if total > 0 else np.nan
    return Fit(
        name="signal+background" + ("" if free_n else f" (n={n:g} fixed)"),
        parameters=parameters,
        nll=float(best.fun),
        n_parameters=3 if free_n else 2,
        n_points=len(counts),
        mu=mu_of(s, exponent, b),
        converged=bool(best.success),
    )


def profile_likelihood(
    nll_of_parameter,
    best_value: float,
    best_nll: float,
    delta: float = 0.5,
    lower_limit: float = 0.0,
    upper_limit: float = 10.0,
) -> tuple[float, float]:
    """Profile-likelihood interval around the optimum.

    `delta` is the increase in -logL defining the interval: 0.5 for 68 %
    (Delta(-2lnL) = 1), 1.353 for a one-sided 95 % bound.
    Returns (lower, upper); an endpoint is NaN when there is no crossing within
    the limits, which means the data do not bound that side.
    """
    target = best_nll + delta

    def excess(x):
        return nll_of_parameter(x) - target

    lower = np.nan
    if best_value > lower_limit and np.isfinite(excess(lower_limit)):
        if excess(lower_limit) > 0:
            lower = float(optimize.brentq(excess, lower_limit, best_value))
        else:
            lower = lower_limit

    upper = np.nan
    if np.isfinite(excess(upper_limit)) and excess(upper_limit) > 0:
        upper = float(optimize.brentq(excess, best_value, upper_limit))
    else:
        # Search outwards from the optimum; if there is no crossing, unbounded.
        high = best_value
        for _ in range(60):
            high = high * 2 if high > 0 else 1e-3
            if high > upper_limit:
                break
            value = excess(high)
            if np.isfinite(value) and value > 0:
                upper = float(optimize.brentq(excess, best_value, high))
                break
    return lower, upper


def deviance(counts: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance: 2*(logL_saturated - logL_model).

    Approximately chi-square distributed with dof degrees of freedom, and the
    correct goodness-of-fit statistic for counts.
    """
    counts = np.asarray(counts, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if np.any((mu <= 0) & (counts > 0)):
        return np.inf
    term = np.where(counts > 0, counts * np.log(counts / mu), 0.0)
    return float(2 * np.sum(term - (counts - mu)))


def pearson_residuals(counts: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """(N - mu)/sqrt(mu), in units of sigma."""
    counts = np.asarray(counts, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return np.where(mu > 0, (counts - mu) / np.sqrt(np.where(mu > 0, mu, 1.0)), np.nan)


def goodness_of_fit(counts: np.ndarray, mu: np.ndarray, dof: int) -> tuple[float, float]:
    """Deviance and its p-value under chi-square with `dof` degrees of freedom."""
    d = deviance(counts, mu)
    if not np.isfinite(d) or dof <= 0:
        return d, 0.0
    return d, float(stats.chi2.sf(d, dof))


def likelihood_ratio(
    nll_null: float, nll_alternative: float, extra_dof: int
) -> tuple[float, float]:
    """Wilks test between nested models.

    Returns (Delta(-2lnL), p). The null is the model with fewer parameters.
    Statistical caveat: if the extra parameter sits on the boundary of its
    space (S >= 0 with true S = 0), the asymptotic distribution is not
    chi-square but a mixture, and the chi-square p-value is conservative.
    """
    if not np.isfinite(nll_null) or not np.isfinite(nll_alternative):
        return np.inf, 0.0
    d = 2 * (nll_null - nll_alternative)
    if d <= 0 or extra_dof <= 0:
        return float(d), 1.0
    return float(d), float(stats.chi2.sf(d, extra_dof))


def chi2_with_systematic(
    rates: np.ndarray, errors: np.ndarray, systematic_fraction: float
) -> tuple[float, float, float]:
    """Chi-square of a common rate with a relative systematic added in quadrature.

    Used to answer how much run-to-run systematic is needed for the seven
    points to be consistent with a single rate. Returns
    (weighted_rate, chi2, p).
    """
    rates = np.asarray(rates, dtype=float)
    errors = np.asarray(errors, dtype=float)
    sigma = np.sqrt(errors**2 + (systematic_fraction * rates) ** 2)
    weights = 1.0 / sigma**2
    mean = float(np.sum(weights * rates) / np.sum(weights))
    chi2 = float(np.sum(((rates - mean) / sigma) ** 2))
    dof = len(rates) - 1
    return mean, chi2, float(stats.chi2.sf(chi2, dof))


def required_systematic(
    rates: np.ndarray, errors: np.ndarray, target_p: float = 0.05
) -> float:
    """Smallest systematic fraction for the single-rate hypothesis not to be
    rejected at level `target_p`. Returns 0 if it is already consistent."""
    _, _, p_without = chi2_with_systematic(rates, errors, 0.0)
    if p_without >= target_p:
        return 0.0

    def excess(f):
        return chi2_with_systematic(rates, errors, f)[2] - target_p

    return float(optimize.brentq(excess, 1e-9, 1.0))
