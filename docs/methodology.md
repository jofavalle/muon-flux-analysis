# Methodology

Normative document. Every rule is numbered so that code and report can cite it,
and the numbers are stable: a rule that stops applying is struck through rather
than removed, so an old citation never silently points at a different rule.

Cite as `methodology.md 3.2` in comments and commit messages.

---

## 1. Scope and observable

**1.1** The measurement consists of eight runs of a CosmicWatch desktop muon
detector, one per nominal angle, described in [dataset.md](dataset.md).

**1.2** The observable is the number of inter-event intervals in a run, `N-1`,
modelled as Poisson with mean `R * T_live`. Not `N`: for `N` events observed in
the window `[t1, tN]` there are `N-1` intervals, and `(N-1)/T` is the unbiased
rate estimator while `N/T` overstates it by `1/N`.

**1.3** The observation window of a run is `tN - t1`, using the full window. It
is never truncated to a whole multiple of any bin size.

**1.4** All inference is performed on raw counts, never on normalised ratios.
The ratios `I(theta)/I(0)` share a denominator and are therefore not
independent; a diagonal chi-square over them is ill-posed. Ratios are computed
in step 04 for presentation and for comparison with the earlier analysis only.

---

## 2. Live time

**2.1** The live time of a run is `T_live = (tN - t1) - sum(DeadTime)`, with the
`DeadTime[us]` column summed over all events and converted to seconds.

**2.2** Every published rate is a live-time rate. Wall-clock rates are reported
alongside only to show the size of the correction.

**2.3** The dead fraction is 9.09 % to 9.28 % across the eight runs, so the
correction lifts every rate by about 10 % and **very nearly cancels in the
ratios between angles**. The report states this explicitly. Correcting is
mandatory, but it must not be presented as fixing the angular shape, which it
does not: it changes the absolute rate and leaves the shape where it was.

---

## 3. Angular models

**3.1 The cosine clamp.** `cos(theta)` is evaluated once, and any value with
`|cos| < 1e-12` is clamped to exact zero. `np.cos(np.deg2rad(90))` returns
6.123e-17, not zero, and raised to a small exponent it yields order-unity
values: `6.123e-17 ** 0.01 = 0.688`. Every model evaluation goes through
`lib_muon.models.clamped_cosine`.

**3.2 Consequence of 3.1, and it is not a technicality.** With `cos(90) = 0`
exactly, any `cos^n` with `n > 0` predicts exactly zero counts at 90 degrees.
Since 446 intervals were observed there, the likelihood of pure `cos^n` is
rigorously zero for every `n > 0`. The model is not a poor fit; it is
impossible, and it has no finite AIC or BIC to compare. This is reported as the
result it is, not worked around.

**3.3** The families fitted are:

| Model | Form | Parameters |
|---|---|---|
| A | `mu = R * T` | 1 |
| B | `mu = R * T * cos^n(theta)` | 2 |
| C | `mu = R * T * cos^2(theta)` | 1 |
| D | `mu = T * (S * cos^n(theta) + B)` | 3 |
| E | `mu = T * (S * cos^2(theta) + B)` | 2 |

**3.4** Models B and C are fitted knowing they will return infinite `-logL`, so
that the record is reproducible rather than asserted.

**3.5** The `B` term of models D and E collects in a single number everything
the three hypotheses of this analysis predict at 90 degrees: wide geometric
acceptance, an accidental component, and an undocumented angular convention
(see [dataset.md](dataset.md) section 4). **These data cannot separate the
three**, and no result may be phrased as though they could.

---

## 4. Inference

**4.1** Point estimates by Poisson maximum likelihood, with the `log(k!)` term
included so that `-logL` values are absolute rather than relative.

**4.2** Parameter uncertainties by profile likelihood: `Delta(-2lnL) = 1` for
the 68 % interval, `Delta(-2lnL) = 2.706` for a one-sided 95 % bound. No
uncertainty is ever reported from the curvature of a grid search.

**4.3** Goodness of fit by Poisson deviance against the saturated model, with a
chi-square p-value on `n_points - n_parameters` degrees of freedom.

**4.4** Nested model comparison by likelihood ratio (Wilks). Where the extra
parameter sits on the boundary of its space, as `S >= 0` does, the caveat that
the asymptotic distribution is a mixture and the p-value conservative is stated
with the number.

**4.5** Non-nested comparison by AIC and BIC. Only finite values are compared;
a model with null likelihood is reported as excluded, not as a large number.

**4.6** No substitute error is ever invented. Where an uncertainty is
unavailable the value is reported as unavailable. The previous analysis
substituted `sigma = 1.0` for missing errors and floored the degrees of freedom
at `max(dof, 1)`, which manufactured a reduced chi-square where there was
nothing to divide by.

---

## 5. Systematics

**5.1 The Poisson-expectation test, and it is mandatory.** Before any candidate
source enters the systematics budget, its observed size is compared against
what counting statistics alone predicts for that same comparison. Only the
**excess** over that expectation enters the table.

Worked example, which is why this rule exists: comparing half a run against the
whole run shifts the rate by up to 5.45 %. Half a run holds half the events and
`1/sqrt(300) = 5.8 %`, so the observed shift is exactly the expected
fluctuation. Its excess is 0.00 %. Entering the 5.45 % as a systematic would
have inflated the budget with statistics wearing a systematic's clothes.

**5.2 Run-to-run repeatability** is measured, not assumed, from the 360 degree
file, which is a second measurement of the same angle as the 0 degree file. It
is the dominant systematic and the only one this data set permits to be
measured.

**5.3** The repeatability discrepancy is 7.45 %, at 1.33 sigma. The report
states both numbers: **as an established effect it is not significant**, but as
the scale of the run-to-run systematic it is the best information available.
Claiming it as a detected difference would be as wrong as ignoring it.

**5.4** Systematics are folded in as a relative term added in quadrature to the
statistical error of each run.

**5.5** Any claim of significance is reported twice: under pure counting
statistics, and with the measured systematic folded in. **The second is the one
that stands.** Reporting only the first is the error this analysis exists to
correct.

**5.6** Where a claim's significance depends on the systematic, the report
states the **threshold**: the systematic at which the claim stops being
significant, next to the systematic actually measured. Here 3.88 % against 7.45 %
measured.

---

## 6. Products and traceability

**6.1** Every script writes with a `_vN` suffix and **no script overwrites a
previous run**. The version lives in the product, never in the script name.

**6.2** Every JSON carries a provenance block: commit hash, branch, whether the
inputs were committed, seed, Python and numpy versions, platform, and the
parameters of the run.

**6.3** `inputs_clean` in that block ignores everything under `results/`. What
must be committed for a result to be reproducible is what produced it, the code
and the raw data. A pipeline dirties its own output directory as it writes, so
including products would mark every result dirty for merely existing, and a
provenance field that always warns teaches the reader to ignore it.

**6.4** Products are paired: `.json` to be read by another script, `.txt` to be
read by a person without tools. Figures are paired as PDF and PNG.

**6.5** Non-finite values are written to JSON as `null`, never as `Infinity` or
`NaN`, which are not valid JSON. A null `nll` means the model had no finite
likelihood, which for models B and C is the result itself, not a missing value.

**6.6** Numbering is the execution order. Steps 00 to 06 are the analysis; step
90 and above are presentation, run afterwards, and perform no fitting: they
read from `results/` so that a figure cannot disagree with the table it came
from.

---

## 7. Quality gate

**7.1** Step 01 checks the conditions the rest of the pipeline assumes, listed
in its header docstring.

**7.2** Malformed lines are counted, their text is kept, and they are reported.
They are never dropped silently.

**7.3** Anomalies already documented in [dataset.md](dataset.md) are reported
but do not fail the gate. A gate that fails on every run is a gate nobody reads,
and within days it is run with `|| true`.

**7.4** The comparison against the documented state is exact and runs in **both
directions**. The gate fires if a new anomaly appears, if a documented one
changes, and equally if a documented one **disappears**: the data would have
changed just as much. The anchors hold the literal text of the rejected line and
the per-angle counts, not merely a category.

**7.5** `--strict` makes documented anomalies count as alerts too, for use when
the question is what is wrong with the data rather than what has changed.

---

## 8. Reporting

**8.1** The conclusion appears at the top of the README, not at the end.

**8.2** A null result is stated as a **bound**, with its confidence level,
never as "nothing was seen".

**8.3** Statements the data cannot support are not made, and the reason they
cannot be supported is given. In particular, no geometric acceptance correction
is attempted, because the detector geometry is not recorded; see
[dataset.md](dataset.md) section 4.

**8.4** Where this analysis contradicts the repository's earlier published
result, the contradiction is stated explicitly along with the cause. The earlier
result, `n = 0.006`, was double-precision rounding, and saying so is more useful
to a reader than quietly replacing the number.
