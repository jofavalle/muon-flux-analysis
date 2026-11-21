# Methodology

Normative document. Every rule is numbered so that code and report can cite it,
and the numbers are stable: a rule that stops applying is struck through rather
than removed, so an old citation never silently points at a different rule.

Cite as `methodology.md 3.2` in comments and commit messages.

---

## 1. Scope and observable

**1.1** The measurement consists of eight runs of a CosmicWatch desktop muon
detector, one per nominal angle, described in [dataset.md](dataset.md).
*Added 2026-11-21:* sections 1 to 8 govern that 2025 angular campaign. The
2026 altitude campaign (dataset.md section 7) is governed by sections 9 and
11, and the second reading of the angular runs by section 10.

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
*Added 2026-11-21:* the 2026 campaign showed what the second hypothesis
really is. The record holds the singles of one detector, most of which are
not muons, and one slab does not fall to zero at 90 degrees even for muons
(section 10). `B` is therefore dominated by composition and geometry; the
statement that these data alone cannot split it stands.

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
*Added 2026-11-21:* section 10 uses an assumed geometry to build a reference
curve, not to correct the data. The rule stands: no rate is corrected for an
acceptance that was never measured.

**8.4** Where this analysis contradicts the repository's earlier published
result, the contradiction is stated explicitly along with the cause. The earlier
result, `n = 0.006`, was double-precision rounding, and saying so is more useful
to a reader than quietly replacing the number.

---

## 9. Detector state (2026 campaign)

**9.1** `SiPM[mV]` is an exact function of the pair `(ADC1, ADC2)`: in the
three data sets, every event that shares a pair with another shares its SiPM
value. It carries no information of its own and is never used to classify.

**9.2** The **pedestal** of a minute is the median ADC1 of its events with
`ADC2 < 1500`. Below that value of ADC2 the low-gain channel barely leaves its
baseline, so the pedestal measures the state of the electronics rather than
the pulses. It is the state monitor.

**9.3** A **burst** event reads `ADC1 > ADC2`. A real pulse never reads more on
the low-gain channel than on the high-gain one; clean data do so in 0 to 0.1 %
of events, the burst of minutes 70 to 73 of Tacachico in 97 %.

**9.4** A minute is **vetoed** if its pedestal exceeds 170, more than 1 % of
its events are burst events, or more than 60 of its events have ADC2 in
`[200, 260)`; one guard minute on each side of a vetoed minute is vetoed too.
The third criterion catches a slow-fault episode whose pedestal stays just
under 170 (Tacachico, minute 319).

**9.5** After the veto, events are kept with `ADC2 >= 260`. The slow fault
adds events just above the trigger threshold (201) and, up to a pedestal of
200, nowhere above ADC2 = 400.

**9.6** The thresholds of 9.4 and 9.5 were chosen **by looking at the Tacachico
file**, the one that has the fault. Its validation (step 07) is therefore not
blind, and is reported as a consistency check, not as a measured efficiency.
The conclusion that the 2025 runs were not taken in the fault state does not
depend on the thresholds: no 2025 event reads `ADC1 > ADC2`, and the 2025
near-threshold rate equals that of the cleanest 2026 state.

**9.7** The `Coincident` flag is not a coincidence and is never a selection.
In the 2026 Tacachico file, with no second detector connected, it reads 1 on
every event of the first three hours, on 87 % of the fourth and on 0.05 % of
the events after that; in the 2025 runs it reads 1 on every event, at a rate
no coincidence can reach.

---

## 10. The reference for a single detector

**10.1** The rate of one scintillator slab tilted by `alpha` is the flux
`I(theta) ~ cos^2(theta)` integrated over the upper hemisphere against the area
the slab presents to each direction. The ratio to the horizontal slab is
`g(alpha)`. `cos^2(theta)` itself is the reference only for a narrow telescope
of two detectors in coincidence.

**10.2** The slab is **assumed** to be 5 x 5 x 1 cm, the CosmicWatch design;
it was never measured (dataset.md 4.1). Then `g(90) = 2/3` exactly. The step
reports `g` also for 0.5 cm, 2 cm and a thin plate so the dependence on the
assumption is visible.

**10.3** The triggers are modelled as a fraction `f` of muons, which follow
`g`, plus a remainder independent of orientation: `I(alpha)/I(0) = f g + 1 - f`.
The remainder (radioactivity, electronic noise, the soft component) is taken
as isotropic over the upper hemisphere, for which a plate sees the same rate
at any tilt. This is an assumption, not a measurement.

**10.4** `f` is fitted at each ADC2 cut with the scale profiled out in closed
form and `f` scanned on a grid of step 0.001 in `[0, 1]`; the interval is where
chi-square rises by 1. No optimiser and no starting point, so the result is the
same on every machine.

---

## 11. Rate against altitude (2026 campaign)

**11.1** Tacachico uses the window `[93 600, 144 000)` s of its first
acquisition, hours 26 to 40, the reference window of the fault report, for
the analysis as presented. Santa Tecla uses its whole file,
`[t_first, t_last)`.

**11.2** The rate of a site is `N / (T - sum DeadTime)` over its window, with
`sigma = sqrt(N) / T_live`. DeadTime is the dead time accumulated since the
previous event.

**11.3** The hour-to-hour spread of a site's live rate beyond Poisson,
`sqrt(var - <sigma^2>)`, is a systematic of the ratio (rule 5.1 applied to
hours). In Tacachico it follows the case temperature.

**11.4** `Delta h` comes from the two mean pressures through the hypsometric
equation with an assumed air column of 25 +- 5 C, plus 200 Pa of weather
between the two days on each site. The absolute accuracy of the sensor cancels
in the difference but not in each altitude from sea level.

**11.5** Predictions are `exp(-Delta h / (gamma c tau))` with `gamma = 1`
(classical) and `gamma = E/m` for `E = 4 GeV` (relativistic, decay only); both
carry the error of `Delta h`.

**11.6** `gamma_min` is the Lorentz factor for which that survival equals the
measured ratio. Its one-sided 95 % bound is taken in log scale, because
`gamma_min` grows steeply as the ratio approaches 1. Every one of these numbers
assumes that each trigger is a muon, which a single detector does not satisfy;
the report says so next to them.

**11.7** The cleaned analysis (state veto and `ADC2 >= 260` at both sites,
whole acquisition) is reported next to the presented one and **never combined
with it**. A fixed ADC2 cut is a different energy threshold at a different case
temperature, because the gain of the SiPM falls as it warms up, so the cleaned
ratio is a different observable, not a correction.
