# Muon angular flux analysis

Angular dependence of the cosmic muon flux, measured with a CosmicWatch desktop
detector at seven angles plus one repeatability control.

## Result

**This setup did not measure the angular dependence of the muon flux, and the
analysis explains why.**

The fraction of flux modulated as `cos^2(theta)` is bounded at

> **< 25.4 % at 95 % confidence**, systematic included.

The data on their own prefer a modulated component of about 12 %, significant
at p = 0.008 under pure counting statistics. **That preference does not
survive.** A run-to-run systematic of 3.88 % is enough to cancel it, and the
repeatability control in this very data set measures **7.45 %**, nearly twice
that. With the measured systematic folded in, p = 0.19.

The single most telling comparison in the whole data set:

| | |
|---|---|
| Difference between two runs at the **same** angle | **7.45 %** |
| The **entire** drop from 0 to 45 degrees | 10.0 % |

The run-to-run systematic is the size of the whole signal the fit was trying to
interpret.

![Summary](results/figures/90_summary_v1.png)

![Repeatability](results/figures/90_repeatability_v1.png)

### Headline numbers

| Quantity | Value |
|---|---|
| Events analysed | 4760, across 8 runs |
| Constant rate | 1.816 [1.788, 1.844] 1/s at 68 % |
| I(90)/I(0) measured | 0.806 +- 0.049 |
| I(90)/I(0) predicted by `cos^2` | 0 |
| chi-square against pure `cos^2`, 7 points | 372.9 |
| Modulated fraction, best fit | 12.5 % |
| Modulated fraction, 95 % upper bound | **25.4 %** |
| Run-to-run systematic, measured | 7.45 % |
| Systematic that cancels the significance | 3.88 % |

---

## Contents

- [What the previous version of this repository published, and why it was wrong](#what-the-previous-version-published)
- [Why the flux looks flat](#why-the-flux-looks-flat)
- [Repository layout](#repository-layout)
- [Reproducing the analysis](#reproducing-the-analysis)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [License](#license)

---

## What the previous version published

Until 2026-08-10 this repository fitted `I(theta)/I(0) = cos^n(theta)` and
reported **n = 0.006**. That number was double-precision rounding, not physics.

`np.cos(np.deg2rad(90))` does not return zero. It returns 6.123e-17. Raised to a
small exponent that is a number of order one:

```
cos(deg2rad(90)) ** 0.01 = 0.688
```

So the grid search found a minimum pinned to the bottom edge of its range, and
the exponent it reported was set by how a floating-point residue behaves under
exponentiation. Two symptoms show how little it meant:

- Dropping the single point at 90 degrees moved the exponent from 0.006 to
  0.364, a factor of 64.
- The exponent depended on whether the grid search refined its minimum: the
  coarse grid gave 0.010, the refinement 0.006. A parameter whose value moves by
  75 % with an implementation detail of the optimiser is not measuring anything.

**Clamping `cos(90)` to exact zero does not fix the fit. It removes it.** With
`cos(90) = 0`, every `cos^n` with n > 0 predicts exactly zero counts at 90
degrees, where 446 intervals were observed. The likelihood is rigorously zero:
`-logL` diverges, and there is no AIC or BIC to compare. The 6.123e-17 was the
only reason the model could be fitted at all.

Other defects corrected in the same pass, each of which changed a published
number:

| Defect | Effect |
|---|---|
| `DeadTime` column parsed and never used | every rate 10 % low |
| Rate estimated as `N/T` instead of `(N-1)/T` | biased high by 1/N |
| Run duration truncated to a whole multiple of `--bin-seconds` | a cosmetic flag moved the published rate by 0.4 % |
| chi-square over ratios sharing a denominator | correlations ignored entirely |
| 0 degree point counted in the degrees of freedom | reduced chi-square understated |
| `sigma = 1.0` substituted for missing errors, `max(dof, 1)` | manufactured a reduced chi-square out of nothing |
| Malformed lines dropped by `except Exception: continue` | one real malformed line, never reported |
| Column order guessed by magnitude, per row | the file header declares the wrong order and nothing said so |

The six usage examples in the old README all invoked `--data-dir second`, a
directory that exists in no commit in the repository's history.

---

## Why the flux looks flat

The measured profile is flat and not even monotonic: it falls to 0.845 at 15
degrees and rises to 0.950 at 25. No `cos^n` with n > 0 can rise anywhere, so
if that rise were real it would rule out the entire family. On its own the rise
is worth 1.47 sigma, which is not enough to assert it; the chi-square of 372.9
is what carries the conclusion.

Three explanations are consistent with the data, and **this data set cannot
distinguish between them**:

1. **Wide geometric acceptance.** A short, wide two-scintillator telescope
   accepts nearly a hemisphere, and its rate barely depends on orientation
   however strongly the underlying flux is modulated. The detector geometry was
   never recorded, so this cannot be corrected or ruled out. It is the principal
   limitation of this analysis.
2. **An accidental component.** `Coincident == 1` for all 4760 events and no
   singles were recorded, so the accidental rate cannot be estimated, subtracted
   or bounded from these files.
3. **An undocumented angular convention.** There is no record of whether the
   angles are zenith or azimuth, what they are measured from, or with what
   precision.

In the fitted models these three collapse into a single background term. No
claim is made as if they could be separated. See
[docs/dataset.md](docs/dataset.md) section 4.

---

## Repository layout

```
muon-angle-analysis/
  README.md                     this document
  docs/
    methodology.md              numbered, citable analysis rules
    dataset.md                  what is known and, above all, what is not
  scripts/
    00_ingest.py                read and inventory the raw runs
    01_validation.py            data quality gate
    02_rates.py                 live-time corrected rates
    03_timeseries.py            per-run stability
    04_normalization.py         I(theta)/I(0) and the cos^2 contrast
    05_fits.py                  Poisson likelihood inference
    06_systematics.py           systematics and the reassessment
    90_report_figures.py        README figures, from results/ only
    check_reproducibility.py    published results against a fresh run
    lib_muon/                   shared library
  results/
    tables/  fits/  figures/    versioned products, _vN
  data/raw/                     the eight raw runs, 256 kB
  tests/                        76 tests, including the full pipeline
  requirements.txt              pinned versions
```

Conventions, described in full in [docs/methodology.md](docs/methodology.md)
section 6:

- **The numbering is the execution order.** 00 to 06 analyse, 90 and above
  present and perform no fitting.
- **No script overwrites a result.** Each run writes `_vN`; the suffix lives in
  the product, never in the script name.
- **Results are versioned.** Every figure and table in this README is in the
  repository, and every number in it is traceable to the run that produced it.
- **Every JSON carries** the commit hash, the parameters and the seed.
- Products come paired: `.json` and `.txt` for tables, PDF and PNG for figures.

---

## Reproducing the analysis

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Check the published numbers

To verify that everything in this README reproduces on your machine, this is the
whole procedure:

```bash
.venv/bin/python scripts/check_reproducibility.py
```

It runs the entire pipeline into a temporary directory, compares every numeric
leaf of its output against the `_v1` products committed here, and removes the
temporary directory afterwards. It needs no prior step and does not touch
`results/`. Expect this, in about twenty seconds:

```
[check] 00_ingest: reproduces exactly
...
[check] OK: 8 product(s) compared, all reproduce within rtol=1e-06, atol=1e-12.
```

Exit codes are `0` if every number reproduces, `1` if any disagrees beyond
floating-point noise or the repository publishes nothing to compare against, and
`2` if the pipeline itself failed to run. The three cases are distinguished on
purpose: "nothing was compared" is not the same claim as "the results do not
reproduce", and reporting the second when the first is true would be exactly the
sort of thing this repository exists to correct.

### Run the pipeline yourself

In order, from the repository root:

```bash
.venv/bin/python scripts/00_ingest.py
.venv/bin/python scripts/01_validation.py
.venv/bin/python scripts/02_rates.py
.venv/bin/python scripts/03_timeseries.py
.venv/bin/python scripts/04_normalization.py
.venv/bin/python scripts/05_fits.py
.venv/bin/python scripts/06_systematics.py
.venv/bin/python scripts/90_report_figures.py
```

Each script writes a new `_vN` alongside the committed results rather than
replacing them, so your run and the published one coexist and can be diffed.

Tests:

```bash
.venv/bin/pytest tests/
```

The suite includes a regression test for the `cos(90)` defect, recovery of a
known exponent from synthetic data, a rate checked against a hand-computed case,
and an end-to-end run of the whole pipeline. CI runs all of it plus the full
pipeline on the real data, on Python 3.11 and 3.13.

---

## Documentation

- **[docs/methodology.md](docs/methodology.md)**: the analysis rules, numbered
  so code and report can cite them. Includes the mandatory rule that every
  candidate systematic is compared against its own Poisson expectation before
  entering the budget (rule 5.1), which is what keeps statistical fluctuation
  from being reported as a systematic.
- **[docs/dataset.md](docs/dataset.md)**: the files, the format, the two known
  defects, and section 4 on what is not recorded about the setup, which
  constrains this analysis more than anything that is.

---

## Limitations

1. **No geometric acceptance correction is possible.** The detector geometry
   was not recorded. This is the largest single limitation and no amount of
   reanalysis addresses it; it needs a tape measure applied to the apparatus.
2. **The accidental rate is not estimable.** No singles were recorded.
3. **The angular convention is not documented.** The mapping from file labels to
   physical angles is assumed, not known.
4. **One repeatability control, at 1.33 sigma.** The dominant systematic rests
   on a single repeated measurement. Repeating every angle would turn it from an
   estimated scale into a measured one.
5. **Longer runs would not help.** At about 1.8 counts per second, six minutes
   already gives a 4 % statistical error, smaller than the 7.45 % run-to-run
   systematic. The measurement is limited by reproducibility, not by statistics.

What would be needed to do better, in order of value, is in
[docs/dataset.md](docs/dataset.md) section 6.

---

## License

MIT. See [LICENSE](LICENSE).

Raw data recorded with a [CosmicWatch](http://www.cosmicwatch.lns.mit.edu/)
desktop muon detector.
