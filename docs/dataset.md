# The data set

Eight runs from a CosmicWatch desktop muon detector, 4760 events, 256 kB, in
`data/raw/`. They are versioned with the repository, which is what allows CI to
run the entire pipeline on the real data rather than on fixtures.

**Section 4 is the most important part of this document.** What is not known
about this setup constrains the analysis more than anything that is.

---

## 1. Files

| File | Nominal angle | Events | Window [s] | Live time [s] | Dead fraction |
|---|---|---|---|---|---|
| `000deg` | 0 | 677 | 372.86 | 338.27 | 9.28 % |
| `005deg` | 5 | 590 | 360.03 | 327.07 | 9.16 % |
| `015deg` | 15 | 494 | 321.20 | 291.84 | 9.14 % |
| `025deg` | 25 | 766 | 444.11 | 403.10 | 9.23 % |
| `035deg` | 35 | 544 | 323.01 | 293.24 | 9.22 % |
| `045deg` | 45 | 712 | 435.41 | 395.27 | 9.22 % |
| `090deg` | 90 | 447 | 304.74 | 277.05 | 9.09 % |
| `360deg` | see 4.5 | 530 | 314.96 | 286.03 | 9.19 % |

Filenames were `<angle>grados` before 2026-08-10 and are `NNNdeg` now, with
fixed width so that `005` sorts before `015` and `090` before `360`.

---

## 2. Format

Tab-separated, with a comment preamble. Ten columns per data line.

### 2.1 The header lies about column order

The files declare:

```
### Event TimeStamp[ms] ADC1 ADC2 SiPM[mV] Temp[C] Pressure[Pa] DeadTime[us] Coincident Name
```

The data actually arrive as:

```
Event TimeStamp[ms] ADC1 ADC2 SiPM[mV] Pressure[Pa] Temp[C] DeadTime[us] Coincident Name
```

Pressure comes **before** temperature. A sample line makes it unambiguous:
column 6 reads 93609.2 and column 7 reads 27.8, which can only be pascals and
degrees Celsius respectively.

Anyone trusting the header gets the two swapped. The original script compensated
with an undocumented per-row magnitude heuristic; the pipeline now decides once
per file using the median against declared physical ranges, and records the
decision in the quality report. If neither assignment fits the ranges, it
raises rather than guessing.

### 2.2 Columns that are parsed and not used

`ADC1`, `ADC2` and `SiPM[mV]` are read and validated but enter no result. There
is no pulse-height calibration recorded, so they cannot be converted into
anything physical.

Pressure and temperature are read and, outside the startup rows of 3.2, are
stable to better than 0.1 % over each run. **No barometric correction is
applied**: the effect over a six-minute run is far below the statistical error,
and the runs are too short and too close together for it to matter.

---

## 3. Known defects

Both are anchored in `scripts/01_validation.py` and reported on every run.

### 3.1 `015deg` is truncated at the top

It lost the first lines of the preamble, and the fourth was cut mid-line, so
the file begins with the fragment:

```
 118  | Hexa address:  0x76
```

This is the **only malformed line in the whole data set**. It is counted and
reported, never dropped silently. The column-name line survived the truncation,
so the file itself reads unambiguously; what was lost is the sensor preamble.

### 3.2 Eight rows carry the BMP280 startup reading

Before the barometric sensor settles it emits a fixed pair, 73157.02 Pa and
23.13 C, physically impossible alongside the 93600 Pa of the rest of the run.

| File | Startup rows |
|---|---|
| `000deg` | 2 |
| `005deg` | 2 |
| `025deg` | 2 |
| `035deg` | 1 |
| `360deg` | 1 |

Always at the very start of the run. They **do not affect event counts or
times**, so no rate changes because of them; they are flagged because they
invalidate any use of pressure or temperature in those rows.

### 3.3 The coincidence filter is a no-op

`Coincident == 1` for all 4760 events across all eight files. Filtering on it
changes nothing.

The consequence matters more than the fact: **with no singles recorded, the
accidental coincidence rate is not estimable from this data set.** It can
neither be subtracted nor bounded. It remains one of the three open hypotheses
for the flat profile, and it is not testable with what was saved.

---

## 4. What is NOT known about the setup

None of the following was recorded. None of it is invented, and each absence
has a consequence for what the analysis can claim.

### 4.1 Detector geometry

Neither the scintillator dimensions nor the separation between them is
recorded.

**Consequence, and it is the principal limitation of this analysis: no
geometric acceptance correction is possible.** A two-scintillator telescope
accepts a cone of directions set by its area and separation. A short, wide
telescope accepts nearly a hemisphere, and its count rate is close to
independent of orientation no matter how strongly the underlying flux is
modulated. This is entirely sufficient on its own to explain a flat profile,
and it cannot be checked, corrected, or ruled out without the geometry.

This is the single highest-value improvement available to this repository, and
it requires a measuring tape applied to the apparatus, not more analysis.

### 4.2 Angular convention

It is not recorded whether the angles are zenith or azimuth, what they are
measured from, with what instrument, or to what precision. There is no record
that 0 degrees is vertical.

**Consequence:** the mapping between the file labels and any physical angle is
assumed, not known. If the labels were azimuthal, a flat profile is precisely
what should be expected, and nothing in the analysis would be wrong except its
premise.

### 4.3 Date and place

Not recorded. The pressure of about 93600 Pa is consistent with a moderate
altitude, of order 700 m, but that is an inference from one number and not a
record.

**Consequence:** the absolute rate cannot be compared against a reference
measurement, since sea-level flux references do not apply without knowing the
altitude and overburden.

### 4.4 Alignment and stability between runs

There is no record of how the detector was repositioned between runs, whether
it was disturbed, or whether anything else changed.

**Consequence:** this is what the 360 degree file measures, and it is the
dominant systematic of the analysis.

### 4.5 What the `360deg` file means

Not recorded. Read literally, 360 degrees is the same orientation as 0 degrees,
which makes the file a **repeatability control**: a second measurement of the
same angle.

The analysis treats it as exactly that, and it is the most valuable file in the
set, because it is the only direct measurement of how much the apparatus moves
between runs for reasons that are not statistical:

```
R(000deg) = 1.9984 +- 0.0769 1/s
R(360deg) = 1.8495 +- 0.0804 1/s
ratio     = 0.9255 +- 0.0537    ->  7.45 % apart, at 1.33 sigma
```

Two caveats stated together, because either alone misleads:

- **As an established effect, 1.33 sigma is not significant.** It cannot be
  claimed that the two runs really differ.
- **As the scale of the run-to-run systematic, it is the best information
  available**, and it is the only such information this data set contains.

If instead the label meant a full rotation about a different axis, or a
different physical configuration, this interpretation fails. There is no way to
tell from the data, and the report says so.

---

## 5. The three hypotheses for the flat profile

The measured profile is flat, non-monotonic, and gives `I(90)/I(0) = 0.806 +-
0.049` where `cos^2` predicts 0. Three explanations are consistent with the
data, and **this data set cannot distinguish between them**:

1. **Wide geometric acceptance** (4.1). Not checkable: the geometry is not
   recorded.
2. **An accidental component** (3.3). Not checkable: no singles were recorded,
   so the accidental rate cannot be estimated.
3. **An undocumented angular convention** (4.2). Not checkable: there is no
   record of what the angles mean.

In the fitted models these three collapse into the single `B` term of
`mu = T * (S * cos^2 + B)`. Nothing in the analysis separates them, and no
statement is made as though it could.

---

## 6. What would be needed to do better

In descending order of value, and all of it requires returning to the
apparatus, not reanalysing these files:

1. **Measure and record the detector geometry** (4.1). Without it there is no
   acceptance correction, and with it most of this analysis becomes possible.
2. **Record singles rates alongside coincidences** (3.3), which makes the
   accidental rate estimable.
3. **Document the angular convention and how it was set** (4.2), with an
   estimate of the pointing uncertainty.
4. **Repeat measurements at each angle**, not only at 0 degrees. One
   repeatability control gives one number at 1.33 sigma; repeating each angle
   turns the dominant systematic from an estimated scale into a measured one.
5. **Longer runs.** At about 1.8 counts per second, six minutes gives roughly
   4 % statistical error per point, which is already smaller than the 7.45 %
   run-to-run systematic. **Longer runs alone would not help**: the measurement
   is limited by reproducibility, not by statistics, and this is the reason to
   fix point 4 before spending more hours counting.
