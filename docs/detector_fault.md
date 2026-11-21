# The detector fault of 2026, and how to tell it apart

The detector of the 2026 altitude campaign ([dataset.md](dataset.md) section 7)
had a mechanical fault. Its board sits loose inside the case and, pushed into
some positions, sends the detector into a state of very high count rate that
persists while the board stays there. The group that took the data documented
the symptoms in an internal report of 18 November 2026, and also listed the
candidate causes: a light leak, light from the detector's own LED reaching the
sensor, or an electrical contact that changes with the position of the board.
That report is the source of the episodes and of the mechanical history. What
follows is the evidence in the data, which is what this repository can check,
produced by `scripts/07_detector_health.py` and governed by methodology
section 9.

Nothing in the files flags the fault. The question was whether the two pulse
readings each event carries are enough to recognise it. They are.

---

## 1. Two observables, not three

Each event carries `ADC1` (low-gain channel), `ADC2` (high-gain channel) and
`SiPM[mV]`. The third is computed by the firmware from the other two: across
588 328 events of the three data sets, 502 159 share their `(ADC1, ADC2)` pair
with another event, and inside every such pair the SiPM value is identical. It
adds nothing.

For all but the largest pulses, ADC1 barely leaves the baseline of its channel:
it only starts to follow the pulse once ADC2 approaches saturation (4095). The
median ADC1 of a minute, over events with `ADC2 < 1500`, is therefore a reading
of the baseline of the electronics, the **pedestal**, and not of the pulses.
That makes it a monitor of the state of the detector.

## 2. Two signatures

| State (Tacachico unless named) | Rate (1/s) | ADC1 median | ADC2 median | ADC1 > ADC2 | gaps of 10 ms or less |
|---|---|---|---|---|---|
| Santa Tecla, whole file | 1.74 | 107 | 1096 | 0.0 % | 0.8 % |
| hours 26 to 40 (reference window) | 1.44 | 144 | 633 | 0.0 % | 0.7 % |
| burst, minutes 70 to 73 | 133.5 | 339 | 230 | 96.9 % | 97.7 % |
| all ten high-rate episodes | 27.6 | 190 | 217 | 24.2 % | 33.8 % |

**Burst.** In one of the ten episodes, minutes 69 to 74, the detector fires
every 6 to 10 ms, the dead time of its own readout, and in 93 % of its events
the low-gain channel reads more than the high-gain one (97 % in minutes 70 to
73). Two more episodes carry it in part: minutes 66 and 67 (37 % of events,
gaps of 7 ms) and 689 to 697 (28 %, gaps of 20 ms). A real pulse never reads
more on the low-gain channel; clean data do so in 0 to 0.1 % of events. This
signature separates fault from signal event by event.

**Slow fault.** In the other seven, the pedestal of ADC1 climbs from its usual
100 to 150 up to 170 to 200, and the extra events pile up just above the ADC2
trigger threshold of 201. Live rate per ADC2 band, median of 10-minute bins of
Tacachico, against the pedestal of the bin:

| Pedestal | Total | [200, 260) | [260, 400) | [400, 700) | [700, 1200) | [1200, 2500) | [2500, 4095) | 4095 |
|---|---|---|---|---|---|---|---|---|
| 90 to 110 | 1.48 | 0.127 | 0.225 | 0.396 | 0.233 | 0.347 | 0.102 | 0.026 |
| 150 to 170 | 1.64 | 0.260 | 0.261 | 0.388 | 0.242 | 0.317 | 0.117 | 0.035 |
| 170 to 185 | 2.08 | 0.654 | 0.282 | 0.394 | 0.239 | 0.334 | 0.120 | 0.033 |
| 185 to 200 | 25.6 | 23.3 | 1.16 | 0.371 | 0.237 | 0.296 | 0.110 | 0.032 |

Between the first and the last class the total rate grows by a factor of 17
and the band just above the threshold by a factor of 183, while the bands from
ADC2 = 400 up change by -15 % to +27 %. Those changes are real, and this table
does not explain them: the classes also differ in time and in case
temperature, and the gain of the sensor depends on the latter. They are two
orders of magnitude below the excess just above threshold, which is the
fault.

A reading consistent with this, **not established by it**: a shift of the
baseline of the electronics that lowers the effective trigger threshold, so
that noise which normally stays under it crosses it, more and more steeply as
the shift grows. It would fit the electrical hypothesis of the group's report.
The files contain no waveforms and no voltages, and the data cannot rule out
the optical ones.

The pedestal is not a function of the case temperature alone: at the same
temperature it differs by tens of counts between hours of Tacachico (118 to 155
at 36 C, 5th to 95th percentile of the minutes kept), while in Santa Tecla 98 %
of the minutes sit between 95 and 114 over a range of 7 C. In the last hours
of the reference window it creeps up to about 170, which is where the
near-threshold rate starts to grow.

## 3. Cleaning

A minute is vetoed if its pedestal exceeds 170, more than 1 % of its events read
`ADC1 > ADC2`, or more than 60 of its events fall in ADC2 `[200, 260)`, plus one
guard minute on each side. The third criterion is there because the episode at
minute 319, with a pedestal of 167, escaped the other two. After the veto,
events are kept with `ADC2 >= 260`.

On Tacachico this removes 311 of 2454 minutes (12.7 %), every minute of the ten
episodes among them. The hourly live rate, tested against a constant:

| Selection | Hours | chi2 / dof | Excess over Poisson |
|---|---|---|---|
| raw: every trigger, every minute | 39 | 143 736 / 38 | |
| veto only | 36 | 509.6 / 35 | 5.6 % |
| veto and `ADC2 >= 260` | 36 | 55.5 / 35 | 1.2 % |

The cleaning brings the whole acquisition close to constant, **not to
constant**: chi-square 55.5 / 35 has p = 0.015, and the reference window
(1.384 /s) and the hours outside it (1.359 /s) still differ by 1.8 %, 3.3 sigma
from the hourly spread, at the same mean case temperature. The residual 1.2 %
excess over Poisson is what the cleaned altitude ratio carries as its
systematic (methodology 11.7). On Santa Tecla, which has no episode, the veto
removes 6 of 1926 minutes and the hourly test stays where it was (36.4 / 30
before the cut, 39.3 / 30 after).

**The thresholds were chosen by looking at Tacachico** (methodology 9.6). This
validation is a consistency check, not a measured efficiency, and a detector
with other settings would need its own.

## 4. The 2025 angular runs

None of the eight 2025 runs shows either signature: no event reads
`ADC1 > ADC2`, no minute is vetoed, the pedestal stays between 69 and 100, the
near-threshold rate is 0.09 to 0.14 /s (the cleanest 2026 state gives 0.11 to
0.13), the fraction of gaps of 10 ms or less is what a Poisson source gives
behind the 5.6 ms dead time (0.2 to 1.2 % against 0.7 to 0.9 %), the live
waiting times pass a KS test against an exponential (p from 0.07 to 0.71, by
Monte Carlo because the scale is estimated from them), and the counts in 10 s
bins are Poisson. The two campaigns share the trigger threshold and the dead-time
pattern of their firmware, so the signatures carry over. The 2025 runs were not
taken in the fault state, and cleaning them changes nothing: `I(90)/I(0)` goes
from 0.806 +- 0.049 to 0.796 +- 0.050. Why their profile is flat is the subject
of step 08 and methodology section 10.

## 5. Checking a detector before using it

A few minutes of data are enough. If the median ADC1 of events with
`ADC2 < 1500` drifts above about 170, if any noticeable fraction of events
reads `ADC1 > ADC2`, or if the events crowd just above the trigger threshold,
the detector is in its fault state and the board needs to be fixed in place
before the run starts. The `Coincident` flag says nothing about any of this
(methodology 9.7), and neither does the `Coincidence detector found.` line
every preamble prints.
