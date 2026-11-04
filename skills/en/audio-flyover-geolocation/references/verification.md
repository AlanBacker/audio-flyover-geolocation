# Verification and confidence

An acoustic result is a **hypothesis**, not a conclusion. Complete at least part A before reporting, and grade
it with part C.

## A. Mandatory acoustic self-checks

0. **Trace check**: `best.trace_check.consistent` must be true. With two events, `NO CONSISTENT SITE` means no
   answer: report that, with the causes in section F, not the best-scoring point.
1. **Overlay** (`result.rank1.<event>.png`): the cyan model curve should sit on the bright line in three places:
   - the approach asymptote;
   - the steep drop at CPA;
   - the recede asymptote.

   If it only fits the CPA section, the d/v combination matched by chance: lower confidence by one level.
2. **Overlays for ranks 2 and 3**: if they look as good as rank 1, be cautious however large the `margin` is.
3. **Heatmap** (`result.heatmap.png`): a single peak with a 0.9-region radius ≤ 150 m is good. If it is a strip,
   check `geometry` and `coarse_plateau`.
4. **Recording start time**: does `rec_start_iso` agree with the metadata, upload time and time clues in the
   picture? Metadata is often the export time, which can be minutes to hours off.
5. **Negative check**: at the best point, which other prop/rotor flights passed close during the recording?
   - The script's silence filter already removes "should have heard but didn't".
   - Still confirm by hand that those stretches are really quiet, not masked by speech or wind.
6. **Parameter robustness**: the best point should barely move (< 100 m) when you vary:
   - `--c` by ±2 %;
   - `--harmonics 1` vs `1,2`;
   - `--lock e1`.

## B. Independent (non-acoustic) corroboration

- **Street view / satellite imagery**: buildings, roads, vegetation, utility poles, skyline in the picture.
- **Terrain and viewshed**: compute the horizon profile from the DEM and compare it with distant hills in the
  picture. Check that landmarks seen in the picture are visible from the best point.
- **Sun / shadows**: use a SunCalc-style tool with the inferred recording UTC to check shadow direction and
  length.
- **Ambient sound**: church bells, train horns, PA announcements, particular birds. Road-noise level should fit
  the distance to roads.
- **Breaking the mirror**: the two mirror candidates of a straight single-event fly-over can often be told apart
  by:
  - whether the aircraft crosses the picture left-to-right or right-to-left;
  - terrain occlusion;
  - road positions.
- **Stereo extension**: for two-channel recordings, inter-channel phase/time difference (GCC-PHAT) estimates how
  the aircraft's bearing changes over time.
  - The direction the bearing flips at CPA breaks the mirror symmetry directly.
  - Not automated in this skill; it needs manual analysis.
  - With an unknown microphone orientation, only left/right can be judged.

## C. Confidence scale

| Grade | Conditions |
|---|---|
| **High** | All of: two events (crossing ≥ 25°), or one event + clock ±2 s and not mirrored; trace check passed; margin ≥ 0.15; overlay fits all three sections; heatmap radius ≤ 150 m; at least one independent corroboration |
| **Medium** | Acoustic conditions met but no independent corroboration; or margin 0.05–0.15; or heatmap radius 150–600 m |
| **Low** | Any of: mirror not ruled out; strip > 600 m; margin < 0.05; overlay fits only one section; ADS-B coverage doubtful |
| **Unusable** | Fit implausible, empty funnel, `NO CONSISTENT SITE`, `DEGENERATE`, or several aircraft with indistinguishable scores |

How to write the location in the report:
- High / medium: coordinates + "about ±R m (0.9 score region)".
- Strip: both end coordinates + width + bearing, with the reason ("clock ±30 s, along-track direction
  unresolvable").
- Mirror: both coordinates side by side, with the information needed to tell them apart.

## D. Common sources of false matches

- Several sorties on the same route (circuit training, flight schools) → a wrong flight can also match. Separate
  them with a second event or with time.
- Tones from ground vehicles, drones or boats (v < 25 m/s or d < 150 m).
- Edited recordings: the time difference between two events is changed → the two-event funnel fails. Look for
  edit marks (sudden background noise changes, broken gating patterns).
- Platform speed changes (1.25× playback) or a wrong sample rate scale all frequencies. v and d are unchanged,
  but ΔT and durations are scaled. Check that mains hum is still at 50/60 Hz.

## E. Synthetic blind-test baseline (`selftest`, 6 random seeds)

Scenario:
- a 50–70 km area, 4 h, about 70 flights (45 propeller + 25 jet);
- a ~24 min recording containing two propeller fly-overs (~915 s apart);
- pink noise, mains hum, speech and jets;
- event 2's aircraft accelerates before CPA and climbs after it, deliberately biasing the linear fit.

| Mode | Result |
|---|---|
| Two events, time unknown | 24–53 m error (5/6). In 1 case the two tracks cross at only 3°: 477 m error, but the truth lies inside the reported 0.9 region (radius 849 m) and a warning is given |
| One event, clock ±2 s, pass turning after CPA | 23–96 m (6/6) |
| One event, clock ±2 s, straight pass | 6–213 m (4/6); in the other 2/6 the mirror side scores slightly higher, and the truth is the flagged mirror alternative |
| One event, clock ±30 s | 2–5 km strips, truth always inside the strip |

In every seed the two-event answer passes the trace check (robust RMS 1.0–1.9 × the fit's own).

This is idealised synthetic data (no ADS-B gaps, no wind, no multipath); expect larger errors on real footage.
It shows **that the pipeline is correct and that its uncertainty reporting is honest**. It is not a real-world
accuracy guarantee. After changing code, rerun `selftest --seed 7/11/23/42/101/2026`.

## F. When no site is consistent

Two events that no candidate explains together (`NO CONSISTENT SITE`, or only low scores whose overlays fit one
event) leave several possibilities. The method cannot tell them apart; say so in the report.

1. **The two events are not from one place**, or two recordings were joined with a wrong time offset. Check
   that the recordings really share a spot and how their clocks relate.
2. **An aircraft is missing from the ADS-B data**: no ADS-B Out (not required for much low-level GA in the US),
   not received at low altitude or behind terrain, or excluded by `--event-classes`, `--start/--end` or the
   bbox. Each event alone then still finds "matching" aircraft (any aircraft with a similar speed and distance
   fits one curve), but no point explains both. Nothing in the audio can rule this out.
3. **The site is outside the bbox or the time window.** Widen them.
4. **A trace followed the wrong line** (another harmonic, music, a vehicle). Check every fit overlay.

Next steps: run each event alone with a clock if one is known, and compare where the single-event strips lie.
Strips that never meet point to cause 1 or 2. Look for another fly-over in the recording, or a visual clue that
narrows the area.
