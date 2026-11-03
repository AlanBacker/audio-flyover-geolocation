# Spectrogram analysis and curve tracing

## 1. Parameter choice

| Parameter | Default | Notes |
|---|---|---|
| `--fs` | 4000 | Resample rate; the tones are 20–600 Hz, so 4 kHz is enough and fast |
| `--win` | 1.0 s | Frequency resolution ≈ 1/win. Use 0.5–1 s for close, steep sections (slope > 3 Hz/s); 2–4 s for distant, faint, gentle curves |
| `--hop` | 0.2 s | Frame step; adjacent frames are correlated, so the fit covariance is inflated by √(win/hop) |
| `--fmin/--fmax` | 20/600 Hz | Helicopter main rotors need `--fmin 8` and a longer window |
| `--bg` | 120 s | Background median window (whitening) |

Slope versus window trade-off: the frequency change within one window, Δf = |slope|·win, should be ≲ 1/win,
i.e. win ≲ 1/√|slope|. Example: a CPA slope of 2 Hz/s → win ≲ 0.7 s.

## 2. Whitening and gating

- **Whitening**: each frequency bin is divided by its running-median background (`--bg`), then every frame is
  flattened. Narrow-band tones stand out; broadband noise (wind, traffic, jets) is flattened.
- **Transient gating** (`gate_transients`): frames where an unusually high fraction of bins is "occupied"
  (speech, knocks, wind gusts) are marked as gated. They are excluded from detection, trace output and fitting,
  and show as black vertical bars in the spectrogram.
- Videos with a lot of speech have many gated frames. That is fine as long as the curve is continuously visible
  in the gaps between them.

## 3. Detection (`detect`)

- In sliding 60 s windows (`--window`, `--step`; the last window is aligned with the end, so the tail is always
  examined), a Viterbi search finds the strongest monotone descending path.
- By default it follows the brightest single line, whichever harmonic that is (`--harmonics 1`). All candidates
  in a band share one harmonic count, the largest K with K × fmax under the spectrogram top, so 40–400 Hz under
  a 600 Hz spectrogram stacks only one anyway. Per-candidate counts were tried and dropped: rows stacked over
  different counts have different noise spread, and both detection and the self-test traces got worse.
  `boxes.diag.json` and `trace` report the count actually used. Stacking pays off in a narrow `trace` box.
- Each window's mean path strength is judged against a baseline (`boxes.diag.json` says which):
  - **relative** (normal case): the recording's own window median, by `--thr` dB (default 0.8) and by 3.5 robust
    standard deviations. Needs at least 6 usable windows, most of them background.
  - **noise-model** (clips shorter than about 2 min, or tonal content through most of the recording): the same
    pipeline run on synthetic Gaussian noise for the same window length, by `--null-db` (default 0.8). Whitening
    makes that level independent of the noise's loudness and colour. It matched the background windows of all
    six self-test recordings within ±0.06 dB. In 73 synthetic noise-only clips (12–300 s, with hum and speech
    bursts) there were no false events, and passes within 1.5 km exceeded it by 1.5–12 dB. Real wind and traffic
    noise were not part of that test.
- Hit windows are chained into groups. Because windows overlap, a group can span two separate passes, so each
  group's ridge is split again:
  - where it loses support (its 5 s mean below the 99.5th percentile of the noise model's) for more than 10 s
    of ungated audio;
  - at its turning points, where it rises again by more than max(3 Hz, 3 %). The flat approach and departure
    stay with their pass.

  A group with a single stretch keeps the window-based box of earlier versions (the six self-tests are
  unchanged). Stretches lasting ≥ `--min-dur` become events; shorter ones are listed as `short_hits`.
- Trend: `drop_pct` compares the medians of the first and last 20 % of an event's supported ridge points.
  - ≥ 1 % → `descending`.
  - ≤ −1 % → `rising`. It is not returned but listed under `rejected` in `boxes.diag.json`: a straight fly-over
    always falls, so a rising tone is a test signal, an engine run-up or a turning source. A strong tone hides
    what lies under it, so its ridge is masked and the stretch searched again. A falling ridge found there
    becomes an event (with a `note`) or a `short_hits` entry, and the tone gets a warning.
  - Otherwise → `flat`. It is kept, but check it with `plot`.
- `warnings` on an event: its ridge rises again by more than the tolerance, or falls by more than 45 % (one
  pass falls by 2v/(c+v), which is 36 % at 75 m/s). It then probably joins two aircraft or two harmonics.
- Measured on synthetic recordings:
  - Two passes 90–180 s apart came out as two events in 12 of 12 cases (v1.1.0: 4 of 12, only at 180 s).
  - Passes 40–60 s apart that are audible at the same time often stay one box, but every such box carried a
    warning.
  - Single passes (150 m–2 km, 144 clips) and 72 noise-only clips gave the same detections as v1.1.0. There
    were no warnings, and no ridges were set aside.
- `status` in `boxes.diag.json`: `ok` = an empty result means no fly-over the detector can see; `limited` = the
  noise model was used, or most frames are gated; `insufficient` = the check could not run as configured (clip
  shorter than `--min-dur`, or almost everything gated), so an empty result says nothing.
- Sorting the output by `score_z` (a robust z-score of relative strength) is more meaningful than by time.
- **Detection is only a suggestion**: always confirm with `plot` and adjust the boxes yourself.

## 4. Tracing (`trace`)

Two passes:
1. **Find the strongest segment**:
   - Stack harmonics inside the box, then find a monotone Viterbi path (`--penalty` penalises slope changes).
   - Test significance against a **null path computed on time-shuffled frames**. It has the same
     max-selection bias but no real continuity; its 99th percentile is the threshold.
   - Keep only the strongest contiguous segment.
2. **Re-trace the whole curve**:
   - Fit that segment with the linear model.
   - If the fit is physically plausible (5 < v < 0.6c, 30 < d < 20000 m), build a corridor along the model
     curve and re-trace the whole curve. The threshold relaxes to the null path's 90th percentile, which
     recovers both tails.

### Diagnosis

| What you see | Meaning | Action |
|---|---|---|
| Green points jump from one harmonic to another | Box too wide; harmonic strengths alternate | Narrow the box to one harmonic (bracket it with fmin/fmax) |
| Green points drift into noise before/after the curve | Tails too faint | Normal; fit uses a robust loss. If it hurts the fit, use `fit --tmin/--tmax` to fit only the clean part |
| Only a short piece near CPA | Asymptotes not visible | v and d are individually unreliable, but k = v²/d still holds; the search windows are designed for this |
| A "step" mid-curve | Two aircraft overlapping, or a power change | Separate them with `--anchors`; if needed use only one side |
| A rising curve | Turning towards you / helicopter manoeuvring | `detect` lists it under `rejected`; trace it with `--no-monotone` if needed; usually unsuitable as a primary event |

### Manual anchors

`--anchors "200:255,230:210,250:186"` gives a few (recording seconds, Hz) points. Tracing is restricted to
±`--corridor` Hz around the curve interpolated through the anchors. For very faint curves this is the most
reliable approach: zoom in with `plot` first and read off 3–5 points.

## 5. Identifying harmonics

- One event shows a set of parallel (same-ratio) curves at f0, 2f0, 3f0, … Check that their frequencies at the
  same moment are in integer ratios.
- Which one is strongest depends on aircraft type and distance. Trace the cleanest one;
  `score --harmonics 1,2` uses both 1× and 2× the traced line.
- 50/60 Hz mains hum and its harmonics are **horizontal** lines (possibly with a small ENF wobble); don't trace
  them by mistake. ENF itself can be used for time forensics, but that is outside this skill's scope.

## 6. Reading the plots (for Claude)

When you open a PNG with Read, check:
- that the curve is smooth and monotone, with plateaus (asymptotes) on both sides;
- that the green points sit on the bright line;
- that the cyan fit line matches through the CPA and on both sides.

If it isn't clear, use `plot` to zoom to ±60 s and ±40 Hz and look again.
