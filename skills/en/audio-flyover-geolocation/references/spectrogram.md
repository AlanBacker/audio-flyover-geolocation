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

- In sliding 60 s windows, a Viterbi search finds monotone descending paths, stacking 3 harmonics.
- A window counts when its mean path strength exceeds the whole-recording median by the threshold (`--thr` dB,
  default 0.8). Windows lasting ≥ `--min-dur` are merged into event boxes.
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
| A rising curve | Turning towards you / helicopter manoeuvring | `--no-monotone`; usually unsuitable as a primary event |

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
