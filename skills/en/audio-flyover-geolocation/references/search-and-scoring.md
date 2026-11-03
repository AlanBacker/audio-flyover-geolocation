# Search funnel and scoring

## Contents
1. Grid and track resampling
2. Funnel A: event-1 corridor
3. Funnel B: inter-event offset
4. Funnel C: silence
5. Scoring: exact model + locking
6. Refinement, hypotheses, ambiguity
7. Reading the output fields
8. Tuning and performance

---

## 1. Grid and track resampling

- An implicit global grid (`Grid(spacing)`, default 200 m) with stable node ids. Only nodes inside track
  corridors are instantiated.
- ADS-B tracks are resampled to 1 s: lat/lon by PCHIP, altitude linearly. Velocity is built from the reported
  gs/track/vrate as ENU, then converted to ECEF. Tracks are split at gaps (> 45 s) and never interpolated across
  a split.
- Observer = grid node lat/lon + DEM elevation + microphone height (default 2 m). All 3D slant distances are
  computed in ECEF.

## 2. Funnel A: event-1 corridor

1. Take only track samples whose class is in `--event-classes` (default prop,rotor,unknown) and whose ground
   speed is in [v_min−3, v_max+3] m/s. With a known recording clock, also restrict to
   rec_start + t_rec ± (clock tolerance + t_tol + d_max/c + 60 s).
2. Generate corridor nodes of half-width d_max around those samples (`corridor_nodes`).
3. For every node, find each flight's CPA (`cpa_events`: sparse distance matrix → local minima → parabolic
   interpolation of R² for a sub-second tCPA). Compute the reception time t_rx = tCPA + d/c (the "delayed CPA",
   exactly the moment the observed curve passes f = f0).
4. Keep a match if:
   - the CPA is not at a track endpoint (`edge`);
   - d ∈ [d_min, d_max] and gs ∈ [v_min, v_max];
   - **k = gs²/d ∈ [k_min, k_max]**;
   - with a known clock, |t_rx − (rec_start + t_rec)| ≤ tolerance + t_tol.

## 3. Funnel B: inter-event offset (two events)

- ΔT = t_rec2 − t_rec1: the difference between the f = f0 times of the two curves. It depends only on the
  recording itself and is accurate to ~1 s.
- On the nodes left by A, repeat CPA matching for event 2, then pair by node: |(t_rx2 − t_rx1) − ΔT| ≤ tol.
  tol defaults to √(t_tol1² + t_tol2²) and can be set with `--offset-tol`.
- `--distinct-aircraft` forbids both events being the same aircraft. It is allowed by default, because an
  aircraft doing two circuits is common.
- This step turns "absolute time unknown" into "relative time known". The 920 s offset in the video does its
  work here.

## 4. Funnel C: silence

- Silence filtering is enabled only when `--rec-duration` (total recording length, seconds) is given.
- Quiet intervals: by default the whole recording minus each event ± guard (`--guard` 25 s) plus the curve
  half-width. Or set them explicitly with `--quiet "a-b,c-d"`.
- The silence check applies to **all** flights, including those outside the bbox, since they can still pass
  near a node.
- A candidate (node + time alignment) is dropped if any prop/rotor/unknown/jet flight passes within `--d-quiet`
  slant distance during a quiet interval. Defaults: prop 3000, rotor 2500, unknown 3000, jet 2500 m.
- Risk: the recording actually contains a weaker fly-over that wasn't detected, and the true answer gets killed.
  With lots of speech or wind, give only intervals you are sure are quiet, or lower d_quiet.

## 5. Scoring

For each candidate (node, flight 1, [flight 2]):
1. `CurveModel.curve` takes the flight's resampled track over CPA ± (half_window+20) s and computes (t_rx, f/f0)
   exactly by emission time.
2. Align event 1's model t_rx(CPA) with the observed t_rec1 to get off0 (the UTC of recording second 0). Then
   search δ ∈ [−delta_max, +delta_max] (default ±4 s; coarse 1 s / fine 0.25 s).
   - **Both events share δ** (`--lock joint`).
   - With `e1`, δ is chosen from event 1 alone.
3. Each event's f0 is scaled independently over 1 ± f_rel (default ±0.6 %; 9 coarse / 25 fine steps).
4. Score = mean strength of the whitened spectrogram sampled along the model curve (and its `--harmonics`
   multiples), clipped to 0–30 dB. `--trace-mode` switches to a Gaussian distance to the traced points instead
   (for very faint curves).
5. With a known clock (`--rec-start`), δ outside ±`--rec-start-tol` is forbidden.

Why lock: if each event aligns its time separately, a wrong location can also "shift" each curve to its best
fit, and scores stop discriminating. A shared δ keeps the strongest constraint, the purely physical time
difference between the two events.

## 6. Refinement, hypotheses, ambiguity

- After coarse scoring, each flight pair keeps up to `nodes_per_pair`=4 nodes more than 400 m apart (so both
  mirror sides survive). The best `--top-pairs` (40) flight pairs are kept.
- Every kept node gets a shrinking 3×3 hill-climb: step `--sub-m` (100 m) → 50 → 25 m, until the centre is best.
- Each flight pair outputs up to 3 hypotheses more than 500 m apart; all are ranked by total score.
- `ambiguity`: other hypotheses with score ≥ `--ambiguity-frac` (0.9) × best and ≥ 500 m away →
  `ambiguous_alternatives`.
- `coarse_plateau`: all nodes of the best flight pair with coarse score ≥ 0.9 × max, summarised as principal-axis
  length, width and bearing.
  - Single event, no clock: this is the along-track strip (≈ 2·v·Δt).
  - When a mirror exists, the width spans across the track.
- `geometry`: bearing of each event's track at CPA and the crossing angle. < 25° gives a warning (along-track
  degeneracy).
- Heatmap: fine scoring within `--heat-radius` (1500 m) of the best point at `--heat-step` (30 m). The area and
  maximum radius of the 0.9 region are the error bar.

## 7. Reading the output fields

| Field | Interpretation |
|---|---|
| `best.total` | Meaningless on its own; only for comparison |
| `margin_vs_2nd` | (1st − 2nd)/1st; > 0.15 strong, 0.05–0.15 medium, < 0.05 weak |
| `robust_z` | Robust z of the 1st place against all hypotheses; > 5 to count as significant |
| `ambiguous_alternatives` | Non-empty → must be listed in the report and ruled out visually / by terrain if possible |
| `coarse_plateau.length_m` | Single event: > 600 m → report a strip; compare with 2·v·clock tolerance to tell whether it is clock degeneracy. With two events coarse scoring discriminates poorly; rely on the heatmap and `geometry` |
| `region_max_radius_m` | The reported error radius (0.9 region) |
| `best.rec_start_utc` / `rec_start_iso` | Inferred recording start in UTC; cross-check with metadata |
| `best.flight1/flight2` | hex / registration / type / class |
| `funnel` (search stats) | Count remaining at each stage; a stage collapsing to 0 → check that stage's window |

## 8. Tuning and performance

- Speed: a 200 m grid, 50×50 km, 4 h of data and 70 flights take about 30–45 s end to end in the synthetic test
  (single-core Python). A country-wide search needs tiling by region (`max_nodes` caps at 3 million nodes), or a
  known clock to narrow the time window first.
- If the funnel is empty, relax in this order:
  1. `--c-uncert 0.06`
  2. `fit --k-margin 0.12 --v-margin 0.3`
  3. `--event-classes prop,rotor,unknown,jet`
  4. check `--start/--end` and the QNH / altitude datum
  5. check ADS-B coverage.
- Too many candidates (> 100k): shrink the bbox, give a clock, or add a second event.
- Never tighten so far that the true answer could be excluded. Keep the funnel wide and let scoring decide.
