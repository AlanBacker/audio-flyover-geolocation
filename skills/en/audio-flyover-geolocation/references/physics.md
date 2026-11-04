# Physics model (Doppler fly-over)

## Contents
1. Notation and the exact relation
2. Closed form for a straight constant-speed pass
3. Reading parameters off the curve
4. Errors and degeneracy (why k = v²/d)
5. Sound speed, temperature, wind
6. Non-ideal motion: turns, acceleration, climbs
7. Sources: propellers, helicopters, multi-engine
8. Numerical checks

---

## 1. Notation and the exact relation

- τ: emission time (when the aircraft produced the sound). t: reception time (when the microphone heard it).
- R(τ): slant range from observer to aircraft. Ṙ = dR/dτ (positive when receding).
- Propagation: t = τ + R(τ)/c.
- For a stationary observer in still air:

  f_obs(t) = f0 · dτ/dt = f0 / (1 + Ṙ(τ)/c)

Ṙ must be evaluated at the **emission time**. Using Ṙ at reception time ("where the aircraft is now") gives a
curve that is not just delayed but **differently shaped**, especially at short range and high speed.
`doppler_from_track` in the scripts computes (t_rx, f, R) exactly for any sampled trajectory, and all scoring uses it.

## 2. Closed form for a straight constant-speed pass

Speed v, CPA slant distance d (3D minimum distance), t_c = reception time of the sound emitted AT the CPA
(where f = f0):

```
T  = t − (t_c − d/c)
s  = [c²T − sqrt(c²v²T² + d²(c² − v²))] / (c² − v²)      # s = τ − τ_cpa
R  = c (T − s)
Ṙ  = v² s / R
f  = f0 / (1 + Ṙ/c)
```
A steady climb or descent is still a straight line in 3D, so the formula still holds (v = 3D speed,
d = 3D minimum distance).

## 3. Reading parameters off the curve

- Asymptotes: f_hi = f0/(1 − v/c), f_lo = f0/(1 + v/c)
- v = c (f_hi − f_lo)/(f_hi + f_lo)
- f0 = 2 f_hi f_lo/(f_hi + f_lo) (the harmonic mean, not the arithmetic mean)
- Slope at CPA: df/dt = −f0 v²/(c d) ⇒ d = f0 v²/(c |slope|)
- Intuition: **v sets how far the tone drops, d sets how fast it drops.**

Example (the video's numbers): f0 = 130 Hz, v = 62.7 m/s, c = 343 m/s → f_lo = 130/(1+0.1828) ≈ 109.9 Hz.

Harmonics: if you traced the n-th harmonic, f0 → n·f0; v and d are unchanged (the ratios do not involve f0).

## 4. Errors and degeneracy

- **Proportionality to c**: v ∝ c and d ∝ c (d = f0 v²/(c|slope|) with v ∝ c ⇒ d ∝ c). A 2 % sound-speed
  error ⇒ 2 % error in each of v and d.
- **v–d degeneracy**: when the asymptotes are unclear (tails buried in noise, removed by gating, or cut off by
  the box), the fit slides along v²/d ≈ constant. Example: truth v=68.1, d=1422 (k=3.26), fit v=59.8,
  d=1055 (k=3.39): v is off by 12 %, d by 26 %, k by only 4 %.
- `search_constraints` therefore outputs:
  - a wide speed window (default ±20 % + the c uncertainty);
  - a narrow **k window** (±8 % or 3σ_k);
  - the d window implied by both, [v_min²/k_max, v_max²/k_min].
- In the search, each candidate aircraft computes k = gs²/d_cpa from **its own ADS-B ground speed** and is
  checked against the k window. This is more precise than bounding v and d independently, and much less likely
  to drop the true answer.
- tCPA uncertainty is usually < 1 s.
- **Fit uncertainties.** Adjacent frames overlap (window/hop), and the residuals of a real pass stay correlated
  for longer still (a turn, a power change, a tracer following a smooth wrong path). The script measures the
  residual correlation time τ = 1 + 2·Σρ_k (summed until ρ drops below 0.05) and inflates the covariance by
  max(win/hop, τ). They are not multiplied, because both describe the same correlation. The fit output has
  `resid_corr_time`, `sigma_inflation` and `n_eff`. In the self-test scenarios τ is 3–38 points, so sigmas
  grow by up to 2.7× over the overlap factor alone; on straight synthetic passes τ ≈ 2 and nothing changes.
- **Sigma is statistical only.** An acceleration before CPA biases the fitted v by far more than sigma (about
  10 σ for the self-tests' event 2). That is why the search relies on wide speed windows and the k window, not
  on sigma. `corr_v_d` (typically 0.8–0.9) shows how strongly v and d trade off along k = v²/d.

## 5. Sound speed, temperature, wind

- c = 20.0468 · √(T + 273.15) (dry air, m/s): 15 °C → 340.3, 25 °C → 346.2. Humidity contributes < 0.5 %.
- Take the temperature from METAR or reanalysis data (ground level, at recording time). Under an inversion the
  near-ground temperature differs from that at aircraft height, and a path average is better. If unsure, keep
  `--c-uncert 0.04`.
- Uniform wind w: effective sound speed along the ray is c_eff = c + w·n (n = unit vector source → observer).
  Downwind and upwind sides become asymmetric, which can be misread as a speed change. In strong wind
  (> 8 m/s), widen `--d-margin` and note it in the report.

## 6. Non-ideal motion

| Motion | What the curve shows | Handling |
|---|---|---|
| Turn after CPA | One asymptote not flat; left/right residual asymmetry | `fit` adds an asymmetry note; scoring uses the real track, and the turn actually **removes the mirror** |
| Acceleration/deceleration, change of climb rate | Different "apparent speeds" on approach and recede | `fit_split` (separate v per side) + union of windows; rely on the k window |
| Orbiting/hovering (helicopter) | Curve may rise or fall | `trace --no-monotone`; usually better to use another event |
| Turning towards the microphone before CPA | Curve rises for a while, then falls | `trace --no-monotone` (`trace` hints at it when the default trace keeps only a short piece) |

## 7. Sources

- Propeller BPF = blades × RPM / 60, rich in harmonics. Typical values:
  - C172 (2 blades, 2300–2700 rpm): 70–90 Hz; PA-28 is similar.
  - DHC-8-400 turboprop (6 blades, ~1020 rpm cruise): ≈ 102 Hz.
  - ATR 72 (6 blades, ~1200 rpm takeoff / ~990 cruise): ≈ 100–120 Hz.
  - King Air 350 (4 blades, ~1700 rpm): ≈ 113 Hz.

  These are order-of-magnitude references. Actual RPM varies with power setting, so do not use them as a hard
  type filter.
- Helicopter main-rotor BPF is 10–25 Hz, which phone microphones often attenuate. The tail rotor (50–120 Hz)
  is easier to use.
- Twins: two close but distinct lines (props not synchronised) → trace just one.
- Jets: mostly broadband, occasionally fan tones in the kHz range that attenuate strongly with distance. Not
  used as events by default, only for the silence check.

## 8. Numerical checks

`physics.doppler_linear` and `doppler_from_track` should agree on a straight trajectory (error < 1e-6 Hz).
The synthetic recordings in `selftest` are generated with exact emission-time physics and can be used to
regression-test any change.
