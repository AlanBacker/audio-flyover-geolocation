# Thunder × lightning-data geolocation (mode T)

This is the method from two earlier colsto videos: *Can you geolocate a video using background thunder sounds?*
(2026-08) and *I located a fan's video using thunder sounds* (2026-09). Precision is at the kilometre level.
It suits recordings with a thunderstorm but no aircraft curve.

## Principle

Each lightning strike at time t_i and position x_i produces thunder. An observer at p hears it at
t_i + |x_i − p|/c, with amplitude falling off with distance. For a candidate point p and recording start time
T0, you can synthesise the "thunder envelope that should be heard" and compare it with the recording's
low-frequency (20–250 Hz) RMS envelope:
- Slant distance: the source height is taken as ~2 km (`--source-h`), r = √(horizontal distance² + h²).
- Amplitude model:
  - A ∝ 1/r^p (`--p`: 1 = amplitude, 2 = energy), with an extra exponential decay;
  - r is floored at 1.5 km so nearby strikes can't dominate;
  - r is capped at 25 km; farther strikes are essentially inaudible.
- Similarity: cosine similarity. A single FFT cross-correlation computes all T0 offsets at once, with
  window-energy normalisation.
- Grid default 2 km. The recording start time is searched within [`--start`, `--end`].

## Data

| Source | Position accuracy | How to get it |
|---|---|---|
| Blitzortung / LightningMaps | ~km (about 3 km off in the video) | Community network. History requires recording the live feed yourself or requesting it; follow their terms of use |
| MET Norway lightning API | ~hundreds of m (about 200 m in the video) | `api.met.no` (needs an identifying User-Agent); covers the Nordics |
| National met services / commercial networks (NLDN, GLD360, ENGLN, …) | ~hundreds of m | Mostly commercially licensed |

Prepare a CSV: `time,lat,lon` (time as ISO or unix seconds).

## Usage

```bash
$AFG thunder --audio storm.mp4 --strikes strikes.csv --bbox 59.5,60.5,10.0,11.5 \
    --start 2026-09-20T15:00Z --end 2026-09-20T17:00Z --out work/thunder.json
```
Outputs candidate points ranked by similarity, with the matching T0.

## Limits and advice

- Kilometre-level precision is normal. It is limited by:
  - lightning location error;
  - source height (cloud-to-cloud vs cloud-to-ground);
  - terrain occlusion, wind and inversions.
- Remote isolated strikes (outside the main storm cell) carry the most information, because they line up only
  at a few positions. The second video weights them for this reason.
- Wind noise, rain and speech contaminate the 20–250 Hz envelope. Cut out the parts where the thunder is clear
  first.
- Use the result as a regional prior, then combine it with aircraft curves (if any) or visual clues.
