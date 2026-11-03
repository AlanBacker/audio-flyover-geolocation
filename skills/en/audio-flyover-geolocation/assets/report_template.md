# Acoustic geolocation report: <footage name>

## Conclusion
- **Location**: <lat, lon> (<place name / landmark, if confirmed>)
- **Error**: about ±<R> m (0.9 score region) | or: strip <end A> — <end B>, width <W> m, bearing <deg>°
- **Recording start time (inferred)**: <UTC ISO> (metadata: <value / none>, difference <Δ>)
- **Confidence**: high / medium / low / unusable — <one-sentence reason>

## Evidence
| Event | tCPA in recording | f0 (Hz) | v (m/s) | d (m) | k=v²/d | Matched flight (hex / registration / type) |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |

- Search scope: <bbox / date / time window>; ADS-B source: <adsb.lol / ADSBx / OpenSky>; altitude datum: <geom / baro+QNH xxxx>
- Funnel: flights <N> → corridor nodes <N> → event 1 <N> → offset <N> → silence <N> → candidates <N>
- Scoring: margin_vs_2nd = <x>, robust_z = <x>, crossing angle = <x>°
- Figures: spectrogram/fit `e1.fit.png`, `e2.fit.png`; model overlay `result.rank1.*.png`; heatmap `result.heatmap.png`

## Ambiguity and alternatives
- <ambiguous_alternatives list: coordinates, score ratio, distance, same flight or not>
- <mirror / strip explanation and the information needed to resolve it>

## Independent corroboration
- <street view / terrain / sun / ambient sound etc., or "not done">

## Limitations
- ADS-B coverage: <low-level GA may be missing / what was checked this time>
- Sound speed / wind: <temperature source, wind speed>
- Footage processing: <signs of editing, transcoding, speed change>
- Other cases that cannot be excluded: <...>

## Method
Doppler curves (emission-time model) × historical ADS-B tracks, 3D slant distance + DEM, inter-event offset and
silence constraints, locked scoring and local refinement (audio-flyover-geolocation skill; method after colsto's
2026-11-02 video).
