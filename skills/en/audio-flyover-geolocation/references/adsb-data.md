# ADS-B data: sources, formats, datums and gaps

## Contents
1. Data sources and downloads
2. readsb trace format
3. OpenSky and generic CSV
4. Altitude datums (the most common precision killer)
5. Aircraft type classification
6. Coverage gaps and regulation
7. Practical advice

---

## 1. Data sources and downloads

| Source | Coverage / granularity | How to get it | Notes |
|---|---|---|---|
| **adsb.lol globe_history** | Global, one release per day; one `trace_full_<hex>.json` per aircraft | GitHub `adsblol/globe_history_YYYY` releases (large files split into `.tar.aa`, `.tar.ab`, …; `cat` them together before extracting) | Free, ODbL licence; first choice |
| **ADS-B Exchange samples** | Global, **1st of each month only** | `https://samples.adsbexchange.com/traces/YYYY/MM/01/<xx>/trace_full_<hex>.json` (xx = last two hex digits); also `readsb-hist` (5 s snapshots) and hires-traces | Samples are free; other dates need a commercial licence; mind the terms of use |
| **OpenSky Network** | Global (wherever receivers cover); state vectors ~1 Hz (archive resolution varies by year) | Trino (`trino.opensky-network.org`, table `minio.osky.state_vectors_data4`, partitioned by `hour`); Python: `pyopensky` / `traffic` | Needs an account with research access |
| Your own readsb / tar1090 | Local station | `/var/globe_history/YYYY/MM/DD/traces/` | Best low-level coverage |

Download only what you need: the **date + time window ±1 h + area padded by ~0.5°**. A single day of
adsb.lol's global archive is GB-scale. Extract it, then filter with `adsb-ingest --bbox`; the script globs
`trace_full_*` recursively and detects gzip automatically.

OpenSky Trino example query (note the `hour` partition condition, otherwise it is extremely slow):
```sql
SELECT time, icao24, lat, lon, velocity, heading, vertrate, baroaltitude, geoaltitude, onground
FROM minio.osky.state_vectors_data4
WHERE hour BETWEEN 1789221600 AND 1789239600          -- whole-hour unix seconds
  AND time BETWEEN 1789223000 AND 1789238000
  AND lat BETWEEN 38.5 AND 39.7 AND lon BETWEEN -95.4 AND -93.8
```
Export to CSV, then run `adsb-ingest --format opensky --src export.csv`.

## 2. readsb trace format

Top level of the file: `icao` (hex), `r` (registration), `t` (ICAO type code), `desc`, `timestamp` (base unix
seconds), `trace` (array). Each point (wiedehopf/readsb `README-json.md`):

```
[dt, lat, lon, alt_baro_ft | "ground" | null, gs_kt, track_deg, flags, vrate_fpm,
 details | null, source, alt_geom_ft, geom_rate_fpm, ias_kt, roll_deg]
flags: 1 = stale, 2 = start of new leg, 4 = vrate is geometric, 8 = altitude is geometric
```
What the script does:
- t = timestamp + dt.
- It starts a new segment on flags&2 or on a gap > `--max-gap` (45 s), and drops ground points.
- Velocity comes from the reported gs/track/vrate, which is much smoother than differenced positions.
  Savitzky-Golay differencing is used only when those fields are missing.

## 3. OpenSky and generic CSV

- `--format opensky`: columns time, icao24, lat, lon, velocity (m/s), heading, vertrate (m/s), baroaltitude (m),
  geoaltitude (m), onground.
- `--format generic`: `hex,t,lat,lon,alt_m,gs_ms,track_deg[,vr_ms,type]`, with alt_m taken as MSL.

## 4. Altitude datums

Slant distance d is almost 1:1 sensitive to altitude error when the aircraft passes near overhead.

- **Pressure altitude (alt_baro)**: altitude in the standard atmosphere relative to 1013.25 hPa, not true
  altitude.
  - True ≈ baro + (QNH − 1013.25) × 8.3 m, plus a temperature error: on cold days true altitude is lower.
  - Take QNH from the METAR of the nearest airport at recording time (e.g. the Iowa State Mesonet ASOS archive)
    and pass `--qnh`.
- **Geometric altitude (alt_geom / geoaltitude)**: GNSS height above the WGS84 ellipsoid (HAE); MSL = HAE − N.
  - N is the geoid undulation. Over the contiguous US it is about −20 to −35 m, so MSL is 20–35 m above HAE.
    Elsewhere it varies (for example about −5 to +5 m in eastern China).
  - Look it up for EGM2008 with GeographicLib GeoidEval and pass `--geoid-n`.
- The DEM (Copernicus GLO-30) is referenced to EGM2008 (≈ MSL). Convert aircraft altitude to MSL before
  subtracting terrain.
- `--alt-source auto`: use geometric altitude where available, otherwise QNH-corrected pressure altitude.

## 5. Aircraft type classification

The scripts use ICAO Doc 8643 description codes, e.g. L1P = land single piston, L2T = land twin turboprop,
L2J = land twin jet, H1T = single-turboshaft helicopter.
- First character: L/S/A/H/G/T = landplane / seaplane / amphibian / helicopter / gyrocopter / tiltrotor.
- Third character: P/T/J/E = piston / turboprop-turboshaft / jet / electric.

The script reads tar1090-db's `icao_aircraft_types.js` (cached in `~/.cache/afg`; a built-in short table is
used offline). It supplements it with the ADS-B emitter category (A1–A7, B1–B7) to give these classes:
- `prop` (L*P, L*T)
- `rotor` (H*, G*, A7)
- `jet`
- `glider` (B1)
- `balloon` (B2)
- `unknown`

The default event classes for search are `prop,rotor,unknown`; jets only take part in the silence check.

## 6. Coverage gaps and regulation

- US 14 CFR 91.225 requires ADS-B Out in:
  - Class A airspace;
  - Class B/C airspace and above it;
  - the Mode C veil (30 nm);
  - above 10,000 ft MSL (excluding below 2,500 ft AGL).

  **Below 10,000 ft outside these areas, GA is not required to have it.** Small aircraft in rural low-level
  airspace may be entirely absent from the data.
- Military, law-enforcement and some HEMS helicopters may switch off or use anonymous/privacy ICAO addresses
  (PIA). Aircraft on the FAA LADD list are hidden on some platforms, but adsb.lol / ADSBx generally don't filter.
- Receiver coverage: at low level (< 1,000 ft AGL) far from receivers, data is intermittent or missing, worse in
  mountains. Track interpolation across gaps > 45 s is not trustworthy.
- Europe: EASA requires ADS-B (Mode S EHS/ELS) for aircraft above 5.7 t or faster than 250 kt. GA often has
  Mode S without position.
- Conclusion: "no match in the data" ≠ "not in this area". State the coverage assumptions in the report.

## 7. Practical advice

1. Ingest with a wide bbox and time window first, and see how many prop/rotor flights `tracks.npz` contains;
   narrow down if there are too many.
2. Check that prop tracks in the target period have continuous low-level data (plot them in Python if needed).
3. The time window must contain tCPA ± travel delay ± clock uncertainty, plus 10 minutes of margin.
