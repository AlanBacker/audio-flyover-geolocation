# Sources

## Method source (colsto)
- colsto, *I discovered a sound that leaks your exact location (and it's everywhere)*, YouTube, 2026-11-02 —
  <https://www.youtube.com/watch?v=UytYrhYLOKI> (full recording: <https://youtu.be/4CxVxuMVkMc>).
  - The description summarises the method:
    - ADS-B + ICAO type classification; a 200 m grid; 3D slant distance + DEM;
    - CPA and propagation delay; the spectrogram constrains v and the CPA slant distance;
    - a 920 s offset + silence constraint; exact emission-time radial-velocity modelling; locked two-event scoring;
    - a 1-minute clip with known time still ranks first in a US-wide search;
    - why high-pass filtering, masking and call compression fail.
  - The author states the code will not be released. This skill's implementation is an independent rewrite.
- colsto, *Can you geolocate a video using background thunder sounds?*, 2026-08-09 — <https://www.youtube.com/watch?v=SnbkIm56TfI>
- colsto, *I located a fan's video using thunder sounds*, 2026-09-30 — <https://www.youtube.com/watch?v=6a7Z0I3LAnE>

## Coverage and discussion
- TechRadar: *YouTuber discovers aircraft noise trick that can geolocate your videos to within 100m* —
  <https://www.techradar.com/computing/cyber-security/youtuber-discovers-aircraft-noise-trick-that-can-geolocate-your-videos-to-within-100m-doppler-effect-pinpoints-locations-from-just-60-seconds-of-audio>
- Simon Willison: *A sound that leaks your exact location* (2026-11-03) — <https://simonwillison.net/2026/Nov/3/a-sound-that-leaks-your-exact-location/>

## Academic background
- B. G. Ferguson, "A ground-based narrow-band passive acoustic technique for estimating the altitude and speed
  of a propeller-driven aircraft," *J. Acoust. Soc. Am.* 92(3), 1403–1407 (1992). The classic method for
  estimating a propeller aircraft's speed, altitude and CPA time from one microphone's narrow-band Doppler
  curve; this skill's v/d/f0 identities agree with it.

## Data and formats
- readsb JSON / trace format: <https://github.com/wiedehopf/readsb/blob/dev/README-json.md>
- Aircraft type table (ICAO Doc 8643 description codes): <https://github.com/wiedehopf/tar1090-db>
  (`db/icao_aircraft_types.js`); ICAO Doc 8643: <https://www.icao.int/publications/doc-8643-aircraft-type-designators>
- adsb.lol history: <https://github.com/adsblol/globe_history_2026> (ODbL)
- ADS-B Exchange sample data: <https://www.adsbexchange.com/data-products/sample-data/> (`samples.adsbexchange.com`,
  1st of each month)
- OpenSky Network Trino: `trino.opensky-network.org` (research access required; Python clients `pyopensky`,
  `traffic`)
- Copernicus DEM GLO-30 (AWS public bucket): <https://copernicus-dem-30m.s3.amazonaws.com/>

## Regulation
- 14 CFR 91.225 (airspace requiring ADS-B Out): <https://www.law.cornell.edu/cfr/text/14/91.225>
