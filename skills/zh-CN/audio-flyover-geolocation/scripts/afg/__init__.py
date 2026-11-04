"""afg - Acoustic Flyover Geolocation toolkit.

Locate where an audio/video recording was made by matching faint aircraft
Doppler curves (and optionally thunder) in its soundtrack against public
flight-tracking (ADS-B) and lightning data.

Modules
-------
common   constants, time parsing, JSON helpers
geo      WGS84 geodesy, ENU frames, implicit search grid
dem      elevation model sampling (GeoTIFF via rasterio, or constant)
physics  sound speed, Doppler models, curve fitting
audio    audio loading, chunked STFT, whitening, ridge tracking, detection
adsb     readsb / ADS-B Exchange / adsb.lol / OpenSky ingestion, classification
tracks   per-flight resampling to a uniform 1 s ECEF time base
search   candidate funnel: CPA events, offset filter, silence filter
scoring  emission-time Doppler modelling and locked two-event scoring
thunder  thunder-envelope geolocation (lightning data)
simulate synthetic scenario generator used by the self-test
"""

__version__ = "1.2.0"
