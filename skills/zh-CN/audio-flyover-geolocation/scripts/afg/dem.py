"""Ground elevation for candidate observer points.

The aircraft altitude in ADS-B is above sea level, the observer is on the
ground: slant distance needs ground elevation at every grid point. Use a real
DEM (Copernicus GLO-30 / USGS 3DEP / SRTM). A constant elevation is accepted
for flat areas or tests but is reported as a precision downgrade.
"""
import math
import os
import urllib.error
import urllib.request
from collections import OrderedDict

import numpy as np

from .common import log

COP30_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def cop30_tile_name(lat_floor, lon_floor):
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def fetch_copernicus(bbox, outdir):
    """Download Copernicus GLO-30 tiles (public AWS open data, no account) covering bbox.
    Ocean-only tiles do not exist (HTTP 404) and are skipped (treated as 0 m)."""
    la0, la1, lo0, lo1 = bbox
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for la in range(math.floor(la0), math.floor(la1) + 1):
        for lo in range(math.floor(lo0), math.floor(lo1) + 1):
            name = cop30_tile_name(la, lo)
            dst = os.path.join(outdir, name + ".tif")
            if not os.path.exists(dst):
                url = COP30_URL.format(name=name)
                try:
                    log("downloading", url)
                    urllib.request.urlretrieve(url, dst + ".part")
                    os.replace(dst + ".part", dst)
                except urllib.error.HTTPError as e:
                    if e.code in (403, 404):
                        log(f"  no tile {name} (ocean?) - treated as sea level")
                        continue
                    raise
            paths.append(dst)
    return paths


class ElevationModel:
    """Bilinear sampling over one or more GeoTIFF tiles (lazy, LRU-cached),
    or a constant. Points outside every tile get `fill` (default 0 = sea)."""

    def __init__(self, paths=None, constant=None, fill=0.0, cache_tiles=6):
        self.constant = constant
        self.fill = float(fill)
        self.tiles = []
        self._cache = OrderedDict()
        self._cache_n = cache_tiles
        if constant is None:
            if not paths:
                raise ValueError("ElevationModel needs DEM paths or a constant")
            try:
                import rasterio  # noqa: F401
            except ImportError as e:
                raise ImportError("Reading a DEM needs `pip install rasterio` (or pass --ground-elev)") from e
            import rasterio
            for p in paths:
                with rasterio.open(p) as ds:
                    b = ds.bounds
                    self.tiles.append((p, b.left, b.bottom, b.right, b.top))
            log(f"DEM: {len(self.tiles)} tile(s)")

    @property
    def is_constant(self):
        return self.constant is not None

    def _load(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            nod = ds.nodata
            tr = ds.transform
        if nod is not None:
            arr[arr == nod] = np.nan
        item = (arr, tr)
        self._cache[path] = item
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return item

    def sample(self, lat, lon):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if self.constant is not None:
            return np.full(lat.shape, float(self.constant))
        out = np.full(lat.shape, np.nan)
        for path, l, b, r, t in self.tiles:
            m = np.isnan(out) & (lon >= l) & (lon <= r) & (lat >= b) & (lat <= t)
            if not m.any():
                continue
            arr, tr = self._load(path)
            inv = ~tr
            col, row = inv * (lon[m], lat[m])
            col = np.asarray(col) - 0.5
            row = np.asarray(row) - 0.5
            c0 = np.clip(np.floor(col).astype(int), 0, arr.shape[1] - 2)
            r0 = np.clip(np.floor(row).astype(int), 0, arr.shape[0] - 2)
            fc = np.clip(col - c0, 0, 1)
            fr = np.clip(row - r0, 0, 1)
            v = (arr[r0, c0] * (1 - fc) * (1 - fr) + arr[r0, c0 + 1] * fc * (1 - fr)
                 + arr[r0 + 1, c0] * (1 - fc) * fr + arr[r0 + 1, c0 + 1] * fc * fr)
            out[m] = v
        out[np.isnan(out)] = self.fill
        return out
