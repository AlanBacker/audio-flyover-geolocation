"""WGS84 geodesy, local ENU frames and the implicit near-uniform search grid.

All slant distances in this toolkit are computed as straight-line ECEF
distances between the aircraft (lat, lon, height) and the observer (lat, lon,
ground elevation + mic height). Heights only need a *consistent* vertical datum
(we use orthometric/MSL for both); the geoid offset cancels to < 1 mm over
the ranges involved.
"""
import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
M_PER_DEG = 111320.0  # only used to lay out grid rows/cols, never for distances


def geodetic_to_ecef(lat, lon, h):
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    h = np.asarray(h, dtype=np.float64)
    s, c = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    x = (n + h) * c * np.cos(lon)
    y = (n + h) * c * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * s
    return np.stack([x, y, z], axis=-1)


def enu_rotation(lat0, lon0):
    """3x3 matrix whose rows are the E, N, U unit vectors expressed in ECEF."""
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([
        [-np.sin(lo), np.cos(lo), 0.0],
        [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
    ])


def enu_vectors_to_ecef(lat, lon, ve, vn, vu):
    """Rotate local ENU vectors (arrays) at (lat, lon) into ECEF components."""
    la, lo = np.radians(lat), np.radians(lon)
    sla, cla, slo, clo = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    vx = -slo * ve - sla * clo * vn + cla * clo * vu
    vy = clo * ve - sla * slo * vn + cla * slo * vu
    vz = cla * vn + sla * vu
    return np.stack([vx, vy, vz], axis=-1)


def ecef_to_enu(xyz, lat0, lon0, h0=0.0):
    ref = geodetic_to_ecef(lat0, lon0, h0)
    return (np.asarray(xyz) - ref) @ enu_rotation(lat0, lon0).T


def radii(lat):
    """Meridional (M) and prime-vertical (N) radii of curvature at lat (deg)."""
    s = np.sin(np.radians(lat))
    w = np.sqrt(1.0 - WGS84_E2 * s * s)
    return WGS84_A * (1.0 - WGS84_E2) / w**3, WGS84_A / w


def offset_latlon(lat0, lon0, east_m, north_m):
    """Small-offset (<~20 km) ENU -> lat/lon using local radii of curvature."""
    m, n = radii(lat0)
    lat = lat0 + np.degrees(np.asarray(north_m) / m)
    lon = lon0 + np.degrees(np.asarray(east_m) / (n * np.cos(np.radians(lat0))))
    return lat, lon


def latlon_to_local_en(lat, lon, lat0, lon0):
    m, n = radii(lat0)
    north = np.radians(np.asarray(lat) - lat0) * m
    east = np.radians(np.asarray(lon) - lon0) * n * np.cos(np.radians(lat0))
    return east, north


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class Grid:
    """Global implicit grid: rows of constant latitude spaced `spacing_m`,
    per-row longitude spacing chosen so east-west spacing is also ~`spacing_m`.
    Node ids are stable int64s (row << 32 | col), so runs can be merged."""

    LAT0 = -90.0
    LON0 = -180.0

    def __init__(self, spacing_m):
        self.g = float(spacing_m)
        self.dlat = self.g / M_PER_DEG

    def _dlon(self, lat):
        return self.g / (M_PER_DEG * np.maximum(np.cos(np.radians(lat)), 1e-6))

    def nodes_in_bbox(self, lat_min, lat_max, lon_min, lon_max):
        k0 = int(np.floor((lat_min - self.LAT0) / self.dlat))
        k1 = int(np.ceil((lat_max - self.LAT0) / self.dlat))
        ids, lats, lons = [], [], []
        for k in range(k0, k1 + 1):
            lat = self.LAT0 + k * self.dlat
            if lat < lat_min - self.dlat or lat > lat_max + self.dlat:
                continue
            dl = self._dlon(lat)
            m0 = int(np.floor((lon_min - self.LON0) / dl))
            m1 = int(np.ceil((lon_max - self.LON0) / dl))
            m = np.arange(m0, m1 + 1, dtype=np.int64)
            ids.append((np.int64(k) << 32) | m)
            lats.append(np.full(m.size, lat))
            lons.append(self.LON0 + m * dl)
        if not ids:
            return np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
        return np.concatenate(ids), np.concatenate(lats), np.concatenate(lons)

    def node_latlon(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        k = ids >> 32
        m = ids & 0xFFFFFFFF
        lat = self.LAT0 + k * self.dlat
        return lat, self.LON0 + m * self._dlon(lat)
