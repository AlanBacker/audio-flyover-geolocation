"""Shared constants and small helpers."""
import datetime as _dt
import json
import sys

import numpy as np

FT = 0.3048            # m per foot
KT = 0.514444          # m/s per knot
FPM = 0.00508          # m/s per foot-per-minute
HPA_TO_M = 8.3         # metres of pressure-altitude per hPa near sea level (~27 ft/hPa)


def log(*args):
    print("[afg]", *args, file=sys.stderr, flush=True)


def parse_time(value):
    """Parse ISO-8601 (assumed UTC if naive) or unix seconds -> float unix seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    s = s.replace("Z", "+00:00")
    t = _dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.timestamp()


def iso(t):
    if t is None or not np.isfinite(t):
        return None
    return _dt.datetime.fromtimestamp(float(t), _dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, cls=_Enc, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_bbox(s):
    """'lat_min,lat_max,lon_min,lon_max' -> tuple of floats (validated)."""
    if s is None:
        return None
    v = [float(x) for x in str(s).split(",")]
    if len(v) != 4:
        raise ValueError("bbox must be lat_min,lat_max,lon_min,lon_max")
    la0, la1, lo0, lo1 = v
    if la0 > la1:
        la0, la1 = la1, la0
    if lo0 > lo1:
        lo0, lo1 = lo1, lo0
    return (la0, la1, lo0, lo1)
