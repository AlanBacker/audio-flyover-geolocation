"""ADS-B ingestion and aircraft classification.

Supported inputs
----------------
readsb trace files  (trace_full_<hex>.json, gzip or plain) as published by
  - ADS-B Exchange free samples: samples.adsbexchange.com/traces/YYYY/MM/DD/xx/
  - adsb.lol globe_history daily releases (github.com/adsblol/globe_history_YYYY)
  - any tar1090/readsb install (/var/globe_history/YYYY/MM/DD/traces/)
OpenSky state_vectors_data4 CSV exports (Trino / pyopensky)
Generic CSV: hex,t,lat,lon,alt_m,gs_ms,track_deg[,vr_ms,type]

readsb trace entry layout (wiedehopf/readsb README-json.md):
  [dt, lat, lon, alt_baro_ft|"ground"|null, gs_kt, track_deg, flags, vrate_fpm,
   details|null, source, alt_geom_ft, geom_rate_fpm, ias_kt, roll_deg]
  flags: 1 stale, 2 new leg, 4 vrate geometric, 8 altitude geometric

Altitude datum (matters for slant distance!)
  alt_baro = pressure altitude (29.92 inHg / 1013.25 hPa), NOT true altitude.
             true ~= baro + (QNH - 1013.25) * 8.3 m   (+ temperature error)
  alt_geom = GNSS height, WGS84 ellipsoid (HAE). MSL = HAE - N (geoid undulation;
             N ~ -20..-35 m over CONUS, so MSL = HAE + 20..35 m).
"""
import csv
import glob
import gzip
import json
import os
import urllib.request

import numpy as np

from .common import FPM, FT, HPA_TO_M, KT, log

TYPES_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types.js"

# Minimal fallback (ICAO Doc 8643 description codes) if the full table is unavailable.
_FALLBACK_TYPES = {
    "C150": "L1P", "C152": "L1P", "C172": "L1P", "C182": "L1P", "C206": "L1P", "C210": "L1P",
    "P28A": "L1P", "P28R": "L1P", "PA32": "L1P", "PA34": "L2P", "PA44": "L2P", "PA46": "L1P",
    "SR20": "L1P", "SR22": "L1P", "BE35": "L1P", "BE36": "L1P", "BE58": "L2P", "M20P": "L1P",
    "DA40": "L1P", "DA42": "L2P", "C208": "L1T", "PC12": "L1T", "TBM7": "L1T", "TBM9": "L1T",
    "BE20": "L2T", "BE9L": "L2T", "B350": "L2T", "DH8A": "L2T", "DH8B": "L2T", "DH8C": "L2T",
    "DH8D": "L2T", "AT43": "L2T", "AT45": "L2T", "AT72": "L2T", "AT75": "L2T", "AT76": "L2T",
    "SF34": "L2T", "C130": "L4T", "DHC6": "L2T", "E120": "L2T",
    "R22": "H1P", "R44": "H1P", "R66": "H1T", "EC35": "H2T", "EC45": "H2T", "AS50": "H1T",
    "B06": "H1T", "B407": "H1T", "S76": "H2T", "A139": "H2T", "H60": "H2T",
    "A319": "L2J", "A320": "L2J", "A321": "L2J", "B737": "L2J", "B738": "L2J", "B739": "L2J",
    "B38M": "L2J", "B39M": "L2J", "E170": "L2J", "E75L": "L2J", "CRJ2": "L2J", "CRJ7": "L2J",
    "CRJ9": "L2J", "B752": "L2J", "B763": "L2J", "B772": "L2J", "B77W": "L2J", "B788": "L2J",
    "B789": "L2J", "A332": "L2J", "A333": "L2J", "A359": "L2J", "C25A": "L2J", "C56X": "L2J",
    "CL30": "L2J", "GLF4": "L2J", "GLF5": "L2J", "GLEX": "L2J", "E55P": "L2J", "PC24": "L2J",
}


def _read_maybe_gz(path):
    with open(path, "rb") as fh:
        b = fh.read()
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    return b


def load_type_table(path=None, cache_dir=None, allow_download=True):
    """ICAO type designator -> Doc 8643 description code (e.g. C172 -> L1P).
    desc code: [L/S/A/H/G/T = land/sea/amphibian/helicopter/gyro/tilt-wing]
               [engine count][J jet / T turboprop / P piston / E electric]"""
    cand = []
    if path:
        cand.append(path)
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "afg")
    cache = os.path.join(cache_dir, "icao_aircraft_types.js")
    cand.append(cache)
    for p in cand:
        if p and os.path.exists(p):
            try:
                return _parse_types(json.loads(_read_maybe_gz(p)))
            except Exception as e:
                log("type table unreadable:", p, e)
    if allow_download:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            log("downloading ICAO type table", TYPES_URL)
            urllib.request.urlretrieve(TYPES_URL, cache)
            return _parse_types(json.loads(_read_maybe_gz(cache)))
        except Exception as e:
            log("type table download failed, using built-in fallback:", e)
    return dict(_FALLBACK_TYPES)


def _parse_types(obj):
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k.upper()] = v.get("desc", "")
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            out[k.upper()] = v[1]
    return out


def classify(type_code, types, category=None):
    """-> 'prop' | 'rotor' | 'jet' | 'glider' | 'balloon' | 'unknown'"""
    desc = types.get((type_code or "").upper(), "") if type_code else ""
    if desc and len(desc) >= 3:
        c0, eng = desc[0], desc[2]
        if c0 in "HG":
            return "rotor"
        if eng == "J":
            return "jet"
        if eng in "PTE":
            return "prop"
    cat = (category or "").upper()
    if cat == "A7":
        return "rotor"
    if cat in ("A3", "A4", "A5", "A6"):
        return "jet"
    if cat == "B1":
        return "glider"
    if cat == "B2":
        return "balloon"
    if (type_code or "").upper() in ("GLID",):
        return "glider"
    if (type_code or "").upper() in ("BALL",):
        return "balloon"
    return "unknown"


class _Acc:
    """Accumulates per-sample rows and per-flight metadata."""

    def __init__(self):
        self.cols = {k: [] for k in ("fid", "t", "lat", "lon", "alt_baro", "alt_geom", "gs", "trk", "vr")}
        self.flights = []

    def add_segment(self, meta, arr):
        if arr["t"].size < 3:
            return
        fid = len(self.flights)
        self.flights.append(meta)
        self.cols["fid"].append(np.full(arr["t"].size, fid, np.int32))
        for k in ("t", "lat", "lon", "alt_baro", "alt_geom", "gs", "trk", "vr"):
            self.cols[k].append(arr[k])

    def finish(self):
        if not self.flights:
            raise RuntimeError("no ADS-B samples inside the requested window/bbox")
        return {k: np.concatenate(v) for k, v in self.cols.items()}, self.flights


def _segments(t, breaks, max_gap):
    """Split index range at explicit breaks and time gaps."""
    cut = np.zeros(t.size, bool)
    cut[1:] = (np.diff(t) > max_gap) | breaks[1:]
    starts = np.flatnonzero(cut | (np.arange(t.size) == 0))
    ends = np.append(starts[1:], t.size)
    return list(zip(starts, ends))


def _f(v):
    return np.nan if v is None or isinstance(v, str) else float(v)


def ingest_readsb(paths, t0, t1, bbox=None, buffer_deg=0.15, max_gap=45.0, types=None, acc=None):
    acc = acc or _Acc()
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "trace_full_*"), recursive=True)
        else:
            files += glob.glob(p)
    log(f"readsb: {len(files)} trace files")
    if bbox:
        la0, la1, lo0, lo1 = bbox[0] - buffer_deg, bbox[1] + buffer_deg, bbox[2] - buffer_deg, bbox[3] + buffer_deg
    nseg = 0
    for i, fn in enumerate(files):
        if i and i % 20000 == 0:
            log(f"  {i}/{len(files)} files, {nseg} segments kept")
        try:
            obj = json.loads(_read_maybe_gz(fn))
        except Exception:
            continue
        tr = obj.get("trace") or []
        if not tr:
            continue
        base = float(obj.get("timestamp", 0))
        tt = base + np.array([e[0] for e in tr], float)
        if tt[-1] < t0 or tt[0] > t1:
            continue
        lat = np.array([_f(e[1]) for e in tr])
        lon = np.array([_f(e[2]) for e in tr])
        m = (tt >= t0) & (tt <= t1) & np.isfinite(lat)
        if bbox:
            m &= (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        alt_raw = [tr[k][3] for k in idx]
        ground = np.array([a == "ground" for a in alt_raw])
        alt_b = np.array([_f(a) for a in alt_raw]) * FT
        gs = np.array([_f(tr[k][4]) for k in idx]) * KT
        trk = np.array([_f(tr[k][5]) for k in idx])
        flags = np.array([int(tr[k][6] or 0) if len(tr[k]) > 6 else 0 for k in idx])
        vr = np.array([_f(tr[k][7]) if len(tr[k]) > 7 else np.nan for k in idx]) * FPM
        alt_g = np.array([_f(tr[k][10]) if len(tr[k]) > 10 else np.nan for k in idx]) * FT
        gr = np.array([_f(tr[k][11]) if len(tr[k]) > 11 else np.nan for k in idx]) * FPM
        vr = np.where(np.isfinite(gr), gr, vr)
        cat = None
        for k in idx:
            if len(tr[k]) > 8 and isinstance(tr[k][8], dict) and tr[k][8].get("category"):
                cat = tr[k][8]["category"]
                break
        typ = obj.get("t") or ""
        meta = {"hex": obj.get("icao", os.path.basename(fn)[11:].split(".")[0]), "type": typ,
                "reg": obj.get("r") or "", "category": cat or "",
                "desc": (types or {}).get(typ.upper(), "") if types else "",
                "cls": classify(typ, types or {}, cat), "src": "readsb"}
        keep = ~ground
        breaks = ((flags & 2) > 0) | ((flags & 1) > 0)
        sub = {"t": tt[idx], "lat": lat[idx], "lon": lon[idx], "alt_baro": alt_b, "alt_geom": alt_g,
               "gs": gs, "trk": trk, "vr": vr}
        # remove ground points, then split
        sub = {k: v[keep] for k, v in sub.items()}
        brk = breaks[keep]
        if sub["t"].size < 3:
            continue
        for s, e in _segments(sub["t"], brk, max_gap):
            seg = {k: v[s:e] for k, v in sub.items()}
            acc.add_segment(dict(meta), seg)
            nseg += 1
    return acc


def ingest_csv(path, t0, t1, bbox=None, fmt="generic", max_gap=45.0, types=None, acc=None):
    """fmt='opensky' expects state_vectors_data4 columns
    (time, icao24, lat, lon, velocity, heading, vertrate, baroaltitude, geoaltitude, onground).
    fmt='generic': hex,t,lat,lon,alt_m,gs_ms,track_deg[,vr_ms,type,alt_kind] (alt_m MSL)."""
    acc = acc or _Acc()
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if fmt == "opensky":
                if str(r.get("onground", "")).lower() in ("true", "1", "t"):
                    continue
                key = r["icao24"]
                rec = (float(r["time"]), _num(r["lat"]), _num(r["lon"]), _num(r.get("baroaltitude")),
                       _num(r.get("geoaltitude")), _num(r.get("velocity")), _num(r.get("heading")),
                       _num(r.get("vertrate")), "")
            else:
                key = r["hex"]
                rec = (float(r["t"]), _num(r["lat"]), _num(r["lon"]), np.nan, _num(r["alt_m"]),
                       _num(r.get("gs_ms")), _num(r.get("track_deg")), _num(r.get("vr_ms")), r.get("type", ""))
            if not (t0 <= rec[0] <= t1) or not np.isfinite(rec[1]):
                continue
            if bbox and not (bbox[0] - 0.15 <= rec[1] <= bbox[1] + 0.15 and bbox[2] - 0.15 <= rec[2] <= bbox[3] + 0.15):
                continue
            rows.setdefault(key, []).append(rec)
    for key, recs in rows.items():
        a = np.array([x[:8] for x in recs], float)
        a = a[np.argsort(a[:, 0])]
        typ = recs[0][8] or ""
        meta = {"hex": key, "type": typ, "reg": "", "category": "", "desc": (types or {}).get(typ.upper(), ""),
                "cls": classify(typ, types or {}), "src": fmt}
        sub = {"t": a[:, 0], "lat": a[:, 1], "lon": a[:, 2], "alt_baro": a[:, 3], "alt_geom": a[:, 4],
               "gs": a[:, 5], "trk": a[:, 6], "vr": a[:, 7]}
        if fmt == "generic":   # generic alt_m is already MSL: store as geom with N=0 handled by caller
            sub["alt_baro"] = np.full(a.shape[0], np.nan)
        for s, e in _segments(sub["t"], np.zeros(a.shape[0], bool), max_gap):
            acc.add_segment(dict(meta), {k: v[s:e] for k, v in sub.items()})
    return acc


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def finalize(acc, alt_source="auto", qnh_hpa=1013.25, geoid_n=0.0):
    """Choose the altitude used for geometry: h_msl (m).
    auto: geometric (HAE - N) where present, else baro corrected by QNH."""
    cols, flights = acc.finish()
    h_geom = cols["alt_geom"] - geoid_n
    h_baro = cols["alt_baro"] + (qnh_hpa - 1013.25) * HPA_TO_M
    if alt_source == "geom":
        h = h_geom
    elif alt_source == "baro":
        h = h_baro
    else:
        h = np.where(np.isfinite(h_geom), h_geom, h_baro)
    cols["h"] = h
    ok = np.isfinite(h) & np.isfinite(cols["lat"]) & np.isfinite(cols["lon"])
    cols = {k: v[ok] for k, v in cols.items()}
    return cols, flights


def save_tracks(path, cols, flights, meta):
    np.savez_compressed(
        path,
        fid=cols["fid"], t=cols["t"], lat=cols["lat"], lon=cols["lon"], h=cols["h"].astype(np.float32),
        alt_baro=cols["alt_baro"].astype(np.float32), alt_geom=cols["alt_geom"].astype(np.float32),
        gs=cols["gs"].astype(np.float32), trk=cols["trk"].astype(np.float32), vr=cols["vr"].astype(np.float32),
        f_hex=np.array([f["hex"] for f in flights]), f_type=np.array([f["type"] for f in flights]),
        f_reg=np.array([f["reg"] for f in flights]), f_cls=np.array([f["cls"] for f in flights]),
        f_desc=np.array([f["desc"] for f in flights]), f_cat=np.array([f["category"] for f in flights]),
        meta=json.dumps(meta),
    )
