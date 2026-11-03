"""Uniform-time resampling of ADS-B flights into ECEF position + velocity.

Velocity comes from the broadcast ground speed / track / vertical rate when
available (ADS-B velocity messages are GNSS-derived, ~0.5 m/s), because
differentiating noisy positions would inject several m/s of error - and the
Doppler shift is directly proportional to radial velocity (1 m/s ~ 0.3 %).
"""
import json

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from .geo import enu_vectors_to_ecef, geodetic_to_ecef


class Flights:
    """Raw per-sample table + per-flight metadata (from tracks.npz)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.fid, self.t = z["fid"], z["t"]
        self.lat, self.lon, self.h = z["lat"], z["lon"], z["h"].astype(np.float64)
        self.gs, self.trk, self.vr = z["gs"].astype(np.float64), z["trk"].astype(np.float64), z["vr"].astype(np.float64)
        self.hex, self.type, self.reg = z["f_hex"], z["f_type"], z["f_reg"]
        self.cls, self.desc = z["f_cls"], z["f_desc"]
        self.meta = json.loads(str(z["meta"]))
        self.nflights = len(self.hex)

    def label(self, fid):
        return f"{self.hex[fid]}/{self.type[fid] or '?'}/{self.reg[fid] or '-'}#{fid}"


class Resampled:
    """Concatenated uniform samples: arrays aligned by sample index.
    Samples of one flight are contiguous and time-ordered; prev/next validity
    is encoded in has_prev / has_next."""

    def __init__(self, fl, dt=1.0, t0=None, t1=None, fids=None):
        self.dt = float(dt)
        out = {k: [] for k in ("fid", "t", "lat", "lon", "h", "gs", "ve", "vn", "vu")}
        order = np.argsort(fl.fid, kind="stable")
        fid_sorted = fl.fid[order]
        bounds = np.flatnonzero(np.diff(fid_sorted)) + 1
        starts = np.r_[0, bounds]
        ends = np.r_[bounds, fid_sorted.size]
        allowed = None if fids is None else set(int(x) for x in fids)
        for s, e in zip(starts, ends):
            ii = order[s:e]
            f = int(fl.fid[ii[0]])
            if allowed is not None and f not in allowed:
                continue
            ii = ii[np.argsort(fl.t[ii])]
            t = fl.t[ii]
            t, uniq = np.unique(t, return_index=True)
            ii = ii[uniq]
            if t.size < 3:
                continue
            a = np.ceil(max(t[0], t0 if t0 is not None else -np.inf) / dt) * dt
            b = np.floor(min(t[-1], t1 if t1 is not None else np.inf) / dt) * dt
            if b - a < 2 * dt:
                continue
            tg = np.arange(a, b + dt / 2, dt)
            lat = PchipInterpolator(t, fl.lat[ii])(tg)
            lon = PchipInterpolator(t, np.degrees(np.unwrap(np.radians(fl.lon[ii]))))(tg)
            hh = fl.h[ii]
            okh = np.isfinite(hh)
            if okh.sum() < 2:
                continue
            h = np.interp(tg, t[okh], hh[okh])
            gs, trk, vr = fl.gs[ii], fl.trk[ii], fl.vr[ii]
            okv = np.isfinite(gs) & np.isfinite(trk)
            if okv.sum() >= 2:
                ve_s = gs[okv] * np.sin(np.radians(trk[okv]))
                vn_s = gs[okv] * np.cos(np.radians(trk[okv]))
                ve = np.interp(tg, t[okv], ve_s)
                vn = np.interp(tg, t[okv], vn_s)
            else:  # fall back to smoothed position derivative
                m_lat = 111132.954 - 559.822 * np.cos(2 * np.radians(lat))
                m_lon = 111412.84 * np.cos(np.radians(lat))
                win = min(len(tg) - (1 - len(tg) % 2), 21)
                if win >= 5:
                    ve = savgol_filter(lon, win, 2, deriv=1, delta=dt) * m_lon
                    vn = savgol_filter(lat, win, 2, deriv=1, delta=dt) * m_lat
                else:
                    ve = np.gradient(lon, dt) * m_lon
                    vn = np.gradient(lat, dt) * m_lat
            okr = np.isfinite(vr)
            vu = np.interp(tg, t[okr], vr[okr]) if okr.sum() >= 2 else np.gradient(h, dt)
            out["fid"].append(np.full(tg.size, f, np.int32))
            out["t"].append(tg)
            out["lat"].append(lat)
            out["lon"].append(((lon + 180) % 360) - 180)
            out["h"].append(h)
            out["ve"].append(ve)
            out["vn"].append(vn)
            out["vu"].append(vu)
            out["gs"].append(np.hypot(ve, vn))
        if not out["t"]:
            raise RuntimeError("no flight samples after resampling (check time window)")
        for k in out:
            setattr(self, k, np.concatenate(out[k]))
        self.ecef = geodetic_to_ecef(self.lat, self.lon, self.h)
        self.vel = enu_vectors_to_ecef(self.lat, self.lon, self.ve, self.vn, self.vu)
        same = (self.fid[1:] == self.fid[:-1]) & (np.abs(np.diff(self.t) - dt) < 1e-6)
        self.has_prev = np.r_[False, same]
        self.has_next = np.r_[same, False]
        self.n = self.t.size
        # flights are contiguous (input grouped by fid): first/last index per flight
        fids, first = np.unique(self.fid, return_index=True)
        cnt = np.bincount(self.fid)
        self._first = dict(zip(fids.tolist(), first.tolist()))
        self._last = {f: a + int(cnt[f]) - 1 for f, a in zip(fids.tolist(), first.tolist())}

    def flight_slice(self, fid, ta=None, tb=None):
        """Index range [i0, i1) of flight fid restricted to [ta, tb]."""
        a, b = self._first[int(fid)], self._last[int(fid)] + 1
        if ta is not None:
            a = max(a, a + int(np.ceil((ta - self.t[a]) / self.dt)))
        if tb is not None:
            b = min(b, self._first[int(fid)] + int(np.floor((tb - self.t[self._first[int(fid)]]) / self.dt)) + 1)
        return a, max(a, b)
