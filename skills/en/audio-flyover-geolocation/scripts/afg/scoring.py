"""Trajectory-exact Doppler modelling and locked multi-event scoring.

For each candidate observer point the *exact* curve it would have heard is
modelled from the real ADS-B trajectory (turns, speed and altitude changes
included) using emission-time radial velocity, then laid on the measured
spectrogram:

  * event 1 is anchored at its measured delayed-CPA recording time; a small
    time shift (delta) and a frequency scale (f0 measurement error / RPM) are
    swept to absorb measurement error;
  * event 2 uses the SAME delta - its placement relative to event 1 is fixed
    purely by flight geometry ('locked' scoring), only its own f0 scale is
    swept. A wrong location therefore cannot fix a wrong inter-event offset.

Brightness score = mean of clipped whitened dB under the modelled curve
(and selected harmonics) over a fixed window; 'trace' score = Gaussian
closeness to hand/auto-traced points (better for very faint curves).

Bright texture under a wrong curve (speech, music, wind) can lift a brightness
score, so `trace_check` then tests whether each hypothesis' model curve
actually reproduces the traced points.
"""
import numpy as np

from .audio import sample_spec
from .common import log
from .geo import geodetic_to_ecef, offset_latlon
from .physics import doppler_from_track, doppler_linear


class CurveModel:
    def __init__(self, R, c=343.0, wind_ecef=None):
        self.R, self.c, self.wind = R, c, wind_ecef

    def curve(self, fid, obs, t_emit_center, half_s=100.0):
        R = self.R
        a, b = R.flight_slice(fid, t_emit_center - half_s - 40, t_emit_center + half_s + 40)
        if b - a < 5:
            return None
        tau = R.t[a:b]
        t_rx, ratio, rng = doppler_from_track(tau, R.ecef[a:b], R.vel[a:b], obs, 1.0, self.c, self.wind)
        k = int(np.argmin(rng))
        if k == 0 or k == len(rng) - 1:
            t_cpa_rx = t_rx[k]
            edge = True
        else:
            y = rng[k - 1:k + 2] ** 2
            den = y[0] - 2 * y[1] + y[2]
            dl = 0.5 * (y[0] - y[2]) / den if den > 0 else 0.0
            t_cpa_rx = np.interp(k + dl, np.arange(len(t_rx)), t_rx)
            edge = False
        return {"t_rx": t_rx, "ratio": ratio, "rng": rng, "t_cpa_rx": t_cpa_rx, "d_cpa": float(rng.min()), "edge": edge}


def _event_scores(spec, cur, off, rec_center, f0, half_w, step, deltas, scales, harmonics, clip=(0.0, 30.0)):
    rec = rec_center + np.arange(-half_w, half_w + step / 2, step)
    utc = rec[None, :] + off + deltas[:, None]
    r = np.interp(utc, cur["t_rx"], cur["ratio"], left=np.nan, right=np.nan)
    valid = np.isfinite(r)
    r = np.where(valid, r, 0.0)
    h = np.asarray(harmonics, float)
    F = f0 * scales[:, None, None, None] * h[None, None, None, :] * r[None, :, :, None]
    T = np.broadcast_to(rec[None, None, :, None], F.shape)
    v = sample_spec(spec, T, F, clip_db=clip)
    v = np.where(valid[None, :, :, None], v, 0.0)
    return v.mean(axis=(2, 3))  # (n_scales, n_deltas)


def _trace_scores(trace, cur, off, f0, deltas, scales, sigma_hz):
    tr_t, tr_f = trace
    utc = tr_t[None, :] + off + deltas[:, None]
    r = np.interp(utc, cur["t_rx"], cur["ratio"], left=np.nan, right=np.nan)
    valid = np.isfinite(r)
    fm = f0 * scales[:, None, None] * np.where(valid, r, 0)[None]
    g = np.exp(-0.5 * ((fm - tr_f[None, None, :]) / sigma_hz) ** 2)
    g = np.where(valid[None], g, 0.0)
    return g.mean(axis=2)


def score_point(model, spec, events, obs, fids, t_cpas, cfg, traces=None, why=None):
    """Score one observer point for one assignment of flights to events.
    Returns dict (total, per-event scores, delta, scales, rec_start_utc) or None.
    why: optional dict; the reason for a None is counted in it ('no_track': too
    little ADS-B around the CPA, 'clock': every time shift lies outside the
    known recording clock)."""
    curves = []
    for fid, tc in zip(fids, t_cpas):
        cu = model.curve(int(fid), obs, float(tc), cfg["half_w"] + 20)
        if cu is None:
            if why is not None:
                why["no_track"] = why.get("no_track", 0) + 1
            return None
        curves.append(cu)
    off0 = curves[0]["t_cpa_rx"] - events[0]["t_rec"]  # rec -> utc offset anchoring event 1
    deltas, scales = cfg["deltas"], cfg["scales"]
    per = []
    for k, (ev, cu) in enumerate(zip(events, curves)):
        rec_center = ev["t_rec"] if k == 0 else cu["t_cpa_rx"] - off0
        if traces is not None and traces[k] is not None:
            S = _trace_scores(traces[k], cu, off0, ev["f0"], deltas, scales, cfg["trace_sigma_hz"])
        else:
            S = _event_scores(spec, cu, off0, rec_center, ev["f0"], cfg["half_w"], cfg["step"], deltas, scales,
                              cfg["harmonics"])
        per.append(S)
    best_s = [S.max(axis=0) for S in per]       # per delta, best scale for each event
    if cfg.get("clock") is not None:            # known recording clock: forbid shifts outside it
        rs, tol = cfg["clock"]
        bad = np.abs(off0 + deltas - rs) > tol
        if bad.all():
            if why is not None:
                why["clock"] = why.get("clock", 0) + 1
            return None
        best_s = [np.where(bad, -np.inf, b) for b in best_s]
    if cfg.get("lock_mode", "joint") == "e1":
        di = int(np.argmax(best_s[0]))
    else:
        di = int(np.argmax(np.sum(best_s, axis=0)))
    ev_scores = [float(b[di]) for b in best_s]
    sc = [float(scales[int(np.argmax(S[:, di]))]) for S in per]
    return {"total": float(np.sum(ev_scores)), "events": ev_scores, "delta": float(deltas[di]), "scales": sc,
            "rec_start_utc": float(off0 + deltas[di]), "d_cpa": [cu["d_cpa"] for cu in curves],
            "t_cpa_rx": [cu["t_cpa_rx"] for cu in curves]}


def make_cfg(fine=True, half_w=60.0, delta_max=4.0, f_rel=0.006, harmonics=(1, 2), lock_mode="joint",
             trace_sigma_hz=0.6):
    if fine:
        return {"half_w": half_w, "step": 0.5, "deltas": np.arange(-delta_max, delta_max + 1e-9, 0.25),
                "scales": 1 + np.linspace(-f_rel, f_rel, 25), "harmonics": list(harmonics),
                "lock_mode": lock_mode, "trace_sigma_hz": trace_sigma_hz}
    return {"half_w": half_w, "step": 1.0, "deltas": np.arange(-delta_max, delta_max + 1e-9, 1.0),
            "scales": 1 + np.linspace(-f_rel, f_rel, 9), "harmonics": list(harmonics)[:1],
            "lock_mode": lock_mode, "trace_sigma_hz": trace_sigma_hz}


def subgrid(lat, lon, spacing, n):
    o = np.arange(-n, n + 1) * spacing
    E, N = np.meshgrid(o, o)
    la, lo = offset_latlon(lat, lon, E.ravel(), N.ravel())
    return la, lo


def effective_delta_max(events, delta_max=4.0, cap=30.0):
    """Time-shift range for scoring: at least event 1's CPA-time tolerance t_tol
    (the funnel admits candidates that far off), in whole seconds, at most cap."""
    t_tol = float(events[0].get("t_tol", 0.0)) if events else 0.0
    return float(min(max(delta_max, np.ceil(t_tol - 1e-9)), max(cap, delta_max)))


def score_candidates(cand, R, spec, events, dem, c=343.0, mic_height=2.0, top_pairs=40, nodes_per_pair=4,
                     sub_m=100.0, sub_n=1, traces=None, harmonics=(1, 2), lock_mode="joint", half_w=60.0,
                     delta_max=4.0, f_rel=0.006, clock=None, min_step_m=25.0, max_iter=8, info=None):
    """clock=(rec_start_utc, tol_s) restricts the time shift to the known recording clock.
    The time shift spans +-max(delta_max, event 1's t_tol), the same CPA-time
    tolerance the funnel used.  Each kept node is refined by a shrinking 3x3
    hill-climb (sub_m -> min_step_m).  info: optional dict filled with counts
    (unique candidates, scored, and why the others were not)."""
    model = CurveModel(R, c)
    nev = len(events)
    dm = effective_delta_max(events, delta_max)
    if dm > delta_max:
        log(f"scoring: time shift widened to +-{dm:.0f} s to cover event 1's CPA-time tolerance "
            f"({events[0]['t_tol']:.1f} s)")
    delta_max = dm
    coarse = make_cfg(False, half_w, delta_max, f_rel, harmonics, lock_mode)
    fine = make_cfg(True, half_w, delta_max, f_rel, harmonics, lock_mode)
    coarse["clock"] = fine["clock"] = clock
    # unique (node, fid1, t1, fid2, t2)
    key = np.c_[cand["node_id"], cand["fid1"], np.round(cand["t1_cpa"]), cand["fid2"], np.round(np.nan_to_num(cand["t2_cpa"]))]
    _, ui = np.unique(key, axis=0, return_index=True)
    cand = cand[np.sort(ui)]
    log(f"scoring: {len(cand)} unique candidates (coarse pass)")
    cs = np.full(len(cand), -np.inf)
    why = {}
    for i, cd in enumerate(cand):
        obs = geodetic_to_ecef(cd["lat"], cd["lon"], cd["elev"] + mic_height)
        fids = [cd["fid1"]] + ([cd["fid2"]] if nev > 1 else [])
        tcs = [cd["t1_cpa"]] + ([cd["t2_cpa"]] if nev > 1 else [])
        r = score_point(model, spec, events, obs, fids, tcs, coarse, traces, why)
        if r is not None:
            cs[i] = r["total"]
        if i and i % 2000 == 0:
            log(f"  coarse {i}/{len(cand)}")
    if info is not None:
        info.update(unique_candidates=int(len(cand)), scored=int(np.isfinite(cs).sum()),
                    rejected_clock=int(why.get("clock", 0)), no_track=int(why.get("no_track", 0)),
                    delta_max_s=float(delta_max),
                    clock=None if clock is None else {"rec_start": float(clock[0]), "tol_s": float(clock[1])})
    if why.get("clock"):
        log(f"scoring: {why['clock']} of {len(cand)} candidates dropped by the recording clock: their implied start "
            f"is more than {clock[1]:.0f} s (+-{delta_max:.0f} s shift) from --rec-start - give score the same "
            "--rec-start-tol as search, or widen it")
    if why.get("no_track"):
        log(f"scoring: {why['no_track']} candidates without enough ADS-B samples around the CPA")
    # keep the best node per flight pair plus spatially distinct runners-up
    # (a straight pass is mirror-symmetric about its track: both sides must survive)
    pair = cand["fid1"].astype(np.int64) * 10_000_000 + (cand["fid2"].astype(np.int64) + 1)
    keep = []
    for p in np.unique(pair):
        idx = np.flatnonzero((pair == p) & np.isfinite(cs))
        idx = idx[np.argsort(-cs[idx])]
        keep += _diverse(idx, cand["lat"], cand["lon"], nodes_per_pair, 400.0)
    keep = np.array(keep, int)
    if keep.size == 0:
        return [], cand, cs
    pbest = {}
    for i in keep:
        pbest[pair[i]] = max(pbest.get(pair[i], -np.inf), cs[i])
    top = sorted(pbest, key=lambda p: -pbest[p])[:top_pairs]
    sel = [i for i in keep if pair[i] in set(top)]
    log(f"scoring: fine pass on {len(sel)} nodes from {len(top)} flight pairs")
    results = []
    for i in sel:
        cd = cand[i]
        fids = [cd["fid1"]] + ([cd["fid2"]] if nev > 1 else [])
        tcs = [cd["t1_cpa"]] + ([cd["t2_cpa"]] if nev > 1 else [])
        best, step, seen = None, sub_m, {}
        clat, clon = float(cd["lat"]), float(cd["lon"])
        for _ in range(max_iter):
            la, lo = subgrid(clat, clon, step, sub_n)
            el = dem.sample(la, lo)
            moved = False
            for a, b, e in zip(la, lo, el):
                kk = (round(a, 6), round(b, 6))
                if kk not in seen:
                    seen[kk] = score_point(model, spec, events, geodetic_to_ecef(a, b, e + mic_height), fids, tcs,
                                           fine, traces)
                r = seen[kk]
                if r is not None and (best is None or r["total"] > best["total"] + 1e-9):
                    best, moved = dict(r, lat=float(a), lon=float(b), elev=float(e)), (a, b) != (clat, clon)
            if best is None:
                break
            clat, clon = best["lat"], best["lon"]
            if not moved:
                if step <= min_step_m:
                    break
                step /= 2.0
        if best:
            best.update(fid1=int(cd["fid1"]), fid2=int(cd["fid2"]), t1_cpa=float(cd["t1_cpa"]),
                        t2_cpa=float(cd["t2_cpa"]) if nev > 1 else None, coarse=float(cs[i]))
            results.append(best)
    # hypotheses = spatially distinct maxima per flight pair (>= 500 m apart)
    hyps = []
    for k in {(r["fid1"], r["fid2"]) for r in results}:
        rr = sorted([r for r in results if (r["fid1"], r["fid2"]) == k], key=lambda r: -r["total"])
        la = np.array([r["lat"] for r in rr])
        lo = np.array([r["lon"] for r in rr])
        for j in _diverse(np.arange(len(rr)), la, lo, 3, 500.0):
            hyps.append(rr[j])
    ranked = sorted(hyps, key=lambda r: -r["total"])
    return ranked, cand, cs


def trace_check(R, hyps, events, traces, fits, c=343.0, mic_height=2.0, half_w=60.0, f_rel=0.006,
                ratio_max=4.0, slack_hz=1.0):
    """Does each hypothesis' model curve reproduce the traced points?  Adds
    hyp["trace_check"] and hyp["scale_at_bound"] in place.

    The model curve uses the hypothesis' own position, clock and f0 scale (as
    scored).  It is compared with the event's traced points within +-half_w s of
    the fitted CPA (the stretch the brightness score sees; the tracer's far
    tails lean towards a straight pass when the aircraft turns), using the
    points the event's linear fit kept as inliers.  Statistic: robust RMS, the
    square root of the median squared residual.  The reference is the same
    statistic for the linear fit; an event passes when
        rms <= max(ratio_max * ref, ref + slack_hz).
    On the synthetic self-tests the true two-event answer stays within 1.9 x
    ref, and wrong aircraft pairs exceed 15 x ref on at least one event.  The
    true pair at a site 2-6 km away reaches 2.7-15 x, so some of those pass;
    the brightness score still ranks them below the true site.  One event alone
    is weaker: other aircraft with a similar speed and distance reproduce the
    curve too."""
    model = CurveModel(R, c)
    nev = len(events)
    for r in hyps:
        fids = [r["fid1"]] + ([r["fid2"]] if nev > 1 else [])
        tcs = [r["t1_cpa"]] + ([r["t2_cpa"]] if nev > 1 else [])
        obs = geodetic_to_ecef(r["lat"], r["lon"], r["elev"] + mic_height)
        per = []
        for k, (fid, tc, ev) in enumerate(zip(fids, tcs, events)):
            tr = traces[k] if traces is not None and k < len(traces) else None
            ft = fits[k] if fits is not None and k < len(fits) else None
            if tr is None or ft is None or len(tr[0]) < 10:
                per.append(None)
                continue
            t, f = np.asarray(tr[0], float), np.asarray(tr[1], float)
            res_l = doppler_linear(t, ft["f0"], ft["v"], ft["d"], ft["t_cpa_rx"], ft.get("c", c)) - f
            mad = 1.4826 * np.median(np.abs(res_l - np.median(res_l))) + 1e-9
            use = (np.abs(res_l) < 4 * mad) & (np.abs(t - ft["t_cpa_rx"]) <= half_w)
            if use.sum() < 10:
                per.append(None)
                continue
            ref = max(float(np.sqrt(np.median(res_l[use] ** 2))), 0.05)
            cu = model.curve(int(fid), obs, float(tc), half_w + 60.0)
            fm = np.full(t.size, np.nan)
            if cu is not None:
                fm = np.interp(t, cu["t_rx"] - r["rec_start_utc"], ev["f0"] * r["scales"][k] * cu["ratio"],
                               left=np.nan, right=np.nan)
            ok = use & np.isfinite(fm)
            cover = float(ok.sum() / use.sum())
            if ok.sum() < 10 or cover < 0.5:     # the model does not reach the traced CPA: wrong time or no track
                per.append({"rms_hz": None, "ref_hz": ref, "ratio": None, "n": int(ok.sum()), "coverage": cover,
                            "ok": False})
                continue
            rms = float(np.sqrt(np.median((fm[ok] - f[ok]) ** 2)))
            per.append({"rms_hz": rms, "ref_hz": ref, "ratio": rms / ref, "n": int(ok.sum()), "coverage": cover,
                        "ok": bool(rms <= max(ratio_max * ref, ref + slack_hz))})
        known = [p["ok"] for p in per if p is not None]
        r["trace_check"] = {"events": per, "consistent": (all(known) if known else None),
                            "rule": f"robust rms <= max({ratio_max:g} x fit rms, fit rms + {slack_hz:g} Hz) "
                                    f"within +-{half_w:g} s of CPA"}
        r["scale_at_bound"] = [bool(abs(sc - 1.0) >= f_rel * (1 - 1e-6)) for sc in r["scales"]]
    return hyps


def rank_consistent(hyps, nev):
    """Two events: hypotheses whose curves fail `trace_check` go below all the
    others (score order is kept inside each group).  One event: unchanged - a
    single curve is reproduced by too many wrong aircraft for the check to rank."""
    if nev < 2:
        return list(hyps)
    return sorted(hyps, key=lambda r: ((r.get("trace_check") or {}).get("consistent") is False, -r["total"]))


def pass_geometry(R, best, nev, min_cross_deg=25.0):
    """Ground-track headings at CPA and, for two events, their crossing angle.

    Each pass pins the observer across its own track (curve shape) and, with
    a clock, along it.  Two passes share one unknown clock, so the along-track
    direction is only broken if the tracks cross at a decent angle: near-
    parallel passes leave a strip along the tracks."""
    out = {"headings_deg": []}
    for fid, tc in [(best["fid1"], best["t1_cpa"])] + ([(best["fid2"], best["t2_cpa"])] if nev > 1 else []):
        a, b = R.flight_slice(fid, tc - 1.5 * R.dt, tc + 1.5 * R.dt)
        if b <= a:
            out["headings_deg"].append(None)
            continue
        i = a + int(np.argmin(np.abs(R.t[a:b] - tc)))
        out["headings_deg"].append(float(np.degrees(np.arctan2(R.ve[i], R.vn[i])) % 360))
    h = out["headings_deg"]
    if nev > 1 and None not in h:
        x = abs(h[0] - h[1]) % 180.0
        out["crossing_angle_deg"] = float(min(x, 180.0 - x))
        if out["crossing_angle_deg"] < min_cross_deg:
            out["warning"] = (f"tracks cross at only {out['crossing_angle_deg']:.0f} deg: position along the tracks is "
                              "weakly constrained - report the heat-map strip, not a point")
    return out


def score_region(cand, cs, frac=0.9, best=None):
    """Coarse-score plateau: every candidate node scoring >= frac * best.

    One straight, steady pass constrains distance-to-track and crossing time,
    but NOT where along the track you stood unless the recording clock is
    known: the plateau is then a strip ~2 * v * clock_tolerance long (plus the
    mirror strip on the other side).  Report this extent instead of a point.
    """
    from .geo import latlon_to_local_en
    sel = np.isfinite(cs)
    if best is not None:  # only the best hypothesis' flight pair: other aircraft are separate hypotheses
        sel &= (cand["fid1"] == best["fid1"]) & (cand["fid2"] == best["fid2"])
    ok = sel & (cs >= frac * np.max(cs[sel])) if sel.any() else np.zeros(len(cs), bool)
    la, lo = cand["lat"][ok], cand["lon"][ok]
    out = {"frac": frac, "n_nodes": int(ok.sum()), "lat": la, "lon": lo, "length_m": 0.0, "width_m": 0.0, "axis_deg": None}
    if la.size >= 2:
        e, n = latlon_to_local_en(la, lo, float(la.mean()), float(lo.mean()))
        X = np.c_[e - e.mean(), n - n.mean()]
        w, V = np.linalg.eigh(X.T @ X)
        p = X @ V  # columns: minor, major axis
        out["length_m"] = float(np.ptp(p[:, 1]))
        out["width_m"] = float(np.ptp(p[:, 0]))
        out["axis_deg"] = float(np.degrees(np.arctan2(V[0, 1], V[1, 1])) % 180)  # bearing of major axis
    return out


def _diverse(idx_sorted, lat, lon, k, min_sep_m):
    from .geo import haversine_m
    picks = []
    for i in idx_sorted:
        if all(haversine_m(lat[i], lon[i], lat[p], lon[p]) > min_sep_m for p in picks):
            picks.append(int(i))
        if len(picks) >= k:
            break
    return picks


def ambiguity(ranked, frac=0.9, min_sep_m=500.0):
    """Competing hypotheses scoring >= frac*best that are >= min_sep_m away
    (mirror side of a straight pass, or a different flight pair)."""
    from .geo import haversine_m
    if not ranked:
        return []
    b = ranked[0]
    out = []
    for r in ranked[1:]:
        dist = float(haversine_m(b["lat"], b["lon"], r["lat"], r["lon"]))
        if r["total"] >= frac * b["total"] and dist >= min_sep_m:
            out.append({"lat": r["lat"], "lon": r["lon"], "score_ratio": r["total"] / b["total"], "dist_m": dist,
                        "same_flights": (r["fid1"], r["fid2"]) == (b["fid1"], b["fid2"])})
    return out


def heatmap(R, spec, events, dem, best, c=343.0, mic_height=2.0, radius_m=1500.0, step_m=30.0, traces=None,
            harmonics=(1, 2), lock_mode="joint", half_w=60.0, delta_max=4.0, f_rel=0.006, clock=None):
    """Dense local score map around the best candidate for its flight pair, with
    the same time-shift range and recording clock as `score_candidates`.
    Returns lat, lon grids and total / per-event score grids."""
    model = CurveModel(R, c)
    cfg = make_cfg(True, half_w, effective_delta_max(events, delta_max), f_rel, harmonics, lock_mode)
    cfg["clock"] = clock
    cfg["scales"] = 1 + np.linspace(-f_rel, f_rel, 13)
    n = int(round(radius_m / step_m))
    o = np.arange(-n, n + 1) * step_m
    E, N = np.meshgrid(o, o)
    la, lo = offset_latlon(best["lat"], best["lon"], E.ravel(), N.ravel())
    el = dem.sample(la, lo)
    nev = len(events)
    fids = [best["fid1"]] + ([best["fid2"]] if nev > 1 else [])
    tcs = [best["t1_cpa"]] + ([best["t2_cpa"]] if nev > 1 else [])
    tot = np.full(la.size, np.nan)
    per = np.full((nev, la.size), np.nan)
    for i, (a, b, e) in enumerate(zip(la, lo, el)):
        r = score_point(model, spec, events, geodetic_to_ecef(a, b, e + mic_height), fids, tcs, cfg, traces)
        if r is not None:
            tot[i] = r["total"]
            per[:, i] = r["events"]
    shp = E.shape
    return la.reshape(shp), lo.reshape(shp), tot.reshape(shp), per.reshape((nev,) + shp), E, N


def model_overlay(R, best, events, dem, c=343.0, mic_height=2.0, half_w=90.0):
    """Modelled curves (recording time, Hz) for plotting over the spectrogram."""
    model = CurveModel(R, c)
    obs = geodetic_to_ecef(best["lat"], best["lon"], best["elev"] + mic_height)
    out = []
    fids = [best["fid1"]] + ([best["fid2"]] if len(events) > 1 else [])
    tcs = [best["t1_cpa"]] + ([best["t2_cpa"]] if len(events) > 1 else [])
    for k, (fid, tc, ev) in enumerate(zip(fids, tcs, events)):
        cu = model.curve(int(fid), obs, float(tc), half_w)
        if cu is None:
            continue
        rec = cu["t_rx"] - best["rec_start_utc"]
        out.append({"t": rec, "f": ev["f0"] * best["scales"][k] * cu["ratio"], "label": f"event {k + 1} model"})
    return out
