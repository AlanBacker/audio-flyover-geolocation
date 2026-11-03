"""Candidate funnel (coarse grid, typically 200 m):

  A  event-1 corridor: grid nodes near speed-matching flights whose 3D CPA
     slant distance and ground speed match event 1 (and its absolute time,
     if known)
  B  offset filter: the same node must hear an event-2-matching flight whose
     delayed-CPA time is (t2_rec - t1_rec) +- tol after event 1
  C  silence filter: no other flight may pass close enough to be heard
     during the parts of the recording where no fly-by is visible

'Delayed CPA time' = t_cpa(emission) + d_cpa / c : the moment the observer
hears the sound emitted at closest approach - exactly where the observed
curve crosses f0 (zero radial velocity). That is what gets matched to the
recording clock.
"""
import numpy as np
from scipy.spatial import cKDTree

from .common import log
from .geo import Grid, geodetic_to_ecef, latlon_to_local_en

EV_DTYPE = [("node", np.int64), ("j", np.int64), ("fid", np.int32), ("t_cpa", np.float64),
            ("t_rx", np.float64), ("d", np.float32), ("gs", np.float32), ("edge", np.bool_)]
CAND_DTYPE = [("node_id", np.int64), ("lat", np.float64), ("lon", np.float64), ("elev", np.float32),
              ("fid1", np.int32), ("t1_cpa", np.float64), ("t1_rx", np.float64), ("d1", np.float32), ("v1", np.float32),
              ("fid2", np.int32), ("t2_cpa", np.float64), ("t2_rx", np.float64), ("d2", np.float32), ("v2", np.float32)]


def cpa_events(node_ecef, R, sample_idx, radius, c=343.0, chunk=3000, tree=None):
    """All closest-point-of-approach events between nodes and the flights whose
    samples are listed in sample_idx (indices into R). A CPA = local minimum of
    slant range along a flight (refined with a parabola on R^2, exact for
    straight flight). Range minima at a track end are flagged edge=True.
    Returns structured array EV_DTYPE (node = row index into node_ecef)."""
    sample_idx = np.asarray(sample_idx, np.int64)
    if sample_idx.size == 0 or len(node_ecef) == 0:
        return np.zeros(0, EV_DTYPE)
    if tree is None:
        tree = cKDTree(R.ecef[sample_idx])
    outs = []
    for s in range(0, len(node_ecef), chunk):
        nodes = node_ecef[s:s + chunk]
        tn = cKDTree(nodes)
        M = tn.sparse_distance_matrix(tree, radius, output_type="ndarray")
        if M.size == 0:
            continue
        i = M["i"].astype(np.int64)
        j = sample_idx[M["j"]]
        d0 = M["v"]
        hp, hn = R.has_prev[j], R.has_next[j]
        jp = np.where(hp, j - 1, j)
        jn = np.where(hn, j + 1, j)
        dp = np.where(hp, np.linalg.norm(R.ecef[jp] - nodes[i], axis=1), np.inf)
        dn = np.where(hn, np.linalg.norm(R.ecef[jn] - nodes[i], axis=1), np.inf)
        is_min = (d0 <= dp) & (d0 < dn)
        if not is_min.any():
            continue
        i, j, d0, dp, dn, hp, hn, jp, jn = (a[is_min] for a in (i, j, d0, dp, dn, hp, hn, jp, jn))
        edge = ~(hp & hn)
        # parabola on squared range through (-1, 0, +1)
        y0, ym, yp = d0**2, np.where(hp, dp**2, d0**2), np.where(hn, dn**2, d0**2)
        den = ym - 2 * y0 + yp
        delta = np.where((den > 0) & ~edge, 0.5 * (ym - yp) / np.where(den > 0, den, 1), 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        r2 = y0 - 0.25 * (ym - yp) * delta
        d = np.sqrt(np.clip(np.minimum(r2, y0), 0, None))
        t_cpa = R.t[j] + delta * R.dt
        gs = R.gs[j] + delta * np.where(delta > 0, R.gs[jn] - R.gs[j], R.gs[j] - R.gs[jp])
        ev = np.zeros(i.size, EV_DTYPE)
        ev["node"], ev["j"], ev["fid"] = i + s, j, R.fid[j]
        ev["t_cpa"], ev["d"], ev["gs"], ev["edge"] = t_cpa, d, gs, edge
        ev["t_rx"] = t_cpa + d / c
        outs.append(ev)
    return np.concatenate(outs) if outs else np.zeros(0, EV_DTYPE)


def corridor_nodes(R, sample_idx, grid, half_width, chunk_s=180.0, step_s=4.0):
    """Grid node ids within horizontal distance `half_width` of the listed samples."""
    sample_idx = np.asarray(sample_idx, np.int64)
    if sample_idx.size == 0:
        return np.zeros(0, np.int64)
    ids_all = []
    fids = R.fid[sample_idx]
    order = np.lexsort((R.t[sample_idx], fids))
    si = sample_idx[order]
    f_s = R.fid[si]
    t_s = R.t[si]
    # chunk boundaries: new flight or > chunk_s elapsed
    newc = np.r_[True, (f_s[1:] != f_s[:-1]) | (np.floor(t_s[1:] / chunk_s) != np.floor(t_s[:-1] / chunk_s))]
    starts = np.flatnonzero(newc)
    ends = np.r_[starts[1:], si.size]
    stride = max(1, int(round(step_s / R.dt)))
    for a, b in zip(starts, ends):
        js = si[a:b:stride]
        if js.size == 0:
            continue
        lat, lon = R.lat[js], R.lon[js]
        pad = (half_width + grid.g) / 111320.0
        padlon = pad / max(np.cos(np.radians(lat.mean())), 0.2)
        ids, nlat, nlon = grid.nodes_in_bbox(lat.min() - pad, lat.max() + pad, lon.min() - padlon, lon.max() + padlon)
        if ids.size == 0:
            continue
        lat0, lon0 = lat.mean(), lon.mean()
        e, n = latlon_to_local_en(nlat, nlon, lat0, lon0)
        te, tn = latlon_to_local_en(lat, lon, lat0, lon0)
        dist, _ = cKDTree(np.c_[te, tn]).query(np.c_[e, n], k=1)
        slack = 0.5 * stride * R.dt * float(np.nanmax(R.gs[js])) + 0.75 * grid.g  # along-track sampling + cell size
        ids_all.append(ids[dist <= half_width + slack])
    if not ids_all:
        return np.zeros(0, np.int64)
    return np.unique(np.concatenate(ids_all))


def _in_window(x, lo, hi):
    return (x >= lo) & (x <= hi)


def _k_ok(ev, e):
    """v^2/d consistency: the candidate's own ADS-B ground speed must predict
    its CPA distance through the curve steepness k (see search_constraints)."""
    if "k_min" not in e:
        return np.ones(len(ev), bool)
    k = ev["gs"] ** 2 / np.maximum(ev["d"], 1.0)
    return _in_window(k, e["k_min"], e["k_max"])


def run_search(R, flights, events, dem, bbox=None, grid_m=200.0, c=343.0, rec_start=None,
               rec_start_tol=None, rec_duration=None, quiet_intervals=None, event_classes=("prop", "rotor", "unknown"),
               d_quiet=None, guard_s=25.0, mic_height=2.0, offset_tol=None, allow_same_aircraft=True,
               max_nodes=3_000_000):
    """events: list of 1 or 2 search-constraint dicts (physics.search_constraints).
    rec_start: absolute UTC (unix) of recording t=0 if known (+-rec_start_tol s).
    quiet_intervals: list of (t0, t1) recording-time spans that show NO fly-by;
                     default = whole recording minus +-guard around observed events.
    d_quiet: {class: slant-distance threshold in m} - any such flight with CPA closer than
             this inside a quiet interval disqualifies the candidate.
    Returns (candidates structured array, funnel stats dict)."""
    grid = Grid(grid_m)
    stats = {}
    d_quiet = d_quiet or {"prop": 3000.0, "rotor": 2500.0, "unknown": 3000.0, "jet": 2500.0}
    cls_arr = np.asarray(flights.cls)
    samp_cls = cls_arr[R.fid]
    in_bbox = np.ones(R.n, bool)
    if bbox:
        in_bbox = (R.lat >= bbox[0]) & (R.lat <= bbox[1]) & (R.lon >= bbox[2]) & (R.lon <= bbox[3])

    e1 = events[0]
    m1 = np.isin(samp_cls, list(event_classes)) & in_bbox
    m1 &= _in_window(R.gs, e1["v_min"] - 3, e1["v_max"] + 3)
    if rec_start is not None:
        tol = (rec_start_tol or 0) + e1["t_tol"] + e1["d_max"] / c + 60
        t1_abs = rec_start + e1["t_rec"]
        m1 &= _in_window(R.t, t1_abs - tol, t1_abs + tol)
    s1 = np.flatnonzero(m1)
    stats["flights_total"] = int(flights.nflights)
    stats["flights_event1_speed_class"] = int(np.unique(R.fid[s1]).size)
    log(f"A: {stats['flights_event1_speed_class']} flights match event-1 class/speed")

    node_ids = corridor_nodes(R, s1, grid, e1["d_max"])
    if bbox:
        nl, no = grid.node_latlon(node_ids)
        node_ids = node_ids[(nl >= bbox[0]) & (nl <= bbox[1]) & (no >= bbox[2]) & (no <= bbox[3])]
    if node_ids.size > max_nodes:
        raise RuntimeError(f"{node_ids.size} corridor nodes - narrow bbox/time window or tile the search")
    stats["corridor_nodes"] = int(node_ids.size)
    log(f"A: {node_ids.size} corridor nodes")
    nlat, nlon = grid.node_latlon(node_ids)
    nelev = dem.sample(nlat, nlon)
    necef = geodetic_to_ecef(nlat, nlon, nelev + mic_height)

    ev1 = cpa_events(necef, R, s1, e1["d_max"] * 1.02, c)
    ok = (~ev1["edge"]) & _in_window(ev1["d"], e1["d_min"], e1["d_max"]) & _in_window(ev1["gs"], e1["v_min"], e1["v_max"])
    ok &= _k_ok(ev1, e1)
    if rec_start is not None:
        ok &= np.abs(ev1["t_rx"] - (rec_start + e1["t_rec"])) <= (rec_start_tol or 0) + e1["t_tol"]
    ev1 = ev1[ok]
    stats["event1_matches"] = int(ev1.size)
    stats["nodes_after_event1"] = int(np.unique(ev1["node"]).size)
    log(f"A: {ev1.size} event-1 CPA matches at {stats['nodes_after_event1']} nodes")
    if ev1.size == 0:
        log("no node hears a flight matching event 1 - widen v/d windows, bbox or time window, or check data coverage")
        stats["candidates"] = 0
        return np.zeros(0, CAND_DTYPE), stats

    # ---- B: offset filter --------------------------------------------------
    if len(events) >= 2:
        e2 = events[1]
        dT = e2["t_rec"] - e1["t_rec"]
        tol = offset_tol if offset_tol is not None else np.hypot(e1["t_tol"], e2["t_tol"])
        cand_nodes = np.unique(ev1["node"])
        m2 = np.isin(samp_cls, list(event_classes)) & _in_window(R.gs, e2["v_min"] - 3, e2["v_max"] + 3)
        t_lo, t_hi = ev1["t_rx"].min() + dT - tol - e2["d_max"] / c - 30, ev1["t_rx"].max() + dT + tol + 30
        m2 &= _in_window(R.t, t_lo, t_hi)
        s2 = np.flatnonzero(m2)
        ev2 = cpa_events(necef[cand_nodes], R, s2, e2["d_max"] * 1.02, c)
        ev2["node"] = cand_nodes[ev2["node"]]
        ok2 = (~ev2["edge"]) & _in_window(ev2["d"], e2["d_min"], e2["d_max"]) & _in_window(ev2["gs"], e2["v_min"], e2["v_max"])
        ok2 &= _k_ok(ev2, e2)
        ev2 = ev2[ok2]
        pairs = _pair(ev1, ev2, dT, tol, allow_same_aircraft) if ev2.size else np.zeros((0, 2), np.int64)
        stats["offset_pairs"] = int(len(pairs))
        stats["nodes_after_offset"] = int(np.unique(ev1["node"][pairs[:, 0]]).size) if len(pairs) else 0
        log(f"B: {len(pairs)} (event1, event2) pairs at {stats['nodes_after_offset']} nodes (dT={dT:.1f}+-{tol:.1f}s)")
    else:
        ev2 = np.zeros(0, EV_DTYPE)
        pairs = np.c_[np.arange(ev1.size), np.full(ev1.size, -1)]

    # ---- C: silence filter -------------------------------------------------
    if rec_duration is not None and len(pairs):
        if quiet_intervals is None:
            busy = [(e["t_rec"] - guard_s - e.get("half_span_s", 60), e["t_rec"] + guard_s + e.get("half_span_s", 60)) for e in events]
            quiet_intervals = _complement(busy, guard_s, rec_duration - guard_s)
        stats["quiet_intervals"] = quiet_intervals
        pnodes = ev1["node"][pairs[:, 0]]
        un = np.unique(pnodes)
        dq_max = max(d_quiet.values())
        t_anchor = ev1["t_rx"][pairs[:, 0]] - e1["t_rec"]  # absolute UTC of recording t=0 per pair
        # every flight counts here, even outside the bbox: it may still pass near a node
        mall = _in_window(R.t, t_anchor.min() - dq_max / c - 120, t_anchor.max() + rec_duration + 120)
        mall &= np.isin(samp_cls, list(d_quiet.keys()))
        sall = np.flatnonzero(mall)
        evq = cpa_events(necef[un], R, sall, dq_max, c)
        evq["node"] = un[evq["node"]]
        thr = np.array([d_quiet.get(k, 0.0) for k in cls_arr])[evq["fid"]]
        evq = evq[evq["d"] <= thr]
        bad = _silence_violations(pairs, ev1, ev2, evq, t_anchor, quiet_intervals)
        stats["pairs_failing_silence"] = int(bad.sum())
        pairs = pairs[~bad]
        stats["pairs_after_silence"] = int(len(pairs))
        stats["nodes_after_silence"] = int(np.unique(ev1["node"][pairs[:, 0]]).size) if len(pairs) else 0
        log(f"C: {len(pairs)} pairs survive silence filter at {stats['nodes_after_silence']} nodes")

    # ---- assemble ----------------------------------------------------------
    cand = np.zeros(len(pairs), CAND_DTYPE)
    if len(pairs):
        a = ev1[pairs[:, 0]]
        k = a["node"]
        cand["node_id"], cand["lat"], cand["lon"], cand["elev"] = node_ids[k], nlat[k], nlon[k], nelev[k]
        cand["fid1"], cand["t1_cpa"], cand["t1_rx"], cand["d1"], cand["v1"] = a["fid"], a["t_cpa"], a["t_rx"], a["d"], a["gs"]
        has2 = pairs[:, 1] >= 0
        cand["fid2"] = -1
        if has2.any():
            b = ev2[pairs[has2, 1]]
            cand["fid2"][has2], cand["t2_cpa"][has2], cand["t2_rx"][has2] = b["fid"], b["t_cpa"], b["t_rx"]
            cand["d2"][has2], cand["v2"][has2] = b["d"], b["gs"]
    stats["candidates"] = int(len(cand))
    return cand, stats


def _pair(ev1, ev2, dT, tol, allow_same_aircraft=True):
    if ev1.size == 0 or ev2.size == 0:
        return np.zeros((0, 2), np.int64)
    o2 = np.lexsort((ev2["t_rx"], ev2["node"]))
    n2, t2 = ev2["node"][o2], ev2["t_rx"][o2]
    tref = min(ev1["t_rx"].min(), t2.min())
    key2 = n2.astype(np.float64) * 1e7 + (t2 - tref)
    k_lo = ev1["node"].astype(np.float64) * 1e7 + (ev1["t_rx"] + dT - tol - tref)
    k_hi = ev1["node"].astype(np.float64) * 1e7 + (ev1["t_rx"] + dT + tol - tref)
    a = np.searchsorted(key2, k_lo, "left")
    b = np.searchsorted(key2, k_hi, "right")
    cnt = b - a
    i1 = np.repeat(np.arange(ev1.size), cnt)
    i2 = o2[np.concatenate([np.arange(x, y) for x, y in zip(a, b)])] if cnt.sum() else np.zeros(0, np.int64)
    p = np.c_[i1, i2].astype(np.int64)
    if len(p) and not allow_same_aircraft:
        p = p[ev1["fid"][p[:, 0]] != ev2["fid"][p[:, 1]]]
    return p


def _complement(busy, lo, hi):
    busy = sorted((max(lo, a), min(hi, b)) for a, b in busy if b > lo and a < hi)
    out, cur = [], lo
    for a, b in busy:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        out.append((cur, hi))
    return out


def _silence_violations(pairs, ev1, ev2, evq, t_anchor, quiet):
    """True for pairs where some flight (not one of the pair's own events)
    reaches its CPA within a quiet interval."""
    bad = np.zeros(len(pairs), bool)
    if evq.size == 0 or not quiet:
        return bad
    pn = ev1["node"][pairs[:, 0]]
    o = np.argsort(pn, kind="stable")
    pn_s = pn[o]
    a = np.searchsorted(pn_s, evq["node"], "left")
    b = np.searchsorted(pn_s, evq["node"], "right")
    cnt = b - a
    if cnt.sum() == 0:
        return bad
    qi = np.repeat(np.arange(evq.size), cnt)
    pi = o[np.concatenate([np.arange(x, y) for x, y in zip(a, b)])]
    rec_t = evq["t_rx"][qi] - t_anchor[pi]
    inq = np.zeros(qi.size, bool)
    for q0, q1 in quiet:
        inq |= (rec_t >= q0) & (rec_t <= q1)
    own1 = (evq["fid"][qi] == ev1["fid"][pairs[pi, 0]]) & (np.abs(evq["t_cpa"][qi] - ev1["t_cpa"][pairs[pi, 0]]) < 30)
    has2 = pairs[pi, 1] >= 0
    j2 = np.where(has2, pairs[pi, 1], 0)
    own2 = has2 & (evq["fid"][qi] == ev2["fid"][j2] if ev2.size else False) & \
        (np.abs(evq["t_cpa"][qi] - (ev2["t_cpa"][j2] if ev2.size else 0)) < 30)
    viol = inq & ~own1 & ~own2
    bad[np.unique(pi[viol])] = True
    return bad
