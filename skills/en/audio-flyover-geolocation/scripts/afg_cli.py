#!/usr/bin/env python3
"""afg - acoustic fly-over geolocation pipeline. Run `python afg_cli.py <command> -h`.

Typical sequence
  spectrogram -> detect -> (plot / trace / fit per event) -> adsb-ingest -> [dem-fetch]
  -> search -> score  ->  verify visually
Other: thunder, audit, simulate, selftest
"""
import argparse
import csv
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from afg import adsb, audio, physics  # noqa: E402
from afg.common import dump_json, iso, load_json, log, parse_bbox, parse_time  # noqa: E402
from afg.dem import ElevationModel, fetch_copernicus  # noqa: E402
from afg.geo import Grid, haversine_m  # noqa: E402


# ----------------------------------------------------------------------------- helpers

def _c(args):
    if getattr(args, "c", None):
        return float(args.c)
    if getattr(args, "temp_c", None) is not None:
        return float(physics.sound_speed(args.temp_c))
    return 343.0


def _dem(args):
    if getattr(args, "dem", None):
        return ElevationModel(paths=args.dem)
    if getattr(args, "ground_elev", None) is not None:
        log(f"WARNING: constant ground elevation {args.ground_elev} m (no DEM) - slant distances degrade in relief")
        return ElevationModel(constant=args.ground_elev)
    raise SystemExit("need --dem TIF [TIF ...] or --ground-elev METRES")


def _read_trace(path):
    t, f = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("t,"):
                continue
            a = line.strip().split(",")
            if len(a) >= 2:
                t.append(float(a[0]))
                f.append(float(a[1]))
    return np.array(t), np.array(f)


def _trace_meta(path):
    meta = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                for kv in line[1:].strip().split():
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        meta[k] = v
            else:
                break
    return meta


def _load_events(paths):
    evs, traces = [], []
    for p in paths:
        e = load_json(p)
        if e.get("implausible"):
            raise SystemExit(f"{p}: fit flagged implausible ({'; '.join(e['implausible'])}). Fix the trace first "
                             "(narrower box around one harmonic / --anchors) or delete the 'implausible' list "
                             "if you have independently confirmed the fit.")
        c = dict(e["constraints"])
        c["half_span_s"] = e.get("half_span_s", 60.0)
        c["name"] = e.get("name", os.path.basename(p))
        evs.append(c)
        tp = e.get("trace_csv")
        traces.append(_read_trace(tp) if tp and os.path.exists(tp) else None)
    return evs, traces


# ----------------------------------------------------------------------------- commands

def cmd_spectrogram(a):
    x, fs, nch = audio.load_audio(a.audio, fs=a.fs, channel=a.channel)
    log(f"audio: {len(x) / fs:.1f} s @ {fs} Hz ({nch} ch)")
    spec = audio.compute_spectrogram(x, fs, a.win, a.hop, a.fmin, a.fmax, a.bg)
    audio.save_spec(spec, a.out + ".spec.npz", a.audio)
    audio.plot_spec(spec, a.out + ".png", a.t0, a.t1, a.fmin, a.fmax,
                    title=f"{os.path.basename(a.audio)}  win={a.win}s hop={a.hop}s")
    print(json.dumps({"spec": a.out + ".spec.npz", "png": a.out + ".png", "duration_s": spec["duration"],
                      "df_hz": float(spec["f"][1] - spec["f"][0]), "frames": int(spec["t"].size)}, indent=2))


def cmd_plot(a):
    spec = audio.load_spec(a.spec)
    ov = []
    for p in a.overlay or []:
        t, f = _read_trace(p)
        ov.append({"t": t, "f": f, "style": ".", "lw": 0, "color": "lime", "label": os.path.basename(p)})
    audio.plot_spec(spec, a.out, a.t0, a.t1, a.fmin, a.fmax, overlays=ov, vmax=a.vmax)
    print(a.out)


def cmd_detect(a):
    spec = audio.load_spec(a.spec)
    ev, info = audio.detect_events(spec, a.fmin, a.fmax, a.harmonics, a.thr, a.min_dur, win_s=a.window,
                                   step_s=a.step, null_db=a.null_db, return_info=True)
    ev.sort(key=lambda e: e["t0"])
    dump_json(ev, a.out)
    diag = os.path.splitext(a.out)[0] + ".diag.json"
    dump_json(info, diag)
    for i, e in enumerate(ev):
        log(f"event {i + 1}: t={e['t0']:.0f}-{e['t1']:.0f}s f={e['fmin']:.1f}-{e['fmax']:.1f}Hz "
            f"peak={e['peak_snr_db']:.1f}dB ({e['basis']} baseline)")
    log(f"detection status: {info['status']} ({info['n_valid_windows']} usable windows, {info['baseline']} baseline)")
    for r in info["reasons"]:
        log("  -", r)
    print(json.dumps({"detection": info, "diag": diag, "events": ev}, indent=2))


def cmd_trace(a):
    spec = audio.load_spec(a.spec)
    anchors = None
    if a.anchors:
        anchors = [tuple(float(v) for v in p.split(":")) for p in a.anchors.split(",")]
    t, f, s = audio.trace_curve(spec, a.t0, a.t1, a.fmin, a.fmax, n_harm=a.harmonics, anchors=anchors,
                                corridor_hz=a.corridor, min_snr=a.min_snr, monotone=not a.no_monotone,
                                penalty=a.penalty)
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# win_s={spec['win_s']} hop_s={spec['hop_s']} spec={os.path.abspath(a.spec)}\n")
        fh.write("t,f,snr_db\n")
        for row in zip(t, f, s):
            fh.write("%.3f,%.4f,%.2f\n" % row)
    png = os.path.splitext(a.out)[0] + ".png"
    audio.plot_spec(spec, png, a.t0, a.t1, a.fmin, a.fmax, points=(t, f), title="trace " + os.path.basename(a.out))
    k = audio.harmonics_used(spec, a.fmax, a.harmonics)
    if k < a.harmonics:
        log(f"note: only {k} of {a.harmonics} harmonics fit under the spectrogram top ({spec['f'][-1]:.0f} Hz) "
            f"for --fmax {a.fmax:g}; a narrower box around one harmonic stacks more")
    print(json.dumps({"points": int(t.size), "csv": a.out, "png": png, "harmonics_used": k,
                      "f_range": [float(f.min()), float(f.max())] if t.size else None}, indent=2))


def cmd_fit(a):
    t, f = _read_trace(a.trace)
    meta = _trace_meta(a.trace)
    sel = np.ones(t.size, bool)
    if a.tmin is not None:
        sel &= t >= a.tmin
    if a.tmax is not None:
        sel &= t <= a.tmax
    t, f = t[sel], f[sel]
    c = _c(a)
    corr = np.sqrt(float(meta.get("win_s", 1.0)) / float(meta.get("hop_s", 1.0)))
    fit = physics.fit_linear(t, f, c=c, corr_factor=corr)
    split = physics.fit_split(t, f, fit, c=c, corr_factor=corr)
    cons = physics.search_constraints(fit, a.v_margin, a.d_margin, a.t_margin, a.c_uncert, k_rel_margin=a.k_margin,
                                      alt=split)
    asym = fit["rms_left_hz"] / fit["rms_right_hz"] if fit["rms_right_hz"] else np.nan
    out = {"name": a.name or os.path.splitext(os.path.basename(a.trace))[0], "trace_csv": os.path.abspath(a.trace),
           "fit": {k: v for k, v in fit.items() if k != "cov"}, "fit_split": split, "constraints": cons,
           "half_span_s": float(max(fit["t_cpa_rx"] - t.min(), t.max() - fit["t_cpa_rx"])),
           "notes": [], "implausible": physics.plausibility(fit)}
    for w in out["implausible"]:
        log("WARNING:", w, "- re-trace with a narrower --fmin/--fmax box around ONE harmonic, or --anchors")
    if np.isfinite(asym) and (asym > 2 or asym < 0.5):
        out["notes"].append("left/right residual asymmetry: aircraft probably turned or changed speed - "
                            "keep wide v/d margins; trajectory scoring will use the real path")
    if split and abs(split["v_out"] - split["v_in"]) > 0.08 * fit["v"]:
        out["notes"].append(f"approach/recede speeds differ ({split['v_in']:.0f} vs {split['v_out']:.0f} m/s): speed or "
                            "climb changed during the pass - v and d individually biased; k=v^2/d window still holds")
    if fit["d"] < 150:
        out["notes"].append("very small CPA distance: could be a ground vehicle/drone, not an aircraft")
    if fit["v"] < 25:
        out["notes"].append("speed < 25 m/s: likely road vehicle, boat, drone or helicopter hover-taxi")
    dump_json(out, a.out)
    tt = np.linspace(t.min(), t.max(), 600)
    spec_path = meta.get("spec")
    if spec_path and os.path.exists(spec_path):
        spec = audio.load_spec(spec_path)
        png = os.path.splitext(a.out)[0] + ".fit.png"
        ovs = [{"t": tt, "f": physics.doppler_linear(tt, fit["f0"], fit["v"], fit["d"], fit["t_cpa_rx"], c),
                "color": "cyan", "label": "linear fit"}]
        if split:
            ovs.append({"t": tt, "f": physics.doppler_split(tt, split["f0"], split["v_in"], split["v_out"], split["d"],
                                                            split["t_cpa_rx"], c), "color": "orange", "label": "split fit"})
        audio.plot_spec(spec, png, t.min() - 10, t.max() + 10, f.min() - 5, f.max() + 5, points=(t, f), overlays=ovs,
                        title=f"fit: f0={fit['f0']:.2f}Hz v={fit['v']:.1f}m/s d={fit['d']:.0f}m tCPA={fit['t_cpa_rx']:.1f}s")
    print(json.dumps(out, indent=2, default=float))


def cmd_adsb_ingest(a):
    t0, t1 = parse_time(a.start), parse_time(a.end)
    bbox = parse_bbox(a.bbox)
    types = adsb.load_type_table(a.types, allow_download=not a.offline)
    if a.format == "readsb":
        acc = adsb.ingest_readsb(a.src, t0, t1, bbox, max_gap=a.max_gap, types=types)
        geoid_n = a.geoid_n
    else:
        acc = None
        for s in a.src:
            acc = adsb.ingest_csv(s, t0, t1, bbox, fmt=a.format, max_gap=a.max_gap, types=types, acc=acc)
        geoid_n = 0.0 if a.format == "generic" else a.geoid_n
    cols, flights = adsb.finalize(acc, a.alt_source, a.qnh, geoid_n)
    meta = {"t0": t0, "t1": t1, "bbox": bbox, "alt_source": a.alt_source, "qnh_hpa": a.qnh, "geoid_n": geoid_n,
            "format": a.format, "src": a.src}
    adsb.save_tracks(a.out, cols, flights, meta)
    cls = {}
    for f in flights:
        cls[f["cls"]] = cls.get(f["cls"], 0) + 1
    print(json.dumps({"segments": len(flights), "samples": int(cols["t"].size), "classes": cls, "out": a.out}, indent=2))


def cmd_dem_fetch(a):
    paths = fetch_copernicus(parse_bbox(a.bbox), a.out)
    print(json.dumps(paths, indent=2))


def _resampled(tracks, t0=None, t1=None, dt=1.0):
    from afg.tracks import Flights, Resampled
    fl = Flights(tracks)
    R = Resampled(fl, dt=dt, t0=t0, t1=t1)
    return fl, R


def cmd_search(a):
    from afg.search import run_search
    events, _ = _load_events(a.events)
    rec_start = parse_time(a.rec_start) if a.rec_start else None
    t0 = parse_time(a.start) if a.start else None
    t1 = parse_time(a.end) if a.end else None
    fl, R = _resampled(a.tracks, t0, t1, a.dt)
    quiet = None
    if a.quiet:
        quiet = [tuple(float(v) for v in s.split("-")) for s in a.quiet.split(",")]
    dq = None
    if a.d_quiet:
        dq = {k: float(v) for k, v in (p.split("=") for p in a.d_quiet.split(","))}
    cand, stats = run_search(R, fl, events, _dem(a), parse_bbox(a.bbox), a.grid, _c(a), rec_start, a.rec_start_tol,
                             a.rec_duration, quiet, tuple(a.event_classes.split(",")), dq, a.guard, a.mic_height,
                             a.offset_tol, not a.distinct_aircraft)
    np.save(a.out, cand)
    stats["c"] = _c(a)
    dump_json(stats, os.path.splitext(a.out)[0] + ".stats.json")
    print(json.dumps(stats, indent=2, default=float))


def cmd_score(a):
    from afg.scoring import ambiguity, heatmap, model_overlay, pass_geometry, score_candidates, score_region
    events, traces = _load_events(a.events)
    if not a.trace_mode:
        traces = None
    cand = np.load(a.cands)
    if cand.size == 0:
        raise SystemExit("no candidates to score")
    tmin = np.nanmin(cand["t1_cpa"]) - 1800
    tmax = np.nanmax(np.r_[cand["t1_cpa"], np.nan_to_num(cand["t2_cpa"], nan=0)]) + 1800
    fl, R = _resampled(a.tracks, tmin, tmax, a.dt)
    spec = audio.load_spec(a.spec)
    dem = _dem(a)
    c = _c(a)
    harm = [float(h) for h in a.harmonics.split(",")]
    clock = (parse_time(a.rec_start), a.rec_start_tol) if a.rec_start else None
    ranked, ucand, cs = score_candidates(cand, R, spec, events, dem, c, a.mic_height, a.top_pairs, 4, a.sub_m, 1,
                                         traces, harm, a.lock, a.half_window, a.delta_max, a.f_rel, clock)
    if not ranked:
        raise SystemExit("nothing scored")
    amb = ambiguity(ranked, a.ambiguity_frac)
    plateau = score_region(ucand, cs, a.ambiguity_frac, ranked[0])
    with open(os.path.splitext(a.out)[0] + ".plateau.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lat", "lon"])
        w.writerows([[f"{x:.6f}", f"{y:.6f}"] for x, y in zip(plateau["lat"], plateau["lon"])])
    for r in ranked:
        r["flight1"] = fl.label(r["fid1"])
        r["flight2"] = fl.label(r["fid2"]) if r["fid2"] >= 0 else None
        r["rec_start_iso"] = iso(r["rec_start_utc"])
        r["maps"] = f"https://www.google.com/maps/search/?api=1&query={r['lat']:.6f},{r['lon']:.6f}"
    tot = np.array([r["total"] for r in ranked])
    summary = {"best": ranked[0], "n_hypotheses": len(ranked),
               "margin_vs_2nd": float((tot[0] - tot[1]) / max(tot[0], 1e-9)) if tot.size > 1 else None,
               "robust_z": float((tot[0] - np.median(tot)) / (1.4826 * np.median(np.abs(tot - np.median(tot))) + 1e-9)) if tot.size > 3 else None,
               "ambiguous_alternatives": amb,
               "coarse_plateau": {k: v for k, v in plateau.items() if k not in ("lat", "lon")},
               "geometry": pass_geometry(R, ranked[0], len(events)),
               "verdict": ("AMBIGUOUS - resolve with another event/visual check" if amb else "unique within scored set")
               + (f"; coarse plateau is {plateau['length_m']:.0f} m long - treat the answer as a strip, not a point"
                  if len(events) == 1 and plateau["length_m"] > 600 else ""),
               "top": ranked[: a.report]}
    if "warning" in summary["geometry"]:
        summary["verdict"] += "; " + summary["geometry"]["warning"]
    base = os.path.splitext(a.out)[0]
    # overlays for top-3
    for k, r in enumerate(ranked[:3]):
        ov = model_overlay(R, r, events, dem, c, a.mic_height)
        if not ov:
            continue
        for e, o in zip(events, ov):
            o["color"] = "cyan"
            lo, hi = np.nanmin(o["f"]), np.nanmax(o["f"])
            audio.plot_spec(spec, f"{base}.rank{k + 1}.{e['name']}.png", e["t_rec"] - 90, e["t_rec"] + 90, lo - 8, hi + 8,
                            overlays=[o], title=f"rank {k + 1} score={r['total']:.2f}  {e['name']}")
    if not a.no_heatmap:
        la, lo, tot_g, per_g, E, N = heatmap(R, spec, events, dem, ranked[0], c, a.mic_height, a.heat_radius,
                                             a.heat_step, traces, harm, a.lock, a.half_window, a.delta_max, a.f_rel)
        _plot_heat(base + ".heatmap.png", E, N, tot_g, per_g, ranked[0])
        k = np.nanargmax(tot_g)
        summary["heatmap_peak"] = {"lat": float(la.ravel()[k]), "lon": float(lo.ravel()[k]), "score": float(np.nanmax(tot_g))}
        thr = np.nanmax(tot_g) * a.region_frac
        m = tot_g >= thr
        summary["region_frac"] = a.region_frac
        summary["region_area_m2"] = float(m.sum() * a.heat_step**2)
        summary["region_max_radius_m"] = float(np.sqrt((E[m] - E.ravel()[k]) ** 2 + (N[m] - N.ravel()[k]) ** 2).max())
        with open(base + ".heatmap.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["lat", "lon", "score"] + [f"event{i + 1}" for i in range(per_g.shape[0])])
            for i in range(la.size):
                w.writerow([f"{la.ravel()[i]:.6f}", f"{lo.ravel()[i]:.6f}", f"{tot_g.ravel()[i]:.4f}"]
                           + [f"{per_g[j].ravel()[i]:.4f}" for j in range(per_g.shape[0])])
    dump_json(summary, a.out)
    print(json.dumps({k: summary[k] for k in summary if k != "top"}, indent=2, default=float))


def _plot_heat(path, E, N, tot, per, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 1 + per.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 4.6), dpi=110)
    axs = np.atleast_1d(axs)
    for ax, (img, ttl) in zip(axs, [(tot, "combined")] + [(per[i], f"event {i + 1}") for i in range(per.shape[0])]):
        im = ax.pcolormesh(E, N, img, shading="auto", cmap="inferno")
        ax.set_aspect("equal")
        ax.set_title(ttl)
        ax.set_xlabel("east (m)")
        ax.set_ylabel("north (m)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"score around {best['lat']:.5f},{best['lon']:.5f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cmd_thunder(a):
    from afg import thunder
    x, fs, _ = audio.load_audio(a.audio, fs=1000)
    env = thunder.observed_envelope(x, fs, rate=a.rate)
    st = {"t": [], "lat": [], "lon": []}
    with open(a.strikes, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            st["t"].append(parse_time(r["time"]))
            st["lat"].append(float(r["lat"]))
            st["lon"].append(float(r["lon"]))
    st = {k: np.array(v) for k, v in st.items()}
    t0, t1 = parse_time(a.start), parse_time(a.end)
    m = (st["t"] >= t0 - 120) & (st["t"] <= t1 + len(env) / a.rate + 120)
    st = {k: v[m] for k, v in st.items()}
    bb = parse_bbox(a.bbox)
    ids, la, lo = Grid(a.grid).nodes_in_bbox(*bb)
    el = _dem(a).sample(la, lo)
    log(f"thunder: {st['t'].size} strikes, {la.size} grid nodes")
    sc, bt = thunder.locate(env, a.rate, st, la, lo, el, t0, t1, _c(a), a.source_h, a.p)
    o = np.argsort(-sc)[: a.report]
    res = [{"lat": float(la[i]), "lon": float(lo[i]), "cosine": float(sc[i]), "rec_start_iso": iso(bt[i])} for i in o]
    with open(os.path.splitext(a.out)[0] + ".grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lat", "lon", "cosine", "rec_start_unix"])
        for i in range(la.size):
            w.writerow([f"{la[i]:.6f}", f"{lo[i]:.6f}", f"{sc[i]:.4f}", f"{bt[i]:.1f}"])
    dump_json({"top": res, "n_nodes": int(la.size), "n_strikes": int(st["t"].size)}, a.out)
    print(json.dumps(res[:10], indent=2))


def cmd_audit(a):
    """Privacy audit of YOUR OWN recording before publishing it."""
    x, fs, nch = audio.load_audio(a.audio, fs=4000)
    spec = audio.compute_spectrogram(x, fs, 1.0, 0.2, 20, 600, 120)
    ev, info = audio.detect_events(spec, 40, 400, 1, a.thr, 12.0, return_info=True)
    dur = len(x) / fs
    if len(ev) >= 2:
        risk = "high"
    elif len(ev) == 1:
        risk = "medium (high if the recording time is known or the area is small)"
    elif info["status"] == "insufficient" or info["short_hits"]:
        risk = "undetermined (the automatic check could not decide; see detection.reasons and inspect the PNG)"
    elif info["status"] == "limited":
        risk = "low, limited check (see detection.reasons; inspect the PNG)"
    else:
        risk = "low"
    out = {"duration_s": dur, "detection": info, "candidate_flyovers": ev, "risk": risk,
           "risk_scope": "covers only sustained aircraft tones (about 40-400 Hz) that this detector can see; "
                         "'low' is not a guarantee - look for smooth descending curves in the PNG",
           "advice": ["cut the audio so no two fly-overs remain with an intact time gap",
                      "replace/strip the audio track, or re-record the narration separately",
                      "a high-pass filter, talking over it, or phone-call codecs do NOT reliably remove it",
                      "also check visual clues, metadata (GPS/EXIF/QuickTime), reflections and voices"]}
    audio.plot_spec(spec, a.out + ".png", fmin=20, fmax=500, title="audit: look for smooth descending curves")
    dump_json(out, a.out + ".json")
    print(json.dumps(out, indent=2))


def cmd_simulate(a):
    from afg.simulate import simulate
    truth, wav, tdir = simulate(a.out, a.lat, a.lon, a.ground, a.seed)
    print(json.dumps({"wav": wav, "traces": tdir, "truth": os.path.join(a.out, "truth.json")}, indent=2))


def cmd_selftest(a):
    """End-to-end blind run on a synthetic scenario. Exit code 0 = pass."""
    from afg.scoring import score_candidates
    from afg.search import run_search
    from afg.simulate import simulate
    from afg.tracks import Flights, Resampled
    work = a.keep or tempfile.mkdtemp(prefix="afg_selftest_")
    t_start = time.time()
    truth, wav, tdir = simulate(work, seed=a.seed)
    log("simulated:", work)
    x, fs, _ = audio.load_audio(wav, fs=4000)
    spec = audio.compute_spectrogram(x, fs, 1.0, 0.2, 20, 600, 120)
    audio.save_spec(spec, os.path.join(work, "rec.spec.npz"), wav)
    boxes = sorted(audio.detect_events(spec, 40, 400, 1), key=lambda e: -e["score_z"])[:2]
    boxes.sort(key=lambda e: e["t0"])
    log("detected boxes:", [(round(b["t0"]), round(b["t1"]), round(b["fmin"], 1), round(b["fmax"], 1)) for b in boxes])
    if len(boxes) < 2:
        log("FAIL: fewer than two fly-overs detected")
        return 1
    c = 343.0
    events, rep = [], []
    for k, b in enumerate(boxes):
        t, f, s = audio.trace_curve(spec, b["t0"], b["t1"], b["fmin"], b["fmax"], n_harm=3)
        fit = physics.fit_linear(t, f, c=c, corr_factor=np.sqrt(1.0 / 0.2))
        cons = physics.search_constraints(fit, alt=physics.fit_split(t, f, fit, c=c, corr_factor=np.sqrt(1.0 / 0.2)))
        cons["half_span_s"] = float(max(fit["t_cpa_rx"] - t.min(), t.max() - fit["t_cpa_rx"]))
        cons["name"] = f"e{k + 1}"
        events.append(cons)
        tr = truth["events"][k]
        rep.append({"event": k + 1, "fit_v": fit["v"], "true_v": tr["speed_ms"], "fit_d": fit["d"], "true_d": tr["d_cpa_m"],
                    "fit_tcpa_rec": fit["t_cpa_rx"], "true_tcpa_rec": tr["t_cpa_rec_s"],
                    "implausible": physics.plausibility(fit)})
    log("event fits vs truth:", json.dumps(rep, default=float))
    if any(r["implausible"] for r in rep):
        out = {"pass": False, "reason": "unphysical event fit (tracing failed)", "event_fits": rep, "workdir": work}
        dump_json(out, os.path.join(work, "selftest.json"))
        print(json.dumps(out, indent=2, default=float))
        return 1
    types = adsb.load_type_table(allow_download=True)
    day = [parse_time(x) for x in truth["day_window"]]
    acc = adsb.ingest_readsb([tdir], day[0], day[1], None, types=types)
    cols, flights = adsb.finalize(acc)
    adsb.save_tracks(os.path.join(work, "tracks.npz"), cols, flights, {})
    fl = Flights(os.path.join(work, "tracks.npz"))
    R = Resampled(fl, dt=1.0)
    dem = ElevationModel(constant=truth["observer"]["ground_elev_m"])
    bbox = tuple(truth["bbox"])
    results = {}
    # Mode A: two events, unknown time, whole area
    hexes = [str(h) for h in fl.hex]
    true_f = [{i for i, h in enumerate(hexes) if h == e["hex"]} for e in truth["events"]]

    def funnel_has_truth(cd, ks):
        ok = np.isin(cd["fid1"], list(true_f[ks[0]]))
        if len(ks) > 1:
            ok &= np.isin(cd["fid2"], list(true_f[ks[1]]))
        if not ok.any():
            return {"truth_pair_in_funnel": False}
        dmin = float(haversine_m(cd["lat"][ok], cd["lon"][ok], truth["observer"]["lat"], truth["observer"]["lon"]).min())
        return {"truth_pair_in_funnel": True, "nearest_true_pair_node_m": dmin}

    cand, stats = run_search(R, fl, events, dem, bbox, 200.0, c, rec_duration=truth["rec_duration_s"])
    log("funnel A:", json.dumps(stats, default=float))
    ranked, _, _ = score_candidates(cand, R, spec, events, dem, c) if cand.size else ([], None, None)
    results["two_event_unknown_time"] = dict(_judge(ranked, truth, fl), funnel=stats, **funnel_has_truth(cand, [0, 1]))
    if ranked:
        # local confidence region (>= 0.9 x peak) of the fine score map: the honest
        # error bar.  Pass also when the truth lies inside a compact region.
        from afg.scoring import heatmap
        la, lo, tot_g, per_g, E, N = heatmap(R, spec, events, dem, ranked[0], c, radius_m=600.0, step_m=40.0)
        _plot_heat(os.path.join(work, "two_event.heatmap.png"), E, N, tot_g, per_g, ranked[0])
        pk = np.nanmax(tot_g)
        m = tot_g >= 0.9 * pk
        dt_ = haversine_m(la, lo, truth["observer"]["lat"], truth["observer"]["lon"])
        kt = np.unravel_index(np.argmin(dt_), dt_.shape)
        kp = np.unravel_index(np.nanargmax(tot_g), tot_g.shape)
        reg = {"region_max_radius_m": float(np.hypot(E[m] - E[kp], N[m] - N[kp]).max()),
               "truth_score_ratio": float(tot_g[kt] / pk), "truth_in_region": bool(m[kt] and dt_[kt] < 60)}
        from afg.scoring import pass_geometry
        r2 = results["two_event_unknown_time"]
        r2["region"] = reg
        r2["geometry"] = pass_geometry(R, ranked[0], 2)
        # honest-error-bar pass: right aircraft, truth inside the reported region,
        # and the error is no larger than the region radius we would report
        if (not r2["pass"] and r2["pair_correct"] and reg["truth_in_region"]
                and r2["error_m"] <= reg["region_max_radius_m"]):
            r2["pass"] = True
            r2["verdict"] = (f"inside reported 0.9-region (radius {reg['region_max_radius_m']:.0f} m)"
                             + ("; " + r2["geometry"]["warning"] if "warning" in r2["geometry"] else ""))
    # Mode B: one event only.  Clock known to +-2 s (file metadata from an NTP-synced
    # phone): expect a point fix, or for a straight pass a flagged mirror pair.
    # Clock known to +-30 s: along-track position is degenerate (~ +-v*30 s), so
    # the test is honesty - the reported score plateau must contain the truth.
    from afg.scoring import score_region
    quiet = [(5.0, events[0]["t_rec"] - 90), (events[0]["t_rec"] + 90, events[1]["t_rec"] - 90),
             (events[1]["t_rec"] + 90, truth["rec_duration_s"] - 5)]
    for k in (0, 1):
        ev_b = [dict(events[k])]
        for tol, tag in ((2.0, "clock2s"), (30.0, "clock30s")):
            cand_b, stats_b = run_search(R, fl, ev_b, dem, bbox, 200.0, c, rec_start=truth["rec_start_unix"],
                                         rec_start_tol=tol, rec_duration=truth["rec_duration_s"], quiet_intervals=quiet)
            log(f"funnel B{k + 1} {tag}:", json.dumps(stats_b, default=float))
            ranked_b, uc, cs = (score_candidates(cand_b, R, spec, ev_b, dem, c, clock=(truth["rec_start_unix"], tol))
                                if cand_b.size else ([], None, None))
            res = dict(_judge(ranked_b, truth, fl, single=k), **funnel_has_truth(cand_b, [k]))
            if ranked_b:
                pl = score_region(uc, cs, best=ranked_b[0])
                dtr = haversine_m(pl["lat"], pl["lon"], truth["observer"]["lat"], truth["observer"]["lon"])
                res["plateau"] = {"length_m": pl["length_m"], "width_m": pl["width_m"], "n_nodes": pl["n_nodes"],
                                  "truth_to_plateau_m": float(dtr.min()) if dtr.size else None}
                if tag == "clock30s":
                    res["pass"] = bool(dtr.size and dtr.min() < 250)
                    res["verdict"] = ("truth inside reported plateau" if res["pass"] else "truth OUTSIDE plateau") + \
                                     f" ({pl['length_m']:.0f} m long)"
            results[f"one_event_{tag}_e{k + 1}"] = res
    ok = all(r["pass"] for r in results.values())
    out = {"pass": ok, "results": results, "event_fits": rep, "workdir": work, "runtime_s": time.time() - t_start}
    dump_json(out, os.path.join(work, "selftest.json"))
    print(json.dumps(out, indent=2, default=float))
    return 0 if ok else 1


def _judge(ranked, truth, fl, single=None):
    """single=None: two-event run; single=k: one-event run using event k.
    Pass = rank-1 within 250 m with the right aircraft, or - for one-event runs
    only - the truth is the flagged mirror alternative (an honest 'ambiguous')."""
    from afg.scoring import ambiguity
    if not ranked:
        return {"pass": False, "reason": "no ranked candidates"}
    ob = truth["observer"]
    want = [e["hex"] for e in truth["events"]]
    want = want if single is None else [want[single]]

    def ids(r):
        return [str(fl.hex[r["fid1"]])] + ([str(fl.hex[r["fid2"]])] if r["fid2"] >= 0 else [])

    def err(r):
        return float(haversine_m(r["lat"], r["lon"], ob["lat"], ob["lon"]))

    best = ranked[0]
    amb = ambiguity(ranked)
    pair_ok = ids(best) == want
    e1 = err(best)
    tot = [r["total"] for r in ranked]
    res = {"error_m": e1, "pair_correct": pair_ok, "rec_start_error_s": abs(best["rec_start_utc"] - truth["rec_start_unix"]),
           "score": best["total"], "margin_vs_2nd": (tot[0] - tot[1]) / tot[0] if len(tot) > 1 else 1.0,
           "n_hypotheses": len(ranked), "ambiguous_alternatives": amb, "best": {"lat": best["lat"], "lon": best["lon"]}}
    if pair_ok and e1 < 250:
        res["pass"], res["verdict"] = True, ("resolved" if not amb else "correct, but flagged ambiguous")
    elif single is not None and amb:
        alt_ok = any(err(r) < 250 and ids(r) == want for r in ranked[1:1 + len(amb)])
        res["pass"] = bool(alt_ok)
        res["verdict"] = "mirror-ambiguous: truth is the flagged alternative" if alt_ok else "wrong"
    else:
        res["pass"], res["verdict"] = False, "wrong"
    return res


# ----------------------------------------------------------------------------- argparse

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def phys(sp):
        sp.add_argument("--c", type=float, help="speed of sound m/s (default 343, or from --temp-c)")
        sp.add_argument("--temp-c", type=float, help="air temperature (deg C) near the ground at recording time")

    def demargs(sp):
        sp.add_argument("--dem", nargs="+", help="GeoTIFF DEM tile(s) (Copernicus GLO-30, 3DEP, SRTM)")
        sp.add_argument("--ground-elev", type=float, help="constant ground elevation (m) instead of a DEM")
        sp.add_argument("--mic-height", type=float, default=2.0)

    s = sub.add_parser("spectrogram", help="audio/video -> whitened narrow-band spectrogram (.spec.npz + .png)")
    s.add_argument("audio")
    s.add_argument("--out", required=True, help="output prefix")
    s.add_argument("--fs", type=int, default=4000)
    s.add_argument("--channel", default="mix")
    s.add_argument("--win", type=float, default=1.0, help="window s (0.5-1 steep/close passes, 2-4 faint/distant)")
    s.add_argument("--hop", type=float, default=0.2)
    s.add_argument("--fmin", type=float, default=20.0)
    s.add_argument("--fmax", type=float, default=600.0)
    s.add_argument("--bg", type=float, default=120.0, help="background median window s")
    s.add_argument("--t0", type=float)
    s.add_argument("--t1", type=float)
    s.set_defaults(fn=cmd_spectrogram)

    s = sub.add_parser("plot", help="render a zoomed spectrogram region (optionally with trace overlays)")
    s.add_argument("spec")
    s.add_argument("--out", required=True)
    for k in ("t0", "t1", "fmin", "fmax", "vmax"):
        s.add_argument("--" + k, type=float)
    s.add_argument("--overlay", nargs="*")
    s.set_defaults(fn=cmd_plot)

    s = sub.add_parser("detect", help="find candidate fly-over boxes (sustained narrow-band tones)")
    s.add_argument("spec")
    s.add_argument("--out", required=True)
    s.add_argument("--fmin", type=float, default=40)
    s.add_argument("--fmax", type=float, default=400)
    s.add_argument("--harmonics", type=int, default=1,
                   help="harmonics stacked per candidate (1 = brightest single line; see references/spectrogram.md)")
    s.add_argument("--thr", type=float, default=0.8, help="min ridge excess (dB) over the recording median")
    s.add_argument("--null-db", type=float, default=0.8,
                   help="min ridge excess (dB) over the calibrated noise model (short clips, tonal recordings)")
    s.add_argument("--min-dur", type=float, default=15.0)
    s.add_argument("--window", type=float, default=60.0, help="detection window (s)")
    s.add_argument("--step", type=float, default=10.0, help="detection window step (s)")
    s.set_defaults(fn=cmd_detect)

    s = sub.add_parser("trace", help="trace one Doppler curve inside a time/frequency box -> CSV")
    s.add_argument("spec")
    s.add_argument("--out", required=True)
    for k in ("t0", "t1", "fmin", "fmax"):
        s.add_argument("--" + k, type=float, required=True)
    s.add_argument("--harmonics", type=int, default=3, help="harmonics stacked to aid tracking (1 = off)")
    s.add_argument("--anchors", help="'t:f,t:f,...' rough points on the curve (restricts search corridor)")
    s.add_argument("--corridor", type=float, default=3.0, help="Hz half-width around anchors")
    s.add_argument("--min-snr", type=float, default=3.0)
    s.add_argument("--penalty", type=float, default=0.3)
    s.add_argument("--no-monotone", action="store_true", help="allow rising frequency (manoeuvring aircraft)")
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("fit", help="fit Doppler model to a trace -> event JSON (v, d, CPA time, f0, search windows)")
    s.add_argument("trace")
    s.add_argument("--out", required=True)
    s.add_argument("--name")
    s.add_argument("--tmin", type=float, help="fit only t >= tmin (e.g. the clean side of the curve)")
    s.add_argument("--tmax", type=float)
    s.add_argument("--v-margin", type=float, default=0.20, help="relative speed window (wide on purpose)")
    s.add_argument("--k-margin", type=float, default=0.08, help="relative window on k = v^2/d (curve steepness)")
    s.add_argument("--d-margin", type=float, default=0.15)
    s.add_argument("--t-margin", type=float, default=4.0)
    s.add_argument("--c-uncert", type=float, default=0.04, help="relative sound-speed uncertainty (0.01 if temp known)")
    phys(s)
    s.set_defaults(fn=cmd_fit)

    s = sub.add_parser("adsb-ingest", help="ADS-B traces/CSV -> tracks.npz (classified, altitude-corrected)")
    s.add_argument("--src", nargs="+", required=True)
    s.add_argument("--format", choices=["readsb", "opensky", "generic"], default="readsb")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--bbox")
    s.add_argument("--alt-source", choices=["auto", "geom", "baro"], default="auto")
    s.add_argument("--qnh", type=float, default=1013.25, help="local altimeter setting hPa (METAR) for baro altitudes")
    s.add_argument("--geoid-n", type=float, default=0.0, help="geoid undulation N (m); MSL = HAE - N (CONUS ~ -20..-35)")
    s.add_argument("--max-gap", type=float, default=45.0)
    s.add_argument("--types", help="icao_aircraft_types.js (tar1090-db) path")
    s.add_argument("--offline", action="store_true")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_adsb_ingest)

    s = sub.add_parser("dem-fetch", help="download Copernicus GLO-30 DEM tiles for a bbox")
    s.add_argument("--bbox", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_dem_fetch)

    s = sub.add_parser("search", help="candidate funnel: event corridor -> offset -> silence")
    s.add_argument("--tracks", required=True)
    s.add_argument("--events", nargs="+", required=True, help="event JSON(s) from `fit`, in recording order")
    s.add_argument("--bbox", required=True)
    s.add_argument("--grid", type=float, default=200.0)
    s.add_argument("--dt", type=float, default=1.0)
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--rec-start", help="known UTC of recording t=0 (single-event mode needs this)")
    s.add_argument("--rec-start-tol", type=float, default=60.0)
    s.add_argument("--rec-duration", type=float, help="recording length s (enables silence filter)")
    s.add_argument("--quiet", help="explicit quiet spans 'a-b,c-d' (recording s)")
    s.add_argument("--d-quiet", help="silence thresholds 'prop=3000,rotor=2500,unknown=3000,jet=2500' (m)")
    s.add_argument("--guard", type=float, default=25.0)
    s.add_argument("--offset-tol", type=float)
    s.add_argument("--event-classes", default="prop,rotor,unknown")
    s.add_argument("--distinct-aircraft", action="store_true", help="forbid the same aircraft for both events")
    s.add_argument("--out", required=True, help="candidates .npy")
    phys(s)
    demargs(s)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("score", help="trajectory-exact Doppler modelling + locked scoring + heat map")
    s.add_argument("--tracks", required=True)
    s.add_argument("--spec", required=True)
    s.add_argument("--events", nargs="+", required=True)
    s.add_argument("--cands", required=True)
    s.add_argument("--dt", type=float, default=1.0)
    s.add_argument("--harmonics", default="1,2", help="multiples of the traced line to score")
    s.add_argument("--lock", choices=["joint", "e1"], default="joint")
    s.add_argument("--trace-mode", action="store_true", help="score against traced points (very faint curves)")
    s.add_argument("--top-pairs", type=int, default=40)
    s.add_argument("--sub-m", type=float, default=100.0)
    s.add_argument("--half-window", type=float, default=60.0)
    s.add_argument("--delta-max", type=float, default=4.0)
    s.add_argument("--f-rel", type=float, default=0.006)
    s.add_argument("--heat-radius", type=float, default=1500.0)
    s.add_argument("--heat-step", type=float, default=30.0)
    s.add_argument("--region-frac", type=float, default=0.9)
    s.add_argument("--ambiguity-frac", type=float, default=0.9, help="flag alternatives scoring >= frac*best")
    s.add_argument("--rec-start", help="known recording start (UTC ISO/unix) - restricts the time shift")
    s.add_argument("--rec-start-tol", type=float, default=5.0, help="clock tolerance (s) for --rec-start")
    s.add_argument("--no-heatmap", action="store_true")
    s.add_argument("--report", type=int, default=15)
    s.add_argument("--out", required=True)
    phys(s)
    demargs(s)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("thunder", help="thunder-envelope geolocation from lightning strike CSV (time,lat,lon)")
    s.add_argument("--audio", required=True)
    s.add_argument("--strikes", required=True)
    s.add_argument("--bbox", required=True)
    s.add_argument("--grid", type=float, default=2000.0)
    s.add_argument("--start", required=True, help="earliest possible recording start (UTC)")
    s.add_argument("--end", required=True, help="latest possible recording start (UTC)")
    s.add_argument("--rate", type=float, default=2.0)
    s.add_argument("--source-h", type=float, default=2000.0)
    s.add_argument("--p", type=float, default=1.0, help="amplitude decay exponent (1 amplitude, 2 energy)")
    s.add_argument("--report", type=int, default=25)
    s.add_argument("--out", required=True)
    phys(s)
    demargs(s)
    s.set_defaults(fn=cmd_thunder)

    s = sub.add_parser("audit", help="privacy audit of your own recording before publishing")
    s.add_argument("audio")
    s.add_argument("--thr", type=float, default=0.8)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("simulate", help="write a synthetic scenario (traces + recording + truth)")
    s.add_argument("--out", required=True)
    s.add_argument("--lat", type=float, default=39.10)
    s.add_argument("--lon", type=float, default=-94.60)
    s.add_argument("--ground", type=float, default=250.0)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("selftest", help="blind end-to-end test on a synthetic scenario")
    s.add_argument("--keep", help="work directory to keep outputs")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_selftest)

    a = p.parse_args(argv)
    r = a.fn(a)
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
