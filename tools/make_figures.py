#!/usr/bin/env python3
"""Draw the README figures (light and dark) from a finished pipeline run.

  bash examples/synthetic_demo.sh work/demo 23
  python tools/make_figures.py work/demo            # -> docs/images/*-{light,dark}.png

The work directory must hold the files the CLI writes: rec.spec.npz, e1/e2 .csv and
.json, tracks.npz, result.json and result.heatmap.csv. truth.json (synthetic runs
only) adds the true recording location to the maps.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "skills", "en", "audio-flyover-geolocation", "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from afg import physics  # noqa: E402
from afg.audio import load_spec  # noqa: E402
from afg.geo import latlon_to_local_en  # noqa: E402

# chart tokens: categorical slots 1-3 (validated all-pairs in both modes), ink, chrome
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a"],
                  seq=["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926", "#199e70"],
                 seq=["#1a1a19", "#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]),
}


def style(th):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "figure.facecolor": th["surface"], "axes.facecolor": th["surface"], "savefig.facecolor": th["surface"],
        "axes.edgecolor": th["axis"], "axes.linewidth": 1.0, "axes.labelcolor": th["ink2"],
        "xtick.color": th["muted"], "ytick.color": th["muted"], "xtick.labelcolor": th["ink2"],
        "ytick.labelcolor": th["ink2"], "text.color": th["ink"], "axes.grid": False,
        "legend.frameon": False, "legend.labelcolor": th["ink2"], "lines.solid_capstyle": "round",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def hgrid(ax, th, axis="y"):
    ax.grid(True, axis=axis, color=th["grid"], linewidth=1.0)
    ax.set_axisbelow(True)


def save(fig, out, name, mode):
    path = os.path.join(out, f"{name}-{mode}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


# ----------------------------------------------------------------------------- figures

def fig_doppler(th, out, mode):
    """Theory: v sets how far the tone drops, d sets how fast."""
    t = np.linspace(-60, 60, 1201)
    f0, c = 100.0, 343.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    panels = [("Speed changes the size of the drop", "d = 1000 m", [(40, 1000), (60, 1000), (80, 1000)],
               lambda v, d: f"{v} m/s", True),
              ("Distance changes how fast it drops", "v = 70 m/s", [(70, 300), (70, 1000), (70, 3000)],
               lambda v, d: f"{d} m", False)]
    for ax, (title, fixed, cases, lab, end_labels) in zip(axes, panels):
        for k, (v, d) in enumerate(cases):
            f = physics.doppler_linear(t, f0, v, d, 0.0, c)
            ax.plot(t, f, color=th["series"][k], lw=2, label=lab(v, d))
            if end_labels:  # the right panel's lines converge, so its legend carries identity
                ax.text(61.5, f[-1], lab(v, d), color=th["ink2"], va="center", fontsize=9)
        ax.axhline(f0, color=th["axis"], lw=1)
        ax.axvline(0, color=th["axis"], lw=1)
        ax.text(-59, f0 + 0.6, "f0 = 100 Hz (tone at the source)", color=th["muted"], fontsize=8.5)
        ax.text(1, 79, "CPA", color=th["muted"], fontsize=8.5)
        ax.set_title(title, color=th["ink"])
        ax.set_xlabel("time relative to closest approach (s)")
        ax.set_xlim(-60, 72)
        hgrid(ax, th)
        ax.legend(loc="upper right", title=fixed, title_fontsize=9, fontsize=9, handlelength=1.6)
        ax.get_legend().get_title().set_color(th["ink2"])
    axes[0].set_ylabel("heard frequency (Hz)")
    axes[0].set_ylim(77, 133)
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "doppler-curves", mode)


def fig_spectrogram(th, work, out, mode):
    spec = load_spec(os.path.join(work, "rec.spec.npz"))
    ev = json.load(open(os.path.join(work, "e1.json"), encoding="utf-8"))
    fit = ev["fit"]
    with open(os.path.join(work, "e1.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    tr = {k: np.array([float(x[k]) for x in rows]) for k in ("t", "f")}
    truth = _truth(work)
    tc = fit["t_cpa_rx"]
    t0, t1 = tc - 82, tc + 82
    f_lo, f_hi = fit["f_lo_asym"], fit["f_hi_asym"]
    lo, hi = f_lo - (f_hi - f_lo) * 0.12, f_hi + (f_hi - f_lo) * 0.12
    ti = (spec["t"] >= t0) & (spec["t"] <= t1)
    fi = (spec["f"] >= lo) & (spec["f"] <= hi)
    W = spec["W"][np.ix_(fi, ti)].astype(float)
    W[:, spec["gated"][ti].astype(bool)] = np.nan
    gray = LinearSegmentedColormap.from_list("g", [th["surface"], th["muted"]])
    gray.set_bad(th["surface"])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    # quiet context: only the upper half of the noise distribution gets ink
    vmin, vmax = np.nanpercentile(W, [60, 99.7])
    ax.imshow(W, origin="lower", aspect="auto", cmap=gray, vmin=vmin, vmax=vmax, interpolation="nearest",
              extent=[spec["t"][ti][0], spec["t"][ti][-1], spec["f"][fi][0], spec["f"][fi][-1]])
    m = (tr["t"] >= t0) & (tr["t"] <= t1)
    ax.scatter(tr["t"][m], tr["f"][m], s=7, color=th["series"][1], linewidths=0, zorder=3)
    tt = np.linspace(t0, t1, 800)
    ax.plot(tt, physics.doppler_linear(tt, fit["f0"], fit["v"], fit["d"], tc, fit["c"]),
            color=th["series"][0], lw=2, zorder=4)
    for y, lab in ((f_hi, f"approach  {f_hi:.1f} Hz"), (f_lo, f"recede  {f_lo:.1f} Hz")):
        ax.axhline(y, color=th["ink2"], lw=1, zorder=2)
        ax.text(t1 - 1, y + 1.2, lab, ha="right", va="bottom", color=th["ink"], fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc=th["surface"], ec="none", alpha=0.85))
    ax.axvline(tc, color=th["ink2"], lw=1, zorder=2)
    ax.text(tc + 1, hi - 2, f"closest approach heard at {tc:.1f} s", va="top", color=th["ink"], fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc=th["surface"], ec="none", alpha=0.85))
    txt = f"fit:  v = {fit['v']:.1f} m/s   d = {fit['d']:.0f} m   f0 = {fit['f0']:.1f} Hz"
    if truth:
        e = truth["events"][0]
        txt += f"\ntruth:  v = {e['speed_ms']:.1f} m/s   d = {e['d_cpa_m']:.0f} m"
    ax.text(t0 + 3, lo + 0.22 * (hi - lo), txt, va="bottom", color=th["ink"], fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc=th["surface"], ec=th["axis"], lw=1))
    ax.set_xlim(t0, t1)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("recording time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Event 1: the traced ridge and the fitted Doppler curve", color=th["ink"])
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=5, color=th["series"][1], label="traced ridge"),
                       Line2D([], [], color=th["series"][0], lw=2, label="Doppler fit (emission-time model)"),
                       Line2D([], [], ls="", marker="s", ms=7, mfc=th["surface"], mec=th["axis"],
                              label="gap = frame gated (speech, knocks)")],
              loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, fontsize=9)
    save(fig, out, "spectrogram-fit", mode)


def fig_map(th, work, out, mode, half_km=12.0):
    z = np.load(os.path.join(work, "tracks.npz"), allow_pickle=True)
    r = json.load(open(os.path.join(work, "result.json"), encoding="utf-8"))
    b = r["best"]
    truth = _truth(work)
    lat0, lon0 = (truth["observer"]["lat"], truth["observer"]["lon"]) if truth else (b["lat"], b["lon"])
    E, N = latlon_to_local_en(z["lat"], z["lon"], lat0, lon0)
    E, N = E / 1e3, N / 1e3
    fid = z["fid"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for f in np.unique(fid):
        k = np.flatnonzero(fid == f)
        if f in (b["fid1"], b["fid2"]):
            continue
        ax.plot(E[k], N[k], color=th["muted"], lw=0.8, alpha=0.55, zorder=1)
    names = []
    for slot, (f, tcpa, d) in enumerate(((b["fid1"], b["t1_cpa"], b["d_cpa"][0]),
                                          (b["fid2"], b["t2_cpa"], b["d_cpa"][1]))):
        k = np.flatnonzero(fid == f)
        col = th["series"][slot]
        ax.plot(E[k], N[k], color=col, lw=2, zorder=3)
        ce, cn = np.interp(tcpa, z["t"][k], E[k]), np.interp(tcpa, z["t"][k], N[k])
        oe, on = latlon_to_local_en(np.array([b["lat"]]), np.array([b["lon"]]), lat0, lon0)
        ax.plot([ce, oe[0] / 1e3], [cn, on[0] / 1e3], color=col, lw=1, zorder=3)
        ax.plot(ce, cn, "o", ms=8, color=col, mec=th["surface"], mew=2, zorder=5)
        ax.annotate(f"CPA, {d:.0f} m slant", (ce, cn), xytext=(8, 8 if slot == 0 else -14),
                    textcoords="offset points", color=th["ink"], fontsize=9)
        names.append(f"event {slot + 1}: {z['f_type'][f]} {z['f_reg'][f]}")
    oe, on = latlon_to_local_en(np.array([b["lat"]]), np.array([b["lon"]]), lat0, lon0)
    if truth:
        ax.plot(0, 0, "o", ms=13, mfc="none", mec=th["ink"], mew=2, zorder=6)
    ax.plot(oe[0] / 1e3, on[0] / 1e3, "o", ms=6, color=th["ink"], zorder=7)
    ax.set_xlim(-half_km, half_km)
    ax.set_ylim(-half_km, half_km)
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    hgrid(ax, th, "both")
    n_fl = len(np.unique(fid))
    h = [Line2D([], [], color=th["muted"], lw=0.8, label=f"other ADS-B tracks ({n_fl} flights in the data)"),
         Line2D([], [], color=th["series"][0], lw=2, label=names[0]),
         Line2D([], [], color=th["series"][1], lw=2, label=names[1]),
         Line2D([], [], ls="", marker="o", ms=6, color=th["ink"], label="estimated recording location")]
    if truth:
        h.append(Line2D([], [], ls="", marker="o", ms=11, mfc="none", mec=th["ink"], mew=2,
                        label="true location (synthetic test)"))
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0, -0.09), ncol=1, fontsize=9)
    ax.set_title("Two fly-overs matched to two ADS-B tracks", color=th["ink"])
    save(fig, out, "search-map", mode)


def fig_heatmap(th, work, out, mode):
    rows = list(csv.DictReader(open(os.path.join(work, "result.heatmap.csv"), encoding="utf-8")))
    lat = np.array([float(x["lat"]) for x in rows])
    lon = np.array([float(x["lon"]) for x in rows])
    r = json.load(open(os.path.join(work, "result.json"), encoding="utf-8"))
    b = r["best"]
    truth = _truth(work)
    E, N = latlon_to_local_en(lat, lon, b["lat"], b["lon"])
    ue, un = np.unique(np.round(E, 1)), np.unique(np.round(N, 1))
    shape = (len(un), len(ue))
    order = np.lexsort((np.round(E, 1), np.round(N, 1)))
    grids = {k: np.array([float(x[k]) for x in rows])[order].reshape(shape) for k in ("event1", "event2", "score")}
    Eg, Ng = E[order].reshape(shape), N[order].reshape(shape)
    cmap = LinearSegmentedColormap.from_list("seq", th["seq"])
    zoom = 400.0
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.4))
    fig.subplots_adjust(left=0.1, right=0.86, top=0.95, bottom=0.13, wspace=0.3, hspace=0.28)
    panels = [("event1", "Event 1 alone: a strip", None), ("event2", "Event 2 alone: another strip", None),
              ("score", "Both, locked in time: a point", None), ("score", f"Zoom to ±{zoom:.0f} m", zoom)]
    for ax, (k, title, lim) in zip(axes.ravel(), panels):
        raw = grids[k]
        g = (raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw))
        im = ax.pcolormesh(Eg, Ng, g, cmap=cmap, vmin=0, vmax=1, shading="auto", rasterized=True)
        ax.set_title(title, color=th["ink"])
        ax.set_aspect("equal")
        if k == "score":
            ax.contour(Eg, Ng, raw, levels=[0.9 * np.nanmax(raw)], colors=[th["series"][1]], linewidths=1.5)
            if truth:
                te, tn = latlon_to_local_en(np.array([truth["observer"]["lat"]]), np.array([truth["observer"]["lon"]]),
                                            b["lat"], b["lon"])
                ax.plot(te, tn, "o", ms=13, mfc="none", mec=th["ink"], mew=2, zorder=6)
            ax.plot(0, 0, "o", ms=6, color=th["ink"], zorder=7)
        if lim:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        elif k == "score":
            ax.add_patch(plt.Rectangle((-zoom, -zoom), 2 * zoom, 2 * zoom, fill=False, ec=th["ink2"], lw=1))
    for ax in axes[1]:
        ax.set_xlabel("east of estimate (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("north of estimate (m)")
    cb = fig.colorbar(im, cax=fig.add_axes([0.89, 0.3, 0.018, 0.45]))
    cb.set_label("score (each panel scaled to its peak)", color=th["ink2"])
    cb.outline.set_visible(False)
    err = ""
    if truth:
        from afg.geo import haversine_m
        err = f"   error {haversine_m(b['lat'], b['lon'], truth['observer']['lat'], truth['observer']['lon']):.0f} m"
    h = [Line2D([], [], color=th["series"][1], lw=1.5,
                label=f"0.9-score region (radius {r['region_max_radius_m']:.0f} m){err}"),
         Line2D([], [], ls="", marker="o", ms=6, color=th["ink"], label="estimate")]
    if truth:
        h.append(Line2D([], [], ls="", marker="o", ms=11, mfc="none", mec=th["ink"], mew=2, label="true location"))
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.08, 0.0), ncol=3, fontsize=9)
    save(fig, out, "score-heatmap", mode)


def _truth(work):
    p = os.path.join(work, "truth.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("work", help="work directory of a finished pipeline run")
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "images"))
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for mode, th in THEMES.items():
        style(th)
        fig_doppler(th, a.out, mode)
        fig_spectrogram(th, a.work, a.out, mode)
        fig_map(th, a.work, a.out, mode)
        fig_heatmap(th, a.work, a.out, mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
