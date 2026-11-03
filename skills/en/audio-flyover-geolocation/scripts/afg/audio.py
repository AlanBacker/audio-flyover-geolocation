"""Audio loading, high-resolution narrow-band spectrograms, whitening,
harmonic summation, ridge tracking and event detection.

Aircraft propeller tones sit at roughly 50-400 Hz (blade-passing frequency and
harmonics), overlapping the human voice. They are usually inaudible but show
up as smooth descending curves once the spectrogram is (a) zoomed to the band,
(b) given enough frequency resolution, and (c) whitened against the
stationary background.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import map_coordinates, median_filter, uniform_filter1d

from .common import log


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def _to_float(x):
    if x.dtype.kind == "f":
        return x.astype(np.float64)
    if x.dtype == np.uint8:
        return (x.astype(np.float64) - 128) / 128.0
    return x.astype(np.float64) / float(np.iinfo(x.dtype).max)


def load_audio(path, fs=None, channel="mix"):
    """Load any audio/video file. Non-WAV inputs are decoded with ffmpeg
    (lossless PCM, no filtering). channel: 'mix' | integer index.
    Returns (x, fs, n_channels)."""
    path = str(path)
    sr, x = None, None
    if path.lower().endswith(".wav"):
        try:
            sr, x = wavfile.read(path)
        except Exception:
            sr = None
    if sr is None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required to decode " + path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-c:a", "pcm_s24le"]
        if fs:
            cmd += ["-ar", str(int(fs))]
        cmd += [tmp.name]
        subprocess.run(cmd, check=True)
        sr, x = wavfile.read(tmp.name)
        os.unlink(tmp.name)
    x = _to_float(np.asarray(x))
    nch = 1 if x.ndim == 1 else x.shape[1]
    if x.ndim > 1:
        x = x.mean(axis=1) if channel == "mix" else x[:, int(channel)]
    if fs and int(fs) != int(sr):
        g = np.gcd(int(fs), int(sr))
        x = signal.resample_poly(x, int(fs) // g, int(sr) // g)
        sr = int(fs)
    return x, int(sr), nch


# ----------------------------------------------------------------------------
# Spectrogram
# ----------------------------------------------------------------------------

def stft_power(x, fs, win_s=1.0, hop_s=0.2, fmax=600.0, pad=4, chunk=1024):
    """Chunked Hann STFT keeping only bins <= fmax (memory-safe for long files).
    Returns f (Hz), t (frame-centre seconds), P (power, float32, F x T)."""
    n = int(round(win_s * fs))
    hop = max(1, int(round(hop_s * fs)))
    nfft = 1 << int(np.ceil(np.log2(n * pad)))
    w = signal.windows.hann(n, sym=False)
    nb = min(nfft // 2 + 1, int(np.floor(fmax * nfft / fs)) + 2)
    if len(x) < n:
        raise ValueError("audio shorter than one analysis window")
    nframes = 1 + (len(x) - n) // hop
    view = np.lib.stride_tricks.sliding_window_view(x, n)
    P = np.empty((nb, nframes), np.float32)
    norm = 1.0 / (fs * np.sum(w**2))
    for s in range(0, nframes, chunk):
        idx = np.arange(s, min(nframes, s + chunk))
        F = np.fft.rfft(view[idx * hop] * w, nfft, axis=1)[:, :nb]
        P[:, idx] = ((F.real**2 + F.imag**2) * norm).T
    t = (np.arange(nframes) * hop + n / 2) / fs
    f = np.arange(nb) * fs / nfft
    return f, t, P


def whiten(P, t, bg_window_s=120.0, flatten=True):
    """dB spectrogram minus a slowly varying per-bin background (block medians,
    interpolated) -> removes hum, HVAC lines, mic colouration. Optionally also
    subtract the per-frame median across frequency (broadband level changes).
    Result is 'dB above local background' (0 = background)."""
    Pdb = 10 * np.log10(P + 1e-30)
    dur = t[-1] - t[0] if len(t) > 1 else 0
    hop = (t[1] - t[0]) if len(t) > 1 else 1.0
    if bg_window_s is None or dur <= bg_window_s:
        B = np.median(Pdb, axis=1, keepdims=True)
        W = Pdb - B
    else:
        blk = max(4, int(round(bg_window_s / hop)))
        centers, meds = [], []
        for s in range(0, Pdb.shape[1], blk // 2):
            e = min(Pdb.shape[1], s + blk)
            meds.append(np.median(Pdb[:, s:e], axis=1))
            centers.append(0.5 * (s + e - 1))
            if e == Pdb.shape[1]:
                break
        meds = np.stack(meds, axis=1)
        centers = np.asarray(centers)
        fr = np.arange(Pdb.shape[1])
        W = np.empty_like(Pdb)
        for i in range(Pdb.shape[0]):
            W[i] = Pdb[i] - np.interp(fr, centers, meds[i])
    if flatten:
        W -= np.median(W, axis=0, keepdims=True)
    return W.astype(np.float32)


def gate_transients(W, occ_db=6.0, k_mad=6.0, dilate=2):
    """Zero frames dominated by broadband/harmonic-rich transients (speech,
    door slams, wind gusts). A fly-by tone lights up only a handful of bins per
    frame; speech lights up a large fraction. Returns (W_gated, mask)."""
    occ = (W > occ_db).mean(axis=0)
    med = np.median(occ)
    mad = 1.4826 * np.median(np.abs(occ - med)) + 1e-6
    bad = occ > max(med + k_mad * mad, 2.5 * med + 0.01)
    if dilate:
        bad = np.convolve(bad.astype(float), np.ones(2 * dilate + 1), "same") > 0
    Wg = W.copy()
    Wg[:, bad] = 0.0
    return Wg, bad


def compute_spectrogram(x, fs, win_s=1.0, hop_s=0.2, fmin=20.0, fmax=600.0, bg_window_s=120.0, gate=True):
    f, t, P = stft_power(x, fs, win_s, hop_s, fmax)
    keep = f >= max(0.0, fmin - 2.0)
    f, P = f[keep], P[keep]
    W = whiten(P, t, bg_window_s)
    mask = np.zeros(t.size, bool)
    if gate:
        W, mask = gate_transients(W)
    return {"f": f, "t": t, "W": W, "Pdb": (10 * np.log10(P + 1e-30)).astype(np.float32), "gated": mask,
            "win_s": win_s, "hop_s": hop_s, "fs": fs, "duration": len(x) / fs}


def save_spec(spec, path, source=""):
    np.savez_compressed(path, f=spec["f"], t=spec["t"], W=spec["W"], Pdb=spec["Pdb"],
                        gated=spec.get("gated", np.zeros(spec["t"].size, bool)),
                        win_s=spec["win_s"], hop_s=spec["hop_s"], fs=spec["fs"],
                        duration=spec["duration"], source=str(source))


def load_spec(path):
    z = np.load(path, allow_pickle=False)
    return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}


def sample_spec(spec, t, f, clip_db=(0.0, 30.0)):
    """Bilinear sample of the whitened spectrogram at (t, f) arrays (any shape);
    outside the spectrogram -> 0. Values clipped to clip_db (brightness)."""
    tt, ff, W = spec["t"], spec["f"], spec["W"]
    ti = (np.asarray(t) - tt[0]) / (tt[1] - tt[0])
    fi = (np.asarray(f) - ff[0]) / (ff[1] - ff[0])
    shp = ti.shape
    v = map_coordinates(W, [fi.ravel(), ti.ravel()], order=1, mode="constant", cval=0.0)
    if clip_db is not None:
        v = np.clip(v, clip_db[0], clip_db[1])
    return v.reshape(shp)


def harmonic_sum(spec, fmin, fmax, n_harm=4, t0=None, t1=None, df=None):
    """H(t, f) = mean_k max(W(t, k f), 0) for fundamental candidates f in [fmin, fmax].
    Stacking harmonics raises SNR by ~10log10(K) dB for propeller tones."""
    t = spec["t"]
    sel = np.ones(t.size, bool)
    if t0 is not None:
        sel &= t >= t0
    if t1 is not None:
        sel &= t <= t1
    tt = t[sel]
    step = df or (spec["f"][1] - spec["f"][0])
    fg = np.arange(fmin, fmax + step / 2, step)
    acc = np.zeros((fg.size, tt.size), np.float32)
    T, F = np.meshgrid(tt, fg)
    nk = 0
    for k in range(1, n_harm + 1):
        if k * fmax > spec["f"][-1]:
            break
        acc += sample_spec(spec, T, k * F, clip_db=(0.0, 40.0))
        nk += 1
    return fg, tt, acc / max(nk, 1)


# ----------------------------------------------------------------------------
# Ridge tracking (Viterbi) and detection
# ----------------------------------------------------------------------------

def track_ridge(S, max_jump=3, penalty=0.5, monotone=False, up_allow=1):
    """Globally optimal ridge through score matrix S (F x T): maximise
    sum(S) - penalty*|jump| with |jump| <= max_jump bins per frame.
    monotone=True forbids rises > up_allow bins (a single straight fly-by
    always decreases in frequency)."""
    F, T = S.shape
    lo_j = -max_jump
    hi_j = up_allow if monotone else max_jump
    score = S[:, 0].astype(np.float64).copy()
    back = np.zeros((F, T), np.int16)
    js = np.arange(lo_j, hi_j + 1)
    for ti in range(1, T):
        cand = np.full((js.size, F), -np.inf)
        for a, j in enumerate(js):
            # moving from bin (b - j) at ti-1 to bin b at ti
            if j >= 0:
                cand[a, j:] = score[: F - j] - penalty * abs(j)
            else:
                cand[a, :F + j] = score[-j:] - penalty * abs(j)
        a_best = np.argmax(cand, axis=0)
        score = cand[a_best, np.arange(F)] + S[:, ti]
        back[:, ti] = js[a_best]
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(score))
    for ti in range(T - 1, 0, -1):
        path[ti - 1] = path[ti] - back[path[ti], ti]
    return path


def trace_curve(spec, t0, t1, fmin, fmax, n_harm=3, max_jump_hz=None, penalty=0.3,
                monotone=True, anchors=None, corridor_hz=3.0, min_snr=3.0, refine=True, model_guided=True,
                c=343.0):
    """Trace one Doppler curve inside a (time, frequency) box.

    Pass 1: optimal monotone ridge, keep its dominant contiguous run.
    Pass 2 (model_guided): fit the physical fly-by shape to that core and
    re-trace inside +-corridor_hz of the fitted curve over the whole box - this
    rejects noise wander and bridges gaps (speech gating, dropouts).
    anchors: optional (t, f) points roughly on the curve (read off a PNG) to
    restrict pass 1 to a corridor - the 'trace it by hand' approach for very
    faint curves. Returns (t, f, snr) of accepted points on the boxed line.
    """
    t, f, s, tt, fg, H = _trace_pass(spec, t0, t1, fmin, fmax, n_harm, max_jump_hz, penalty, monotone,
                                     anchors, corridor_hz, min_snr, refine)
    if not model_guided or t.size < 25:
        return t, f, s
    from .physics import doppler_linear, fit_linear
    try:
        fit = fit_linear(t, f, c=c)
    except Exception:
        return t, f, s
    if not (5 < fit["v"] < 0.6 * c and 30 < fit["d"] < 20000):
        return t, f, s
    model = doppler_linear(tt, fit["f0"], fit["v"], fit["d"], fit["t_cpa_rx"], c)
    # step 2 Hz anchors along the model -> corridor re-trace
    anc = list(zip(tt[::max(1, len(tt) // 200)], model[::max(1, len(tt) // 200)]))
    t2, f2, s2, *_ = _trace_pass(spec, t0, t1, fmin, fmax, n_harm, max_jump_hz, penalty, monotone,
                                 anc, max(corridor_hz, 0.02 * fit["f0"]), min_snr, refine, keep_runs="all")
    return (t2, f2, s2) if t2.size >= t.size * 0.8 else (t, f, s)


def _trace_pass(spec, t0, t1, fmin, fmax, n_harm, max_jump_hz, penalty, monotone, anchors, corridor_hz,
                min_snr, refine, keep_runs="dominant"):
    df = spec["f"][1] - spec["f"][0]
    hop = spec["t"][1] - spec["t"][0]
    fg, tt, H = harmonic_sum(spec, fmin, fmax, n_harm=n_harm, t0=t0, t1=t1)
    gated = np.asarray(spec.get("gated", np.zeros(spec["t"].size, bool)), bool)
    gated = np.interp(tt, spec["t"], gated.astype(float)) > 0.5
    hb = H[:, ~gated] if (~gated).any() else H
    med = np.median(hb)
    mad = 1.4826 * np.median(np.abs(hb - med)) + 1e-6
    if anchors:
        a = np.asarray(sorted(anchors), float)
        line = np.interp(tt, a[:, 0], a[:, 1])
        mask = np.abs(fg[:, None] - line[None, :]) <= corridor_hz
        H = np.where(mask, H, -5.0)
    if max_jump_hz is None:
        max_jump_hz = max(10.0 * hop, 2 * df)  # allow up to ~10 Hz/s sweep (close, fast passes)
    mj = max(1, int(np.ceil(max_jump_hz / df)))
    path = track_ridge(H, max_jump=mj, penalty=penalty, monotone=monotone)
    f_tr = fg[path]
    snr = H[path, np.arange(tt.size)]
    if refine:  # parabolic sub-bin refinement
        p = np.clip(path, 1, fg.size - 2)
        ym, y0, yp = H[p - 1, np.arange(tt.size)], H[p, np.arange(tt.size)], H[p + 1, np.arange(tt.size)]
        den = ym - 2 * y0 + yp
        safe = np.abs(den) > 1e-6
        delta = np.where(safe, 0.5 * (ym - yp) / np.where(safe, den, 1.0), 0.0)
        f_tr = fg[p] + np.clip(delta, -0.5, 0.5) * df
    # Significance against a NULL ridge: the optimal path through the same box
    # with frames shuffled in time (same max-selection bias, no real continuity).
    # Compare 3-s smoothed ridge levels; in pass 1 keep the dominant contiguous
    # run (gated frames bridge runs but are never output).
    k = max(3, int(round(3.0 / hop)))
    vs = uniform_filter1d(np.where(gated, np.nan_to_num(snr), snr), k)
    Hn = H[:, np.random.default_rng(0).permutation(H.shape[1])]
    pn = track_ridge(Hn, max_jump=mj, penalty=penalty, monotone=monotone)
    vn = uniform_filter1d(Hn[pn, np.arange(Hn.shape[1])], k)
    thr = max(min_snr, float(np.percentile(vn, 99.0)))
    ok = (vs >= thr) & ~gated
    if keep_runs == "dominant":
        sm = uniform_filter1d((ok | gated).astype(float), size=max(3, int(round(4 / hop))))
        run = sm > 0.4
        lab = np.cumsum(np.r_[run[0], (run[1:] & ~run[:-1])]) * run
        if lab.max() > 0:
            w = np.bincount(lab, weights=np.where(ok, snr - med, 0.0))
            w[0] = 0
            ok &= lab == int(np.argmax(w))
    else:  # corridor pass: the corridor itself is the constraint; accept weaker points
        ok = (vs >= max(min_snr, float(np.percentile(vn, 90.0)))) & ~gated
    return tt[ok], f_tr[ok], snr[ok], tt, fg, H


def detect_events(spec, fmin=40.0, fmax=400.0, n_harm=3, thr_db=0.8, min_dur_s=15.0,
                  merge_gap_s=10.0, win_s=60.0, step_s=10.0, k_mad=3.5):
    """Find time spans containing a sustained, smoothly varying narrow-band
    tone (candidate fly-bys). Slides a window over the harmonic-summed
    spectrogram, finds the optimal monotone ridge in each window and flags
    windows whose mean ridge level stands out from the recording's own
    distribution (noise ridges also pick maxima, so the comparison is relative).
    Returns list of dicts: t0, t1, fmin, fmax (padded box), peak_snr_db (ridge
    excess over median, dB), score_z."""
    df = spec["f"][1] - spec["f"][0]
    fg, tt, H = harmonic_sum(spec, fmin, fmax, n_harm=n_harm, df=max(df, 0.5))
    hop = tt[1] - tt[0]
    dfg = fg[1] - fg[0]
    gated = np.asarray(spec.get("gated", np.zeros(spec["t"].size, bool)), bool)
    gated = np.interp(tt, spec["t"], gated.astype(float)) > 0.5
    nw = max(5, int(round(win_s / hop)))
    ns = max(1, int(round(step_s / hop)))
    mj = max(1, int(np.ceil(10.0 * hop / dfg)))
    spans, scores, paths = [], [], []
    for s in range(0, max(1, H.shape[1] - nw + 1), ns):
        sub = H[:, s:s + nw]
        p = track_ridge(sub, max_jump=mj, penalty=0.3, monotone=True)
        ok = ~gated[s:s + sub.shape[1]]          # gated (speech) frames carry no information
        vals = sub[p, np.arange(sub.shape[1])]
        spans.append((s, s + sub.shape[1]))
        scores.append(float(vals[ok].mean()) if ok.sum() > 0.3 * ok.size else np.nan)
        paths.append(fg[p][ok] if ok.any() else fg[p])
    scores = np.asarray(scores)
    scores = np.where(np.isfinite(scores), scores, np.nanmedian(scores))
    base = np.median(scores)
    mad = 1.4826 * np.median(np.abs(scores - base)) + 1e-6
    hit = scores > base + max(thr_db, k_mad * mad)
    groups = []
    for i in np.flatnonzero(hit):
        if groups and (spans[i][0] - spans[groups[-1][-1]][1]) * hop <= merge_gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    for g in groups:
        a, b = spans[g[0]][0], spans[g[-1]][1] - 1
        if (b - a) * hop < min_dur_s:
            continue
        # one continuous ridge over the whole span -> follows a single harmonic line
        sub = H[:, a:b + 1]
        p = track_ridge(sub, max_jump=mj, penalty=0.3, monotone=True)
        v = sub[p, np.arange(sub.shape[1])]
        strong = (v > np.median(H) + 3 * (1.4826 * np.median(np.abs(H - np.median(H))) + 1e-6)) & ~gated[a:b + 1]
        fs_ = fg[p][strong] if strong.sum() >= 10 else fg[p]
        lo, hi = np.percentile(fs_, 2), np.percentile(fs_, 98)
        pad_f = max(4.0, 0.06 * hi)
        out.append({"t0": float(max(tt[0], tt[a] - 15.0)), "t1": float(min(tt[-1], tt[b] + 15.0)),
                    "fmin": float(max(fmin, lo - pad_f)), "fmax": float(min(fmax, hi + pad_f)),
                    "peak_snr_db": float(scores[g].max() - base), "score_z": float((scores[g].max() - base) / mad),
                    "core_t0": float(tt[a]), "core_t1": float(tt[b])})
    return out


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def plot_spec(spec, path, t0=None, t1=None, fmin=None, fmax=None, overlays=None, title="",
              vmax=None, points=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, f, W = spec["t"], spec["f"], spec["W"]
    ts = (t >= (t0 if t0 is not None else -np.inf)) & (t <= (t1 if t1 is not None else np.inf))
    fs_ = (f >= (fmin if fmin is not None else -np.inf)) & (f <= (fmax if fmax is not None else np.inf))
    img = W[np.ix_(fs_, ts)]
    fig, ax = plt.subplots(figsize=(14, 6), dpi=110)
    vm = vmax if vmax is not None else max(6.0, float(np.percentile(img, 99.7)))
    ax.imshow(img, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vm,
              extent=[t[ts][0], t[ts][-1], f[fs_][0], f[fs_][-1]], interpolation="nearest")
    for ov in overlays or []:
        ax.plot(ov["t"], ov["f"], ov.get("style", "-"), color=ov.get("color", "cyan"),
                lw=ov.get("lw", 1.0), alpha=ov.get("alpha", 0.8), label=ov.get("label"))
    if points is not None:
        ax.plot(points[0], points[1], ".", color="lime", ms=2)
    if any(ov.get("label") for ov in overlays or []):
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("recording time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(title or "whitened spectrogram (dB above background)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
