"""Acoustic Doppler physics for an aircraft passing a fixed microphone.

Conventions
-----------
tau      emission time (when the aircraft produced the sound)
t        reception time (when the microphone heard it):  t = tau + R(tau)/c
R        observer-aircraft slant range;  Rdot = dR/dtau (>0 receding)
f0       rest (source) frequency of the tone being tracked (any harmonic)

Exact relation for a stationary observer in still air, evaluated at the
EMISSION time (this is the point the video stresses - using the radial
velocity at reception time produces a wrong curve, not just a delayed one):

    f_obs(t) = f0 / (1 + Rdot(tau)/c)        ( = f0 * dtau/dt )

With a uniform wind w the ground-frame propagation speed along the ray is
c_eff = c + w.n (n = unit vector source->observer); replace c by c_eff.

Linear constant-velocity pass (speed v, CPA slant distance d) closed form,
with tc = reception time of the sound emitted AT the CPA (where f = f0):
    T = t - (tc - d/c)
    s = [c^2 T - sqrt(c^2 v^2 T^2 + d^2 (c^2 - v^2))] / (c^2 - v^2)   (tau - tau_cpa)
    R = c (T - s),  Rdot = v^2 s / R,  f = f0 / (1 + Rdot/c)
Useful identities:
    f(-inf) = f0/(1 - v/c),  f(+inf) = f0/(1 + v/c)
    v  = c (f_hi - f_lo)/(f_hi + f_lo),   f0 = 2 f_hi f_lo/(f_hi + f_lo)
    df/dt at tc = -f0 v^2 / (c d)   ->  d = f0 v^2 / (c |slope|)
v sets the size of the frequency drop, d sets how fast the drop happens.
"""
import numpy as np
from scipy.optimize import least_squares


def sound_speed(temp_c=15.0):
    """Speed of sound in dry air (m/s); humidity adds < 0.5 %."""
    return 20.0468 * np.sqrt(np.asarray(temp_c, dtype=float) + 273.15)


def doppler_linear(t, f0, v, d, tc, c=343.0):
    t = np.asarray(t, dtype=float)
    tau_c = tc - d / c
    T = t - tau_c
    c2, v2 = c * c, v * v
    s = (c2 * T - np.sqrt(c2 * v2 * T * T + d * d * (c2 - v2))) / (c2 - v2)
    R = c * (T - s)
    rdot = v2 * s / np.maximum(R, 1e-9)
    return f0 / (1.0 + rdot / c)


def doppler_split(t, f0, v_in, v_out, d, tc, c=343.0):
    """Linear pass with a different speed before (v_in) and after (v_out) the CPA.

    Cheap stand-in for speed/climb changes during the pass (departures,
    arrivals, power changes): each side of the curve gets its own asymptote
    while f0, d and tc stay shared."""
    t = np.asarray(t, dtype=float)
    return np.where(t < tc, doppler_linear(t, f0, v_in, d, tc, c), doppler_linear(t, f0, v_out, d, tc, c))


def cpa_slope(f0, v, d, c=343.0):
    return -f0 * v * v / (c * d)


def doppler_from_track(tau, pos, vel, obs, f0=1.0, c=343.0, wind=None):
    """Exact emission-time model along an arbitrary sampled trajectory.

    tau: (N,) emission times; pos/vel: (N,3) aircraft position/velocity in any
    Cartesian frame (ECEF or ENU); obs: (3,) observer position (same frame).
    wind: optional (3,) wind vector in the same frame (m/s).
    Returns reception times t (N,), observed frequency (N,), slant range (N,).
    """
    D = pos - obs
    R = np.linalg.norm(D, axis=1)
    rdot = np.einsum("ij,ij->i", D, vel) / np.maximum(R, 1e-9)
    if wind is None:
        ceff = np.full(R.shape, c)
    else:
        n = -D / np.maximum(R, 1e-9)[:, None]
        ceff = c + n @ np.asarray(wind, dtype=float)
    t = tau + R / ceff
    return t, f0 / (1.0 + rdot / ceff), R


# ----------------------------------------------------------------------------
# Fitting a traced curve
# ----------------------------------------------------------------------------

def _initial_guess(t, f, c):
    o = np.argsort(t)
    t, f = t[o], f[o]
    q = max(3, len(t) // 6)
    f_hi, f_lo = np.median(f[:q]), np.median(f[-q:])
    if f_hi <= f_lo:
        raise ValueError("traced curve does not decrease in frequency - not a single fly-by "
                         "(check the box/anchors, or the event may be two overlapping aircraft)")
    f0 = 2 * f_hi * f_lo / (f_hi + f_lo)
    v = c * (f_hi - f_lo) / (f_hi + f_lo)
    below = np.where(f <= f0)[0]
    tc = t[below[0]] if below.size else t[len(t) // 2]
    sel = np.abs(f - f0) < 0.25 * (f_hi - f_lo)
    slope = None
    if sel.sum() >= 4:
        slope = np.polyfit(t[sel], f[sel], 1)[0]
    if slope is None or slope >= 0:
        slope = -(f_hi - f_lo) / max(t[-1] - t[0], 1.0) * 3
    d = f0 * v * v / (c * abs(slope))
    return np.array([f0, v, np.clip(d, 50, 30000), tc])


def fit_linear(t, f, c=343.0, weights=None, corr_factor=1.0, robust=True):
    """Fit the closed-form linear-pass model to traced (t, f) points.

    corr_factor: sqrt(window/hop) of the spectrogram - adjacent frames are not
    independent, so formal sigmas are inflated by this factor.
    Returns dict with params, 1-sigma errors, residual diagnostics.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    w = np.ones_like(t) if weights is None else np.asarray(weights, float)
    x0 = _initial_guess(t, f, c)

    def resid(x, tt, ff, ww):
        return (doppler_linear(tt, x[0], x[1], x[2], x[3], c) - ff) * ww

    lo = [0.5 * f.min(), 1.0, 5.0, t.min() - 600]
    hi = [2.0 * f.max(), 0.85 * c, 60000.0, t.max() + 600]
    best = None
    for vs in (1.0, 1.25, 1.6, 0.8):
        for ds in (1.0, 0.5, 2.0):
            xi = x0.copy()
            xi[1] = np.clip(xi[1] * vs, lo[1] + 1e-3, hi[1] - 1e-3)
            xi[2] = np.clip(xi[2] * ds, lo[2] + 1e-3, hi[2] - 1e-3)
            xi[0] = np.clip(xi[0], lo[0] + 1e-6, hi[0] - 1e-6)
            xi[3] = np.clip(xi[3], lo[3] + 1e-3, hi[3] - 1e-3)
            try:
                r = least_squares(resid, xi, args=(t, f, w), bounds=(lo, hi),
                                  loss="soft_l1" if robust else "linear",
                                  f_scale=max(0.2, 0.01 * np.median(f) * 0.05),
                                  x_scale=[max(0.01 * x0[0], 0.05), 5.0, 100.0, 1.0])
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        raise RuntimeError("fit failed")
    x = best.x
    res = doppler_linear(t, *x, c) - f
    mad = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
    inl = np.abs(res) < 4 * mad
    # refit on inliers with linear loss for honest covariance
    r2 = least_squares(resid, x, args=(t[inl], f[inl], w[inl]), bounds=(lo, hi), loss="linear",
                       x_scale=[max(0.01 * x0[0], 0.05), 5.0, 100.0, 1.0])
    x = r2.x
    res_in = doppler_linear(t[inl], *x, c) - f[inl]
    n, p = inl.sum(), 4
    s2 = float(np.sum((res_in * w[inl]) ** 2) / max(n - p, 1))
    J = r2.jac
    try:
        cov = np.linalg.inv(J.T @ J) * s2 * corr_factor**2
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.nan)
        sig = np.full(4, np.nan)
    res_all = doppler_linear(t, *x, c) - f
    left = t < x[3]
    rms_l = float(np.sqrt(np.mean(res_all[left & inl] ** 2))) if (left & inl).any() else np.nan
    rms_r = float(np.sqrt(np.mean(res_all[~left & inl] ** 2))) if (~left & inl).any() else np.nan
    return {
        "f0": x[0], "v": x[1], "d": x[2], "t_cpa_rx": x[3],
        "sigma": {"f0": sig[0], "v": sig[1], "d": sig[2], "t_cpa_rx": sig[3]},
        "cov": cov,
        "rms_hz": float(np.sqrt(np.mean(res_in**2))),
        "rms_left_hz": rms_l, "rms_right_hz": rms_r,
        "n_points": int(len(t)), "n_inliers": int(n),
        "f_hi_asym": x[0] / (1 - x[1] / c), "f_lo_asym": x[0] / (1 + x[1] / c),
        "cpa_slope_hz_s": cpa_slope(x[0], x[1], x[2], c),
        "c": c,
    }


def fit_split(t, f, lin, c=343.0, corr_factor=1.0):
    """Refit with separate approach/recede speeds, starting from the linear fit.

    Returns a fit-like dict (v = mean of the two, v_in/v_out, d, t_cpa_rx, ...).
    A large |v_out - v_in| means the aircraft changed speed or climb rate
    across the pass, which biases the constant-speed v (and d ~ v^2) - the
    search windows should then cover both fits."""
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    res_l = doppler_linear(t, lin["f0"], lin["v"], lin["d"], lin["t_cpa_rx"], c) - f
    mad = 1.4826 * np.median(np.abs(res_l - np.median(res_l))) + 1e-9
    inl = np.abs(res_l) < 5 * mad
    tt, ff = t[inl], f[inl]

    def resid(x):
        return doppler_split(tt, *x, c=c) - ff

    lo = [0.5 * f.min(), 1.0, 1.0, 5.0, t.min() - 600]
    hi = [2.0 * f.max(), 0.85 * c, 0.85 * c, 60000.0, t.max() + 600]
    x0 = np.clip([lin["f0"], lin["v"], lin["v"], lin["d"], lin["t_cpa_rx"]], np.add(lo, 1e-6), np.subtract(hi, 1e-6))
    try:
        r = least_squares(resid, x0, bounds=(lo, hi), x_scale=[max(0.01 * lin["f0"], 0.05), 5.0, 5.0, 100.0, 1.0])
    except Exception:
        return None
    x = r.x
    n, p = inl.sum(), 5
    rr = resid(x)
    s2 = float(np.sum(rr**2) / max(n - p, 1))
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * s2 * corr_factor**2
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sig = np.full(5, np.nan)
    v_lo, v_hi = sorted((x[1], x[2]))
    return {"f0": x[0], "v": 0.5 * (x[1] + x[2]), "v_in": x[1], "v_out": x[2], "d": x[3], "t_cpa_rx": x[4],
            "v_span": (v_lo, v_hi),
            "sigma": {"f0": sig[0], "v": float(np.hypot(sig[1], sig[2]) / 2 + 0.5 * (v_hi - v_lo)),
                      "v_in": sig[1], "v_out": sig[2], "d": sig[3], "t_cpa_rx": sig[4]},
            "rms_hz": float(np.sqrt(np.mean(rr**2))), "n_inliers": int(n), "c": c}


def search_constraints(fit, v_rel_margin=0.20, d_rel_margin=0.15, t_margin_s=4.0,
                       c_rel_uncert=0.04, nsig=3.0, k_rel_margin=0.08, alt=None):
    """Turn a fit into deliberately generous search windows.

    Key point: v and d are individually ill-determined when the curve's tails
    (asymptotes) are faint or the aircraft changed speed/climb during the pass,
    but their combination k = v^2/d = c |slope_at_CPA| / f0 is pinned by the
    steep middle of the curve.  So the windows are: a WIDE speed window, a
    tight k window, and the d window implied by both.  In the search each
    ADS-B candidate's own ground speed then predicts its d = gs^2 / k.

    Sound-speed error scales v, d and k by the same factor (c_rel_uncert).
    alt: optional second fit (e.g. fit_split); the windows become the union.
    """
    if alt is not None:
        a = search_constraints(fit, v_rel_margin, d_rel_margin, t_margin_s, c_rel_uncert, nsig, k_rel_margin)
        b = search_constraints(alt, v_rel_margin, d_rel_margin, t_margin_s, c_rel_uncert, nsig, k_rel_margin)
        lo_t = min(a["t_rec"] - a["t_tol"], b["t_rec"] - b["t_tol"])
        hi_t = max(a["t_rec"] + a["t_tol"], b["t_rec"] + b["t_tol"])
        for key in ("v_min", "d_min", "k_min"):
            a[key] = min(a[key], b[key])
        for key in ("v_max", "d_max", "k_max"):
            a[key] = max(a[key], b[key])
        a.update(t_rec=0.5 * (lo_t + hi_t), t_tol=0.5 * (hi_t - lo_t))
        return a
    v, d, s = fit["v"], fit["d"], fit["sigma"]
    dv = max(nsig * np.nan_to_num(s["v"], nan=v * 0.1), v * v_rel_margin) + v * c_rel_uncert
    k = v * v / d
    cov = fit.get("cov")
    sk = np.nan
    if cov is not None and np.all(np.isfinite(cov[1:3, 1:3])):
        J = np.array([2 * v / d, -v * v / (d * d)])
        sk = float(np.sqrt(max(J @ cov[1:3, 1:3] @ J, 0.0)))
    dk = max(nsig * np.nan_to_num(sk, nan=0.1 * k), k * k_rel_margin) + k * c_rel_uncert
    v_min, v_max = max(v - dv, 5.0), v + dv
    k_min, k_max = max(k - dk, 1e-3), k + dk
    dd = max(nsig * np.nan_to_num(s["d"], nan=d * 0.3), d * d_rel_margin) + d * c_rel_uncert
    dt = max(nsig * np.nan_to_num(s["t_cpa_rx"], nan=t_margin_s), t_margin_s)
    return {
        "v_min": v_min, "v_max": v_max,
        "k_min": k_min, "k_max": k_max,
        "d_min": max(min(d - dd, v_min**2 / k_max), 30.0), "d_max": max(d + dd, v_max**2 / k_min),
        "t_rec": fit["t_cpa_rx"], "t_tol": dt,
        "f0": fit["f0"], "f0_rel_tol": max(0.004, 3 * np.nan_to_num(s["f0"], nan=0) / fit["f0"]),
    }


def plausibility(fit, v_max=150.0, d_max=15000.0, rel_sigma_max=0.5):
    """Reasons a fit is NOT a usable low-level aircraft pass (empty list = ok).

    A tracer that wandered through noise or hopped harmonics typically yields
    v near c or d at the bound; feeding that to the search makes a corridor
    tens of km wide (slow and meaningless).  Props/helicopters audible on the
    ground fly < ~110 m/s and pass within a few km.
    """
    why = []
    if not np.isfinite(fit["v"]) or fit["v"] > v_max:
        why.append(f"v={fit['v']:.0f} m/s > {v_max:.0f} (trace probably followed noise or switched harmonics)")
    if not np.isfinite(fit["d"]) or fit["d"] > d_max:
        why.append(f"d={fit['d']:.0f} m > {d_max:.0f} (no usable curvature)")
    for k in ("v", "d"):
        s = fit["sigma"][k]
        if np.isfinite(s) and s > rel_sigma_max * fit[k]:
            why.append(f"sigma_{k}/{k}={s / fit[k]:.2f} (poorly constrained)")
    if fit.get("n_inliers", 99) < 20:
        why.append(f"only {fit.get('n_inliers')} inlier points")
    return why
