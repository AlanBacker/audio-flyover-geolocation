"""Thunder-envelope geolocation (the author's earlier, coarser method).

Observed: band-pass 20-250 Hz -> RMS envelope -> light smoothing.
Model at each candidate point: each strike contributes a pulse at
t_strike + slant/c (acoustic source ~2 km up for CG strikes) with amplitude
~ 1/r^p (p=1 for an RMS *amplitude* envelope, p=2 if you compare energy),
capped near the source, cut beyond audibility range (~20-25 km: thunder is
refracted upward and rarely heard further).
Match: cosine similarity between observed envelope and the model window,
evaluated at every time offset with FFT cross-correlation; the model is
normalised *within each window* (the clip knows nothing about strikes
outside it). Precision is limited by strike-location error (free networks
~1-5 km; national/commercial networks ~100-500 m) and by strike geometry
(strikes from several directions are needed to triangulate).
"""
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from .geo import geodetic_to_ecef


def observed_envelope(x, fs, band=(20.0, 250.0), rate=2.0, smooth_s=1.5):
    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    hop = int(round(fs / rate))
    n = len(y) // hop
    env = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1))
    env = gaussian_filter1d(env, smooth_s * rate)
    env = env - np.median(env)
    return np.clip(env, 0, None)


def model_envelope(node_ecef, strikes_ecef, strike_t, t0, n, rate, c=343.0, p=1.0, r_cap=1500.0,
                   r_max=25000.0, smooth_s=1.5, decay_s=2.5, weights=None):
    r = np.linalg.norm(strikes_ecef - node_ecef, axis=1)
    m = r <= r_max
    tarr = strike_t[m] + r[m] / c
    amp = 1.0 / np.maximum(r[m], r_cap) ** p
    if weights is not None:
        amp = amp * weights[m]
    idx = np.round((tarr - t0) * rate).astype(int)
    ok = (idx >= 0) & (idx < n)
    env = np.zeros(n)
    np.add.at(env, idx[ok], amp[ok])
    k = np.exp(-np.arange(int(decay_s * rate * 5)) / (decay_s * rate))
    env = signal.fftconvolve(env, k)[:n]
    return gaussian_filter1d(env, smooth_s * rate)


def sliding_cosine(model, obs):
    """cos(model[k:k+m], obs) for every k (FFT); returns array of len(model)-m+1."""
    m = len(obs)
    if len(model) < m:
        return np.zeros(0)
    num = signal.fftconvolve(model, obs[::-1], mode="valid")
    csum = np.r_[0, np.cumsum(model**2)]
    den = np.sqrt(csum[m:] - csum[:-m]) * np.linalg.norm(obs)
    return np.where(den > 1e-30, num / np.maximum(den, 1e-30), 0.0)


def locate(obs_env, rate, strikes, nodes_lat, nodes_lon, nodes_elev, t_start, t_end, c=343.0, source_h=2000.0,
           p=1.0, min_strikes=5, weights=None):
    """Grid scan. strikes: dict t (unix), lat, lon. Returns score, best offset per node."""
    s_ecef = geodetic_to_ecef(strikes["lat"], strikes["lon"], np.full(len(strikes["lat"]), source_h))
    n = int((t_end - t_start) * rate) + len(obs_env)
    score = np.zeros(len(nodes_lat))
    best_t = np.full(len(nodes_lat), np.nan)
    necef = geodetic_to_ecef(nodes_lat, nodes_lon, nodes_elev)
    for i in range(len(nodes_lat)):
        r = np.linalg.norm(s_ecef - necef[i], axis=1)
        if (r < 25000).sum() < min_strikes:
            continue
        me = model_envelope(necef[i], s_ecef, strikes["t"], t_start, n, rate, c, p, weights=weights)
        cs = sliding_cosine(me, obs_env)
        if cs.size:
            k = int(np.argmax(cs))
            score[i], best_t[i] = cs[k], t_start + k / rate
    return score, best_t
