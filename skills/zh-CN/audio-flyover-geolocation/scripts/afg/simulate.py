"""Synthetic ground-truth scenario: realistic ADS-B traces (readsb format) and a
physically exact recording (emission-time Doppler, 1/R spreading, voice,
hum, wind, pink noise) at a hidden observer. Used by `selftest` to prove the
pipeline end to end, and handy for rehearsing a case before touching real data.
"""
import gzip
import json
import os

import numpy as np
from scipy.io import wavfile

from .common import FPM, FT, KT, dump_json, iso, parse_time
from .geo import geodetic_to_ecef, offset_latlon

PROPS = [  # type, desc, blades, rpm  ->  BPF = blades*rpm/60
    ("C172", "L1P", 2, 2400), ("SR22", "L1P", 3, 2600), ("PC12", "L1T", 4, 1700),
    ("BE20", "L2T", 4, 2000), ("C208", "L1T", 3, 1900), ("DH8D", "L2T", 6, 1020), ("P28A", "L1P", 2, 2500),
]
JETS = ["B738", "CRJ9", "E75L", "A320", "C56X"]


def _fly(rng, t0, dur, x0, y0, hdg, v, z, maneuvers=(), dt=0.1):
    n = int(dur / dt) + 1
    t = t0 + np.arange(n) * dt
    turn = np.zeros(n)
    acc = np.zeros(n)
    climb = np.zeros(n)
    for m in maneuvers:
        s = (t >= t0 + m["t"]) & (t < t0 + m["t"] + m["dur"])
        turn[s] += m.get("turn", 0.0)
        acc[s] += m.get("acc", 0.0)
        climb[s] += m.get("climb", 0.0)
    h = np.radians(hdg + np.cumsum(turn) * dt)
    spd = np.clip(v + np.cumsum(acc) * dt, 20, 300)
    vx, vy = spd * np.sin(h), spd * np.cos(h)
    x = x0 + np.cumsum(vx) * dt - vx[0] * dt
    y = y0 + np.cumsum(vy) * dt - vy[0] * dt
    zz = z + np.cumsum(climb) * dt
    return {"t": t, "x": x, "y": y, "z": zz, "vx": vx, "vy": vy, "vz": climb, "spd": spd, "hdg": np.degrees(h) % 360}


def _random_maneuvers(rng, dur, k=3):
    out = []
    for _ in range(rng.integers(0, k + 1)):
        out.append({"t": rng.uniform(0, dur), "dur": rng.uniform(10, 60), "turn": rng.uniform(-2.5, 2.5),
                    "acc": rng.uniform(-0.3, 0.3), "climb": rng.uniform(-3, 3)})
    return out


def _ecef_track(tr, lat0, lon0):
    la, lo = offset_latlon(lat0, lon0, tr["x"], tr["y"])
    return la, lo, geodetic_to_ecef(la, lo, tr["z"])


def simulate(outdir, lat0=39.10, lon0=-94.60, ground=250.0, seed=7, n_prop=45, n_jet=25, area_km=70.0,
             day="2026-09-08T14:00:00Z", span_h=4.0, c=343.0, fs=4000, t1_rec=240.0, gap=920.0, tail=300.0,
             tone_amp=0.005, noise_rms=0.03):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    t_day = parse_time(day)
    half = area_km * 500.0
    obs_x, obs_y = rng.uniform(-6000, 6000, 2)
    obs_lat, obs_lon = offset_latlon(lat0, lon0, obs_x, obs_y)
    obs_ecef = geodetic_to_ecef(obs_lat, obs_lon, ground + 2.0)
    T1 = t_day + rng.uniform(0.35, 0.55) * span_h * 3600
    flights = []

    def target(Tc, r_h, agl, v, man, typ):
        hdg = rng.uniform(0, 360)
        hr = np.radians(hdg)
        side = rng.choice([-1, 1])
        px, py = obs_x + side * r_h * np.cos(hr), obs_y - side * r_h * np.sin(hr)
        pre = rng.uniform(600, 1200)
        dur = pre + rng.uniform(600, 1200)
        x0, y0 = px - v * pre * np.sin(hr), py - v * pre * np.cos(hr)
        man = [dict(m, t=m["t"] + pre) for m in man]
        tr = _fly(rng, Tc - pre, dur, x0, y0, hdg, v, ground + agl, man)
        return {"tr": tr, "type": typ, "cls": "prop"}

    p1 = PROPS[rng.integers(len(PROPS))]
    p2 = PROPS[rng.integers(len(PROPS))]
    flights.append(target(T1, rng.uniform(600, 1300), rng.uniform(500, 900), rng.uniform(75, 100),
                          [{"t": 25, "dur": 35, "turn": 1.5}], p1))
    flights.append(target(T1 + gap, rng.uniform(700, 1500), rng.uniform(500, 900), rng.uniform(55, 75),
                          [{"t": -60, "dur": 50, "acc": 0.25}, {"t": 10, "dur": 40, "climb": 2.0}], p2))
    rec_start = T1 + 2.0 - t1_rec  # approx (exact reception offsets come from physics)
    rec_dur = t1_rec + gap + tail
    win = (rec_start - 90, rec_start + rec_dur + 30)

    def min_dist_in_window(tr):
        s = (tr["t"] >= win[0]) & (tr["t"] <= win[1])
        if not s.any():
            return np.inf
        _, _, e = _ecef_track({k: v[s] for k, v in tr.items()}, lat0, lon0)
        return float(np.linalg.norm(e - obs_ecef, axis=1).min())

    def rand_flight(kind):
        for _ in range(200):
            t0 = t_day + rng.uniform(-0.5, span_h) * 3600
            if kind == "prop":
                typ = PROPS[rng.integers(len(PROPS))]
                v, z, dur = rng.uniform(45, 110), ground + rng.uniform(350, 3000), rng.uniform(900, 3600)
                man = _random_maneuvers(rng, dur)
            else:
                typ = JETS[rng.integers(len(JETS))]
                v, z, dur = rng.uniform(150, 240), ground + rng.uniform(3000, 11000), rng.uniform(600, 1500)
                man = []
            x0, y0 = rng.uniform(-half, half, 2)
            tr = _fly(rng, t0, dur, x0, y0, rng.uniform(0, 360), v, z, man)
            if min_dist_in_window(tr) > (5000 if kind == "prop" else 4500):
                return {"tr": tr, "type": typ, "cls": kind}
        raise RuntimeError("could not place distractor")

    # a few distractors deliberately pass close to the observer OUTSIDE the recording
    for _ in range(4):
        Tc = T1 + rng.choice([-1, 1]) * rng.uniform(2500, 6000)
        f = target(Tc, rng.uniform(500, 2000), rng.uniform(400, 1200), rng.uniform(55, 100), [], PROPS[rng.integers(len(PROPS))])
        if min_dist_in_window(f["tr"]) > 5000:
            flights.append(f)
    flights += [rand_flight("prop") for _ in range(n_prop)]
    flights += [rand_flight("jet") for _ in range(n_jet)]

    # ---------------- ADS-B traces ------------------------------------------------
    tdir = os.path.join(outdir, "traces")
    truth_f = []
    for i, f in enumerate(flights):
        tr = f["tr"]
        hexid = f"{0xA00000 + i * 7919:06x}"
        f["hex"] = hexid
        typ = f["type"][0] if f["cls"] == "prop" else f["type"]
        ts = [tr["t"][0]]
        while ts[-1] < tr["t"][-1]:
            step = rng.choice([1, 1, 2, 2, 3, 4, 5])
            if rng.random() < 0.02:
                step = rng.uniform(8, 25)
            ts.append(ts[-1] + step + rng.uniform(-0.3, 0.3))
        ts = np.array(ts[:-1])
        g = lambda k: np.interp(ts, tr["t"], tr[k])  # noqa: E731
        x, y = g("x") + rng.normal(0, 2.5, ts.size), g("y") + rng.normal(0, 2.5, ts.size)
        la, lo = offset_latlon(lat0, lon0, x, y)
        z, spd, vz = g("z"), g("spd"), g("vz")
        hdg = np.interp(ts, tr["t"], np.degrees(np.unwrap(np.radians(tr["hdg"])))) % 360
        entries = []
        for k in range(ts.size):
            vr_fpm = int(round(vz[k] / FPM / 64) * 64)
            entries.append([round(ts[k] - ts[0], 2), round(la[k], 6), round(lo[k], 6),
                            int(round((z[k] + rng.normal(0, 3)) / FT / 25) * 25),   # baro (QNH = std here)
                            round(spd[k] / KT + rng.normal(0, 0.25), 1),
                            round((hdg[k] + rng.normal(0, 0.15)) % 360, 1), 2 if k == 0 else 0,
                            vr_fpm, None, "adsb_icao",
                            int(round((z[k] + rng.normal(0, 4)) / FT / 25) * 25),   # geometric (N = 0 here)
                            vr_fpm, None, None])
        obj = {"icao": hexid, "r": f"N{100 + i}SIM", "t": typ, "dbFlags": 0, "desc": "SIMULATED",
               "timestamp": round(ts[0], 3), "trace": entries}
        d = os.path.join(tdir, hexid[-2:])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"trace_full_{hexid}.json"), "wb") as fh:
            fh.write(gzip.compress(json.dumps(obj).encode()))
        truth_f.append({"hex": hexid, "type": typ, "cls": f["cls"]})

    # ---------------- audio ---------------------------------------------------
    n = int(rec_dur * fs)
    ta = rec_start + np.arange(n) / fs
    audio = np.zeros(n)
    ev_truth = []
    for i, f in enumerate(flights):
        tr = f["tr"]
        s = (tr["t"] >= win[0] - 60) & (tr["t"] <= win[1])
        if s.sum() < 10:
            continue
        sub = {k: v[s] for k, v in tr.items()}
        _, _, e = _ecef_track(sub, lat0, lon0)
        R = np.linalg.norm(e - obs_ecef, axis=1)
        if R.min() > 12000:
            continue
        trx = sub["t"] + R / c
        m = (ta >= trx[0]) & (ta <= trx[-1])
        if not m.any():
            continue
        tau = np.interp(ta[m], trx, sub["t"])
        Rt = np.interp(ta[m], trx, R)
        if f["cls"] == "prop":
            typ, _, blades, rpm = f["type"]
            bpf = blades * rpm / 60.0 * (1 + rng.uniform(-0.01, 0.01))
            drift = np.cumsum(rng.normal(0, 2e-5, sub["t"].size))
            phase = 2 * np.pi * bpf * np.cumsum(np.r_[0, np.diff(sub["t"])] * (1 + drift))
            ph = np.interp(tau, sub["t"], phase)
            sig = np.zeros(m.sum())
            for k, a in enumerate([1.0, 0.7, 0.5, 0.35, 0.25, 0.18], start=1):
                sig += a * np.sin(k * ph + rng.uniform(0, 2 * np.pi))
            audio[m] += tone_amp * 1000.0 / Rt * sig
            if i < 2:
                j = int(np.argmin(R))
                ev_truth.append({"event": i + 1, "hex": f["hex"], "type": typ, "bpf_hz": bpf, "t_cpa_emit": iso(sub["t"][j]),
                                 "d_cpa_m": float(R[j]), "t_cpa_rx_unix": float(sub["t"][j] + R[j] / c),
                                 "t_cpa_rec_s": float(sub["t"][j] + R[j] / c - rec_start), "speed_ms": float(sub["spd"][j])})
        else:
            noise = rng.normal(0, 1, m.sum())
            from scipy import signal as _s
            noise = _s.sosfilt(_s.butter(2, 600, fs=fs, output="sos"), noise)
            audio[m] += 0.02 * 3000.0 / Rt * noise
    # ambient: pink noise + hum + voice bursts + wind
    w = rng.normal(0, 1, n)
    W = np.fft.rfft(w)
    fr = np.fft.rfftfreq(n, 1 / fs)
    W[1:] /= np.sqrt(fr[1:] / 50.0)
    W[0] = 0
    pink = np.fft.irfft(W, n)
    audio += noise_rms * pink / pink.std()
    tt = np.arange(n) / fs
    enf = 60 + 0.02 * np.sin(2 * np.pi * tt / 97.0)
    audio += 0.002 * np.sin(2 * np.pi * np.cumsum(enf) / fs) + 0.001 * np.sin(2 * np.pi * 3 * np.cumsum(enf) / fs)
    t_v = 0.0
    while True:
        t_v += rng.exponential(35.0)
        if t_v > rec_dur - 6:
            break
        L = rng.uniform(1.5, 5.0)
        a, b = int(t_v * fs), int((t_v + L) * fs)
        tv = np.arange(b - a) / fs
        pitch = rng.uniform(100, 220) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * tv))
        ph = 2 * np.pi * np.cumsum(pitch) / fs
        v = sum((1.0 / k) * np.sin(k * ph) * (1 + np.cos(2 * np.pi * k / 7)) for k in range(1, 13))
        env = np.sin(np.pi * tv / L) ** 0.5 * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.5 * tv)))
        audio[a:b] += rng.uniform(0.05, 0.12) * v * env
    peak = np.max(np.abs(audio))
    audio = 0.9 * audio / peak
    wav = os.path.join(outdir, "recording.wav")
    wavfile.write(wav, fs, (audio * 32767).astype(np.int16))
    truth = {"observer": {"lat": float(obs_lat), "lon": float(obs_lon), "ground_elev_m": ground},
             "rec_start_utc": iso(rec_start), "rec_start_unix": rec_start, "rec_duration_s": rec_dur,
             "events": ev_truth, "region_center": [lat0, lon0], "area_km": area_km,
             "bbox": [lat0 - area_km / 222.0, lat0 + area_km / 222.0,
                      lon0 - area_km / (222.0 * np.cos(np.radians(lat0))), lon0 + area_km / (222.0 * np.cos(np.radians(lat0)))],
             "day_window": [iso(t_day - 3600), iso(t_day + (span_h + 1) * 3600)], "c": c, "fs": fs,
             "n_flights": len(flights)}
    dump_json(truth, os.path.join(outdir, "truth.json"))
    return truth, wav, tdir
