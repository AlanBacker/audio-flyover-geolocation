"""Regression tests for the search funnel, the fit uncertainties and the scoring checks.

Run from the repository root:  python -m unittest discover -s tests -v
Everything is synthetic; nothing is read from disk or the network.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "skills", "en", "audio-flyover-geolocation", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))

import afg_cli  # noqa: E402
from afg import audio, physics  # noqa: E402
from afg.geo import geodetic_to_ecef, offset_latlon  # noqa: E402
from afg.scoring import CurveModel, effective_delta_max, rank_consistent, trace_check  # noqa: E402
from afg.search import EV_DTYPE, _crossing_stats, _pair  # noqa: E402

LAT0, LON0, GROUND = 39.10, -94.60, 250.0


def brute_pairs(ev1, ev2, dT, tol):
    return sorted((i, j) for i in range(ev1.size) for j in range(ev2.size)
                  if ev1["node"][i] == ev2["node"][j] and abs(ev2["t_rx"][j] - ev1["t_rx"][i] - dT) <= tol)


def straight_flight(heading_deg=60.0, v=70.0, alt=700.0, t0=1.79e9, dur=600.0, cpa_east=0.0, cpa_north=800.0):
    """A minimal stand-in for tracks.Resampled: one straight, level flight at 1 s.

    Its closest approach to (LAT0, LON0) is near t0 + dur/2, with the CPA point
    (cpa_east, cpa_north) metres away."""
    h = np.radians(heading_deg)
    s = (np.arange(int(dur) + 1) - dur / 2) * v
    e, n = cpa_east + s * np.sin(h), cpa_north + s * np.cos(h)
    la, lo = offset_latlon(LAT0, LON0, e, n)
    ecef = geodetic_to_ecef(la, lo, np.full(la.size, GROUND + alt))
    t = t0 + np.arange(la.size, dtype=float)
    vel = np.gradient(ecef, t, axis=0)
    first, last = 0, la.size - 1

    def flight_slice(fid, ta=None, tb=None):
        a, b = first, last + 1
        if ta is not None:
            a = max(a, int(np.ceil(ta - t[0])))
        if tb is not None:
            b = min(b, int(np.floor(tb - t[0])) + 1)
        return a, max(a, b)

    return types.SimpleNamespace(t=t, ecef=ecef, vel=vel, dt=1.0, flight_slice=flight_slice, lat=la, lon=lo,
                                 ve=np.full(la.size, v * np.sin(h)), vn=np.full(la.size, v * np.cos(h)))


class Pairing(unittest.TestCase):
    def events(self, nodes, seed=0, n=300, dT=900.0):
        rng = np.random.default_rng(seed)
        ev1, ev2 = np.zeros(n, EV_DTYPE), np.zeros(n, EV_DTYPE)
        ev1["node"], ev2["node"] = nodes[rng.integers(0, nodes.size, n)], nodes[rng.integers(0, nodes.size, n)]
        ev1["t_rx"] = 1.79e9 + rng.uniform(0, 7200, n)
        ev2["t_rx"] = np.r_[ev1["t_rx"][: n // 2] + dT + rng.normal(0, 4, n // 2), 1.79e9 + rng.uniform(0, 9000, n - n // 2)]
        ev2["node"][: n // 2] = ev1["node"][: n // 2]
        return ev1, ev2

    def test_matches_brute_force(self):
        for nodes in (np.arange(40), (np.int64(60000) << 32) | np.arange(40)):   # compact and global Grid ids
            for dT in (900.0, -350.0):
                with self.subTest(global_ids=bool(nodes[0] > 1e9), dT=dT):
                    ev1, ev2 = self.events(nodes, dT=dT)
                    got = sorted(map(tuple, _pair(ev1, ev2, dT, 6.0).tolist()))
                    self.assertEqual(got, brute_pairs(ev1, ev2, dT, 6.0))
                    self.assertTrue(got)

    def test_global_ids_keep_the_time_term(self):
        """The defect report's case: an hour off must give no pairs, also with ~1e14 node ids."""
        nodes = (np.int64(60000) << 32) | np.arange(2000)
        ev1, ev2 = self.events(nodes)
        ev2["t_rx"][:] = ev1["t_rx"] + 900.0
        ev2["node"][:] = ev1["node"]
        self.assertEqual(len(_pair(ev1, ev2, 900.0, 8.0)), ev1.size)
        self.assertEqual(len(_pair(ev1, ev2, 900.0 + 3600.0, 8.0)), 0)

    def test_crossing_angle(self):
        R = types.SimpleNamespace(ve=np.array([0.0, 70.0, -10.0, 10.0]), vn=np.array([70.0, 0.0, 70.0, 70.0]))
        a = _crossing_stats(R, np.array([0, 2]), np.array([1, 3]))
        self.assertAlmostEqual(a["max"], 90.0, places=6)
        self.assertAlmostEqual(a["min"], 2 * np.degrees(np.arctan2(10, 70)), places=6)
        self.assertEqual(a["pairs_below_25"], 1)


class FitUncertainty(unittest.TestCase):
    def test_corr_time(self):
        rng = np.random.default_rng(1)
        self.assertLess(physics.corr_time(rng.normal(size=4000)), 1.3)
        x = np.zeros(20000)
        e = rng.normal(size=x.size)
        for i in range(1, x.size):
            x[i] = 0.8 * x[i - 1] + e[i]
        self.assertAlmostEqual(physics.corr_time(x), 9.0, delta=1.5)          # (1 + 0.8) / (1 - 0.8)

    def curve(self, noise):
        t = np.arange(0, 160, 0.2)
        return t, physics.doppler_linear(t, 110.0, 60.0, 900.0, 80.0) + noise

    def test_white_residuals_keep_the_overlap_factor(self):
        t, f = self.curve(np.random.default_rng(2).normal(0, 0.2, 800))
        fit = physics.fit_linear(t, f, corr_factor=np.sqrt(5.0))
        self.assertAlmostEqual(fit["sigma_inflation"], np.sqrt(5.0), places=6)

    def test_correlated_residuals_widen_sigma(self):
        rng = np.random.default_rng(3)
        x = np.zeros(800)
        for i in range(1, x.size):
            x[i] = 0.97 * x[i - 1] + rng.normal(0, 0.05)
        t, f = self.curve(x)
        fit = physics.fit_linear(t, f, corr_factor=np.sqrt(5.0))
        self.assertGreater(fit["resid_corr_time"], 10.0)
        self.assertAlmostEqual(fit["sigma_inflation"], np.sqrt(fit["resid_corr_time"]), places=6)
        self.assertLess(fit["n_eff"], fit["n_inliers"] / 10.0)
        self.assertGreater(fit["corr_v_d"], 0.5)           # v and d trade off along k = v^2/d

    def test_constraints_report_the_fitted_distance(self):
        t, f = self.curve(np.random.default_rng(4).normal(0, 0.2, 800))
        fit = physics.fit_linear(t, f, corr_factor=np.sqrt(5.0))
        c = physics.search_constraints(fit)
        self.assertLess(c["d_fit_min"], fit["d"])
        self.assertGreater(c["d_fit_max"], fit["d"])
        self.assertLessEqual(c["d_min"], c["d_fit_min"])
        self.assertGreaterEqual(c["d_max"], c["d_fit_max"])


class Scoring(unittest.TestCase):
    def test_time_shift_covers_the_funnel_tolerance(self):
        self.assertEqual(effective_delta_max([{"t_tol": 2.0}], 4.0), 4.0)
        self.assertEqual(effective_delta_max([{"t_tol": 4.16}], 4.0), 5.0)
        self.assertEqual(effective_delta_max([{"t_tol": 20.4}], 4.0), 21.0)
        self.assertEqual(effective_delta_max([{"t_tol": 400.0}], 4.0), 30.0)

    def hypothesis(self, R, east, north, rec_start):
        la, lo = offset_latlon(LAT0, LON0, east, north)
        return {"lat": float(la), "lon": float(lo), "elev": GROUND, "fid1": 0, "fid2": -1, "t1_cpa": R.t[300],
                "t2_cpa": None, "rec_start_utc": rec_start, "scales": [1.0], "total": 1.0}

    def traced(self, R, rec_start, f0=110.0, noise=0.15):
        """What the microphone at (LAT0, LON0) hears, as traced points in recording time."""
        cu = CurveModel(R).curve(0, geodetic_to_ecef(LAT0, LON0, GROUND + 2.0), R.t[300], 200.0)
        t = np.arange(20.0, 260.0, 0.2)
        f = np.interp(t, cu["t_rx"] - rec_start, f0 * cu["ratio"]) + np.random.default_rng(5).normal(0, noise, t.size)
        return t, f

    def test_trace_check_passes_the_true_site_and_fails_wrong_ones(self):
        R = straight_flight()
        rs = R.t[300] - 140.0
        t, f = self.traced(R, rs)
        fit = physics.fit_linear(t, f)
        ev = [{"f0": fit["f0"], "t_rec": fit["t_cpa_rx"], "t_tol": 4.0}]
        good = self.hypothesis(R, 0.0, 0.0, rs)
        good["scales"] = [110.0 / fit["f0"]]
        far = dict(self.hypothesis(R, 0.0, 2500.0, rs), scales=good["scales"])     # 2.5 km off: other curve shape
        late = dict(good, rec_start_utc=rs - 300.0)                                  # curve 300 s away in time
        trace_check(R, [good, far, late], ev, [(t, f)], [fit])
        self.assertTrue(good["trace_check"]["consistent"], good["trace_check"])
        self.assertLess(good["trace_check"]["events"][0]["ratio"], 2.0)
        self.assertFalse(far["trace_check"]["consistent"], far["trace_check"])
        self.assertFalse(late["trace_check"]["consistent"])
        self.assertIsNone(late["trace_check"]["events"][0]["rms_hz"])
        self.assertEqual(good["scale_at_bound"], [abs(good["scales"][0] - 1) >= 0.006 * (1 - 1e-6)])

    def test_no_trace_means_no_verdict(self):
        R = straight_flight()
        h = self.hypothesis(R, 0.0, 0.0, R.t[300] - 140.0)
        trace_check(R, [h], [{"f0": 110.0, "t_rec": 140.0}], [None], [None])
        self.assertIsNone(h["trace_check"]["consistent"])

    def test_rank_consistent(self):
        hs = [{"total": 9.0, "trace_check": {"consistent": False}}, {"total": 8.0, "trace_check": {"consistent": True}},
              {"total": 7.0, "trace_check": {"consistent": None}}]
        self.assertEqual([h["total"] for h in rank_consistent(hs, 2)], [8.0, 7.0, 9.0])
        self.assertEqual([h["total"] for h in rank_consistent(hs, 1)], [9.0, 8.0, 7.0])


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = afg_cli.main(argv)
    return code, buf.getvalue()


class Cli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        fs = 4000
        t = np.arange(120 * fs) / fs
        ph = 2 * np.pi * np.cumsum(physics.doppler_linear(t, 100.0, 60.0, 300.0, 60.0)) / fs
        audible = np.clip(np.minimum(t - 38.0, 82.0 - t) / 2.0, 0, 1)        # heard from 40 to 80 s only
        x = 0.3 * audible * np.sin(ph) + 0.02 * np.random.default_rng(7).normal(size=t.size)
        spec = audio.compute_spectrogram(x, fs, 1.0, 0.2, 20, 600, 120)
        self.spec = os.path.join(self.dir.name, "rec.spec.npz")
        audio.save_spec(spec, self.spec, "synthetic")

    def test_detect_stdout_names_the_events_file(self):
        out = os.path.join(self.dir.name, "boxes.json")
        code, stdout = run(["detect", self.spec, "--out", out])
        self.assertEqual(code, 0)
        res = json.loads(stdout)
        self.assertEqual(res["events_file"], out)
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), res["events"])          # the file holds the event list itself

    def test_trace_without_hint_when_it_covers_the_box(self):
        out = os.path.join(self.dir.name, "e1.csv")
        code, stdout = run(["trace", self.spec, "--t0", "40", "--t1", "80", "--fmin", "60", "--fmax", "125",
                            "--out", out])
        full = json.loads(stdout)
        self.assertGreater(full["box_coverage"], 0.9)
        self.assertIsNone(full["hint"])

    def test_trace_hint_when_only_a_piece_is_traced(self):
        """The report's case: 57 points over 11 s of a 96 s box (the curve rose before it fell)."""
        t = np.arange(44.3, 55.5, 0.2)
        piece = (t, 100.0 - 0.5 * (t - 44.3), np.full(t.size, 8.0))
        out = os.path.join(self.dir.name, "e1.csv")
        with mock.patch.object(afg_cli.audio, "trace_curve", return_value=piece):
            code, stdout = run(["trace", self.spec, "--t0", "5", "--t1", "101", "--fmin", "60", "--fmax", "125",
                                "--out", out])
            part = json.loads(stdout)
            self.assertLess(part["box_coverage"], 0.2)
            self.assertIn("--no-monotone", part["hint"])
            code, stdout = run(["trace", self.spec, "--t0", "5", "--t1", "101", "--fmin", "60", "--fmax", "125",
                                "--no-monotone", "--out", out])
            self.assertNotIn("--no-monotone", json.loads(stdout)["hint"])


if __name__ == "__main__":
    unittest.main()
