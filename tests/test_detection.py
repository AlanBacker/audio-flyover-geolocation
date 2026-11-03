"""Regression tests for harmonic stacking, event detection and the privacy audit.

Run from the repository root:  python -m unittest discover -s tests -v
Everything is synthetic; nothing is read from disk or the network.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "skills", "en", "audio-flyover-geolocation", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))

import afg_cli  # noqa: E402
from afg import audio  # noqa: E402
from afg.physics import doppler_linear  # noqa: E402

FS = 4000


def ambient(dur, seed, voices=True):
    """Pink noise, mains hum and speech-like bursts, as in afg.simulate."""
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    W = np.fft.rfft(rng.normal(0, 1, n))
    fr = np.fft.rfftfreq(n, 1 / FS)
    W[1:] /= np.sqrt(fr[1:] / 50.0)
    W[0] = 0
    pink = np.fft.irfft(W, n)
    x = 0.03 * pink / pink.std()
    ph = 2 * np.pi * np.cumsum(60 + 0.02 * np.sin(2 * np.pi * np.arange(n) / FS / 97.0)) / FS
    x += 0.002 * np.sin(ph) + 0.001 * np.sin(3 * ph)
    t_v = 0.0
    while voices:
        t_v += rng.exponential(35.0)
        if t_v > dur - 6:
            break
        L = rng.uniform(1.5, 5.0)
        a, b = int(t_v * FS), int((t_v + L) * FS)
        tv = np.arange(b - a) / FS
        pitch = rng.uniform(100, 220) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * tv))
        p = 2 * np.pi * np.cumsum(pitch) / FS
        v = sum((1.0 / k) * np.sin(k * p) * (1 + np.cos(2 * np.pi * k / 7)) for k in range(1, 13))
        env = np.sin(np.pi * tv / L) ** 0.5 * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.5 * tv)))
        x[a:b] += rng.uniform(0.05, 0.12) * v * env
    return x


def flyby(dur, f0=100.0, v=55.0, d=150.0, tc=None, amp=0.005):
    """Straight pass with three harmonics and 1/R amplitude."""
    t = np.arange(int(dur * FS)) / FS
    tc = t[-1] / 2 if tc is None else tc
    ph = 2 * np.pi * np.cumsum(doppler_linear(t, f0, v, d, tc)) / FS
    R = np.sqrt(d * d + (v * (t - tc + d / 343.0)) ** 2)
    return amp * 1000.0 / R * sum(a * np.sin(k * ph) for k, a in enumerate((1.0, 0.7, 0.5), start=1))


def strong(dur, seed=1, **kw):
    """The defect report's construction: a clear pass over weak white noise."""
    return flyby(dur, **kw) + 0.002 * np.random.default_rng(seed).normal(size=int(dur * FS))


def spec_of(x):
    return audio.compute_spectrogram(x, FS, 1.0, 0.2, 20, 600, 120)


class HarmonicSum(unittest.TestCase):
    def setUp(self):
        f = np.arange(18.0, 600.01, 0.25)
        t = np.arange(0.0, 20.0, 0.2)
        W = np.zeros((f.size, t.size), np.float32)
        W[np.argmin(np.abs(f - 200.0))] = 30.0        # a bright line at 200 Hz only
        self.spec = {"f": f, "t": t, "W": W}

    def test_band_that_fits_stacks_every_harmonic(self):
        fg, _, H, K = audio.harmonic_sum(self.spec, 40, 200, n_harm=3, df=0.5, return_counts=True)
        self.assertEqual(K, 3)
        self.assertAlmostEqual(float(H[np.argmin(np.abs(fg - 100.0))].max()), 10.0, places=4)  # 1 of 3 bright

    def test_count_used_is_reported(self):
        self.assertEqual(audio.harmonic_sum(self.spec, 40, 400, n_harm=3, df=0.5, return_counts=True)[3], 1)
        self.assertEqual(audio.harmonics_used(self.spec, 300.0, 3), 2)
        _, info = audio.detect_events(spec_of(strong(20)), n_harm=3, return_info=True)
        self.assertEqual(info["harmonics"], dict(info["harmonics"], requested=3, used=1))
        self.assertIn("note", info["harmonics"])

    def test_band_above_the_top_keeps_the_fundamental(self):
        fg, _, H = audio.harmonic_sum(self.spec, 150, 700, n_harm=3, df=0.5)
        self.assertAlmostEqual(float(H[np.argmin(np.abs(fg - 200.0))].max()), 30.0, places=4)


class ShortRecordings(unittest.TestCase):
    def test_report_clips_are_detected(self):
        for dur in (20, 69, 79):                         # 1, 1 and 2 detection windows
            with self.subTest(dur=dur):
                ev, info = audio.detect_events(spec_of(strong(dur)), return_info=True)
                self.assertEqual(len(ev), 1)
                self.assertEqual(info["baseline"], "noise-model")
                self.assertEqual(info["status"], "limited")
                self.assertTrue(info["reasons"])
                self.assertLess(ev[0]["fmin"], 100.0)
                self.assertGreater(ev[0]["fmax"], 100.0)

    def test_realistic_level_short_pass(self):
        for d in (300, 1000):
            with self.subTest(d=d):
                ev = audio.detect_events(spec_of(ambient(20, 7) + flyby(20, f0=110, v=60, d=d)))
                self.assertEqual(len(ev), 1)

    def test_shorter_than_min_dur_is_insufficient(self):
        spec = spec_of(strong(9))
        ev, info = audio.detect_events(spec, return_info=True)
        self.assertEqual(ev, [])
        self.assertEqual(info["status"], "insufficient")
        self.assertEqual(len(info["short_hits"]), 1)
        self.assertEqual(len(audio.detect_events(spec, min_dur_s=5.0)), 1)

    def test_fully_gated_is_insufficient(self):
        spec = spec_of(strong(20))
        spec["gated"] = np.ones(spec["t"].size, bool)
        ev, info = audio.detect_events(spec, return_info=True)
        self.assertEqual(ev, [])
        self.assertEqual(info["status"], "insufficient")


class NoFalseAlarms(unittest.TestCase):
    def test_noise_only(self):
        for dur, seeds in ((20, range(3)), (69, range(3)), (300, range(1))):
            for seed in seeds:
                with self.subTest(dur=dur, seed=seed):
                    ev, info = audio.detect_events(spec_of(ambient(dur, 100 + seed)), return_info=True)
                    self.assertEqual(ev, [])
                    self.assertNotEqual(info["status"], "insufficient")
        self.assertEqual(info["baseline"], "relative")       # 300 s: enough windows for a median
        self.assertEqual(info["status"], "ok")

    def test_noise_model_matches_background(self):
        # whitening makes the noise ridge level independent of loudness and colour
        levels = [audio.detect_events(spec_of(a * np.random.default_rng(9).normal(size=60 * FS)),
                                      return_info=True)[1]["median_noise_excess_db"] for a in (0.001, 0.3)]
        self.assertAlmostEqual(levels[0], levels[1], places=3)               # float32 rounding only
        _, info = audio.detect_events(spec_of(ambient(300, 11)), return_info=True)
        self.assertLess(abs(info["median_noise_excess_db"]), 0.3)

    def test_stationary_hum(self):
        t = np.arange(60 * FS) / FS
        x = ambient(60, 4) + 0.01 * (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t))
        self.assertEqual(audio.detect_events(spec_of(x)), [])


class Coverage(unittest.TestCase):
    def test_pass_filling_most_of_the_recording(self):
        x = ambient(150, 3) + flyby(150, f0=95, v=45, d=250, amp=0.01)
        ev, info = audio.detect_events(spec_of(x), return_info=True)
        self.assertGreaterEqual(len(ev), 1)
        self.assertEqual(info["baseline"], "noise-model")

    def test_pass_at_the_very_end(self):
        x = ambient(130, 5) + flyby(130, f0=105, v=60, d=200, tc=124.0)
        spec = spec_of(x)
        ev = audio.detect_events(spec, min_dur_s=10.0)
        self.assertTrue(ev)
        self.assertGreater(max(e["core_t1"] for e in ev), spec["t"][-1] - 1.0)


class Audit(unittest.TestCase):
    def audit(self, x):
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "clip.wav")
            wavfile.write(wav, FS, (0.9 * x / np.abs(x).max() * 32767).astype(np.int16))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                afg_cli.main(["audit", wav, "--out", os.path.join(d, "audit")])
            return json.loads(buf.getvalue())

    def test_short_flyover_is_not_low(self):
        out = self.audit(strong(20))
        self.assertTrue(out["risk"].startswith("medium"), out["risk"])
        self.assertEqual(len(out["candidate_flyovers"]), 1)

    def test_too_short_is_undetermined(self):
        out = self.audit(strong(9))
        self.assertTrue(out["risk"].startswith("undetermined"), out["risk"])
        self.assertEqual(out["detection"]["status"], "insufficient")

    def test_quiet_recording_is_low(self):
        out = self.audit(ambient(120, 9))
        self.assertTrue(out["risk"].startswith("low"), out["risk"])
        self.assertIn("risk_scope", out)


if __name__ == "__main__":
    unittest.main()
