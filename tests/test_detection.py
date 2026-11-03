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


def burst(dur, tc, half=13.0, **kw):
    """A pass audible only for about +-half s around tc (26 s events with 2 s fades)."""
    x = flyby(dur, tc=tc, **kw)
    t = np.arange(x.size) / FS
    return x * np.clip((half - np.abs(t - tc)) / 2.0, 0, 1)


def two_passes(rising_tone=False):
    """The second defect report's construction: passes at 45 s and 135 s of a 180 s recording."""
    x = ambient(180, 21) + burst(180, 45.0, f0=105, v=60, d=250) + burst(180, 135.0, f0=95, v=55, d=300)
    if rising_tone:                                  # a quieter tone climbing 70 -> 115 Hz under both
        t = np.arange(x.size) / FS
        x = x + 0.004 * np.sin(2 * np.pi * np.cumsum(70 + 0.25 * t) / FS)
    return x


def pass_under_tone():
    """A loud rising tone with a brief pass (about 10 s) underneath it: the ridge follows the tone."""
    t = np.arange(60 * FS) / FS
    return (ambient(60, 31) + 0.01 * np.sin(2 * np.pi * np.cumsum(70 + 0.5 * t) / FS)
            + burst(60, 30.0, half=4.0, f0=110, v=60, d=200, amp=0.01))


def rising_tone(dur=30):
    """A test tone sweeping up from 100 Hz at 1.2 Hz/s: tonal, but no fly-over can rise."""
    t = np.arange(int(dur * FS)) / FS
    return np.sin(2 * np.pi * np.cumsum(100 + 1.2 * t) / FS) + 0.05 * np.random.default_rng(3).normal(size=t.size)


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

    def test_a_few_frames_do_not_crash(self):
        for dur, frames in ((1.0, 1), (1.4, 3), (1.8, 5), (2.0, 6)):
            with self.subTest(dur=dur):
                spec = spec_of(np.zeros(int(dur * FS)))
                self.assertEqual(spec["t"].size, frames)
                self.assertEqual(spec["gated"].shape, spec["t"].shape)
                ev, info = audio.detect_events(spec, return_info=True)
                self.assertEqual(ev, [])
                self.assertEqual(info["status"], "insufficient")
                self.assertEqual(info["rejected"], [])
        with self.assertRaises(ValueError):          # less than one analysis window: callers report it
            spec_of(np.zeros(int(0.7 * FS)))


class SeparatePasses(unittest.TestCase):
    def check_two(self, ev):
        self.assertEqual(len(ev), 2, [(e["core_t0"], e["core_t1"]) for e in ev])
        for e, tc in zip(sorted(ev, key=lambda e: e["t0"]), (45.0, 135.0)):
            self.assertLess(e["core_t0"], tc)
            self.assertGreater(e["core_t1"], tc)
            self.assertLess(e["t1"] - e["t0"], 90.0)
            self.assertEqual(e["trend"], "descending")
        self.assertLessEqual(ev[0]["t1"], ev[1]["t0"])

    def test_two_passes_a_minute_apart(self):
        spec = spec_of(two_passes())
        for kw in ({}, {"win_s": 20.0, "step_s": 5.0}):
            with self.subTest(**kw):
                self.check_two(audio.detect_events(spec, **kw))

    def test_two_passes_over_a_rising_tone(self):
        ev, info = audio.detect_events(spec_of(two_passes(rising_tone=True)), return_info=True)
        self.check_two(ev)
        self.assertTrue(info["rejected"])
        self.assertTrue(all(e["trend"] == "rising" for e in info["rejected"]))


class Trend(unittest.TestCase):
    def test_flyover_falls(self):
        ev = audio.detect_events(spec_of(strong(20)))
        self.assertEqual(ev[0]["trend"], "descending")
        self.assertGreater(ev[0]["drop_pct"], 5.0)
        self.assertEqual(ev[0]["warnings"], [])

    def test_rising_tone_is_set_aside(self):
        ev, info = audio.detect_events(spec_of(rising_tone()), return_info=True)
        self.assertEqual(ev, [])
        self.assertEqual(len(info["rejected"]), 1)
        self.assertEqual(info["rejected"][0]["trend"], "rising")
        self.assertLess(info["rejected"][0]["drop_pct"], -10.0)

    def test_search_continues_under_a_set_aside_tone(self):
        ev, info = audio.detect_events(spec_of(pass_under_tone()), min_dur_s=12.0, return_info=True)
        self.assertEqual(ev, [])
        self.assertTrue(info["rejected"][0]["warnings"])
        self.assertTrue(any(h["t0"] < 30.0 < h["t1"] for h in info["short_hits"]), info["short_hits"])

    def test_turning_points(self):
        tp = audio._turning_points
        self.assertEqual(tp(np.array([5.0, 5.5, 4.8, 5.2]), 3.0), [0, 3])           # wiggles do not split
        self.assertEqual(tp(np.r_[np.linspace(120, 90, 30), np.linspace(91, 110, 20)], 3.0), [0, 29, 49])
        sw = audio._swings(np.r_[np.linspace(120, 90, 30), np.full(10, 90.0), np.linspace(91, 130, 5),
                                 np.full(10, 130.0), np.linspace(129, 100, 30)], 3.0)
        self.assertEqual([s[2] for s in sw], [True, False, True])
        self.assertEqual((sw[1][0], sw[1][1]), (40, 44))   # the flat bottom and top stay with the falls
        self.assertEqual(tp(np.linspace(70, 115, 50), 3.0), [0, 49])


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


def run_cli(cmd, x, *extra):
    """Write x as a 16-bit WAV, run one afg_cli command on it, return (exit code, stdout, out prefix files)."""
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "clip.wav")
        peak = np.abs(x).max() if x.size else 0.0
        wavfile.write(wav, FS, (0.9 * x / peak * 32767 if peak else x).astype(np.int16))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            code = afg_cli.main([cmd, wav, "--out", os.path.join(d, "out"), *extra])
        return code, buf.getvalue(), sorted(os.listdir(d))


class Audit(unittest.TestCase):
    def audit(self, x):
        code, stdout, _ = run_cli("audit", x)
        self.assertEqual(code, 0)
        return json.loads(stdout)

    def test_rising_tone_is_not_a_flyover(self):
        out = self.audit(rising_tone())
        self.assertFalse(out["risk"].startswith(("medium", "high")), out["risk"])
        self.assertEqual(out["candidate_flyovers"], [])
        self.assertEqual(len(out["detection"]["rejected"]), 1)
        self.assertTrue(any("rising" in n for n in out["notes"]))

    def test_pass_under_a_rising_tone_is_not_cleared(self):
        out = self.audit(pass_under_tone())
        self.assertTrue(out["risk"].startswith("undetermined"), out["risk"])

    def test_two_separate_passes_are_high(self):
        out = self.audit(two_passes())
        self.assertEqual(out["risk"], "high")
        self.assertEqual(len(out["candidate_flyovers"]), 2)

    def test_very_short_files(self):
        for dur in (0.0, 0.7, 1.0, 1.4, 2.0):
            with self.subTest(dur=dur):
                out = self.audit(np.zeros(int(dur * FS)))
                self.assertTrue(out["risk"].startswith("undetermined"), out["risk"])
                self.assertEqual(out["detection"]["status"], "insufficient")
                self.assertTrue(out["detection"]["reasons"])
                if dur < 1.0:
                    self.assertIsNone(out["png"])

    def test_spectrogram_of_a_sub_window_clip_exits_cleanly(self):
        code, stdout, files = run_cli("spectrogram", np.zeros(int(0.7 * FS)))
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(files, ["clip.wav"])

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
