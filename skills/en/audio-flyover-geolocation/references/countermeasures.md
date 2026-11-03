# Privacy countermeasures: what works and what doesn't

The end of the video is devoted to this. Conclusion: **only destroying the information itself (cutting it,
replacing it) is reliable; signal-processing "masking" mostly fails.**

| Measure | Effect | Why |
|---|---|---|
| **Cut out the fly-over** | ✅ Works | No curve means no v/d/tCPA. Cut at least one fly-over, or change the time interval between two fly-overs (the two-event method depends on it) |
| **Replace / remove the original audio** (re-dub, narration only) | ✅ Most reliable | The background sound field disappears entirely |
| Publish only very short clips (< 20 s, without the CPA) | ⚠️ Partial | With only one asymptote v and d are undetermined, but if the steep CPA section is included, k = v²/d still works |
| **High-pass filter** (e.g. > 200 Hz) | ❌ Doesn't work | Propeller BPF harmonics extend to several hundred Hz and overlap the voice band. Filtering only attenuates; narrow-band tones can still be recovered in the whitened spectrogram |
| **"Drowning it out"** (music, white noise, talking) | ❌ Mostly doesn't work | Broadband masking only raises the noise floor; a long narrow-band tone still has enough integrated SNR. Narrow-band parts of music are discontinuous and easy to tell apart |
| **Phone-call codecs / noise suppression** | ❌ Unreliable | Speech codecs keep narrow-band tones between 100 and 4000 Hz. Noise suppression usually gates only when nobody is speaking; while you talk, the background goes through with your voice |
| Speed / pitch change | ⚠️ Partial | A uniform speed change scales f0 and ΔT together; it can be undone from the 50/60 Hz hum or a known duration |
| Adding a fake Doppler curve | ⚠️ Risky | Could mislead, but only "looks real" if it is consistent with real ADS-B, and it doesn't erase the real curve |

## Pre-publication checklist

1. `$AFG audit my.mp4 --out work/audit` → look at `risk` and `work/audit.png` (look for smooth descending
   curves).
2. Also check:
   - EXIF/GPS and QuickTime `©xyz` location tags, plus creation time;
   - in the picture: road signs, skyline, sun shadows, reflections;
   - other ambient sounds (bells, PA announcements, trains);
   - ENF (mains frequency), which can also be used for time forensics.
3. If risk is medium/high: cut the fly-over or replace the audio, then audit again.

## Points to explain to the user

- Inaudible to the ear ≠ absent from the spectrogram.
- Two fly-overs + their time difference are more dangerous than one fly-over + exact time, because the recording
  time isn't needed.
- Risk grows with nearby GA airfields / flight schools and helicopter route density.
