#!/usr/bin/env bash
# End-to-end CLI run on a synthetic scenario: simulate a recording plus ADS-B traces,
# then locate the recording with the same commands you would use on real footage.
#
#   bash examples/synthetic_demo.sh [workdir] [seed]
#
# Afterwards `python tools/make_figures.py <workdir>` redraws the README figures.
set -euo pipefail

W=${1:-work/demo}
SEED=${2:-23}
PY=${PYTHON:-python}
AFG="$PY $(dirname "$0")/../skills/en/audio-flyover-geolocation/scripts/afg_cli.py"
mkdir -p "$W"

$AFG simulate --out "$W" --seed "$SEED" > /dev/null
$AFG spectrogram "$W/recording.wav" --out "$W/rec" > /dev/null
$AFG detect "$W/rec.spec.npz" --out "$W/boxes.json" > /dev/null

# the two strongest detections, in time order
read -r B1 B2 <<< "$($PY -c "
import json, sys
b = sorted(json.load(open(sys.argv[1])), key=lambda e: -e['score_z'])[:2]
b.sort(key=lambda e: e['t0'])
print(*[f\"{e['t0']:.1f},{e['t1']:.1f},{e['fmin']:.1f},{e['fmax']:.1f}\" for e in b])" "$W/boxes.json")"

i=1
for B in $B1 $B2; do
  IFS=, read -r t0 t1 f0 f1 <<< "$B"
  $AFG trace "$W/rec.spec.npz" --t0 "$t0" --t1 "$t1" --fmin "$f0" --fmax "$f1" --out "$W/e$i.csv" > /dev/null
  $AFG fit "$W/e$i.csv" --out "$W/e$i.json" > /dev/null
  i=$((i + 1))
done

# search window, area and recording length come from the scenario file here;
# on real footage they come from what you know about the video
read -r S E BB DUR <<< "$($PY -c "
import json, sys
t = json.load(open(sys.argv[1]))
print(t['day_window'][0], t['day_window'][1], ','.join(map(str, t['bbox'])), t['rec_duration_s'])" "$W/truth.json")"

$AFG adsb-ingest --src "$W/traces" --start "$S" --end "$E" --bbox "$BB" --out "$W/tracks.npz" > /dev/null
$AFG search --tracks "$W/tracks.npz" --events "$W/e1.json" "$W/e2.json" --bbox "$BB" \
    --ground-elev 250 --rec-duration "$DUR" --out "$W/cands.npy" > /dev/null
$AFG score --tracks "$W/tracks.npz" --spec "$W/rec.spec.npz" --events "$W/e1.json" "$W/e2.json" \
    --cands "$W/cands.npy" --ground-elev 250 --out "$W/result.json" > "$W/score.out"

$PY -c "
import json, math, sys
r = json.load(open(sys.argv[1])); t = json.load(open(sys.argv[2]))['observer']; b = r['best']
d = 6371000 * math.hypot(math.radians(b['lat'] - t['lat']),
                         math.radians(b['lon'] - t['lon']) * math.cos(math.radians(t['lat'])))
print(f\"estimate {b['lat']:.5f},{b['lon']:.5f}  error {d:.0f} m  0.9-region radius {r['region_max_radius_m']:.0f} m\")
print(f\"flights  {b['flight1']} / {b['flight2']}  recording start {b['rec_start_iso']}\")
print(f\"verdict  {r['verdict']}\")" "$W/result.json" "$W/truth.json"
