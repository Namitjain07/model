#!/usr/bin/env bash
# AIRTLab violence-detection dataset (350 staged Full-HD clips: kicks, punches, pushes, slaps, chokes,
# stabs, clubbing, gunshots vs. hugs, high-fives, handshakes, jumping, gesticulating).
# Released "freely for research and educational purposes" - cite: Bianculli et al., Data in Brief 33
# (2020), doi:10.1016/j.dib.2020.106587.  Files go to data/airtlab/ (git-ignored; do not commit/redistribute).
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://raw.githubusercontent.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/master/violence-detection-dataset
OUT=data/airtlab; mkdir -p $OUT/violent/cam1 $OUT/violent/cam2 $OUT/non-violent/cam1 $OUT/non-violent/cam2
for f in action-class-occurrences.csv violent-action-classes.csv nonviolent-action-classes.csv; do
  [ -s $OUT/$f ] || curl -fsSL --retry 3 -o $OUT/$f "$BASE/$f"
done
{ for c in cam1 cam2; do
    for i in $(seq 1 115); do echo "violent/$c/$i.mp4"; done
    for i in $(seq 1 60);  do echo "non-violent/$c/$i.mp4"; done
  done; } > $OUT/.manifest
fetch() { [ -s "$OUT/$1" ] && exit 0; curl -fsSL --retry 4 --retry-delay 2 -m 120 -o "$OUT/$1.part" "$BASE/$1" && mv "$OUT/$1.part" "$OUT/$1"; }
export -f fetch; export BASE OUT
xargs -P 6 -I{} bash -c 'fetch {} || echo "FAILED {}" >&2' < $OUT/.manifest
echo "clips: $(find $OUT -name '*.mp4' | wc -l) / 350 | size: $(du -sh $OUT | cut -f1)"
