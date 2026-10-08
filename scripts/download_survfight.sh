#!/usr/bin/env bash
# Surveillance Camera Fight Dataset (Akti, Tataroglu, Ekenel, IPTA 2019; MIT license): 150 fight + 150 non-fight
# 2-second clips cut from YouTube surveillance-style footage (cafes, streets, buses, ...). Cite the IPTA 2019 paper.
# videos.txt maps clips to their source YouTube video -> use it to group CV folds (clips from one video must not straddle).
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://raw.githubusercontent.com/seymanurakti/fight-detection-surv-dataset/master
OUT=data/survfight; mkdir -p $OUT/fight $OUT/noFight
[ -s $OUT/videos.txt ] || curl -fsSL --retry 3 -o $OUT/videos.txt "$BASE/videos.txt"
{ for i in $(seq -f "%03g" 1 150); do echo "fight/fi$i.mp4"; echo "noFight/nofi$i.mp4"; done; } > $OUT/.manifest
fetch() { [ -s "$OUT/$1" ] && exit 0; curl -fsSL --retry 4 --retry-delay 2 -m 60 -o "$OUT/$1.part" "$BASE/$1" && mv "$OUT/$1.part" "$OUT/$1"; }
export -f fetch; export BASE OUT
xargs -P 8 -I{} bash -c 'fetch {} || echo "FAILED {}" >&2' < $OUT/.manifest
echo "fight: $(ls $OUT/fight | wc -l)/150  noFight: $(ls $OUT/noFight | wc -l)/150  size: $(du -sh $OUT | cut -f1)"
