"""False-alarm sweep: run the pipeline over every clip in data/videos and tabulate alerts.

All bundled clips are NORMAL activity, so any FALL/UNRESPONSIVE/FIGHT alert here is a false alarm.
"""
import glob, json, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_video import make_backend, run
from safety.pipeline import SafetyPipeline, learned_pipeline

backend = make_backend(sys.argv[1] if len(sys.argv) > 1 else "yolo")
LEARNED = "learned" in sys.argv[2:]          # python scripts/eval_clips.py yolo learned
rows, total_s, total_alerts = [], 0.0, 0
for v in sorted(glob.glob("data/videos/*")):
    c = cv2.VideoCapture(v); fps = c.get(cv2.CAP_PROP_FPS) or 10; c.release()
    every = max(1, round(fps / 10))                       # emulate ~10 FPS processing on the board
    s = run(v, backend, every=every, max_seconds=120, quiet=True, pipe=learned_pipeline(backend, os.environ.get('FIGHT_MODEL'), thr=float(os.environ['FIGHT_THR']) if os.environ.get('FIGHT_THR') else None) if LEARNED else SafetyPipeline(backend),
            sheet=f"outputs/sheets/{os.path.splitext(os.path.basename(v))[0]}_eval.jpg")
    bad = [a for a in s["alerts"] if a["kind"] in ("FALL", "DOWN", "UNRESPONSIVE", "FIGHT")]
    total_s += s["seconds"]; total_alerts += len(bad)
    print(f"{s['video']:36s} {s['seconds']:6.1f}s people={s['frames_with_people']:.0%} max={s['max_people']} "
          f"{s['ms_mean']:5.1f}ms  false_alarms={len(bad)} {[(a['kind'], a['t']) for a in bad]}")
print(f"\nTOTAL {total_s/60:.1f} min of normal footage, {total_alerts} false alarms")
