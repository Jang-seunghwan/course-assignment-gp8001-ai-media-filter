"""clips -> transcripts/<clip_id>.txt  (faster-whisper small, auto language, "(no speech)" if silent)

    python 01_transcribe.py              # all clips in data/clips, skips ones already done
    python 01_transcribe.py --overwrite
"""
import argparse
import time

from common import CLIPS_DIR, TRANSCRIPTS_DIR, transcribe

ap = argparse.ArgumentParser()
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

TRANSCRIPTS_DIR.mkdir(exist_ok=True)
clips = sorted(CLIPS_DIR.glob("*.mp4"))
if not clips:
    raise SystemExit(f"No .mp4 files in {CLIPS_DIR}")

for clip in clips:
    out = TRANSCRIPTS_DIR / f"{clip.stem}.txt"
    if out.exists() and not args.overwrite:
        continue
    t0 = time.time()
    text = transcribe(clip)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"{clip.stem}  {time.time() - t0:5.1f}s  {text[:70]}")
print(f"done: {len(clips)} clips -> {TRANSCRIPTS_DIR}")
