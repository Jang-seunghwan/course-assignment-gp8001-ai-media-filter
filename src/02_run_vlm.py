"""clips + transcripts + prompts -> results/P0.jsonl, results/P1.jsonl

    python 02_run_vlm.py                       # test split, P0 and P1  -> results/P0.jsonl, P1.jsonl
    python 02_run_vlm.py --split dev --prompts P1   # prompt tuning   -> results/dev_P1.jsonl
    python 02_run_vlm.py --clips c001 c002     # specific clips (written to the split's file)

Already-finished clip/prompt pairs are skipped, so an interrupted run can be restarted.
Use --overwrite after changing a prompt.
"""
import argparse
import json

import pandas as pd

from common import (CLIPS_DIR, LABELS_CSV, PROMPT_FILES, RESULTS_DIR, TRANSCRIPTS_DIR,
                    classify, transcribe)

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test", choices=["test", "dev", "all"])
ap.add_argument("--prompts", nargs="+", default=["P0", "P1"], choices=list(PROMPT_FILES))
ap.add_argument("--clips", nargs="*", help="clip ids; default = every clip of --split in labels.csv")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()


def select_clips():
    if args.clips:
        return args.clips
    labels = pd.read_csv(LABELS_CSV, dtype=str) if LABELS_CSV.exists() else pd.DataFrame()
    if labels.empty:
        print(f"[warn] {LABELS_CSV.name} has no rows -> running on every mp4 in {CLIPS_DIR}")
        return sorted(p.stem for p in CLIPS_DIR.glob("*.mp4"))
    if args.split != "all":
        labels = labels[labels["split"] == args.split]
    return labels["clip_id"].tolist()


def get_transcript(clip_id):
    path = TRANSCRIPTS_DIR / f"{clip_id}.txt"
    if not path.exists():
        path.write_text(transcribe(CLIPS_DIR / f"{clip_id}.mp4") + "\n", encoding="utf-8")
    return path.read_text(encoding="utf-8")


clip_ids = select_clips()
RESULTS_DIR.mkdir(exist_ok=True)
for prompt in args.prompts:
    out_path = RESULTS_DIR / (f"{prompt}.jsonl" if args.split == "test" else f"{args.split}_{prompt}.jsonl")
    done = set()
    if out_path.exists() and not args.overwrite:
        done = {json.loads(line)["clip_id"] for line in out_path.read_text().splitlines() if line.strip()}
    elif out_path.exists():
        out_path.unlink()
    todo = [c for c in clip_ids if c not in done]
    print(f"== {prompt}: {len(todo)} to run, {len(done)} already in {out_path.name}")
    for clip_id in todo:
        video = CLIPS_DIR / f"{clip_id}.mp4"
        if not video.exists():
            print(f"[skip] {video} not found")
            continue
        res = {"clip_id": clip_id, **classify(video, get_transcript(clip_id), prompt)}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"{clip_id} {prompt} score={res['score']} code={res['code']} {res['sec']}s | {res['reason']}")
