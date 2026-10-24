"""Build the website's media and data from the finished experiment (no model is run).

    python tools/build_site_assets.py --project ../project

Reads the experiment folder (data/clips/*.mp4, data/labels.csv, results/P0.jsonl, results/P1.jsonl) and
writes docs/assets/{clips,thumbs,posters,frames}/ and docs/assets/clips.json. Needs ffmpeg and Pillow.
The source clips are not part of this repository.
"""
import argparse
import io
import json
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

# Grid order. 9 sludge clips the team checked by hand, 8 normal clips without large real faces,
# and c012, the one normal clip P1 flagged by mistake.
CLIPS = ["c044", "c024", "c020", "c025", "c066", "c208", "c159", "c234", "c012",
         "c037", "c192", "c072", "c160", "c230", "c075", "c186", "c001", "c255"]
NOTES = {"c012": "Our check: the model was wrong — this clip has a single screen."}
# @handle watermarks, blurred for as long as they are on screen:
# clip -> [(t0, t1, x, y, w, h)] in seconds and fractions of the source frame.
BLUR = {
    "c012": [(0.0, 3.2, 0.06, 0.630, 0.34, 0.070)],     # "@FALSEFITT" on the jacket, fades out by 3 s
    "c192": [(0.0, 1.6, 0.37, 0.070, 0.26, 0.055)],     # "@FALSEFITT" top centre, fades out by 1.5 s
    "c025": [(0.0, 13.0, 0.17, 0.835, 0.66, 0.060)],    # "@HORSECRAFTSTUDIOS" bottom, whole clip
}
SECONDS, W, H = 12, 540, 960
CRF, MAX_BYTES = 28, 2_000_000
FRAMES = {"c020": [0.2, 0.4, 0.6, 0.8], "c012": [None]}   # explanation frames (fraction of the full clip)
C012_FRAME_AT = 5.0                                       # seconds into the web clip, after the watermark


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    st = info["streams"][0]
    num, den = map(int, st["r_frame_rate"].split("/"))
    return st["width"], st["height"], num / den, float(info["format"]["duration"])


def filter_graph(cid, src_w, src_h, fps):
    parts, cur = [f"[0:v]{'fps=30,' if fps > 30.5 else ''}setsar=1[s0]"], "s0"
    for k, (t0, t1, x, y, w, h) in enumerate(BLUR.get(cid, [])):
        parts.append(f"[{cur}]split[m{k}][c{k}];[c{k}]crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},gblur=sigma=18[b{k}];"
                     f"[m{k}][b{k}]overlay=x=W*{x}:y=H*{y}:enable='between(t,{t0},{t1})'[s{k + 1}]")
        cur = f"s{k + 1}"
    if abs(src_w / src_h - W / H) < 0.01:
        parts.append(f"[{cur}]scale={W}:{H},format=yuv420p[out]")
    else:   # not 9:16: fit inside, fill the rest with a blurred, darkened copy of the same video
        parts.append(f"[{cur}]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                     f"gblur=sigma=24,eq=brightness=-0.2[bg];[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
                     f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[out]")
    return ";".join(parts)


def encode(cid, src, dst):
    w, h, fps, _ = probe(src)
    crf = CRF
    while True:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(SECONDS),
                        "-filter_complex", filter_graph(cid, w, h, fps), "-map", "[out]", "-map", "0:a:0?",
                        "-c:v", "libx264", "-profile:v", "main", "-preset", "slow", "-crf", str(crf),
                        "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-movflags", "+faststart", str(dst)], check=True)
        if dst.stat().st_size <= MAX_BYTES or crf >= 40:
            return crf
        crf += 2


def grab(path, t):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(out)).convert("RGB")


def result_rows(project, prompt):
    rows = (project / "results" / f"{prompt}.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["clip_id"]: r for r in map(json.loads, filter(None, rows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True, help="experiment folder with data/ and results/")
    args = ap.parse_args()
    project = args.project.resolve()
    labels = pd.read_csv(project / "data" / "labels.csv", dtype=str).set_index("clip_id")
    p0, p1 = result_rows(project, "P0"), result_rows(project, "P1")
    for sub in ("clips", "thumbs", "posters", "frames"):
        (ASSETS / sub).mkdir(parents=True, exist_ok=True)

    clips = []
    for cid in CLIPS:
        lab = labels.loc[cid]
        assert lab["split"] == "test", cid
        src = project / "data" / "clips" / lab["file"]
        web = ASSETS / "clips" / f"{cid}.mp4"
        crf = encode(cid, src, web)
        frame = grab(web, 1.0)                                         # poster and thumbnail: the 1 s frame
        frame.save(ASSETS / "posters" / f"{cid}.jpg", quality=75)
        side = min(frame.size)
        top = (frame.height - side) // 2
        frame.crop((0, top, side, top + side)).resize((360, 360), Image.LANCZOS).save(
            ASSETS / "thumbs" / f"{cid}.jpg", quality=75)
        entry = {"id": cid, "video": f"assets/clips/{cid}.mp4", "thumb": f"assets/thumbs/{cid}.jpg",
                 "poster": f"assets/posters/{cid}.jpg", "label": "sludge" if lab["final_br"] == "1" else "not sludge"}
        for key, res in (("p0", p0[cid]), ("p1", p1[cid])):
            entry[key] = {"score": res["score"], "reason": res["reason"], "sec": res["sec"]}  # copied verbatim
        entry["note"] = NOTES.get(cid, "")
        clips.append(entry)
        print(f"{cid} {entry['label']:10s} p0={entry['p0']['score']:.3f} p1={entry['p1']['score']:.3f} "
              f"crf={crf} {web.stat().st_size / 1e6:.2f} MB")

    full = project / "data" / "clips" / labels.loc["c020", "file"]
    dur = probe(full)[3]
    for i, frac in enumerate(FRAMES["c020"], 1):
        im = grab(full, dur * frac)
        im.thumbnail((270, 480))
        im.save(ASSETS / "frames" / f"c020_{i}.jpg", quality=80)
    im = grab(ASSETS / "clips" / "c012.mp4", C012_FRAME_AT)
    im.thumbnail((270, 480))
    im.save(ASSETS / "frames" / "c012_1.jpg", quality=80)

    (ASSETS / "clips.json").write_text(json.dumps(clips, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(f.stat().st_size for f in ASSETS.rglob("*") if f.is_file())
    print(f"wrote {len(clips)} clips, docs/assets = {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
