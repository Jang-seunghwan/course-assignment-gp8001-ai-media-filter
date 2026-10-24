"""Shared helpers for 01/02 scripts and the Gradio demo.

Pipeline for one clip: Whisper transcript + 1 fps frames -> Qwen3-VL-4B -> JSON {score, code, reason}.
"""
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

# Must run before anything imports huggingface_hub (gradio, transformers, faster_whisper),
# because the hub library reads these variables once at import time. Import this module first.
# Export HF_HOME to choose where model weights are cached.


def _cached(repo, filename):
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return any((hub / f"models--{repo.replace('/', '--')}" / "snapshots").glob(f"*/{filename}"))


ROOT = Path(__file__).resolve().parent.parent      # repository root: data/, transcripts/, prompts/, results/
CLIPS_DIR = ROOT / "data" / "clips"
LABELS_CSV = ROOT / "data" / "labels.csv"
TRANSCRIPTS_DIR = ROOT / "transcripts"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"

MODEL_ID = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
WHISPER_SIZE = "small"
FPS = 1.0
MAX_PIXELS = 360 * 640          # per-frame resolution cap
MAX_SECONDS = 60
MAX_NEW_TOKENS = 150
MAX_TRANSCRIPT_CHARS = 1500     # keep the prompt short for talky clips
NO_SPEECH = "(no speech)"

# Once both models are downloaded, skip hub version checks: they hang on a flaky network
# (seen here: 3 min model load instead of 13 s). Set HF_HUB_OFFLINE=0 to force online.
if _cached(MODEL_ID, "config.json") and _cached(f"Systran/faster-whisper-{WHISPER_SIZE}", "model.bin"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROMPT_FILES = {"P0": "p0_brainrot.txt", "P1": "p1_brainrot.txt"}
# Scope of the Kaggle run: brainrot = B1 only (CLAUDE_CODE_TASK_kaggle.md §0). B2-B4 are out of scope.
SUBTYPES = {
    "B1": "Split-screen sludge (화면 분할 sludge)",
    "none": "No subtype",
}


# ---------------------------------------------------------------- utilities
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / PROMPT_FILES[name]).read_text(encoding="utf-8")


def fill_prompt(template: str, transcript: str) -> str:
    # str.replace, not str.format: the template contains literal JSON braces.
    return template.replace("{transcript}", transcript.strip()[:MAX_TRANSCRIPT_CHARS] or NO_SPEECH)


def video_duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def trim_to_limit(path, max_seconds: int = MAX_SECONDS) -> str:
    """Return a path to a copy cut to the first `max_seconds` (or the original if already short)."""
    if video_duration(path) <= max_seconds + 0.5:
        return str(path)
    out = Path(tempfile.mkdtemp()) / (Path(path).stem + "_60s.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-t", str(max_seconds),
                    "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out)], check=True)
    return str(out)


# ---------------------------------------------------------------- whisper
_whisper = None


def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        try:
            _whisper = WhisperModel(WHISPER_SIZE, device="cuda", compute_type="float16")
        except Exception as e:  # missing cuDNN/cuBLAS etc. -> CPU is fine for <=60 s clips
            print(f"[whisper] CUDA init failed ({e}); falling back to CPU int8")
            _whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _whisper


def has_audio(path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(out.stdout.strip())


def transcribe(path) -> str:
    """Language is auto-detected. Clips without an audio track or speech give NO_SPEECH."""
    if not has_audio(path):
        return NO_SPEECH
    try:
        segments, _info = get_whisper().transcribe(str(path), vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        print(f"[whisper] {Path(path).name}: {e}")
        text = ""
    return text or NO_SPEECH


# ---------------------------------------------------------------- VLM
_vlm = None


def get_vlm():
    global _vlm
    if _vlm is None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        model.eval()
        _vlm = (model, AutoProcessor.from_pretrained(MODEL_ID))
    return _vlm


def generate(video_path, prompt: str, sample: bool = False):
    """Returns (reply text, P(yes) at the first yes/no token or None)."""
    import torch
    from qwen_vl_utils import process_vision_info
    model, processor = get_vlm()
    messages = [{"role": "user", "content": [
        {"type": "video", "video": str(video_path), "fps": FPS, "max_pixels": MAX_PIXELS},
        {"type": "text", "text": prompt},
    ]}]
    # Video preprocessing as in the Qwen3-VL model card / qwen-vl-utils README.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    video_metadata = None
    if videos is not None:
        videos, video_metadata = map(list, zip(*videos))
    inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadata,
                       do_resize=False, return_tensors="pt", **video_kwargs).to(model.device)
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=sample,
                      output_logits=True, return_dict_in_generate=True)
    if sample:  # only used for the single retry after a JSON parse failure
        gen_kwargs.update(temperature=0.7, top_p=0.8, top_k=20)
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    reply = processor.decode(new_tokens, skip_special_tokens=True)
    return reply, _yes_probability(new_tokens, out.logits, processor.tokenizer)


_yes_no = None


def _yes_no_ids(tokenizer):
    """Every vocabulary token that reads as yes / no (any case, with or without leading space or quote)."""
    global _yes_no
    if _yes_no is None:
        yes, no = [], []
        for i, tok in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
            word = (tok or "").replace("\u0120", " ").strip().strip('"').lower()
            if word == "yes":
                yes.append(i)
            elif word == "no":
                no.append(i)
        _yes_no = (yes, no)
    return _yes_no


def _yes_probability(new_tokens, logits, tokenizer):
    """Spec ch.5 fallback score: at the first generated yes/no token (the "brainrot" verdict, which
    the yes/no prompt puts first in the JSON), P(yes) / (P(yes) + P(no)) from the model's logits."""
    yes, no = _yes_no_ids(tokenizer)
    yes_set, no_set = set(yes), set(no)
    for step, tok in enumerate(new_tokens):
        if tok in yes_set or tok in no_set:
            probs = logits[step][0].float().softmax(-1)
            p_yes, p_no = probs[yes].sum().item(), probs[no].sum().item()
            return p_yes / (p_yes + p_no) if p_yes + p_no > 0 else None
    return None


def parse_json(raw: str, p_yes=None):
    """Grab the first {...} in the reply and normalise fields. Returns None on failure.

    Two answer formats are accepted, decided by the prompt file:
      {"score": 0.8, ...}        -> score as stated by the model ("verbal")
      {"brainrot": "yes", ...}   -> score = P(yes) of the verdict token ("yesprob")
    """
    for pattern in (r"\{.*?\}", r"\{.*\}"):
        m = re.search(pattern, raw, re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
            if "brainrot" in obj:
                verdict = str(obj["brainrot"]).strip().lower()
                if verdict not in ("yes", "no"):
                    continue
                score = p_yes if p_yes is not None else float(verdict == "yes")
                mode = "yesprob"
            else:
                score, verdict, mode = float(obj["score"]), None, "verbal"
                if 1 < score <= 100:          # model answered in percent
                    score /= 100
        except (ValueError, KeyError, TypeError):
            continue
        code = str(obj.get("code", "none")).strip().upper()
        return {"score": min(max(score, 0.0), 1.0), "verdict": verdict, "score_mode": mode,
                "code": code if code in SUBTYPES else "none",
                "reason": str(obj.get("reason", "")).strip()}
    return None


def classify(video_path, transcript: str, prompt_name: str = "P1") -> dict:
    """One VLM call (+1 retry on parse failure). score is None if both attempts fail."""
    prompt = fill_prompt(load_prompt(prompt_name), transcript)
    t0 = time.time()
    raw, p_yes = generate(video_path, prompt)
    parsed, retried = parse_json(raw, p_yes), False
    if parsed is None:
        # Greedy decoding would just repeat the same answer, so the retry samples
        # with the model card's recommended settings.
        retried = True
        raw, p_yes = generate(video_path, prompt, sample=True)
        parsed = parse_json(raw, p_yes)
    result = {"prompt": prompt_name, "score": None, "code": None, "reason": None}
    if parsed:
        result.update(parsed)
        if prompt_name == "P0":
            result["code"] = None      # P0 does not ask for a subtype
    result.update(parse_ok=parsed is not None, retried=retried,
                  sec=round(time.time() - t0, 2), raw=raw)
    return result


# ---------------------------------------------------------------- review pages
def frame_strip(path, n: int, height: int = 150) -> list:
    """n evenly spaced frames as base64 JPEG strings, for self-contained HTML review pages."""
    import base64
    dur = video_duration(path)
    frames = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                              "-vf", f"scale=-2:{height}", "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                             capture_output=True)
        if out.stdout:
            frames.append(base64.b64encode(out.stdout).decode())
    return frames


PAGE_CSS = """
:root { --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --line:#e7e5e4; --card:#ffffff; --br:#c2410c; --ok:#15803d; }
@media (prefers-color-scheme: dark) { :root { --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --line:#44403c; --card:#292524; --br:#fb923c; --ok:#4ade80; } }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
p.lead { color:var(--muted); margin:0 0 16px; max-width:70ch; }
.row { display:grid; grid-template-columns:230px 1fr; gap:12px; padding:10px; margin-bottom:8px;
       background:var(--card); border:1px solid var(--line); border-radius:8px; }
.meta b { font-size:15px; } .meta div { margin-top:2px; } .muted { color:var(--muted); }
.tag { display:inline-block; padding:1px 7px; border-radius:10px; border:1px solid var(--line); font-size:12px; margin-right:4px; }
.tag.br { border-color:var(--br); color:var(--br); } .tag.ok { border-color:var(--ok); color:var(--ok); }
.frames { display:flex; gap:4px; overflow-x:auto; } .frames img { height:150px; border-radius:4px; }
table { border-collapse:collapse; margin:4px 0 12px; } td, th { border:1px solid var(--line); padding:4px 10px; text-align:left; }
.fill { display:inline-block; min-width:160px; border-bottom:1px solid var(--muted); height:1.2em; }
a { color:inherit; }
@media (max-width:700px) { .row { grid-template-columns:1fr; } .frames img { height:110px; } }
"""
