"""Upload demo: mp4 -> Whisper -> Qwen3-VL (P1 prompt) -> score bar, subtype, reason, warning banner.

    python app.py            # http://127.0.0.1:7860  (VS Code forwards the port automatically)
    python app.py --share    # temporary public gradio.live link
"""
import argparse
import html

# common must be imported before gradio: it sets HF_HOME / HF_HUB_OFFLINE, which the
# hub library (pulled in by gradio) reads only once at import time.
from common import SUBTYPES, classify, get_vlm, get_whisper, transcribe, trim_to_limit

import gradio as gr

DEFAULT_THRESHOLD = 0.7   # brainrot is a matter of taste, so start high and let the user lower it


def analyze(video_path):
    if not video_path:
        raise gr.Error("Upload an mp4 first.")
    clip = trim_to_limit(video_path)          # only the first 60 s are analysed
    transcript = transcribe(clip)
    res = classify(clip, transcript, "P1")
    res["transcript"] = transcript
    return res


def render(res, threshold):
    """Returns (score bar html, details markdown, banner group visibility, banner html)."""
    if not res:
        return "", "", gr.update(visible=False), ""
    if res["score"] is None:
        return ("<p>The model's answer could not be parsed.</p>",
                f"Raw output:\n```\n{res['raw']}\n```", gr.update(visible=False), "")
    s = res["score"]
    color = "#d9480f" if s >= threshold else "#2b8a3e"
    bar = f"""
    <div style="font-size:15px;margin-bottom:6px">Brainrot score <b>{s:.2f}</b>
      <span style="opacity:.7">(threshold {threshold:.2f})</span></div>
    <div style="position:relative;height:22px;border-radius:11px;background:#e9ecef;overflow:hidden">
      <div style="width:{s*100:.0f}%;height:100%;background:{color}"></div>
      <div style="position:absolute;top:0;left:{threshold*100:.0f}%;width:2px;height:100%;background:#212529"></div>
    </div>"""
    details = (f"**Type:** {res['code']} — {SUBTYPES.get(res['code'], '')}\n\n"
               f"**Reason:** {res['reason']}\n\n"
               f"<details><summary>Transcript / timing</summary>\n\n{res['transcript']}\n\n"
               f"({res['sec']:.1f} s VLM time)</details>")
    flagged = s >= threshold
    banner = (f"<div style='padding:14px;border-radius:10px;background:#fff4e6;border:2px solid #d9480f;"
              f"color:#862e0c;font-size:17px'>⚠️ <b>This looks like brainrot</b> "
              f"({html.escape(SUBTYPES.get(res['code'], ''))}). Score {s:.2f} is above your threshold.</div>")
    return bar, details, gr.update(visible=flagged), banner if flagged else ""


with gr.Blocks(title="AI Media Filter") as demo:
    gr.Markdown("# AI Media Filter\nUpload a short clip (≤60 s). A zero-shot video model "
                "(Qwen3-VL-4B) scores it against our brainrot definition, which this version narrows to "
                "one type: split-screen sludge (B1, two or more unrelated videos on one screen). "
                "This is a class demo, not a validated classifier.")
    state = gr.State(None)
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Clip (mp4)", sources=["upload"])
            threshold = gr.Slider(0, 1, value=DEFAULT_THRESHOLD, step=0.05, label="Warning threshold")
            run = gr.Button("Analyze", variant="primary")
        with gr.Column():
            with gr.Group(visible=False) as banner_box:
                banner = gr.HTML()
                with gr.Row():
                    skip = gr.Button("Skip it")
                    keep = gr.Button("Keep watching")
            bar = gr.HTML()
            details = gr.Markdown()

    outputs = [bar, details, banner_box, banner]
    run.click(analyze, video, state).then(render, [state, threshold], outputs)
    threshold.change(render, [state, threshold], outputs)       # no re-inference
    # The two buttons are mock-ups: they just dismiss the banner.
    for b in (skip, keep):
        b.click(lambda: gr.update(visible=False), None, banner_box)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    get_whisper(); get_vlm()                  # load both models before the first request
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
