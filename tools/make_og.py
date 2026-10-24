"""docs/assets/og.png (1200x630): the method figure from docs/index.html, for link previews and the README.

    python tools/make_og.py        # needs playwright + chromium; loads the same Google Fonts as the site
"""
import re
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
icons = re.search(r'<svg width="0" height="0".*?</svg>', html, re.S).group(0)
pipe = re.search(r'<svg class="pipe-svg".*?</svg>', html, re.S).group(0)

page = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Source+Code+Pro:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; width: 1200px; height: 630px; background: #f6f5f1; font-family: "Source Sans 3", sans-serif; color: #15171c; }}
.top {{ display: flex; justify-content: space-between; align-items: center; padding: 34px 44px 0; }}
.logo {{ display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 22px; }}
.logo i {{ width: 30px; height: 30px; border-radius: 9px; background: #2f3bd1; position: relative; }}
.logo i::after {{ content: ""; position: absolute; left: 8px; top: 7px; width: 14px; height: 16px; border: 3px solid #fff; border-radius: 3px; box-sizing: border-box; }}
.course {{ font-size: 15px; font-weight: 600; letter-spacing: 1.2px; color: #2f3bd1; }}
h1 {{ font-size: 40px; line-height: 1.1; letter-spacing: -1px; margin: 26px 44px 0; }}
.legend {{ display: flex; gap: 10px; align-items: center; font-size: 15px; color: #646a73; margin: 12px 44px 0; }}
.tag {{ font-size: 12px; font-weight: 700; letter-spacing: .6px; padding: 3px 8px; border-radius: 6px; }}
.tag.img {{ background: #eceefd; color: #2f3bd1; }} .tag.text {{ background: #e3f4f1; color: #0b8a78; margin-left: 10px; }}
.card {{ margin: 22px 44px 0; background: #fff; border: 1px solid #e2e0da; border-radius: 18px; padding: 16px 18px; }}
.card svg {{ width: 100%; height: auto; font-family: "Source Sans 3", sans-serif; }}
.card .mono {{ font-family: "Source Code Pro", monospace; }}
.cap {{ margin: 18px 46px 0; font-size: 18px; line-height: 1.5; color: #3a3f47; }}
.cap b {{ color: #0b8a78; }}
</style></head><body>{icons}
<div class="top"><div class="logo"><i></i>AI Media Filter</div><div class="course">GP8001 · GROUP K3 · COURSE PROJECT</div></div>
<h1>One clip, three inputs, one sentence back.</h1>
<div class="legend"><span class="tag img">IMAGE</span> what the model sees <span class="tag text">TEXT</span> what the model reads and writes</div>
<div class="card">{pipe}</div>
<p class="cap"><b>Text is used three times:</b> the clip’s speech goes in as a transcript, our definition goes in as prompt text, and the verdict comes out as a score with a one-sentence reason.</p>
</body></html>"""

with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "og.html"
    src.write_text(page, encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1200, "height": 630})
        pg.goto(src.as_uri(), wait_until="networkidle")
        pg.evaluate("document.fonts.ready")
        pg.screenshot(path=str(ROOT / "docs" / "assets" / "og.png"))
        browser.close()
print("wrote docs/assets/og.png")
