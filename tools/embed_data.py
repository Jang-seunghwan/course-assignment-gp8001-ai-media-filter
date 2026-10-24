"""Copy docs/assets/site.json and clips.json into the fallback blocks of docs/index.html and the team line
of README.md. Run after editing site.json:

    python tools/embed_data.py

The live site reads the JSON files directly; the copies only matter when index.html is opened as a file
and for the README.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

site = json.loads((ASSETS / "site.json").read_text(encoding="utf-8"))
clips = json.loads((ASSETS / "clips.json").read_text(encoding="utf-8"))


def embed(html, block_id, data):
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    pattern = re.compile(rf'(<script type="application/json" id="{block_id}">)(.*?)(</script>)', re.S)
    assert pattern.search(html), block_id
    return pattern.sub(lambda m: m.group(1) + text + m.group(3), html)


index = ROOT / "docs" / "index.html"
html = embed(embed(index.read_text(encoding="utf-8"), "site-data", site), "clips-data", clips)
index.write_text(html, encoding="utf-8")

readme = ROOT / "README.md"
if readme.exists():
    team = f"{site['course']} · {site['group']} · " + ", ".join(site["members"])
    text = re.sub(r"(<!-- team:start -->\n).*?(\n<!-- team:end -->)", lambda m: m.group(1) + team + m.group(2),
                  readme.read_text(encoding="utf-8"), flags=re.S)
    readme.write_text(text, encoding="utf-8")
print(f"embedded site.json and {len(clips)} clips into docs/index.html; README team line updated")
