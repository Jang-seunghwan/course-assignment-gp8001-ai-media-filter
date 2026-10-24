"""Headless check of the site (spec ch.7): screenshots, console errors, P0/P1 toggle, on-screen text vs results.

    python -m http.server 8765 -d docs &          # or pass --url https://<user>.github.io/<repo>/
    python tools/check_site.py --url http://127.0.0.1:8765/ --out /tmp/site-shots

Needs `pip install playwright` and `playwright install chromium`.
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SIZES = {"1440": (1440, 900), "1024": (1024, 768), "390": (390, 844)}


def results(prompt):
    rows = (ROOT / "results" / f"{prompt}.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["clip_id"]: r for r in map(json.loads, filter(None, rows))}


def open_clip(page, idx):
    page.locator(f'#grid .tile[data-index="{idx}"]').click()
    page.wait_for_function("document.querySelector('#chip').classList.contains('ok') || "
                           "document.querySelector('#wcard').classList.contains('up')", timeout=6000)
    page.wait_for_timeout(350)                       # let the card finish sliding up


def verdicts(page, prompt):
    """Open every clip with the given prompt; return {id: (verdict, score text, reason text or None)}."""
    page.locator(f'.seg button[data-prompt="{prompt}"]').click()
    ids = page.evaluate("[...document.querySelectorAll('#grid .tile')].map(t => t.getAttribute('aria-label').split(' ')[2])")
    out = {}
    for i, cid in enumerate(ids):
        open_clip(page, i)
        if page.evaluate("document.querySelector('#wcard').classList.contains('up')"):
            sub = page.inner_text("#w-sub")
            out[cid] = ("warn", sub.split("score ")[1].split(" ")[0], page.inner_text("#w-reason"),
                        page.is_visible("#w-note"))
        else:
            out[cid] = ("pass", page.inner_text("#chip").split("· ")[1], None, False)
        page.keyboard.press("Escape")
        page.wait_for_selector("#viewer", state="hidden")
    badges = page.evaluate("document.querySelectorAll('#grid .badge').length")
    return out, badges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-text", action="store_true", help="only screenshots")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    problems = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, (w, h) in SIZES.items():
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            errors = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(args.url, wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(800)
            page.screenshot(path=args.out / f"{name}_full.png", full_page=True)
            for sec in ("top", "problem", "method", "results", "feed"):
                page.evaluate(f"document.getElementById('{sec}').scrollIntoView({{block: 'start'}})")
                page.wait_for_timeout(400)
                page.screenshot(path=args.out / f"{name}_{sec}.png")
            # viewer states: warning (c044), pass chip (c024), model error (c012), analyzing
            page.evaluate("document.getElementById('feed').scrollIntoView({block: 'start'})")
            idx = page.evaluate("Object.fromEntries([...document.querySelectorAll('#grid .tile')]"
                                ".map(t => [t.getAttribute('aria-label').split(' ')[2], t.dataset.index]))")
            page.locator(f'#grid .tile[data-index="{idx["c044"]}"]').click()
            page.wait_for_timeout(500)
            page.screenshot(path=args.out / f"{name}_analyzing.png")
            page.wait_for_selector("#wcard.up", timeout=6000)
            page.wait_for_timeout(400)
            page.screenshot(path=args.out / f"{name}_warning.png")
            page.keyboard.press("Escape")
            page.locator(f'#grid .tile[data-index="{idx["c024"]}"]').click()
            page.wait_for_selector("#chip.ok", timeout=6000)
            page.wait_for_timeout(150)
            page.screenshot(path=args.out / f"{name}_pass.png")
            page.keyboard.press("Escape")
            page.locator(f'#grid .tile[data-index="{idx["c012"]}"]').click()
            page.wait_for_selector("#wcard.up", timeout=6000)
            page.wait_for_timeout(400)
            page.screenshot(path=args.out / f"{name}_error.png")
            page.keyboard.press("Escape")
            page.wait_for_timeout(200)
            page.screenshot(path=args.out / f"{name}_grid_badges.png")
            if errors:
                problems.append(f"{name}: console errors {errors}")
            print(f"{name}: screenshots done, console errors: {len(errors)}")

            if name == "1440" and not args.skip_text:
                r = {"p0": results("P0"), "p1": results("P1")}
                shown = {}
                for prompt in ("p1", "p0"):
                    shown[prompt], badges = verdicts(page, prompt)
                    if badges != len(shown[prompt]):
                        problems.append(f"{prompt}: {badges} badges after viewing {len(shown[prompt])} clips")
                    for cid, (v, score, reason, note) in shown[prompt].items():
                        res = r[prompt][cid]
                        want = "warn" if res["score"] >= 0.5 else "pass"
                        if v != want or score != f"{res['score']:.2f}":
                            problems.append(f"{prompt} {cid}: shown {v} {score}, file {want} {res['score']:.4f}")
                        if reason is not None and reason != f"“{res['reason']}”":
                            problems.append(f"{prompt} {cid}: reason differs")
                        if note != (cid == "c012"):
                            problems.append(f"{prompt} {cid}: note shown={note}")
                page.locator('.seg button[data-prompt="p1"]').click()
                if page.evaluate("document.querySelectorAll('#grid .badge').length"):
                    problems.append("badges not cleared on toggle")
                changed = [c for c in shown["p1"] if shown["p1"][c][0] != shown["p0"][c][0]]
                print(f"verdict changes P1 -> P0: {len(changed)} clips: {changed}")
                if len(changed) < 3:
                    problems.append("fewer than 3 verdict changes")
                pair = page.locator("#pair .reason").all_inner_texts()
                if pair != [f"“{r['p0']['c020']['reason']}”", f"“{r['p1']['c020']['reason']}”"]:
                    problems.append("P0/P1 example reasons differ from results")
                n_warn = {k: sum(v[0] == "warn" for v in shown[k].values()) for k in shown}
                print(f"clips with a warning: P1 {n_warn['p1']}, P0 {n_warn['p0']}")
            page.close()
        browser.close()
    print("PROBLEMS:" if problems else "all checks passed", *problems, sep="\n  ")


if __name__ == "__main__":
    main()
