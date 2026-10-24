"""labels.csv + results -> results/metrics.md (+ results/errors.csv)

    python 03_evaluate.py                 # test split, threshold 0.5 -> results/metrics.md
    python 03_evaluate.py --split dev     # -> results/dev_metrics.md
"""
import argparse
import json

import pandas as pd

from common import LABELS_CSV, RESULTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test", choices=["test", "dev"])
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--prompts", nargs="+", default=["P0", "P1"])
args = ap.parse_args()
TH = args.threshold
PREFIX = "" if args.split == "test" else f"{args.split}_"

labels = pd.read_csv(LABELS_CSV, dtype=str).fillna("")
labels = labels[labels["split"] == args.split].copy()
if labels.empty or (labels["final_br"] == "").any():
    raise SystemExit(f"labels.csv: need split={args.split} rows with final_br filled in")
labels["final_br"] = labels["final_br"].astype(int)
labels["final_code"] = labels["final_code"].str.strip().str.upper()


def load_results(prompt):
    path = RESULTS_DIR / f"{PREFIX}{prompt}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows).drop_duplicates("clip_id", keep="last")
    df = labels.merge(df, on="clip_id", how="left")
    df["pred"] = (df["score"].fillna(0) >= TH).astype(int)    # no score counts as "not flagged"
    return df


def ratio(num, den):
    return f"{num / den:.3f}" if den else "–"


results = {p: df for p, df in ((p, load_results(p)) for p in args.prompts) if df is not None}
if not results:
    raise SystemExit("no results files found")

summary, errors = [], []
for prompt, df in results.items():
    y, p = df["final_br"], df["pred"]
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    summary.append({
        "프롬프트": prompt, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": ratio(tp, tp + fp), "Recall": ratio(tp, tp + fn), "Accuracy": ratio(tp + tn, len(df)),
        "파싱 실패": int(df["score"].isna().sum()),
        "재시도": int(df["retried"].fillna(False).astype(bool).sum()),
        "클립당 처리 시간(초)": f"{df['sec'].mean():.1f}",
        "서로 다른 점수 값": int(df["score"].round(3).nunique()),
    })
    for _, r in df[df["pred"] != df["final_br"]].iterrows():
        errors.append({"prompt": prompt, "clip_id": r["clip_id"], "type": "FP" if r["pred"] else "FN",
                       "kaggle_id": r["kaggle_id"], "final_br": r["final_br"], "score": r["score"],
                       "pred_code": r.get("code"), "reason": r.get("reason")})

mode = results[next(iter(results))]["score_mode"].dropna()
mode_txt = {"yesprob": "yes 토큰 확률", "verbal": "모델이 말한 점수"}.get(mode.iloc[0] if len(mode) else "", "?")
n_pos = int(labels["final_br"].sum())
lines = [f"# 결과 ({args.split} {len(labels)}개: sludge {n_pos}, non-sludge {len(labels) - n_pos} · "
         f"임계값 {TH} · 점수 = {mode_txt})", "",
         "## 혼동행렬", "", pd.DataFrame(summary).to_markdown(index=False), "",
         "- 점수가 임계값 이상이면 양성이다. 정답은 Kaggle의 수정된 라벨(Sludge=1)이다.",
         "- 파싱 실패는 재시도 1회 후에도 JSON을 못 읽은 클립이며, 양성이 아닌 것으로 셌다.",
         "- 처리 시간은 VLM 호출 1회(프레임 디코딩 포함)의 평균이다. 전사는 클립당 한 번만 하며 따로 잰 값은 "
         "`timing.md`에 있다.", ""]
if args.split == "test":
    lines += ["참고: Visual-Qwen의 정확도 96.67%. 같은 Kaggle test split(300개)으로 평가했지만 fine-tuning한 모델이고, "
              "우리 test는 라벨 감사에서 뒤집힌 클립을 뺐으므로 직접 비교가 아니다.", ""]

lines += ["## 하위 유형 일치율", "", "해당 없음(유형 1개). brainrot을 B1 화면 분할 sludge 한 유형으로 좁혔다.", "",
          "## 라벨러 불일치", "", "해당 없음. 정답은 Kaggle의 수정된 라벨이며 사람 라벨러가 없다 (DEVIATIONS.md).", ""]

edf = pd.DataFrame(errors)
if len(edf):
    edf["score"] = edf["score"].round(3)
    edf["pred_code"] = edf["pred_code"].fillna("–")
edf.to_csv(RESULTS_DIR / f"{PREFIX}errors.csv", index=False)
lines += ["## 틀린 클립", "", edf.drop(columns=["reason"]).to_markdown(index=False) if len(edf) else "없음", "",
          f"이유 문장까지 포함한 목록은 `{PREFIX}errors.csv`에 있다.", ""]

out = RESULTS_DIR / f"{PREFIX}metrics.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:14]))
print(f"wrote {out}")
