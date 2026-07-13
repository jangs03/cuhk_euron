"""로컬 채점: 예측 csv vs 정답이 있는 training_qa.csv.

카테고리별 규칙:
  single/combination/emotion/object_interaction — 정확 일치
  multi    — 글자 집합이 같아야 정답 (순서 무시, 부분점수 없음)
  sequence — 글자 순서까지 정확히 일치

사용:
  python src/evaluate.py --pred val_pred.csv --gold data/training_qa.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import config


def normalize(ans: str) -> str:
    return "".join(ch for ch in str(ans).upper() if ch in "ABCD")


def is_correct(pred: str, gold: str, category: str) -> bool:
    pred, gold = normalize(pred), normalize(gold)
    if category == "multi":
        return set(pred) == set(gold) and len(pred) == len(set(pred))
    return pred == gold  # single류 및 sequence는 정확 일치


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default=str(config.TRAIN_QA))
    args = ap.parse_args()

    pred = pd.read_csv(args.pred, dtype=str)
    pred_col = "prediction" if "prediction" in pred.columns else "answer"
    pred = pred.rename(columns={pred_col: "pred"})[["qa_id", "pred"]]
    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    merged = pred.merge(
        gold[["qa_id", "source", "category", "answer"]], on="qa_id",
    )
    if merged.empty:
        raise SystemExit("no overlapping qa_id between pred and gold")

    merged["correct"] = [
        is_correct(p, g, c)
        for p, g, c in zip(merged["pred"], merged["answer"], merged["category"])
    ]

    print(f"overall accuracy: {merged['correct'].mean():.4f}  (n={len(merged)})\n")
    print("by source/category:")
    print(
        merged.groupby(["source", "category"])["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "acc"})
        .round(4)
    )


if __name__ == "__main__":
    main()
