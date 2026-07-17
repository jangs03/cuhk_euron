"""multi 문항의 P(YES) threshold 튜닝 — 재추론 없이 확률 사이드카만으로.

decoding=logits로 검증을 한 번 돌리면 <out>.probs.csv에 보기별 P(YES)가 저장된다.
이 스크립트는 threshold를 스윕해서 multi 정확도가 최대가 되는 값을 찾고,
--apply를 주면 그 threshold로 예측 csv의 multi 답을 다시 써준다 (GPU 불필요).

사용:
  # 1) 최적 threshold 탐색 (검증셋)
  python src/tune_yes_threshold.py --probs val_pred_v8.csv.probs.csv --gold data/training_qa.csv

  # 2) 찾은 threshold를 테스트 예측에 적용 (예: 0.35)
  python src/tune_yes_threshold.py --probs submission_v8.csv.probs.csv \
      --apply submission_v8.csv --threshold 0.35
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd


def answers_at(probs: pd.DataFrame, threshold: float) -> pd.Series:
    """qa_id별 P(YES)>=threshold인 글자 집합 (없으면 최고 확률 1개)."""
    def pick(g):
        yes = sorted(g.loc[g["prob"] >= threshold, "letter"])
        if not yes:
            yes = [g.loc[g["prob"].idxmax(), "letter"]]
        return "".join(yes)
    return probs.groupby("qa_id", sort=False).apply(pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="<out>.probs.csv 경로")
    ap.add_argument("--gold", default="data/training_qa.csv",
                    help="정답 csv (탐색 모드에서 사용)")
    ap.add_argument("--apply", default="",
                    help="예측 csv 경로 — 주면 --threshold로 multi 답을 재작성")
    ap.add_argument("--threshold", type=float, default=None,
                    help="--apply와 함께: 적용할 threshold")
    args = ap.parse_args()

    probs = pd.read_csv(args.probs, dtype={"qa_id": str, "letter": str})
    probs = probs[probs["category"] == "multi"].drop_duplicates(
        subset=["qa_id", "letter"], keep="last")
    if probs.empty:
        raise SystemExit("no multi rows in probs file")

    if args.apply:
        if args.threshold is None:
            raise SystemExit("--apply에는 --threshold가 필요합니다")
        pred = pd.read_csv(args.apply, dtype=str)
        col = "prediction" if "prediction" in pred.columns else "answer"
        new_ans = answers_at(probs, args.threshold)
        mask = pred["qa_id"].isin(new_ans.index)
        pred.loc[mask, col] = pred.loc[mask, "qa_id"].map(new_ans)
        pred.to_csv(args.apply, index=False)
        print(f"applied threshold={args.threshold} to {mask.sum()} multi rows "
              f"→ {args.apply}")
        return

    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    gold = gold.set_index("qa_id")["answer"].str.upper()

    print(f"multi questions: {probs['qa_id'].nunique()}")
    print(f"{'threshold':>9} | {'acc':>6} | mean #selected")
    best = (0.5, -1.0)
    for t in np.arange(0.05, 0.96, 0.05):
        ans = answers_at(probs, t)
        common = ans.index.intersection(gold.index)
        acc = float(np.mean([set(ans[q]) == set(gold[q]) for q in common]))
        n_sel = ans.str.len().mean()
        marker = ""
        if acc > best[1]:
            best = (float(t), acc)
            marker = " ◀"
        print(f"{t:9.2f} | {acc:6.4f} | {n_sel:.2f}{marker}")
    print(f"\nbest threshold = {best[0]:.2f}  (multi acc {best[1]:.4f})")
    print(f"적용: python src/tune_yes_threshold.py --probs <test의 probs.csv> "
          f"--apply <submission.csv> --threshold {best[0]:.2f}")


if __name__ == "__main__":
    main()
