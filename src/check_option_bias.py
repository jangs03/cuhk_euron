"""보기 위치 편향(option position bias) 진단 + 디바이어싱 prior 추정.

배경: VLM은 내용과 무관하게 특정 보기 글자(흔히 A/B)를 선호하는 경향이 있어
객관식 정확도를 깎는다. 이를 사전분포로 추정해 로짓에서 빼면 공짜로 정확도가 오른다.
(PriDe / contextual calibration — "Unexplored Flaws in Multiple-Choice VQA Evaluations",
 arXiv:2511.22341)

--decoding logits 로 검증을 한 번 돌리면 <out>.probs.csv 가 생기고, 이 스크립트가
① 모델 예측 분포 vs 정답 분포를 비교해 편향 크기를 보여주고
② run_baseline.py --option-prior 에 그대로 넣을 문자열을 출력한다. (GPU 불필요)

사용:
  python src/check_option_bias.py --probs val_pred_v9.csv.probs.csv \\
      --gold data/train_nonvisual_fused_prompt.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

SINGLE_CATEGORIES = {"single", "combination", "emotion", "object_interaction"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="<out>.probs.csv 경로")
    ap.add_argument("--gold", default="data/training_qa.csv")
    ap.add_argument("--n-options", type=int, default=4,
                    help="분석할 보기 개수 (4지선다만 분석. HARn single 3지선다는 제외)")
    args = ap.parse_args()

    probs = pd.read_csv(args.probs, dtype={"qa_id": str, "letter": str})
    probs = probs[probs["category"].isin(SINGLE_CATEGORIES)]
    probs = probs.drop_duplicates(subset=["qa_id", "letter"], keep="last")
    if probs.empty:
        raise SystemExit("probs 파일에 single류 카테고리 행이 없습니다")

    # 보기 개수가 --n-options인 문항만 (3지선다와 섞이면 편향 추정이 왜곡됨)
    sizes = probs.groupby("qa_id")["letter"].nunique()
    keep = sizes[sizes == args.n_options].index
    probs = probs[probs["qa_id"].isin(keep)]
    letters = sorted(probs["letter"].unique())
    print(f"분석 대상: {len(keep)}문항 ({args.n_options}지선다), 보기 {letters}\n")

    # ① 모델의 평균 예측 확률 = 사전 편향 추정치
    marginal = probs.groupby("letter")["prob"].mean()
    marginal = marginal / marginal.sum()
    # ② 실제 선택(argmax) 분포
    picked = probs.loc[probs.groupby("qa_id")["prob"].idxmax(), "letter"]
    picked_rate = picked.value_counts(normalize=True).reindex(letters).fillna(0)

    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    gold = gold[gold["qa_id"].isin(keep)].set_index("qa_id")["answer"].str.upper()
    gold_rate = gold.value_counts(normalize=True).reindex(letters).fillna(0)

    print(f"{'글자':>4} | {'모델 평균확률':>12} | {'모델 선택률':>10} | {'정답 비율':>9} | 편차")
    print("-" * 62)
    for L in letters:
        dev = picked_rate[L] - gold_rate[L]
        flag = " ⚠️" if abs(dev) > 0.05 else ""
        print(f"{L:>4} | {marginal[L]:11.3f} | {picked_rate[L]:9.3f} | "
              f"{gold_rate[L]:8.3f} | {dev:+.3f}{flag}")

    max_dev = float((picked_rate - gold_rate).abs().max())
    uniform = 1.0 / len(letters)
    print(f"\n최대 편차: {max_dev:.3f}  (정답 분포는 {'균등에 가까움' if gold_rate.std() < 0.05 else '불균등'})")

    if max_dev < 0.05:
        print("→ 편향이 작습니다. 디바이어싱 이득은 크지 않을 수 있습니다.")
    else:
        print("→ 편향이 있습니다. 아래 prior로 디바이어싱을 시도해 보세요.")

    # 디바이어싱 prior: 모델 마진 ÷ 정답 분포 (정답이 균등이면 모델 마진 그대로)
    prior = marginal / gold_rate.replace(0, uniform)
    prior = prior / prior.sum()
    s = ",".join(f"{L}:{prior[L]:.4f}" for L in letters)
    print(f"\n--option-prior '{s}'")
    print("\n검증 방법: 같은 설정으로 이 prior만 추가해 재실행 → 정확도 비교")
    print("주의: prior를 뽑은 셋(hold-out)에 그대로 적용하면 과적합 소지가 있습니다. "
          "개선폭이 작으면 채택 보류하세요.")


if __name__ == "__main__":
    main()
