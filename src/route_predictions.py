"""카테고리별로 다른 모델의 예측을 골라 하나의 제출 파일로 합친다 (추가 추론 불필요).

모델마다 잘하는 카테고리가 다를 때, 카테고리별 승자의 답만 모아 쓰는 방식.
이미 만들어진 예측 csv들을 후처리로 합치므로 GPU가 필요 없고 몇 초면 끝난다.

사용:
  # 검증셋에서 라우팅 효과 확인 (--gold를 주면 채점까지)
  python src/route_predictions.py --out routed.csv --gold data/training_qa.csv \\
      --map "*=val_qwen3.csv;sequence=val_base.csv"

  # 테스트 제출 파일 생성 (같은 매핑을 test 예측에 적용)
  python src/route_predictions.py --out submission_routed.csv \\
      --map "*=sub_qwen3.csv;sequence=sub_base.csv"

매핑 형식:  "<category>=<csv>;<category>=<csv>;*=<기본 csv>"
  `*` 는 명시되지 않은 나머지 카테고리에 쓸 기본 파일 (필수).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from evaluate import is_correct

CAT_ORDER = ["single", "combination", "emotion", "multi", "sequence",
             "object_interaction"]


def parse_map(spec: str) -> dict[str, str]:
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--map 형식 오류: '{part}' (카테고리=파일.csv)")
        cat, path = part.split("=", 1)
        out[cat.strip()] = path.strip()
    if "*" not in out:
        raise SystemExit("--map 에 기본값 '*=<csv>' 가 필요합니다")
    return out


def load_pred(path: str) -> pd.Series:
    df = pd.read_csv(path, dtype=str)
    col = "prediction" if "prediction" in df.columns else "answer"
    return df.set_index("qa_id")[col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True,
                    help='"*=기본.csv;multi=qwen3.csv;sequence=base.csv"')
    ap.add_argument("--out", required=True, help="합쳐진 예측 csv 출력 경로")
    ap.add_argument("--qa", default="",
                    help="카테고리 정보를 가진 qa csv (기본: --gold 사용)")
    ap.add_argument("--gold", default="",
                    help="정답 csv — 주면 라우팅 전후 정확도를 비교 출력")
    args = ap.parse_args()

    mapping = parse_map(args.map)
    qa_path = args.qa or args.gold
    if not qa_path:
        raise SystemExit("--qa 또는 --gold 중 하나는 필요합니다 (카테고리 정보)")
    qa = pd.read_csv(qa_path, dtype=str, keep_default_na=False)

    preds = {}
    for cat, path in mapping.items():
        if not Path(path).exists():
            raise SystemExit(f"파일 없음: {cat} → {path}")
        preds[cat] = load_pred(path)

    # 카테고리별로 해당 파일의 답을 선택
    chosen, source, missing = [], [], 0
    for _, row in qa.iterrows():
        cat = str(row["category"]).strip()
        key = cat if cat in preds else "*"
        s = preds[key]
        qid = str(row["qa_id"])
        if qid in s.index:
            chosen.append(s[qid])
        else:                                  # 해당 파일에 없으면 기본 파일로
            base = preds["*"]
            chosen.append(base[qid] if qid in base.index else "A")
            missing += 1
        source.append(mapping[key])

    out = pd.DataFrame({"qa_id": qa["qa_id"], "prediction": chosen})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"라우팅 매핑:")
    for cat in list(mapping):
        n = sum(1 for c in qa["category"] if str(c).strip() == cat) if cat != "*" else "-"
        print(f"  {cat:20s} → {Path(mapping[cat]).name}"
              + (f"  ({n}문항)" if cat != "*" else "  (나머지 전부)"))
    print(f"\n→ {args.out}  ({len(out)}행"
          + (f", 누락 {missing}건은 기본 파일 사용" if missing else "") + ")")

    if not args.gold:
        return

    # ---- 채점: 라우팅 vs 각 구성 파일 단독 ----
    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    gold = gold[["qa_id", "category", "answer"]]

    def score(series: pd.Series):
        m = gold[gold["qa_id"].isin(series.index)].copy()
        m["pred"] = m["qa_id"].map(series)
        m["ok"] = [is_correct(p, g, c)
                   for p, g, c in zip(m["pred"], m["answer"], m["category"])]
        return m.groupby("category")["ok"].mean(), float(m["ok"].mean()), len(m)

    routed = out.set_index("qa_id")["prediction"]
    cats = [c for c in CAT_ORDER if c in set(gold["category"])]
    rows = []
    for name, s in [(f"단독: {Path(p).name}", preds[c])
                    for c, p in mapping.items()] + [("**라우팅 결과**", routed)]:
        by_cat, overall, n = score(s)
        rows.append((name, [by_cat.get(c, float("nan")) for c in cats], overall, n))

    w = max(len(r[0]) for r in rows)
    print(f"\n{'':<{w}}  " + "  ".join(f"{c[:9]:>9}" for c in cats) + "     전체")
    print("-" * (w + 12 * len(cats) + 12))
    for name, vals, overall, n in rows:
        cells = "  ".join(f"{v:9.3f}" if v == v else f"{'-':>9}" for v in vals)
        print(f"{name:<{w}}  {cells}   {overall:.4f} (n={n})")

    best_single = max(r[2] for r in rows[:-1])
    gain = rows[-1][2] - best_single
    print(f"\n라우팅 이득: {gain:+.4f} (단독 최고 {best_single:.4f} 대비)")
    if gain <= 0:
        print("  → 이득 없음. 단독 최고 모델을 그대로 제출하세요.")
    else:
        print("  ⚠️ 카테고리별 표본이 작아(27~99문항) 과적합 소지가 있습니다.")
        print("     차이가 큰 카테고리만 라우팅하는 보수적 버전도 함께 비교해 보세요.")


if __name__ == "__main__":
    main()
