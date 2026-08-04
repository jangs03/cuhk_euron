"""여러 모델의 예측을 결합한다 (추가 추론 불필요).

probs 사이드카(`<pred>.probs.csv`)가 있으면 **확률을 평균**하고, 없으면 예측 문자열
**다수결**로 결합한다. 카테고리별 처리:
  single류  : 보기별 확률 평균 → argmax
  multi     : 보기별 P(YES) 평균 → threshold
  sequence  : 확률 형식이 달라 다수결 (동률이면 가장 강한 모델의 답)

약한 모델을 넣어도 되는지는 실력 격차·오류 상관에 달리므로, --weight accuracy 로
성능에 비례한 가중을 주고 조합을 바꿔가며 검증셋에서 비교하는 것을 권장한다.

사용:
  # 3모델 동일 가중
  python src/ensemble.py --out ens.csv --gold data/training_qa.csv \\
      --runs "Qwen3=val_qwen3.csv" "Gemma3=val_gemma.csv" "LLaVA=val_llavaov.csv"

  # 성능 비례 가중 (약한 모델 포함 실험)
  python src/ensemble.py --out ens4.csv --gold data/training_qa.csv --weight accuracy \\
      --runs "Qwen3=val_qwen3.csv" "Gemma3=val_gemma.csv" \\
             "LLaVA=val_llavaov.csv" "Qwen2.5=val_base.csv"
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from evaluate import is_correct

SINGLE_CATEGORIES = {"single", "combination", "emotion", "object_interaction"}
CAT_ORDER = ["single", "combination", "emotion", "multi", "sequence",
             "object_interaction"]


def load_run(path: str):
    """(예측 Series, 확률 dict) — 확률은 {qa_id: {letter: prob}}."""
    pred = pd.read_csv(path, dtype=str)
    col = "prediction" if "prediction" in pred.columns else "answer"
    series = pred.set_index("qa_id")[col]

    probs = {}
    p = Path(str(path) + ".probs.csv")
    if p.exists():
        df = pd.read_csv(p, dtype={"qa_id": str, "letter": str})
        df = df.drop_duplicates(subset=["qa_id", "letter"], keep="last")
        for qid, g in df.groupby("qa_id", sort=False):
            # sequence는 'step1:C' 형태라 단일 글자만 사용
            d = {L: v for L, v in zip(g["letter"], g["prob"])
                 if isinstance(L, str) and len(L) == 1}
            if d:
                probs[qid] = d
    return series, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help='"이름=예측.csv"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--qa", default="", help="카테고리 정보 csv (기본: --gold)")
    ap.add_argument("--gold", default="", help="정답 csv — 주면 단독 vs 앙상블 비교")
    ap.add_argument("--weight", choices=["equal", "accuracy"], default="equal",
                    help="accuracy=검증 정확도에 비례한 가중 (--gold 필요)")
    ap.add_argument("--yes-threshold", type=float, default=0.5,
                    help="multi에서 평균 P(YES) 채택 기준")
    args = ap.parse_args()

    qa_path = args.qa or args.gold
    if not qa_path:
        raise SystemExit("--qa 또는 --gold 가 필요합니다 (카테고리 정보)")
    qa = pd.read_csv(qa_path, dtype=str, keep_default_na=False)

    runs = {}
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f'--runs 형식 오류: "{spec}" (이름=경로.csv)')
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            raise SystemExit(f"파일 없음: {name} → {path}")
        runs[name] = load_run(path)

    # 결합 범위는 모든 구성원이 답한 문항으로 한정한다.
    # (qa csv 전체를 순회하면 예측 없는 문항이 임의값으로 채워져 점수가 무너진다)
    common = set.intersection(*(set(s.index) for s, _ in runs.values()))
    before = len(qa)
    qa = qa[qa["qa_id"].isin(common)]
    if qa.empty:
        raise SystemExit("모든 구성원이 공통으로 답한 문항이 없습니다 — 파일을 확인하세요")
    sizes = {n: len(s) for n, (s, _) in runs.items()}
    if len(set(sizes.values())) > 1:
        print("⚠️  구성원별 문항 수가 다릅니다:")
        for n, s in sizes.items():
            print(f"     {n:24s} {s:>5}문항")
    if len(qa) < before:
        print(f"결합 범위: {len(qa)}문항 (qa csv {before}문항 중 전원이 답한 것만)\n")

    gold = None
    if args.gold:
        gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
        gold = gold.set_index("qa_id")

    # ---- 가중치 결정 ----
    weights = {n: 1.0 for n in runs}
    if args.weight == "accuracy":
        if gold is None:
            raise SystemExit("--weight accuracy 에는 --gold 가 필요합니다")
        accs = {}
        for n, (series, _) in runs.items():
            ids = [i for i in series.index if i in gold.index]
            accs[n] = sum(is_correct(series[i], gold.loc[i, "answer"],
                                     gold.loc[i, "category"]) for i in ids) / max(len(ids), 1)
        best = max(accs.values())
        # 최고 모델을 1.0으로 두고 정확도 비율로 가중 (약한 모델은 목소리가 작아짐)
        weights = {n: (a / best) ** 3 for n, a in accs.items()}   # 세제곱: 격차 강조
        print("가중치 (검증 정확도 기반):")
        for n in runs:
            print(f"  {n:24s} acc={accs[n]:.4f}  weight={weights[n]:.3f}")
        print()

    n_prob, n_vote = 0, 0
    rows = []
    for _, r in qa.iterrows():
        qid, cat = str(r["qa_id"]), str(r["category"]).strip()
        letters = [L for L in "ABCD" if str(r.get(L, "")).strip()]

        # 확률을 가진 모델이 2개 이상이면 확률 평균, 아니면 다수결
        have = [(n, runs[n][1][qid]) for n in runs
                if qid in runs[n][1] and cat != "sequence"]
        if len(have) >= 2:
            agg = defaultdict(float)
            wsum = 0.0
            for n, d in have:
                w = weights[n]
                wsum += w
                for L in letters:
                    agg[L] += w * float(d.get(L, 0.0))
            for L in agg:
                agg[L] /= max(wsum, 1e-9)
            if cat == "multi":
                yes = sorted(L for L in letters if agg.get(L, 0) >= args.yes_threshold)
                ans = "".join(yes) if yes else max(agg, key=agg.get)
            else:
                ans = max(agg, key=agg.get)
            n_prob += 1
        else:                                   # 다수결 (동률이면 가중치 큰 쪽)
            votes = Counter()
            for n, (series, _) in runs.items():
                if qid in series.index:
                    votes[str(series[qid])] += weights[n]
            ans = votes.most_common(1)[0][0] if votes else "A"
            n_vote += 1
        rows.append({"qa_id": qid, "prediction": ans})

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"→ {args.out}  ({len(out)}행 | 확률평균 {n_prob} · 다수결 {n_vote})")

    if gold is None:
        return

    # ---- 채점: 각 단독 vs 앙상블 ----
    cats = [c for c in CAT_ORDER if c in set(qa["category"])]

    def score(series: pd.Series):
        # 공통 문항으로만 채점 — 구성원과 앙상블을 같은 기준으로 비교
        ids = [i for i in series.index if i in gold.index and i in common]
        ok = pd.DataFrame({
            "category": [gold.loc[i, "category"] for i in ids],
            "ok": [is_correct(series[i], gold.loc[i, "answer"],
                              gold.loc[i, "category"]) for i in ids]})
        return ok.groupby("category")["ok"].mean(), float(ok["ok"].mean()), len(ok)

    table = [(n, *score(runs[n][0])) for n in runs]
    table.append(("**앙상블**", *score(out.set_index("qa_id")["prediction"])))

    w = max(len(t[0]) for t in table)
    print(f"\n{'':<{w}}  " + "  ".join(f"{c[:9]:>9}" for c in cats) + "      전체")
    print("-" * (w + 12 * len(cats) + 12))
    for name, by_cat, overall, n in table:
        cells = "  ".join(f"{by_cat.get(c, float('nan')):9.3f}" for c in cats)
        print(f"{name:<{w}}  {cells}   {overall:.4f} (n={n})")

    best_single = max(t[2] for t in table[:-1])
    gain = table[-1][2] - best_single
    print(f"\n앙상블 이득: {gain:+.4f} (단독 최고 {best_single:.4f} 대비)")
    if gain <= 0:
        print("  → 이득 없음. 구성원을 줄이거나(약한 모델 제외) --weight accuracy 를 시도해 보세요.")
    else:
        print("  → 구성원을 바꿔가며(3모델 vs 4모델) 비교해 최적 조합을 찾으세요.")


if __name__ == "__main__":
    main()
