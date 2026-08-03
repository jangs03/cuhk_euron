"""여러 실험 결과를 한 표로 비교 (ablation 표 자동 생성).

non-visual 6종 실험처럼 여러 예측 csv를 한 번에 채점하고, 기준(baseline) 대비
카테고리별 증감을 표로 출력한다. GPU 불필요.

사용:
  python src/compare_runs.py --gold data/train_nonvisual_fused_prompt.csv \\
      --runs "Visual only=val_v9_visual.csv" \\
             "Visual+IMU=val_v9_imu.csv" \\
             "Visual+Radar=val_v9_radar.csv" \\
             "Visual+Skeleton=val_v9_skel.csv" \\
             "Visual+IMU+Skeleton=val_v9_imu_skel.csv" \\
             "Visual+All=val_v9_all.csv"

첫 번째 run이 기준선이 된다. --markdown 을 주면 노션에 붙일 표로 출력.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from evaluate import is_correct

CAT_ORDER = ["single", "combination", "emotion", "multi", "sequence",
             "object_interaction"]


def score(pred_path: str, gold: pd.DataFrame) -> tuple[pd.Series, float, int]:
    pred = pd.read_csv(pred_path, dtype=str)
    col = "prediction" if "prediction" in pred.columns else "answer"
    m = pred[["qa_id", col]].rename(columns={col: "pred"}).merge(gold, on="qa_id")
    if m.empty:
        raise SystemExit(f"{pred_path}: gold와 겹치는 qa_id 없음")
    m["ok"] = [is_correct(p, g, c)
               for p, g, c in zip(m["pred"], m["answer"], m["category"])]
    by_cat = m.groupby("category")["ok"].mean()
    return by_cat, float(m["ok"].mean()), len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/training_qa.csv")
    ap.add_argument("--runs", nargs="+", required=True,
                    help='"이름=경로.csv" 형식. 첫 번째가 기준선')
    ap.add_argument("--markdown", action="store_true", help="노션용 마크다운 표로 출력")
    ap.add_argument("--all-rows", action="store_true",
                    help="문항 수가 달라도 각자 전체로 채점 (기본: 공통 문항으로만 비교)")
    args = ap.parse_args()

    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    gold = gold[["qa_id", "category", "answer"]]

    # 실행별 qa_id 수집 — 중단된 실행이 섞이면 비교가 왜곡되므로 먼저 확인
    loaded = {}
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f'--runs 형식 오류: "{spec}" (이름=경로.csv)')
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"⚠️  건너뜀 (파일 없음): {name} → {path}")
            continue
        ids = set(pd.read_csv(path, dtype=str)["qa_id"])
        loaded[name] = (path, ids)

    if not loaded:
        raise SystemExit("채점할 결과가 없습니다")

    sizes = {n: len(i) for n, (_, i) in loaded.items()}
    if len(set(sizes.values())) > 1:
        biggest = max(sizes.values())
        print("⚠️  문항 수가 다릅니다 (중단된 실행이 있을 수 있음):")
        for n, s in sizes.items():
            flag = "" if s == biggest else "  ← 미완료"
            print(f"     {n:28s} {s:>5}문항{flag}")
        if not args.all_rows:
            common = set.intersection(*(i for _, i in loaded.values()))
            print(f"   → 공통 {len(common)}문항으로만 비교합니다 "
                  f"(전체로 보려면 --all-rows)\n")
            gold = gold[gold["qa_id"].isin(common)]
        else:
            print("   → --all-rows: 각자 전체로 채점 (직접 비교는 부정확할 수 있음)\n")

    results, counts = {}, {}
    for name, (path, _ids) in loaded.items():
        by_cat, overall, n = score(path, gold)
        results[name] = by_cat
        counts[name] = (overall, n)

    if not results:
        raise SystemExit("채점할 결과가 없습니다")

    cats = [c for c in CAT_ORDER if any(c in r.index for r in results.values())]
    cat_n = gold["category"].value_counts().to_dict()   # 카테고리별 문항 수
    base_name = next(iter(results))
    base = results[base_name]

    header = ["실험"] + cats + ["전체", "vs 기준"]
    rows = []
    for name, by_cat in results.items():
        overall, n = counts[name]
        cells = [f"{by_cat.get(c, float('nan')):.3f}" if c in by_cat.index else "-"
                 for c in cats]
        delta = "기준" if name == base_name else f"{overall - counts[base_name][0]:+.4f}"
        rows.append([name] + cells + [f"**{overall:.4f}** (n={n})", delta])

    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
        print(" | ".join(h.ljust(w[i]) for i, h in enumerate(header)))
        print("-+-".join("-" * x for x in w))
        for r in rows:
            print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))

    # 카테고리별 최고 조합 — 카테고리 조건부 채택 판단용
    print("\n카테고리별 최고 조합 (조건부 적용 후보):")
    winners, oracle = {}, []
    for c in cats:
        vals = {n: r[c] for n, r in results.items() if c in r.index}
        best = max(vals, key=vals.get)
        winners[c] = best
        oracle.append(vals[best])
        gain = vals[best] - base.get(c, 0)
        mark = "" if best == base_name else f"  (기준 대비 {gain:+.3f})"
        print(f"  {c:20s} {best:24s} {vals[best]:.3f}{mark}")

    # 실험 이름에서 modality를 추론해 --nonvisual 스펙 문자열 생성
    def mods_of(name: str) -> list[str] | None:
        low = name.lower()
        if "all" in low:
            return ["imu", "radar", "skeleton"]
        found = [m for m in ("imu", "radar", "skeleton") if m in low]
        if found:
            return found
        return [] if ("only" in low or "visual" == low.strip()) else None

    spec_parts = []
    for c, name in winners.items():
        mods = mods_of(name)
        if mods is None:          # 이름에서 추론 불가 → 스펙 생략
            spec_parts = None
            break
        if mods:
            spec_parts.append(f"{c}={','.join(mods)}")
    if spec_parts:
        print(f"\n카테고리 조건부 센서 큐:\n  --nonvisual \"{';'.join(spec_parts)}\"")

    # ── 최종 승자 요약 — 셀 12/13에 그대로 붙여 쓸 수 있게 출력 ──
    best_name = max(counts, key=lambda n: counts[n][0])
    best_acc = counts[best_name][0]
    print(f"\n{'=' * 66}\n제출용 요약\n{'=' * 66}")
    print(f"단독 최고: {best_name}  ({best_acc:.4f})")

    # 카테고리 라우팅: 승자가 갈리는 카테고리만 매핑
    routed = {c: n for c, n in winners.items() if n != best_name}
    if routed:
        est = sum(results[winners[c]][c] * cat_n[c] for c in cats) / sum(cat_n.values())
        print(f"\n카테고리 라우팅 예상: {est:.4f}  (단독 최고 대비 {est - best_acc:+.4f})")
        pairs = [f"*={Path(loaded[best_name][0]).name}"]
        pairs += [f"{c}={Path(loaded[n][0]).name}" for c, n in routed.items()]
        print("  라우팅으로 바뀌는 카테고리:")
        for c, n in routed.items():
            gain = results[n][c] - results[best_name].get(c, 0)
            print(f"    {c:20s} → {n:24s} {results[n][c]:.3f} ({gain:+.3f}, {cat_n[c]}문항)")
        print(f"\n  python src/route_predictions.py --out routed.csv \\\\")
        print(f"      --gold <gold.csv> --map \"{';'.join(pairs)}\"")
        print("  ※ 표본이 작은 카테고리(<50문항)의 승자는 노이즈일 수 있습니다.")
    else:
        print("모든 카테고리에서 같은 설정이 최고 — 라우팅 불필요")


if __name__ == "__main__":
    main()
