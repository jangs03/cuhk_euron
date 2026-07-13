"""메인 추론 스크립트: QA csv를 읽어 VLM으로 답을 예측하고 submission csv를 만든다.

사용 예:
  # 테스트 추론 → 제출 파일
  python src/run_baseline.py --qa data/test_qa.csv --out submission.csv

  # 로컬 검증 (hold-out user 9, 24만)
  python src/run_baseline.py --qa data/training_qa.csv --out val_pred.csv --val-users 9,24

중간에 끊겨도 --out 파일에 있는 qa_id는 건너뛰므로 재실행하면 이어서 돈다.
"""
import argparse
import csv
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm

import config
import data_utils
from parse_answer import parse_answer
from prompts import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=str(config.TEST_QA), help="qa csv 경로")
    ap.add_argument("--media-root", default=str(config.MEDIA_ROOT),
                    help="쉼표로 여러 개 가능. 예: data/cache,data (앞이 우선 = 캐시 우선)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=config.DEFAULT_NUM_FRAMES)
    ap.add_argument("--colormap", action="store_true", help="depth를 JET 컬러맵으로 변환")
    ap.add_argument("--modality", default="IR",
                    help="IR / Depth_Color / Depth / Thermal (없으면 선호 순서로 fallback). "
                         "실데이터 확인 결과 IR이 가장 선명해서 기본값.")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (디버그용)")
    ap.add_argument("--val-users", default="", help="예: 9,24 — 이 user들 행만 추론")
    args = ap.parse_args()

    df = data_utils.load_qa(args.qa)
    media_roots = [Path(r.strip()) for r in args.media_root.split(",") if r.strip()]

    if args.val_users:
        users = {int(u) for u in args.val_users.split(",")}
        df = df[df["path"].map(data_utils.extract_user).isin(users)]
        print(f"val users {sorted(users)}: {len(df)} rows")

    if args.limit:
        df = df.head(args.limit)

    # resume: 기존 out 파일에 있는 qa_id는 건너뜀
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path)["qa_id"].astype(str))
        print(f"resume: {len(done)} already answered")

    from vlm import load_model
    model = load_model(args.model)

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["qa_id", "answer"])

        n_err = 0
        for _, row in tqdm(df.iterrows(), total=len(df)):
            qa_id = str(row["qa_id"])
            if qa_id in done:
                continue
            category = str(row["category"]).strip()
            options = data_utils.get_options(row)
            letters = list(options.keys())
            try:
                # test path는 modality 파일을 직접 가리킴 → 원하는 modality로 교체 시도,
                # 해당 modality가 없는 클립이면 원본 경로로 fallback
                rel = data_utils.swap_modality(row["path"], args.modality)
                try:
                    media = data_utils.resolve_media(rel, media_roots)
                except FileNotFoundError:
                    media = data_utils.resolve_media(row["path"], media_roots)
                frames = data_utils.sample_frames(
                    media, args.frames, args.colormap, modality=args.modality)
                prompt = build_prompt(str(row["question"]), options, category)
                raw = model.answer(frames, prompt)
                ans = parse_answer(raw, category, letters)
            except Exception:
                n_err += 1
                traceback.print_exc()
                ans = parse_answer("", category, letters)  # 형식에 맞는 fallback
            writer.writerow([qa_id, ans])
            f.flush()

    print(f"done → {out_path}  (errors/fallbacks: {n_err})")


if __name__ == "__main__":
    main()
