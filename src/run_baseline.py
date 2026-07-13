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
from parse_answer import parse_answer, parse_yes_no
from prompts import build_binary_prompt, build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=str(config.TEST_QA), help="qa csv 경로")
    ap.add_argument("--media-root", default=str(config.MEDIA_ROOT),
                    help="쉼표로 여러 개 가능. 예: data/cache,data (앞이 우선 = 캐시 우선)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=config.DEFAULT_NUM_FRAMES)
    ap.add_argument("--seq-frames", type=int, default=16,
                    help="sequence 문항 전용 프레임 수 (순서 판단에 더 많은 프레임 필요)")
    ap.add_argument("--multi-mode", choices=["binary", "joint"], default="binary",
                    help="multi 문항: binary=보기별 yes/no 분해(권장), joint=한 번에 질문")
    ap.add_argument("--crop-person", action="store_true",
                    help="배경 차분으로 사람 활동 영역만 crop (고정 카메라 가정, "
                         "캐시 프레임에도 즉석 적용 가능) — v5 검증에서 성능 하락, 비권장")
    ap.add_argument("--sampling", choices=["uniform", "motion"], default="uniform",
                    help="motion=모션 에너지 기반 keyframe 샘플링 "
                         "(원본 비디오에서 후보 4n장을 읽으므로 --media-root data 권장)")
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
            # Kaggle 채점기는 'prediction' 컬럼명을 요구함 (설명 페이지의 'answer' 예시는 오류)
            writer.writerow(["qa_id", "prediction"])

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
                n_frames = args.seq_frames if category == "sequence" else args.frames
                frames, pos = data_utils.sample_frames(
                    media, n_frames, args.colormap, modality=args.modality,
                    crop_person=args.crop_person, sampling=args.sampling,
                    return_pos=True)

                # 클립 길이/타임스탬프: 검증 결과 emotion(+5%p)·multi에만 도움이 되고
                # single/sequence에는 노이즈였음 → 해당 카테고리에만 적용 (v4)
                duration, times = None, None
                if category in ("emotion", "multi"):
                    # 캐시(이미지 dir)에는 길이 정보가 없으니 원본 루트에서 시도
                    duration = data_utils.get_duration(media, args.modality)
                    if duration is None:
                        for root in media_roots:
                            try:
                                orig = data_utils.resolve_media(row["path"], root)
                            except FileNotFoundError:
                                continue
                            duration = data_utils.get_duration(orig, args.modality)
                            if duration:
                                break
                    if duration and len(frames) > 1:
                        # 실제 샘플 위치 기반 타임스탬프 (motion 샘플링은 비균등)
                        times = [p * duration for p in pos]

                if category == "multi" and args.multi_mode == "binary":
                    # 보기별로 "영상에 등장하나?"를 따로 물어 yes인 것을 모은다
                    yes = [L for L, opt in options.items()
                           if parse_yes_no(model.answer(
                               frames, build_binary_prompt(opt, duration), times))]
                    if yes:
                        ans = "".join(sorted(yes))
                    else:  # 전부 no면 joint 질문으로 fallback
                        prompt = build_prompt(str(row["question"]), options, category, duration)
                        ans = parse_answer(model.answer(frames, prompt, times), category, letters)
                else:
                    prompt = build_prompt(str(row["question"]), options, category, duration)
                    raw = model.answer(frames, prompt, times)
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
