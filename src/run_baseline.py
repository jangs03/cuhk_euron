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
import math
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
    ap.add_argument("--decoding", choices=["generate", "logits"], default="generate",
                    help="logits=자유 생성 대신 보기 글자/YES 토큰의 로그확률로 답 선택. "
                         "single류는 글자 확률 비교, multi(binary)는 P(YES)+threshold. "
                         "sequence는 항상 generate. 확률은 <out>.probs.csv에 저장됨")
    ap.add_argument("--yes-threshold", type=float, default=0.5,
                    help="decoding=logits에서 multi의 P(YES) 채택 기준. "
                         "tune_yes_threshold.py로 검증셋에서 튜닝 가능")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                    help="bitsandbytes 양자화 (VRAM 절감용, T4 등 저사양 GPU). "
                         "속도가 목적이면 --model ...-AWQ 체크포인트 권장")
    ap.add_argument("--crop-person", action="store_true",
                    help="배경 차분으로 사람 활동 영역만 crop (고정 카메라 가정, "
                         "캐시 프레임에도 즉석 적용 가능) — v5 검증에서 성능 하락, 비권장")
    ap.add_argument("--sampling", choices=["uniform", "motion", "auto"], default="auto",
                    help="auto=카테고리별 자동 선택 (v6 검증 결과: object_interaction/emotion만 "
                         "motion, 나머지 uniform). motion 카테고리는 비디오가 있는 루트를 "
                         "자동으로 우선 사용")
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
    model = load_model(args.model,
                       quant=None if args.quant == "none" else args.quant)

    write_header = not out_path.exists()
    # logits 디코딩 시 확률 사이드카: threshold 재튜닝/앙상블에 재사용 (재추론 불필요)
    probs_path = out_path.with_suffix(out_path.suffix + ".probs.csv")
    probs_f = probs_writer = None
    if args.decoding == "logits":
        probs_header = not probs_path.exists()
        probs_f = open(probs_path, "a", newline="", encoding="utf-8")
        probs_writer = csv.writer(probs_f)
        if probs_header:
            probs_writer.writerow(["qa_id", "category", "letter", "prob"])

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
                # 카테고리별 샘플링 전략 (v6 검증: motion은 obj_interaction +7.4%p,
                # emotion +3.0%p / HAU single -11.1%p, multi -4.0%p)
                sampling = args.sampling
                if sampling == "auto":
                    sampling = ("motion" if category in ("object_interaction", "emotion")
                                else "uniform")

                # test path는 modality 파일을 직접 가리킴 → 원하는 modality로 교체 시도,
                # 해당 modality가 없는 클립이면 원본 경로로 fallback
                rel = data_utils.swap_modality(row["path"], args.modality)
                rel_candidates = [rel] if rel == str(row["path"]) else [rel, str(row["path"])]

                media = None
                if sampling == "motion":
                    # 모션 샘플링은 후보 프레임이 많아야 함 → 비디오가 있는 루트 우선
                    for rc in rel_candidates:
                        for root in media_roots:
                            try:
                                cand = data_utils.resolve_media(rc, root)
                            except FileNotFoundError:
                                continue
                            if data_utils.has_video(cand):
                                media = cand
                                break
                        if media is not None:
                            break
                if media is None:
                    try:
                        media = data_utils.resolve_media(rel, media_roots)
                    except FileNotFoundError:
                        media = data_utils.resolve_media(row["path"], media_roots)
                n_frames = args.seq_frames if category == "sequence" else args.frames
                frames, pos = data_utils.sample_frames(
                    media, n_frames, args.colormap, modality=args.modality,
                    crop_person=args.crop_person, sampling=sampling,
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

                use_logits = args.decoding == "logits" and category != "sequence"

                if category == "multi" and args.multi_mode == "binary":
                    if use_logits:
                        # 보기별 P(YES)를 뽑아 threshold로 채택 (하드 YES/NO보다 조절 가능)
                        probs = {L: model.yes_probability(
                                     frames, build_binary_prompt(opt, duration), times)
                                 for L, opt in options.items()}
                        for L, p in probs.items():
                            probs_writer.writerow([qa_id, category, L, f"{p:.4f}"])
                        probs_f.flush()
                        yes = sorted(L for L, p in probs.items()
                                     if p >= args.yes_threshold)
                        if not yes:  # 최소 1개: 가장 확률 높은 보기
                            yes = [max(probs, key=probs.get)]
                        ans = "".join(yes)
                    else:
                        # 보기별로 "영상에 등장하나?"를 따로 물어 yes인 것을 모은다
                        yes = [L for L, opt in options.items()
                               if parse_yes_no(model.answer(
                                   frames, build_binary_prompt(opt, duration), times))]
                        if yes:
                            ans = "".join(sorted(yes))
                        else:  # 전부 no면 joint 질문으로 fallback
                            prompt = build_prompt(str(row["question"]), options,
                                                  category, duration)
                            ans = parse_answer(model.answer(frames, prompt, times),
                                               category, letters)
                elif use_logits:
                    # single류: 보기 글자 토큰의 로그확률 직접 비교 (생성/파싱 노이즈 제거)
                    prompt = build_prompt(str(row["question"]), options, category, duration)
                    lp = model.option_logprobs(frames, prompt, letters, times)
                    z = max(lp.values())
                    total = sum(math.exp(v - z) for v in lp.values())
                    for L in letters:
                        probs_writer.writerow(
                            [qa_id, category, L, f"{math.exp(lp[L] - z) / total:.4f}"])
                    probs_f.flush()
                    ans = max(lp, key=lp.get)
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

    if probs_f is not None:
        probs_f.close()
        print(f"probs → {probs_path}  (threshold 재튜닝: src/tune_yes_threshold.py)")
    print(f"done → {out_path}  (errors/fallbacks: {n_err})")


if __name__ == "__main__":
    main()
