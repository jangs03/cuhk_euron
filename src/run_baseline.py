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
from prompts import build_binary_prompt, build_nonvisual_block, build_prompt


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
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="모델 저장소의 코드를 실행 (InternVL 등 일부 모델에 필요). "
                         "신뢰할 수 있는 공식 저장소에만 사용할 것")
    # ── non-visual 센서 큐 (fused csv 필요: *_nonvisual_fused_prompt.csv) ──
    ap.add_argument("--nonvisual", default="",
                    help="프롬프트에 주입할 센서 큐. 전역: 'imu,skeleton' / "
                         "카테고리별: 'emotion=imu,skeleton;multi=imu' "
                         "(명시 안 된 카테고리는 미적용). 빈 값이면 Visual only")
    ap.add_argument("--nonvisual-categories", default="",
                    help="전역 지정 시 적용할 카테고리 제한 (예: emotion,multi). "
                         "카테고리별 지정을 쓰면 불필요")
    ap.add_argument("--nonvisual-min-quality", choices=["good", "partial", "poor"],
                    default="partial",
                    help="이 등급 미만인 modality는 프롬프트에서 제외 (기본 partial)")
    ap.add_argument("--nonvisual-dedup", action="store_true",
                    help="센서 블록의 중복/무정보 문장 제거 (토큰 절약)")
    ap.add_argument("--max-side", type=int, default=448,
                    help="프레임 리사이즈 긴 변 픽셀 (해상도 실험용: 448 → 672 등)")
    ap.add_argument("--ir-preprocess", default="",
                    help="IR 밝기 전처리 manifest 경로 (ir_preprocess_manifest.csv). "
                         "어두운 클립에만 팀 공통 감마 보정을 적용. "
                         "manifest에 없는 클립은 원본 사용")
    ap.add_argument("--option-prior", default="",
                    help="보기 위치 편향 제거 (decoding=logits 전용). "
                         "예: 'A:0.31,B:0.24,C:0.23,D:0.22'. "
                         "src/check_option_bias.py 출력값을 그대로 붙여넣으면 됨")
    ap.add_argument("--crop-person", action="store_true",
                    help="배경 차분으로 사람 활동 영역만 crop (고정 카메라 가정, "
                         "캐시 프레임에도 즉석 적용 가능) — v5 검증에서 성능 하락, 비권장")
    ap.add_argument("--sampling", choices=["uniform", "motion", "stratified", "auto"],
                    default="auto",
                    help="auto=카테고리별 자동 선택: object_interaction/emotion=motion, "
                         "sequence=stratified(S1: 4구간 층화+구간별 모션 상위), 나머지 uniform. "
                         "비균등 샘플링 카테고리는 비디오가 있는 루트를 자동 우선 사용")
    ap.add_argument("--category", default="",
                    help="쉼표로 카테고리 필터 (빠른 검증용). 예: sequence 또는 multi,sequence")
    ap.add_argument("--colormap", action="store_true", help="depth를 JET 컬러맵으로 변환")
    ap.add_argument("--modality", default="IR",
                    help="IR / Depth_Color / Depth / Thermal (없으면 선호 순서로 fallback). "
                         "실데이터 확인 결과 IR이 가장 선명해서 기본값.")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (디버그용)")
    ap.add_argument("--val-users", default="", help="예: 9,24 — 이 user들 행만 추론")
    args = ap.parse_args()

    df = data_utils.load_qa(args.qa)
    media_roots = [Path(r.strip()) for r in args.media_root.split(",") if r.strip()]

    # non-visual 설정: fused csv에 큐 컬럼이 있어야 함
    from prompts import NONVISUAL_COLUMNS, modalities_for, parse_nonvisual_spec
    nv_spec = parse_nonvisual_spec(args.nonvisual)
    nv_cats = {c.strip() for c in args.nonvisual_categories.split(",") if c.strip()}
    if nv_spec:
        used = {m for mods in nv_spec.values() for m in mods}
        unknown = used - set(NONVISUAL_COLUMNS)
        if unknown:
            raise SystemExit(f"알 수 없는 modality: {sorted(unknown)} "
                             f"(가능: {sorted(NONVISUAL_COLUMNS)})")
        missing = [NONVISUAL_COLUMNS[m][0] for m in used
                   if NONVISUAL_COLUMNS[m][0] not in df.columns]
        if missing:
            raise SystemExit(
                f"--nonvisual {args.nonvisual} 인데 컬럼이 없습니다: {missing}\n"
                f"  → --qa 를 *_nonvisual_fused_prompt.csv 로 지정하세요")
        desc = ("; ".join(f"{k}={','.join(v)}" for k, v in nv_spec.items())
                if list(nv_spec) != ["*"] else ",".join(nv_spec["*"]))
        print(f"nonvisual: {desc} | quality>={args.nonvisual_min_quality}"
              f"{' | dedup' if args.nonvisual_dedup else ''}"
              f"{' | categories=' + str(sorted(nv_cats)) if nv_cats else ''}")

    ir_prep = None
    if args.ir_preprocess:
        from ir_preprocess import IRPreprocessor
        ir_prep = IRPreprocessor(args.ir_preprocess)
        print(f"IR 전처리: {len(ir_prep.by_key)}개 키 로드 "
              f"(version={ir_prep.config.get('version')})")

    option_prior = None
    if args.option_prior:
        option_prior = {kv.split(":")[0].strip().upper(): float(kv.split(":")[1])
                        for kv in args.option_prior.split(",") if ":" in kv}
        print(f"option prior (debias): {option_prior}")

    if args.val_users:
        users = {int(u) for u in args.val_users.split(",")}
        df = df[df["path"].map(data_utils.extract_user).isin(users)]
        print(f"val users {sorted(users)}: {len(df)} rows")

    if args.category:
        cats = {c.strip() for c in args.category.split(",") if c.strip()}
        df = df[df["category"].isin(cats)]
        print(f"category filter {sorted(cats)}: {len(df)} rows")

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
                       quant=None if args.quant == "none" else args.quant,
                       trust_remote_code=args.trust_remote_code)

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
                # emotion +3.0%p / HAU single -11.1%p, multi -4.0%p.
                # sequence는 S1 층화 motion — 구간 커버리지 + 구간별 대표 장면)
                sampling = args.sampling
                if sampling == "auto":
                    if category in ("object_interaction", "emotion"):
                        sampling = "motion"
                    elif category == "sequence":
                        sampling = "stratified"
                    else:
                        sampling = "uniform"

                # test path는 modality 파일을 직접 가리킴 → 원하는 modality로 교체 시도,
                # 해당 modality가 없는 클립이면 원본 경로로 fallback
                rel = data_utils.swap_modality(row["path"], args.modality)
                rel_candidates = [rel] if rel == str(row["path"]) else [rel, str(row["path"])]

                media = None
                if sampling in ("motion", "stratified"):
                    # 비균등 샘플링은 후보 프레임이 많아야 함 → 비디오가 있는 루트 우선
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
                # 샘플링은 완전 결정적(난수 없음) → 같은 클립·같은 인자면 항상 동일 프레임.
                # non-visual ablation의 "동일 IR frames" 조건이 코드로 보장됨.
                ir_record = (ir_prep.record_for(row["path"], args.modality)
                             if ir_prep is not None else None)
                frames, pos = data_utils.sample_frames(
                    media, n_frames, args.colormap, modality=args.modality,
                    crop_person=args.crop_person, sampling=sampling,
                    max_side=args.max_side, return_pos=True,
                    ir_prep=ir_prep, ir_record=ir_record)

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

                # non-visual 센서 큐 블록 (카테고리별로 다른 조합 가능)
                nv_block = ""
                if nv_spec and (not nv_cats or category in nv_cats):
                    mods = modalities_for(nv_spec, category)
                    if mods:
                        nv_block = build_nonvisual_block(
                            row, mods, args.nonvisual_min_quality, args.nonvisual_dedup)

                if category == "multi" and args.multi_mode == "binary":
                    if use_logits:
                        # 보기별 P(YES)를 뽑아 threshold로 채택 (하드 YES/NO보다 조절 가능)
                        probs = {L: model.yes_probability(
                                     frames, build_binary_prompt(opt, duration, nv_block),
                                     times)
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
                                   frames, build_binary_prompt(opt, duration, nv_block),
                                   times))]
                        if yes:
                            ans = "".join(sorted(yes))
                        else:  # 전부 no면 joint 질문으로 fallback
                            prompt = build_prompt(str(row["question"]), options,
                                                  category, duration, nv_block)
                            ans = parse_answer(model.answer(frames, prompt, times),
                                               category, letters)
                elif use_logits:
                    # single류: 보기 글자 토큰의 로그확률 직접 비교 (생성/파싱 노이즈 제거)
                    prompt = build_prompt(str(row["question"]), options, category,
                                          duration, nv_block)
                    lp = model.option_logprobs(frames, prompt, letters, times,
                                               prior=option_prior)
                    z = max(lp.values())
                    total = sum(math.exp(v - z) for v in lp.values())
                    for L in letters:
                        probs_writer.writerow(
                            [qa_id, category, L, f"{math.exp(lp[L] - z) / total:.4f}"])
                    probs_f.flush()
                    ans = max(lp, key=lp.get)
                else:
                    prompt = build_prompt(str(row["question"]), options, category,
                                          duration, nv_block)
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
    if ir_prep is not None:
        print(ir_prep.coverage())
    print(f"done → {out_path}  (errors/fallbacks: {n_err})")


if __name__ == "__main__":
    main()
