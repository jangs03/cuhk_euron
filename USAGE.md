# 코드 사용법 설명서 (USAGE)

각 스크립트의 역할·인자·사용 예시 정리. 설계 배경은 [PIPELINE.md](PIPELINE.md) 참고.

---

## 0. 설치

```bash
git clone https://github.com/jangs03/cuhk_euron.git
cd cuhk_euron
pip install -r requirements.txt
```

- GPU 필요 (Qwen2.5-VL 추론). 로컬 GPU가 없으면 **Colab 사용** → [notebooks/colab_baseline.ipynb](notebooks/colab_baseline.ipynb)를 열고 Run all (아래 6절).
- 로컬에서 랜덤 제출/채점/전처리만 할 거면 GPU 불필요.

## 1. 데이터 준비

다운로드한 대회 데이터를 `data/` 아래에 이렇게 배치하면 **모든 스크립트가 기본 인자로 동작**:

```
data/
├── training_qa.csv          ← Training/training_qa.csv 복사
├── test_qa.csv              ← Testing/test_qa.csv 복사
├── HARn/                    ← Training/data/HARn.zip 해제
├── HAU/                     ← Training/data/HAU.zip 해제
└── large_model_track_test/  ← Testing/data/large_model_track_test.zip 해제
```

다른 위치에 두려면 각 스크립트의 `--qa`, `--media-root`, `--harn-root`로 지정하면 된다.

> 클립 구조: `<클립>/<modality>/<modality>.mp4`, modality는 `IR` / `Depth_Color` / `Depth` (HAU는 `Thermal`도).
> **IR이 가장 선명해서 코드 기본값**. 자세한 비교는 PIPELINE.md 2절.

## 2. `make_random_submission.py` — 형식 확인용 랜덤 제출

제출 파이프라인이 정상인지 확인 + chance-level 파악용. GPU 불필요.

```bash
python src/make_random_submission.py                      # → random_submission.csv
python src/make_random_submission.py --qa data/test_qa.csv --out my_random.csv
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--qa` | `data/test_qa.csv` | 질문 csv |
| `--out` | `random_submission.csv` | 출력 파일 |

## 3. `preprocess_harn.py` — 미디어 전처리 (선택, 반복 실험 시 권장)

클립당 N프레임을 미리 JPEG로 캐시 + 클립 인덱스 csv 생성. HARn/HAU 모두 사용 가능.

```bash
python src/preprocess_harn.py --harn-root data/HARn --out data/cache/HARn \
    --index data/cache/harn_index.csv --frames 16 --workers 4
python src/preprocess_harn.py --harn-root data/HAU --out data/cache/HAU \
    --index data/cache/hau_index.csv
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--harn-root` | `data/HARn` | 압축 해제된 미디어 루트 |
| `--out` | `data/cache/HARn` | 프레임 캐시 출력 루트 (원본 구조 미러링) |
| `--index` | `data/cache/harn_index.csv` | 클립 인덱스 csv |
| `--frames` | 16 | 클립당 저장 프레임 수 |
| `--size` | 448 | 긴 변 리사이즈(px) |
| `--modality` | `IR` | 캐시할 modality (없는 클립은 자동 fallback) |
| `--colormap` | off | depth를 JET 컬러맵으로 변환 |
| `--workers` | 4 | 병렬 프로세스 수 |

끝나면 인덱스 csv에서 user/action 분포와 `error` 컬럼(깨진 클립)을 확인할 것.

## 4. `run_baseline.py` — 메인 추론 (VLM)

QA csv를 읽어 VLM으로 답을 예측하고 제출 csv를 만든다.

```bash
# 로컬 검증 (hold-out user 9, 24 / 우선 100문항만)
python src/run_baseline.py --qa data/training_qa.csv --out val_pred.csv \
    --val-users 9,24 --limit 100 --media-root data/cache,data

# 테스트 전체 추론 → 제출 파일
python src/run_baseline.py --qa data/test_qa.csv --out submission.csv \
    --media-root data/cache,data
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--qa` | `data/test_qa.csv` | 질문 csv |
| `--out` | `submission.csv` | 출력 (qa_id,prediction) |
| `--media-root` | `data` | 쉼표로 여러 개, **앞이 우선** (`data/cache,data` = 캐시 우선) |
| `--model` | `Qwen/Qwen2.5-VL-3B-Instruct` | HF의 VLM이면 대부분 그대로 교체 가능 (아래 모델 표 참고) |
| `--trust-remote-code` | off | InternVL 등 저장소 코드 실행이 필요한 모델용 |
| `--frames` | 8 | 클립당 입력 프레임 수 |
| `--seq-frames` | 16 | sequence 문항 전용 프레임 수 |
| `--multi-mode` | `binary` | multi 문항 처리: `binary`=보기별 yes/no 분해(권장), `joint`=한 번에 질문 |
| `--crop-person` | off | 배경 차분으로 사람 영역 crop — **v5 검증에서 전면 하락 (비권장, 기록용)** |
| `--sampling` | `auto` | 카테고리별 자동 선택: object_interaction·emotion=motion, **sequence=stratified(S1: 4구간 층화+구간별 모션 상위)**, 나머지=uniform. 비균등 샘플링 카테고리는 비디오가 있는 루트를 자동 우선 사용 |
| `--category` | (없음) | 쉼표로 카테고리 필터 — 빠른 검증용. 예: `--category sequence` (45문항, ~1분) |
| `--decoding` | `generate` | `logits`=자유 생성 대신 로그확률로 답 선택 (single류=글자 확률 비교, multi=P(YES)+threshold, sequence는 항상 generate). 보기별 확률이 `<out>.probs.csv`에 저장됨 |
| `--yes-threshold` | 0.5 | logits 디코딩에서 multi의 P(YES) 채택 기준 |
| `--seq-mode` | `stepwise` | sequence 답 결정 (아래 표 참고) |
| `--tta K` | 1 | **보기 순서 순열 TTA** — 원본 포함 K회 추론 후 집계. 위치 편향을 구조적으로 상쇄 (single류=확률 평균, sequence=다수결, multi(binary)는 순서 무관이라 미적용). 추론 K배 |
| `--quant` | `none` | `4bit`/`8bit` = bitsandbytes 양자화 (VRAM 절감용). 속도 목적이면 아래 AWQ 권장 |
| `--nonvisual` | (없음) | 센서 큐 지정. 전역 `imu,skeleton` 또는 **카테고리별** `emotion=imu,skeleton;multi=imu`. **`--qa`를 fused csv로 지정해야 함** |
| `--nonvisual-categories` | 전체 | 전역 지정 시 적용 카테고리 제한 (카테고리별 지정을 쓰면 불필요) |
| `--nonvisual-min-quality` | `partial` | 이 등급 미만 modality는 제외 (poor 데이터 차단) |
| `--nonvisual-dedup` | off | 센서 블록의 중복/무정보 문장 제거 (~7% 토큰 절감) |
| `--max-side` | 448 | 프레임 리사이즈 긴 변 (해상도 실험: 448 → 672) |
| `--ir-preprocess` | (없음) | IR 밝기 보정 manifest 경로 — 어두운 클립만 감마 보정 (아래 참고) |
| `--option-prior` | (없음) | 보기 위치 편향 제거 (logits 전용). `check_option_bias.py` 출력을 붙여넣기 |

### Non-visual 센서 큐 ablation (6종 실험)

`train/test_nonvisual_fused_prompt.csv`를 `--qa`로 쓰면 센서 큐 컬럼이 함께 로드됩니다.
**샘플링은 완전 결정적이라 모든 실험이 동일한 IR 프레임을 사용합니다** (비시각 입력만 변수).

```bash
BASE="--qa data/train_nonvisual_fused_prompt.csv --val-users 9,24 \
  --media-root data/cache,data --decoding logits"

python src/run_baseline.py $BASE --out val_visual.csv                              # ① Visual only
python src/run_baseline.py $BASE --out val_imu.csv       --nonvisual imu           # ② +IMU
python src/run_baseline.py $BASE --out val_radar.csv     --nonvisual radar         # ③ +Radar
python src/run_baseline.py $BASE --out val_skel.csv      --nonvisual skeleton      # ④ +Skeleton
python src/run_baseline.py $BASE --out val_imuskel.csv   --nonvisual imu,skeleton  # ⑤ +IMU+Skeleton
python src/run_baseline.py $BASE --out val_all.csv       --nonvisual imu,radar,skeleton  # ⑥ +All
```

결과를 한 표로 비교 (`--markdown`이면 노션용):

```bash
python src/compare_runs.py --gold data/train_nonvisual_fused_prompt.csv --markdown \
  --runs "Visual only=val_visual.csv" "Visual+IMU=val_imu.csv" "Visual+Radar=val_radar.csv" \
         "Visual+Skeleton=val_skel.csv" "Visual+IMU+Skeleton=val_imuskel.csv" "Visual+All=val_all.csv"
```

### 카테고리별로 다른 센서 조합 적용 (v7식 조건부 최적화)

`compare_runs.py`가 카테고리별 승자를 보고 **적용 명령을 자동 생성**합니다:

```
카테고리 조건부 적용 명령:
  --nonvisual "emotion=imu,skeleton;multi=imu;object_interaction=skeleton"
```

이 명령을 그대로 붙여 전체 검증을 한 번 더 돌려 실제 이득을 확인하세요.
명시되지 않은 카테고리(single 등)는 센서 큐가 붙지 않아 토큰도 절약됩니다.
`*=imu;emotion=imu,skeleton`처럼 기본값 + 예외 형태도 가능합니다.

### IR 밝기 manifest 생성 (`analyze_ir.py`)

클립별 밝기 통계를 분석해 적용할 감마를 미리 정한 표를 만듭니다. GPU 불필요.

```bash
python src/analyze_ir.py --roots data/HAU,data/HARn,data/large_model_track_test \
    --out data/cache/ir_preprocess_manifest.csv --workers 4
```

- 약 4,120클립, `--workers 4`로 **10~20분** 예상
- 중단돼도 다시 실행하면 이어서 진행 (resume), 200개마다 중간 저장
- 팀원 노트북과 **동일한 계산식·동일한 컬럼**이라 결과를 합쳐 써도 됨
- 끝나면 밝기 상태 분포와 "보정 적용 대상 비율"을 출력

### IR 밝기 전처리 (`--ir-preprocess`)

팀 공통 규격(`ir_gamma_v2`)의 사전 분석 결과를 읽어, **어두운 클립에만** 감마 보정을
추론 시점에 적용합니다. 샘플링 전략은 그대로 두고 **선택된 프레임에만** 적용됩니다.

```bash
python src/run_baseline.py ... --ir-preprocess data/cache/ir_preprocess_manifest.csv
```

| 규칙 | 동작 |
|---|---|
| `global_state == normal` 또는 `gamma >= 1.0` | **원본 유지** (보정 안 함) |
| 어두운 클립 | 감마 보정 + **밝은 영역 보호 마스크** (실측: 어두운 픽셀 +47, 밝은 픽셀 +1) |
| manifest에 없는 클립 | 원본 사용 (부분 커버리지 허용) — 실행 후 적용률 출력 |

필요 파일: `ir_preprocess_manifest.csv` (필수), `preprocess_config.json` (같은 폴더에 있으면 자동 사용)

### sequence 답 결정 방식 (`--seq-mode`)

| 모드 | 방식 | forward/문항 | 파싱 실패 |
|---|---|---|---|
| **`stepwise`** (기본) | "남은 보기 중 **가장 먼저** 일어난 것"을 3번 질의해 순서를 세움 | **3** | 없음 |
| `score` | 가능한 순열 24개를 전부 로그확률 채점 | 24 | 없음 |
| `generate` | 자유 생성 후 텍스트 파싱 (구버전) | 1 | **있음** |

`generate`는 모델이 `ANSWER: DBCA` 형식을 안 지키면 fallback으로 `ABCD`가 채워져
**무작위 수준(1/24 = 0.042)으로 붕괴**합니다. 실제로 LLaVA-OneVision이 0.044를 기록했습니다.

`stepwise`는 매 단계가 single류와 같은 단일 선택이라 기존 로짓 비교를 그대로 쓰고,
**항상 유효한 순열**이 나오며, `score` 대비 **8배 빠릅니다**.

### 카테고리 라우팅 (`route_predictions.py`)

모델마다 잘하는 카테고리가 다를 때, **카테고리별 승자의 답만 모아** 하나의 제출 파일로 합칩니다.
이미 만든 예측 csv를 후처리로 합치므로 **추가 추론이 없습니다**(GPU 0분).

```bash
# 검증셋에서 이득 확인 (--gold를 주면 단독 vs 라우팅 비교표 출력)
python src/route_predictions.py --out routed.csv --gold data/training_qa.csv \
    --map "*=val_qwen3.csv;sequence=val_base.csv"

# 이득이 있으면 같은 매핑을 테스트 예측에 적용
python src/route_predictions.py --out submission_routed.csv --qa data/test_qa.csv \
    --map "*=sub_qwen3.csv;sequence=sub_base.csv"
```

매핑에는 기본값 `*=<csv>`가 반드시 필요합니다. `compare_runs.py`가 검증 결과를 보고
**이 명령을 그대로 만들어 출력**하므로 복사해 쓰면 됩니다.

> ⚠️ 카테고리별 표본이 작아(27~99문항) 승자가 노이즈일 수 있습니다. 차이가 큰 카테고리만
> 라우팅하는 보수적 버전과 함께 비교하세요.

### 보기 위치 편향 진단 (`check_option_bias.py`)

VLM은 내용과 무관하게 특정 보기 글자를 선호합니다. logits 디코딩으로 검증을 돌린 뒤:

```bash
python src/check_option_bias.py --probs val_visual.csv.probs.csv \
    --gold data/train_nonvisual_fused_prompt.csv
# → 편차가 크면 출력된 --option-prior '...' 를 붙여 재실행 후 비교
```

### 모델 교체 (`--model`만 바꾸면 됨)

`vlm.py`가 `AutoModelForImageTextToText`로 자동 로드하므로 HF의 VLM 대부분이 코드 수정 없이 동작합니다.

| 모델 | 크기 | 특징 | 비고 |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | 현재 기준선 (v1~v9 튜닝 대상) | — |
| **`Qwen/Qwen3-VL-8B-Instruct`** | 8B | Qwen2.5-VL의 직계 후속, 비디오 이해 대폭 향상 | **1순위 시도** |
| `Qwen/Qwen3-VL-32B-Instruct` | 32B | 상용 모델급 성능 | A100 필요(+`--quant 4bit`) |
| `Qwen/Qwen3-VL-4B-Instruct` | 4B | T4에서도 여유 | 저사양용 |
| `llava-hf/llava-onevision-qwen2-7b-ov-hf` | 7B | **다중 이미지·비디오 특화**, SigLIP 인코더 | ~16GB |
| `google/gemma-3-12b-it` | 12B | **다른 계열**(Google) — 앙상블 다양성 최대 | ~24GB, HF 라이선스 동의 필요 |
| `google/gemma-3-4b-it` | 4B | 위의 경량판 | ~10GB |
| `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | 7B | 4bit, 속도 ~2배·VRAM 1/3 | `pip install autoawq` |

❌ **쓸 수 없는 모델**: InternVL(`internvl_chat`), Phi-4-multimodal(`phi4mm`)은 표준 chat
template 대신 자체 인터페이스(`model.chat()`, `<|image_N|>` 규약)를 써서 이 파이프라인과
맞지 않습니다. 억지로 로드하면 **가중치 0개로 실려 무작위 출력**이 나오므로 로더가 중단시킵니다.
쓰려면 모델별 전용 어댑터가 필요합니다.

⚠️ **Qwen3-VL은 4B / 8B / 32B만 존재합니다 (7B 없음).** 없는 ID를 주면 즉시 안내 후 중단됩니다.

**새 모델 첫 실행 시 반드시**: `--limit 5`로 먼저 확인하세요. 로드 로그
`[vlm] <모델> | <클래스> | dtype=... | attn=...`가 뜨고, 프레임이 실제로 모델에 전달되는지
자동 검사됩니다. 모델별 이미지 플레이스홀더 규약이 달라 **영상이 조용히 무시되면 점수가
왜곡**되므로, 그 경우 명시적으로 중단하고 전용 어댑터가 필요하다고 알려줍니다.
InternVL·Phi 계열은 저장소 코드 실행이 필요해 공식 저장소에 한해 자동 허용됩니다.

**속도 최적화 가이드** (효과 순, 조합 가능):

| 기법 | 방법 | 효과 | 비고 |
|---|---|---|---|
| 로짓 디코딩 | `--decoding logits` | single류 생성 64토큰 → forward 1회 | 정확도 개선 겸용 |
| **AWQ 4-bit** | `--model Qwen/Qwen2.5-VL-7B-Instruct-AWQ` + `pip install autoawq` | 디코딩 ~1.5-2×, VRAM ~1/3 (T4에서도 7B 가능) | **정확도 검증 필수** (보통 -1%p 이내) |
| FlashAttention-2 | `pip install flash-attn --no-build-isolation` (설치돼 있으면 자동 사용) | 긴 비전 시퀀스에서 attention 가속 | Colab 빌드 오래 걸릴 수 있음, 없으면 SDPA 자동 |
| bnb 4bit | `--quant 4bit` | VRAM ~1/4 (속도는 비슷하거나 소폭 감소) | T4/L4에서 큰 모델 올릴 때 |
| 프레임/해상도 축소 | `--frames 6`, max_side 축소 | 비전 토큰 수 비례 가속 | 정확도 트레이드오프 — 검증 필수 |

**threshold 튜닝** (`tune_yes_threshold.py`) — 재추론 없이 사이드카 확률만으로:

```bash
# 검증 probs로 최적 threshold 탐색
python src/tune_yes_threshold.py --probs val_pred_v8.csv.probs.csv --gold data/training_qa.csv
# 찾은 값을 테스트 예측에 적용 (GPU 불필요)
python src/tune_yes_threshold.py --probs submission_v8.csv.probs.csv --apply submission_v8.csv --threshold 0.35
```
| `--modality` | `IR` | 사용할 modality |
| `--val-users` | (없음) | 예: `9,24` — 해당 user 문항만 추론 (검증용) |
| `--limit` | 0 | 앞에서 N문항만 (디버그용) |
| `--colormap` | off | depth JET 컬러맵 |

**동작 특성:**
- **Resume**: `--out` 파일에 이미 있는 qa_id는 건너뜀 → 중단돼도 같은 명령 재실행하면 이어서 돈다.
  처음부터 다시 하려면 out 파일을 삭제.
- **설정 변경 감지**: 실행마다 `<out>.meta.json`에 설정을 기록하고, 같은 출력 파일에
  **다른 설정으로 이어 쓰려 하면 무엇이 바뀌었는지 보여주고 중단**합니다.
  (모델·TTA·해상도 등을 바꿨는데 EXP 태그를 안 바꾸면 옛 답이 재사용되어 실험이 조용히 무효가 되는 사고 방지.)
  의도한 재개라면 `--allow-config-change`.
- 클립 하나가 깨져도 죽지 않고 형식에 맞는 fallback 답을 쓰고 계속 진행 (마지막에 에러 수 출력).
- 답은 카테고리 형식으로 자동 보정됨 (single→한 글자, multi→글자 집합, sequence→ABCD 순열).

## 5. `evaluate.py` — 로컬 채점

정답이 있는 training_qa 기준으로 대회 규칙 그대로 채점 (multi=집합 일치, sequence=순서 일치).

```bash
python src/evaluate.py --pred val_pred.csv --gold data/training_qa.csv
```

출력: 전체 정확도 + source/category별 정확도 표. **category별 표에서 병목을 찾는 게 개선의 시작점.**

## 6. Colab에서 실행 (팀 표준 워크플로)

```
[로컬 VS Code] 코드 수정 → git push
        ↓
[Colab] notebooks/colab_baseline.ipynb → Runtime > Run all
        (pull → 데이터 다운로드 → 전처리 → 추론 → Drive에 submission 저장)
```

1. Colab에서 GitHub의 `notebooks/colab_baseline.ipynb` 열기 (런타임: **L4 이상**)
2. 셀 0에서 `DRIVE_OUT`/`MODEL`/`MODALITY` 확인 (T4면 MODEL을 3B로)
3. Run all → 완료 후 Drive `cuhk/submission.csv`를 Kaggle에 제출
4. VM이 리셋되면 다시 Run all (데이터 재다운로드), 코드만 바꿨으면 셀 2부터 재실행

Drive(`cuhk/`)에 저장되는 것: `submission.csv`, `val_pred.csv`,
**`cache.tar`(전처리 캐시 — 다음 세션에서 자동 복원되어 재전처리 생략)**, 클립 인덱스 csv 3개.
`MODALITY`를 바꿔 재전처리하려면 Drive의 `cache.tar`와 인덱스 csv를 지우고 재실행할 것.

## 7. 실험 방법 (예시)

```bash
# modality 비교: IR vs Depth_Color (검증 점수로 판단)
python src/run_baseline.py --qa data/training_qa.csv --out val_ir.csv --val-users 9,24 --modality IR
python src/run_baseline.py --qa data/training_qa.csv --out val_dc.csv --val-users 9,24 --modality Depth_Color
python src/evaluate.py --pred val_ir.csv
python src/evaluate.py --pred val_dc.csv

# 프레임 수 실험
python src/run_baseline.py ... --frames 16
```

- 프롬프트 수정: [src/prompts.py](src/prompts.py)만 건드리면 됨
- 모델 교체/API 추가: [src/vlm.py](src/vlm.py)의 `answer(frames, prompt) -> str` 인터페이스 구현

## 8. 자주 나는 문제

| 증상 | 원인/해결 |
|---|---|
| `media not found: ...` | `data/` 배치가 1절과 다름 → `--media-root`로 실제 위치 지정 |
| CUDA out of memory | `--model Qwen/Qwen2.5-VL-3B-Instruct`로 낮추거나 `--frames` 줄이기 |
| 추론이 너무 느림 | 전처리 캐시 사용 (`--media-root data/cache,data`), `--limit`으로 먼저 소규모 확인 |
| 에러 수(`errors/fallbacks`)가 많음 | 전처리 인덱스 csv의 `error` 컬럼으로 어떤 클립이 깨졌는지 확인 |
| 검증 점수가 리더보드와 크게 다름 | val user에 과적합됐을 수 있음 → hold-out user를 바꿔 재확인 |
