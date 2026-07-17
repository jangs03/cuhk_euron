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
| `--model` | `Qwen/Qwen2.5-VL-3B-Instruct` | VRAM 16GB↑면 `Qwen/Qwen2.5-VL-7B-Instruct` 권장 |
| `--frames` | 8 | 클립당 입력 프레임 수 |
| `--seq-frames` | 16 | sequence 문항 전용 프레임 수 |
| `--multi-mode` | `binary` | multi 문항 처리: `binary`=보기별 yes/no 분해(권장), `joint`=한 번에 질문 |
| `--crop-person` | off | 배경 차분으로 사람 영역 crop — **v5 검증에서 전면 하락 (비권장, 기록용)** |
| `--sampling` | `auto` | 카테고리별 자동 선택: object_interaction·emotion=motion(keyframe), 나머지=uniform. motion 카테고리는 비디오가 있는 루트를 자동 우선 사용 |
| `--decoding` | `generate` | `logits`=자유 생성 대신 로그확률로 답 선택 (single류=글자 확률 비교, multi=P(YES)+threshold, sequence는 항상 generate). 보기별 확률이 `<out>.probs.csv`에 저장됨 |
| `--yes-threshold` | 0.5 | logits 디코딩에서 multi의 P(YES) 채택 기준 |
| `--quant` | `none` | `4bit`/`8bit` = bitsandbytes 양자화 (VRAM 절감용). 속도 목적이면 아래 AWQ 권장 |

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
