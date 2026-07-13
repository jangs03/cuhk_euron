# CUHK-X Large Model Track — 파이프라인 설계 문서 (팀 공유용)

> Kaggle CUHK-X competition, Large Model Track 베이스라인의 전체 구조 설명.
> 실행 명령어 요약은 [README.md](README.md) 참고.

---

## 1. 대회 요약

| 항목 | 내용 |
|---|---|
| Task | privacy-preserving(depth, RGB 없음) 홈 액티비티 비디오에 대한 **객관식 VQA** |
| 벤치마크 | **HAU** (action understanding) + **HARn** (action reasoning) |
| 평가 | 전체 정확도 (문항별 동일 가중치, **부분점수 없음**) |
| 제출 | `qa_id,answer` 형식 csv |
| 규정 | 모델/API 제한 없음. 테스트 라벨 학습 금지, 테스트 수동 라벨링 금지 |
| Split | **cross-subject** — train: user 1–9, 16–24 / test: user 10, 11, 25, 26 |

### 문항 유형과 답 형식

| source | category | 답 형식 | 채점 |
|---|---|---|---|
| HAU | single / combination / emotion | 한 글자 (`A`) | 정확 일치 |
| HAU | multi | 정답 글자 전부 (`ABD`) | **집합** 일치 (순서 무시, 부분점수 0) |
| HAU | sequence | 시간순 네 글자 (`DBCA`) | **순서까지** 정확 일치 |
| HARn | single | 한 글자 — **보기 3개** (D 빈칸) | 정확 일치 |
| HARn | object_interaction | 한 글자 | 정확 일치 |

→ **multi와 sequence가 점수 병목**: multi는 부분 선택도 0점, sequence는 4! = 24 순열 중 하나를 정확히 맞혀야 함 (찍으면 4.2%).

---

## 2. 데이터 구조 (실제 다운로드 데이터 기준 — Kaggle 페이지 설명과 다름!)

```
Training/
├── training_qa.csv       # qa_id, source, path, category, question, A~D, answer
├── modality_list.csv     # 클립별 보유/누락 modality 목록
└── data/
    ├── HARn.zip  →  HARn/<action>/<user>/<trial>/<modality>/<modality>.mp4
    └── HAU.zip   →  HAU/<user>/<trial>/<modality>/<modality>.mp4
Testing/
├── test_qa.csv           # answer 대신 빈 prediction 컬럼
└── data/
    └── large_model_track_test.zip → large_model_track_test/<id>/<modality>/<modality>.mp4
```

- **Training의 `path` = 클립 디렉토리** (예: `HARn/0_Wash_face/user16/1-1-2`) — modality는 우리가 선택.
- **Testing의 `path` = modality 파일 직접 지정** (예: `.../LM_test_0066/Depth/Depth.mp4`) —
  코드가 `swap_modality`로 원하는 modality 경로로 바꿔 읽는다 (없으면 원본 fallback).
- 테스트 클립은 `LM_test_XXXX`로 익명화되어 user 추출 불가 (당연히 cross-subject).

### Modality별 품질 (실데이터 육안 확인 결과 — 중요!)

| Modality | 내용 | 품질 | HARn | HAU |
|---|---|---|:--:|:--:|
| **IR** | 적외선 | **사진 수준으로 선명 — 기본값** | ✓ | ✓ |
| Depth_Color | 컬러화된 depth | 양호 (형태/자세 구분 가능) | ✓(일부 없음) | ✓ |
| Depth | 8-bit 정규화 depth | 윤곽만 남음 — VLM 입력으로 비추천 | ✓ | ✓ |
| Thermal | 열화상 (25fps) | 미확인 | — | ✓ |

→ 코드 기본 modality는 **IR**, 없으면 Depth_Color → Depth 순 fallback (`data_utils.MODALITIES`).

---

## 3. 전체 파이프라인

```
 ┌──────────────────────────────────────────────────────────────┐
 │ (0) HARn 전처리  ·  src/preprocess_harn.py                    │
 │   harn/<action>/<user>/<seq> 스캔                             │
 │   → 클립 인덱스 csv (액션/유저/프레임수/해상도/에러)                │
 │   → 클립당 16프레임 샘플링 + 16bit→8bit 정규화 + 리사이즈           │
 │   → data/cache/harn/ 에 원본 구조 그대로 JPEG 캐시               │
 └──────────────────────┬───────────────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ (1) Data Layer  ·  src/data_utils.py                          │
 │   QA csv 로드 → path를 실제 미디어로 해석                         │
 │   media root 여러 개 지원: "data/cache,data" → 캐시 우선          │
 │   대소문자/하위폴더 변형에 관대한 경로 매칭 + glob fallback          │
 └──────────────────────┬───────────────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ (2) Frame Sampling  ·  src/data_utils.py                      │
 │   비디오(OpenCV) 또는 이미지 시퀀스에서 N프레임 균등 샘플링           │
 │   (기본 8, sequence 문항은 16+ 권장) · 긴 변 448 리사이즈          │
 │   옵션: --colormap (depth → JET 컬러맵, 가시성 실험)              │
 └──────────────────────┬───────────────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ (3) Prompt Builder  ·  src/prompts.py                         │
 │   시스템: "depth 비디오 프레임(시간순)" 컨텍스트 명시                │
 │   카테고리별 지시:                                              │
 │     single류 → "한 글자만"   multi → "정답 전부 (예: ABD)"        │
 │     sequence → "시간순 네 글자 (예: DBCA)"                       │
 │   출력 형식 강제: "ANSWER: <letters>"                           │
 └──────────────────────┬───────────────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ (4) VLM Inference  ·  src/vlm.py                              │
 │   기본: Qwen2.5-VL-3B/7B-Instruct, zero-shot, greedy           │
 │   인터페이스: answer(frames: list[PIL], prompt: str) -> str     │
 │   → 이 시그니처만 맞추면 다른 로컬 모델/상용 API로 교체 가능           │
 └──────────────────────┬───────────────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ (5) Answer Parser  ·  src/parse_answer.py                     │
 │   "ANSWER: ..." 라인에서 유효 글자 추출 ('A, C' 'A and C' 허용)    │
 │   카테고리별 형식 강제:                                          │
 │     single류 → 첫 유효 글자 1개 (HARn 3지선다는 A–C만)             │
 │     multi → 중복 제거 + 정렬  ·  sequence → ABCD 순열로 보정       │
 │   파싱 실패 시 형식에 맞는 fallback (제출 파일이 항상 유효하도록)      │
 └──────────────────────┬───────────────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ (6) Output  ·  src/run_baseline.py                            │
 │   submission.csv (qa_id,answer) · 행 단위 flush → resume 지원    │
 └──────────────────────────────────────────────────────────────┘
```

---

## 4. HARn 전처리 상세 (`src/preprocess_harn.py`)

클립 3,912개(HARn 3,098)를 추론 때마다 디코딩하면 느리다.
전처리에서 **한 번만 디코딩해서 JPEG 캐시**해 두면 이후 실험 iteration이 빨라진다.

처리 단계:

1. **클립 탐색** — 미디어를 직접 포함하는 leaf 디렉토리의 이름이 modality(`IR` 등)면
   그 부모를 클립 1개로 인식. 클립별 보유 modality 목록도 수집.
2. **modality 선택** — `--modality`(기본 IR) 우선, 없으면 선호 순서 fallback.
3. **프레임 읽기** — 선택한 modality의 mp4에서 균등 샘플링 (기본 16프레임).
   이미지 시퀀스 변형 구조도 지원하며, 16-bit 이미지는 percentile 정규화로 8-bit 변환.
4. **리사이즈 + 저장** — 긴 변 448px, JPEG q92,
   `data/cache/HARn/<action>/<user>/<trial>/<modality>/frame_XXX.jpg` (원본 구조 미러링).
5. **인덱스 생성** — 클립별 `rel_path, action, user, seq, modality, available(보유 modality),
   media_type, src_frames, saved_frames, width, height, error` 기록.
   → EDA, user 분포 확인, 깨진 클립 파악에 사용.

```bash
# 학습 미디어 (HARn과 HAU 모두 같은 구조라 둘 다 적용 가능)
python src/preprocess_harn.py --harn-root data/HARn --out data/cache/HARn --frames 16 --workers 4
python src/preprocess_harn.py --harn-root data/HAU  --out data/cache/HAU  --index data/cache/hau_index.csv
# 테스트 미디어
python src/preprocess_harn.py --harn-root data/large_model_track_test \
    --out data/cache/large_model_track_test --index data/cache/test_index.csv

# 이후 추론: 캐시 우선, 없으면 원본으로 fallback
python src/run_baseline.py --qa data/Testing/test_qa.csv --media-root data/cache,data --out submission.csv
```

전처리는 **선택 사항** — 실제 데이터가 클립당 mp4 하나라서 `run_baseline.py`가 원본에서
직접 읽어도 동작한다. 캐시는 반복 실험 속도용.

캐시가 원본 경로 구조를 미러링하므로 추론 코드는 캐시 존재 여부를 몰라도 된다
(`resolve_media`가 앞 루트부터 순서대로 시도).

---

## 5. 로컬 검증 전략 (cross-subject 모사)

테스트는 **학습에 없는 사람**(user 10, 11, 25, 26)이므로, 로컬 검증도 사람 단위로 나눠야
리더보드와 상관있는 숫자가 나온다.

- `path`에서 user id를 파싱 (`data_utils.extract_user`)
- train user 중 **9, 24를 hold-out** (필요시 변경) → 이 유저 문항만 추론 후 채점
- 채점(`src/evaluate.py`)은 대회 규칙 그대로: multi=집합 일치, sequence=순서 일치

```bash
python src/run_baseline.py --qa data/training_qa.csv --out val_pred.csv --val-users 9,24
python src/evaluate.py --pred val_pred.csv --gold data/training_qa.csv
# → 전체 + source/category별 정확도 출력
```

**주의**: 같은 유저의 문항으로 프롬프트를 튜닝하고 같은 유저로 검증하면 과적합.
프롬프트/하이퍼파라미터 실험은 hold-out 유저 점수로만 판단할 것.

---

## 6. 파일 맵

| 파일 | 역할 | 주요 수정 포인트 |
|---|---|---|
| `src/config.py` | 경로/기본값 | 데이터 위치, 기본 모델, 프레임 수 |
| `src/preprocess_harn.py` | HARn 전처리 + 인덱스 | depth 정규화 방식, 프레임 수 |
| `src/data_utils.py` | 로드/경로 해석/샘플링 | 실데이터 경로가 가정과 다를 때 |
| `src/prompts.py` | 프롬프트 | **프롬프트 엔지니어링은 여기만** |
| `src/vlm.py` | 모델 래퍼 | 모델 교체/API 백엔드 추가 |
| `src/parse_answer.py` | 출력 파싱 | 새 모델의 출력 습관 대응 |
| `src/run_baseline.py` | 추론 루프 | 배치/few-shot 등 구조 변경 |
| `src/evaluate.py` | 로컬 채점 | (대회 규칙 고정 — 수정 불필요) |
| `src/make_random_submission.py` | 랜덤 제출 | sanity check 용 |

---

## 7. 실험 로드맵 (우선순위 순)

1. **베이스라인 제출** — 랜덤 제출로 파이프라인 확인 → Qwen2.5-VL zero-shot 제출.
2. **카테고리별 진단** — `evaluate.py`의 category별 정확도로 병목 파악.
   예상 병목: sequence(순열 정확 일치), multi(부분점수 없음).
3. **modality 실험** — IR(기본) vs Depth_Color vs 두 modality 프레임을 함께 입력.
   HAU는 Thermal도 확인. `--modality` 플래그로 바로 실험 가능.
4. **프레임 수 실험** — sequence 문항만 16~32프레임으로 늘려보기 (category별 분기).
5. **프롬프트 개선** — IR/depth 영상 특성 설명, step-by-step 유도, 카테고리별 지시 다듬기.
6. **Few-shot** — training_qa에서 같은 category 예시 1~2개 삽입 (hold-out 유저 제외).
7. **Self-consistency** — temperature 샘플링 k회 → 다수결 (특히 multi/sequence).
8. **모델 업그레이드** — 7B → 72B(양자화) 또는 상용 API (Gemini/GPT-4o 등, 규정상 허용).
9. **LoRA SFT** — training_qa로 Qwen2.5-VL 미세조정. 반드시 hold-out 유저로 검증.
10. **멀티모달 fusion** — mmWave/IMU/Skeleton 공개 시 추가 입력으로 활용.

---

## 8. 규정 관련 주의사항

- 테스트셋 답을 학습에 쓰는 것 금지, 테스트 문항 수동 라벨링 금지.
- 공개 리더보드는 테스트 일부만 반영 — **public 점수에 과적합하지 말 것** (최종은 private).
- 외부 API 사용은 허용 (Large Model Track).
