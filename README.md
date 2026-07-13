# CUHK-X Large Model Track — Baseline

Kaggle CUHK-X competition (Large Model Track) 베이스라인.
Depth 비디오에 대한 객관식 VQA를 **로컬 VLM(Qwen2.5-VL) zero-shot**으로 푸는 파이프라인입니다.

> **팀원용 상세 설계 문서: [PIPELINE.md](PIPELINE.md)** (모듈별 설명, HARn 전처리, 검증 전략, 실험 로드맵)
> **코드 사용법 설명서: [USAGE.md](USAGE.md)** (스크립트별 인자·예시·트러블슈팅)

## 1. 파이프라인 설계

```
                ┌─────────────────────────────────────────────────┐
                │                    Data Layer                    │
                │  training_qa.csv / test_qa.csv  +  media(zip 해제)│
                └───────────────┬─────────────────────────────────┘
                                │ path 컬럼으로 클립 위치 해석
                                ▼
                ┌─────────────────────────────────────────────────┐
                │              Frame Sampling (OpenCV)             │
                │  HAU : HAU/data/Depth/<user>/<seq>/Depth.mp4     │
                │  HARn: harn/<action>/<user>/<seq>/ (dir → 자동탐색)│
                │  → 클립당 N프레임 균등 샘플링 (+선택: depth colormap) │
                └───────────────┬─────────────────────────────────┘
                                ▼
                ┌─────────────────────────────────────────────────┐
                │            Prompt Builder (category별)           │
                │  single/combination/emotion/object_interaction   │
                │    → "한 글자만"                                  │
                │  multi     → "해당하는 글자 전부 (예: ABD)"          │
                │  sequence  → "시간 순서대로 네 글자 (예: DBCA)"       │
                │  + "ANSWER: X" 형식 강제                          │
                └───────────────┬─────────────────────────────────┘
                                ▼
                ┌─────────────────────────────────────────────────┐
                │        VLM Inference (Qwen2.5-VL 3B/7B)          │
                │  프레임 N장 + 질문 + 보기 → greedy decoding          │
                └───────────────┬─────────────────────────────────┘
                                ▼
                ┌─────────────────────────────────────────────────┐
                │         Answer Parser (category별 검증/보정)       │
                │  - 유효 글자만 추출, 카테고리 형식 강제                │
                │  - sequence: ABCD의 순열로 보정                    │
                │  - 파싱 실패 시 안전한 fallback                     │
                └───────────────┬─────────────────────────────────┘
                                ▼
                        submission.csv (qa_id,prediction)
```

**로컬 검증(cross-subject 모사):** 테스트가 unseen user(10,11,25,26)이므로,
train user 중 일부(예: 9, 24)를 hold-out 하여 `training_qa.csv`로 로컬 정확도를 측정합니다.
`path`에서 user id를 파싱해 분리합니다.

## 2. 폴더 구조

```
cuhk_euron/
├── data/                     ← 대회 데이터 배치 (아래처럼 맞추면 기본 인자로 동작)
│   ├── training_qa.csv       (다운로드한 Training/training_qa.csv 복사)
│   ├── test_qa.csv           (Testing/test_qa.csv 복사)
│   ├── HARn/                 (Training/data/HARn.zip 해제)
│   │   └── <action>/<user>/<trial>/{IR,Depth,Depth_Color}/<modality>.mp4
│   ├── HAU/                  (Training/data/HAU.zip 해제 — HAU/<user>/<trial>/<modality>/...)
│   └── large_model_track_test/  (Testing/data/large_model_track_test.zip 해제)
├── src/
│   ├── config.py             경로/기본 하이퍼파라미터
│   ├── preprocess_harn.py    HARn 전처리 (프레임 캐시 + 클립 인덱스 csv)
│   ├── data_utils.py         CSV 로드, 미디어 경로 해석, 프레임 샘플링, user 추출
│   ├── prompts.py            카테고리별 프롬프트 생성
│   ├── parse_answer.py       모델 출력 → 형식에 맞는 답 (검증 + fallback)
│   ├── vlm.py                Qwen2.5-VL 래퍼 (백엔드 교체 가능)
│   ├── run_baseline.py       메인 추론 스크립트 → submission.csv
│   ├── evaluate.py           로컬 채점 (카테고리별 규칙 적용)
│   └── make_random_submission.py  형식 확인용 랜덤 제출 파일
└── requirements.txt
```

## 3. 실행 방법

### Colab에서 실행 (권장 워크플로)

```
[로컬 VS Code] 코드 수정 → git push
        ↓
[Colab GPU] notebooks/colab_baseline.ipynb 실행
            (git pull → 데이터 다운로드 → 전처리 → 추론 → 결과를 Drive에 저장)
```

[notebooks/colab_baseline.ipynb](notebooks/colab_baseline.ipynb)를 Colab에서 열고
셀 0의 `REPO_URL`만 본인 저장소로 바꾼 뒤 위에서부터 실행하면 됩니다.
모든 셀이 재실행 가능해서 VM이 리셋되면 그냥 다시 Run all 하면 되고,
submission은 Drive에 저장되므로 세션이 끊겨도 이어서(resume) 돌아갑니다.

### 로컬에서 실행

```bash
pip install -r requirements.txt

# 0) 데이터 배치: data/ 아래에 csv와 zip 해제 결과를 둔다

# 1) 형식 확인용 랜덤 제출 (제출 파이프라인 sanity check)
python src/make_random_submission.py

# 2) HARn 전처리 (선택이지만 권장): 프레임 캐시 + 클립 인덱스 생성 (기본 modality: IR)
python src/preprocess_harn.py --harn-root data/HARn --out data/cache/HARn --frames 16

# 3) 로컬 검증: hold-out user(9,24)에 대해 추론 + 채점
python src/run_baseline.py --qa data/training_qa.csv --out val_pred.csv --val-users 9,24 --limit 100 --media-root data/cache,data
python src/evaluate.py --pred val_pred.csv --gold data/training_qa.csv

# 4) 테스트 추론 → 제출 파일 (캐시 우선, 없으면 원본 사용)
python src/run_baseline.py --qa data/test_qa.csv --out submission.csv --media-root data/cache,data
```

- GPU 메모리가 작으면 `--model Qwen/Qwen2.5-VL-3B-Instruct`(기본값), 충분하면 `--model Qwen/Qwen2.5-VL-7B-Instruct`.
- 중간에 끊겨도 `--out` 파일에 이미 저장된 qa_id는 건너뛰고 이어서 실행됩니다(resume).
- `--frames 8`로 프레임 수 조절, `--colormap`으로 depth를 컬러맵으로 변환(가시성 향상 실험용).

## 4. 개선 아이디어 (베이스라인 이후)

1. **Modality 선택** — 실데이터 확인 결과 **IR이 가장 선명**(기본값). Depth_Color 병행 실험 가치 있음 (`--modality`).
2. **프레임 수/해상도 튜닝** — sequence 문항은 프레임을 늘리면(16~32) 순서 판단이 좋아짐.
2. **Few-shot** — training_qa에서 같은 category 예시 1~2개를 프롬프트에 포함.
3. **카테고리별 프롬프트 엔지니어링** — 특히 multi(부분점수 없음)와 sequence가 점수 병목.
4. **모델 앙상블 / self-consistency** — 온도 샘플링 k회 후 다수결.
5. **LoRA fine-tuning** — training_qa로 Qwen2.5-VL을 SFT (cross-subject 검증 필수).
6. **멀티모달 확장** — mmWave/IMU/Skeleton 공개 시 feature fusion.
7. **외부 API** — 규정상 허용되므로 Gemini/GPT-4o/Claude 등 상용 VLM으로 교체 가능 (`vlm.py`에 백엔드 추가).
