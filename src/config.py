from pathlib import Path

# 프로젝트 루트 기준 경로 (src/ 의 부모)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRAIN_QA = DATA_DIR / "training_qa.csv"
TEST_QA = DATA_DIR / "test_qa.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

# 미디어 루트: csv의 path 컬럼이 이 디렉토리 기준 상대경로라고 가정
MEDIA_ROOT = DATA_DIR

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_NUM_FRAMES = 8
MAX_NEW_TOKENS = 64

LETTERS = ["A", "B", "C", "D"]
