"""형식 확인용 랜덤 베이스라인: 카테고리별 유효한 형식의 랜덤 답 생성.

제출 파이프라인이 정상 동작하는지 확인하고, 리더보드의 chance-level을 가늠하는 용도.

사용:
  python src/make_random_submission.py [--qa data/test_qa.csv] [--out random_submission.csv]
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
import data_utils

rng = random.Random(42)


def random_answer(category: str, letters: list[str]) -> str:
    if category == "multi":
        k = rng.randint(1, len(letters))
        return "".join(sorted(rng.sample(letters, k)))
    if category == "sequence":
        perm = letters[:]
        rng.shuffle(perm)
        return "".join(perm)
    return rng.choice(letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=str(config.TEST_QA))
    ap.add_argument("--out", default="random_submission.csv")
    args = ap.parse_args()

    df = data_utils.load_qa(args.qa)
    rows = ["qa_id,answer"]
    for _, row in df.iterrows():
        letters = list(data_utils.get_options(row).keys())
        rows.append(f"{row['qa_id']},{random_answer(str(row['category']).strip(), letters)}")

    Path(args.out).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows) - 1} rows → {args.out}")


if __name__ == "__main__":
    main()
