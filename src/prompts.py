"""카테고리별 프롬프트 생성. 모델이 'ANSWER: X' 형식으로 답하도록 강제한다.

검증 데이터 분석(2026-07) 반영:
- multi: 모델이 1개만 고르는 편향(예측 80%가 1글자, 정답 63%는 2글자 이상)
  → 보기별 개별 판정 + "보통 2개" 힌트
- emotion: 보기가 감정이 아니라 행동의 태도/속도(Urgently/Carefully/Calmly/Slowly/Quickly)
  → 움직임의 속도·스타일로 판단하도록 지시
- object_interaction: 미세 물체 구분(sponge/napkin/cloth)이라 행동 맥락 추론 유도
"""

CATEGORY_INSTRUCTIONS = {
    "single": (
        "Choose the ONE option that best describes the action performed. "
        "Reply with exactly one letter, e.g. 'ANSWER: B'."
    ),
    "combination": (
        "Choose the ONE option that best describes the combination of actions. "
        "Reply with exactly one letter, e.g. 'ANSWER: B'."
    ),
    "emotion": (
        "The options describe the MANNER or PACE of the person's movement "
        "(e.g. urgently, carefully, calmly, slowly, quickly) — not a facial emotion; "
        "the face is not clearly visible. Compare consecutive frames to judge how fast "
        "and in what style the person moves, then choose the ONE best option. "
        "Reply with exactly one letter, e.g. 'ANSWER: B'."
    ),
    "object_interaction": (
        "Identify the object the person is interacting with. Look closely at what is in "
        "the person's hands, and use the activity being performed to infer which object "
        "is most plausible. Choose the ONE best option. "
        "Reply with exactly one letter, e.g. 'ANSWER: B'."
    ),
    "multi": (
        "Check EACH option one by one: does that action appear anywhere in the video? "
        "Select ALL options that appear. Typically 2 of the options are correct "
        "(sometimes 1 or 3) — do not stop after finding just one. "
        "Reply with all correct letters concatenated, e.g. 'ANSWER: AC'."
    ),
    "sequence": (
        "The frames are numbered in temporal order. Determine when each of the four "
        "actions happens and order ALL FOUR options chronologically (earliest first). "
        "Reply with all four letters in chronological order, e.g. 'ANSWER: DBCA'."
    ),
}

SYSTEM_PROMPT = (
    "You are an expert at analyzing infrared/depth (privacy-preserving, non-RGB) videos "
    "of a single person doing everyday activities at home. The input frames are sampled "
    "uniformly from the clip and labeled 'Frame 1', 'Frame 2', ... in temporal order. "
    "Answer the multiple-choice question. "
    "End your reply with 'ANSWER: <letters>' and nothing after it."
)


def build_prompt(question: str, options: dict, category: str) -> str:
    opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    instruction = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["single"])
    return (
        f"Question: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"{instruction}"
    )
