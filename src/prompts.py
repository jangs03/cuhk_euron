"""카테고리별 프롬프트 생성. 모델이 'ANSWER: X' 형식으로 답하도록 강제한다."""

CATEGORY_INSTRUCTIONS = {
    "single": (
        "Choose the ONE best option. "
        "Reply with exactly one letter, e.g. 'ANSWER: B'."
    ),
    "combination": (
        "Choose the ONE option that best describes the combination of actions. "
        "Reply with exactly one letter, e.g. 'ANSWER: B'."
    ),
    "emotion": (
        "Choose the ONE option that best describes the person's emotion. "
        "Reply with exactly one letter, e.g. 'ANSWER: B'."
    ),
    "object_interaction": (
        "Choose the ONE best option about the object interaction. "
        "Reply with exactly one letter, e.g. 'ANSWER: B'."
    ),
    "multi": (
        "Multiple options may be correct. Select ALL correct options. "
        "Reply with all correct letters concatenated, e.g. 'ANSWER: ABD'."
    ),
    "sequence": (
        "Order ALL the options chronologically as they happen in the video. "
        "Reply with all four letters in chronological order, e.g. 'ANSWER: DBCA'."
    ),
}

SYSTEM_PROMPT = (
    "You are an expert at analyzing depth (privacy-preserving, non-RGB) videos of a "
    "single person doing everyday activities at home. The frames are sampled uniformly "
    "from the clip in temporal order. Answer the multiple-choice question. "
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
