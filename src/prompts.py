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


def build_prompt(question: str, options: dict, category: str,
                 duration: float | None = None, nonvisual: str = "") -> str:
    opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    instruction = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["single"])
    dur_line = ""
    if duration:
        # 클립 길이는 행동 속도/태도 판단의 핵심 단서 (짧은 클립 = 서두른 동작)
        dur_line = f"(The full clip is {duration:.1f} seconds long; frames span it evenly.)\n"
    nv = f"{nonvisual}\n\n" if nonvisual else ""
    # 답 형식 지시(instruction)는 항상 마지막 — 형식 준수율이 가장 높은 위치
    return (
        f"{dur_line}"
        f"{nv}"
        f"Question: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"{instruction}"
    )


def build_sequence_step_prompt(question: str, remaining: dict,
                               placed: list[str], nonvisual: str = "") -> str:
    """sequence 단계별 질의 — 남은 보기 중 '가장 먼저 일어난 것' 하나를 고른다.

    4개 순열(24가지)을 통째로 채점하는 대신 3번의 단일 선택으로 순서를 세운다.
    한 번에 4개를 나열하게 하는 것보다 질문이 단순하고, 매 단계가 single류와
    같은 형태라 로짓 비교를 그대로 쓸 수 있다 (파싱 불필요, 항상 유효한 순열).
    """
    opts_text = "\n".join(f"{k}. {v}" for k, v in remaining.items())
    nv = f"{nonvisual}\n\n" if nonvisual else ""
    done = ""
    if placed:
        done = ("Already determined order so far: "
                + " -> ".join(placed) + "\n\n")
    return (
        f"{nv}"
        f"Question: {question}\n\n"
        f"{done}"
        f"Remaining options:\n{opts_text}\n\n"
        "Among the REMAINING options above, which one happens EARLIEST in the video? "
        "The frames are in temporal order. "
        "Reply with exactly one letter, e.g. 'ANSWER: B'."
    )


def build_binary_prompt(action: str, duration: float | None = None,
                        nonvisual: str = "") -> str:
    """multi 이진 분해용: 보기 하나가 영상에 등장하는지 yes/no로 묻는다."""
    dur_line = f"(The full clip is {duration:.1f} seconds long.)\n" if duration else ""
    nv = f"{nonvisual}\n\n" if nonvisual else ""
    return (
        f"{dur_line}"
        f"{nv}"
        f'Question: Does the action "{action}" appear at ANY point in this video?\n'
        "Check every frame carefully before deciding. "
        "Reply with 'ANSWER: YES' or 'ANSWER: NO' only."
    )


# ─────────────────────────── non-visual 센서 큐 ───────────────────────────
# fused csv(train/test_nonvisual_fused_prompt.csv)의 컬럼 매핑
NONVISUAL_COLUMNS = {
    "imu": ("imu_prompt_block", "imu_quality"),
    "radar": ("radar_prompt_block", "radar_quality"),
    "skeleton": ("skeleton_prompt_block", "sk_quality"),
}
NONVISUAL_HEADER = (
    "[NONVISUAL SENSOR CUES]\n"
    "These cues describe body motion and spatial change. "
    "They are supporting evidence, not action labels. "
    "If a cue conflicts with what you see in the frames, trust the frames."
)
_QUALITY_RANK = {"good": 3, "partial": 2, "poor": 1, "": 0}


def _dedup_lines(block: str) -> str:
    """중복/무정보 문장 제거 (--nonvisual-dedup).

    실측 확인된 문제: IMU 블록에 같은 뜻의 문장이 2개
    ("Body movement is strongest in the late part" ≒ "The strongest body movement
    occurs in the late part"), Skeleton의 'Reliability: good'은 94.7% 클립에서
    동일해 정보가 없다."""
    seen, out = set(), []
    for ln in block.split("\n"):
        s = ln.strip()
        if s.lower().startswith("- reliability: good"):
            continue  # 상수 라인 → 토큰 낭비
        # 의미 중복 판정: 어순/관사 무시한 단어 집합
        key = frozenset(w for w in s.lower().strip("-. ").split()
                        if w not in ("the", "a", "is", "of", "in", "occurs", "part"))
        if s.startswith("-"):
            if key in seen:
                continue
            seen.add(key)
        out.append(ln)
    return "\n".join(out)


def parse_nonvisual_spec(spec: str) -> dict[str, list[str]]:
    """--nonvisual 값을 카테고리별 modality 매핑으로 파싱.

    전역 지정:      "imu,skeleton"          → {"*": ["imu","skeleton"]}
    카테고리별 지정: "emotion=imu,skeleton;multi=imu;object_interaction=skeleton"
                   → {"emotion": [...], "multi": [...], "object_interaction": [...]}
                   (명시되지 않은 카테고리는 센서 큐 없음)
    두 형식을 섞을 수도 있다: "*=imu;emotion=imu,skeleton"
    """
    spec = (spec or "").strip()
    if not spec:
        return {}
    if "=" not in spec:
        mods = [m.strip().lower() for m in spec.split(",") if m.strip()]
        return {"*": mods} if mods else {}
    out: dict[str, list[str]] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--nonvisual 형식 오류: '{part}' (카테고리=modality 형태)")
        cat, mods = part.split("=", 1)
        out[cat.strip()] = [m.strip().lower() for m in mods.split(",") if m.strip()]
    return out


def modalities_for(spec_map: dict[str, list[str]], category: str) -> list[str]:
    """해당 카테고리에 적용할 modality 목록 (없으면 빈 리스트 = 센서 큐 미적용)."""
    if category in spec_map:
        return spec_map[category]
    return spec_map.get("*", [])


def build_nonvisual_block(row, modalities: list[str], min_quality: str = "partial",
                          dedup: bool = False) -> str:
    """fused csv 행에서 선택된 modality의 큐 블록을 조립.

    modalities: ["imu"], ["imu","skeleton"], ["imu","radar","skeleton"] 등
    min_quality: 이 등급 미만(기본 poor/빈값)이면 해당 modality 제외
    """
    floor = _QUALITY_RANK.get(min_quality, 2)
    parts = []
    for m in modalities:
        col, qcol = NONVISUAL_COLUMNS[m]
        block = str(row.get(col, "") or "").strip()
        qual = str(row.get(qcol, "") or "").strip().lower()
        if not block or _QUALITY_RANK.get(qual, 0) < floor:
            continue  # 품질 미달 → 프롬프트에서 제외 (팀 설계 원칙)
        parts.append(_dedup_lines(block) if dedup else block)
    if not parts:
        return ""
    return NONVISUAL_HEADER + "\n\n" + "\n\n".join(parts)
