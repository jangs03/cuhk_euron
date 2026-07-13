"""모델 출력 텍스트 → 카테고리 형식에 맞는 답으로 파싱/보정."""
import re

SINGLE_CATEGORIES = {"single", "combination", "emotion", "object_interaction"}


def _extract_letter_run(raw: str, valid: str) -> str:
    """'ANSWER: ...' 뒤의 한 줄에서 유효 글자 추출 ('A, C' 같은 구분자 허용).
    ANSWER 패턴이 없으면 본문에서 마지막으로 등장하는 유효 글자 뭉치를 쓴다."""
    raw = raw.upper()
    m = re.search(r"ANSWER\s*[:\-]?\s*([^\n]*)", raw)
    if m:
        # 유효 글자로만 이루어진 단어 토큰만 인정 → 'AND' 같은 단어의 A/D 오인 방지
        flat = "".join(re.findall(rf"\b[{valid}]+\b", m.group(1)))
        if flat:
            return flat
    runs = re.findall(rf"\b([{valid}]+)\b", raw)
    return runs[-1] if runs else ""


def parse_answer(raw: str, category: str, valid_letters: list[str]) -> str:
    """raw 모델 출력을 제출 형식으로. 실패 시 안전한 fallback을 반환한다."""
    valid = "".join(valid_letters)
    run = _extract_letter_run(raw or "", valid)

    if category in SINGLE_CATEGORIES:
        for ch in run:
            if ch in valid:
                return ch
        return valid_letters[0]  # fallback

    if category == "multi":
        chosen = sorted(set(ch for ch in run if ch in valid))
        return "".join(chosen) if chosen else valid_letters[0]

    if category == "sequence":
        # 순서 유지 + 중복 제거 후, 빠진 글자를 원래 순서로 뒤에 붙여 순열로 보정
        seen = []
        for ch in run:
            if ch in valid and ch not in seen:
                seen.append(ch)
        for ch in valid_letters:
            if ch not in seen:
                seen.append(ch)
        return "".join(seen)

    return valid_letters[0]
