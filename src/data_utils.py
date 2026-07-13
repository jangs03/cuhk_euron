"""CSV 로드, 미디어 경로 해석, 프레임 샘플링, user id 추출."""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# 클립 디렉토리 안의 modality 하위 폴더 이름 (선호 순서 = fallback 순서)
# 실데이터 확인 결과: IR이 가장 선명(사진 수준), Depth_Color는 양호, Depth는 윤곽만 남음
MODALITIES = ["IR", "Depth_Color", "Depth", "Thermal"]


def load_qa(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, keep_default_na=False)  # 빈 D 컬럼을 NaN이 아닌 ""로
    return df


def get_options(row) -> dict:
    """행에서 유효한 보기만 {letter: text}로 반환 (HARn single은 D가 빈칸)."""
    opts = {}
    for letter in config.LETTERS:
        text = str(row.get(letter, "")).strip()
        if text and text.lower() != "nan":
            opts[letter] = text
    return opts


def extract_user(path: str) -> int | None:
    """path에서 user id 추출 (실제 데이터 구조 기준).
    HARn: HARn/<action>/<user>/<trial>          예: HARn/0_Wash_face/user16/1-1-2
    HAU : HAU/<user>/<trial>                     예: HAU/user3/2-1
    테스트 클립은 LM_test_XXXX로 익명화 → None 반환.
    """
    parts = Path(str(path)).parts
    lowered = [p.lower() for p in parts]
    idx = None
    if "harn" in lowered:
        idx = lowered.index("harn") + 2
    elif "hau" in lowered:
        idx = lowered.index("hau") + 1
    if idx is not None and idx < len(parts):
        m = re.fullmatch(r"(?:user[_\s]?)?(\d+)", parts[idx], re.IGNORECASE)
        if m:
            return int(m.group(1))
    # fallback: 'userNN' 토큰 탐색
    for p in parts:
        m = re.fullmatch(r"user[_\s]?(\d+)", p, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def resolve_media(rel_path: str, media_roots) -> Path:
    """csv의 path를 실제 파일/디렉토리 경로로 해석. 대소문자/루트 변형에 관대하게.

    media_roots: Path 하나 또는 Path 리스트. 리스트면 앞의 루트부터 시도하므로
    캐시 우선 검색이 가능하다 (예: [data/cache, data]).
    """
    if isinstance(media_roots, (str, Path)):
        media_roots = [media_roots]
    media_roots = [Path(r) for r in media_roots]

    rel_path = str(rel_path).strip().replace("\\", "/")
    variants = [
        rel_path,
        rel_path.lower(),
        rel_path.replace("HARn", "harn"),
        rel_path.replace("harn", "HARn"),
    ]
    for root in media_roots:
        for v in variants:
            c = root / v
            if c.exists():
                return c
    # 마지막 수단: glob으로 탐색 (한 단계 하위 폴더에 압축이 풀린 경우)
    tail = Path(rel_path).name
    for root in media_roots:
        if not root.exists():
            continue
        hits = list(root.glob(f"**/{tail}"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"media not found: {rel_path} (roots={media_roots})")


def get_duration(media_path: Path, modality: str = "IR") -> float | None:
    """클립 길이(초). 원본 비디오 메타데이터에서 계산 (fps, frame count).
    이미지 캐시 디렉토리처럼 비디오가 없으면 None."""
    p = Path(media_path)
    if p.is_dir():
        pref = [modality] + [m for m in MODALITIES if m != modality]
        for m in pref:
            if (p / m).is_dir():
                p = p / m
                break
        vids = sorted(v for v in p.rglob("*") if v.suffix.lower() in VIDEO_EXTS)
        if not vids:
            return None
        p = vids[0]
    if p.suffix.lower() not in VIDEO_EXTS:
        return None
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0 and total > 0:
        return float(total / fps)
    return None


def swap_modality(rel_path: str, modality: str) -> str:
    """test path처럼 '<Mod>/<Mod>.mp4'로 끝나는 경로를 원하는 modality로 교체.
    해당 패턴이 아니면 (클립 디렉토리 경로면) 그대로 반환."""
    p = Path(str(rel_path).strip())
    if p.suffix.lower() in VIDEO_EXTS and p.parent.name in MODALITIES:
        return str(p.parent.parent / modality / f"{modality}{p.suffix}").replace("\\", "/")
    return str(rel_path)


def _uniform_indices(total: int, n: int) -> list[int]:
    if total <= 0:
        return []
    n = min(n, total)
    return sorted({int(round(i)) for i in np.linspace(0, total - 1, n)})


def _read_video_frames(video_path: Path, n: int) -> tuple[list[np.ndarray], list[float]]:
    """비디오에서 n프레임 균등 샘플링. (frames, 상대 위치 0~1 리스트) 반환."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, pos = [], []
    if total > 0:
        for idx in _uniform_indices(total, n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
                pos.append(idx / max(total - 1, 1))
    else:  # 프레임 수 메타데이터가 없으면 전부 읽고 샘플링
        buf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(frame)
        idxs = _uniform_indices(len(buf), n)
        frames = [buf[i] for i in idxs]
        pos = [i / max(len(buf) - 1, 1) for i in idxs]
    cap.release()
    return frames, pos


def _read_dir_frames(dir_path: Path, n: int,
                     modality: str = "IR") -> tuple[list[np.ndarray], list[float]]:
    """클립 디렉토리인 경우: modality 하위 폴더를 선호 순서대로 선택한 뒤
    그 안의 비디오 또는 이미지 시퀀스를 읽는다. (frames, 상대 위치) 반환.
    (training path = 클립 디렉토리, 내부는 <modality>/<modality>.mp4 구조)"""
    pref = [modality] + [m for m in MODALITIES if m != modality]
    for m in pref:
        sub = dir_path / m
        if sub.is_dir():
            dir_path = sub
            break
    videos = sorted(p for p in dir_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if videos:
        videos.sort(key=lambda p: ("depth" not in p.name.lower(), p.name))
        return _read_video_frames(videos[0], n)
    images = sorted(p for p in dir_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if images:
        idxs = _uniform_indices(len(images), n)
        frames = [cv2.imread(str(images[i])) for i in idxs]
        pos = [i / max(len(images) - 1, 1) for i in idxs]
        return frames, pos
    raise FileNotFoundError(f"no video/images inside: {dir_path}")


def _pick_motion_indices(frames: list[np.ndarray], n: int) -> list[int]:
    """모션 에너지 누적분포를 n등분해 keyframe 인덱스 선택 (시간순 유지).

    행동이 벌어지는 구간에서 촘촘히, 정지 구간에서 성기게 뽑는다.
    균등 샘플링이 짧게 등장하는 행동을 놓치는 문제(multi/sequence) 대응."""
    if len(frames) <= n:
        return list(range(len(frames)))
    small = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        small.append(cv2.resize(g, (80, 60)))
    energy = [1e-6]  # 첫 프레임 몫
    for a, b in zip(small, small[1:]):
        energy.append(float(cv2.absdiff(a, b).mean()) + 1e-6)
    cum = np.cumsum(energy)
    targets = np.linspace(cum[0], cum[-1], n)
    idx = sorted({int(np.searchsorted(cum, t, side="left")) for t in targets})
    # 중복 제거로 모자라면 아직 안 뽑힌 인덱스로 채움
    pool = [i for i in range(len(frames)) if i not in idx]
    while len(idx) < n and pool:
        idx.append(pool.pop(0))
    return sorted(idx)[:n]


def person_crop_frames(frames: list[np.ndarray], pad: float = 0.15,
                       thresh: int = 25, min_area: float = 0.01) -> list[np.ndarray]:
    """고정 카메라 가정의 사람 영역 crop.

    프레임들의 중앙값을 배경으로 보고, 배경과 달라지는(=사람이 지나간) 픽셀의
    합집합 bbox로 모든 프레임을 동일하게 crop한다 (프레임 간 구도 일관성 유지).
    움직임 신호가 너무 작거나 화면 대부분이면 원본을 그대로 반환한다.
    """
    if len(frames) < 3:
        return frames
    grays = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        grays.append(g)
    bg = np.median(np.stack(grays), axis=0).astype(grays[0].dtype)

    # 2프레임 이상에서 움직임이 감지된 픽셀만 인정 (단발 노이즈 제거)
    count = np.zeros(bg.shape, np.uint16)
    for g in grays:
        count += (cv2.absdiff(g, bg) > thresh).astype(np.uint16)
    mask = ((count >= 2) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)

    # 유의미한 크기의 움직임 덩어리들의 합집합 bbox
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bg.shape[:2]
    boxes = [cv2.boundingRect(c) for c in contours
             if cv2.contourArea(c) > 0.003 * w * h]
    if not boxes:
        return frames
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)

    area = (x1 - x0) * (y1 - y0)
    if area < min_area * w * h or area > 0.9 * w * h:
        return frames  # 신호가 너무 작거나 crop 이득이 없음
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w, x1 + px), min(h, y1 + py)
    return [f[y0:y1, x0:x1] for f in frames]


def sample_frames(media_path: Path, n: int, colormap: bool = False,
                  max_side: int = 448, modality: str = "IR",
                  crop_person: bool = False, sampling: str = "uniform",
                  return_pos: bool = False):
    """미디어(파일 or 클립 디렉토리)에서 n프레임 샘플링 → PIL 리스트.

    sampling='uniform'  : 균등 간격 (기본)
    sampling='motion'   : 후보를 넉넉히(4n, 최소 32장) 읽은 뒤 모션 에너지 기준 keyframe 선택
    return_pos=True     : (frames, 상대 위치 0~1 리스트) 튜플 반환 — 타임스탬프 계산용
    """
    n_read = n if sampling == "uniform" else max(4 * n, 32)
    if media_path.is_dir():
        raw, pos = _read_dir_frames(media_path, n_read, modality)
    else:
        raw, pos = _read_video_frames(media_path, n_read)

    valid = [(f, p) for f, p in zip(raw, pos) if f is not None]
    raw = [f for f, _ in valid]
    pos = [p for _, p in valid]

    if sampling == "motion" and len(raw) > n:
        sel = _pick_motion_indices(raw, n)
        raw = [raw[i] for i in sel]
        pos = [pos[i] for i in sel]

    if crop_person:
        raw = person_crop_frames(raw)

    frames = []
    for f in raw:
        if f is None:
            continue
        if colormap:  # depth 가시성 향상: grayscale → JET 컬러맵
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            f = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w = f.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            f = cv2.resize(f, (int(w * scale), int(h * scale)))
        frames.append(Image.fromarray(f))
    if not frames:
        raise RuntimeError(f"no frames decoded from {media_path}")
    if return_pos:
        return frames, pos[:len(frames)]
    return frames
