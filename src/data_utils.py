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


def _read_video_frames(video_path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for idx in _uniform_indices(total, n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    else:  # 프레임 수 메타데이터가 없으면 전부 읽고 샘플링
        buf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(frame)
        frames = [buf[i] for i in _uniform_indices(len(buf), n)]
    cap.release()
    return frames


def _read_dir_frames(dir_path: Path, n: int, modality: str = "Depth") -> list[np.ndarray]:
    """클립 디렉토리인 경우: modality 하위 폴더를 선호 순서대로 선택한 뒤
    그 안의 비디오 또는 이미지 시퀀스를 읽는다.
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
        picked = [images[i] for i in _uniform_indices(len(images), n)]
        return [cv2.imread(str(p)) for p in picked]
    raise FileNotFoundError(f"no video/images inside: {dir_path}")


def sample_frames(media_path: Path, n: int, colormap: bool = False,
                  max_side: int = 448, modality: str = "Depth") -> list[Image.Image]:
    """미디어(파일 or 클립 디렉토리)에서 n프레임 균등 샘플링 → PIL 리스트."""
    if media_path.is_dir():
        raw = _read_dir_frames(media_path, n, modality)
    else:
        raw = _read_video_frames(media_path, n)

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
    return frames
