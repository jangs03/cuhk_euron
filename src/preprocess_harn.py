"""HARn 전처리: HARn/<action>/<user>/<trial>/<modality>/<modality>.mp4 클립을 스캔해서

  1) 클립 인덱스 csv 생성  — 클립별 메타데이터(액션/유저/보유 modality/프레임 수/해상도)
  2) 선택한 modality(기본 Depth)에서 클립당 N프레임 균등 샘플링
     → JPEG 캐시 저장 (원본 디렉토리 구조 미러링, <clip>/<modality>/frame_XXX.jpg)

캐시가 원본 구조를 그대로 미러링하므로, 전처리 후 추론은 --media-root만 바꾸면 된다:

  python src/preprocess_harn.py --harn-root data/HARn --out data/cache/HARn --frames 16
  python src/run_baseline.py --qa data/test_qa.csv --media-root data/cache,data --out submission.csv

같은 구조인 HAU에도 그대로 쓸 수 있다 (--harn-root data/HAU --out data/cache/HAU).

depth 특화 처리:
  - 16-bit depth 이미지 → 1~99 percentile 클리핑 후 8-bit 정규화 (대비 향상)
  - --colormap 시 JET 컬러맵 적용 (VLM 가시성 실험용)
"""
import argparse
import csv
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from tqdm import tqdm

from data_utils import IMAGE_EXTS, MODALITIES, VIDEO_EXTS, _uniform_indices, extract_user


def find_clips(harn_root: Path) -> dict[Path, list[str]]:
    """{클립 디렉토리: 보유 modality 목록} 반환.

    실제 구조는 <clip>/<modality>/<modality>.mp4 이므로, 미디어를 직접 포함하는
    leaf 디렉토리의 이름이 modality면 그 부모를 클립으로 본다.
    modality 폴더 없이 미디어가 바로 들어 있는 변형 구조도 허용."""
    media_exts = VIDEO_EXTS | IMAGE_EXTS
    clips: dict[Path, list[str]] = {}
    for p in harn_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in media_exts:
            leaf = p.parent
            if leaf.name in MODALITIES:
                clips.setdefault(leaf.parent, [])
                if leaf.name not in clips[leaf.parent]:
                    clips[leaf.parent].append(leaf.name)
            else:
                clips.setdefault(leaf, [])
    return dict(sorted(clips.items()))


def to_uint8(img: np.ndarray) -> np.ndarray:
    """16-bit depth 등 비표준 dtype을 percentile 정규화로 8-bit 변환."""
    if img.dtype == np.uint8:
        return img
    valid = img[img > 0]
    if valid.size == 0:
        return np.zeros(img.shape[:2], np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    out = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return out.astype(np.uint8)


def read_clip_frames(clip_dir: Path, n: int) -> tuple[list[np.ndarray], str, int]:
    """클립 디렉토리에서 n프레임 균등 샘플링. (frames, media_type, 원본 프레임 수) 반환."""
    videos = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if videos:
        videos.sort(key=lambda p: ("depth" not in p.name.lower(), p.name))
        cap = cv2.VideoCapture(str(videos[0]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        if total > 0:
            for idx in _uniform_indices(total, n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, f = cap.read()
                if ok:
                    frames.append(f)
        else:
            buf = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                buf.append(f)
            total = len(buf)
            frames = [buf[i] for i in _uniform_indices(total, n)]
        cap.release()
        return frames, "video", total

    images = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if images:
        picked = [images[i] for i in _uniform_indices(len(images), n)]
        frames = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in picked]
        return frames, "images", len(images)

    return [], "empty", 0


def process_clip(job: tuple) -> dict:
    (clip_dir_s, modalities, harn_root_s, out_root_s,
     n_frames, size, colormap, modality) = job
    clip_dir, harn_root, out_root = Path(clip_dir_s), Path(harn_root_s), Path(out_root_s)
    rel = clip_dir.relative_to(harn_root)
    root_name = harn_root.name  # 'HARn', 'HAU' 등 — csv path와 맞추기 위해 유지
    row = {
        "rel_path": str(Path(root_name) / rel).replace("\\", "/"),
        "action": rel.parts[0] if len(rel.parts) >= 1 else "",
        "user": extract_user(str(Path(root_name) / rel)),
        "seq": rel.parts[-1],
        "modality": "", "available": ";".join(modalities),
        "media_type": "", "src_frames": 0, "saved_frames": 0,
        "width": 0, "height": 0, "error": "",
    }
    try:
        # 원하는 modality 우선, 없으면 선호 순서 fallback
        src_dir, used = clip_dir, ""
        if modalities:
            pref = [modality] + [m for m in MODALITIES if m != modality]
            used = next((m for m in pref if m in modalities), modalities[0])
            src_dir = clip_dir / used
        row["modality"] = used

        frames, media_type, total = read_clip_frames(src_dir, n_frames)
        row["media_type"], row["src_frames"] = media_type, total
        if not frames:
            row["error"] = "no frames"
            return row

        # 캐시도 <clip>/<modality>/ 구조를 미러링 → resolve/sampling이 그대로 동작
        out_dir = out_root / rel / used if used else out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            if f is None:
                continue
            if f.ndim == 2:  # grayscale (16-bit depth 포함)
                f = to_uint8(f)
            elif f.dtype != np.uint8:
                f = to_uint8(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            if colormap:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                f = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            h, w = f.shape[:2]
            scale = size / max(h, w)
            if scale < 1:
                f = cv2.resize(f, (int(w * scale), int(h * scale)))
            cv2.imwrite(str(out_dir / f"frame_{i:03d}.jpg"), f,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            row["saved_frames"] += 1
            row["height"], row["width"] = f.shape[:2]
    except Exception:
        row["error"] = traceback.format_exc(limit=1).strip().splitlines()[-1]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harn-root", default="data/HARn", help="HARn.zip 해제 루트 (HAU도 가능)")
    ap.add_argument("--out", default="data/cache/HARn", help="프레임 캐시 출력 루트")
    ap.add_argument("--index", default="data/cache/harn_index.csv")
    ap.add_argument("--frames", type=int, default=16, help="클립당 저장 프레임 수")
    ap.add_argument("--size", type=int, default=448, help="긴 변 기준 리사이즈")
    ap.add_argument("--colormap", action="store_true", help="depth를 JET 컬러맵으로")
    ap.add_argument("--modality", default="IR",
                    help="캐시할 modality (없는 클립은 선호 순서로 fallback). IR이 가장 선명.")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    harn_root = Path(args.harn_root)
    if not harn_root.exists():
        raise SystemExit(f"harn root not found: {harn_root}")

    clips = find_clips(harn_root)
    print(f"found {len(clips)} clips under {harn_root}")

    jobs = [(str(c), mods, str(harn_root), args.out, args.frames, args.size,
             args.colormap, args.modality)
            for c, mods in clips.items()]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            rows = list(tqdm(pool.imap_unordered(process_clip, jobs), total=len(jobs)))
    else:
        rows = [process_clip(j) for j in tqdm(jobs)]

    index_path = Path(args.index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["rel_path"]))

    n_err = sum(1 for r in rows if r["error"])
    users = sorted({r["user"] for r in rows if r["user"] is not None})
    actions = sorted({r["action"] for r in rows})
    print(f"\nindex → {index_path}")
    print(f"clips: {len(rows)}  errors: {n_err}")
    print(f"users ({len(users)}): {users}")
    print(f"actions ({len(actions)}): {actions[:20]}{' ...' if len(actions) > 20 else ''}")
    if n_err:
        print("\nfirst errors:")
        for r in [r for r in rows if r["error"]][:5]:
            print(f"  {r['rel_path']}: {r['error']}")


if __name__ == "__main__":
    main()
