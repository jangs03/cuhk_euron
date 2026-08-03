"""IR 밝기 분석 → ir_preprocess_manifest.csv 생성 (팀 공통 규격 `ir_gamma_v2`).

클립마다 프레임 20장을 뽑아 밝기 통계를 내고, 적용할 감마를 미리 정해 표로 저장한다.
추론(`run_baseline.py --ir-preprocess`)은 이 표를 조회만 하므로 빠르다.

계산식은 팀원 노트북(CUHK_IR_preprocessing_addition_v2)과 동일하게 맞췄다.
같은 config면 같은 manifest가 나오므로 팀원 결과와 합쳐 써도 된다.

사용:
  # 전체 (HAU + HARn + test) — 약 4,120클립
  python src/analyze_ir.py --roots data/HAU,data/HARn,data/large_model_track_test \\
      --out data/cache/ir_preprocess_manifest.csv --workers 4

  # 이미 처리된 클립은 건너뛰므로 중단돼도 다시 실행하면 이어서 진행 (resume)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 팀 공통 설정 (preprocess_config.json과 동일 — 바꾸면 규격이 달라지니 주의)
CONFIG = {
    "version": "ir_gamma_v2",
    "analysis_sample_count": 20,
    "analysis_width": 160,
    "analysis_height": 120,
    "dark_pixel_threshold": 25,
    "bright_pixel_threshold": 245,
    "dark_median_threshold": 45,
    "dark_ratio_threshold": 0.45,
    "low_contrast_threshold": 55,
    "highlight_bright_ratio_threshold": 0.05,
    "highlight_p98_threshold": 250,
    "overexposed_bright_ratio_threshold": 0.12,
    "full_strength_end": 80,
    "protection_start": 160,
    "mask_blur_sigma": 1.5,
    "gamma_rules": [
        {"max_median_exclusive": 12, "gamma": 0.52},
        {"max_median_exclusive": 20, "gamma": 0.58},
        {"max_median_exclusive": 35, "gamma": 0.72},
        {"max_median_exclusive": 55, "gamma": 0.85},
        {"max_median_exclusive": 256, "gamma": 1.0},
    ],
}
CONFIG_SHA256 = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True, ensure_ascii=False).encode("utf-8")
).hexdigest()

COLUMNS = ["frame_count", "fps", "width", "height", "analysis_frames_requested",
           "analysis_frames_read", "p2", "p5", "p50", "p95", "p98", "dark_ratio",
           "bright_ratio", "contrast", "global_state", "highlight_clipping",
           "display_condition", "gamma", "preprocess_version", "config_sha256",
           "relative_video_path"]


def find_ir_videos(root: Path, modality: str = "IR") -> list[Path]:
    """루트 아래의 <modality>/<modality>.mp4 파일 전부."""
    out = []
    for cur, _dirs, files in os.walk(root):
        p = Path(cur)
        if p.name != modality:
            continue
        for f in files:
            if Path(f).suffix.lower() == ".mp4":
                out.append(p / f)
    return sorted(out)


def analyze(video_path: Path) -> dict | None:
    """프레임 20장 샘플링 → 밝기 통계. 열 수 없으면 None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        return None

    n = min(int(CONFIG["analysis_sample_count"]), total)
    size = (int(CONFIG["analysis_width"]), int(CONFIG["analysis_height"]))
    buf, read = [], 0
    for idx in np.linspace(0, total - 1, n, dtype=np.int64):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        buf.append(cv2.resize(gray, size, interpolation=cv2.INTER_AREA).reshape(-1))
        read += 1
    cap.release()
    if not buf:
        return None

    px = np.concatenate(buf).astype(np.float32)
    p2, p5, p50, p95, p98 = np.percentile(px, [2, 5, 50, 95, 98])
    return {
        "frame_count": total, "fps": fps, "width": w, "height": h,
        "analysis_frames_requested": n, "analysis_frames_read": read,
        "p2": float(p2), "p5": float(p5), "p50": float(p50),
        "p95": float(p95), "p98": float(p98),
        "dark_ratio": float(np.mean(px <= CONFIG["dark_pixel_threshold"])),
        "bright_ratio": float(np.mean(px >= CONFIG["bright_pixel_threshold"])),
        "contrast": float(p95 - p5),
    }


def classify(stats: dict) -> dict:
    median, dark = float(stats["p50"]), float(stats["dark_ratio"])
    bright, contrast, p98 = (float(stats["bright_ratio"]),
                             float(stats["contrast"]), float(stats["p98"]))
    if median < CONFIG["dark_median_threshold"] or dark >= CONFIG["dark_ratio_threshold"]:
        state = "dark"
    elif contrast < CONFIG["low_contrast_threshold"]:
        state = "low_contrast"
    else:
        state = "normal"

    clipping = (bright >= CONFIG["highlight_bright_ratio_threshold"]
                or p98 >= CONFIG["highlight_p98_threshold"])
    if state == "dark" and clipping:
        display = "mixed"
    elif (state == "normal" and clipping
          and bright >= CONFIG["overexposed_bright_ratio_threshold"]):
        display = "overexposed"
    else:
        display = state
    return {"global_state": state, "highlight_clipping": bool(clipping),
            "display_condition": display}


def choose_gamma(median: float) -> float:
    for rule in CONFIG["gamma_rules"]:
        if float(median) < rule["max_median_exclusive"]:
            return float(rule["gamma"])
    return 1.0


def process(job: tuple) -> dict | None:
    video_s, root_s = job
    video, root = Path(video_s), Path(root_s)
    stats = analyze(video)
    if stats is None:
        return None
    cond = classify(stats)
    gamma = 1.0 if cond["global_state"] == "normal" else choose_gamma(stats["p50"])
    rel = Path(root.name) / video.relative_to(root)
    return {**stats, **cond, "gamma": float(gamma),
            "preprocess_version": CONFIG["version"], "config_sha256": CONFIG_SHA256,
            "relative_video_path": rel.as_posix()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="data/HAU,data/HARn,data/large_model_track_test",
                    help="쉼표로 구분한 미디어 루트")
    ap.add_argument("--out", default="data/cache/ir_preprocess_manifest.csv")
    ap.add_argument("--config-out", default="data/cache/preprocess_config.json")
    ap.add_argument("--modality", default="IR")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (테스트용)")
    args = ap.parse_args()

    jobs = []
    for r in args.roots.split(","):
        root = Path(r.strip())
        if not root.exists():
            print(f"[건너뜀] 폴더 없음: {root}")
            continue
        vids = find_ir_videos(root, args.modality)
        print(f"{root}: {len(vids)}개 {args.modality} 영상")
        jobs += [(str(v), str(root)) for v in vids]

    if not jobs:
        raise SystemExit("분석할 영상이 없습니다 — --roots 경로를 확인하세요")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.config_out).write_text(
        json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")

    done_rows, done_keys = [], set()
    if out_path.exists():                       # resume: 처리된 클립은 건너뜀
        prev = pd.read_csv(out_path)
        done_rows = prev.to_dict("records")
        done_keys = set(prev["relative_video_path"].astype(str))
        print(f"resume: 기존 {len(done_keys)}개 유지")

    todo = [j for j in jobs
            if (Path(Path(j[1]).name) / Path(j[0]).relative_to(j[1])).as_posix()
            not in done_keys]
    if args.limit:
        todo = todo[:args.limit]
    print(f"분석 대상: {len(todo)}개 (전체 {len(jobs)})")
    if not todo:
        print("이미 완료 — 할 일 없음")
        return

    rows, failed = list(done_rows), 0
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            it = pool.imap_unordered(process, todo, chunksize=8)
            for i, rec in enumerate(tqdm(it, total=len(todo)), 1):
                if rec is None:
                    failed += 1
                else:
                    rows.append(rec)
                if i % 200 == 0:                # 중간 저장 (중단 대비)
                    pd.DataFrame(rows)[COLUMNS].to_csv(out_path, index=False)
    else:
        for i, job in enumerate(tqdm(todo), 1):
            rec = process(job)
            if rec is None:
                failed += 1
            else:
                rows.append(rec)
            if i % 200 == 0:
                pd.DataFrame(rows)[COLUMNS].to_csv(out_path, index=False)

    df = pd.DataFrame(rows)[COLUMNS].sort_values("relative_video_path")
    df.to_csv(out_path, index=False)

    print(f"\nmanifest → {out_path}  ({len(df)}개, 실패 {failed})")
    print(f"config   → {args.config_out}")
    print("\n밝기 상태 분포:")
    for k, v in df["global_state"].value_counts().items():
        print(f"  {k:12s} {v:5d}  ({v / len(df) * 100:.1f}%)")
    corrected = int((df["gamma"] < 1.0).sum())
    print(f"\n보정 적용 대상: {corrected}/{len(df)} ({corrected / len(df) * 100:.1f}%)"
          f"  — 나머지는 원본 유지")


if __name__ == "__main__":
    main()
