"""IR 밝기 전처리 (팀 공통 `ir_gamma_v2` 규격) 적용 모듈.

사전 분석 결과(`ir_preprocess_manifest.csv`)를 읽어 클립별로 정해진 감마를
추론 시점에 프레임에 적용한다. 분석은 팀원의 노트북에서 수행하고, 여기서는
**적용만** 담당한다 (동일 규격 보장을 위해 계산식을 그대로 옮김).

핵심 규칙
  - global_state == 'normal' 또는 gamma >= 1.0  → 원본 유지 (보정 안 함)
  - 어두운 클립만 감마 보정하되, **밝은 영역은 보호 마스크로 제외**
    (mask = 1 - smoothstep((gray - full_strength_end) / (protection_start - ...)))
  - manifest에 없는 클립은 조용히 원본 사용 (부분 커버리지 허용)

사용:
    prep = IRPreprocessor("data/cache/ir_preprocess_manifest.csv")
    rec = prep.record_for("HAU/user8/3-3-1", "IR")
    frame = prep.apply(frame_bgr, rec)
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 노트북 preprocess_config.json 기본값 — config 파일이 없을 때 사용
DEFAULT_CONFIG = {
    "version": "ir_gamma_v2",
    "full_strength_end": 80,
    "protection_start": 160,
    "mask_blur_sigma": 1.5,
}


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _gamma_lut(gray: np.ndarray, gamma: float) -> np.ndarray:
    values = np.arange(256, dtype=np.float32) / 255.0
    table = np.clip((values ** float(gamma)) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, table)


class IRPreprocessor:
    def __init__(self, manifest_path, config_path=None):
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"IR manifest가 없습니다: {manifest_path}")
        df = pd.read_csv(manifest_path)

        self.config = dict(DEFAULT_CONFIG)
        cfg = Path(config_path) if config_path else \
            manifest_path.parent / "preprocess_config.json"
        if cfg.exists():
            self.config.update(json.loads(cfg.read_text(encoding="utf-8")))

        # 조회 인덱스: 정규화 전체 경로 + 데이터셋 접두어를 뗀 꼬리 경로
        # (팀원마다 루트가 달라 'HAU/...' / 'TEST/...' 접두어가 다를 수 있음)
        self.by_key: dict[str, dict] = {}
        for rec in df.to_dict("records"):
            key = str(rec.get("relative_video_path", "")).replace("\\", "/").strip("./")
            if not key:
                continue
            rec = dict(rec)
            self.by_key[key.lower()] = rec
            parts = key.split("/")
            if len(parts) > 1:                       # 접두어 제거 형태도 등록
                self.by_key.setdefault("/".join(parts[1:]).lower(), rec)
        self.hits = 0
        self.misses = 0

    def _candidate_keys(self, rel_path: str, modality: str) -> list[str]:
        p = str(rel_path).replace("\\", "/").strip("./")
        keys = []
        if p.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            # test 경로: '.../<Mod>/<Mod>.mp4' → 원하는 modality로 교체한 형태도 시도
            parent = "/".join(p.split("/")[:-2])
            keys += [p, f"{parent}/{modality}/{modality}.mp4"]
        else:                                        # training 경로: 클립 디렉토리
            keys.append(f"{p}/{modality}/{modality}.mp4")
        # 접두어를 뗀 꼬리 형태 (HAU/… ↔ TEST/… 불일치 대응)
        extra = []
        for k in keys:
            parts = k.split("/")
            if len(parts) > 1:
                extra.append("/".join(parts[1:]))
        return keys + extra

    def record_for(self, rel_path: str, modality: str = "IR") -> dict | None:
        """클립에 해당하는 manifest 레코드 (없으면 None → 보정 없음)."""
        for k in self._candidate_keys(rel_path, modality):
            rec = self.by_key.get(k.lower())
            if rec is not None:
                self.hits += 1
                return rec
        self.misses += 1
        return None

    def apply(self, frame_bgr: np.ndarray, record: dict | None) -> np.ndarray:
        """프레임에 감마 보정 적용 (BGR 입출력). record가 None이면 원본 반환."""
        if record is None or frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        gamma = float(record.get("gamma", 1.0) or 1.0)
        state = str(record.get("global_state", "normal"))
        if state == "normal" or gamma >= 1.0:        # 정상 클립은 원본 유지
            return frame_bgr

        gray = (cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if frame_bgr.ndim == 3 else frame_bgr)
        gamma_img = _gamma_lut(gray, gamma)

        # 밝은 영역 보호: 어두운 픽셀은 보정 100%, protection_start 이상은 0%
        start = float(self.config["full_strength_end"])
        end = float(self.config["protection_start"])
        gray_f = gray.astype(np.float32)
        mask = 1.0 - _smoothstep((gray_f - start) / max(end - start, 1.0))

        sigma = float(self.config.get("mask_blur_sigma", 0) or 0)
        if sigma > 0:                                # 마스크만 블러 (밴딩 방지)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)

        out = gray_f * (1.0 - mask) + gamma_img.astype(np.float32) * mask
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    def coverage(self) -> str:
        total = self.hits + self.misses
        if not total:
            return "IR 전처리: 조회 없음"
        return (f"IR 전처리 적용: {self.hits}/{total} "
                f"({self.hits / total * 100:.1f}%) — 나머지는 원본 사용")
