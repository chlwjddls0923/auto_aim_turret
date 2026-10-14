"""측정 결과를 저장한다 (CSV에 추가 + 스냅샷 이미지)."""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np

from . import CAPTURES, LOGS
from .camera import LockedSettings
from .color import ColorSample

MEASUREMENTS_CSV = LOGS / "measurements.csv"

FIELDS = [
    "timestamp", "name",
    "R", "G", "B",
    "R_sstd", "G_sstd", "B_sstd",
    "R_tstd", "G_tstd", "B_tstd",
    "H_deg", "S_pct", "V_pct",
    "L", "a", "b",
    "Y", "hex", "color_name",
    "n_frames", "n_pixels", "clip_ratio", "dark_ratio",
    "roi_x", "roi_y", "roi_w", "roi_h",
    "awb_ae_locked", "exposure_us", "analogue_gain", "gain_r", "gain_b", "lens_position",
    "snapshot",
]


def _safe(name: str) -> str:
    """이름을 파일 이름에 쓸 수 있게 정리한다. 한글은 그대로 둔다."""
    s = re.sub(r"[^\w\uAC00-\uD7A3\u3131-\u314E\u314F-\u3163._-]+", "_", name.strip())
    return s[:60] or "unnamed"


def save_measurement(
    sample: ColorSample,
    name: str,
    roi: tuple[int, int, int, int],
    frame_bgr: np.ndarray | None = None,
    locked: LockedSettings | None = None,
) -> tuple[Path, Path | None]:
    """
    측정 1건을 CSV에 추가하고 스냅샷 이미지를 저장한다.

    frame_bgr : BGR 순서의 원본 프레임 (cv2.imwrite에 바로 넣을 수 있다)
    locked    : '측정 당시'의 잠금 설정. auto 모드에서 측정했다면 반드시 None을 넘긴다.
    반환값    : (csv 경로, 스냅샷 경로 또는 None)
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    snap_path: Path | None = None

    if frame_bgr is not None:
        snap = frame_bgr.copy()
        x, y, w, h = roi
        cv2.rectangle(snap, (x, y), (x + w, y + h), (0, 255, 255), 2)
        CAPTURES.mkdir(parents=True, exist_ok=True)
        snap_path = CAPTURES / f"{ts}_{_safe(name)}.jpg"
        cv2.imwrite(str(snap_path), snap, [cv2.IMWRITE_JPEG_QUALITY, 95])

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "R": f"{sample.r:.2f}", "G": f"{sample.g:.2f}", "B": f"{sample.b:.2f}",
        "R_sstd": f"{sample.r_std:.2f}", "G_sstd": f"{sample.g_std:.2f}", "B_sstd": f"{sample.b_std:.2f}",
        "R_tstd": f"{sample.r_tstd:.3f}", "G_tstd": f"{sample.g_tstd:.3f}", "B_tstd": f"{sample.b_tstd:.3f}",
        "H_deg": f"{sample.h:.2f}", "S_pct": f"{sample.s:.2f}", "V_pct": f"{sample.v:.2f}",
        "L": f"{sample.L:.3f}", "a": f"{sample.a:.3f}", "b": f"{sample.bb:.3f}",
        "Y": f"{sample.y:.2f}", "hex": sample.hex, "color_name": sample.color_name,
        "n_frames": sample.n_frames, "n_pixels": sample.n_pixels,
        "clip_ratio": f"{sample.clip_ratio:.4f}", "dark_ratio": f"{sample.dark_ratio:.4f}",
        "roi_x": roi[0], "roi_y": roi[1], "roi_w": roi[2], "roi_h": roi[3],
        "awb_ae_locked": "yes" if locked is not None else "no",
        "exposure_us": locked.exposure_time if locked else "",
        "analogue_gain": f"{locked.analogue_gain:.3f}" if locked else "",
        "gain_r": f"{locked.colour_gains[0]:.4f}" if locked else "",
        "gain_b": f"{locked.colour_gains[1]:.4f}" if locked else "",
        "lens_position": f"{locked.lens_position:.3f}" if locked and locked.lens_position is not None else "",
        "snapshot": snap_path.name if snap_path else "",
    }

    MEASUREMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not MEASUREMENTS_CSV.exists()
    with MEASUREMENTS_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)

    return MEASUREMENTS_CSV, snap_path


def load_measurements() -> list[dict]:
    """저장된 측정값을 모두 읽는다 (기능 4 비교 리포트에서 사용)."""
    if not MEASUREMENTS_CSV.exists():
        return []
    with MEASUREMENTS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
