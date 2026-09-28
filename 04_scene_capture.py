#!/usr/bin/env python3
"""
[배경 색 분포 수집] 고정된 카메라로 배경 전체의 색 분포를 CSV + 이미지로 저장한다.

목적:
  대상(피아 식별 표지)이 화면 어디에 나타날지 알 수 없다. 그래서 배경 어디에도 거의
  나타나지 않는 색을 찾아 표지 색으로 사용한다.
  결과 파일은 외부로 보내 분석하므로, meta.json에 모든 조건과 컬럼 정의를 기록한다.
  이렇게 하면 파일만 보고도 내용을 이해할 수 있다.

화면 없이(SSH로) 동작한다. 해상도(기본 640x480)와 채널 처리는 03_preview.py와 같다.

예시:
  python3 04_scene_capture.py --self-test                          # 첫 실행: PNG 색 순서 확인
  python3 04_scene_capture.py --label morning_sunny                # 1회 촬영
  python3 04_scene_capture.py --label morning_sunny --interval 600 --count 12   # 10분 간격 12회
  python3 04_scene_capture.py --label test --allow-auto            # config 없이 자동 모드 (권장하지 않음)

Ctrl+C 한 번: 현재 촬영을 마친 뒤 종료 / 두 번: 즉시 종료.
결과는 촬영마다 파일에 기록되므로, 중간에 멈춰도 그때까지의 결과는 모두 안전하다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorlab import ROOT
from colorlab.camera import ColorCamera, DEFAULT_CONFIG_PATH
from colorlab.color import CLIP_THR, DARK_THR, clip_dark_masks, hue_name, rgb_to_hsv360, rgb_to_lab
from colorlab.store import _safe as safe_name

OUT_ROOT = ROOT / "04_scene_captures"

SIZE_LIMIT_BYTES = 20 * 1024 * 1024  # pixels.csv 전체 크기 상한
CLIP_STRONG_WARN = 0.02  # 한 촬영의 포화(clipped) 비율이 이 값을 넘으면 강한 경고
DARK_WARN = 0.05  # 어두운 픽셀 비율 경고(참고용, 그림자/검은 물체는 정상일 수 있어 느슨하게 둔다)
HUE_EMPTY_RATIO = 0.001  # 이 비율 미만인 Hue bin = "빈" bin
LOW_CHROMA_WARN = 0.05  # 유채색 픽셀이 이보다 적으면 Hue 분포를 신뢰할 수 없다
HIGH_GAIN_WARN = 7.9  # 이 값 이상이면 노이즈가 크다 (ov5647 자동 노출 최대값은 8.0)
LOCK_DRIFT_WARN = 0.05  # 실제 프레임의 노출/게인이 config와 5% 넘게 다르면 고정이 풀린 것이다

L_STEP, L_BINS = 10, 10  # L: 0~100, 간격 10
AB_STEP, AB_BINS, AB_MIN = 8, 32, -128  # a/b: -128~128, 간격 8
HUE_STEP, HUE_BINS = 10, 36  # H: 0~360, 간격 10

PIXELS_COLS = ["capture_id", "timestamp", "label", "x", "y", "R", "G", "B",
               "H", "S", "V", "L", "a", "b", "clipped", "dark"]
HIST_LAB_COLS = ["capture_id", "label", "L_bin_start", "a_bin_start", "b_bin_start", "count", "ratio"]
HIST_HUE_COLS = ["capture_id", "label", "hue_bin_start", "count", "ratio", "total_chromatic_pixels"]


# ---------------------------------------------------------------------------
# 파일 입출력
# ---------------------------------------------------------------------------
def save_png_rgb(path: Path, rgb: np.ndarray) -> None:
    """실제 RGB 배열을 표준 PNG로 저장한다 (cv2.imwrite는 BGR을 기대하므로 변환한다)."""
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                     [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"Failed to save PNG: {path}")


def read_png_rgb_cv2(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def write_atomic(path: Path, text: str) -> None:
    """임시 파일에 쓴 뒤 교체한다. 그래서 중간에 멈춰도 반쯤 쓰인 파일이 남지 않는다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_csv(path: Path, rows: list[list]) -> None:
    """한 촬영의 행을 메모리에서 만든 뒤 한 번에 추가 + fsync한다."""
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())


def init_csv(path: Path, cols: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(cols)


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------
def r2(a: np.ndarray) -> np.ndarray:
    """소수점 둘째 자리로 반올림한다. -0.00이 출력되지 않도록 +0.0을 더한다."""
    return np.round(a, 2) + 0.0


def grid_blocks(rgb: np.ndarray, step: int):
    """step x step 블록마다 평균 RGB와, 블록 안에 포화/어두운 픽셀이 하나라도 있는지를 구한다. 가장자리 블록은 더 작은 실제 크기로 평균한다."""
    h, w = rgb.shape[:2]
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    sums = np.add.reduceat(np.add.reduceat(rgb.astype(np.float64), ys, axis=0), xs, axis=1)
    bh = np.diff(np.append(ys, h))
    bw = np.diff(np.append(xs, w))
    mean = sums / np.outer(bh, bw)[..., None]
    clip, dark = clip_dark_masks(rgb)
    cb = np.add.reduceat(np.add.reduceat(clip.astype(np.int32), ys, axis=0), xs, axis=1) > 0
    db = np.add.reduceat(np.add.reduceat(dark.astype(np.int32), ys, axis=0), xs, axis=1) > 0
    return xs, ys, mean, cb, db


def analyze(rgb: np.ndarray, sat_min: float, val_min: float) -> dict:
    """합성 이미지 한 장을 전체 해상도로 분석한다."""
    hsv = rgb_to_hsv360(rgb.astype(np.float64))  # H 0~360, S/V 0~100
    lab = rgb_to_lab(rgb.astype(np.float64))  # L 0~100, a/b 약 -128~127
    H, S, V = hsv[..., 0], hsv[..., 1] / 100.0, hsv[..., 2] / 100.0
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    total = int(L.size)

    clip, dark = clip_dark_masks(rgb)

    li = np.clip((L // L_STEP).astype(np.int64), 0, L_BINS - 1)  # L=100 -> 마지막 bin (90~100)
    ai = np.clip(((A - AB_MIN) // AB_STEP).astype(np.int64), 0, AB_BINS - 1)
    bi = np.clip(((B - AB_MIN) // AB_STEP).astype(np.int64), 0, AB_BINS - 1)
    lab_counts = np.bincount(((li * AB_BINS + ai) * AB_BINS + bi).ravel(),
                             minlength=L_BINS * AB_BINS * AB_BINS)

    chroma = (S >= sat_min) & (V >= val_min)
    n_chroma = int(chroma.sum())
    hi = np.clip((H[chroma] // HUE_STEP).astype(np.int64), 0, HUE_BINS - 1)
    hue_counts = np.bincount(hi, minlength=HUE_BINS)

    # 전체 촬영에 걸친 백분위수 계산에 쓰는 L 히스토그램 (0.1 간격)
    l_fine = np.bincount(np.clip(np.rint(L * 10).astype(np.int64), 0, 1000).ravel(), minlength=1001)

    return {
        "total": total,
        "clip_mask": clip, "dark_mask": dark,
        "clip_ratio": float(clip.mean()), "dark_ratio": float(dark.mean()),
        "lab_counts": lab_counts,
        "hue_counts": hue_counts, "n_chroma": n_chroma,
        "chromatic_ratio": n_chroma / total,
        "L_mean": float(L.mean()),
        "L_p10": float(np.percentile(L, 10)), "L_p50": float(np.percentile(L, 50)),
        "L_p90": float(np.percentile(L, 90)),
        "l_fine": l_fine,
    }


def clipmap_image(rgb: np.ndarray, clip: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """어둡게 만든 흑백 이미지 위에 포화 = 빨강, 어두움 = 파랑으로 표시한다. RGB를 반환한다."""
    y = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    base = (y.astype(np.float32) * 0.5).astype(np.uint8)
    out = np.dstack([base, base, base])
    out[clip] = (255, 0, 0)
    out[dark] = (0, 0, 255)
    return out


def grab_median(cam: ColorCamera, n: int):
    """연속 n 프레임의 픽셀별 중앙값(uint8)을 구한다. n이 짝수이면 가운데 두 값의 평균을 반올림한다."""
    frames = [np.ascontiguousarray(cam.read_rgb()) for _ in range(n)]
    meta = cam.metadata()
    med = np.median(np.stack(frames, axis=0), axis=0)
    return np.rint(med).astype(np.uint8), meta


def estimate_pixels_bytes(w: int, h: int, step: int, count: int, label: str) -> int:
    gw, gh = math.ceil(w / step), math.ceil(h / step)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(
        [count, "2026-01-01 00:00:00", label, w - step, h - step,
         "123.45", "123.45", "123.45", "359.99", "0.55", "0.55", "53.24", "-12.34", "-45.67", 0, 0])
    row = len(buf.getvalue().encode("utf-8"))
    header = len((",".join(PIXELS_COLS) + "\n").encode("utf-8"))
    return header + gw * gh * count * row


# ---------------------------------------------------------------------------
# 요약 / meta
# ---------------------------------------------------------------------------
def hue_range_name(start: int, end: int) -> str:
    """start~end 각도 범위 안의 색 이름을 순서대로 모두 나열한다 (end가 360을 넘으면 한 바퀴 돌아간다)."""
    names: list[str] = []
    for h in range(start, end, HUE_STEP):
        n = hue_name((h + HUE_STEP / 2) % 360, 100, 100)
        if not names or names[-1] != n:
            names.append(n)
    return "/".join(names)


def merge_empty_ranges(empty_bins: list[int]) -> list[tuple[int, int]]:
    """빈 bin 인덱스를 연속 구간으로 묶는다. 350~360과 0~10은 이어진 것으로 본다."""
    if not empty_bins:
        return []
    s = set(empty_bins)
    if len(s) == HUE_BINS:
        return [(0, 360)]
    # 비어 있지 않은 bin 바로 다음부터 시작해 한 바퀴 돌며 묶는다
    start_idx = next(i for i in range(HUE_BINS) if i not in s)
    ranges, cur = [], None
    for k in range(1, HUE_BINS + 1):
        i = (start_idx + k) % HUE_BINS
        if i in s:
            cur = [i, i] if cur is None else [cur[0], i]
        elif cur is not None:
            ranges.append(tuple(cur))
            cur = None
    if cur is not None:
        ranges.append(tuple(cur))
    return [(a * HUE_STEP, ((b + 1) * HUE_STEP) if b >= a else (b + 1) * HUE_STEP + 360) for a, b in ranges]


def fmt_range(a: int, b: int) -> str:
    return f"{a}~{b % 360 if b > 360 else b} deg" + (" (wraps past 360 deg)" if b > 360 else "")


def l_percentile(fine: np.ndarray, q: float) -> float:
    c = np.cumsum(fine)
    return float(np.searchsorted(c, q / 100.0 * c[-1])) / 10.0


def build_summary(meta: dict, caps: list[dict]) -> str:
    lk = meta["lock"]
    out = []
    out.append("=" * 70)
    out.append(" Background color distribution summary (04_scene_capture.py)")
    out.append("=" * 70)
    out.append(f"Label       : {meta['label']}")
    out.append(f"Folder      : {meta['output_dir']}")
    out.append(f"Time        : {meta['started_at']} ~ {meta['updated_at']}  ({meta['status']})")
    out.append(f"Captures    : {len(caps)} / {meta['capture_plan']['count']}, interval {meta['capture_plan']['interval_sec']} sec, "
               f"median of {meta['capture_plan']['frames_per_capture']} frames per capture")
    out.append(f"Camera      : {meta['camera'].get('camera_model')} {meta['camera'].get('main_size')} "
               f"(sensor {meta['camera'].get('sensor_output_size')}, {meta['camera'].get('sensor_bit_depth')}bit, "
               f"ScalerCrop {meta['camera'].get('scaler_crop')})")
    if lk["locked"]:
        lp = f", LensPosition {lk['LensPosition']}" if lk.get("LensPosition") is not None else ""
        out.append(f"Locked      : ExposureTime {lk['ExposureTime']}us, AnalogueGain {lk['AnalogueGain']:.3f}, "
                   f"ColourGains {lk['ColourGains']}{lp}")
    else:
        out.append("!!! AUTO mode (AWB/AE on) - exposure/white balance change each capture, so you cannot compare with real values !!!")
    out.append(f"Hue pixels  : only pixels with S >= {meta['thresholds']['hue_sat_min']} and V >= {meta['thresholds']['hue_val_min']} (gray/dark pixels excluded)")
    out.append("")

    out.append("-" * 70)
    out.append(" Quality / brightness per capture")
    out.append("-" * 70)
    out.append(" id  time                clip%   dark%  color%   L mean  L p10 / p50 / p90")
    for c in caps:
        out.append(f" {c['capture_id']:02d}  {c['timestamp']}  {c['clip_ratio']*100:6.2f}  {c['dark_ratio']*100:6.2f}"
                   f"  {c['chromatic_ratio']*100:6.2f}   {c['L_mean']:6.2f}   {c['L_p10']:5.1f} / {c['L_p50']:5.1f} / {c['L_p90']:5.1f}")
    out.append("")

    tot_hue = np.sum([c["_hue_counts"] for c in caps], axis=0) if caps else np.zeros(HUE_BINS, np.int64)
    tot_chroma = int(tot_hue.sum())
    out.append("-" * 70)
    out.append(" Hue distribution (all captures added, ratio of colored pixels)")
    out.append("-" * 70)
    if tot_chroma == 0:
        out.append(" No colored pixels.")
    else:
        ratio = tot_hue / tot_chroma
        out.append(" Top 5 most common Hue bins:")
        for i in np.argsort(-ratio)[:5]:
            s0 = int(i) * HUE_STEP
            out.append(f"   {s0:3d}~{s0 + HUE_STEP:3d} deg  {ratio[i]*100:6.2f}%   ({hue_range_name(s0, s0 + HUE_STEP)})")
        empty = [i for i in range(HUE_BINS) if ratio[i] < HUE_EMPTY_RATIO]
        out.append("")
        out.append(f" Empty Hue ranges (ratio below {HUE_EMPTY_RATIO*100:.1f}%) = sign color candidates:")
        if not empty:
            out.append("   None - every Hue appears at least a little.")
        for a, b in merge_empty_ranges(empty):
            out.append(f"   {fmt_range(a, b)}  ({(b - a) // HUE_STEP} bins, about {hue_range_name(a, b)})")
        if len(caps) > 1:
            out.append("")
            out.append(" Number of empty bins per capture (check if it changes with lighting):")
            for c in caps:
                n = c["chromatic_pixels"]
                e = int(np.sum(c["_hue_counts"] / n < HUE_EMPTY_RATIO)) if n else HUE_BINS
                out.append(f"   capture {c['capture_id']:02d}: {e}")
    out.append("")

    out.append("-" * 70)
    out.append(" Overall brightness (L*) distribution (all captures, all pixels)")
    out.append("-" * 70)
    if caps:
        fine = np.sum([c["_l_fine"] for c in caps], axis=0)
        mean = float(np.sum(fine * np.arange(1001) / 10.0) / fine.sum())
        out.append(f" mean {mean:.2f}   p10 {l_percentile(fine, 10):.1f}   p50 {l_percentile(fine, 50):.1f}   p90 {l_percentile(fine, 90):.1f}  (0~100)")
    out.append("")

    out.append("-" * 70)
    out.append(" Warnings")
    out.append("-" * 70)
    warns = list(meta["warnings"])
    for c in caps:
        warns += c["warnings"]
    out += [f" - {w}" for w in warns] if warns else [" None"]
    out.append("")
    out.append("For file descriptions and column definitions, see the \"files\" section in meta.json.")
    return "\n".join(out) + "\n"


def files_doc(grid_step: int, sat_min: float, val_min: float) -> dict:
    return {
        "capture_XX.png": {
            "description": "Composite image: per-pixel median of N frames. Lossless PNG, standard RGB order (correct colors in normal viewers).",
        },
        "capture_XX_clipmap.png": {
            "description": f"On a darkened grayscale image: clipped pixels (any channel >= {CLIP_THR}) = red, dark pixels (all channels <= {DARK_THR}) = blue.",
        },
        "pixels.csv": {
            "description": f"The composite image sampled on a {grid_step}px grid. Each row = the mean RGB of the {grid_step}x{grid_step} block whose top-left corner is (x,y), "
                           "plus HSV/Lab converted from that mean RGB. Edge blocks are averaged over their smaller size. All captures are added to this file.",
            "columns": {
                "capture_id": "Capture number (starts at 1). Matches meta.json captures[].capture_id",
                "timestamp": "Capture time YYYY-MM-DD HH:MM:SS",
                "label": "Shooting condition label (ends with [AUTO] in auto mode)",
                "x": "Block top-left x (px, 0=left)",
                "y": "Block top-left y (px, 0=top)",
                "R": "Block mean R, 0~255 (true RGB order), 2 decimal places",
                "G": "Block mean G, 0~255",
                "B": "Block mean B, 0~255",
                "H": "Hue, 0~360 deg (below 360)",
                "S": "Saturation, 0~1",
                "V": "Value (brightness), 0~1",
                "L": "CIELAB L*, 0~100 (standard range, not the OpenCV 8bit scale)",
                "a": "CIELAB a*, about -128~127 (negative=green, positive=red/magenta)",
                "b": "CIELAB b*, about -128~127 (negative=blue, positive=yellow)",
                "clipped": f"1 if the block has at least one pixel with any channel >= {CLIP_THR} (color not reliable)",
                "dark": f"1 if the block has at least one pixel with all channels <= {DARK_THR} (color not reliable)",
            },
            "notes": "H/S/V/L/a/b use D65 sRGB. H has little meaning for blocks with low saturation, so read it together with S.",
        },
        "hist_lab.csv": {
            "description": "3D CIELAB histogram made from 'all pixels' of the composite image. Bins with count 0 are left out. All captures are added to this file.",
            "columns": {
                "capture_id": "Capture number",
                "label": "Shooting condition label",
                "L_bin_start": f"Start of L* bin (0,10,...,90). Bin width {L_STEP}. L*=100 goes into the 90~100 bin",
                "a_bin_start": f"Start of a* bin (-128,-120,...,120). Bin width {AB_STEP}",
                "b_bin_start": f"Start of b* bin (-128,-120,...,120). Bin width {AB_STEP}",
                "count": "Number of pixels in this bin",
                "ratio": "count / total pixels of this capture",
            },
        },
        "hist_hue.csv": {
            "description": f"Hue histogram. Uses only pixels with S >= {sat_min} and V >= {val_min} (gray and dark pixels excluded). "
                           f"{HUE_BINS} bins of {HUE_STEP} deg. Bins with count 0 are included, so each capture always has {HUE_BINS} rows.",
            "columns": {
                "capture_id": "Capture number",
                "label": "Shooting condition label",
                "hue_bin_start": f"Start of Hue bin (0,10,...,350). Bin width {HUE_STEP} deg",
                "count": "Number of colored pixels in this bin",
                "ratio": "count / total_chromatic_pixels (0 if there are no colored pixels)",
                "total_chromatic_pixels": "Total number of colored pixels in this capture that meet the condition",
            },
        },
        "summary.txt": {"description": "Summary for people to read. Rewritten after every capture."},
        "meta.json": {"description": "This file. Rewritten after every capture. If status is not completed, the run was stopped midway."},
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test(args) -> int:
    from PIL import Image

    tmp = OUT_ROOT / "_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    ok = True
    print("=== PNG color order self-test ===")

    # 1) 합성 테스트 패턴: 순색 줄무늬 + 무작위 영역
    rng = np.random.default_rng(0)
    img = np.zeros((60, 120, 3), np.uint8)
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0), (250, 128, 3)]):
        img[:30, i * 20:(i + 1) * 20] = c
    img[30:] = rng.integers(0, 256, (30, 120, 3), dtype=np.uint8)
    p = tmp / "pattern.png"
    save_png_rgb(p, img)
    back_cv = read_png_rgb_cv2(p)
    back_pil = np.array(Image.open(p).convert("RGB"))  # cv2와 독립된 디코더
    r1 = np.array_equal(img, back_cv)
    r2_ = np.array_equal(img, back_pil)
    print(f"  [test pattern] read back with cv2 matches: {r1}")
    print(f"  [test pattern] read back with PIL matches: {r2_}  <- means other programs show the same colors")
    print(f"  [test pattern] red stripe value read with PIL: {tuple(int(v) for v in back_pil[10, 10])} (expected (255, 0, 0))")
    ok &= r1 and r2_ and tuple(back_pil[10, 10]) == (255, 0, 0)

    # 2) 실제 카메라 중앙값 합성 이미지
    cam = ColorCamera(main_size=(args.width, args.height), channel_swap=not args.no_swap)
    try:
        cam.start()
    except RuntimeError as e:
        print(f"  [camera] skipped - cannot open the camera ({e}). Check if another program is using it.")
        cam = None
    if cam is not None:
        try:
            if args.config.exists():
                cam.load_config(args.config)
            else:
                time.sleep(2.0)
            comp, _ = grab_median(cam, args.frames)
            p2 = tmp / "camera_composite.png"
            save_png_rgb(p2, comp)
            r3 = np.array_equal(comp, np.array(Image.open(p2).convert("RGB")))
            print(f"  [camera composite {comp.shape[1]}x{comp.shape[0]}] read back with PIL matches: {r3}")
            ok &= r3
        finally:
            cam.close()

    print(f"  Test files: {tmp}")
    if not ok:
        print("Result: FAIL - the color order is wrong. Please report this.")
        return 1
    if cam is None:
        print("Result: PARTIAL PASS - test pattern PASS, but the camera composite image was not checked. Free the camera and run again.")
        return 3
    print("Result: PASS - the color order saved in PNG is correct.")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Capture the color distribution of the whole background (CSV + PNG)")
    ap.add_argument("--label", help="Shooting condition label (e.g. morning_sunny). Written to files/CSV")
    ap.add_argument("--count", type=int, default=1, help="Number of captures")
    ap.add_argument("--interval", type=float, default=600.0, help="Time between capture starts (sec). Used when count is 2 or more")
    ap.add_argument("--frames", type=int, default=15, help="Number of frames in a row to combine per capture (median)")
    ap.add_argument("--grid-step", type=int, default=4, help="Grid step for pixels.csv (px)")
    ap.add_argument("--sat-min", type=float, default=0.25, help="Minimum saturation S for the Hue histogram (0~1)")
    ap.add_argument("--val-min", type=float, default=0.15, help="Minimum value V for the Hue histogram (0~1)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--allow-auto", action="store_true", help="Allow shooting in auto mode when there is no config (not recommended)")
    ap.add_argument("--force", action="store_true", help=f"Continue even if the expected pixels.csv size is over {SIZE_LIMIT_BYTES//1024//1024}MB")
    ap.add_argument("--no-swap", action="store_true", help="Do not swap channels (only if 01_test_camera.py said Case B)")
    ap.add_argument("--self-test", action="store_true", help="Only check PNG color order, then exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args)

    if not args.label:
        ap.error("--label is required (e.g. --label morning_sunny)")
    if args.count < 1 or args.frames < 1 or args.grid_step < 1:
        ap.error("--count, --frames, --grid-step must be 1 or more")

    # ---- 파일 크기 먼저 확인 ----------------------------------------
    est = estimate_pixels_bytes(args.width, args.height, args.grid_step, args.count, args.label)
    print(f"[estimate] total pixels.csv size about {est/1024/1024:.1f} MB "
          f"({math.ceil(args.width/args.grid_step)}x{math.ceil(args.height/args.grid_step)} blocks x {args.count} captures)")
    if est > SIZE_LIMIT_BYTES:
        step = args.grid_step
        while estimate_pixels_bytes(args.width, args.height, step, args.count, args.label) > SIZE_LIMIT_BYTES:
            step += 1
        print(f"[warning] Over {SIZE_LIMIT_BYTES//1024//1024}MB. We recommend --grid-step {step} or more "
              f"(expected {estimate_pixels_bytes(args.width, args.height, step, args.count, args.label)/1024/1024:.1f} MB).")
        if not args.force:
            print("        To continue anyway, add --force. Exiting without capturing.")
            return 2
        print("        --force given - continuing anyway.")

    # ---- config ----------------------------------------------------
    use_auto = False
    if not args.config.exists():
        if not args.allow_auto:
            print(f"[error] Camera config file not found: {args.config}")
            print("       First run 03_preview.py in the real environment and press f to lock and save AWB/AE.")
            print("       (To shoot in auto mode without config, use --allow-auto, but results cannot be compared with real use)")
            return 1
        use_auto = True
        print("[warning] No config, so shooting in auto mode. All output files will be marked as auto mode.")

    label_rec = args.label + ("[AUTO]" if use_auto else "")

    # ---- 카메라 ------------------------------------------------------
    cam = ColorCamera(main_size=(args.width, args.height), channel_swap=not args.no_swap)
    try:
        cam.start()
    except RuntimeError as e:
        print(f"[error] Cannot open the camera: {e}")
        print("       If another program (such as 03_preview.py) is using the camera, close it.")
        return 1

    run_warnings: list[str] = []
    try:
        if use_auto:
            time.sleep(3.0)
        else:
            s = cam.load_config(args.config)
            print(f"[loaded] {args.config}")
            for line in s.summary_lines():
                print("   ", line)
            for w in cam.load_warnings:
                print(f"  [warning] {w}")
                run_warnings.append(f"config mismatch: {w}")
            if not s.converged:
                run_warnings.append("The config values were locked before they converged. We recommend locking again with f in 03_preview.py.")
            if s.analogue_gain >= HIGH_GAIN_WARN:
                run_warnings.append(f"Locked gain {s.analogue_gain:.2f} is high - the light is dim, so there is a lot of noise. If possible, add more light and make the config again.")
        for w in run_warnings:
            print(f"  [warning] {w}")

        cam_info = cam.sensor_info()
        L = cam.locked
        lock_info = {
            "locked": not use_auto,
            "mode": "AUTO (AWB/AE on, cannot compare)" if use_auto else "LOCKED (AWB/AE off)",
            "config_path": str(args.config) if not use_auto else None,
            "config_warnings": list(cam.load_warnings) if not use_auto else [],
            "ExposureTime": L.exposure_time if L else None,
            "AnalogueGain": L.analogue_gain if L else None,
            "ColourGains": list(L.colour_gains) if L else None,
            "LensPosition": L.lens_position if L else None,
            "locked_at": L.locked_at if L else None,
            "converged": L.converged if L else None,
        }

        # ---- 출력 폴더 ------------------------------------------------
        start_ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = OUT_ROOT / f"{start_ts}_{safe_name(args.label)}{'_AUTO' if use_auto else ''}"
        out_dir.mkdir(parents=True, exist_ok=False)
        init_csv(out_dir / "pixels.csv", PIXELS_COLS)
        init_csv(out_dir / "hist_lab.csv", HIST_LAB_COLS)
        init_csv(out_dir / "hist_hue.csv", HIST_HUE_COLS)

        meta = {
            "tool": "04_scene_capture.py",
            "purpose": "Collect the color distribution of the whole background with a fixed camera. The goal is to pick colors that are rare in the background as friend/enemy sign colors.",
            "label": args.label,
            "label_recorded": label_rec,
            "output_dir": str(out_dir),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "capture_plan": {
                "count": args.count,
                "interval_sec": args.interval,
                "frames_per_capture": args.frames,
                "composite": "One uint8 image: per-pixel median of N frames in a row, rounded (removes sensor noise and short movements)",
            },
            "camera": cam_info,
            "lock": lock_info,
            "color_conventions": {
                "RGB": "0~255, true R,G,B order (the picamera2 'RGB888' buffer is BGR in memory, so we swap it)",
                "H": "0~360 deg", "S": "0~1", "V": "0~1",
                "L": "0~100", "a": "about -128~127", "b": "about -128~127",
                "reference": "sRGB, D65, OpenCV float32 conversion (not the 8bit scale)",
            },
            "thresholds": {
                "clip_thr": CLIP_THR, "dark_thr": DARK_THR,
                "hue_sat_min": args.sat_min, "hue_val_min": args.val_min,
                "grid_step": args.grid_step,
                "hue_empty_ratio": HUE_EMPTY_RATIO,
                "clip_strong_warn_ratio": CLIP_STRONG_WARN,
                "dark_warn_ratio": DARK_WARN,
            },
            "files": files_doc(args.grid_step, args.sat_min, args.val_min),
            "warnings": run_warnings,
            "captures": [],
        }

        def flush_meta_summary(caps_full: list[dict]):
            meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta["captures"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in caps_full]
            write_atomic(out_dir / "meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
            write_atomic(out_dir / "summary.txt", build_summary(meta, caps_full))

        flush_meta_summary([])
        print(f"[output] {out_dir}")

        # ---- Ctrl+C: 한 번 = 현재 촬영을 마치고 종료, 두 번 = 즉시 종료 --------
        stop = {"flag": False}

        def on_sigint(sig, frm):
            if stop["flag"]:
                raise KeyboardInterrupt
            stop["flag"] = True
            print("\n[stop requested] Finishing the current capture, then exiting. (Press again to exit now)", flush=True)

        signal.signal(signal.SIGINT, on_sigint)

        caps: list[dict] = []
        t_next = time.monotonic()
        for i in range(1, args.count + 1):
            if stop["flag"]:
                break
            # 다음 촬영까지 대기
            while not stop["flag"]:
                remain = t_next - time.monotonic()
                if remain <= 0:
                    break
                m, s_ = divmod(int(math.ceil(remain)), 60)
                print(f"\r  Next capture ({i:02d}/{args.count:02d}) in {m:02d}:{s_:02d}   ", end="", flush=True)
                time.sleep(min(1.0, remain))
            if stop["flag"]:
                print()
                break
            print("\r" + " " * 50 + "\r", end="")
            t_next = time.monotonic() + args.interval

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[capture {i:02d}/{args.count:02d}] {ts}  collecting {args.frames} frames...", flush=True)
            comp, fmeta = grab_median(cam, args.frames)
            an = analyze(comp, args.sat_min, args.val_min)

            img_name = f"capture_{i:02d}.png"
            clip_name = f"capture_{i:02d}_clipmap.png"
            save_png_rgb(out_dir / img_name, comp)
            save_png_rgb(out_dir / clip_name, clipmap_image(comp, an["clip_mask"], an["dark_mask"]))

            # pixels.csv
            xs, ys, bmean, bclip, bdark = grid_blocks(comp, args.grid_step)
            bh = rgb_to_hsv360(bmean)
            bl = rgb_to_lab(bmean)
            R, G, B = r2(bmean[..., 0]), r2(bmean[..., 1]), r2(bmean[..., 2])
            Hh, Ss, Vv = r2(bh[..., 0]), r2(bh[..., 1] / 100.0), r2(bh[..., 2] / 100.0)
            Ll, Aa, Bb = r2(bl[..., 0]), r2(bl[..., 1]), r2(bl[..., 2])
            Hh = np.where(Hh >= 360.0, 0.0, Hh)  # 359.995가 360.00으로 반올림되는 것을 막는다
            rows = []
            for yi, y in enumerate(ys):
                for xi, x in enumerate(xs):
                    rows.append([i, ts, label_rec, int(x), int(y),
                                 f"{R[yi, xi]:.2f}", f"{G[yi, xi]:.2f}", f"{B[yi, xi]:.2f}",
                                 f"{Hh[yi, xi]:.2f}", f"{Ss[yi, xi]:.2f}", f"{Vv[yi, xi]:.2f}",
                                 f"{Ll[yi, xi]:.2f}", f"{Aa[yi, xi]:.2f}", f"{Bb[yi, xi]:.2f}",
                                 int(bclip[yi, xi]), int(bdark[yi, xi])])
            append_csv(out_dir / "pixels.csv", rows)

            # hist_lab.csv
            lc = an["lab_counts"]
            nz = np.nonzero(lc)[0]
            li, rem = np.divmod(nz, AB_BINS * AB_BINS)
            ai, bi = np.divmod(rem, AB_BINS)
            append_csv(out_dir / "hist_lab.csv", [
                [i, label_rec, int(l) * L_STEP, AB_MIN + int(a) * AB_STEP, AB_MIN + int(b) * AB_STEP,
                 int(lc[k]), f"{lc[k] / an['total']:.6f}"]
                for k, l, a, b in zip(nz, li, ai, bi)])

            # hist_hue.csv
            hc, nch = an["hue_counts"], an["n_chroma"]
            append_csv(out_dir / "hist_hue.csv", [
                [i, label_rec, k * HUE_STEP, int(hc[k]), f"{(hc[k] / nch) if nch else 0:.6f}", nch]
                for k in range(HUE_BINS)])

            # 이번 촬영의 경고
            cw = []
            if an["clip_ratio"] > CLIP_STRONG_WARN:
                cw.append(f"[STRONG WARNING] capture {i:02d}: clipped pixels {an['clip_ratio']*100:.1f}% (> {CLIP_STRONG_WARN*100:.0f}%). "
                          "Colors in bright areas are cut off and differ from real colors. Lower the exposure and make the config again (press f in 03_preview.py).")
            if an["dark_ratio"] > DARK_WARN:
                cw.append(f"capture {i:02d}: dark pixels {an['dark_ratio']*100:.1f}% (> {DARK_WARN*100:.0f}%). "
                          "Colors in those areas are not reliable (blue parts of the clipmap). You can ignore this if they are shadows/black objects.")
            if an["chromatic_ratio"] < LOW_CHROMA_WARN:
                cw.append(f"capture {i:02d}: only {an['chromatic_ratio']*100:.1f}% colored pixels, so the Hue distribution is not very reliable.")
            if L is not None:
                fe, fg = float(fmeta.get("ExposureTime", 0)), float(fmeta.get("AnalogueGain", 0))
                if abs(fe - L.exposure_time) > LOCK_DRIFT_WARN * L.exposure_time or \
                        abs(fg - L.analogue_gain) > LOCK_DRIFT_WARN * L.analogue_gain:
                    cw.append(f"capture {i:02d}: real frame exposure/gain ({fe:.0f}us, {fg:.2f}) differ from config. The lock may be lost.")

            cap = {
                "capture_id": i,
                "timestamp": ts,
                "image": img_name,
                "clipmap": clip_name,
                "clip_ratio": round(an["clip_ratio"], 6),
                "dark_ratio": round(an["dark_ratio"], 6),
                "chromatic_ratio": round(an["chromatic_ratio"], 6),
                "chromatic_pixels": an["n_chroma"],
                "total_pixels": an["total"],
                "L_mean": round(an["L_mean"], 3),
                "L_p10": round(an["L_p10"], 3), "L_p50": round(an["L_p50"], 3), "L_p90": round(an["L_p90"], 3),
                "pixels_csv_rows": len(rows),
                "frame_metadata": {
                    "ExposureTime": fmeta.get("ExposureTime"),
                    "AnalogueGain": fmeta.get("AnalogueGain"),
                    "DigitalGain": fmeta.get("DigitalGain"),
                    "ColourGains": list(fmeta["ColourGains"]) if "ColourGains" in fmeta else None,
                    "LensPosition": fmeta.get("LensPosition"),
                    "Lux": fmeta.get("Lux"),
                    "ColourTemperature": fmeta.get("ColourTemperature"),
                },
                "warnings": cw,
                "_hue_counts": an["hue_counts"],
                "_l_fine": an["l_fine"],
            }
            caps.append(cap)
            flush_meta_summary(caps)

            print(f"  saved: {img_name}, {clip_name} | clipped {an['clip_ratio']*100:.2f}%  dark {an['dark_ratio']*100:.2f}%  "
                  f"colored {an['chromatic_ratio']*100:.1f}%  L mean {an['L_mean']:.1f}", flush=True)
            for w in cw:
                print(f"  {'!!!' if w.startswith('[STRONG') else '[warning]'} {w}", flush=True)

        meta["status"] = "completed" if len(caps) == args.count else "interrupted"
        flush_meta_summary(caps)
        print(f"\n[done] {len(caps)}/{args.count} captures ({meta['status']})")
        print(f"       {out_dir}")
        print(f"       summary: {out_dir / 'summary.txt'}")
        return 0

    except KeyboardInterrupt:
        print("\n[exit now] Results up to the last finished capture are kept in the files.")
        return 130
    finally:
        cam.close()


if __name__ == "__main__":
    sys.exit(main())
