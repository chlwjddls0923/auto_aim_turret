#!/usr/bin/env python3
"""
색 공간 변환 확인 스크립트

  python3 02_check_colorspace.py            # 기준값 테스트만 (카메라 불필요)
  python3 02_check_colorspace.py --camera   # + 카메라 중앙 영역의 실제 RGB/HSV/LAB/Y 측정

기준값 테스트: colorlab.color의 변환 결과를 알려진 sRGB (D65) 기준값과 비교한다.
  - CIELAB  : L* 0~100, a*/b* 약 -128~127
  - HSV     : H 0~360 도, S/V 0~100%
  - 밝기 Y  : BT.601, 0~255
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorlab.color import rgb_to_hsv360, rgb_to_lab, rgb_to_luma, hue_name, measure_roi

LAB_TRUTH = [
    ("red", (255, 0, 0), (53.2408, 80.0925, 67.2032)),
    ("green", (0, 255, 0), (87.7347, -86.1827, 83.1793)),
    ("blue", (0, 0, 255), (32.2970, 79.1875, -107.8602)),
    ("white", (255, 255, 255), (100.0, 0.0, 0.0)),
    ("gray", (128, 128, 128), (53.585, 0.0, 0.0)),
]
HUE_TRUTH = [
    ("red", (255, 0, 0), 0), ("yellow", (255, 255, 0), 60), ("green", (0, 255, 0), 120),
    ("cyan", (0, 255, 255), 180), ("blue", (0, 0, 255), 240),
]
Y_TRUTH = [((255, 0, 0), 76.245), ((0, 255, 0), 149.685), ((0, 0, 255), 29.07), ((255, 255, 255), 255.0)]


def selftest() -> bool:
    ok = True
    print("=== CIELAB reference check (tolerance 0.6) ===")
    for name, rgb, exp in LAB_TRUTH:
        got = rgb_to_lab(np.array(rgb, dtype=np.float64)).reshape(3)
        err = max(abs(got[i] - exp[i]) for i in range(3))
        passed = err < 0.6
        ok &= passed
        print(f"  {'OK  ' if passed else 'FAIL'} {name:6s} {str(rgb):16s} "
              f"L*={got[0]:7.2f} a*={got[1]:8.2f} b*={got[2]:8.2f}  (error {err:.3f})")

    print("\n=== HSV range (H=0~360, S/V=0~100) ===")
    for name, rgb, eh in HUE_TRUTH:
        h, s, v = rgb_to_hsv360(np.array(rgb, dtype=np.float64)).reshape(3)
        passed = abs(h - eh) < 0.6 and 0.0 <= h < 360.0
        ok &= passed
        print(f"  {'OK  ' if passed else 'FAIL'} {name:6s} {str(rgb):16s} "
              f"H={h:6.1f}deg (expected {eh:3d})  S={s:5.1f}%  V={v:5.1f}%  -> {hue_name(h, s, v)}")

    print("\n=== Brightness Y (BT.601) ===")
    for rgb, exp in Y_TRUTH:
        y = float(rgb_to_luma(np.array(rgb, dtype=np.float64)))
        passed = abs(y - exp) < 0.01
        ok &= passed
        print(f"  {'OK  ' if passed else 'FAIL'} {str(rgb):16s} Y={y:7.2f} (expected {exp})")

    print("\nResult:", "ALL PASS" if ok else "SOME FAILED")
    return ok


def camera_check(n_frames: int, warmup: float) -> None:
    from colorlab.camera import ColorCamera

    print(f"\n=== Camera measurement (center 20% area, mean of {n_frames} frames) ===")
    cam = ColorCamera()
    try:
        cam.start()
    except RuntimeError as e:
        print(f"  [FAIL] Cannot open the camera: {e}")
        print("  Check if another program (03_preview.py, rpicam-hello, etc.) is using the camera. Close it and run again.")
        print("  Check: ps aux | grep -E 'rpicam|03_preview.py' | grep -v grep")
        return
    try:
        time.sleep(warmup)
        frames = cam.grab_rgb_frames(n_frames)
        h, w = frames[0].shape[:2]
        rw, rh = int(w * 0.2), int(h * 0.2)
        roi = (w // 2 - rw // 2, h // 2 - rh // 2, rw, rh)
        s = measure_roi(frames, roi)
        print(f"  ROI = {roi}")
        for line in s.summary_lines():
            print("  " + line)
        print("\n  If you hold a red object in front, the first RGB value (R) must be the largest. Then the channel order is right.")
    finally:
        cam.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true", help="also measure the camera center area")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--warmup", type=float, default=3.0)
    args = ap.parse_args()

    ok = selftest()
    if args.camera:
        camera_check(args.frames, args.warmup)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
