#!/usr/bin/env python3
"""
06_calibrate_lens.py - 렌즈의 focal length를 픽셀 단위로 측정하고, 크기별 표지판이
거리마다 화면에서 얼마나 크게 보이는지 보여 준다.

    focal_px = pixel_height * distance / object_height

사용법 (VNC 데스크톱 터미널):
    cd ~/src
    python3 06_calibrate_lens.py --object-height 0.297 --distance 2.3     # A4 세로, 2.3 m 거리

  1. 높이를 아는 기준 물체(예: A4 용지, 높이 0.297 m)를 지정한 거리에, 화면의 "중앙"에
     둔다. 광각 렌즈는 가장자리를 휘게 하므로 가장자리 근처에서 재면 값이 틀어진다.
  2. 마우스로 드래그해서 박스가 물체의 위쪽과 아래쪽 가장자리에 닿게 한다.
     박스의 "높이"만 사용한다. 창은 카메라가 보는 그대로(회전 없음) 표시되며,
     저장되는 학습용 사진과 같다.
  3. SPACE: 측정값을 하나 추가한다. 여러 번, 다른 거리(d 키)에서도 측정한다.
     최종 값은 모든 측정값의 median이다.
  4. s: 03_config/lens.json에 저장하고 표지판 크기 표를 출력한다.

키 (먼저 창을 클릭한다):
    drag      기준 물체 둘레에 박스를 그린다 (위쪽과 아래쪽 가장자리)
    SPACE     측정한다 (목록에 값 하나를 추가)
    d         터미널에서 새 거리를 입력한다 (미터)
    o         터미널에서 새 물체 높이를 입력한다 (미터)
    u         마지막 측정값을 지운다
    t         현재 median으로 표를 출력한다
    s         03_config/lens.json에 저장하고 표를 출력한다
    q / ESC   종료한다

팁: --display-width 0 을 주면 1280x960 이미지 전체를 보여 주므로 박스 가장자리를 더 정확히
맞출 수 있다 (화면 픽셀 하나가 이미지 픽셀 두 개가 아니라 하나에 대응한다).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorlab import CONFIG
from colorlab import lens
from colorlab.camera import ColorCamera, DEFAULT_CONFIG_PATH, DEFAULT_MAIN_SIZE
from colorlab.roi import RoiSelector, clamp_roi, display_to_main

WINDOW = "06 Lens Calibration"
LENS_PATH = CONFIG / "lens.json"
CENTER_WARN = 0.2  # 박스 중심이 화면 중심에서 프레임 크기의 20 % 넘게 떨어지면 경고한다


def ask_float(prompt: str, current: float) -> float:
    """터미널에서 양수 하나를 읽는다. 빈 입력이면 현재 값을 유지한다."""
    print(f"\n{prompt} (now {current:g}, Enter = keep). The window is frozen while you type.")
    try:
        text = input(" > ").strip()
    except (EOFError, KeyboardInterrupt):
        return current
    if not text:
        return current
    try:
        value = float(text)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"[WARN] '{text}' is not a positive number. Keeping {current:g}.")
        return current


def print_report(focal_px: float, size: tuple[int, int]) -> None:
    w, h = size
    print(f"\n[LENS] focal = {focal_px:.1f} px  (at {w}x{h}, image center)")
    print(f"       {lens.px_per_degree(focal_px):.2f} px per degree  (use this for the aiming dead zone)")
    print(f"       field of view: horizontal {lens.fov_deg(w, focal_px):.1f} deg, "
          f"vertical {lens.fov_deg(h, focal_px):.1f} deg")
    print("\nSign height on screen (pixels):")
    for line in lens.format_size_table(focal_px):
        print("  " + line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the lens focal length in pixels")
    ap.add_argument("--object-height", type=float, required=True, help="real height of the reference object (m), e.g. 0.297 for A4")
    ap.add_argument("--distance", type=float, required=True, help="distance from the camera to the object (m)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--allow-auto", action="store_true", help="run without a camera config (auto mode, not recommended)")
    ap.add_argument("--out", type=Path, default=LENS_PATH, help="where to save the result")
    ap.add_argument("--width", type=int, default=DEFAULT_MAIN_SIZE[0])
    ap.add_argument("--height", type=int, default=DEFAULT_MAIN_SIZE[1])
    ap.add_argument("--display-width", type=int, default=640, help="window width (0 = full size, more exact)")
    ap.add_argument("--no-swap", action="store_true", help="do not reverse channels (only if 01_test_camera.py said 'Case B')")
    args = ap.parse_args()

    if args.object_height <= 0 or args.distance <= 0:
        print("[ERROR] --object-height and --distance must be greater than 0.")
        return 1
    if not args.config.exists() and not args.allow_auto:
        print(f"[ERROR] Camera config file not found: {args.config}")
        print("        Press f in 03_preview.py to make it, or add --allow-auto.")
        return 1

    cam = ColorCamera(main_size=(args.width, args.height), channel_swap=not args.no_swap)
    print(f"[START] Opening the camera at {args.width}x{args.height}...")
    try:
        cam.start()
    except Exception as e:
        print(f"[ERROR] Could not open the camera: {e}", file=sys.stderr)
        return 1

    object_h = args.object_height
    distance = args.distance
    measures: list[dict] = []
    sel = RoiSelector()
    status = "Drag a box from the top to the bottom edge of the object, then SPACE"

    try:
        if args.config.exists():
            s = cam.load_config(args.config)
            print(f"[LOCKED] {args.config}")
            for line in s.summary_lines():
                print("    " + line)
            for w in cam.load_warnings:
                print(f"[WARN] {w}")
            hint = cam.resolution_hint()
            if hint:
                print(f"[WARN] {hint}")
        else:
            print("[WARN] No config file, running in auto mode.")
        info = cam.sensor_info()

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, sel.on_mouse)
        print(f"[READY] object {object_h:g} m at {distance:g} m. Click the '{WINDOW}' window, then press keys.")

        while True:
            frame = cam.read_bgr()
            mh, mw = frame.shape[:2]
            if args.display_width <= 0 or args.display_width == mw:
                scale, disp = 1.0, frame.copy()
            else:
                scale = args.display_width / float(mw)
                disp = cv2.resize(frame, (args.display_width, int(mh * scale)), interpolation=cv2.INTER_AREA)
            dh, dw = disp.shape[:2]

            # 중앙 십자선: 물체를 여기에 둔다
            cx, cy = dw // 2, dh // 2
            cv2.line(disp, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
            cv2.line(disp, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)

            drect = sel.live_rect()
            roi_main = None
            if drect is not None:
                drect = clamp_roi(drect, dw, dh)
                roi_main = display_to_main(drect, scale, mw, mh)
                cv2.rectangle(disp, (drect[0], drect[1]), (drect[0] + drect[2], drect[1] + drect[3]), (0, 255, 0), 1)

            lines = [f"object {object_h:g} m   distance {distance:g} m   measures {len(measures)}"]
            if roi_main is not None:
                live_f = lens.focal_from_measure(roi_main[3], distance, object_h)
                lines.append(f"box height {roi_main[3]} px  ->  focal {live_f:.1f} px")
            else:
                lines.append("drag a box around the object")
            if measures:
                lines.append(f"median focal {lens.median(m['focal_px'] for m in measures):.1f} px   (s = save)")
            lines.append(status)
            box_h = 10 + 20 * len(lines)
            disp[:box_h] = (disp[:box_h] * 0.4).astype(disp.dtype)
            for i, text in enumerate(lines):
                cv2.putText(disp, text, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            elif key == ord(" "):
                if roi_main is None:
                    status = "No box yet - drag one first"
                    continue
                px = roi_main[3]
                f = lens.focal_from_measure(px, distance, object_h)
                off_x = abs(roi_main[0] + roi_main[2] / 2 - mw / 2) / mw
                off_y = abs(roi_main[1] + roi_main[3] / 2 - mh / 2) / mh
                measures.append({"distance_m": distance, "object_height_m": object_h,
                                 "pixel_height": px, "focal_px": f,
                                 "box": list(roi_main)})
                med = lens.median(m["focal_px"] for m in measures)
                print(f"\n[MEASURE {len(measures)}] {px} px at {distance:g} m for {object_h:g} m -> "
                      f"focal {f:.1f} px   (median {med:.1f} px)")
                if max(off_x, off_y) > CENTER_WARN:
                    print("  [WARN] The box is far from the center. A wide-angle lens bends the edges, "
                          "so this value may be off. Move the object to the center if you can.")
                status = f"measured: focal {f:.1f} px (median {med:.1f})"

            elif key == ord("d"):
                distance = ask_float("New distance in meters", distance)
                status = f"distance = {distance:g} m"

            elif key == ord("o"):
                object_h = ask_float("New object height in meters", object_h)
                status = f"object height = {object_h:g} m"

            elif key == ord("u"):
                if measures:
                    gone = measures.pop()
                    print(f"\n[UNDO] removed {gone['pixel_height']} px at {gone['distance_m']:g} m")
                    status = "last measurement removed"

            elif key in (ord("t"), ord("s")):
                if not measures:
                    status = "No measurements yet (SPACE)"
                    print("\n[INFO] No measurements yet. Press SPACE first.")
                    continue
                med = lens.median(m["focal_px"] for m in measures)
                if key == ord("s"):
                    data = {
                        "focal_px": round(med, 2),
                        "method": "median of measurements, pinhole model, image center",
                        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "resolution": info["main_size"],
                        "sensor_output_size": info["sensor_output_size"],
                        "scaler_crop": info["scaler_crop"],
                        "camera_model": info["camera_model"],
                        "rotate": 0,
                        "px_per_degree": round(lens.px_per_degree(med), 3),
                        "fov_deg": {"horizontal": round(lens.fov_deg(mw, med), 2),
                                    "vertical": round(lens.fov_deg(mh, med), 2)},
                        "measurements": [{**m, "focal_px": round(m["focal_px"], 2)} for m in measures],
                    }
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
                    print(f"\n[SAVED] {args.out}  ({len(measures)} measurements)")
                    status = f"saved: focal {med:.1f} px"
                print_report(med, (mw, mh))

    except KeyboardInterrupt:
        print("\n[STOPPED]")
    finally:
        cv2.destroyAllWindows()
        cam.close()
    print("[EXIT] Camera closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
