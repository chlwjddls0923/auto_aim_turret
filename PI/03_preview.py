#!/usr/bin/env python3
"""
[기능 1 + 2] 실시간 미리보기 + ROI 색상 측정   (OpenCV 창 / VNC 화면용)

사용법:
    cd ~/src
    python3 03_preview.py                 # auto 모드로 시작
    python3 03_preview.py --load          # 저장된 카메라 설정을 불러와 잠근 상태로 시작
    python3 03_preview.py --view-fps 15   # VNC가 느리면: 창 갱신만 초당 15회로 제한

VNC 화면에 'ColorLab' 창이 열린다. 창을 클릭해 포커스를 준 다음 키를 누른다.

  마우스 드래그 : 측정할 사각형 영역(ROI)을 고른다
  SPACE 또는 c  : ROI를 측정한다 (최근 N개 프레임의 평균)
  s             : 마지막 측정값을 이름을 붙여 저장한다  <- 이름은 '터미널'에 입력한다
                  (입력하는 동안 창이 멈춘 것처럼 보이는 것은 정상이다)
  f             : auto 모드로 돌아가 안정화한 뒤 AWB/AE(/초점)를 잠그고 + config JSON을 저장한다
  a             : auto 모드로 돌아간다
  l             : 저장된 config JSON을 불러와 다시 잠근다
  [ / ]         : 평균할 프레임 수를 줄인다 / 늘린다
  r             : ROI를 화면 중앙으로 초기화한다
  h             : 도움말 표시/숨김
  p             : 현재 카메라 메타데이터를 터미널에 출력한다
  q 또는 ESC    : 종료
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# VNC(XWayland)에서 실행될 때를 위해 기본 DISPLAY를 설정한다.
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorlab.camera import ColorCamera, DEFAULT_CONFIG_PATH
from colorlab.color import measure_roi, quick_mean_rgb
from colorlab.overlay import Overlay, panel
from colorlab.store import save_measurement

WINDOW = "ColorLab"
SWATCH = 44  # 색 견본 한 변의 길이 (px)


class RoiSelector:
    """마우스 드래그로 ROI를 고른다. 좌표는 '표시 화면' 기준이다."""

    def __init__(self):
        self.dragging = False
        self.start = (0, 0)
        self.cur = (0, 0)
        self.rect: tuple[int, int, int, int] | None = None  # 표시 화면 좌표

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.cur = (x, y)
            x0, x1 = sorted((self.start[0], x))
            y0, y1 = sorted((self.start[1], y))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                self.rect = (x0, y0, x1 - x0, y1 - y0)

    def live_rect(self) -> tuple[int, int, int, int] | None:
        if self.dragging:
            x0, x1 = sorted((self.start[0], self.cur[0]))
            y0, y1 = sorted((self.start[1], self.cur[1]))
            return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        return self.rect


def clamp_roi(roi, w, h):
    x, y, rw, rh = roi
    x = max(0, min(int(x), w - 2))
    y = max(0, min(int(y), h - 2))
    rw = max(2, min(int(rw), w - x))
    rh = max(2, min(int(rh), h - y))
    return (x, y, rw, rh)


def print_load_result(cam: ColorCamera, path: Path, s) -> None:
    print(f"\n[LOADED] {path}")
    for line in s.summary_lines():
        print("   ", line)
    for w in cam.load_warnings:
        print(f"  [WARNING] {w}")
    if cam.load_warnings:
        print("  [WARNING] Camera conditions differ from when the config was saved. "
              "Run with the same --main-width/--main-height, or press f to lock again.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-width", type=int, default=640, help="default is 640x480, same as data_collect/inference")
    ap.add_argument("--main-height", type=int, default=480)
    ap.add_argument("--display-width", type=int, default=0, help="VNC window width (0=original size, no resize)")
    ap.add_argument("--frames", type=int, default=10, help="number of frames to average when measuring")
    ap.add_argument("--view-fps", type=float, default=0,
                    help="max window update rate (0=no limit). If VNC is choppy, use about 15. Does not affect measuring")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--load", action="store_true", help="load config and lock right at start")
    ap.add_argument("--settle", type=float, default=3.0, help="minimum settle time before locking AWB/AE (seconds)")
    ap.add_argument("--max-settle", type=float, default=10.0, help="maximum time to wait for convergence (seconds)")
    ap.add_argument(
        "--no-swap", action="store_true",
        help="do not reverse channels (only if 01_test_camera.py said 'Case B')",
    )
    args = ap.parse_args()

    n_frames = max(1, args.frames)

    cam = ColorCamera(
        main_size=(args.main_width, args.main_height),
        channel_swap=not args.no_swap,
    )
    print("[START] Opening camera...")
    cam.start()
    time.sleep(1.0)

    if args.load:
        try:
            s = cam.load_config(args.config)
            print_load_result(cam, args.config, s)
        except FileNotFoundError:
            print(f"[WARNING] Config file not found: {args.config} (starting in auto mode)")

    ov = Overlay(default_size=15)
    if not ov.korean_ok:
        print("[INFO] No Korean font found. Korean names you type will show as '?' on screen.")
        print("       To fix: sudo apt install -y fonts-nanum")

    sel = RoiSelector()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, sel.on_mouse)

    last_sample = None
    last_frame_bgr = None
    last_roi_main = None
    last_locked = None  # '측정 순간'의 잠금 설정 (auto 모드면 None)
    last_warns: list[str] = []
    show_help = True
    status_msg = "Drag to select ROI, SPACE to measure"
    status_until = time.time() + 5
    status_color = (0, 255, 255)

    def flash(msg: str, sec: float = 3.0, color=(0, 255, 255)):
        nonlocal status_msg, status_until, status_color
        status_msg = msg
        status_until = time.time() + sec
        status_color = color
        print(f"    {msg}")

    # f 키: 먼저 안내 메시지가 있는 프레임을 보여 주고, 다음 루프에서 잠근다.
    pending_lock = False
    pending_lock_frame = -1

    fps, t_prev = 0.0, time.time()
    view_interval = 1.0 / args.view_fps if args.view_fps > 0 else 0.0
    t_last_show = 0.0
    frame_idx = 0
    print("[READY] Click the 'ColorLab' window on the VNC screen, then press keys. (q=quit)")

    try:
        while True:
            frame_idx += 1
            frame_bgr = cam.read_bgr()
            mh, mw = frame_bgr.shape[:2]

            if args.display_width <= 0 or args.display_width == mw:
                scale = 1.0
                disp = frame_bgr.copy()  # 오버레이가 스냅샷에 들어가지 않도록 복사한다
            else:
                scale = args.display_width / float(mw)
                disp = cv2.resize(frame_bgr, (args.display_width, int(mh * scale)))
            dh, dw = disp.shape[:2]

            # 아직 ROI가 없으면 화면 중앙 20% 영역을 쓴다
            drect = sel.live_rect()
            if drect is None:
                cw, ch = int(dw * 0.2), int(dh * 0.2)
                drect = (dw // 2 - cw // 2, dh // 2 - ch // 2, cw, ch)
                sel.rect = drect
            drect = clamp_roi(drect, dw, dh)

            # 표시 화면 좌표 -> 원본(main) 좌표
            roi_main = clamp_roi(
                (drect[0] / scale, drect[1] / scale, drect[2] / scale, drect[3] / scale), mw, mh
            )

            # 실시간 값 (현재 프레임만 -> 가볍고 빠르다)
            rgb_now = cam.to_rgb(frame_bgr)
            lr, lg, lb = quick_mean_rgb(rgb_now, roi_main)
            live_hex = "#%02X%02X%02X" % tuple(int(round(min(255.0, max(0.0, c)))) for c in (lr, lg, lb))
            live_y = 0.299 * lr + 0.587 * lg + 0.114 * lb

            # --- 그리기 ---------------------------------------------
            cv2.rectangle(disp, (drect[0], drect[1]),
                          (drect[0] + drect[2], drect[1] + drect[3]), (0, 255, 255), 2)

            panel(disp, 0, 0, dw, 46)
            lock_txt = ("LOCKED" if cam.is_locked else "AUTO")
            lock_col = (0, 255, 0) if cam.is_locked else (0, 200, 255)

            ov.begin(disp)
            ov.text(8, 5, f"[{lock_txt}]", lock_col, 17)
            meta_bits = []
            if cam.is_locked:
                meta_bits.append(f"exp {cam.locked.exposure_time}us")
                meta_bits.append(f"gain {cam.locked.analogue_gain:.2f}")
                meta_bits.append(
                    f"cg({cam.locked.colour_gains[0]:.2f},{cam.locked.colour_gains[1]:.2f})")
                if cam.locked.lens_position is not None:
                    meta_bits.append(f"lens {cam.locked.lens_position:.2f}")
            ov.text(90, 6, "  ".join(meta_bits), (220, 220, 220), 14)
            ov.text(dw - 190, 6, f"{fps:4.1f} fps   N={n_frames}", (200, 200, 200), 14)
            ov.text(8, 25, f"ROI {roi_main[2]}x{roi_main[3]}px  live: "
                    + f"RGB {lr:5.1f},{lg:5.1f},{lb:5.1f}  {live_hex}",
                    (255, 255, 255), 14)

            # 너무 어두우면 경고한다
            if live_y < 3:
                ov.text(8, dh - 28,
                        "!! Frame is nearly black - check lens cap / film / lighting",
                        (0, 100, 255), 16)

            # 측정 결과 패널
            if last_sample is not None:
                ph = 132 + (18 if last_warns else 0)
                pw = min(370, dw)
                panel(disp, 0, dh - ph, pw, ph)
                y = dh - ph + 6
                ov.text(8, y, "-- LAST MEASURE --", (0, 255, 255), 14)
                y += ov.line_height(14)
                for line in last_sample.summary_lines(en=not ov.korean_ok):
                    ov.text(8, y, line, (235, 235, 235), 13)
                    y += ov.line_height(13)
                if last_warns:
                    bits = []
                    if last_sample.clip_ratio > 0.01:
                        bits.append(f"clip {last_sample.clip_ratio*100:.1f}%")
                    if last_sample.dark_ratio > 0.01:
                        bits.append(f"dark {last_sample.dark_ratio*100:.1f}%")
                    if last_locked is None:
                        bits.append("AUTO mode")
                    ov.text(8, y, "! " + "  ".join(bits), (0, 80, 255), 13)
                # 색 견본: 패널 오른쪽에 공간이 있을 때만 그린다 (좁은 창에서 범위 초과 오류를 막는다)
                sx = pw + 6
                sy = dh - ph + 4
                if sx + SWATCH <= dw and sy >= 0:
                    disp[sy:sy + SWATCH, sx:sx + SWATCH] = (
                        int(last_sample.b), int(last_sample.g), int(last_sample.r))
                    cv2.rectangle(disp, (sx, sy), (sx + SWATCH, sy + SWATCH), (255, 255, 255), 1)

            # 도움말
            if show_help:
                lines = [
                    "drag: ROI   SPACE/c: measure   s: save (name in terminal)",
                    "f: lock AWB/AE   a: auto   l: load config   r: reset ROI",
                    "[ / ]: frames to average   p: metadata   h: help   q: quit",
                ]
                hh = 8 + len(lines) * ov.line_height(13)
                panel(disp, 0, 46, dw, hh)
                yy = 50
                for ln in lines:
                    ov.text(6, yy, ln, (200, 255, 200), 13)
                    yy += ov.line_height(13)

            if time.time() < status_until or pending_lock:
                sy = 46 + (8 + 3 * ov.line_height(13) if show_help else 0)
                panel(disp, 0, sy, dw, 24)
                ov.text(8, sy + 3, status_msg, status_color, 15)

            disp = ov.end()
            # VNC로 보내는 데이터를 줄이기 위해 창 갱신만 건너뛴다 (카메라/측정은 매 프레임 계속 돈다)
            # 잠금 대기 중에는 안내 메시지 프레임이 확실히 보이도록 항상 갱신한다
            if pending_lock or time.time() - t_last_show >= view_interval:
                cv2.imshow(WINDOW, disp)
                t_last_show = time.time()

            key = cv2.waitKey(1) & 0xFF

            # --- 대기 중인 잠금 실행 ------------------------------------
            # f 키를 누른 다음 프레임(= 안내 메시지가 있는 프레임)이 표시된 뒤에 실행한다.
            if pending_lock and frame_idx > pending_lock_frame:
                for _ in range(3):  # Qt가 창을 실제로 다시 그릴 시간을 준다
                    cv2.waitKey(30)
                s = cam.lock_from_current(settle_sec=args.settle, max_wait=args.max_settle)
                p = cam.save_config(args.config)
                pending_lock = False
                print("\n[LOCKED] AWB/AE turned off." + (" Focus is locked too." if s.lens_position is not None else ""))
                for line in s.summary_lines():
                    print("   ", line)
                print(f"[SAVED] {p}")
                if s.converged:
                    flash("AWB/AE locked + config saved", 3.0, (0, 255, 0))
                else:
                    print(f"  [WARNING] Exposure/white balance did not converge within {args.max_settle:.0f} s. "
                          "Press f again after the light is stable.")
                    flash("locked, but NOT converged! Press f again when light is stable",
                          6.0, (0, 80, 255))
                continue

            # --- 키 처리 --------------------------------------------
            if key in (ord("q"), 27):
                break

            elif key in (ord(" "), ord("c")):
                frames = cam.grab_rgb_frames(n_frames)
                last_sample = measure_roi(frames, roi_main)
                last_frame_bgr = frame_bgr.copy()
                last_roi_main = roi_main
                last_locked = cam.locked if cam.is_locked else None
                last_warns = last_sample.quality_warnings(locked=last_locked is not None)
                print(f"\n[MEASURE] ROI={roi_main}  mean of {n_frames} frames  "
                      f"({'locked' if last_locked else 'auto'} mode)")
                for line in last_sample.summary_lines():
                    print("   ", line)
                print(f"    Quality  clipped {last_sample.clip_ratio*100:.2f}%  dark {last_sample.dark_ratio*100:.2f}%")
                for w in last_warns:
                    print(f"  [WARNING] {w}")
                if last_warns:
                    flash("Measured WITH WARNINGS (see terminal) - s to save",
                          4.0, (0, 80, 255))
                else:
                    flash("Measured - press s to save")

            elif key == ord("s"):
                if last_sample is None:
                    flash("Measure first (SPACE)")
                else:
                    print("\n" + "=" * 50)
                    print("Type a name for this measurement (example: candidate_red_sunny)")
                    print("Press just Enter to cancel. (It is normal that the window is frozen while you type)")
                    try:
                        name = input(" Name > ").strip()
                    except (EOFError, KeyboardInterrupt):
                        name = ""
                    if name:
                        csv_p, img_p = save_measurement(
                            last_sample, name, last_roi_main, last_frame_bgr, last_locked
                        )
                        print(f"[SAVED] CSV      : {csv_p}")
                        print(f"[SAVED] Snapshot : {img_p}")
                        flash(f"saved: {name}")
                    else:
                        flash("save cancelled")
                    print("=" * 50 + "\n")

            elif key == ord("f"):
                pending_lock = True
                pending_lock_frame = frame_idx
                flash(f"settling in AUTO... (min {args.settle:.0f} s, max {args.max_settle:.0f} s) then locking",
                      args.max_settle + 2)

            elif key == ord("a"):
                cam.set_auto()
                flash("back to AUTO mode (AWB/AE on)")

            elif key == ord("l"):
                try:
                    s = cam.load_config(args.config)
                    print_load_result(cam, args.config, s)
                    if cam.load_warnings:
                        flash("loaded, but CONDITIONS DIFFER - see terminal",
                              5.0, (0, 80, 255))
                    else:
                        flash("config loaded + locked", 3.0, (0, 255, 0))
                except FileNotFoundError:
                    flash("no config file (press f to lock first)")

            elif key == ord("["):
                n_frames = max(1, n_frames - 1)
                flash(f"N = {n_frames}", 1.5)
            elif key == ord("]"):
                n_frames = min(60, n_frames + 1)
                flash(f"N = {n_frames}", 1.5)

            elif key == ord("r"):
                sel.rect = None
                flash("ROI reset", 1.5)

            elif key == ord("h"):
                show_help = not show_help

            elif key == ord("p"):
                meta = cam.metadata()
                print("\n[METADATA]")
                for k in ("ColourGains", "ExposureTime", "AnalogueGain", "DigitalGain",
                          "Lux", "ColourTemperature", "FrameDuration", "LensPosition", "ScalerCrop"):
                    if k in meta:
                        print(f"    {k:18s} = {meta[k]}")

            # fps 계산
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

    except KeyboardInterrupt:
        print("\n[STOPPED]")
    finally:
        cv2.destroyAllWindows()
        cam.close()
        print("[EXIT] Camera closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
