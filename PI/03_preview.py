#!/usr/bin/env python3
"""
[기능 1 + 2] 실시간 미리보기 + ROI 색 측정   (OpenCV 창 / VNC 화면용)

사용법:
    cd ~/src
    python3 03_preview.py                 # 자동 모드로 시작한다 (1280x960)
    python3 03_preview.py --load          # 저장된 카메라 설정을 불러와 잠근 상태로 시작한다
    python3 03_preview.py --view-fps 15   # VNC가 느리면: 창 갱신만 초당 15회로 제한한다

VNC 화면에 'ColorLab' 창이 열린다. 창을 클릭해 포커스를 준 뒤 키를 누른다.

  마우스 드래그 : 측정할 사각형 영역(ROI)을 고른다
  SPACE 또는 c  : ROI를 측정한다 (최근 N 프레임의 평균)
  s             : 마지막 측정값을 이름을 붙여 저장한다  <- 이름은 터미널에 입력한다
                  (입력하는 동안 창이 멈춘 것처럼 보이는 것은 정상이다)
  f             : 자동 모드로 돌아가 안정될 때까지 기다린 뒤 화이트 밸런스를 잠그고 config JSON을 저장한다.
                  기본은 "화이트 밸런스만 고정 + 노출 자동"이다 (시간대에 따라 밝기가 바뀌어도
                  카메라가 알아서 맞춘다). --fixed-exposure로 실행하면 예전처럼 노출 / gain까지 고정한다.
  a             : 자동 모드로 돌아간다
  l             : 저장된 config JSON을 불러와 다시 잠근다

  수동 조정 (잠긴 상태에서만):
  - / +         : 노출 자동일 때: 밝기 보정(EV) -0.25 / +0.25. 포화가 나면 - 로 조금 어둡게 한다
                  노출 고정일 때: 노출 시간 -10% / +10%   (= 도 + 로 동작한다)
  , / .         : analogue gain -0.25 / +0.25 (노출 고정일 때만)
  b             : ROI 기준으로 화이트 밸런스를 맞춘다. ROI를 흰 종이에 두고 b를 누른다.
                  카메라가 자동 모드이면 현재 값을 먼저 잠근다.
  w             : 조정한 값을 config JSON에 저장한다 (대문자 F도 같다. VNC에서 Shift 조합이 안 먹을 때가 있어 w를 쓴다)

  g             : 지금 화면을 오버레이 없이 원본 PNG(무손실)로 저장한다 → 03_captures/png/
                  같은 이름의 .json 에 카메라 설정(색 게인, 노출, EV, 해상도)을 함께 남긴다. 색 측정은 이 PNG 로만 한다.
  m             : 테스트 직전 점검. 검출기로 타겟을 찾아 아군/적군 판정과 여유(margin), 기준 조각 세 개의 상태,
                  노출 조정 방향을 화면(10초)과 터미널에 보여 준다. 처음 누를 때 모델을 올리느라 약 10초 걸린다.
                  검출기가 필요하므로 ~/src/pipeline/venv/bin/python 03_preview.py 로 실행해야 한다.

  [ / ]         : 평균 낼 프레임 수를 줄인다 / 늘린다
  r             : ROI를 화면 중앙으로 되돌린다
  h             : 도움말을 보이거나 숨긴다
  p             : 현재 카메라 metadata를 터미널에 출력한다
  q 또는 ESC    : 종료한다

위쪽에 항상 표시된다: 포화 % (어느 한 채널이라도 >= 250인 픽셀, 전체 프레임 기준),
어두움 % (모든 채널 <= 5), ROI RGB spread (R, G, B의 최대 - 최소; 작을수록 회색에 가깝다).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

# VNC 데스크톱(XWayland)에서 실행하도록 기본 DISPLAY를 설정한다.
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorlab import CAPTURES
from colorlab.camera import ColorCamera, DEFAULT_CONFIG_PATH, DEFAULT_MAIN_SIZE
from colorlab.color import CLIP_THR, DARK_THR, measure_roi, quick_mean_rgb
from colorlab.overlay import Overlay, panel
from colorlab.roi import RoiSelector, clamp_roi, display_to_main
from colorlab.store import save_measurement

WINDOW = "ColorLab"
SWATCH = 44  # 색 견본 한 변의 길이 (px)

EXPOSURE_STEP = 0.10  # -/+ 로 노출을 10 %씩 바꾼다
GAIN_STEP = 0.25  # ,/. 로 analogue gain을 0.25씩 바꾼다
EV_STEP = 0.25  # 노출 자동일 때 -/+ 로 밝기 보정(EV)을 0.25씩 바꾼다
WB_OK_SPREAD = 10.0  # max(R,G,B) - min(R,G,B)가 이 값 이하이면 화이트 밸런스가 맞은 것으로 본다
TOP_PANEL_H = 66


def print_load_result(cam: ColorCamera, path: Path, s) -> None:
    print(f"\n[LOADED] {path}")
    for line in s.summary_lines():
        print("   ", line)
    for w in cam.load_warnings:
        print(f"  [WARNING] {w}")
    hint = cam.resolution_hint()
    if hint:
        print(f"  [WARNING] {hint}")
    elif cam.load_warnings:
        print("  [WARNING] Camera conditions differ from when the config was saved. "
              "Press f to lock again.")


def frame_quality(rgb: np.ndarray) -> tuple[float, float]:
    """전체 프레임의 (포화 %, 어두움 %)를 구한다. 빠르게 하려고 한 픽셀씩 건너뛰어 쓴다."""
    sub = rgb[::2, ::2]
    sat = float((sub >= CLIP_THR).any(axis=-1).mean()) * 100.0
    dark = float((sub <= DARK_THR).all(axis=-1).mean()) * 100.0
    return sat, dark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-width", type=int, default=DEFAULT_MAIN_SIZE[0],
                    help="camera output width (standard 1280, same field of view as the old 640x480)")
    ap.add_argument("--main-height", type=int, default=DEFAULT_MAIN_SIZE[1])
    ap.add_argument("--display-width", type=int, default=640,
                    help="VNC window width (0 = original size). Measuring always uses the full image")
    ap.add_argument("--frames", type=int, default=10, help="number of frames to average when measuring")
    ap.add_argument("--view-fps", type=float, default=0,
                    help="max window update rate (0=no limit). If VNC is choppy, use about 15. Does not affect measuring")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--load", action="store_true", help="load config and lock right at start")
    ap.add_argument("--settle", type=float, default=3.0, help="minimum settle time before locking AWB/AE (seconds)")
    ap.add_argument("--max-settle", type=float, default=10.0, help="maximum time to wait for convergence (seconds)")
    ap.add_argument("--fixed-exposure", action="store_true",
                    help="f / b lock exposure and gain too (old way). Default: only white balance is fixed, exposure stays auto")
    ap.add_argument(
        "--no-swap", action="store_true",
        help="do not reverse channels (only if 01_test_camera.py said 'Case B')",
    )
    ap.add_argument("--png-dir", type=Path, default=CAPTURES / "png", help="where g saves PNG frames")
    ap.add_argument("--pipeline-config", type=Path,
                    default=Path("~/src/pipeline/tools/pipeline_config.json").expanduser(),
                    help="pipeline config for the m check (detector + color classifier)")
    args = ap.parse_args()

    n_frames = max(1, args.frames)

    cam = ColorCamera(
        main_size=(args.main_width, args.main_height),
        channel_swap=not args.no_swap,
    )
    print(f"[START] Opening camera at {args.main_width}x{args.main_height}...")
    try:
        cam.start()
    except Exception as e:
        print(f"[ERROR] Could not open the camera: {e}", file=sys.stderr)
        return 1
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
    last_locked = None  # 측정 시점의 잠금 설정 (자동 모드이면 None)
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

    def need_lock() -> bool:
        if cam.is_locked:
            return False
        flash("Lock first (press f, or l to load the config)", 3.0, (0, 80, 255))
        return True

    def report_tuning(what: str) -> None:
        """새 값과 새 프레임의 포화 비율을 출력한다."""
        sat, dark = frame_quality(cam.read_rgb())
        lk = cam.locked
        expo = (f"exposure AUTO EV {lk.exposure_value:+.2f}" if lk.auto_exposure
                else f"exposure {lk.exposure_time} us, gain {lk.analogue_gain:.2f}")
        print(f"\n[TUNE] {what}: {expo}, "
              f"cg({lk.colour_gains[0]:.3f}, {lk.colour_gains[1]:.3f})  "
              f"saturated {sat:.2f}%  dark {dark:.2f}%   (w saves)")
        flash(f"{what}: {expo}  sat {sat:.1f}%  (w saves)",
              3.0, (0, 255, 0) if sat <= 1.0 else (0, 80, 255))

    def white_balance(roi_main) -> None:
        """ROI를 회색/흰색으로 보고 R, B gain을 G에 맞게 조정한다."""
        if not cam.is_locked:
            cam.lock_now(note="locked as is for white balance", auto_exposure=not args.fixed_exposure)
            print("\n[WB] Camera was in auto mode, so the current values were locked first.")
        before = measure_roi(cam.grab_rgb_frames(n_frames), roi_main)
        r, g, b = before.r, before.g, before.b
        if min(r, g, b) < 20:
            flash("WB: ROI is too dark. Use white paper in good light.", 4.0, (0, 80, 255))
            return
        if before.clip_ratio > 0.01:
            flash("WB: ROI is saturated. Lower the exposure (-) first.", 4.0, (0, 80, 255))
            return
        old_r, old_b = cam.locked.colour_gains
        new_gains = (old_r * g / r, old_b * g / b)
        cam.adjust(colour_gains=new_gains, note="white balance (b)")
        after = measure_roi(cam.grab_rgb_frames(n_frames), roi_main)
        spread = max(after.r, after.g, after.b) - min(after.r, after.g, after.b)
        ok = spread <= WB_OK_SPREAD
        print(f"\n[WB] ROI before RGB ({r:.1f}, {g:.1f}, {b:.1f})  "
              f"gains ({old_r:.3f}, {old_b:.3f}) -> ({cam.locked.colour_gains[0]:.3f}, "
              f"{cam.locked.colour_gains[1]:.3f})")
        print(f"     ROI after  RGB ({after.r:.1f}, {after.g:.1f}, {after.b:.1f})  "
              f"spread {spread:.1f}  -> {'OK' if ok else 'press b again'}   (w saves)")
        if ok:
            flash(f"WB OK: spread {spread:.1f} (<= {WB_OK_SPREAD:.0f}). Press w to save.", 4.0, (0, 255, 0))
        else:
            flash(f"WB: spread {spread:.1f} > {WB_OK_SPREAD:.0f}. Press b again.", 4.0, (0, 80, 255))

    def true_bgr(raw_bgr):
        """cam.read_bgr() 버퍼를 cv2.imread 와 같은 진짜 BGR 순서로 바꾼다 (채널 설정과 무관하게)."""
        return np.ascontiguousarray(cam.to_rgb(raw_bgr)[:, :, ::-1])

    def save_png(raw_bgr) -> None:
        """오버레이 없는 원본 프레임을 PNG(무손실)로, 카메라 설정은 같은 이름의 json 으로 저장한다."""
        args.png_dir.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        path = args.png_dir / f"{stem}.png"
        cv2.imwrite(str(path), true_bgr(raw_bgr))
        meta = cam.metadata()
        info = dict(file=path.name, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    locked=cam.locked.as_dict() if cam.is_locked else None,
                    actual={k: (list(meta[k]) if isinstance(meta.get(k), tuple) else meta.get(k))
                            for k in ("ExposureTime", "AnalogueGain", "DigitalGain", "ColourGains",
                                      "Lux", "ColourTemperature")},
                    camera=cam.sensor_info())
        path.with_suffix(".json").write_text(json.dumps(info, indent=1, ensure_ascii=True, default=str),
                                             encoding="utf-8")
        print(f"\n[PNG] {path}  (+ .json with camera settings)")
        flash(f"PNG saved: {path.name}", 3.0, (0, 255, 0))

    pipe = None           # m 키용 파이프라인 (처음 누를 때 한 번만 올린다)
    m_view = None         # m 결과 (화면 표시용)
    m_until = 0.0

    def run_check(raw_bgr):
        """m 키: 검출 → 판정 → 기준 조각 상태 → 안내. 반환: 화면에 그릴 dict, 실패하면 None."""
        nonlocal pipe
        if pipe is None:
            tools_dir = str(args.pipeline_config.expanduser().resolve().parent)
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            try:
                from target_pipeline import Pipeline
            except ImportError as e:
                print(f"\n[CHECK] Cannot import the pipeline ({e}).")
                print("        Run 03 with the pipeline venv:  ~/src/pipeline/venv/bin/python 03_preview.py --load")
                flash("m needs ~/src/pipeline/venv/bin/python (see terminal)", 5.0, (0, 80, 255))
                return None
            print("\n[CHECK] Loading the detector (about 10 s, only the first time)...")
            with contextlib.redirect_stdout(sys.stderr):
                pipe = Pipeline(str(args.pipeline_config.expanduser()))
        out = pipe.run(true_bgr(raw_bgr))
        margin = pipe.color.margin_ab
        lines, problems = [], []
        refs = out["refs"]
        if refs is None:
            lines.append("refs: relative mode OFF (set color.ref_patches in pipeline_config.json)")
            problems.append("no_ref_patches")
        else:
            st = refs["status"]
            lines.append("refs: " + "  ".join(f"{k} {v}" for k, v in st.items()))
            if st["white"] == "saturated":
                problems.append("white patch saturated -> lower exposure (-)")
            elif st["white"] == "dark":
                problems.append("white patch dark -> raise exposure (+)")
            for k in ("friend", "enemy"):
                if st[k] != "ok":
                    problems.append(f"{k} patch {st[k]}")
        dets = sorted(out["dets"], key=lambda d: d.cx)
        sides = [d.side for d in dets]
        for d in dets:
            if d.mode == "relative":
                gap = abs(d.d_friend_ab - d.d_enemy_ab)
                lines.append(f"{d.side:7s} x{d.cx:5.0f}  gap {gap:5.1f} (margin {margin:g})  "
                             f"d_f {d.d_friend_ab:5.1f} d_e {d.d_enemy_ab:5.1f}  n_px {d.n_px}  {d.reason}")
                if gap < margin and refs is not None and refs["lab"].get("white") is not None:
                    L = float(refs["lab"]["white"][0])
                    hint = ("lower exposure (-)" if L > 85 else "raise exposure (+)" if L < 60
                            else "check target placement / color pair")
                    problems.append(f"gap {gap:.1f} < margin at x{d.cx:.0f}: white L* {L:.0f} -> {hint}")
            else:
                lines.append(f"{d.side:7s} x{d.cx:5.0f}  mode {d.mode}  n_px {d.n_px}  {d.reason}")
        if not dets:
            problems.append("no target detected")
        if "unknown" in sides:
            problems.append("unknown target(s) -> the pipeline would HOLD")
        if dets and not ("friend" in sides and "enemy" in sides):
            problems.append("expected one friend and one enemy")
        verdict = "PASS" if not problems else "CHECK"
        print(f"\n[CHECK] {verdict}   action={out['decision']['action']}")
        for ln in lines + [f"! {q_}" for q_ in problems]:
            print("   ", ln)
        ref_boxes = [] if pipe.color.ref is None else list(pipe.color.ref.rois.items())
        return dict(verdict=verdict, lines=lines, problems=problems, dets=dets, ref_boxes=ref_boxes,
                    ref_status=None if refs is None else refs["status"])

    # f 키: 먼저 안내 문구가 있는 프레임을 보여 주고, 다음 루프에서 잠근다.
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
                disp = frame_bgr.copy()  # 오버레이가 스냅숏에 들어가지 않도록 복사한다
            else:
                scale = args.display_width / float(mw)
                disp = cv2.resize(frame_bgr, (args.display_width, int(mh * scale)),
                                  interpolation=cv2.INTER_AREA)
            dh, dw = disp.shape[:2]

            # 아직 ROI가 없으면 화면 중앙 20 % 영역을 쓴다
            drect = sel.live_rect()
            if drect is None:
                cw, ch = int(dw * 0.2), int(dh * 0.2)
                drect = (dw // 2 - cw // 2, dh // 2 - ch // 2, cw, ch)
                sel.rect = drect
            drect = clamp_roi(drect, dw, dh)
            roi_main = display_to_main(drect, scale, mw, mh)  # 표시 좌표 -> main 이미지 좌표

            # 실시간 값 (현재 프레임만 사용 -> 가볍고 빠르다)
            rgb_now = cam.to_rgb(frame_bgr)
            lr, lg, lb = quick_mean_rgb(rgb_now, roi_main)
            live_hex = "#%02X%02X%02X" % tuple(int(round(min(255.0, max(0.0, c)))) for c in (lr, lg, lb))
            live_y = 0.299 * lr + 0.587 * lg + 0.114 * lb
            spread = max(lr, lg, lb) - min(lr, lg, lb)
            sat_pct, dark_pct = frame_quality(rgb_now)

            # --- 그리기 ---------------------------------------------
            cv2.rectangle(disp, (drect[0], drect[1]),
                          (drect[0] + drect[2], drect[1] + drect[3]), (0, 255, 255), 2)

            panel(disp, 0, 0, dw, TOP_PANEL_H)
            lock_txt = ("LOCKED" if cam.is_locked else "AUTO")
            lock_col = (0, 255, 0) if cam.is_locked else (0, 200, 255)

            ov.begin(disp)
            ov.text(8, 5, f"[{lock_txt}]", lock_col, 17)
            meta_bits = []
            if cam.is_locked:
                if cam.locked.auto_exposure:
                    meta_bits.append(f"exp AUTO EV{cam.locked.exposure_value:+.2f}")
                else:
                    meta_bits.append(f"exp {cam.locked.exposure_time}us")
                    meta_bits.append(f"gain {cam.locked.analogue_gain:.2f}")
                meta_bits.append(
                    f"cg({cam.locked.colour_gains[0]:.2f},{cam.locked.colour_gains[1]:.2f})")
                if cam.locked.lens_position is not None:
                    meta_bits.append(f"lens {cam.locked.lens_position:.2f}")
            ov.text(90, 6, "  ".join(meta_bits), (220, 220, 220), 14)
            ov.text(dw - 150, 6, f"{fps:4.1f} fps  N={n_frames}", (200, 200, 200), 14)
            # 품질 줄 (항상 표시한다)
            sat_col = (0, 255, 0) if sat_pct <= 1.0 else (0, 80, 255)
            spr_col = (0, 255, 0) if spread <= WB_OK_SPREAD else (0, 200, 255)
            ov.text(8, 25, f"sat {sat_pct:4.1f}%", sat_col, 14)
            ov.text(100, 25, f"dark {dark_pct:4.1f}%", (220, 220, 220), 14)
            ov.text(200, 25, f"ROI spread {spread:5.1f}", spr_col, 14)
            ov.text(8, 44, f"ROI {roi_main[2]}x{roi_main[3]}px  RGB {lr:5.1f},{lg:5.1f},{lb:5.1f}  {live_hex}",
                    (255, 255, 255), 14)

            # 프레임이 너무 어두우면 경고한다
            if live_y < 3:
                ov.text(8, dh - 28,
                        "!! Frame is nearly black - check lens cap / film / lighting",
                        (0, 100, 255), 16)

            # 마지막 측정값 패널
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
                # 색 견본: 패널 오른쪽에 자리가 있을 때만 그린다 (좁은 창에서 범위를 벗어나지 않게 한다)
                sx = pw + 6
                sy = dh - ph + 4
                if sx + SWATCH <= dw and sy >= 0:
                    disp[sy:sy + SWATCH, sx:sx + SWATCH] = (
                        int(last_sample.b), int(last_sample.g), int(last_sample.r))
                    cv2.rectangle(disp, (sx, sy), (sx + SWATCH, sy + SWATCH), (255, 255, 255), 1)

            # m 점검 결과 (10초 동안): 타겟 박스, 기준 조각 박스, 아래쪽 결과 패널
            if m_view is not None and time.time() < m_until:
                col = {"enemy": (196, 79, 126), "friend": (161, 163, 20), "unknown": (0, 200, 255)}
                for d in m_view["dets"]:
                    c1 = col.get(d.side, (255, 255, 255))
                    p1 = (int(d.x1 * scale), int(d.y1 * scale))
                    cv2.rectangle(disp, p1, (int(d.x2 * scale), int(d.y2 * scale)), c1, 2)
                    ov.text(p1[0], max(0, p1[1] - 18), d.side, c1, 13)
                for name, (x, y, w, h) in m_view["ref_boxes"]:
                    ok = m_view["ref_status"] is not None and m_view["ref_status"].get(name) == "ok"
                    c2 = (0, 255, 0) if ok else (0, 80, 255)
                    cv2.rectangle(disp, (int(x * scale), int(y * scale)),
                                  (int((x + w) * scale), int((y + h) * scale)), c2, 1)
                    ov.text(int(x * scale), int((y + h) * scale) + 2, name, c2, 12)
                rows = [f"CHECK: {m_view['verdict']}"] + m_view["lines"] + [f"! {q_}" for q_ in m_view["problems"]]
                rh = ov.line_height(13)
                py = dh - 8 - rh * len(rows)
                panel(disp, 0, py - 4, dw, rh * len(rows) + 12)
                for i, ln in enumerate(rows):
                    if i == 0:
                        c3 = (0, 255, 0) if m_view["verdict"] == "PASS" else (0, 80, 255)
                    else:
                        c3 = (0, 80, 255) if ln.startswith("!") else (235, 235, 235)
                    ov.text(6, py + i * rh, ln, c3, 13)

            # 도움말
            help_lines = [
                "drag: ROI  SPACE/c: measure  s: save  f: lock  a: auto  l: load",
                "-/+: brightness (EV) or exposure  ,/.: gain  b: WB (ROI)  w: save",
                "g: PNG  m: check  [ ]: frames  r: reset ROI  p: metadata  h: help  q: quit",
            ]
            if show_help:
                hh = 8 + len(help_lines) * ov.line_height(13)
                panel(disp, 0, TOP_PANEL_H, dw, hh)
                yy = TOP_PANEL_H + 4
                for ln in help_lines:
                    ov.text(6, yy, ln, (200, 255, 200), 13)
                    yy += ov.line_height(13)

            if time.time() < status_until or pending_lock:
                sy = TOP_PANEL_H + (8 + len(help_lines) * ov.line_height(13) if show_help else 0)
                panel(disp, 0, sy, dw, 24)
                ov.text(8, sy + 3, status_msg, status_color, 15)

            disp = ov.end()
            # VNC로 보내는 데이터를 줄이려고 창 갱신만 건너뛴다 (카메라 / 측정은 매 프레임 그대로 돈다).
            # 잠금을 기다리는 동안에는 안내 프레임이 실제로 보이도록 항상 갱신한다.
            if pending_lock or time.time() - t_last_show >= view_interval:
                cv2.imshow(WINDOW, disp)
                t_last_show = time.time()

            key = cv2.waitKey(1) & 0xFF

            # --- 대기 중인 잠금 -------------------------------------------
            # 안내 문구가 있는 프레임(f를 누른 뒤의 프레임)이 표시된 다음에 실행한다.
            if pending_lock and frame_idx > pending_lock_frame:
                for _ in range(3):  # Qt가 창을 실제로 다시 그릴 시간을 준다
                    cv2.waitKey(30)
                s = cam.lock_from_current(settle_sec=args.settle, max_wait=args.max_settle,
                                          auto_exposure=not args.fixed_exposure)
                p = cam.save_config(args.config)
                pending_lock = False
                what = ("White balance fixed, exposure stays AUTO." if s.auto_exposure
                        else "AWB/AE turned off (white balance and exposure fixed).")
                print("\n[LOCKED] " + what + (" Focus is locked too." if s.lens_position is not None else ""))
                for line in s.summary_lines():
                    print("   ", line)
                print(f"[SAVED] {p}  ({args.main_width}x{args.main_height})")
                if s.converged:
                    flash("AWB/AE locked + config saved", 3.0, (0, 255, 0))
                else:
                    print(f"  [WARNING] Exposure/white balance did not converge within {args.max_settle:.0f} s. "
                          "Press f again after the light is stable.")
                    flash("locked, but NOT converged! Press f again when light is stable",
                          6.0, (0, 80, 255))
                continue

            # --- 키 처리 --------------------------------------------------
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

            elif key in (ord("w"), ord("F")):
                if not need_lock():
                    p = cam.save_config(args.config)
                    print(f"\n[SAVED] {p}  ({args.main_width}x{args.main_height})")
                    for line in cam.locked.summary_lines():
                        print("   ", line)
                    flash("tuned values saved to config", 3.0, (0, 255, 0))

            elif key in (ord("-"), ord("+"), ord("=")):
                if not need_lock():
                    sign = -1.0 if key == ord("-") else 1.0
                    if cam.locked.auto_exposure:
                        # 노출 자동: 밝기 보정값(EV)만 바꾼다
                        cam.adjust(exposure_value=cam.locked.exposure_value + sign * EV_STEP)
                        report_tuning(f"EV {sign * EV_STEP:+.2f}")
                    else:
                        cam.adjust(exposure_time=cam.locked.exposure_time * (1.0 + sign * EXPOSURE_STEP))
                        report_tuning(f"exposure {sign * EXPOSURE_STEP * 100:+.0f}%")

            elif key in (ord(","), ord(".")):
                if not need_lock():
                    if cam.locked.auto_exposure:
                        flash("Exposure is AUTO: use - / + to change brightness (EV)", 3.0, (0, 200, 255))
                    else:
                        sign = -1.0 if key == ord(",") else 1.0
                        cam.adjust(analogue_gain=cam.locked.analogue_gain + sign * GAIN_STEP)
                        report_tuning(f"gain {sign * GAIN_STEP:+.2f}")

            elif key == ord("b"):
                white_balance(roi_main)

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

            elif key == ord("g"):
                save_png(frame_bgr)

            elif key == ord("m"):
                res = run_check(frame_bgr)
                if res is not None:
                    m_view, m_until = res, time.time() + 10.0
                    flash(f"check: {res['verdict']} (details in terminal)", 4.0,
                          (0, 255, 0) if res["verdict"] == "PASS" else (0, 80, 255))

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
