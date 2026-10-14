#!/usr/bin/env python3
"""
08_detect_check.py — 검출 + 아군/적군 판정 동작 확인 (확인 전용. 서보·발사는 전혀 건드리지 않는다. pigpio 를 import 하지 않는다)

사용법:
    cd ~/src && pipeline/venv/bin/python 08_detect_check.py
    cd ~/src && pipeline/venv/bin/python 99_fake_camera.py --image some.png 08_detect_check.py --load   # 카메라 없이

카메라 설정은 03_preview.py --load 와 같은 방식으로 03_config/camera_config.json 을 불러와 잠근다.
판정은 pipeline/tools/target_pipeline.Pipeline 을 그대로 쓴다 (실전과 같은 코드 경로, 모델은 시작할 때 한 번만 올린다).
화면은 640 폭으로 계속 갱신하고, 검출+판정은 별도 스레드에서 최신 프레임으로 반복한다 (1회 1~2 s).
설정 파일과 color_refs.json 은 읽기만 한다.

화면:
  박스 색   적군 빨강 / 아군 초록 / unknown 노랑.  라벨 'ENEMY a*+47', 'FRIEND a*-36', '? few_px'
  TARGET    실전에서 조준할 박스(판정이 engage 일 때 적군 중 h 최대) — 굵은 테두리
  HOLD      unknown 이 있어 실전이라면 보류할 때 — 화면 상단에 크게
  상태줄    검출 시간, 마지막 검출 후 경과 시간, 박스 수, 판정 모드, color_refs.json 상태와 안전비

키 (창을 클릭해 포커스를 준 뒤):
  q / ESC   종료
  g         지금 프레임을 판정해서 저장: 원본 PNG + 오버레이 PNG + 판정 JSON → 08_checks/
            (누른 순간의 프레임으로 다시 판정한 뒤 저장하므로 1~2 s 뒤에 저장된다)
  m         판정 모드 순환: 설정값 → a_sign → 설정값 (이 실행에서만. 설정 파일은 바꾸지 않는다)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

# VNC 데스크톱(XWayland)에서 실행하도록 기본 DISPLAY를 설정한다.
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # colorlab

import colorlab  # noqa: E402  (저장 폴더는 실행 시점의 colorlab.ROOT 기준 — 99_fake_camera 가 샌드박스로 바꾼다)
from colorlab.camera import ColorCamera, DEFAULT_CONFIG_PATH, DEFAULT_MAIN_SIZE  # noqa: E402

# 코드 위치는 colorlab 패키지 기준 (99_fake_camera 는 __file__ 을 샌드박스로 바꿔 실행하므로 __file__ 을 쓰지 않는다)
SRC = Path(colorlab.__file__).resolve().parent.parent
sys.path.insert(0, str(SRC / "pipeline" / "tools"))   # target_pipeline, colorlib

WINDOW = "DetectCheck"
FONT = cv2.FONT_HERSHEY_SIMPLEX
SIDE_COLOR = {"enemy": (0, 0, 255), "friend": (0, 255, 0), "unknown": (0, 255, 255)}   # BGR
SIDE_TEXT = {"enemy": "ENEMY", "friend": "FRIEND"}
DEFAULT_PIPELINE_CONFIG = SRC / "pipeline" / "tools" / "pipeline_config.json"


# ---------------------------------------------------------------- 판정 준비


def refs_state(cc, refs_path: str | None, min_safety: float) -> tuple[bool, str]:
    """(color_refs.json 을 쓸 수 있나, 상태 문구). 파일은 읽기만 한다."""
    if cc.cfg is not None:
        return True, f"refs loaded, safety {cc.cfg.get('safety_ratio', 0):.2f}"
    if not refs_path or not os.path.isfile(refs_path):
        return False, "refs MISSING"
    try:
        with open(refs_path, encoding="utf-8") as f:
            j = json.load(f)
    except (OSError, ValueError):
        return False, "refs unreadable"
    s = float(j.get("safety_ratio", 0) or 0)
    if s < min_safety:
        return False, f"refs BLOCKED, safety {s:.2f} < {min_safety:g}"
    if j.get("jpeg_input"):
        return False, "refs BLOCKED (jpeg_input)"
    return False, "refs invalid (need enemy + friend)"


def a_sign_classifier(T, config_path: str):
    """설정의 color 블록 그대로 mode 만 a_sign 으로 바꾼 ColorClassifier (m 키용). Pipeline 이 만드는 것과 같은 방식."""
    cfg, rel = T.read_config(config_path)
    col = {k: v for k, v in cfg.get("color", {}).items() if not k.startswith("_")}
    refs = rel(col.pop("refs", None))
    col["mode"] = "a_sign"
    return T.ColorClassifier(refs, **col), refs, float(col.get("min_safety_ratio", 3.0))


def mode_text(cc) -> str:
    if cc.mode == "relative" and cc.ref is None:
        return "relative (no ref patches -> absolute)"
    return cc.mode


# ---------------------------------------------------------------- 검출 스레드


class CheckWorker(threading.Thread):
    """최신 프레임으로 Pipeline.run 을 반복한다. g 저장 요청이 있으면 그 프레임을 먼저 판정해서 저장한다."""

    def __init__(self, pipe, alt_cc, colorlib, refs_ok: bool, refs_desc: str):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.cc = {"config": pipe.color, "a_sign": alt_cc}
        self.C = colorlib
        self.refs_ok, self.refs_desc = refs_ok, refs_desc
        self.want = "config"               # m 키가 바꾼다. 다음 판정부터 적용
        self._frame, self._fid, self._seen = None, 0, 0
        self._save = None                  # (frame, camera_meta, out_dir)
        self.result: dict | None = None
        self.stop = threading.Event()

    def submit(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._fid += 1

    def request_save(self, frame: np.ndarray, meta: dict, out_dir: Path) -> None:
        self._save = (frame, meta, out_dir)

    def run_once(self, frame: np.ndarray) -> dict:
        key = self.want
        self.pipe.color = self.cc[key]
        t = time.perf_counter()
        out = self.pipe.run(frame)
        ms = (time.perf_counter() - t) * 1000
        # 화면용 a*: 판정이 a* 를 잰 경우 그 값, 아니면(no_refs 등) 같은 고채도 픽셀 선택으로 잰다 (판정에는 쓰지 않음)
        a_disp = []
        for d in out["dets"]:
            if d.a_star is not None:
                a_disp.append(d.a_star)
            else:
                st = self.C.patch_stats(frame, d.xyxy, sat_min=self.pipe.color.sat_min)
                a_disp.append(None if st["lab"] is None else float(st["lab"][1]))
        a_sign = self.pipe.color.mode == "a_sign"
        return dict(out=out, frame=frame, a=a_disp, ms=ms, t=time.monotonic(), wall=time.strftime("%H:%M:%S"),
                    key=key, mode=mode_text(self.pipe.color), refs_ok=a_sign or self.refs_ok,
                    refs="refs not used (a_sign)" if a_sign else self.refs_desc)

    def run(self) -> None:
        while not self.stop.is_set():
            job, self._save = self._save, None
            if job is not None:
                frame = job[0]
            elif self._frame is None or self._fid == self._seen:
                time.sleep(0.01)
                continue
            else:
                frame, self._seen = self._frame, self._fid
            try:
                res = self.run_once(frame)
            except Exception as e:          # 판정 실패는 알리고 계속 돈다
                print(f"\n[WARN] detect/classify failed: {e}", flush=True)
                time.sleep(0.5)
                continue
            self.result = res
            print(result_line(res), flush=True)
            if job is not None:
                try:
                    save_check(res, job[1], job[2])
                except Exception as e:
                    print(f"[WARN] save failed: {e}", flush=True)


# ---------------------------------------------------------------- 문구


def box_label(d, a) -> str:
    a_txt = f"a*{a:+.0f}" if a is not None else "a*?"
    if d.side in SIDE_TEXT:
        return f"{SIDE_TEXT[d.side]} {a_txt}"
    return f"? {d.reason}" + (f" {a_txt}" if a is not None else "")


def hold_reason(out) -> str:
    """decide() 의 보류 이유를 화면용 ASCII 로 (cv2 글꼴은 한글을 못 그린다)."""
    unk = [d.reason for d in out["dets"] if d.side == "unknown"]
    return f"unknown {len(unk)}: {', '.join(sorted(set(unk)))}"


def result_line(res) -> str:
    out = res["out"]
    f = lambda v: f"{v:5.1f}" if v is not None else "    -"
    boxes = [f"({d.cx:4.0f},{d.h:3.0f}) {d.side:7s} a* {f(a)} d_f {f(d.d_friend_ab)} d_e {f(d.d_enemy_ab)}"
             + ("" if d.reason == "ok" else f" [{d.reason}]")
             for d, a in sorted(zip(out["dets"], res["a"]), key=lambda p: p[0].cx)]
    dec = out["decision"]
    return (f"[{res['wall']}] {res['ms']:5.0f} ms  {res['key']:6s} {len(out['dets'])} box | "
            + (" | ".join(boxes) if boxes else "-") + f"  -> {dec['action']}: {dec['reason']}")


# ---------------------------------------------------------------- 그리기


def text(img, s: str, org, scale: float, color, thick: int = 1) -> None:
    """검은 외곽선을 깐 글자 (밝은 벽, 어두운 벽 어디서나 읽힌다)."""
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def panel(img, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = cv2.convertScaleAbs(roi, alpha=1 - alpha)


def banner(img, s: str, color, y: int, scale: float = 0.9) -> None:
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, 2)
    x = max(6, (img.shape[1] - tw) // 2)
    panel(img, x - 8, y - th - 8, tw + 16, th + 16, 0.65)
    cv2.putText(img, s, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def draw_boxes(img, res, scale: float) -> None:
    out = res["out"]
    tgt = out["decision"]["target"]
    for d, a in zip(out["dets"], res["a"]):
        col = SIDE_COLOR.get(d.side, SIDE_COLOR["unknown"])
        p1 = (int(round(d.x1 * scale)), int(round(d.y1 * scale)))
        p2 = (int(round(d.x2 * scale)), int(round(d.y2 * scale)))
        is_tgt = d is tgt
        cv2.rectangle(img, p1, p2, col, 4 if is_tgt else 2)
        label = box_label(d, a)
        lx = min(p1[0], img.shape[1] - cv2.getTextSize(label, FONT, 0.5, 3)[0][0] - 2)   # 오른쪽 끝에서 잘리지 않게
        text(img, label, (max(0, lx), max(14, p1[1] - 6)), 0.5, col)
        if is_tgt:
            tx = min(p1[0], img.shape[1] - cv2.getTextSize("TARGET", FONT, 0.65, 4)[0][0] - 2)
            text(img, "TARGET", (max(0, tx), min(img.shape[0] - 4, p2[1] + 20)), 0.65, col, 2)


def draw_banners(img, res, refs_ok: bool, current_key: str, top: int) -> None:
    """HOLD / NO TARGET / COLOR REFS NOT USABLE."""
    y = top + 34
    if res is not None:
        out = res["out"]
        if not out["dets"]:
            banner(img, "NO TARGET", (255, 255, 255), y)
        elif out["decision"]["action"] == "hold":
            banner(img, f"HOLD - {hold_reason(out)}", (0, 255, 255), y)
        y += 46
    if not refs_ok and current_key != "a_sign":
        banner(img, "COLOR REFS NOT USABLE - press m for a_sign", (0, 128, 255), y, 0.6)


def status_lines(res, cam_label: str, fps: float, mode_desc: str, refs_desc: str, pending: str) -> list[str]:
    if res is None:
        l1 = f"{cam_label}  view {fps:4.1f} fps  detecting... (first run is slow)"
    else:
        age = time.monotonic() - res["t"]
        l1 = (f"{cam_label}  view {fps:4.1f} fps  det {res['ms']:.0f} ms  age {age:4.1f} s  "
              f"boxes {len(res['out']['dets'])}  -> {res['out']['decision']['action']}")
    return [l1, f"mode {mode_desc}{pending}  |  {refs_desc}", "q quit  g save  m mode"]


def draw_status(img, lines: list[str], scale: float = 0.45) -> int:
    lh = int(22 * scale / 0.45)
    panel(img, 0, 0, img.shape[1], 8 + lh * len(lines))
    for i, s in enumerate(lines):
        cv2.putText(img, s, (6, lh * (i + 1) - 2), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return 8 + lh * len(lines)


# ---------------------------------------------------------------- 저장


def camera_meta(cam: ColorCamera) -> dict:
    meta = cam.metadata()
    return dict(locked=cam.locked.as_dict() if cam.is_locked else None,
                actual={k: (list(meta[k]) if isinstance(meta.get(k), tuple) else meta.get(k))
                        for k in ("ExposureTime", "AnalogueGain", "DigitalGain", "ColourGains",
                                  "Lux", "ColourTemperature")})


def save_check(res, cam_info: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    frame, out = res["frame"], res["out"]
    cv2.imwrite(str(out_dir / f"{stem}.png"), frame)
    ov = frame.copy()
    draw_boxes(ov, res, 1.0)
    top = draw_status(ov, [f"{res['wall']}  det {res['ms']:.0f} ms  boxes {len(out['dets'])}  "
                           f"-> {out['decision']['action']}", f"mode {res['mode']}  |  {res.get('refs', '')}"], 0.6)
    draw_banners(ov, res, res["refs_ok"], "a_sign" if res["refs_ok"] else res["key"], top)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), ov)
    tgt = out["decision"]["target"]
    rnd = lambda d: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(d).items()}
    info = dict(
        file=f"{stem}.png", overlay=f"{stem}_overlay.png", saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        mode=res["mode"], mode_key=res["key"], refs=res.get("refs"), detect_ms=round(res["ms"], 1),
        dets=[dict(rnd(d), label=box_label(d, a), a_star_display=None if a is None else round(a, 2), is_target=d is tgt)
              for d, a in zip(out["dets"], res["a"])],
        decision=dict(action=out["decision"]["action"], reason=out["decision"]["reason"],
                      target_index=next((i for i, d in enumerate(out["dets"]) if d is tgt), None)),
        refs_reading=None if out["refs"] is None else dict(ok=out["refs"]["ok"], status=out["refs"]["status"]),
        camera=cam_info)
    (out_dir / f"{stem}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[SAVED] {out_dir / stem}.png  (+ _overlay.png, .json)", flush=True)
    return out_dir / f"{stem}.png"


# ---------------------------------------------------------------- 본체


def main() -> int:
    ap = argparse.ArgumentParser(description="Detection + friend/enemy check (no servo, no firing)")
    ap.add_argument("--main-width", type=int, default=DEFAULT_MAIN_SIZE[0])
    ap.add_argument("--main-height", type=int, default=DEFAULT_MAIN_SIZE[1])
    ap.add_argument("--display-width", type=int, default=640, help="window width (0 = original size)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="camera config (same as 03 --load)")
    ap.add_argument("--load", action="store_true", help="accepted for symmetry with 03; the config is always loaded")
    ap.add_argument("--no-swap", action="store_true", help="do not reverse channels (only if 01_test_camera.py said 'Case B')")
    ap.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=None, help="where g saves (default: ~/src/08_checks)")
    args = ap.parse_args()
    out_dir = args.out_dir or (colorlab.ROOT / "08_checks")

    import colorlib as C            # noqa: E402  (pipeline/tools)
    import target_pipeline as T     # noqa: E402

    cam = ColorCamera(main_size=(args.main_width, args.main_height), channel_swap=not args.no_swap)
    print(f"[START] Opening camera at {args.main_width}x{args.main_height}...")
    try:
        cam.start()
    except Exception as e:
        print(f"[ERROR] Could not open the camera: {e}", file=sys.stderr)
        return 1
    time.sleep(1.0)
    cam_label = "AUTO"
    try:
        s = cam.load_config(args.config)
        cam_label = "LOCKED"
        print(f"[LOCKED] {args.config}")
        for line in s.summary_lines():
            print("    " + line)
        for w in cam.load_warnings:
            print(f"[WARN] {w}")
        hint = cam.resolution_hint()
        if hint:
            print(f"[WARN] {hint}")
    except FileNotFoundError:
        print(f"[WARN] Camera config not found: {args.config} -> camera stays in AUTO mode (colors will drift)")

    worker = None
    try:
        print("[START] Loading the detector + color classifier (about 10 s)...")
        cfg_path = str(args.pipeline_config.expanduser())
        with contextlib.redirect_stdout(sys.stderr):    # ultralytics 로딩 로그
            pipe = T.Pipeline(cfg_path)
            alt_cc, refs_path, min_safety = a_sign_classifier(T, cfg_path)
        refs_ok, refs_desc = refs_state(pipe.color, refs_path, min_safety)
        print(f"[COLOR] mode {mode_text(pipe.color)}  |  {refs_desc}  ({refs_path})")
        if pipe.color.mode == "a_sign":
            refs_ok = True
        elif not refs_ok:
            print("[COLOR] COLOR REFS NOT USABLE: every target will be unknown -> HOLD. "
                  "Press m to check with the a_sign mode (this run only).")

        worker = CheckWorker(pipe, alt_cc, C, refs_ok, refs_desc)
        worker.start()

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        print("[READY] Click the 'DetectCheck' window, then press keys. (q quit, g save, m mode)")
        fps, t_prev = 0.0, time.time()
        flash, flash_until = "", 0.0
        while True:
            frame = np.ascontiguousarray(cam.read_rgb()[:, :, ::-1])   # cv2.imread 와 같은 진짜 BGR
            worker.submit(frame)
            mh, mw = frame.shape[:2]
            scale = 1.0 if args.display_width <= 0 or args.display_width == mw else args.display_width / mw
            disp = cv2.resize(frame, (int(mw * scale), int(mh * scale)), interpolation=cv2.INTER_AREA) \
                if scale != 1.0 else frame.copy()

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3) if fps else 1.0 / max(now - t_prev, 1e-3)
            t_prev = now
            res = worker.result
            if res is not None:
                draw_boxes(disp, res, scale)
            cur_cc = worker.cc[worker.want]
            pending = "" if res is None or res["key"] == worker.want else "  (switching...)"
            ref_txt = "refs not used (a_sign)" if cur_cc.mode == "a_sign" else refs_desc
            top = draw_status(disp, status_lines(res, cam_label, fps, mode_text(cur_cc), ref_txt, pending))
            draw_banners(disp, res, refs_ok, worker.want, top)
            if now < flash_until:
                text(disp, flash, (8, disp.shape[0] - 10), 0.55, (0, 255, 0), 1)
            cv2.imshow(WINDOW, disp)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("g"):
                worker.request_save(frame, camera_meta(cam), out_dir)
                flash, flash_until = f"saving to {out_dir.name}/ after the next detection...", now + 3
                print("\n[SAVE] judging this frame, then saving...", flush=True)
            elif k == ord("m"):
                if worker.cc["config"].mode == "a_sign":
                    flash, flash_until = "config mode is already a_sign", now + 3
                else:
                    worker.want = "a_sign" if worker.want == "config" else "config"
                    msg = f"mode -> {mode_text(worker.cc[worker.want])} (this run only, config file unchanged)"
                    flash, flash_until = msg, now + 3
                    print(f"\n[MODE] {msg}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if worker is not None:
            worker.stop.set()
            worker.join(timeout=5)
        cam.close()
        cv2.destroyAllWindows()
        print("[END]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
