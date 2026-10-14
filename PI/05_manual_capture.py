#!/usr/bin/env python3
"""
05_manual_capture.py - 키보드로 서보를 움직이고, 움직일 때마다 사진을 한 장씩 저장한다.

05_sweep_capture.py와 같은 폴더 / CSV / label 구성을 쓰므로 두 스크립트가 하나의 데이터셋을 함께 쓸 수 있다.
labels.csv의 열 구성은 colorlab/dataset.py(CSV_COLUMNS)를 통해 05_sweep_capture.py와 공유하며,
mode 열에는 "manual"이 기록된다.
차이점: 정해진 범위를 자동으로 훑는 대신, 화면을 보면서 손으로 조준한다.
대상이 화면 밖으로 나가는 것을 눈으로 볼 수 있으므로 배경만 찍힌 사진이 쌓이지 않는다.

키 (카메라 창이나 터미널에서 누르며, Enter는 필요 없다):
    a / d     한 번 누르면 yaw를 왼쪽 / 오른쪽으로 계속 움직이며 한 칸(step)마다 사진 한 장.
              s를 누르거나 이동 한계(중앙 +/- --yaw-limit, 기본 30도)에 닿으면 멈춘다.
              움직이는 중에 반대 키를 누르면 방향이 바뀐다.
    A / D     자동 이동을 크게 (한 칸 = step 3배)
    s         자동 이동 멈춤
    z / c     yaw를 a / d와 같은 방향으로 한 칸 움직이기만 한다 (사진 없음, 조준 위치 맞추기용)
    Z / C     크게 이동 (step 3배), 사진 없음
    i / k     pitch 위 / 아래로 한 칸 움직인 뒤 사진 한 장 (I / K는 step 3배)
              (자동 이동 중에 z / c / i / k를 누르면 자동 이동은 멈춘다)
    SPACE     움직이지 않고 사진 한 장
    [ / ]     step을 줄이기 / 늘리기
    r         중앙으로 되돌린다 (사진 없음). 메뉴에서도, 촬영 중에도 쓸 수 있다.
              중앙값은 03_config/servo_center.json에서 읽는다
              (colorlab/servo.py, 파일이 없으면 기본값 yaw 95, pitch 90).
    q / ESC   이 label 촬영을 멈추고 메뉴로 돌아간다 (메뉴에서는 종료)
    Ctrl+C    지금까지 저장한 것은 그대로 두고 종료한다
  * 한글 입력 상태에서는 키가 전달되지 않는다. 키보드를 영문 입력 상태로 둔다.

메뉴:
    1 enemy / 2 friend / 3 both - 하나를 고르면 그 label로 촬영이 시작된다.

SG90 참고:
  - 명령은 us 단위로 보낸다. dead band(10 us = 약 0.9도)보다 작은 변화는 무시된다.
  - 수동 조작은 양방향으로 움직이므로 여기서는 backlash를 없앨 수 없다. 저장되는 각도는
    측정값이 아니라 명령값이다. 각도는 label이 아니라 참고값일 뿐이므로
    학습용 이미지에는 문제가 없다.
  - 키를 누르고 있으면 반복 입력이 쌓인다. 쌓인 이동은 한 번의 이동과 사진 한 장으로 합친다.

예시:
    python3 05_manual_capture.py
    python3 05_manual_capture.py --note left_1.8m_front
    python3 05_manual_capture.py --step 3 --settle 0.35
    python3 05_manual_capture.py --no-servo          # 카메라만 쓰고, 서보는 움직이지 않는다
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import select
import signal
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorlab.camera import ColorCamera, DEFAULT_CONFIG_PATH, DEFAULT_MAIN_SIZE
from colorlab.datacheck import dataset_mismatches, print_old_data_stop
from colorlab.dataset import CSV_COLUMNS
from colorlab.servo import SERVO_CENTER_PATH, load_center

# ---------------------------------------------------------------- 상수

YAW_PIN = 18
PITCH_PIN = 13

# 서보 중앙값은 03_config/servo_center.json 한 곳에서 읽는다 (없으면 yaw 95, pitch 90).
# 05_sweep_capture.py와 같은 값을 쓰며, 바꿀 때는 98_servo_center.py로 찾아서 저장한다.
CENTER_YAW, CENTER_PITCH = load_center()

SERVO_MIN_US = 500.0
SERVO_MAX_US = 2500.0
US_PER_DEG = (SERVO_MAX_US - SERVO_MIN_US) / 180.0   # 약 11.1 us/도
MIN_STEP_US = 10.0                                   # SG90의 dead band (이보다 작은 명령은 무시된다)

# 05_sweep_capture.py처럼 이 스크립트 옆에 둔다 (실행한 위치의 폴더가 아니다)
DATASET_DIR = Path(__file__).resolve().parent / "05_dataset"

LABEL_KEYS = {"1": "enemy", "2": "friend", "3": "both"}
LABEL_DESC = {"enemy": "enemy only", "friend": "friend only", "both": "friend + enemy"}
QUIT_KEYS = ("q", "\x1b")

# labels.csv 열 구성은 colorlab/dataset.py의 CSV_COLUMNS를 05_sweep과 함께 쓴다.
CSV_MODE = "manual"  # labels.csv의 mode 열에 기록되는 값

_quit = False
_stop_session = False


def _on_sigint(signum, frame):
    global _quit, _stop_session
    _quit = True
    _stop_session = True
    print("\n[INT] Stopping. Files saved so far are kept.", flush=True)


# ---------------------------------------------------------------- 키보드


class RawKeyboard:
    """Enter 없이 키 하나를 읽는다. SSH에서도 동작한다."""

    def __enter__(self):
        try:
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except (termios.error, ValueError):
            self.fd = None      # 예: 파이프로 실행한 경우
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def get(self, timeout: float = 0.0) -> str | None:
        if self.fd is None:
            return None
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None

    def flush(self) -> None:
        if self.fd is not None:
            termios.tcflush(self.fd, termios.TCIFLUSH)


# ---------------------------------------------------------------- 카메라 / 화면


def capture_bgr(cam: ColorCamera, rotate: int) -> np.ndarray:
    bgr = np.ascontiguousarray(cam.read_rgb()[:, :, ::-1])
    if rotate == 90:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif rotate == 270:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


class LiveView:
    WIN = "05 Manual Capture"

    def __init__(self, cam: ColorCamera, rotate: int, kb: RawKeyboard):
        self.cam = cam
        self.rotate = rotate
        self.kb = kb
        self.lines: list[str] = []
        self.pending: list[str] = []
        self.flash_until = 0.0
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        w, h = (480, 640) if rotate in (90, 270) else (640, 480)
        cv2.resizeWindow(self.WIN, w, h)

    def tick(self) -> None:
        frame = capture_bgr(self.cam, self.rotate)
        self._draw(frame)
        cv2.imshow(self.WIN, frame)
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            self.pending.append(chr(k))
        t = self.kb.get(0)
        if t:
            self.pending.append(t)

    def wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while True:
            self.tick()
            if time.monotonic() >= end:
                break

    def take_key(self) -> str | None:
        return self.pending.pop(0) if self.pending else None

    def drain_keys(self) -> list[str]:
        """쌓인 키를 한꺼번에 가져온다 (누르고 있는 키의 반복 입력을 합친다)."""
        keys, self.pending = self.pending, []
        return keys

    def clear_keys(self) -> None:
        self.pending.clear()
        self.kb.flush()

    def next_key(self) -> str | None:
        while not self.pending and not _quit:
            self.tick()
        return self.take_key()

    def flash(self) -> None:
        self.flash_until = time.monotonic() + 0.15

    def _draw(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        if self.lines:
            box_h = 12 + 22 * len(self.lines)
            roi = img[:box_h]
            roi[:] = (roi * 0.45).astype(np.uint8)
            for i, line in enumerate(self.lines):
                cv2.putText(img, line, (8, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # 중앙 십자선: 화면 안 어디에 대상을 둘지 정하는 데 도움이 된다
        cx, cy = w // 2, h // 2
        cv2.line(img, (cx - 14, cy), (cx + 14, cy), (0, 255, 255), 1)
        cv2.line(img, (cx, cy - 14), (cx, cy + 14), (0, 255, 255), 1)
        if time.monotonic() < self.flash_until:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)

    def close(self) -> None:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------- 서보


class Servos:
    def __init__(self, enabled: bool, settle: float, sleeper=time.sleep):
        self.enabled = enabled
        self.settle = settle
        self.sleep = sleeper
        self.pos: dict[int, float] = {}
        self.pi = None
        if enabled:
            import pigpio

            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise SystemExit("Cannot connect to pigpiod. Run 'sudo pigpiod' first.")

    @staticmethod
    def angle_to_us(angle: float) -> int:
        angle = max(0.0, min(180.0, angle))
        return int(round(SERVO_MIN_US + angle * US_PER_DEG))

    def write(self, pin: int, angle: float) -> float:
        angle = max(0.0, min(180.0, angle))
        if self.enabled:
            self.pi.set_servo_pulsewidth(pin, self.angle_to_us(angle))
        self.pos[pin] = angle
        return angle

    def goto(self, pin: int, angle: float, settle: float | None = None) -> float:
        a = self.write(pin, angle)
        self.sleep(self.settle if settle is None else settle)
        return a

    def get(self, pin: int, default: float) -> float:
        return self.pos.get(pin, default)

    def center(self, yaw: float, pitch: float) -> None:
        self.write(PITCH_PIN, pitch)
        self.write(YAW_PIN, yaw)
        self.sleep(max(0.6, self.settle))

    def release(self) -> None:
        if self.enabled and self.pi is not None:
            for pin in (YAW_PIN, PITCH_PIN):
                self.pi.set_servo_pulsewidth(pin, 0)
            self.pi.stop()


# ---------------------------------------------------------------- 저장


def _sync(f) -> None:
    f.flush()
    os.fsync(f.fileno())


class LabelStore:
    def __init__(self, root: Path, label: str):
        self.label = label
        self.dir = root / label
        self.img_dir = self.dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "labels.csv"
        self.sessions_path = self.dir / "sessions.jsonl"
        self._pattern = re.compile(rf"^{re.escape(label)}_(\d{{6}})\.jpg$")

    def next_index(self) -> int:
        biggest = 0
        for p in self.img_dir.glob(f"{self.label}_*.jpg"):
            m = self._pattern.match(p.name)
            if m:
                biggest = max(biggest, int(m.group(1)))
        if self.csv_path.exists():
            with self.csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    m = self._pattern.match(row.get("filename") or "")
                    if m:
                        biggest = max(biggest, int(m.group(1)))
        return biggest + 1

    def open_csv(self):
        is_new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        if not is_new:
            with self.csv_path.open(newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header != CSV_COLUMNS:
                raise SystemExit(f"[ERROR] {self.csv_path} has different columns: {header}\n"
                                 f"        Expected: {CSV_COLUMNS}\n"
                                 "        Move the old folder away and run again.")
        f = self.csv_path.open("a", newline="", encoding="utf-8")
        w = csv.writer(f)
        if is_new:
            w.writerow(CSV_COLUMNS)
            _sync(f)
        return f, w

    def save_image(self, name: str, bgr: np.ndarray, jpeg_quality: int) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed: {name}")
        tmp = self.img_dir / (name + ".tmp")
        with tmp.open("wb") as f:
            f.write(buf.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.img_dir / name)

    def log_session(self, record: dict) -> None:
        with self.sessions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            _sync(f)


def count_saved(root: Path) -> dict[str, int]:
    out = {}
    for label in LABEL_KEYS.values():
        d = root / label / "images"
        out[label] = len(list(d.glob(f"{label}_*.jpg"))) if d.exists() else 0
    return out


def quality(img: np.ndarray) -> tuple[float, float, float]:
    clip = float((img >= 250).any(axis=2).mean())
    dark = float((img <= 5).all(axis=2).mean())
    return clip, dark, float(img.mean())


# ---------------------------------------------------------------- 수동 촬영 session


# a / d: 한 번 누르면 s를 누를 때까지 yaw를 계속 움직이며 한 칸마다 사진을 찍는다 (자동 이동).
# 키 -> (방향, step 배수). 방향 +1 = a쪽(왼쪽), -1 = d쪽(오른쪽)
AUTO_KEYS = {"a": (+1.0, 1), "d": (-1.0, 1), "A": (+1.0, 3), "D": (-1.0, 3)}
STOP_KEY = "s"  # 자동 이동 멈춤

# 한 번 누를 때 한 번만 움직이는 키 -> (축, 방향, step 배수, 이동 후 촬영 여부)
MOVE_KEYS = {
    # z / c: a / d와 같은 방향으로 움직이기만 하고 촬영하지 않는다 (조준 위치 맞추기용)
    "z": ("yaw", +1.0, 1, False), "c": ("yaw", -1.0, 1, False),
    "Z": ("yaw", +1.0, 3, False), "C": ("yaw", -1.0, 3, False),
    # i / k: pitch 위 / 아래로 움직인 뒤 사진 한 장
    "i": ("pitch", +1.0, 1, True), "k": ("pitch", -1.0, 1, True),
    "I": ("pitch", +1.0, 3, True), "K": ("pitch", -1.0, 3, True),
}
SHOT_KEY = " "  # SPACE: 움직이지 않고 사진 한 장


def center_now(servos: Servos, args) -> None:
    """서보를 중앙값으로 정렬한다. 촬영은 하지 않고, 터미널에 정렬했다고 출력한다."""
    servos.center(args.center_yaw, args.center_pitch)
    src = SERVO_CENTER_PATH.name if SERVO_CENTER_PATH.exists() else "default"
    print(f"\n[CENTER] Centered: yaw {args.center_yaw:.1f}, pitch {args.center_pitch:.1f} "
          f"(from {src}, no photo)", flush=True)


def run_manual(label: str, args, cam: ColorCamera, servos: Servos, view: LiveView) -> None:
    """한 label에 대해 수동 촬영 session을 하나 실행한다. 이동 한 번에 사진 한 장을 찍는다."""
    global _stop_session
    _stop_session = False

    store = LabelStore(args.out, label)
    idx = store.next_index()
    first_idx = idx
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    locked = "yes" if cam.is_locked else "no"
    step = args.step
    # yaw 이동 한계: 중앙 +/- yaw_limit. 기구나 케이블에 걸리지 않도록 이 밖으로는 움직이지 않는다
    yaw_lo = args.center_yaw - args.yaw_limit
    yaw_hi = args.center_yaw + args.yaw_limit

    store.log_session({
        "session": session, "event": "start", "mode": "manual",
        "label": label, "note": args.note, "first_index": first_idx,
        "center": {"yaw": args.center_yaw, "pitch": args.center_pitch},
        "step_deg": step, "big_step_mult": 3, "settle_sec": args.settle,
        "rotate": args.rotate, "resolution": [args.width, args.height],
        "servo_enabled": not args.no_servo,
        "config": str(args.config) if cam.is_locked else None,
        "locked_settings": cam.locked.as_dict() if cam.is_locked else None,
        "note_on_angles": "All saved angles are command values. The SG90 cannot read its real angle.",
    })

    print(f"\n[START] Manual capture, label '{label}' ({LABEL_DESC[label]}) -> {store.dir}")
    print(f"        File numbers start at {label}_{first_idx:06d}")
    print("        a/d: keep moving + shooting (s: stop)   z/c: yaw move only   i/k: pitch + shot")
    print("        capital = x3   SPACE: shot   [ ]: step   r: center   q: stop")
    print(f"        yaw limit: {yaw_lo:.1f} ~ {yaw_hi:.1f} deg (center {args.center_yaw:.1f} "
          f"+/- {args.yaw_limit:g})")

    csv_file, writer = store.open_csv()
    saved = warn_clip = warn_dark = 0
    t0 = time.time()

    def status(extra: str = "") -> None:
        view.lines = [
            f"MANUAL {label.upper()}  saved {saved}  [{'LOCKED' if cam.is_locked else 'AUTO'}]",
            f"yaw {servos.get(YAW_PIN, args.center_yaw):6.1f}  "
            f"pitch {servos.get(PITCH_PIN, args.center_pitch):6.1f}  step {step:.1f}deg",
            extra or "a/d auto+shot  s stop  z/c move  i/k pitch  SPACE shot  q quit",
        ]

    def shoot(reason: str) -> None:
        nonlocal idx, saved, warn_clip, warn_dark
        img = capture_bgr(cam, args.rotate)
        clip, dark, mean = quality(img)
        name = f"{label}_{idx:06d}.jpg"
        store.save_image(name, img, args.jpeg_quality)
        writer.writerow([name, label, session, CSV_MODE, args.note,
                         f"{servos.get(YAW_PIN, args.center_yaw):.1f}",
                         f"{servos.get(PITCH_PIN, args.center_pitch):.1f}",
                         datetime.now().isoformat(timespec="seconds"),
                         f"{clip:.4f}", f"{dark:.4f}", f"{mean:.1f}", locked])
        _sync(csv_file)
        idx += 1
        saved += 1
        warn_clip += clip > 0.01
        warn_dark += dark > 0.05
        view.flash()
        print(f"\r  saved {saved}  {name}  yaw {servos.get(YAW_PIN, args.center_yaw):6.1f}  "
              f"pitch {servos.get(PITCH_PIN, args.center_pitch):6.1f}  ({reason})   ",
              end="", flush=True)

    def clamp_yaw(target: float) -> tuple[float, bool]:
        """yaw를 한계 안으로 자른다. (자른 값, 한계에 닿았는지)"""
        if target <= yaw_lo:
            return yaw_lo, True
        if target >= yaw_hi:
            return yaw_hi, True
        return target, False

    auto_dir = 0.0  # 자동 이동 방향 (0이면 멈춤)
    auto_mult = 1

    def stop_auto(why: str) -> None:
        nonlocal auto_dir
        if auto_dir:
            auto_dir = 0.0
            print(f"\n[AUTO] Stopped ({why}).", flush=True)
            status(f"auto stopped ({why})")

    try:
        status()
        while not _quit and not _stop_session:
            view.tick()
            keys = view.drain_keys()

            d_yaw = d_pitch = 0.0
            moved = False
            want_shot = False  # 이번에 모인 한 번 이동 키 중 촬영하는 키(i/k)가 있었는지
            for k in keys:
                if k in QUIT_KEYS:
                    _stop_session = True
                    break
                if k in AUTO_KEYS:
                    new_dir, auto_mult = AUTO_KEYS[k]
                    if new_dir != auto_dir:
                        side = "a (left)" if new_dir > 0 else "d (right)"
                        print(f"\n[AUTO] Moving toward {side}, one photo per step. Press s to stop.", flush=True)
                    auto_dir = new_dir
                    status("auto: moving + shooting (s: stop)")
                elif k == STOP_KEY:
                    stop_auto("s key")
                elif k in MOVE_KEYS:
                    stop_auto("manual move")  # 다른 이동 키를 누르면 자동 이동은 멈춘다
                    axis, sign, mult, shot = MOVE_KEYS[k]
                    want_shot |= shot
                    delta = sign * step * mult
                    if axis == "yaw":
                        d_yaw += delta
                    else:
                        d_pitch += delta
                    moved = True
                elif k == SHOT_KEY:
                    status("shot")
                    shoot("manual")
                elif k == "[":
                    step = max(round(MIN_STEP_US / US_PER_DEG, 1), round(step - 0.5, 1))
                    status(f"step {step:.1f}deg")
                elif k == "]":
                    step = min(15.0, round(step + 0.5, 1))
                    status(f"step {step:.1f}deg")
                elif k == "r":
                    stop_auto("centering")
                    status("centering")
                    center_now(servos, args)
                    status(f"centered: yaw {args.center_yaw:.1f} pitch {args.center_pitch:.1f} (no photo)")

            if _stop_session:
                break

            if moved and (d_yaw or d_pitch):
                # 한 번 이동: 쌓인 반복 입력을 한 번의 이동으로 합친다
                if d_pitch:
                    servos.goto(PITCH_PIN,
                                servos.get(PITCH_PIN, args.center_pitch) + d_pitch, settle=0.0)
                if d_yaw:
                    target, hit = clamp_yaw(servos.get(YAW_PIN, args.center_yaw) + d_yaw)
                    servos.goto(YAW_PIN, target, settle=0.0)
                    if hit:
                        print(f"\n[LIMIT] yaw {target:.1f} is the limit.", flush=True)
                if not want_shot:
                    # z / c: 움직이기만 한다. 기다리지 않아서 누르고 있으면 계속 부드럽게 움직인다
                    status("moved (no photo)")
                    continue
                status("moving")
                view.wait(args.settle)          # 흔들림이 멈출 때까지 기다린다 (화면은 계속 갱신된다)
                status("shot")
                for _ in range(args.shots):
                    shoot("move")
                view.clear_keys()               # 기다리는 동안 들어온 반복 입력은 버린다
                continue

            if auto_dir:
                # 자동 이동: 한 칸 움직이고, 흔들림이 멎으면 찍고, 다시 반복한다.
                # 기다리는 동안 눌린 키(s, 반대 방향 등)는 다음 반복에서 처리된다.
                cur = servos.get(YAW_PIN, args.center_yaw)
                target, hit = clamp_yaw(cur + auto_dir * step * auto_mult)
                if abs(target - cur) < 1e-6:   # 이미 한계에 있다
                    print(f"\n[LIMIT] yaw {cur:.1f} is the limit.", flush=True)
                    stop_auto("limit reached")
                    continue
                servos.goto(YAW_PIN, target, settle=0.0)
                status("auto: moving + shooting (s: stop)")
                view.wait(args.settle)
                for _ in range(args.shots):
                    shoot("auto")
                if hit:
                    print(f"\n[LIMIT] yaw {target:.1f} is the limit.", flush=True)
                    stop_auto("limit reached")

    finally:
        print()
        csv_file.close()
        store.log_session({
            "session": session, "event": "end", "label": label, "mode": "manual",
            "first_index": first_idx, "last_index": idx - 1 if saved else None,
            "saved_photos": saved, "seconds": round(time.time() - t0, 1),
        })

    last = f"{label}_{idx - 1:06d}" if saved else "-"
    print(f"[DONE] Saved {saved} photos in {time.time() - t0:.0f} sec "
          f"({label}_{first_idx:06d} ~ {last})")
    if warn_clip:
        print(f"[WARN] {warn_clip} photos have more than 1% clipped (too bright) pixels. "
              "Lower the exposure and make the config again.")
    if warn_dark:
        print(f"[WARN] {warn_dark} photos have more than 5% dark pixels. Think about adding more light.")


# ---------------------------------------------------------------- 메인


def show_menu(view: LiveView, root: Path, locked: bool) -> None:
    counts = count_saved(root)
    print("\n" + "=" * 60)
    for key, label in LABEL_KEYS.items():
        print(f"  {key} : {label:<7} ({LABEL_DESC[label]:<15})  saved so far: {counts[label]}")
    print("  r : center the servos (no photo)")
    print("  q : quit")
    print("=" * 60)
    print("Place the signs, then press a key (in the camera window or here).", flush=True)
    view.lines = [f"READY  [{'LOCKED' if locked else 'AUTO'}]",
                  "1: enemy   2: friend   3: both   r: center   q: quit",
                  f"saved  enemy {counts['enemy']}  friend {counts['friend']}  both {counts['both']}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="", help="Free note saved in labels.csv (e.g. left_1.8m_front)")
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                    help="Camera config saved by 03_preview.py (press f there)")
    ap.add_argument("--allow-auto", action="store_true",
                    help="Take photos in auto mode without a camera config file (not recommended)")

    ap.add_argument("--step", type=float, default=2.0, help="Degrees per step (key press or auto step)")
    ap.add_argument("--yaw-limit", type=float, default=30.0,
                    help="Yaw may move at most this many degrees left/right of the center")
    ap.add_argument("--center-yaw", type=float, default=CENTER_YAW)
    ap.add_argument("--center-pitch", type=float, default=CENTER_PITCH)
    ap.add_argument("--shots", type=int, default=1, help="Photos saved per move")
    ap.add_argument("--settle", type=float, default=0.4, help="Wait after a move before the shot (sec)")

    ap.add_argument("--width", type=int, default=DEFAULT_MAIN_SIZE[0])
    ap.add_argument("--height", type=int, default=DEFAULT_MAIN_SIZE[1])
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="Rotation before saving. Default 0: all data is saved as the camera sees it")
    ap.add_argument("--no-swap", action="store_true", help="Do not swap the R/B channels")
    ap.add_argument("--no-servo", action="store_true", help="Test the camera only")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    if args.step * US_PER_DEG < MIN_STEP_US:
        print(f"[ERROR] Step {args.step} deg = {args.step * US_PER_DEG:.1f} us is smaller than "
              f"the SG90 dead band ({MIN_STEP_US} us).")
        print(f"        Set it to {MIN_STEP_US / US_PER_DEG:.2f} deg or more.")
        return 1

    if not args.config.exists() and not args.allow_auto:
        print(f"[ERROR] Camera config file not found: {args.config}")
        print("        Press f in 03_preview.py to save fixed values, or add --allow-auto.")
        return 1

    # 출력 폴더에 다른 설정으로 찍은 사진이 이미 있으면 멈춘다
    old = dataset_mismatches(args.out, (args.width, args.height), args.rotate)
    if old:
        print_old_data_stop(args.out, old, (args.width, args.height), f", rotate {args.rotate}")
        return 1

    signal.signal(signal.SIGINT, _on_sigint)

    print("[START] Opening the camera...")
    cam = ColorCamera(main_size=(args.width, args.height), channel_swap=not args.no_swap)
    try:
        cam.start()
    except Exception as e:
        print(f"[ERROR] Could not open the camera: {e}", file=sys.stderr)
        return 1

    servos = None
    view = None
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
            print("[WARN] No config file, so photos are taken in auto mode. Colors may drift.")

        with RawKeyboard() as kb:
            view = LiveView(cam, args.rotate, kb)
            servos = Servos(enabled=not args.no_servo, settle=args.settle, sleeper=view.wait)

            if servos.enabled:
                print(f"[CENTER] Moving to the center (yaw {args.center_yaw}, pitch {args.center_pitch})...")
                view.lines = ["CENTERING...", "servos move to the center"]
            servos.center(args.center_yaw, args.center_pitch)

            while not _quit:
                show_menu(view, args.out, cam.is_locked)
                key = view.next_key()
                if _quit or key is None or key in QUIT_KEYS:
                    break
                if key == "r":
                    view.lines = ["CENTERING...", "servos move to the center"]
                    center_now(servos, args)
                    continue
                if key not in LABEL_KEYS:
                    if key.isprintable():
                        print(f"[INFO] Key '{key}' is not used. Press 1, 2, 3 or q.")
                    continue
                run_manual(LABEL_KEYS[key], args, cam, servos, view)
                view.clear_keys()

    finally:
        if servos is not None:
            if servos.enabled:
                try:
                    print("[CENTER] Returning to the center before releasing the servos...")
                    if view is not None:
                        view.lines = ["PARKING...", "servos return to the center"]
                    servos.center(args.center_yaw, args.center_pitch)
                except Exception as e:
                    print(f"[WARN] Could not return to the center: {e}")
            servos.release()
        if view is not None:
            view.close()
        cam.close()

    print("[EXIT] Camera closed, servos parked at the center and released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
