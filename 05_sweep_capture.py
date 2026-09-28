#!/usr/bin/env python3
"""
05_sweep_capture.py - servo를 단계별로 움직이며 학습용 사진을 자동으로 찍는다.

표지판을 세워 두고 키보드로 라벨을 고르면, pitch를 몇 단계로 바꾸고
각 pitch 단계마다 yaw를 왼쪽에서 오른쪽으로 sweep하면서 각 위치에서 멈춘 뒤
사진을 저장한다. 한 번 sweep이 끝나면 다시 메뉴로 돌아오므로, 표지판을 옮기고
다음 라벨 키를 누르면 된다.

VNC 데스크톱에서 실행한다. '05 Sweep Capture' 창에 카메라 화면이 항상 보인다
(메뉴에서 기다릴 때, sweep 중 servo가 움직이고 멈추기를 기다릴 때 모두).
사진이 저장되는 순간 화면 테두리가 빨갛게 깜빡인다.

servo 위치:
    시작할 때 항상 중앙(yaw 50, pitch 90)으로 정렬한 뒤 메뉴를 띄운다.
    끝날 때(q, Ctrl+C, 오류 모두) 중앙으로 돌려놓은 뒤 힘을 푼다. 그래서 다음 실행 때
    시작 정렬이 거의 움직이지 않는다. 재부팅/전원 차단으로 끝난 경우에는 다음 시작 때
    처진 위치에서 중앙으로 한 번 크게 움직인다.
    (발사 servo는 건드리지 않는다. 수집할 때는 고무줄을 걸지 않는다.)

키 (카메라 창 또는 터미널 어느 쪽에서 눌러도 된다, Enter 불필요):
    1   적군(enemy) 라벨로 sweep 시작
    2   아군(friend) 라벨로 sweep 시작
    3   아군+적군(both) 라벨로 sweep 시작
    q   sweep 중이면 sweep만 멈춤 / 메뉴에서는 프로그램 종료 (ESC도 같음)
    Ctrl+C  (터미널에서) 지금 사진까지 저장하고 프로그램 종료
  * 한글 입력 모드에서는 q가 'ㅂ'으로 들어가 동작하지 않는다. 영문 모드로 바꿔서 누른다.

카메라 설정:
    03_preview.py에서 f로 저장한 03_config/camera_config.json을 03/04와 같은 방식
    (colorlab)으로 불러와 AWB/AE를 고정한다. 해상도/화각이 저장할 때와 다르면 경고한다.

저장 (라벨별 폴더에 계속 이어서 저장):
    05_dataset/<라벨>/
        images/        <라벨>_000001.jpg, <라벨>_000002.jpg, ...
        labels.csv     사진마다 한 줄: 라벨, 명령 각도, 품질 값, sweep 번호
        sessions.jsonl sweep마다 한 줄: 촬영 조건, 시작/끝 번호
    - 파일 번호는 폴더 안의 가장 큰 번호(이미지와 labels.csv 모두 확인) + 1부터 시작한다.
    - 폴더가 이미 있으면(재부팅 후 다시 실행 등) 그대로 이어서 저장한다.
    - 사진은 임시 파일에 먼저 쓴 뒤 이름을 바꾸므로, 저장 중 전원이 꺼져도
      깨진 사진 파일이 남지 않는다. labels.csv도 줄마다 디스크에 바로 기록한다.

SG90 대응:
  - 명령은 us 단위로 보낸다. 최소 스텝(기본 10 us ~ 0.9 deg)보다 작은 변화는
    반응하지 않으므로, 시작할 때 각도 간격이 그보다 큰지 확인한다.
  - backlash를 없애기 위해 yaw와 pitch 모두 항상 "올라가는 방향"으로만 목표에 접근한다.
    되돌아갈 때는 목표보다 충분히 아래로 먼저 내려갔다가 다시 올라온다.
  - SG90은 자기 각도를 읽을 수 없으므로, 저장되는 값은 모두 "명령 각도"다.

예시:
    python3 05_sweep_capture.py
    python3 05_sweep_capture.py --note left_1.8m_front     # 메모는 labels.csv에 기록된다
    python3 05_sweep_capture.py --yaw-range 12 --yaw-step 4 --pitch-offsets -6,-3,0,3,6
    python3 05_sweep_capture.py --no-servo --yaw-range 0 --pitch-offsets 0   # 카메라만 확인
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

# VNC(XWayland)에서 실행될 때를 위해 DISPLAY를 기본 지정한다 (03_preview.py와 같음).
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from colorlab.camera import ColorCamera, DEFAULT_CONFIG_PATH  # noqa: E402

# ======================================================================
# [TODO 2026-09-30 직접 확인 후 수정] 스윕 범위
#
#   카메라를 90도 돌려 달아서 가로 화각이 약 41도(중심에서 좌우 약 20도)다.
#   기본 ±20도로 돌면 가운데 둔 표지판이 스윕 양 끝에서 화면 밖으로 나갈 수 있다.
#
#   확인 방법 (VNC 터미널, 카메라 연결 후):
#     python3 05_sweep_capture.py --out /tmp/range_test --pitch-offsets 0
#     -> 표지판을 정면 가운데 두고 1을 누른다
#     -> '05 Sweep Capture' 창에서 yaw -20 과 +20 일 때 표지판이 화면 안에 있는지 본다
#        (창 위쪽에 현재 yaw 값이 표시된다)
#     -> /tmp/range_test 는 테스트용이라 확인 후 지워도 된다 (진짜 데이터에 섞이지 않음)
#
#   결정한 값을 YAW_RANGE에 넣고 YAW_RANGE_CONFIRMED를 True로 바꾼다.
#   (확인 전까지는 실행할 때마다 터미널에 [TODO] 알림이 나온다)
# ======================================================================
YAW_RANGE = 20.0  # 중앙 기준 좌우 범위 (deg)  <- 여기 수정 (예: 12.0)
YAW_STEP = 4.0  # yaw 간격 (deg)
PITCH_OFFSETS = "-6,-3,0,3,6"  # pitch 단계 (중앙 기준 deg)
YAW_RANGE_CONFIRMED = False  # 확인 후 True

# ---------------------------------------------------------------- 기본값

YAW_PIN = 18
PITCH_PIN = 13

CENTER_YAW = 50.0
CENTER_PITCH = 90.0

SERVO_MIN_US = 500.0
SERVO_MAX_US = 2500.0
US_PER_DEG = (SERVO_MAX_US - SERVO_MIN_US) / 180.0  # ~11.1 us/deg

MIN_STEP_US = 10.0  # SG90 dead band. 이보다 작은 변화는 무시된다
BACKLASH_MARGIN = 5.0  # 되돌아갈 때 다시 올라오기 전에 목표보다 얼마나 아래로 내려갈지 (deg)

DATASET_DIR = HERE / "05_dataset"

# 키 -> 라벨 (폴더 이름과 파일 이름 앞부분으로도 쓴다)
LABEL_KEYS = {"1": "enemy", "2": "friend", "3": "both"}
LABEL_DESC = {"enemy": "enemy only", "friend": "friend only", "both": "friend + enemy"}
QUIT_KEYS = ("q", "Q", "\x1b")  # q 또는 ESC

# 학습에 직접 쓰는 것은 filename, label, session(학습/검증을 sweep 단위로 나눌 때)뿐이다.
# 나머지는 불량 사진을 걸러내거나 조건별로 분석할 때 쓴다 (각도는 sweep 끝에서 표지판이 화면 밖으로
# 나간 사진을 찾을 때 쓴다).
CSV_COLUMNS = ["filename", "label", "session", "note", "yaw_cmd", "pitch_cmd", "timestamp",
               "clip_ratio", "dark_ratio", "mean_bgr", "config_locked"]

_stop_sweep = False  # 지금 sweep만 멈춤 (q)
_quit = False  # 프로그램 종료 (Ctrl+C)


def _on_sigint(signum, frame):
    global _stop_sweep, _quit
    _stop_sweep = True
    _quit = True
    print("\n[STOP REQUESTED] Finishing the photo at this position, then cleaning up...")


# ---------------------------------------------------------------- 키 입력


class RawKeyboard:
    """Enter 없이 키 하나를 바로 읽는다. 터미널이 아니면(파이프 등) 아무것도 읽지 않는다."""

    def __enter__(self):
        self.fd = None
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)  # Ctrl+C(SIGINT)는 그대로 동작한다
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def get(self, timeout: float = 0.0) -> str | None:
        if self.fd is None:
            return None
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None


# ---------------------------------------------------------------- 카메라 / 화면


def capture_bgr(cam: ColorCamera, rotate: int) -> np.ndarray:
    """실제 RGB로 읽은 뒤 저장용 BGR로 바꾸고 회전한다 (03/04와 같은 채널 처리)."""
    bgr = np.ascontiguousarray(cam.read_rgb()[:, :, ::-1])
    if rotate == 90:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif rotate == 270:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


class LiveView:
    """
    카메라 화면을 계속 보여 주는 창. 기다리는 동안에도 화면을 갱신하고,
    창과 터미널에서 들어온 키를 모아 둔다.
    """

    WIN = "05 Sweep Capture"

    def __init__(self, cam: ColorCamera, rotate: int, kb: RawKeyboard):
        self.cam = cam
        self.rotate = rotate
        self.kb = kb
        self.lines: list[str] = []  # 화면 위쪽에 표시할 상태 글자
        self.pending: list[str] = []  # 아직 처리하지 않은 키
        self.flash_until = 0.0
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        w, h = (480, 640) if rotate in (90, 270) else (640, 480)
        cv2.resizeWindow(self.WIN, w, h)

    def tick(self) -> None:
        """새 프레임 1장을 보여 주고, 들어온 키를 모은다 (약 1/30초)."""
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
        """time.sleep 대신 쓴다. 기다리는 동안에도 화면이 멈추지 않는다."""
        end = time.monotonic() + seconds
        while True:
            self.tick()
            if time.monotonic() >= end:
                break

    def take_key(self) -> str | None:
        return self.pending.pop(0) if self.pending else None

    def next_key(self) -> str | None:
        """키가 들어올 때까지 화면을 갱신하며 기다린다."""
        while not self.pending and not _quit:
            self.tick()
        return self.take_key()

    def flash(self) -> None:
        """사진을 저장했다는 표시 (테두리 빨간색)."""
        self.flash_until = time.monotonic() + 0.15

    def _draw(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        if self.lines:
            box_h = 12 + 22 * len(self.lines)
            roi = img[:box_h]
            roi[:] = (roi * 0.45).astype(np.uint8)  # 글자가 잘 보이게 위쪽을 어둡게
            for i, line in enumerate(self.lines):
                cv2.putText(img, line, (8, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if time.monotonic() < self.flash_until:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)

    def close(self) -> None:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------- servos


class Servos:
    def __init__(self, enabled: bool, settle: float, sleeper=time.sleep):
        self.enabled = enabled
        self.settle = settle
        self.sleep = sleeper  # 기다리는 방법 (LiveView.wait를 넣으면 화면이 멈추지 않는다)
        self.pos: dict[int, float] = {}
        self.pi = None
        if enabled:
            import pigpio  # 지연 import: --no-servo면 pigpio 없이도 동작한다

            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise SystemExit("Cannot connect to pigpiod. Run 'sudo systemctl enable --now pigpiod' first.")

    @staticmethod
    def angle_to_us(angle: float) -> float:
        angle = max(0.0, min(180.0, angle))
        return SERVO_MIN_US + angle * US_PER_DEG

    def write(self, pin: int, angle: float) -> float:
        us = round(self.angle_to_us(angle))
        if self.enabled:
            self.pi.set_servo_pulsewidth(pin, us)
        self.pos[pin] = angle
        return us

    def goto(self, pin: int, angle: float, settle: float | None = None) -> float:
        """이동한 뒤 흔들림이 멎을 때까지 한 번만 기다린다. 증가 방향으로만 사용한다."""
        if self.pos.get(pin) == angle:  # 이미 그 각도에서 기다렸으면 다시 기다리지 않는다
            return round(self.angle_to_us(angle))
        us = self.write(pin, angle)
        self.sleep(self.settle if settle is None else settle)
        return us

    def approach_from_below(self, pin: int, angle: float) -> float:
        """항상 아래쪽(더 작은 각도)에서 올라오며 목표에 도달한다. backlash를 없앤다."""
        cur = self.pos.get(pin)
        if cur is None or cur > angle - 0.1:
            under = max(0.0, angle - BACKLASH_MARGIN)
            self.write(pin, under)
            self.sleep(max(0.35, self.settle))
        return self.goto(pin, angle, settle=max(0.35, self.settle))

    def center(self, yaw: float, pitch: float, wait: float = 0.5) -> None:
        """
        중앙으로 보낸다. servo는 파이에 각도를 알려주지 못하지만, 신호를 받으면
        내부 회로가 스스로 그 각도로 간다. 두 servo를 동시에 돌리면 순간 전류가
        커지므로 pitch -> yaw 순서로 하나씩 움직인다.
        """
        if not self.enabled:
            return
        self.write(PITCH_PIN, pitch)
        self.sleep(wait)
        self.write(YAW_PIN, yaw)
        self.sleep(wait)

    def release(self):
        if self.enabled and self.pi is not None:
            for pin in (YAW_PIN, PITCH_PIN):
                self.pi.set_servo_pulsewidth(pin, 0)
            self.pi.stop()


# ---------------------------------------------------------------- 라벨 폴더


class LabelStore:
    """라벨 하나의 저장 폴더. 번호를 이어 붙이고, 끊겨도 안전하게 저장한다."""

    def __init__(self, root: Path, label: str):
        self.label = label
        self.dir = root / label
        self.img_dir = self.dir / "images"
        self.csv_path = self.dir / "labels.csv"
        self.sessions_path = self.dir / "sessions.jsonl"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self._pattern = re.compile(rf"^{re.escape(label)}_(\d+)\.(?:jpg|png)$")
        self._cleanup_partial()

    def _cleanup_partial(self) -> None:
        """이전 실행이 중간에 끊겼을 때 남은 흔적을 정리한다."""
        # 이름을 바꾸기 전에 끊긴 임시 사진
        for tmp in self.img_dir.glob("*.tmp"):
            tmp.unlink(missing_ok=True)
        # 줄 중간에 끊긴 CSV 마지막 줄은 잘라낸다
        if self.csv_path.exists():
            data = self.csv_path.read_bytes()
            if data and not data.endswith(b"\n"):
                cut = data.rfind(b"\n") + 1
                with self.csv_path.open("r+b") as f:
                    f.truncate(cut)
                print(f"[FIX] Removed a broken last line in {self.csv_path}")

    def next_index(self) -> int:
        """이미지 파일과 labels.csv에서 가장 큰 번호를 찾아 +1 한다."""
        biggest = 0
        for p in self.img_dir.iterdir():
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
            # 열 구성이 다른 옛 CSV에 이어 쓰면 열이 어긋나므로 멈춘다
            with self.csv_path.open(newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header != CSV_COLUMNS:
                raise SystemExit(f"[ERROR] {self.csv_path} has different columns: {header}\n"
                                 f"        Expected: {CSV_COLUMNS}\n"
                                 "        Move the old folder away (e.g. rename it) and run again.")
        f = self.csv_path.open("a", newline="", encoding="utf-8")
        w = csv.writer(f)
        if is_new:
            w.writerow(CSV_COLUMNS)
            _sync(f)
        return f, w

    def save_image(self, name: str, bgr: np.ndarray, jpeg_quality: int) -> None:
        """임시 파일에 쓴 뒤 이름을 바꾼다. 중간에 꺼져도 깨진 사진이 남지 않는다."""
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


def _sync(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def count_saved(root: Path) -> dict[str, int]:
    out = {}
    for label in LABEL_KEYS.values():
        d = root / label / "images"
        out[label] = len(list(d.glob(f"{label}_*.jpg"))) if d.exists() else 0
    return out


# ---------------------------------------------------------------- 품질


def quality(img: np.ndarray) -> tuple[float, float, float]:
    """clip(너무 밝음) 비율, 어두운 비율, 평균 밝기를 반환한다."""
    clip = float((img >= 250).any(axis=2).mean())
    dark = float((img <= 5).all(axis=2).mean())
    mean = float(img.mean())
    return clip, dark, mean


# ---------------------------------------------------------------- sweep


def parse_offsets(text: str) -> list[float]:
    return [float(t) for t in text.replace(" ", "").split(",") if t]


def run_sweep(label: str, args, cam: ColorCamera, servos: Servos, view: LiveView,
              yaw_offsets: list[float], pitch_offsets: list[float]) -> None:
    """라벨 하나로 sweep 한 번을 실행하고, 라벨 폴더에 이어서 저장한다."""
    global _stop_sweep
    _stop_sweep = False

    store = LabelStore(args.out, label)
    idx = store.next_index()
    first_idx = idx
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(pitch_offsets) * len(yaw_offsets) * args.shots
    locked = "yes" if cam.is_locked else "no"

    store.log_session({
        "session": session, "event": "start", "label": label, "note": args.note,
        "first_index": first_idx, "planned_photos": total,
        "center": {"yaw": args.center_yaw, "pitch": args.center_pitch},
        "yaw_offsets": yaw_offsets, "pitch_offsets": pitch_offsets,
        "shots_per_pose": args.shots, "settle_sec": args.settle, "rotate": args.rotate,
        "resolution": [args.width, args.height], "servo_enabled": not args.no_servo,
        "config": str(args.config) if cam.is_locked else None,
        "locked_settings": cam.locked.as_dict() if cam.is_locked else None,
        "note_on_angles": "All saved angles are command values. The SG90 cannot read its real angle.",
    })

    print(f"\n[START] Label '{label}' ({LABEL_DESC[label]}) -> {store.dir}")
    print(f"        {total} photos planned, file numbers start at {label}_{first_idx:06d}")
    print("        Press q to stop this sweep. Ctrl+C quits the program.")

    def status(extra: str) -> None:
        view.lines = [f"SWEEP {label.upper()}  {saved}/{total}  "
                      f"[{'LOCKED' if cam.is_locked else 'AUTO'}]",
                      extra, "q: stop this sweep"]

    csv_file, writer = store.open_csv()
    saved = warn_clip = warn_dark = 0
    t0 = time.time()
    try:
        for p_off in pitch_offsets:  # 낮은 쪽에서 높은 쪽으로
            if _stop_sweep:
                break
            pitch = args.center_pitch + p_off
            status(f"moving: pitch {p_off:+.1f}")
            servos.approach_from_below(PITCH_PIN, pitch)

            # 행마다 yaw는 왼쪽 끝으로 돌아가서 한 방향으로만 sweep한다
            servos.approach_from_below(YAW_PIN, args.center_yaw + yaw_offsets[0])

            for y_off in yaw_offsets:
                while (k := view.take_key()) is not None:  # 기다리는 동안 눌린 키 확인
                    if k in QUIT_KEYS:
                        _stop_sweep = True
                if _stop_sweep:
                    break
                yaw = args.center_yaw + y_off
                status(f"yaw {y_off:+.1f}  pitch {p_off:+.1f}")
                servos.goto(YAW_PIN, yaw)  # goto 안에서 한 번만 기다린다 (화면은 계속 갱신)

                for _ in range(args.shots):
                    img = capture_bgr(cam, args.rotate)
                    clip, dark, mean = quality(img)
                    name = f"{label}_{idx:06d}.jpg"
                    store.save_image(name, img, args.jpeg_quality)
                    writer.writerow([name, label, session, args.note,
                                     f"{yaw:.1f}", f"{pitch:.1f}",
                                     datetime.now().isoformat(timespec="seconds"),
                                     f"{clip:.4f}", f"{dark:.4f}", f"{mean:.1f}", locked])
                    _sync(csv_file)
                    idx += 1
                    saved += 1
                    warn_clip += clip > 0.01
                    warn_dark += dark > 0.05
                    view.flash()

                done = saved / total if total else 1.0
                elapsed = time.time() - t0
                eta = elapsed / done - elapsed if done > 0 else 0
                print(f"\r  {saved}/{total}  yaw {y_off:+5.1f}  pitch {p_off:+5.1f}  "
                      f"time left ~{eta:4.0f} sec", end="", flush=True)
    finally:
        print()
        csv_file.close()
        store.log_session({
            "session": session, "event": "end", "label": label,
            "first_index": first_idx, "last_index": idx - 1 if saved else None,
            "saved_photos": saved, "stopped_early": saved < total,
            "seconds": round(time.time() - t0, 1),
        })

    status_word = "STOPPED" if saved < total else "DONE"
    last = f"{label}_{idx - 1:06d}" if saved else "-"
    print(f"[{status_word}] Saved {saved}/{total} photos in {time.time() - t0:.0f} sec  "
          f"({label}_{first_idx:06d} ~ {last})")
    if warn_clip:
        print(f"[WARN] {warn_clip} photos have more than 1% clipped (too bright) pixels. "
              "It is better to lower the exposure and make the config again.")
    if warn_dark:
        print(f"[WARN] {warn_dark} photos have more than 5% dark pixels. Think about adding more light.")


# ---------------------------------------------------------------- 메인


def show_menu(view: LiveView, root: Path, locked: bool) -> None:
    counts = count_saved(root)
    # 터미널
    print("\n" + "=" * 60)
    for key, label in LABEL_KEYS.items():
        print(f"  {key} : {label:<7} ({LABEL_DESC[label]:<15})  saved so far: {counts[label]}")
    print("  q : quit")
    print("=" * 60)
    print("Place the signs, then press a key (in the camera window or here).", flush=True)
    # 카메라 창
    view.lines = [f"READY  [{'LOCKED' if locked else 'AUTO'}]",
                  "1: enemy   2: friend   3: both   q: quit",
                  f"saved  enemy {counts['enemy']}  friend {counts['friend']}  both {counts['both']}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="", help="Free note saved in labels.csv (e.g. left_1.8m_front)")
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                    help="Camera config saved by 03_preview.py (press f there)")
    ap.add_argument("--allow-auto", action="store_true",
                    help="Take photos in auto mode without a camera config file (not recommended)")

    ap.add_argument("--yaw-range", type=float, default=YAW_RANGE, help="Left/right range from the center (deg)")
    ap.add_argument("--yaw-step", type=float, default=YAW_STEP)
    ap.add_argument("--pitch-offsets", default=PITCH_OFFSETS, help="List of pitch offsets from the center (deg)")
    ap.add_argument("--center-yaw", type=float, default=CENTER_YAW)
    ap.add_argument("--center-pitch", type=float, default=CENTER_PITCH)

    ap.add_argument("--shots", type=int, default=1, help="Number of photos to save at each position")
    ap.add_argument("--settle", type=float, default=0.4, help="Time to wait after a move for the shaking to stop (sec)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--rotate", type=int, default=90, choices=[0, 90, 180, 270],
                    help="Rotation before saving. Default is 90, the same as the old collection code")
    ap.add_argument("--no-swap", action="store_true", help="Do not swap the R/B channels")
    ap.add_argument("--no-servo", action="store_true", help="Test the camera only, do not move the servos")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    if not YAW_RANGE_CONFIRMED:
        print(f"[TODO] Yaw range +/-{YAW_RANGE:g} deg is not checked yet. "
              "See the TODO block at the top of 05_sweep_capture.py.")

    # --- 스텝 확인 (SG90 dead band) ------------------------------------
    step_us = args.yaw_step * US_PER_DEG
    if args.yaw_step > 0 and step_us < MIN_STEP_US:
        print(f"[ERROR] Yaw step {args.yaw_step} deg = {step_us:.1f} us, which is smaller than the dead band ({MIN_STEP_US} us).")
        print(f"       Set it to {MIN_STEP_US / US_PER_DEG:.2f} deg or more.")
        return 1

    pitch_offsets = sorted(parse_offsets(args.pitch_offsets))
    if len(pitch_offsets) > 1:
        gaps = [b - a for a, b in zip(pitch_offsets, pitch_offsets[1:])]
        if min(gaps) * US_PER_DEG < MIN_STEP_US:
            print(f"[ERROR] The pitch spacing is smaller than the dead band. Make it {MIN_STEP_US / US_PER_DEG:.2f} deg or more.")
            return 1

    if args.yaw_step <= 0:
        yaw_offsets = [0.0]
    else:
        n = int(round(2 * args.yaw_range / args.yaw_step))
        yaw_offsets = [-args.yaw_range + i * args.yaw_step for i in range(n + 1)]

    if not args.config.exists() and not args.allow_auto:
        print(f"[ERROR] Camera config file not found: {args.config}")
        print("       Press f in 03_preview.py to save fixed values, or add --allow-auto.")
        return 1

    signal.signal(signal.SIGINT, _on_sigint)

    # --- 카메라: 03/04와 같은 방식으로 config를 불러와 고정 ---------------
    print("[START] Opening the camera...")
    cam = ColorCamera(main_size=(args.width, args.height), channel_swap=not args.no_swap)
    cam.start()
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
            if cam.load_warnings:
                print("       The camera setup is different from when the config was saved. "
                      "Run with the same --width/--height, or lock again in 03_preview.py.")
        else:
            print("[WARN] No config file, so photos are taken in auto mode. Colors may not stay the same.")

        n_photos = len(pitch_offsets) * len(yaw_offsets) * args.shots
        print(f"[PLAN] Each sweep: {len(pitch_offsets)} pitch steps x {len(yaw_offsets)} yaw positions "
              f"x {args.shots} photos = {n_photos} photos")

        with RawKeyboard() as kb:
            view = LiveView(cam, args.rotate, kb)
            servos = Servos(enabled=not args.no_servo, settle=args.settle, sleeper=view.wait)

            # 시작: 어디에 있든 항상 중앙으로 정렬한 뒤 메뉴를 띄운다
            if servos.enabled:
                print(f"[CENTER] Moving to the center (yaw {args.center_yaw}, pitch {args.center_pitch})...")
                view.lines = ["CENTERING...", "servos move to the center"]
                servos.center(args.center_yaw, args.center_pitch)

            while not _quit:
                show_menu(view, args.out, cam.is_locked)
                key = view.next_key()
                if _quit or key is None or key in QUIT_KEYS:
                    break
                if key not in LABEL_KEYS:
                    print(f"[INFO] Key '{key}' is not used. Press 1, 2, 3 or q.")
                    continue
                run_sweep(LABEL_KEYS[key], args, cam, servos, view, yaw_offsets, pitch_offsets)
    finally:
        if servos is not None:
            # 끝: 중앙에 돌려놓은 뒤 힘을 푼다. 다음 실행 때 시작 정렬이 거의 움직이지 않는다
            if servos.enabled:
                try:
                    print("[CENTER] Returning to the center before releasing the servos...")
                    if view is not None:
                        view.lines = ["PARKING...", "servos return to the center"]
                    servos.center(args.center_yaw, args.center_pitch)
                except Exception as e:  # 화면/카메라 오류가 나도 힘은 반드시 푼다
                    print(f"[WARN] Could not return to the center: {e}")
            servos.release()
        if view is not None:
            view.close()
        cam.close()

    print("[EXIT] Camera closed, servos parked at the center and released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
