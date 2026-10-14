#!/usr/bin/env python3
"""
98_servo_center.py - servo의 실제 "정면" 각도를 찾아 03_config/servo_center.json에 저장한다.

05_sweep_capture.py와 05_manual_capture.py는 모두 이 파일에서 중앙값을 읽는다
(colorlab/servo.py, 파일이 없으면 yaw 95, pitch 90).

SG90은 개체와 조립 상태마다 "몇 도 = 정면"이 다르다. 저장된 중앙으로 보냈는데
총이 정면을 보지 않으면, 이 도구로 키보드를 눌러 가며 정면이 되는 각도를 찾는다.
카메라는 쓰지 않는다 (총을 직접 보면서 맞춘다).

키 (터미널에서 바로 입력, Enter 불필요):
    a / d   yaw  -1 / +1 도        A / D   yaw  -5 / +5 도
    s / w   pitch -1 / +1 도       S / W   pitch -5 / +5 도
    r       저장된 원래 중앙 값으로 되돌리기
    q       끝내기 -> 찾은 값을 03_config/servo_center.json에 저장할지 묻는다 (y/n)

backlash 대응: 05와 같이 항상 작은 각도 쪽(아래)에서 올라오며 목표에 도달한다.
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from colorlab.servo import SERVO_CENTER_PATH, load_center, save_center  # noqa: E402

YAW_PIN = 18  # 물리 12번 핀
PITCH_PIN = 13  # 물리 33번 핀
BACKLASH_MARGIN = 3.0  # 되돌아갈 때 목표보다 이만큼 아래로 내려갔다가 올라온다 (deg)
SETTLE = 0.3


def angle_to_us(angle: float) -> int:
    angle = max(0.0, min(180.0, angle))
    return round(500 + angle * 2000 / 180)


class Servo:
    def __init__(self, pi, pin: int, angle: float):
        self.pi, self.pin, self.angle = pi, pin, None
        self.move(angle)

    def move(self, target: float) -> None:
        target = max(0.0, min(180.0, target))
        # 내려가는 방향이면 목표보다 아래로 갔다가 올라온다 (backlash 제거)
        if self.angle is None or target < self.angle:
            self.pi.set_servo_pulsewidth(self.pin, angle_to_us(target - BACKLASH_MARGIN))
            time.sleep(SETTLE)
        self.pi.set_servo_pulsewidth(self.pin, angle_to_us(target))
        time.sleep(SETTLE)
        self.angle = target


def get_key(timeout: float = 0.2) -> str | None:
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if r else None


def main() -> int:
    if not sys.stdin.isatty():
        print("[ERROR] Run this in a terminal.")
        return 1
    import pigpio

    pi = pigpio.pi()
    if not pi.connected:
        print("[ERROR] Cannot connect to pigpiod. Run 'sudo systemctl start pigpiod'.")
        return 1

    orig_yaw, orig_pitch = load_center()
    src = SERVO_CENTER_PATH.name if SERVO_CENTER_PATH.exists() else "default (no file yet)"
    print(f"[START] Current center from {src}: yaw {orig_yaw:.1f}, pitch {orig_pitch:.1f}")
    print("  a/d: yaw -1/+1   A/D: yaw -5/+5   s/w: pitch -1/+1   S/W: pitch -5/+5")
    print("  r: back to the saved center   q: finish")
    print("Move until the gun points straight ahead.\n")

    pitch = Servo(pi, PITCH_PIN, orig_pitch)  # 하나씩 움직여 순간 전류를 줄인다
    yaw = Servo(pi, YAW_PIN, orig_yaw)
    steps = {"a": ("yaw", -1), "d": ("yaw", 1), "A": ("yaw", -5), "D": ("yaw", 5),
             "s": ("pitch", -1), "w": ("pitch", 1), "S": ("pitch", -5), "W": ("pitch", 5)}

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            print(f"\r  yaw {yaw.angle:5.1f} deg ({angle_to_us(yaw.angle)} us)   "
                  f"pitch {pitch.angle:5.1f} deg ({angle_to_us(pitch.angle)} us)   ", end="", flush=True)
            k = get_key()
            if k is None:
                continue
            if k in steps:
                which, d = steps[k]
                s = yaw if which == "yaw" else pitch
                s.move(s.angle + d)
            elif k == "r":
                pitch.move(orig_pitch)
                yaw.move(orig_yaw)
            elif k in ("q", "Q", "\x1b"):
                break

        print(f"\n\n[FOUND] yaw {yaw.angle:.1f}, pitch {pitch.angle:.1f}"
              f"   (was yaw {orig_yaw:.1f}, pitch {orig_pitch:.1f})")
        if (yaw.angle, pitch.angle) == (orig_yaw, orig_pitch):
            print("[INFO] Same as the saved values. Nothing to change.")
        else:
            print(f"Save to {SERVO_CENTER_PATH}? (y/n) ", end="", flush=True)
            while (k := get_key(1.0)) is None:
                pass
            print(k)
            if k in ("y", "Y"):
                p = save_center(yaw.angle, pitch.angle, note="98_servo_center.py")
                print(f"[SAVED] {p}: yaw {yaw.angle:.1f}, pitch {pitch.angle:.1f}")
            else:
                print("[SKIP] Not saved.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        pi.set_servo_pulsewidth(YAW_PIN, 0)
        pi.set_servo_pulsewidth(PITCH_PIN, 0)
        pi.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
