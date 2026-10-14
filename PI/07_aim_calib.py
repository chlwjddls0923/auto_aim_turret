#!/usr/bin/env python
"""
07_aim_calib.py - 거리별 조준점 보정 + 적군 자동 사격 도구.
  YOLO로 타겟을 찾고, 08_detect_check.py와 같은 판정(target_pipeline.Pipeline: 색으로 아군/적군/unknown)을 붙여
  p 키로 "적군만" 자동 조준해서 쏜다. 수동 조준(a/d/w/s, f)과 실사격 기록(e)으로 "타겟이 화면 어디에 있을 때
  맞는가"(조준점)를 찾는 기존 기능도 그대로다.

왜 필요한가:
  카메라는 총에 손으로 붙어 있어 총구와 어긋나 있다. 검출 박스 중심을 화면 중앙에 맞춰도 맞지 않는다.
  그래서 거리마다 서보를 손으로 움직여 실제로 맞는 자세를 찾고, 그때 카메라 화면에서 타겟이 보이는 위치
  (cx, cy)와 박스 높이 h를 기록한다. 이것이 그 거리(= 그 h)의 "조준점"이다.
  실전 aim()(pipeline/tools/target_pipeline.py의 Aimer)은 calib/aim_table.json을 읽어
  "h가 이 값이면 타겟을 (cx, cy)에 맞춰라"로 쓴다.

실행 (YOLO 검출기가 필요하므로 pipeline venv로):
    ~/src/pipeline/venv/bin/python 07_aim_calib.py
    ~/src/pipeline/venv/bin/python 07_aim_calib.py --distance 1.8 --step 1.0
    ~/src/pipeline/venv/bin/python 07_aim_calib.py --live-detect      # 박스를 실시간으로도 보고 싶을 때 (아래 '화면')
    ~/src/pipeline/venv/bin/python 07_aim_calib.py --fake pipeline/samples/friend_hallB_1.5m.jpg --no-servo --calib-dir /tmp/aim_test
        (카메라·서보 없이 격자·검출·클릭 기록을 확인한다. --calib-dir로 보내 실제 표와 섞이지 않게 한다)

키 (카메라 창이나 터미널에서, Enter 없이):
    a / d     yaw 왼쪽 / 오른쪽 한 칸 (05_manual_capture.py와 같은 step, 기본 2도)
    w / s     pitch 위 / 아래 한 칸
    [ / ]     step 줄이기 / 늘리기 (최소 0.9도 = SG90 dead band)
    r         중앙값 복귀 (03_config/servo_center.json)
    f         발사: 발사 서보(GPIO12)를 FIRE_READY -> FIRE_SHOOT로 0.5도씩 나눠 천천히 보내고(--fire-duration, 기본 1.0초),
              FIRE_HOLD_SEC 머문 뒤 같은 속도로 FIRE_READY로 복귀한다. 급가속으로 총 본체가 흔들리는 것을 줄인다.
              발사 중(접근·유지·복귀)에는 yaw/pitch 명령을 보내지 않고, 끝난 뒤 FIRE_COOLDOWN_SEC(0.3초)까지
              키 입력을 버린다(q / ESC / Ctrl+C만 받는다). 화면에 FIRE 0.5초 표시.
              안전장치: 시작 직후 5초(FIRE_ARM_SEC)는 f가 동작하지 않는다 (서보 초기화 중 오발 방지).
              시작 시와 종료 시에는 반드시 FIRE_READY 위치로 둔다. 각도 두 값은 위쪽 상수 FIRE_READY / FIRE_SHOOT
              (실제 걸쇠에 맞춰 조정).
    p         적군 자동 조준 + 발사 (한 번):
                1) 전체 프레임 YOLO 검출 + 아군/적군 판정 (08과 같은 코드: pipeline_config.json의 color 설정,
                   color_refs.json). 07의 규칙 = "아군이 아니면 쏜다" (타겟은 적군만 / 적군+아군 두 경우뿐이므로):
                     아군(friend)으로 판정된 박스만 빼고, 나머지(enemy 또는 unknown) 중 h가 가장 큰(가까운) 것.
                     아군만 있거나 박스가 없으면 안 쏜다.
                   (실전 공용 규칙 target_pipeline.decide는 unknown이면 보류한다 - run_match.py는 그대로다)
                   고른 적군 박스의 높이 h로 거리 추정 d = h_coef / h
                   (h_coef = --h-coef로 주면 그 값, 아니면 조준 기록 중 h가 있는 것의 h x 거리 중앙값, 없으면 59.1).
                   기본은 거리를 확인하지 않는다: 박스 크기와 상관없이 적군이면 쏜다 (추정 거리는 출력·로그에만 남긴다).
                   --check-distance를 주면 거리를 확인한다: 기록 거리를 벗어나면 쏘지 않고, 타겟이 실제로 선택 거리에
                   있다고 보고 맞는 --h-coef 값을 알려 준다.
                2) 조준점: 기본은 선택 거리(-/+, 기본 1.5 m)에 가장 가까운 기록 거리의 조준점.
                   --check-distance면 추정 거리에 가장 가까운 기록 거리(+/-AUTO_DIST_TOL m)의 조준점 = 그 거리 기록의 cx, cy 중앙값
                   (잘못 찍은 기록 몇 개에 흔들리지 않게 평균 대신 중앙값). 기록이 없으면 쏘지 않는다.
                3) 타겟이 조준점 +/-6 px 안에 올 때까지 yaw/pitch를 계속 나눠 움직인다 (횟수 제한 없음, q/ESC로 중단).
                   목표를 넘어설 때마다(오차 부호가 바뀌면) 그 축의 이동량 배율과 최소 이동 폭(0.9도 -> 0.1도까지)을 절반씩 줄여
                   좌우로 오가며 좁혀 들어간다.
                   한 번 움직일 때마다 화면 전체가 실제로 몇 px 밀렸는지(위상 상관) 재서 타겟 위치를 따라가고,
                   그 값으로 "px/도"를 다시 배운다 (렌즈 보정값이 없어서 처음엔 --px-per-deg로 가정).
                   이동 후 예측 위치 주변을 ROI로 다시 검출 + 판정해서 위치를 고친다. 그 박스가 아군으로
                   판정되면 그 자리에서 멈추고 쏘지 않는다.
                4) 이동 뒤 박스를 놓쳤으면 새 프레임으로 최대 3번 다시 찾는다(ROI -> 전체 프레임). 다시 찾은 위치로
                   오차를 다시 재고, 확인된 박스가 6 px 안이고 아군이 아닐 때만 f와 같은 발사.
                   같은 방향으로 계속 가는데 아직 멀면(12 px 넘게) 줄였던 이동 폭을 1.5배씩 다시 키운다.
                   발사가 시작되면 평소 07 동작으로 돌아온다.
                5) 맞았으면 e -> 발사 순간의 값(박스 중앙 cx, cy, h, w, conf, yaw, pitch, 그때의 원본 프레임 PNG)을
                   source=auto로 저장한다. 안 맞았으면 아무것도 안 누르면 된다 (저장 안 됨).
                   (맞으면 머리가 떨어져 e 시점엔 박스를 못 찾을 수 있으므로 e를 누른 순간이 아니라 발사 순간 값을 쓴다)
                   이 'e 대기'는 서보를 움직이거나(a/d/w/s/r) p를 다시 누르면 사라진다.
              자율 동작 중 q / ESC를 누르면 그 자리에서 중단한다 (프로그램은 끝나지 않는다).
              시작 직후 5초(발사 잠금) 동안은 동작하지 않는다. 결과는 calib/auto_shots.csv에 한 줄씩 남는다.
              * 클릭 표시(o)로는 쏘지 않는다. 판정 설정이 없으면 모두 unknown(아군 아님) -> 아군 구분 없이 쏜다.
    - / +     선택 거리 0.1 m 감소 / 증가 (1.5 ~ 2.0 m, '='도 '+'로 쓴다)
    클릭      화면을 클릭하면 그 자리에 o1, o2, o3 ... 순서대로 표시가 남는다 (e로 저장할 때까지 지워지지 않는다)
              명중 기록용 클릭은 타겟 '머리 자리'를 찍는다 (p는 이것을 박스 중앙 기준으로 바꿔 쓴다: + 0.35 x h 아래)
    c         클릭 표시 모두 지우기
    e         명중 기록: 지금 프레임을 전체 해상도로 새로 검출 -> 원본 PNG 저장 + CSV 한 줄
                p로 쏜 직후(서보를 움직이기 전)이고 클릭 표시가 없으면 -> p 발사 순간의 값을 기록 (source=auto, 위 p의 5)
                클릭 표시가 있으면 -> **마지막 표시 하나**의 좌표를 기록 (source=click). 그 점을 감싸는 검출 박스가
                                     있으면 박스의 h·w·conf를 함께 적는다(없으면 비움). 저장 후 표시는 모두 지운다
                (클릭 표시가 없을 때)
                박스 1개       -> 그 중심 (cx, cy)와 h를 기록 (source=detect)
                박스 2개 이상  -> 아군 박스는 빼고, 선택 거리와 h가 맞는 것(h ~ 59.1/d)을 고른다. 애매하면 클릭해 고른다
                아군 박스만    -> 화면에서 타겟 위치를 클릭하면 그 좌표를 기록
                박스 없음      -> 화면에서 타겟 위치를 클릭하면 그 좌표를 기록 (source=click, h는 비움)
              클릭을 기다리는 중에 e나 ESC를 다시 누르면 취소
    q         종료 -> 거리별 요약 출력, calib/aim_table.json 저장
    Ctrl+C    종료 (기록은 그대로 남는다)
  * 한글 입력 상태에서는 키가 전달되지 않는다. 키보드를 영문 입력 상태로 둔다.

화면:
    05_manual_capture.py처럼 카메라 영상 + 50 px 격자와 좌표 숫자, 상단에 거리·yaw·pitch·이 거리의 기록 수.
    기록된 조준점은 거리별 색 점으로, 거리별 평균은 큰 원으로 계속 표시한다.
    검출은 기본적으로 e를 누를 때 한 번만 돈다(전체 프레임 1280x960, 약 1.2 s). 그 결과 박스를 2초간 보여 주고 기록한다.
    --live-detect를 주면 뒤에서 박스를 실시간으로도 그린다: 마지막 타겟 주변 ROI 384 추적(약 0.15 s)만 계속 돌고,
    전체 프레임 검출은 직전 전체 검출이 끝난 뒤 LIVE_FULL_INTERVAL_SEC(2초)마다 한 번으로 제한한다. 검출이 없으면 NO TARGET.
    어느 모드든 e의 기록값은 그 순간 프레임의 전체 프레임 검출이다 (라이브 박스의 지연과 무관).

산출물 (기본 ~/src/pipeline/calib/):
    aim_points.csv            한 줄 = 명중 1회:
                              timestamp, distance_m, cx, cy, h_px, w_px, conf, yaw_cmd, pitch_cmd, source, png
    shots/<거리>m_<번호>.png  그때의 원본 프레임 (오버레이 없음)
    aim_table.json            종료 시 CSV 전체(이전 세션 포함)로 다시 만든다: h_px(오름차순) -> cx, cy. Aimer가 읽는다

주의:
    * 카메라 위치가 바뀌면 표 전체가 무효다. 보정 전에 카메라가 총에 단단히 고정됐는지 확인한다.
    * 받침·막대를 색칠하면 박스 높이 h가 달라진다 -> 다시 측정한다.
    * e는 **실제로 맞았을 때만** 누른다. 빗나간 자세를 기록하면 표가 틀어진다. (f로 쏘고 -> 맞았으면 e)
    * 발사 서보는 종료 후에도 READY 각도에서 펄스를 유지한다(걸쇠를 붙잡고 있도록, 06_data_collect.py와 같다).
    * 같은 거리에서 cx·cy 표준편차가 15 px를 넘으면 서보 재현성이나 카메라 흔들림을 의심한다 (종료 시 경고).
    * 05_manual_capture.py는 건드리지 않는다. 서보·중앙값·키 입력 코드는 그 파일에서 그대로 가져왔다.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import re
import select
import signal
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
# venv의 cv2(Qt)는 글꼴 폴더를 못 찾아 경고를 반복 출력한다 -> 시스템 DejaVu를 알려 준다
if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # colorlab
sys.path.insert(0, str(HERE / "pipeline" / "tools"))   # target_pipeline (검출기). ultralytics는 모델을 만들 때 읽는다

from colorlab.camera import ColorCamera, DEFAULT_CONFIG_PATH, DEFAULT_MAIN_SIZE  # noqa: E402
from colorlab.servo import SERVO_CENTER_PATH, load_center  # noqa: E402

# ---------------------------------------------------------------- 상수 (서보 부분은 05_manual_capture.py와 같다)

YAW_PIN = 18
PITCH_PIN = 13
FIRE_PIN = 12            # 발사 서보 (Lecture/06_data_collect.py와 같은 핀)
CENTER_YAW, CENTER_PITCH = load_center()

# 발사 각도 (deg). 실제 걸쇠에 맞춰 여기서 조정한다 (06_data_collect.py의 값에서 시작)
FIRE_READY = 90.0        # 대기(걸쇠 잠김) 위치. 시작·종료 시 반드시 이 위치
FIRE_SHOOT = 160.0       # 발사 위치
FIRE_HOLD_SEC = 0.5      # SHOOT 도달 후 유지하는 시간. 지나면 천천히 READY로 복귀
FIRE_STEP_DEG = 0.5      # 발사 서보를 나눠 움직이는 한 칸 (약 6 us)
FIRE_DURATION_SEC = 0.5  # READY -> SHOOT 접근 시간 (복귀도 같은 속도). --fire-duration으로 바꾼다
FIRE_COOLDOWN_SEC = 0.3  # 발사가 끝난 뒤 이 시간까지 키 입력을 받지 않는다
FIRE_ARM_SEC = 5.0       # 시작 후 이 시간 동안은 f가 동작하지 않는다 (서보 초기화 중 오발 방지)
FIRE_FLASH_SEC = 0.5     # 화면 FIRE 표시 시간

SERVO_MIN_US = 500.0
SERVO_MAX_US = 2500.0
US_PER_DEG = (SERVO_MAX_US - SERVO_MIN_US) / 180.0   # 약 11.1 us/도
MIN_STEP_US = 10.0                                   # SG90의 dead band

DIST_MIN, DIST_MAX, DIST_STEP = 1.5, 2.0, 0.1
H_COEF = 59.1          # h_px ~ H_COEF / d(m). RESUME.md §4의 b2 실측값. 박스가 여러 개일 때 고르는 데만 쓴다 (색칠 후 바뀔 수 있다)
GRID_PX = 50
LIVE_FULL_INTERVAL_SEC = 2.0   # --live-detect: 전체 프레임 검출은 이 간격으로 1회, 나머지는 ROI 추적만
MARK_BOX_PAD = 10.0            # 클릭 표시를 저장할 때, 이 여유(px) 안에서 점을 감싸는 검출 박스의 h를 함께 적는다
E_SHOW_SEC = 2.0               # e로 검출한 박스를 화면에 보여 주는 시간
STD_WARN_PX = 15.0     # 같은 거리에서 cx·cy 표준편차가 이보다 크면 경고
FLASH_SEC = 0.8

# 자율 조준 (p 키)
AUTO_DIST_TOL = 0.2        # 추정 거리와 가장 가까운 기록 거리가 이만큼(m) 넘게 다르면 조준점이 없다고 보고 쏘지 않는다
AUTO_TOL_PX = 6.0          # 타겟이 조준점에서 이 px 안이면 발사. 10-06: 12 px에서는 9~10 px 오차로도 쏴서 빗나감
                           # (서보 최소 0.9도 = 약 11 px라 6 px 안팎이 한계. 6~11 px 오차는 0.9도 한 번으로 넘어온다)
HEAD_OFFSET_FRAC = 0.35    # 타겟 머리 중심이 YOLO 박스 중앙보다 h(박스 높이)의 이만큼 위 (10-04 사진 12개 0.34~0.37, 10-06 021: 8/37)
AUTO_MAX_ITER = 0          # 서보를 나눠 움직이는 최대 횟수. 0 = 제한 없음: 오차가 AUTO_TOL_PX 안에 들 때까지 계속 (q/ESC로 중단)
AUTO_STEP_FLOOR_DEG = 0.1  # 목표를 넘어설 때마다(오차 부호가 바뀌면) 그 축의 최소 이동 폭을 절반으로 줄인다. 이 값까지.
                           # 0.9도(11~15 px)씩만 움직이면 +/-7 px 사이를 오가며 6 px 안에 못 든다 (10-06 4번).
                           # 명령은 절대 각도라 작은 이동도 쌓이면 실제로 움직인다
AUTO_FIND_TRIES = 3        # 박스를 놓쳤을 때 다시 찾는 횟수 (매번 새 프레임: ROI 검출 -> 안 되면 전체 프레임 검출)
AUTO_LEARN_MIN_DEG = 1.8   # px/도는 이만큼 이상 움직였을 때만 다시 잰다 (0.9도 이동은 측정 잡음이 커서 15~19로 튀었다)
AUTO_MAX_STEP_DEG = 8.0    # 한 번에 움직이는 최대 각도
AUTO_SETTLE_SEC = 1.0      # 서보 이동 뒤 흔들림이 멎을 때까지 기다리는 시간
AUTO_PX_PER_DEG = 12.0     # 처음 가정하는 px/도. 10-06 조준 기록 사진 쌍의 화면 이동량에서 잰 값(yaw 11.8~12.7, pitch 10.9~11.3).
                           # p를 누를 때마다 실제 이동량으로 다시 고친다. 크게 잡을수록 덜 움직인다(넘치지 않는 쪽)
AUTO_PITCH_LIMIT = 30.0    # 자율 조준에서 pitch는 중앙 +/- 이 각도 안에서만
AUTO_SHIFT_MIN_RESP = 0.05 # 위상 상관 응답이 이보다 작으면 화면 이동량 측정을 믿지 않고 예측값을 쓴다
AUTO_TRACK_PX = 40.0       # 이동 뒤 예측 위치에서 이 거리 안의 박스만 같은 타겟으로 본다
AUTO_SHOW_SEC = 4.0        # 자율 조준 결과(조준점·타겟·오차)를 화면에 보여 주는 시간
SIDE_COLORS = {"enemy": (0, 0, 255), "friend": (0, 200, 0), "unknown": (0, 255, 255)}   # 08과 같은 색 (BGR)
MIN_STEP_DEG = MIN_STEP_US / US_PER_DEG              # 약 0.9도. 이보다 작은 명령은 SG90이 따라오지 않는다
AUTO_COLUMNS = ["timestamp", "result", "judge", "source", "h_px", "dist_est", "h_coef", "aim_dist", "aim_cx", "aim_cy",
                "t0_cx", "t0_cy", "t_cx", "t_cy", "err_x", "err_y", "iters", "yaw_cmd", "pitch_cmd",
                "px_per_deg_yaw", "px_per_deg_pitch", "note"]

CALIB_DIR = HERE / "pipeline" / "calib"
PIPELINE_CONFIG = HERE / "pipeline" / "tools" / "pipeline_config.json"
CSV_COLUMNS = ["timestamp", "distance_m", "cx", "cy", "h_px", "w_px", "conf", "yaw_cmd", "pitch_cmd", "source", "png"]
PNG_RE = re.compile(r"^(\d+\.\d)m_(\d+)\.png$")
QUIT_KEYS = ("q", "\x1b")
WIN = "07 Aim Calib"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 거리별 표시 색 (BGR)
DIST_COLORS = {
    1.5: (80, 80, 255),      # 빨강
    1.6: (0, 165, 255),      # 주황
    1.7: (0, 230, 230),      # 노랑
    1.8: (80, 220, 80),      # 녹색
    1.9: (255, 200, 60),     # 하늘
    2.0: (255, 100, 255),    # 자홍
}

_quit = False


def _on_sigint(signum, frame):
    global _quit
    _quit = True
    print("\n[INT] Stopping. Records saved so far are kept.", flush=True)


def dist_key(d: float) -> float:
    return round(float(d), 1)


def dist_color(d: float) -> tuple[int, int, int]:
    return DIST_COLORS.get(dist_key(d), (200, 200, 200))


def expected_h(distance: float) -> float:
    """선택 거리에서 기대되는 박스 높이(px)."""
    return H_COEF / float(distance)


# ---------------------------------------------------------------- 키보드 (05_manual_capture.py와 같다)


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


# ---------------------------------------------------------------- 서보 (05_manual_capture.py와 같다)


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

    # 발사 서보 (06_data_collect.py의 set_servo(FIRE_PIN, ...)와 같은 펄스 계산)
    def fire_ready(self) -> None:
        self.write(FIRE_PIN, FIRE_READY)

    def fire_shoot(self) -> None:
        self.write(FIRE_PIN, FIRE_SHOOT)

    def fire_ramp(self, target: float, duration: float, stop: threading.Event | None = None) -> bool:
        """발사 서보를 지금 각도에서 target까지 FIRE_STEP_DEG씩 나눠 움직인다. 전체 각도를 duration에 맞춘 속도.
        (FIRE_READY~FIRE_SHOOT 전체 폭을 duration에 가도록 속도를 정하므로, 중간에서 출발해도 속도는 같다)
        stop이 켜지면 그 자리에서 멈추고 False. 별도 스레드에서 부르므로 GUI용 self.sleep이 아니라 시간으로 기다린다."""
        cur = self.get(FIRE_PIN, FIRE_READY)
        n = max(1, math.ceil(abs(target - cur) / FIRE_STEP_DEG - 1e-9))
        full = max(1, math.ceil(abs(FIRE_SHOOT - FIRE_READY) / FIRE_STEP_DEG - 1e-9))
        dt = max(0.0, duration) / full
        t0 = time.monotonic()
        for i in range(1, n + 1):
            self.write(FIRE_PIN, cur + (target - cur) * i / n)
            wait = t0 + dt * i - time.monotonic()          # 누적 오차 없이 일정 간격
            if wait > 0 and stop is not None:
                if stop.wait(wait):
                    return False
            elif wait > 0:
                time.sleep(wait)
            elif stop is not None and stop.is_set():
                return False
        return True

    def release(self) -> None:
        """yaw/pitch 펄스를 끊는다. 발사 서보는 끊지 않는다: READY 각도에서 펄스를 유지해 걸쇠를 붙잡고 있게 둔다
        (06_data_collect.py도 pi.stop()만 하고 펄스를 끊지 않는다). 호출 전에 fire_ready()가 되어 있어야 한다."""
        if self.enabled and self.pi is not None:
            for pin in (YAW_PIN, PITCH_PIN):
                self.pi.set_servo_pulsewidth(pin, 0)
            self.pi.stop()


# ---------------------------------------------------------------- 카메라


class RealCamera:
    """colorlab ColorCamera를 감싼다. 05_manual_capture.py처럼 config를 불러와 LOCKED로 시작한다. frame()은 BGR."""

    def __init__(self, size: tuple[int, int], config: Path | None):
        self.cam = ColorCamera(main_size=size)
        self.cam.start()
        self.label = "AUTO"
        if config is not None:
            s = self.cam.load_config(config)
            print(f"[LOCKED] {config}")
            for line in s.summary_lines():
                print("    " + line)
            for w in self.cam.load_warnings:
                print(f"[WARN] {w}")
            hint = self.cam.resolution_hint()
            if hint:
                print(f"[WARN] {hint}")
            self.label = "LOCKED"
        else:
            print("[WARN] No camera config, so the camera runs in auto mode (detection still works).")

    def frame(self) -> np.ndarray:
        return np.ascontiguousarray(self.cam.read_rgb()[:, :, ::-1])

    def close(self) -> None:
        self.cam.close()


class FakeCamera:
    """카메라 없이 시험할 때: 사진 한 장을 계속 돌려준다."""

    def __init__(self, path: Path, size: tuple[int, int]):
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"[ERROR] Cannot read the fake image: {path}")
        if (img.shape[1], img.shape[0]) != tuple(size):
            print(f"[WARN] Fake image is {img.shape[1]}x{img.shape[0]}; resizing to {size[0]}x{size[1]}.")
            img = cv2.resize(img, tuple(size))
        self.img = img
        self.label = "FAKE"

    def frame(self) -> np.ndarray:
        return self.img.copy()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- 검출 (뒤에서 도는 라이브 검출 + 기록용 동기 검출)


class LiveDetector(threading.Thread):
    """--live-detect일 때만 만든다. 최신 프레임을 뒤에서 계속 검출한다:
    마지막 타겟 주변 ROI(검출기 설정의 roi, 384 px, 약 0.15 s) 추적만 돌고, 전체 프레임 검출(약 1.2 s)은
    직전 전체 검출이 끝난 뒤 full_interval(2 s)이 지나야 다시 한 번 돈다. 추적할 타겟이 없고 전체 검출도 아직 이르면 쉰다.
    detect_now()는 기록용으로 전체 프레임을 동기로 검출한다."""

    def __init__(self, detector, full_interval: float = LIVE_FULL_INTERVAL_SEC, color=None):
        super().__init__(daemon=True)
        self.det = detector
        self.color = color                 # 있으면 라이브 박스도 아군/적군 판정해서 색으로 보여 준다
        self.full_interval = full_interval
        self.lock = threading.Lock()       # 모델 호출을 직렬화한다 (라이브 검출 vs 기록 검출)
        self._frame = None
        self._frame_id = 0
        self._seen = 0
        self.result: dict | None = None    # dets, t(monotonic), roi(bool), ms
        self.pause = threading.Event()
        self.stop = threading.Event()

    @staticmethod
    def plan(roi, last_full: float, now: float, interval: float) -> str | None:
        """다음 검출 종류: 'full'(전체 프레임, interval마다 1회) | 'roi'(타겟 추적) | None(쉼)."""
        if now - last_full >= interval:
            return "full"
        return "roi" if roi is not None else None

    def submit(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._frame_id += 1

    def run(self) -> None:
        roi, last_full = None, -1e9
        while not self.stop.is_set():
            if self.pause.is_set() or self._frame is None or self._frame_id == self._seen:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            mode = self.plan(roi, last_full, now, self.full_interval)
            if mode is None:
                time.sleep(0.05)
                continue
            frame, self._seen = self._frame, self._frame_id
            t = time.perf_counter()
            with self.lock:
                try:
                    dets = self.det.detect(frame, roi if mode == "roi" else None)
                    if self.color is not None:
                        reading = self.color.read_refs(frame)
                        dets = [self.color.classify(frame, d, reading) for d in dets]
                except Exception as e:      # 라이브 검출 실패는 화면만 비우고 계속 돈다
                    print(f"\n[WARN] live detect failed: {e}", flush=True)
                    dets = []
            ms = (time.perf_counter() - t) * 1000
            if mode == "full":
                last_full = time.monotonic()   # 끝난 시각 기준: 전체 검출 뒤에는 항상 interval 동안 ROI 추적만 돈다
            roi = (dets[0].cx, dets[0].cy) if dets else None
            self.result = dict(dets=dets, t=time.monotonic(), roi=mode == "roi", ms=ms)

    def detect_now(self, frame: np.ndarray) -> list:
        """기록용: 전체 프레임을 지금 검출한다 (라이브 검출이 끝나기를 기다린다)."""
        with self.lock:
            return self.det.detect(frame)


def choose_box(dets: list, distance: float, tol: float = 0.25, gap: float = 0.15) -> tuple:
    """(선택된 박스, 애매한지). 박스 높이 h가 선택 거리의 기대값(H_COEF/d)과 가장 가까운 것을 고른다.
    1개면 그대로. 2개 이상이면 1등이 기대값의 +/-tol 안에 있고 2등보다 gap(비율) 이상 더 가까워야 '분명'하다.
    아니면 애매 -> 사용자가 클릭으로 고른다."""
    if not dets:
        return None, False
    if len(dets) == 1:
        return dets[0], False
    he = expected_h(distance)
    score = lambda d: abs(math.log(max(d.h, 1e-6) / he))  # noqa: E731
    ranked = sorted(dets, key=score)
    s0, s1 = score(ranked[0]), score(ranked[1])
    clear = s0 <= math.log(1 + tol) and (s1 - s0) >= math.log(1 + gap)
    return ranked[0], not clear


def nearest_box(dets: list, x: float, y: float):
    return min(dets, key=lambda d: (d.cx - x) ** 2 + (d.cy - y) ** 2)


# ---------------------------------------------------------------- 자율 조준 계산 (p 키, 창·서보 없이 시험할 수 있게 함수로 뺐다)


def h_coef_from_rows(rows: list[dict]) -> tuple[float, str]:
    """거리 추정 계수 (h_px x 거리). 조준 기록 중 박스 높이가 있는 것(= 지금 타겟을 실제로 잰 값)의 중앙값,
    하나도 없으면 H_COEF(예전 타겟 b2 실측 59.1)."""
    v = [r["h_px"] * r["distance_m"] for r in rows if r.get("h_px")]
    if v:
        return float(np.median(v)), f"measured x{len(v)}"
    return H_COEF, "default"


def aim_point_for(rows: list[dict], distance: float, tol: float = AUTO_DIST_TOL):
    """distance에 가장 가까운 기록 거리의 조준점. 반환 (cx, cy, 기록 거리, 기록 수) 또는 None(tol 안에 기록 없음).
    한 거리의 조준점은 cx, cy 각각의 중앙값이다 (잘못 찍은 클릭 몇 개에 흔들리지 않게).
    클릭 기록은 머리 자리이므로 박스 중앙 기준으로 바꿔서 계산한다 (아래)."""
    ds = sorted({r["distance_m"] for r in rows})
    if not ds:
        return None
    d = min(ds, key=lambda x: abs(x - distance))
    if abs(d - distance) > tol + 1e-9:
        return None
    rs = [r for r in rows if r["distance_m"] == d]
    # p는 박스 중앙을 맞추므로 조준점도 박스 중앙 기준으로 낸다.
    # 클릭 기록(source=click)은 사람이 '머리 자리'를 찍은 것 -> 박스 중앙 = 머리 + HEAD_OFFSET_FRAC x h 아래.
    # h는 그 기록의 박스 높이, 없으면 같은 거리 기록들의 h 중앙값, 그것도 없으면 기대값 59.1/d
    hs = [r["h_px"] for r in rs if r.get("h_px")]
    h_ref = float(np.median(hs)) if hs else expected_h(d)
    pts = [(r["cx"], r["cy"] + (HEAD_OFFSET_FRAC * (r.get("h_px") or h_ref) if r.get("source") == "click" else 0.0))
           for r in rs]
    return float(np.median([x for x, _ in pts])), float(np.median([y for _, y in pts])), d, len(rs)


def color_status(color) -> str:
    """판정 설정 상태 한 줄 (화면 상단·시작 출력용, 영어)."""
    if color is None:
        return "color: OFF -> no friend check, p shoots any box"
    mode = getattr(color, "mode", "?")
    if mode == "a_sign":
        return "color: a_sign (a* sign fallback)"
    parts = []
    if getattr(color, "ref", None) is not None:
        parts.append("ref patches")
    cfg = getattr(color, "cfg", None)
    if cfg is not None:
        parts.append(f"color_refs safety {cfg.get('safety_ratio', 0):.1f}")
    if not parts:
        return f"color: {mode}, NO REFS -> all unknown, no friend check"
    return f"color: {mode} ({' + '.join(parts)})"


def judge_text(dets: list) -> str:
    """판정 결과 한 줄 (터미널·로그용, 영어): 'enemy 1, friend 1, unknown 1 (few_px)'."""
    if not dets:
        return "no box"
    n = {k: sum(1 for d in dets if d.side == k) for k in ("enemy", "friend", "unknown")}
    reasons = sorted({d.reason for d in dets if d.side == "unknown"})
    return ", ".join(f"{k} {v}" for k, v in n.items() if v) + (f" ({', '.join(reasons)})" if reasons else "")


def decide_not_friend(dets: list) -> dict:
    """07 p의 결정 규칙: 아군이 아닌 박스(enemy / unknown) 중 h가 가장 큰 것. 반환 dict(action, target, reason).
    action: 'engage' | 'no_target'(박스 없음) | 'friend_only'."""
    if not dets:
        return dict(action="no_target", target=None, reason="no box")
    cand = [d for d in dets if d.side != "friend"]
    if not cand:
        return dict(action="friend_only", target=None, reason="friend only")
    t = max(cand, key=lambda d: d.h)
    return dict(action="engage", target=t, reason=f"not friend ({t.side})")


def servo_step(err_px: float, px_per_deg: float, min_deg: float = MIN_STEP_DEG, tol: float = 0.0,
               max_deg: float = AUTO_MAX_STEP_DEG, scale: float = 1.0) -> float:
    """화면 오차(조준점 - 타겟, px) -> 서보 각도 변화(도). px_per_deg는 '서보 +1도에 타겟이 화면에서 몇 px 움직이나'(부호 포함).
    오차가 tol 안이면 0 (그 축은 맞았다). scale은 이동량 배율(목표를 넘어설 때마다 자율 조준이 절반으로 줄인다).
    계산값이 min_deg보다 작으면 min_deg만큼 움직인다 (처음 min_deg는 SG90 dead band 0.9도, 역시 넘어설 때마다 절반)."""
    if px_per_deg == 0 or abs(err_px) <= tol:
        return 0.0
    deg = max(-max_deg, min(max_deg, scale * err_px / px_per_deg))
    if abs(deg) < min_deg:
        deg = math.copysign(min_deg, deg)
    return deg


def image_shift(a: np.ndarray, b: np.ndarray, scale: int = 4) -> tuple[float, float, float]:
    """프레임 a -> b에서 화면 전체가 밀린 양 (dx, dy, 응답 0~1). 서보를 돌린 뒤 타겟이 어디로 갔는지 따라가는 데 쓴다.
    1/scale로 줄인 흑백 영상의 위상 상관(phase correlation). 응답이 작으면 믿을 수 없다."""
    def prep(img):
        small = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ga, gb = prep(a), prep(b)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    return float(dx) * scale, float(dy) * scale, float(resp)


def learn_gain(old: float, moved_deg: float, shift_px: float) -> float:
    """실제로 움직인 각도와 화면 이동량으로 px/도를 다시 잰다. 이동이 dead band보다 작거나 값이 이상하면 이전 값을 쓴다.
    부호도 함께 배운다 (서보 방향이 가정과 반대여도 다음 이동부터 맞는 쪽으로 간다)."""
    if abs(moved_deg) < AUTO_LEARN_MIN_DEG - 1e-6 or abs(shift_px) < 3.0:
        return old
    k = shift_px / moved_deg
    return k if 5.0 <= abs(k) <= 30.0 else old    # 실측 10~13. 10-06에 35가 한 번 튀었다 -> 범위 밖은 무시


# ---------------------------------------------------------------- 저장


def _sync(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def _num(s: str) -> float | None:
    s = (s or "").strip()
    return float(s) if s else None


def parse_row(raw: dict) -> dict:
    """CSV 문자열 행 -> 숫자 행 (빈 칸은 None)."""
    r = dict(raw)
    r["distance_m"] = dist_key(float(raw["distance_m"]))
    for k in ("cx", "cy", "h_px", "w_px", "conf", "yaw_cmd", "pitch_cmd"):
        r[k] = _num(raw.get(k))
    return r


class AimStore:
    """calib/aim_points.csv + calib/shots/*.png. 이전 세션의 기록도 함께 읽어 화면과 요약에 쓴다."""

    def __init__(self, calib_dir: Path):
        self.dir = Path(calib_dir)
        self.shots = self.dir / "shots"
        self.shots.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "aim_points.csv"
        self.table_path = self.dir / "aim_table.json"
        self.rows: list[dict] = []
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with self.csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != CSV_COLUMNS:
                    raise SystemExit(f"[ERROR] {self.csv_path} has different columns: {reader.fieldnames}\n"
                                     f"        Expected: {CSV_COLUMNS}\n"
                                     "        Move the old file away and run again.")
                self.rows = [parse_row(r) for r in reader if r.get("png")]

    def count(self, distance: float | None = None) -> int:
        if distance is None:
            return len(self.rows)
        return sum(1 for r in self.rows if r["distance_m"] == dist_key(distance))

    def next_index(self, distance: float) -> int:
        d = dist_key(distance)
        biggest = 0
        for name in [r["png"] for r in self.rows] + [p.name for p in self.shots.glob(f"{d:.1f}m_*.png")]:
            m = PNG_RE.match(name)
            if m and float(m.group(1)) == d:
                biggest = max(biggest, int(m.group(2)))
        return biggest + 1

    def append(self, row: dict, frame: np.ndarray) -> Path:
        """PNG(원본 프레임)를 먼저 저장하고 CSV에 한 줄 더한다. 둘 다 디스크에 내려쓴다."""
        png = self.shots / row["png"]
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"PNG encoding failed: {png}")
        tmp = png.with_suffix(".png.tmp")
        with tmp.open("wb") as f:
            f.write(buf.tobytes())
            _sync(f)
        os.replace(tmp, png)

        is_new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(CSV_COLUMNS)
            w.writerow([row[k] for k in CSV_COLUMNS])
            _sync(f)
        self.rows.append(parse_row({k: str(row[k]) for k in CSV_COLUMNS}))
        return png


# ---------------------------------------------------------------- 요약 / 조준표


def summarize(rows: list[dict]) -> dict[float, dict]:
    """거리별 통계 (거리 오름차순). h는 검출 기록(source=detect)만의 평균이고, 검출 기록이 없는 거리는 기대값 H_COEF/d를 쓴다."""
    def std(a: np.ndarray) -> float:
        return float(a.std(ddof=1)) if len(a) >= 2 else 0.0

    out: dict[float, dict] = {}
    for d in sorted({r["distance_m"] for r in rows}):
        rs = [r for r in rows if r["distance_m"] == d]
        cx = np.array([r["cx"] for r in rs], float)
        cy = np.array([r["cy"] for r in rs], float)
        hs = np.array([r["h_px"] for r in rs if r["h_px"] is not None], float)
        n_click = sum(1 for r in rs if r["source"] == "click")
        out[d] = dict(
            distance_m=d, n=len(rs), n_detect=len(rs) - n_click, n_click=n_click,
            cx=float(cx.mean()), cy=float(cy.mean()), cx_std=std(cx), cy_std=std(cy),
            h=float(hs.mean()) if len(hs) else expected_h(d), h_std=std(hs),
            h_expected=expected_h(d), h_src="measured" if len(hs) else "expected",
            warn=max(std(cx), std(cy)) > STD_WARN_PX,
        )
    return out


def build_aim_table(summary: dict[float, dict], frame_size: tuple[int, int], source_csv: Path | str) -> dict:
    """Aimer(target_pipeline.py)가 읽는 형식. h_px 오름차순으로 cx, cy를 1차 보간한다."""
    per = sorted(summary.values(), key=lambda s: s["h"])
    rnd = lambda v: round(v, 2) if isinstance(v, float) else v  # noqa: E731
    return {
        "_comment": ("07_aim_calib.py가 만든 거리별 조준점 표. 검출 박스 높이 h_px -> 조준점 (cx, cy)를 선형 보간한다"
                     "(범위 밖은 끝값). '박스 높이가 h이면 타겟을 화면의 (cx, cy)에 두면 맞는다'. "
                     "카메라 위치가 바뀌거나 받침·막대를 색칠하면 무효 -> 다시 측정."),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": str(source_csv),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "h_coef": H_COEF,
        "n_points": int(sum(s["n"] for s in per)),
        "h_px": [round(s["h"], 2) for s in per],
        "cx": [round(s["cx"], 1) for s in per],
        "cy": [round(s["cy"], 1) for s in per],
        "per_distance": [{k: rnd(v) for k, v in s.items()} for s in per],
    }


def print_summary(summary: dict[float, dict]) -> list[str]:
    """거리별 요약을 출력하고 경고 문장 목록을 돌려준다."""
    warns: list[str] = []
    print("\n" + "=" * 78)
    print("  dist    n  (det/clk)      cx  +/-sd      cy  +/-sd     h_px  +/-sd   h~59.1/d")
    print("-" * 78)
    for d, s in summary.items():
        h_txt = f"{s['h']:7.1f}" if s["h_src"] == "measured" else f"{s['h']:6.1f}*"
        print(f"  {d:.1f} m {s['n']:3d}  ({s['n_detect']:2d}/{s['n_click']:2d})   {s['cx']:7.1f} {s['cx_std']:5.1f}   "
              f"{s['cy']:7.1f} {s['cy_std']:5.1f}   {h_txt} {s['h_std']:5.1f}   {s['h_expected']:6.1f}")
        if s["warn"]:
            warns.append(f"{d:.1f} m: cx/cy std {s['cx_std']:.1f}/{s['cy_std']:.1f} px > {STD_WARN_PX:g} px "
                         "-> check servo repeatability or camera shake")
        if s["h_src"] == "expected":
            warns.append(f"{d:.1f} m: no detect records, h uses the expected value {s['h']:.1f} px (*)")
    print("=" * 78)
    for w in warns:
        print(f"[WARN] {w}")
    if len(summary) == 1:
        print("[INFO] Only one distance recorded: the table is a constant. Record at 2 or more distances.")
    return warns


# ---------------------------------------------------------------- 그리기


def text(img, s: str, org: tuple[int, int], scale: float, color, thick: int = 1) -> None:
    """검은 외곽선을 깐 글자 (밝은 벽, 어두운 벽 어디서나 읽힌다)."""
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def panel(img, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = cv2.convertScaleAbs(roi, alpha=1 - alpha)


def draw_grid(img, step: int = GRID_PX) -> None:
    """50 px 격자. 100 px 선은 조금 밝게. x 좌표는 아래 가장자리, y 좌표는 왼쪽 가장자리에 적는다."""
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (150, 150, 150) if x % (2 * step) == 0 else (105, 105, 105), 1)
        text(img, str(x), (x + 2, h - 5), 0.38, (230, 230, 230))
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (150, 150, 150) if y % (2 * step) == 0 else (105, 105, 105), 1)
        if y:
            text(img, str(y), (2, y - 3), 0.38, (230, 230, 230))


def draw_dets(img, dets: list, highlight=None, color=(0, 255, 0)) -> None:
    """판정된 박스(reason != 'unclassified')는 적군 빨강 / 아군 초록 / unknown 노랑, 아니면 color. highlight는 굵게."""
    for d in dets:
        judged = getattr(d, "reason", "unclassified") != "unclassified"
        if judged:
            c = SIDE_COLORS.get(d.side, color)
        else:
            c = (0, 255, 255) if d is highlight else color
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, c, 3 if d is highlight else 1)
        cx, cy = int(round(d.cx)), int(round(d.cy))
        cv2.drawMarker(img, (cx, cy), c, cv2.MARKER_CROSS, 18, 1)
        label = f"h{d.h:.0f} {d.conf:.2f} ({cx},{cy})"
        if judged:
            label = (d.side.upper() if d.side != "unknown" else f"? {d.reason}") + "  " + label
        text(img, label, (p1[0], max(14, p1[1] - 6)), 0.45, c)


def draw_points(img, rows: list[dict], summary: dict[float, dict], current: float, p_aim=None) -> None:
    """기록된 조준점(작은 점, 클릭 기록은 빈 원)과 거리별 평균(큰 원)."""
    for r in rows:
        p = (int(round(r["cx"])), int(round(r["cy"])))
        cv2.circle(img, p, 4, dist_color(r["distance_m"]), -1 if r["source"] == "detect" else 1, cv2.LINE_AA)
    for d, s in summary.items():
        c = dist_color(d)
        p = (int(round(s["cx"])), int(round(s["cy"])))
        cv2.circle(img, p, 16, c, 3 if d == current else 2, cv2.LINE_AA)
        if d == current:
            cv2.drawMarker(img, p, c, cv2.MARKER_CROSS, 44, 1)
        text(img, f"{d:.1f}m n{s['n']}", (p[0] + 19, p[1] + 5), 0.45, c)
    if p_aim is not None:      # p가 실제로 박스 중앙을 맞추는 곳 (클릭 기록을 박스 중앙 기준으로 바꾼 중앙값)
        q = (int(round(p_aim[0])), int(round(p_aim[1])))
        cv2.drawMarker(img, q, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
        text(img, f"p aim ({q[0]},{q[1]})", (q[0] + 14, q[1] + 20), 0.45, (255, 0, 255))


def draw_legend(img, store: AimStore, current: float) -> None:
    w = img.shape[1]
    dists = [dist_key(DIST_MIN + i * DIST_STEP) for i in range(int(round((DIST_MAX - DIST_MIN) / DIST_STEP)) + 1)]
    x0, y0, lh = w - 190, 8, 20
    panel(img, x0 - 6, y0 - 4, 196, lh * len(dists) + 10)
    for i, d in enumerate(dists):
        y = y0 + lh * i + 14
        cv2.rectangle(img, (x0, y - 11), (x0 + 12, y - 1), dist_color(d), -1)
        mark = ">" if d == current else " "
        text(img, f"{mark}{d:.1f} m  n={store.count(d)}", (x0 + 18, y), 0.45, (255, 255, 255))


def draw_header(img, lines: list[str]) -> None:
    panel(img, 0, 0, 700, 12 + 22 * len(lines))
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, 26 + 22 * i), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_banner(img, s: str, color, y: int = 130, scale: float = 1.0) -> None:
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, 2)
    x = max(8, (img.shape[1] - tw) // 2)
    panel(img, x - 10, y - th - 10, tw + 20, th + 20, 0.6)
    cv2.putText(img, s, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------- 보정 도구 본체


class AimCalib:
    """키·마우스 -> 서보 이동 / 기록. 창 없이도(window=False) 기록·그리기 함수를 그대로 쓸 수 있다 (시험용)."""

    def __init__(self, args, cam, servos: Servos, detector, store: AimStore, window: bool = True, color=None):
        self.args = args
        self.cam = cam
        self.servos = servos
        self.store = store
        self.distance = dist_key(min(DIST_MAX, max(DIST_MIN, args.distance)))
        self.step = args.step
        self.yaw_lo = args.center_yaw - args.yaw_limit
        self.yaw_hi = args.center_yaw + args.yaw_limit
        self.size = (args.width, args.height)
        self.pending: dict | None = None      # kind: 'pick'(박스 고르기) | 'click'(타겟 위치 클릭), frame, dets
        self.flash_until = 0.0
        self.flash_text = ""
        self.fire_armed_at = time.monotonic() + FIRE_ARM_SEC   # 이 시각 전에는 f 무시
        self.fire_duration = float(getattr(args, "fire_duration", FIRE_DURATION_SEC))
        self.fire_thread: threading.Thread | None = None       # 접근 -> 유지 -> 복귀를 도는 스레드 (살아 있으면 발사 중)
        self.fire_phase = ""                                   # 'SHOOT' | 'HOLD' | 'RETURN'
        self.fire_stop = threading.Event()
        self.fire_done_at = -1e9                               # 마지막 발사가 끝난 시각 (쿨다운 계산)
        self.fire_flash_until = 0.0
        self.marks: list[tuple[int, int]] = []   # 클릭 표시 (누른 순서). e로 저장하거나 c로 지울 때까지 남는다
        self.keys: list[str] = []
        self.clicks: list[tuple[int, int]] = []
        self.kb: RawKeyboard | None = None
        self.detector = detector
        self.color = color                    # 아군/적군 판정 (target_pipeline.ColorClassifier). None이면 아군 구분 없이 쏜다
        self.color_status = color_status(color)
        self.live = (LiveDetector(detector, color=color)
                     if (detector is not None and getattr(args, "live_detect", False)) else None)
        self.last_e: dict | None = None       # 마지막 e 검출 결과: dets, chosen, until (화면에 잠시 보여 준다)
        self.last_auto: dict | None = None    # p로 쏜 직후의 발사 순간 값 (e로 source=auto 저장). 서보가 움직이면 지운다
        # p 자율 조준: 서보 1도에 타겟이 화면에서 움직이는 px (부호 포함). 'a'(yaw +)는 총이 왼쪽 -> 타겟이 오른쪽(+x),
        # 'w'(pitch +)는 총이 위 -> 타겟이 아래(+y)로 가정하고 시작해서, 실제 이동을 잴 때마다 고친다 (세션 동안 유지)
        self.k_yaw = float(getattr(args, "px_per_deg", AUTO_PX_PER_DEG))
        self.k_pitch = float(getattr(args, "px_per_deg", AUTO_PX_PER_DEG))
        self.auto_tol = float(getattr(args, "auto_tol", AUTO_TOL_PX))
        self.h_coef = getattr(args, "h_coef", None)   # 거리 계수 직접 지정 (None이면 조준 기록에서)
        self.check_distance = bool(getattr(args, "check_distance", False))   # False: 박스 크기와 상관없이 쏜다
        self.auto_view: dict | None = None    # 화면 표시용: aim, target, box, text, until
        self.window = window
        if window:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, int(self.size[0] * args.view_scale), int(self.size[1] * args.view_scale))
            cv2.setMouseCallback(WIN, self._on_mouse)

    # ---- 입력 ----
    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((int(x), int(y)))      # WINDOW_NORMAL에서는 창 크기와 상관없이 이미지 좌표가 온다

    def _poll_keys(self) -> None:
        if self.window:
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                self.keys.append(chr(k))
        if self.kb is not None:
            t = self.kb.get(0)
            if t:
                self.keys.append(t)

    def clear_keys(self) -> None:
        self.keys.clear()
        if self.kb is not None:
            self.kb.flush()

    # ---- 화면 ----
    def show(self, img: np.ndarray) -> None:
        if self.window:
            cv2.imshow(WIN, img)
        self._poll_keys()

    def wait(self, seconds: float) -> None:
        """서보가 자리 잡는 동안 화면을 계속 갱신한다 (키는 모아 두고 나중에 처리)."""
        end = time.monotonic() + seconds
        while True:
            self.show(self.render(self.cam.frame()))
            if time.monotonic() >= end:
                break

    def yaw(self) -> float:
        return self.servos.get(YAW_PIN, self.args.center_yaw)

    def pitch(self) -> float:
        return self.servos.get(PITCH_PIN, self.args.center_pitch)

    def render(self, frame: np.ndarray, live: dict | None = None, note: str = "") -> np.ndarray:
        """오버레이를 그린 복사본을 돌려준다 (원본 frame은 건드리지 않는다 -> PNG는 원본 그대로 저장된다)."""
        img = frame.copy()
        if self.pending is not None:
            img = self.pending["frame"].copy()     # 클릭할 프레임을 그대로 보여 준다
        draw_grid(img)
        summary = summarize(self.store.rows)
        draw_points(img, self.store.rows, summary, self.distance, aim_point_for(self.store.rows, self.distance, float("inf")))

        if self.pending is not None:
            draw_dets(img, self.pending["dets"], color=(0, 200, 255))
            if self.pending["kind"] == "pick":
                draw_banner(img, "PICK: click the correct box   (e / ESC: cancel)", (0, 255, 255))
            else:
                draw_banner(img, "NO TARGET - click the target position   (e / ESC: cancel)", (0, 120, 255))
            det_line = f"recording: {len(self.pending['dets'])} box(es) in this frame"
        elif self.detector is None:
            det_line = "det: off (--no-detect, e records a click)"
        elif self.live is None:
            det_line = f"det: on e/p only (full frame)   {self.color_status}"
            if self.last_e is not None and time.monotonic() < self.last_e["until"]:
                draw_dets(img, self.last_e["dets"], highlight=self.last_e["chosen"], color=(0, 200, 255))
        elif live is None:
            det_line = "det: live, waiting for the first result..."
        else:
            dets = live["dets"]
            draw_dets(img, dets)
            age = time.monotonic() - live["t"]
            det_line = (f"det live {len(dets)} box  {'roi' if live['roi'] else 'full'} {live['ms']:.0f} ms  {age:.1f} s ago   "
                        f"{self.color_status}")
            if not dets:
                draw_banner(img, "NO TARGET", (0, 0, 255), y=120, scale=1.2)

        n_here, n_all = self.store.count(self.distance), self.store.count()
        draw_header(img, [
            f"AIM CALIB  dist {self.distance:.1f} m (h~{expected_h(self.distance):.1f}px)  "
            f"recorded here {n_here}  total {n_all}  [{self.cam.label}]",
            f"yaw {self.yaw():6.1f}  pitch {self.pitch():6.1f}  step {self.step:.1f} deg   fire {self.fire_state()}"
            + ("" if self.servos.enabled else "  (servo off)"),
            det_line,
            note or "a/d yaw  w/s pitch  [ ] step  r center  f fire  p auto(enemy)  click mark  c clear  -/+ dist  e record  q quit",
        ] + (["AUTO SHOT: e = HIT -> save as source=auto   (missed: just move on)"]
             if self.last_auto is not None and not self.marks else []))
        draw_legend(img, self.store, self.distance)

        now = time.monotonic()
        av = self.auto_view
        if av is not None and now < av["until"]:
            ax, ay = int(round(av["aim"][0])), int(round(av["aim"][1]))
            cv2.drawMarker(img, (ax, ay), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
            cv2.circle(img, (ax, ay), int(self.auto_tol), (255, 0, 255), 1, cv2.LINE_AA)
            if av.get("target") is not None:
                tx, ty = int(round(av["target"][0])), int(round(av["target"][1]))
                cv2.circle(img, (tx, ty), 10, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.line(img, (tx, ty), (ax, ay), (255, 255, 0), 1, cv2.LINE_AA)
            if av.get("box") is not None:
                draw_dets(img, [av["box"]], highlight=av["box"])
            draw_banner(img, av["text"], (255, 0, 255), y=img.shape[0] - 140, scale=0.7)
        if now < self.fire_flash_until:
            draw_banner(img, "FIRE", (0, 0, 255), y=215, scale=1.8)
        for i, (mx, my) in enumerate(self.marks, 1):
            last = i == len(self.marks)                  # e로 저장되는 것은 마지막 표시: 굵은 노란색
            c = (0, 255, 255) if last else (255, 255, 255)
            cv2.circle(img, (mx, my), 9 if last else 7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(img, (mx, my), 9 if last else 7, c, 2, cv2.LINE_AA)
            text(img, f"o{i}", (mx + 12, my - 8), 0.55, c, 2 if last else 1)
        if now < self.flash_until:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 220, 0), 10)
            draw_banner(img, self.flash_text, (0, 255, 0), y=img.shape[0] - 90)   # 타겟 띠(y 389~632)를 가리지 않게 아래쪽
        return img

    # ---- 기록 ----
    def record(self, frame: np.ndarray, cx: float, cy: float, h: float | None, w: float | None,
               conf: float | None, source: str) -> Path:
        d = self.distance
        idx = self.store.next_index(d)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "distance_m": f"{d:.1f}",
            "cx": f"{cx:.1f}", "cy": f"{cy:.1f}",
            "h_px": "" if h is None else f"{h:.1f}",
            "w_px": "" if w is None else f"{w:.1f}",
            "conf": "" if conf is None else f"{conf:.3f}",
            "yaw_cmd": f"{self.yaw():.1f}", "pitch_cmd": f"{self.pitch():.1f}",   # p 직후 기록이면 서보가 안 움직였으므로 발사 순간 값
            "source": source,
            "png": f"{d:.1f}m_{idx:03d}.png",
        }
        png = self.store.append(row, frame)
        self.pending = None
        self.flash_until = time.monotonic() + FLASH_SEC
        self.flash_text = f"RECORDED  {d:.1f} m  #{idx}  ({cx:.0f}, {cy:.0f})"
        h_txt = f"h {h:.1f}" if h is not None else "h -"
        print(f"\n[RECORDED] {d:.1f} m #{idx}  cx {cx:.1f} cy {cy:.1f} {h_txt}  {source}  "
              f"yaw {self.yaw():.1f} pitch {self.pitch():.1f}  -> {png.name}", flush=True)
        return png

    def record_auto(self) -> Path:
        """p 발사 순간의 값으로 기록한다 (source=auto). e 키에서 부른다."""
        a = self.last_auto
        self.last_auto = None
        png = self.record(a["frame"], a["cx"], a["cy"], a["h"], a["w"], a["conf"], "auto")
        print(f"        (auto shot at {a['time']}: box center ({a['cx']:.1f}, {a['cy']:.1f}) h {a['h']:.1f}, "
              f"yaw {a['yaw']:.1f} pitch {a['pitch']:.1f})", flush=True)
        return png

    def begin_record(self, frame: np.ndarray) -> str:
        """e 키. 반환: 'recorded' | 'pick' | 'click'."""
        if self.last_auto is not None and not self.marks:
            self.record_auto()
            return "recorded"
        dets: list = []
        if self.detector is not None:
            self.show(self.render(frame, note="DETECTING... (full frame)"))
            if self.live is not None:
                self.live.pause.set()
                try:
                    dets = self.live.detect_now(frame)
                finally:
                    self.live.pause.clear()
            else:
                dets = self.detector.detect(frame)
        if self.color is not None and dets:      # 판정해서 아군 박스를 고르지 않게 한다 (10-06: e가 아군 박스를 기록함)
            reading = self.color.read_refs(frame)
            dets = [self.color.classify(frame, d, reading) for d in dets]
        self.last_e = dict(dets=dets, chosen=None, until=time.monotonic() + E_SHOW_SEC)
        if self.marks:
            # 클릭 표시가 있으면 마지막 표시만 저장한다. 그 점을 감싸는 박스(여유 MARK_BOX_PAD px)가 있으면 h·w·conf를 함께 적는다
            mx, my = self.marks[-1]
            n = len(self.marks)
            box = [d for d in dets if d.x1 - MARK_BOX_PAD <= mx <= d.x2 + MARK_BOX_PAD and d.y1 - MARK_BOX_PAD <= my <= d.y2 + MARK_BOX_PAD]
            box = nearest_box(box, mx, my) if box else None
            self.last_e["chosen"] = box
            self.record(frame, float(mx), float(my), box.h if box else None, box.w if box else None,
                        box.conf if box else None, "click")
            print(f"        (saved o{n} of {n} marks" + (f", box h{box.h:.1f} attached)" if box else ", no box around it)"), flush=True)
            self.marks.clear()
            return "recorded"
        if not dets:
            self.pending = dict(kind="click", frame=frame, dets=[])
            print("\n[NO TARGET] Click the target position in the window (e / ESC to cancel).", flush=True)
            return "click"
        cand = [d for d in dets if d.side != "friend"]
        if not cand:
            self.pending = dict(kind="click", frame=frame, dets=dets)
            print("\n[FRIEND ONLY] Only friend boxes. Click the target (head) position (e / ESC to cancel).", flush=True)
            return "click"
        det, ambiguous = choose_box(cand, self.distance)
        if ambiguous:
            self.pending = dict(kind="pick", frame=frame, dets=cand)
            hs = ", ".join(f"h{d.h:.0f}" for d in cand)
            print(f"\n[PICK] {len(cand)} boxes ({hs}), expected h~{expected_h(self.distance):.1f}. "
                  "Click the correct box (e / ESC to cancel).", flush=True)
            return "pick"
        if len(dets) > 1:
            print(f"\n[INFO] {len(dets)} boxes; picked h{det.h:.0f} (expected h~{expected_h(self.distance):.1f}).", flush=True)
        self.last_e["chosen"] = det
        self.record(frame, det.cx, det.cy, det.h, det.w, det.conf, "detect")
        return "recorded"

    def on_click(self, x: int, y: int) -> bool:
        """마우스 클릭. 기록 대기 중(박스 고르기 / 위치 클릭)이면 기록하고 True, 아니면 표시(o1, o2, ...)를 더하고 False."""
        if self.pending is None:
            self.marks.append((x, y))
            print(f"\n[MARK] o{len(self.marks)} ({x}, {y})  (e saves the last mark, c clears)", flush=True)
            return False
        frame = self.pending["frame"]
        if self.pending["kind"] == "pick":
            det = nearest_box(self.pending["dets"], x, y)
            if self.last_e is not None:
                self.last_e["chosen"] = det
                self.last_e["until"] = time.monotonic() + E_SHOW_SEC
            self.record(frame, det.cx, det.cy, det.h, det.w, det.conf, "detect")
        else:
            self.record(frame, float(x), float(y), None, None, None, "click")
        return True

    def cancel_pending(self) -> None:
        if self.pending is not None:
            self.pending = None
            print("\n[CANCEL] Not recorded.", flush=True)

    # ---- 자율 조준 (p) ----
    def _detect(self, frame: np.ndarray, roi=None) -> list:
        """지금 프레임 검출 (라이브 검출이 돌고 있으면 잠시 멈추고 같은 잠금으로)."""
        if self.detector is None:
            return []
        if self.live is not None:
            self.live.pause.set()
            try:
                with self.live.lock:
                    return self.detector.detect(frame, roi)
            finally:
                self.live.pause.clear()
        return self.detector.detect(frame, roi)

    def _abort_requested(self) -> bool:
        """자율 조준 중 q / ESC가 눌렸으면 True (그 키는 소비한다 -> 프로그램은 끝나지 않는다)."""
        if any(k in QUIT_KEYS for k in self.keys):
            self.keys = [k for k in self.keys if k not in QUIT_KEYS]
            return True
        return False

    def _auto_show(self, aim, target, box, msg: str, frame: np.ndarray | None = None) -> None:
        self.auto_view = dict(aim=aim, target=target, box=box, text=msg, until=time.monotonic() + AUTO_SHOW_SEC)
        self.show(self.render(self.cam.frame() if frame is None else frame))

    def _auto_log(self, row: dict) -> None:
        path = self.store.dir / "auto_shots.csv"
        is_new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(AUTO_COLUMNS)
            w.writerow([row.get(k, "") for k in AUTO_COLUMNS])
            _sync(f)

    def _judge(self, frame: np.ndarray, roi=None) -> tuple[list, dict]:
        """검출 + 아군/적군 판정 + 07 결정(decide_not_friend: 아군이 아니면 대상).
        판정기가 없으면 모든 박스가 unknown(= 아군 아님)이라 대상이 된다."""
        dets = self._detect(frame, roi)
        if self.color is not None and dets:
            reading = self.color.read_refs(frame)
            dets = [self.color.classify(frame, d, reading) for d in dets]
        return dets, decide_not_friend(dets)

    def _confirm_enemy(self, frame: np.ndarray, tx: float, ty: float):
        """(tx, ty) 주변 ROI를 다시 검출·판정. 반환 (가장 가까운 박스 또는 None, 그 ROI의 박스들)."""
        dets, _ = self._judge(frame, (tx, ty))
        near = [d for d in dets if math.hypot(d.cx - tx, d.cy - ty) <= AUTO_TRACK_PX]
        return (nearest_box(near, tx, ty) if near else None), dets

    def _find_target(self, tx: float, ty: float):
        """놓친 타겟을 (tx, ty) 근처에서 다시 찾는다. 새 프레임으로 ROI 검출, 안 되면 전체 프레임 검출을 AUTO_FIND_TRIES번.
        반환 (박스 또는 None, 마지막 프레임). 10-06: 먼 타겟(h 31 px)이 ROI 검출 한 번에 빠져 조준 다 해 놓고 취소됨."""
        cur = None
        for _ in range(AUTO_FIND_TRIES):
            cur = self.cam.frame()
            near, _ = self._confirm_enemy(cur, tx, ty)
            if near is None:
                dets, _ = self._judge(cur)
                close = [d for d in dets if math.hypot(d.cx - tx, d.cy - ty) <= AUTO_TRACK_PX]
                near = nearest_box(close, tx, ty) if close else None
            if near is not None:
                return near, cur
        return None, cur

    def auto_aim(self, frame: np.ndarray) -> str:
        """p 키 한 번: 검출 + 아군/적군 판정 -> 적군이면 거리 추정 -> 조준점까지 서보 이동(닫힌 루프) -> 재확인 -> 발사.
        반환: 'fired' | 'busy' | 'arming' | 'no_detector' | 'no_enemy' | 'no_aim_point' |
              'not_converged' | 'not_enemy' | 'limit' | 'aborted' | 'fire_blocked'."""
        if self.pending is not None or self.fire_busy():
            return "busy"
        now = time.monotonic()
        if now < self.fire_armed_at:
            print(f"\n[AUTO] blocked: arming, {self.fire_armed_at - now:.1f} s left", flush=True)
            return "arming"
        if self.detector is None:
            print("\n[AUTO] needs the detector (run without --no-detect).", flush=True)
            return "no_detector"
        self.last_auto = None
        log = dict(timestamp=datetime.now().isoformat(timespec="seconds"), source="detect")

        # 1. 검출 + 판정 + 결정
        self.show(self.render(frame, note="AUTO: detecting + friend/enemy check (full frame)..."))
        dets, dec = self._judge(frame)
        judged = judge_text(dets)
        log["judge"] = judged
        self.last_e = dict(dets=dets, chosen=dec["target"], until=time.monotonic() + AUTO_SHOW_SEC)   # 판정 색으로 보여 준다
        if dec["action"] != "engage":
            why = "friend only" if dec["action"] == "friend_only" else "no target"
            print(f"\n[AUTO] {judged}: {why}. Not firing.   ({self.color_status})", flush=True)
            self.flash_until, self.flash_text = time.monotonic() + 2.5, f"AUTO: {why} - no fire"
            self._auto_log({**log, "result": "no_enemy"})
            return "no_enemy"
        box = dec["target"]
        tx, ty = box.cx, box.cy
        if self.h_coef:
            coef, coef_src = float(self.h_coef), "--h-coef"
        else:
            coef, coef_src = h_coef_from_rows(self.store.rows)
        dist = coef / max(box.h, 1e-6)
        dist_txt = f"h {box.h:.1f} px -> {dist:.2f} m (h_coef {coef:.1f}, {coef_src})"
        log.update(h_px=f"{box.h:.1f}", h_coef=f"{coef:.2f}", dist_est=f"{dist:.2f}", t0_cx=f"{tx:.1f}", t0_cy=f"{ty:.1f}")

        if self.check_distance:
            aim = aim_point_for(self.store.rows, dist)
        else:   # 거리 확인 안 함: 선택 거리에 가장 가까운 기록 거리의 조준점 (추정 거리는 참고로만 남긴다)
            aim = aim_point_for(self.store.rows, self.distance, tol=float("inf"))
            dist_txt += "  [distance not checked]"
        if aim is None:
            have = ", ".join(f"{d:.1f}" for d in sorted({r['distance_m'] for r in self.store.rows})) or "none"
            print(f"\n[AUTO] {judged}: TARGET {box.side} at ({tx:.0f}, {ty:.0f})  {dist_txt}\n"
                  f"       No aim point within +/-{AUTO_DIST_TOL} m (recorded: {have} m). Not firing.\n"
                  f"       If the target is really at {self.distance:.1f} m (selected), run with "
                  f"--h-coef {box.h * self.distance:.1f}", flush=True)
            self.flash_until, self.flash_text = time.monotonic() + 2.5, f"AUTO: no aim point for {dist:.2f} m"
            self._auto_log({**log, "result": "no_aim_point"})
            return "no_aim_point"
        ax, ay, d_ref, n_ref = aim
        log.update(aim_dist=f"{d_ref:.1f}", aim_cx=f"{ax:.1f}", aim_cy=f"{ay:.1f}")
        print(f"\n[AUTO] {judged}: TARGET {box.side} at ({tx:.0f}, {ty:.0f})  {dist_txt}\n"
              f"       aim point {d_ref:.1f} m = ({ax:.0f}, {ay:.0f})  (median of {n_ref} records)   "
              f"px/deg yaw {self.k_yaw:+.1f} pitch {self.k_pitch:+.1f}", flush=True)

        def stop(result: str, msg: str) -> str:
            print(f"[AUTO] {msg}. Not firing.", flush=True)
            self._auto_show((ax, ay), (tx, ty), box, f"AUTO: {msg} - no fire")
            log.update(t_cx=f"{tx:.1f}", t_cy=f"{ty:.1f}", iters=it, yaw_cmd=f"{self.yaw():.1f}", pitch_cmd=f"{self.pitch():.1f}")
            self._auto_log({**log, "result": result, "note": msg})
            return result

        # 2. 닫힌 루프: 움직이고 -> 화면 이동량으로 따라가고 -> ROI 재검출·재판정으로 위치와 적군 여부를 확인
        pitch_lo, pitch_hi = self.args.center_pitch - AUTO_PITCH_LIMIT, self.args.center_pitch + AUTO_PITCH_LIMIT
        prev, it, result, confirmed = frame, 0, "not_converged", True     # confirmed: 지금 위치의 박스가 적군으로 확인됨
        min_y = min_p = MIN_STEP_DEG          # 축별 최소 이동 폭. 목표를 넘어설 때마다 절반 (AUTO_STEP_FLOOR_DEG까지)
        scale_y = scale_p = 1.0               # 축별 이동량 배율. 목표를 넘어설 때마다 절반 (px/도를 잘못 알아 매번 넘어서는 경우)
        sign_y = sign_p = 0                   # 직전 오차 부호 (넘어섰는지 보려고)
        while True:
            if not confirmed:      # 지난 이동 뒤 박스를 놓쳤다 -> 다시 찾아서 확인된 위치로 오차를 잰다 (못 찾으면 안 쏜다)
                found, prev = self._find_target(tx, ty)
                if found is None:
                    return stop("not_enemy", f"target box lost ({AUTO_FIND_TRIES} tries near ({tx:.0f}, {ty:.0f}))")
                if found.side == "friend":
                    box = found
                    return stop("not_enemy", f"tracked box is now friend ({found.reason})")
                box, tx, ty, confirmed = found, found.cx, found.cy, True
            ex, ey = ax - tx, ay - ty
            self._auto_show((ax, ay), (tx, ty), box, f"AUTO #{it}  err ({ex:+.0f}, {ey:+.0f}) px", prev)
            if abs(ex) <= self.auto_tol and abs(ey) <= self.auto_tol:
                result = "converged"
                break
            # 목표를 넘어섰으면(오차 부호가 바뀌면) 그 축의 최소 이동 폭을 절반으로 -> 좌우로 오가며 좁혀 들어간다
            sy_now = 0 if abs(ex) <= self.auto_tol else (1 if ex > 0 else -1)
            sp_now = 0 if abs(ey) <= self.auto_tol else (1 if ey > 0 else -1)
            if sy_now and sign_y and sy_now != sign_y:
                min_y, scale_y = max(AUTO_STEP_FLOOR_DEG, min_y / 2), max(1 / 16, scale_y / 2)
            elif sy_now and sy_now == sign_y and abs(ex) > 2 * self.auto_tol:
                # 같은 방향으로 계속 가는데 아직 멀다 -> 줄였던 폭을 다시 키운다 (10-06: 41번 기어가다 중단)
                min_y, scale_y = min(MIN_STEP_DEG, min_y * 1.5), min(1.0, scale_y * 1.5)
            if sp_now and sign_p and sp_now != sign_p:
                min_p, scale_p = max(AUTO_STEP_FLOOR_DEG, min_p / 2), max(1 / 16, scale_p / 2)
            elif sp_now and sp_now == sign_p and abs(ey) > 2 * self.auto_tol:
                min_p, scale_p = min(MIN_STEP_DEG, min_p * 1.5), min(1.0, scale_p * 1.5)
            sign_y, sign_p = sy_now or sign_y, sp_now or sign_p
            dyaw = servo_step(ex, self.k_yaw, min_y, self.auto_tol, scale=scale_y)
            dpitch = servo_step(ey, self.k_pitch, min_p, self.auto_tol, scale=scale_p)
            dpitch = min(pitch_hi, max(pitch_lo, self.pitch() + dpitch)) - self.pitch()
            if AUTO_MAX_ITER and it >= AUTO_MAX_ITER:
                break
            if self._abort_requested():
                return stop("aborted", "aborted (q/ESC)")
            it += 1
            y0, p0 = self.yaw(), self.pitch()
            self.move(dyaw, dpitch)
            moved_y, moved_p = self.yaw() - y0, self.pitch() - p0
            if abs(moved_y) < 1e-9 and abs(moved_p) < 1e-9:
                return stop("limit", f"servo limit reached (yaw {self.yaw():.1f}, pitch {self.pitch():.1f}), "
                                     f"err ({ex:+.0f}, {ey:+.0f}) px")
            self.wait(AUTO_SETTLE_SEC)
            cur = self.cam.frame()
            sx, sy, resp = image_shift(prev, cur)
            if resp >= AUTO_SHIFT_MIN_RESP:
                self.k_yaw = learn_gain(self.k_yaw, moved_y, sx)
                self.k_pitch = learn_gain(self.k_pitch, moved_p, sy)
                how = f"shift ({sx:+.0f}, {sy:+.0f}) resp {resp:.2f}"
            else:
                sx, sy = moved_y * self.k_yaw, moved_p * self.k_pitch    # 측정 실패: 배운 px/도로 예측
                how = f"shift predicted ({sx:+.0f}, {sy:+.0f}) (resp {resp:.2f} too low)"
            tx, ty = tx + sx, ty + sy
            near, _ = self._confirm_enemy(cur, tx, ty)
            if near is not None:
                if near.side == "friend":
                    box = near
                    return stop("not_enemy", f"tracked box is now {near.side} ({near.reason})")
                box, tx, ty, confirmed = near, near.cx, near.cy, True
                how += f"  box found ({near.side})"
            else:
                confirmed = False
                how += "  box lost (using shift)"
            print(f"  #{it} move yaw {moved_y:+.2f} pitch {moved_p:+.2f} (x{scale_y:.2f}/{scale_p:.2f}, min {min_y:.2f}/{min_p:.2f}) "
                  f"-> yaw {self.yaw():.2f} pitch {self.pitch():.2f}  "
                  f"{how}  target ({tx:.0f}, {ty:.0f})  px/deg {self.k_yaw:+.1f}/{self.k_pitch:+.1f}", flush=True)
            prev = cur

        ex, ey = ax - tx, ay - ty
        log.update(t_cx=f"{tx:.1f}", t_cy=f"{ty:.1f}", err_x=f"{ex:.1f}", err_y=f"{ey:.1f}", iters=it,
                   yaw_cmd=f"{self.yaw():.1f}", pitch_cmd=f"{self.pitch():.1f}",
                   px_per_deg_yaw=f"{self.k_yaw:.2f}", px_per_deg_pitch=f"{self.k_pitch:.2f}")
        if result == "not_converged":
            return stop("not_converged", f"not converged after {it} moves (err {ex:+.0f}, {ey:+.0f} px)")
        # 3. 여기 오면 지금 위치는 항상 확인된 박스다 (놓친 경우는 루프 맨 위에서 다시 찾아 오차를 다시 쟀다)
        if not self.fire():
            self._auto_log({**log, "result": "fire_blocked"})
            return "fire_blocked"
        # 발사 순간 값 (prev = 판정에 쓴 마지막 프레임 = 쏠 때 화면). 맞았으면 e로 source=auto 저장
        self.last_auto = dict(frame=prev, cx=float(tx), cy=float(ty), h=float(box.h), w=float(box.w),
                              conf=float(box.conf), yaw=self.yaw(), pitch=self.pitch(),
                              time=datetime.now().strftime("%H:%M:%S"))
        print(f"[AUTO] FIRED at {box.side.upper()} ({result}) after {it} moves, err ({ex:+.0f}, {ey:+.0f}) px.  "
              "HIT -> press e to save (source=auto). Missed -> do nothing.", flush=True)
        self._auto_show((ax, ay), (tx, ty), box, f"AUTO FIRE ({box.side})  err ({ex:+.0f}, {ey:+.0f}) px  {it} moves")
        self._auto_log({**log, "result": f"fired_{result}", "note": dist_txt})
        return "fired"

    # ---- 발사 ----
    def firing(self) -> bool:
        t = self.fire_thread
        return t is not None and t.is_alive()

    def fire_busy(self) -> bool:
        """발사 중이거나 끝난 지 FIRE_COOLDOWN_SEC가 안 됐으면 True -> 키 입력을 받지 않는다."""
        return self.firing() or time.monotonic() < self.fire_done_at + FIRE_COOLDOWN_SEC

    def fire_state(self) -> str:
        now = time.monotonic()
        if self.firing():
            return self.fire_phase or "SHOOT"
        if now < self.fire_done_at + FIRE_COOLDOWN_SEC:
            return "WAIT"
        if now < self.fire_armed_at:
            return f"ARMING {math.ceil(self.fire_armed_at - now)}s"
        return "READY"

    def fire(self) -> bool:
        """f 키. 스레드에서 READY -> SHOOT 천천히 접근, FIRE_HOLD_SEC 유지, 같은 속도로 READY 복귀
        (메인 루프가 검출로 막혀 있어도 제때 돌아온다)."""
        now = time.monotonic()
        if now < self.fire_armed_at:
            print(f"\n[FIRE] blocked: arming, {self.fire_armed_at - now:.1f} s left", flush=True)
            return False
        if self.fire_busy():
            print("\n[FIRE] blocked: previous shot not finished", flush=True)
            return False
        self.fire_stop.clear()
        self.fire_phase = "SHOOT"
        t = threading.Thread(target=self._fire_sequence, daemon=True)
        self.fire_thread = t
        t.start()
        self.fire_flash_until = now + FIRE_FLASH_SEC
        print(f"\n[FIRE] READY {FIRE_READY:g} -> SHOOT {FIRE_SHOOT:g} in {self.fire_duration:g} s, hold {FIRE_HOLD_SEC:g} s, "
              f"back in {self.fire_duration:g} s   yaw {self.yaw():.1f} pitch {self.pitch():.1f} dist {self.distance:.1f} m"
              + ("" if self.servos.enabled else "  (servo off)"), flush=True)
        return True

    def _fire_sequence(self) -> None:
        try:
            if not self.servos.fire_ramp(FIRE_SHOOT, self.fire_duration, self.fire_stop):
                return
            self.fire_phase = "HOLD"
            if self.fire_stop.wait(FIRE_HOLD_SEC):
                return
            self.fire_phase = "RETURN"
            self.servos.fire_ramp(FIRE_READY, self.fire_duration, self.fire_stop)
        finally:
            self.fire_done_at = time.monotonic()
            self.fire_phase = ""

    def fire_safe(self) -> None:
        """종료 시: 발사 동작을 멈추고, 지금 각도에서 같은 속도로 천천히 READY로 돌아간다."""
        t = self.fire_thread
        if t is not None and t.is_alive():
            self.fire_stop.set()
            t.join(timeout=2 * self.fire_duration + FIRE_HOLD_SEC + 1.0)
        self.fire_thread = None
        self.servos.fire_ramp(FIRE_READY, self.fire_duration)

    # ---- 서보 ----
    def center(self) -> None:
        if self.firing():
            return
        self.last_auto = None
        self.servos.center(self.args.center_yaw, self.args.center_pitch)
        src = SERVO_CENTER_PATH.name if SERVO_CENTER_PATH.exists() else "default"
        print(f"\n[CENTER] yaw {self.args.center_yaw:.1f}, pitch {self.args.center_pitch:.1f} (from {src})", flush=True)

    def move(self, d_yaw: float, d_pitch: float) -> None:
        if self.firing():          # 발사 서보가 움직이는 동안 yaw/pitch 명령을 보내지 않는다
            return
        self.last_auto = None      # 자세가 바뀌면 p 발사 순간 값은 더 이상 지금 자세가 아니다
        if d_pitch:
            self.servos.goto(PITCH_PIN, self.pitch() + d_pitch, settle=0.0)
        if d_yaw:
            target = min(self.yaw_hi, max(self.yaw_lo, self.yaw() + d_yaw))
            self.servos.goto(YAW_PIN, target, settle=0.0)
            if target in (self.yaw_lo, self.yaw_hi):
                print(f"\n[LIMIT] yaw {target:.1f} is the limit.", flush=True)

    def set_distance(self, delta: float) -> None:
        self.distance = dist_key(min(DIST_MAX, max(DIST_MIN, self.distance + delta)))
        print(f"\n[DIST] {self.distance:.1f} m  (expected h~{expected_h(self.distance):.1f} px, "
              f"recorded {self.store.count(self.distance)})", flush=True)

    # ---- 메인 루프 ----
    def run(self) -> None:
        if self.live is not None:
            self.live.start()
        while not _quit:
            frame = self.cam.frame()
            if self.live is not None and self.pending is None:
                self.live.submit(frame)

            keys, self.keys = self.keys, []
            clicks, self.clicks = self.clicks, []
            if self.fire_busy():
                # 발사 중 + 끝난 뒤 FIRE_COOLDOWN_SEC: 키는 버린다 (yaw/pitch 명령도 나가지 않는다). 종료 키만 받는다
                keys = [k for k in keys if k in QUIT_KEYS]
            d_yaw = d_pitch = 0.0
            for k in keys:
                if k in QUIT_KEYS:
                    if self.pending is not None:
                        self.cancel_pending()
                    else:
                        return
                elif k == "a":
                    d_yaw += self.step
                elif k == "d":
                    d_yaw -= self.step
                elif k == "w":
                    d_pitch += self.step
                elif k == "s":
                    d_pitch -= self.step
                elif k == "[":
                    self.step = max(round(MIN_STEP_US / US_PER_DEG, 1), round(self.step - 0.5, 1))
                elif k == "]":
                    self.step = min(15.0, round(self.step + 0.5, 1))
                elif k == "r":
                    self.center()
                elif k == "f":
                    self.fire()
                elif k == "p" and self.pending is None:
                    d_yaw = d_pitch = 0.0              # p 앞에 눌린 수동 이동은 버린다 (자율 조준이 자세를 정한다)
                    self.auto_aim(frame)
                    self.clear_keys()              # 자율 조준 중에 쌓인 키는 버린다 (발사 중 키 차단은 위에서 이어진다)
                    frame = self.cam.frame()
                    break
                elif k == "c" and self.marks:
                    print(f"\n[MARK] cleared {len(self.marks)} marks", flush=True)
                    self.marks.clear()
                elif k == "-" and self.pending is None:
                    self.set_distance(-DIST_STEP)
                elif k in ("+", "=") and self.pending is None:
                    self.set_distance(+DIST_STEP)
                elif k == "e":
                    if self.pending is not None:
                        self.cancel_pending()
                    else:
                        self.begin_record(frame)
                        self.clear_keys()          # 검출하는 동안 쌓인 반복 입력은 버린다
                        frame = self.cam.frame()
            if d_yaw or d_pitch:
                self.move(d_yaw, d_pitch)
            for x, y in clicks:
                self.on_click(x, y)
            self.show(self.render(frame, self.live.result if self.live is not None else None))

    def finish(self) -> dict | None:
        """종료 시 요약 출력 + aim_table.json 저장 (CSV 전체 기준). 기록이 없으면 None."""
        if self.live is not None:
            self.live.stop.set()
        rows = self.store.rows
        if not rows:
            print("\n[DONE] No records. aim_table.json was not written.")
            return None
        summary = summarize(rows)
        print_summary(summary)
        table = build_aim_table(summary, self.size, self.store.csv_path)
        self.store.table_path.write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[SAVED] {self.store.table_path}  ({len(summary)} distances, {table['n_points']} points)")
        print(f"        csv {self.store.csv_path}   png {self.store.shots}/")
        return table

    def close(self) -> None:
        if self.window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


# ---------------------------------------------------------------- 메인


def main() -> int:
    ap = argparse.ArgumentParser(description="Aim-point calibration per distance (record where the target appears when a shot hits)")
    ap.add_argument("--distance", type=float, default=DIST_MIN, help=f"Starting distance in m ({DIST_MIN}~{DIST_MAX})")
    ap.add_argument("--step", type=float, default=2.0, help="Degrees per key press (same as 05_manual_capture.py)")
    ap.add_argument("--yaw-limit", type=float, default=30.0, help="Yaw may move at most this far from the center")
    ap.add_argument("--center-yaw", type=float, default=CENTER_YAW)
    ap.add_argument("--center-pitch", type=float, default=CENTER_PITCH)
    ap.add_argument("--settle", type=float, default=0.4, help="Wait after centering (sec)")
    ap.add_argument("--fire-duration", type=float, default=FIRE_DURATION_SEC,
                    help=f"Seconds for the fire servo to go READY->SHOOT in {FIRE_STEP_DEG} deg steps (return uses the same speed)")

    ap.add_argument("--px-per-deg", type=float, default=AUTO_PX_PER_DEG,
                    help="p auto aim: starting guess of pixels per servo degree (learned from real moves)")
    ap.add_argument("--auto-tol", type=float, default=AUTO_TOL_PX, help="p auto aim: fire when the target is within this many px")
    ap.add_argument("--check-distance", action="store_true",
                    help="p auto aim: fire only if the box-height distance matches a recorded distance (default: no check)")
    ap.add_argument("--h-coef", type=float, default=None,
                    help="p auto aim: distance = h_coef / box height. Default: from the hit records (h x distance)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Camera config (03_preview.py, key f)")
    ap.add_argument("--allow-auto", action="store_true", help="Run without a camera config (auto exposure)")
    ap.add_argument("--pipeline-config", type=Path, default=PIPELINE_CONFIG, help="pipeline_config.json (detector block)")
    ap.add_argument("--calib-dir", type=Path, default=CALIB_DIR, help="Where aim_points.csv, shots/, aim_table.json go")
    ap.add_argument("--live-detect", action="store_true",
                    help=f"Also draw boxes live: ROI tracking only, a full frame at most every {LIVE_FULL_INTERVAL_SEC:g} s "
                         "(default: detect only when e is pressed)")

    ap.add_argument("--fake", type=Path, default=None, help="Image file instead of the camera (test)")
    ap.add_argument("--no-servo", action="store_true", help="Do not move the servos (test)")
    ap.add_argument("--no-detect", action="store_true", help="Do not load the detector (UI/servo check; e records clicks only)")
    ap.add_argument("--width", type=int, default=DEFAULT_MAIN_SIZE[0])
    ap.add_argument("--height", type=int, default=DEFAULT_MAIN_SIZE[1])
    ap.add_argument("--view-scale", type=float, default=1.0, help="Window size relative to the frame")
    args = ap.parse_args()

    if not (0.0 <= args.fire_duration <= 10.0):
        print("[ERROR] --fire-duration must be 0~10 s.")
        return 1
    if args.step * US_PER_DEG < MIN_STEP_US:
        print(f"[ERROR] Step {args.step} deg = {args.step * US_PER_DEG:.1f} us is smaller than the SG90 dead band ({MIN_STEP_US} us).")
        return 1
    if not (DIST_MIN - 1e-9 <= args.distance <= DIST_MAX + 1e-9):
        print(f"[ERROR] --distance must be {DIST_MIN}~{DIST_MAX} m.")
        return 1
    if args.fake is None and not args.config.exists() and not args.allow_auto:
        print(f"[ERROR] Camera config file not found: {args.config}")
        print("        Press f in 03_preview.py to save fixed values, or add --allow-auto.")
        return 1

    print("=" * 70)
    print("  07 AIM CALIB - record where the target appears in the frame when a shot HITS")
    print("  * The camera must be fixed to the gun. If it moves, the whole table is invalid.")
    print("  * Repainting the stand/rod changes h -> measure again.")
    print("  * Press e ONLY when the shot actually hit.")
    print(f"  * f = fire (GPIO{FIRE_PIN}: READY {FIRE_READY:g} -> SHOOT {FIRE_SHOOT:g} in {args.fire_duration:g} s, "
          f"{FIRE_STEP_DEG} deg steps, hold {FIRE_HOLD_SEC:g} s, back at the same speed). Disabled for the first {FIRE_ARM_SEC:g} s.")
    print(f"  * Output: {args.calib_dir}  (aim_points.csv, shots/, aim_table.json)")
    print("=" * 70)

    signal.signal(signal.SIGINT, _on_sigint)

    detector = color = None
    if not args.no_detect:
        t0 = time.perf_counter()
        print(f"[LOAD] detector + friend/enemy check from {args.pipeline_config} ...", flush=True)
        from target_pipeline import Pipeline
        with contextlib.redirect_stdout(sys.stderr):   # ultralytics 로딩 로그
            pipe = Pipeline(str(args.pipeline_config))   # 08_detect_check.py와 같은 검출기 + 판정기
        detector, color = pipe.detector, pipe.color
        blank = np.full((args.height, args.width, 3), 128, np.uint8)
        detector.detect(blank)                                               # 첫 추론이 느린 것을 미리 치른다
        if args.live_detect:
            detector.detect(blank, (args.width / 2, args.height / 2))
        print(f"[LOAD] done in {time.perf_counter() - t0:.1f} s   live detect {'ON' if args.live_detect else 'off (e/p only)'}", flush=True)
        print(f"[COLOR] {color_status(color)}", flush=True)

    store = AimStore(args.calib_dir)
    if store.rows:
        per = ", ".join(f"{d:.1f}m x{v['n']}" for d, v in summarize(store.rows).items())
        print(f"[CSV] {store.csv_path}: {len(store.rows)} records from before ({per})")

    if args.fake is not None:
        cam = FakeCamera(args.fake, (args.width, args.height))
        print(f"[FAKE] {args.fake}")
    else:
        print("[START] Opening the camera...")
        try:
            cam = RealCamera((args.width, args.height), args.config if args.config.exists() else None)
        except Exception as e:
            print(f"[ERROR] Could not open the camera: {e}", file=sys.stderr)
            return 1

    app = None
    servos = None
    try:
        with RawKeyboard() as kb:
            servos = Servos(enabled=not args.no_servo, settle=args.settle)
            servos.fire_ready()                      # 시작: 발사 서보를 먼저 READY에 둔다
            if servos.enabled:
                print(f"[FIRE] ready position {FIRE_READY:g} deg (f is enabled after {FIRE_ARM_SEC:g} s)")
            app = AimCalib(args, cam, servos, detector, store, window=True, color=color)
            app.kb = kb
            servos.sleep = app.wait                  # 서보 대기 중에도 화면이 갱신된다
            if servos.enabled:
                print(f"[CENTER] Moving to the center (yaw {args.center_yaw}, pitch {args.center_pitch})...")
            servos.center(args.center_yaw, args.center_pitch)
            print(f"[READY] dist {app.distance:.1f} m   a/d yaw  w/s pitch  [ ] step  r center  f fire  p auto(enemy only)  -/+ dist  e record  q quit", flush=True)
            app.run()
    finally:
        if app is not None:
            app.finish()
        if servos is not None:
            servos.sleep = time.sleep
            if app is not None:
                app.fire_safe()                      # 종료: 발사 동작 중단 + 천천히 READY
            else:
                servos.fire_ready()
            if servos.enabled:
                try:
                    print(f"[FIRE] ready position {FIRE_READY:g} deg")
                    print("[CENTER] Returning to the center before releasing the servos...")
                    servos.center(args.center_yaw, args.center_pitch)   # 기다리는 동안 발사 서보도 READY에 도달한다
                except Exception as e:
                    print(f"[WARN] Could not return to the center: {e}")
            servos.release()
        if app is not None:
            app.close()
        cam.close()
    print("[EXIT]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
