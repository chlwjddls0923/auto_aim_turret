"""
서보 중앙값을 한 곳에서 관리한다.

중앙값은 코드가 아니라 측정값이다 (서보 개체와 조립 상태마다 다르고, 서보를 바꾸면 다시 잰다).
그래서 카메라 설정(camera_config.json)과 같은 03_config/ 폴더의 servo_center.json에 저장하고,
05_sweep_capture.py, 05_manual_capture.py, 98_servo_center.py가 모두 여기서 읽고 쓴다.

파일이 없으면 기본값(yaw 95, pitch 90)을 쓴다. 03_config/는 git에 올라가지 않으므로
장비마다 다른 값이 커밋 기록에 섞이지 않는다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import CONFIG

SERVO_CENTER_PATH = CONFIG / "servo_center.json"

# 파일이 없을 때 쓰는 기본값 (deg)
DEFAULT_CENTER_YAW = 95.0
DEFAULT_CENTER_PITCH = 90.0


def load_center(path: Path = SERVO_CENTER_PATH) -> tuple[float, float]:
    """(yaw, pitch) 중앙값을 읽는다. 파일이 없거나 깨졌으면 기본값을 돌려준다."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return float(data["center_yaw"]), float(data["center_pitch"])
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_CENTER_YAW, DEFAULT_CENTER_PITCH


def save_center(yaw: float, pitch: float, path: Path = SERVO_CENTER_PATH, note: str = "") -> Path:
    """(yaw, pitch) 중앙값을 저장한다. 98_servo_center.py가 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "center_yaw": round(float(yaw), 1),
        "center_pitch": round(float(pitch), 1),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path
