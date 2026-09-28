"""colorlab - 라즈베리파이 카메라용 색상 측정 도구."""

from pathlib import Path

# 프로젝트 루트 (= /home/pa8/src)
ROOT = Path(__file__).resolve().parent.parent

# 03_preview.py가 사용하는 폴더 (파일을 저장하는 코드가 필요할 때 생성한다)
CAPTURES = ROOT / "03_captures"
LOGS = ROOT / "03_logs"
CONFIG = ROOT / "03_config"

__all__ = ["ROOT", "CAPTURES", "LOGS", "CONFIG"]
