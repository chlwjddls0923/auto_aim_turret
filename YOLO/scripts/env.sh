# source 전용 (scripts/*.sh 가 자동으로 읽음). 프로젝트 루트로 이동하고, 캐시·설정을 프로젝트 폴더 안에 가둔다.
# 공용 계정(GB10)에서 ~/.config/Ultralytics, ~/.cache/pip, ~/.cache/torch 를 건드리지 않기 위함.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics" TORCH_HOME="$ROOT/.cache/torch" PIP_CACHE_DIR="$ROOT/.cache/pip" MPLCONFIGDIR="$ROOT/.cache/matplotlib"
mkdir -p "$YOLO_CONFIG_DIR" "$TORCH_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$ROOT/logs"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"
cd "$ROOT"
DS="$ROOT/handover/dataset"        # 학습 데이터셋 (tools/package_dataset.py 출력)
