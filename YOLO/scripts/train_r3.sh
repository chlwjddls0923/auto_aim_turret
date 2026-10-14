#!/usr/bin/env bash
# train_r3.sh — r3 학습: r2 가중치에서 시작, 전체 데이터(handover/dataset 의 train/val), 입력 1280, hsv_h=0.3.
# 끝나면 val(적군/아군)과 사람 라벨 test 를 tools/eval_cross_color.py 로 평가해 runs/<NAME>/eval_summary.csv 에 기록.
#   BATCH=8 WORKERS=8 bash scripts/train_r3.sh                 # DGX A100 MIG 10GB (실측 GPU_mem 4.5~4.7G, 약 41 s/epoch)
#   EPOCHS=2 NAME=r3_probe bash scripts/train_r3.sh            # 새 장비에서는 먼저 2 epoch 로 GPU_mem·s/epoch 확인
#   bash scripts/train_r3.sh                                   # GB10 기본 BATCH=32 (여유 보고 64. batch=-1 AutoBatch 는 통합 메모리에서 과대 산정 위험 → 금지)
# 데이터를 바꿔 다시 학습할 때는 NAME=r4 처럼 이름을 바꾼다.
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=${NAME:-r3}; IMGSZ=${IMGSZ:-1280}
yolo detect train model=${MODEL:-runs/r2/weights/best.pt} data="$DS/data.yaml" imgsz=$IMGSZ epochs=${EPOCHS:-100} patience=30 \
  batch=${BATCH:-32} workers=${WORKERS:-10} cache=ram device=0 seed=0 \
  hsv_h=0.3 hsv_s=0.7 hsv_v=0.4 translate=0.1 scale=0.5 fliplr=0.5 mosaic=1.0 close_mosaic=15 \
  project="$ROOT/runs" name="$NAME" exist_ok=True 2>&1 | tee "logs/$NAME.log"
rm -f "runs/$NAME/eval_summary.csv"
for L in list_val_enemy list_val_friend list_test_human; do
  python tools/eval_cross_color.py --model "runs/$NAME/weights/best.pt" --list "$DS/$L.txt" --imgsz $IMGSZ --tag "$NAME" --append-csv "runs/$NAME/eval_summary.csv" --out "runs/$NAME/eval_${L#list_}.csv"
done
