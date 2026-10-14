#!/usr/bin/env bash
# train_r4.sh — r4: COCO yolo11n.pt 에서 새로 (r3_coco 와 같은 레시피), 데이터 handover/dataset_r4 (기존 + finetune_data(10.08) train/val).
#   EPOCHS=2 NAME=r4_probe bash scripts/train_r4.sh     # 먼저 s/epoch 측정
#   bash scripts/train_r4.sh                            # 본 학습 (epochs 100, patience 30, batch 32)
set -euo pipefail
source "$(dirname "$0")/env.sh"
DS4="$ROOT/handover/dataset_r4"
NAME=${NAME:-r4}; IMGSZ=${IMGSZ:-1280}
yolo detect train model=${MODEL:-weights/yolo11n.pt} data="$DS4/data.yaml" imgsz=$IMGSZ epochs=${EPOCHS:-100} patience=${PATIENCE:-30} \
  batch=${BATCH:-32} workers=${WORKERS:-8} cache=ram device=0 seed=0 \
  hsv_h=0.3 hsv_s=0.7 hsv_v=0.4 translate=0.1 scale=0.5 fliplr=0.5 mosaic=1.0 close_mosaic=15 \
  project="$ROOT/runs" name="$NAME" exist_ok=True 2>&1 | tee "logs/$NAME.log"
