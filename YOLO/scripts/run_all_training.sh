#!/usr/bin/env bash
# run_all_training.sh — 한 번에 순서대로: r2 기준선 평가 → r3 학습(+평가) → r3 NCNN export → 3-arm 실험.
# 오래 걸리므로 터미널과 분리해서 실행한다:
#   BATCH=8 WORKERS=8 setsid nohup bash scripts/run_all_training.sh > logs/run_all.out 2>&1 &
# 진행 상황: logs/STEP_* 표식 파일, 실패하면 logs/FAILED, 전부 끝나면 logs/ALL_DONE.
set -uo pipefail
source "$(dirname "$0")/env.sh"
rm -f logs/FAILED logs/ALL_DONE
step() {  # step <이름> <명령...> : 표식이 있으면 건너뜀
  local name=$1; shift
  [ -f "logs/STEP_$name" ] && { echo "== $name 완료됨 — 건너뜀"; return 0; }
  echo "== $name 시작 $(date '+%F %T')"
  if "$@"; then date '+%F %T' > "logs/STEP_$name"; echo "== $name 끝 $(date '+%F %T')"; else echo "$name" > logs/FAILED; echo "== $name 실패"; exit 1; fi
}
r2_baseline() {  # 라벨링에 쓴 r2 의 기준선 (학습 해상도 640 과 배포 해상도 1280)
  rm -f runs/r2/eval_summary.csv
  for SZ in 640 1280; do for L in list_val_enemy list_val_friend list_test_human; do
    python tools/eval_cross_color.py --model runs/r2/weights/best.pt --list "$DS/$L.txt" --imgsz $SZ --tag "r2@$SZ" --append-csv runs/r2/eval_summary.csv || return 1
  done; done
}
step r2_baseline r2_baseline
step r3 bash scripts/train_r3.sh
step r3_ncnn python tools/export_ncnn.py --model runs/r3/weights/best.pt --out weights/ncnn_r3 --name r3 --check-image "$DS/val/images/$(ls "$DS/val/images" | head -1)"
step 3arm bash scripts/train_3arm.sh
date '+%F %T' > logs/ALL_DONE
