#!/usr/bin/env bash
# train_3arm.sh — 색 불변성 3-arm 실험. 목적: 색지를 바꿔도 재학습이 필요 없는(색맹) 학습 설정 고르기.
#   ARM  : h0015 (hsv_h=0.015) / h03 (hsv_h=0.3) / gray (흑백 데이터셋 — tools/make_gray.py 가 handover/dataset_gray/ 를 만듦)
#   SIDE : enemy (b1+b2 적군만 학습 → 아군 전체 검출률) / friend (아군만 학습 → 적군 b2 전체 검출률)
# 6 run, 모두 yolo11n.pt(COCO)에서 시작. 모델 선택(best.pt)은 **같은 색** val 로만 한다(교차색에 오염되지 않게).
# 각 run 뒤 평가 3종: cross(학습에 없던 반대색 전체) / self_val(같은 색 val) / test_human(방 A 사람 라벨 98장, 적군색).
# 예상 시간 ≈ s/epoch × 180 (적군 852 + 아군 571 = 1,423장 → arm 하나가 r3 60 epoch 분량). 끝난 run 은 건너뛴다(중간에 끊긴 run 은 처음부터 다시 학습).
#   BATCH=8 WORKERS=8 bash scripts/train_3arm.sh             # DGX
#   ARMS="gray" SIDES="friend" bash scripts/train_3arm.sh    # 일부만
# 결과: runs/3arm/summary.csv + 마지막에 출력되는 표 (runs/3arm/table.txt 에도 저장)
set -euo pipefail
source "$(dirname "$0")/env.sh"
EPOCHS=${EPOCHS:-60}; BATCH=${BATCH:-32}; WORKERS=${WORKERS:-10}; ARMS=${ARMS:-"h0015 h03 gray"}; SIDES=${SIDES:-"enemy friend"}
OUT="$ROOT/runs/3arm"; SUMMARY="$OUT/summary.csv"; mkdir -p "$OUT"
for ARM in $ARMS; do
  case $ARM in
    h0015) D="$DS"; HSV_H=0.015 ;;
    h03)   D="$DS"; HSV_H=0.3 ;;
    gray)  D="${DS}_gray"; HSV_H=0.015
           [ -f "$D/data.yaml" ] || python tools/make_gray.py --src "$DS" --dst "$D" --workers "$WORKERS" ;;
    *) echo "알 수 없는 ARM: $ARM (h0015|h03|gray)"; exit 1 ;;
  esac
  for SIDE in $SIDES; do
    case $SIDE in
      enemy)  CROSS=list_eval_friend_all.txt ;;
      friend) CROSS=list_eval_enemy_b2_all.txt ;;
      *) echo "알 수 없는 SIDE: $SIDE (enemy|friend)"; exit 1 ;;
    esac
    RUN=${ARM}_${SIDE}; BEST="$OUT/$RUN/weights/best.pt"
    if [ ! -f "$OUT/$RUN/TRAINED" ]; then
      yolo detect train model=weights/yolo11n.pt data="$D/data_${SIDE}_only.yaml" imgsz=1280 epochs=$EPOCHS patience=20 \
        batch=$BATCH workers=$WORKERS cache=ram device=0 seed=0 \
        hsv_h=$HSV_H hsv_s=0.7 hsv_v=0.4 translate=0.1 scale=0.5 fliplr=0.5 mosaic=1.0 close_mosaic=15 \
        project="$OUT" name="$RUN" exist_ok=True 2>&1 | tee "logs/3arm_$RUN.log"
      touch "$OUT/$RUN/TRAINED"
    else echo "== $RUN 학습 완료됨 — 건너뜀"; fi
    if [ ! -f "$OUT/$RUN/DONE" ]; then
      python tools/eval_cross_color.py --model "$BEST" --list "$D/$CROSS" --imgsz 1280 --tag "$RUN" --append-csv "$SUMMARY" --out "$OUT/$RUN/cross_boxes.csv"
      python tools/eval_cross_color.py --model "$BEST" --list "$D/list_val_${SIDE}.txt" --imgsz 1280 --tag "$RUN" --append-csv "$SUMMARY"
      python tools/eval_cross_color.py --model "$BEST" --list "$D/list_test_human.txt" --imgsz 1280 --tag "$RUN" --append-csv "$SUMMARY"
      touch "$OUT/$RUN/DONE"
    fi
  done
done
python - "$SUMMARY" <<'PY' | tee "$OUT/table.txt"
import sys, pandas as pd
S = pd.read_csv(sys.argv[1]).drop_duplicates(['tag', 'list'], keep='last')
S['arm'] = S.tag.str.rsplit('_', n=1).str[0]; S['train_side'] = S.tag.str.rsplit('_', n=1).str[1]
S['eval'] = S['list'].map(lambda s: 'self_val' if s.startswith('list_val') else ('test_human' if s.startswith('list_test') else 'cross'))
cols = ['recall', 'mean_conf', 'center_err_med', 'h_abs_err_med', 'fp_per_frame']
print('===== 3-arm 결과 (cross = 학습에 없던 반대색 전체, self_val = 같은 색 val, test_human = 방 A 사람 라벨 98장·적군색) =====')
print(S.set_index(['arm', 'train_side', 'eval'])[['images'] + cols].sort_index().round(3).to_string())
print('\n반대색 recall (행 arm × 열 학습색):'); print(S[S['eval'] == 'cross'].pivot(index='arm', columns='train_side', values='recall').round(3).to_string())
print('※ cross·self_val 의 GT 는 의사 라벨 → 절대값이 아니라 arm 간 상대 비교로 읽는다. test_human 만 사람 라벨.')
PY
