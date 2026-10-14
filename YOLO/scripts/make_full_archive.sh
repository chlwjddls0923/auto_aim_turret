#!/usr/bin/env bash
# make_full_archive.sh — 프로젝트 전체를 단일 압축 파일로 묶는다 (다른 서버로 이전용).
#   bash scripts/make_full_archive.sh            →  _archive/yolo_target_full.tar.zst + .sha256   (풀면 YOLO/ 폴더)
#   PLAIN=1 bash scripts/make_full_archive.sh    →  zstd 가 없는 곳으로 보낼 때: 압축 없는 .tar
#   EXCLUDE_RE='^(runs/r3|runs/3arm|logs)/' bash scripts/make_full_archive.sh   → 학습이 도는 중에 만들 때: 쓰이고 있는 폴더 제외
# 포함: 원본 데이터, 의사 라벨·검수 자료, 학습 데이터셋(handover/dataset), 가중치(r2·r3·3-arm best.pt, NCNN), 학습 결과(csv·yaml), 코드, 문서.
# 제외: 캐시(.cache, *.cache, __pycache__), venv, last.pt·학습 그림, 다시 만들 수 있는 것(handover/dataset_gray, 중간 데이터셋 ds*), 아군 원본의 중복 zip.
# 내부에 SHA256SUMS(전 파일)를 넣는다. 다시 실행하면 덮어쓴다.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd); NAME=$(basename "$ROOT"); OUTDIR=${OUTDIR:-$ROOT/_archive}; mkdir -p "$OUTDIR"
for f in README_FIRST.md RESUME.md; do [ -f $f ] || { echo "$f 없음"; exit 1; }; done
{
  { ls README_FIRST.md RESUME.md TRAINING_DESIGN.md ./*.ipynb 2>/dev/null || true; } | sed 's#^\./##'
  find data/images "data/images(1)" "data2/enemy(26.10.02.1754)" "data2/friend(26.10.02.1839)" data2_labels data2_pseudo data2_pseudo_r2 friend_pseudo \
       handover/dataset tools scripts docs weights -type f ! -name "*.cache" ! -path "*/__pycache__/*"
  find test -maxdepth 1 -type f -name "*.jpg"
  find runs -type f \( -name results.csv -o -name args.yaml -o -name "eval_*.csv" -o -name summary.csv -o -name table.txt -o -name cross_boxes.csv -o -name sweep.json \)
  find runs/r2 runs/r3 runs/3arm -type f \( -name best.pt -o -name results.png \) 2>/dev/null || true
  find logs -type f \( -name "*.log" -o -name "*.out" -o -name "STEP_*" -o -name ALL_DONE \) 2>/dev/null || true
} | grep -Ev "${EXCLUDE_RE:-^$}" | LC_ALL=C sort -u > "$OUTDIR/.files"
xargs -d '\n' sha256sum < "$OUTDIR/.files" > SHA256SUMS
{ cat "$OUTDIR/.files"; echo SHA256SUMS; } | sed "s#^#$NAME/#" > "$OUTDIR/.members"
if [ "${PLAIN:-0}" = 1 ]; then
  OUT="$OUTDIR/yolo_target_full.tar"; tar -C "$ROOT/.." --verbatim-files-from -T "$OUTDIR/.members" -cf "$OUT"
else
  OUT="$OUTDIR/yolo_target_full.tar.zst"; ZSTD=$(command -v zstd || echo /data/miniconda3/bin/zstd)
  tar -C "$ROOT/.." --verbatim-files-from -T "$OUTDIR/.members" -cf - | "$ZSTD" -3 -T8 -q -f -o "$OUT"
fi
(cd "$OUTDIR" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
echo "파일 $(wc -l < "$OUTDIR/.members")개 | 원본 합계 $(xargs -d '\n' du -cb < "$OUTDIR/.files" | tail -1 | awk '{printf "%.0f MB", $1/1e6}') | 압축본 $(du -h "$OUT" | cut -f1) | $(cat "$OUT.sha256")"
rm -f "$OUTDIR/.files" "$OUTDIR/.members"
