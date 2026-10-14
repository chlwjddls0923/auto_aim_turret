#!/usr/bin/env bash
# setup_gb10.sh — 압축을 풀고 venv·패키지 설치(docs/GB10_CHECKLIST.md 2~3번)를 끝낸 뒤 실행. 다시 실행해도 안전.
#   bash scripts/setup_gb10.sh
# 하는 일: (1) SHA256SUMS 로 전 파일 무결성 확인  (2) 데이터셋 장수 확인  (3) 패키지 import + GPU 동작 확인 (체크리스트 4번)
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "== 1/3 무결성 (SHA256SUMS)"
sha256sum -c --quiet SHA256SUMS
echo "   OK ($(wc -l < SHA256SUMS) 파일)"

echo "== 2/3 데이터셋"
for s in train val test; do
  n_img=$(ls "$DS/$s/images" | wc -l); n_lbl=$(ls "$DS/$s/labels" | grep -c '\.txt$'); n_inv=$(grep -c ",$s," "$DS/inventory.csv")
  echo "   $s: images $n_img / labels $n_lbl / inventory $n_inv"
  [ "$n_img" = "$n_inv" ] && [ "$n_lbl" = "$n_inv" ] || { echo "장수 불일치"; exit 1; }
done

echo "== 3/3 패키지·GPU 확인"
python - <<'PY'
import platform, sys
import torch, torchvision, ultralytics, cv2, numpy, pandas
print(f"python {sys.version.split()[0]} | {platform.machine()} | torch {torch.__version__} | torchvision {torchvision.__version__} | ultralytics {ultralytics.__version__} | cv2 {cv2.__version__} | numpy {numpy.__version__} | pandas {pandas.__version__}")
assert torch.cuda.is_available(), "CUDA 사용 불가 — torch 빌드와 드라이버 불일치 (체크리스트 3번: torch 를 먼저, cu130 인덱스로 설치)"
name, cap, archs = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list()
print(f"GPU {name} | capability {cap} | arch_list {archs}")
if f"sm_{cap[0]}{cap[1]}" not in archs and f"sm_{cap[0]}0" not in archs: print(f"⚠️  arch_list 에 sm_{cap[0]}{cap[1]} 없음 — 아래 conv 테스트가 'no kernel image' 로 실패하면 빌드 불일치")
x = torch.randn(1, 3, 960, 1280, device="cuda"); conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
with torch.autocast("cuda"): y = conv(x)
print(f"conv 960x1280 OK ({y.dtype})")
from ultralytics.utils import SETTINGS, USER_CONFIG_DIR
print(f"ultralytics 설정 폴더 {USER_CONFIG_DIR} (프로젝트 폴더 안이어야 함) | weights_dir {SETTINGS['weights_dir']}")
PY
echo "✅ setup 완료. 다음에 할 일은 RESUME.md 의 '다음 할 일'."
