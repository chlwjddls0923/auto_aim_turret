# GB10 환경 체크리스트

GB10(aarch64, CUDA 13.0, 통합 메모리 121 GiB, **공용 계정**)에서 이 프로젝트를 이어가기 위한 환경 확인 목록이다.
압축을 풀고 확인하는 순서는 [README_FIRST.md](../README_FIRST.md), 프로젝트 내용은 [RESUME.md](../RESUME.md)에 있다.

모든 명령은 프로젝트 폴더(`YOLO/`)에서 실행한다. 공용 계정이므로 **개인 폴더 밖에는 아무것도 쓰지 않는다.**
이 목록은 DGX(x86)에서 작성했고 **GB10에서 실제로 돌려 본 적은 없다.** aarch64 휠 유무처럼 GB10에서만 알 수 있는 항목은 직접 확인해야 한다.

## 항목별 확인

| # | 확인 | 명령 | 통과 기준 / 실패 시 |
|---|---|---|---|
| 1 | 플랫폼 | `uname -m`, `nvidia-smi` | `aarch64`, CUDA 13.0(드라이버 580). 다르면 3번의 인덱스 URL을 그 CUDA 버전에 맞춘다 |
| 2 | 개인 폴더 격리 | `source scripts/env.sh` 후 `echo $YOLO_CONFIG_DIR $TORCH_HOME $PIP_CACHE_DIR` | 셋 다 `YOLO/.cache/` 아래. venv도 `YOLO/venv`. `~/.config/Ultralytics`를 만들거나 고치지 않는다 |
| 3 | 설치 순서 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` → `pip install ultralytics pandas` | **torch를 먼저**, 그다음 ultralytics. 순서가 바뀌면 ultralytics가 다른 torch를 끌어와 CUDA가 꺼진다(DGX에서 실제로 발생). 설치 뒤 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`가 `+cu130 True`인지 본다. cu130 aarch64 휠이 없으면 cu128 인덱스 또는 NGC pytorch 컨테이너. `pandas`는 ultralytics 8.4가 같이 설치하지 않으므로 따로 넣는다(`tools/`의 평가·패키징 스크립트가 사용) |
| 4 | 무결성과 GPU | `bash scripts/setup_gb10.sh` | `SHA256SUMS` 전부 OK, train 1423 / val 228 / test 98, `cuda True`, 장치 이름 GB10, capability `(12, 1)`, arch_list에 `sm_120` 또는 `sm_121`, `conv 960x1280 OK`. `no kernel image` 에러면 torch 빌드 불일치 → 3번 다시 |
| 5 | AMP | 학습 로그의 `AMP: checks passed ✅` | 실패하면 FP32로 자동 전환되어 느려진다(학습은 가능). 검사용 `weights/yolo26n.pt`가 들어 있어 다운로드가 필요 없다 |
| 6 | 메모리 | `free -h` | `cache=ram`은 약 5.7 GB를 쓴다(DGX 실측). 가용 20 GiB 이상 유지 |
| 7 | batch | 학습 로그의 `GPU_mem` 열 | 32에서 시작, 여유가 있으면 `BATCH=64`. **`batch=-1`(AutoBatch)은 쓰지 않는다** — 통합 메모리에서 과대 산정 위험. 참고: DGX에서 batch 8 → 4.5~4.7 G이므로 32 ≈ 18 G, 64 ≈ 36 G 예상 |
| 8 | workers | `nproc` | 코어 20개 기준 `WORKERS=8`~`12`. 다른 사용자가 CPU를 쓰고 있으면 낮춘다 |
| 9 | 소요 시간 | `EPOCHS=2 NAME=probe bash scripts/train_r3.sh` 로그의 epoch 당 시간 | 본 학습 전에 먼저 잰다. 참고: DGX A100 MIG에서 batch 8로 약 41 s/epoch(val 포함) |
| 10 | OpenCV | `python -c "import cv2"` | `libGL` 에러면 `pip uninstall -y opencv-python && pip install opencv-python-headless` |
| 11 | NCNN export | `pip install ncnn pnnx` 후 `python tools/export_ncnn.py --model <best.pt> --out weights/ncnn_new --name new` | pnnx의 aarch64 휠이 없으면 GB10에서는 export할 수 없다. r2와 r3의 NCNN 4종은 이미 `weights/ncnn/`, `weights/ncnn_r3/`에 있다. **새 모델을 학습하면 export할 x86 장비가 따로 필요하다** — 이 항목을 가장 먼저 확인해 둘 것 |
| 12 | 다른 사용자 | 시작 전 `nvidia-smi`, `free -h` | 다른 사람 작업이 돌고 있으면 batch를 줄이거나 기다린다. 공용 계정이라 `ps -u`로는 구분되지 않는다 — 내 프로세스는 `ls -l /proc/<PID>/cwd`가 내 폴더인 것. **`pkill python` 금지, 내 PID만 kill** |

## 학습을 다시 돌릴 때

r3와 3-arm은 DGX에서 이미 끝냈다(결과는 `runs/`). 색칠 후 데이터가 들어오거나 라벨을 고쳐서 다시 학습할 때만 아래를 쓴다.

```bash
tmux new -s <내이름>_train
source scripts/env.sh
python tools/package_dataset.py                       # 데이터가 바뀌었으면 데이터셋부터 다시 (sanity 통과 확인)
EPOCHS=2 NAME=probe bash scripts/train_r3.sh          # GPU_mem·s/epoch 확인
NAME=r4 bash scripts/train_r3.sh                      # 본 학습 (이름을 바꿔 r3 결과를 덮어쓰지 않게)
```

시작 로그의 `train: ... N images, 0 backgrounds, 0 corrupt`에서 N이 `handover/dataset/inventory.csv`의 train 장수와 같은지 본다.
loss가 NaN이거나 mAP가 0에 머물면 AMP 문제일 수 있다 → `scripts/train_r3.sh`의 yolo 명령에 `amp=False`를 붙여 다시.
