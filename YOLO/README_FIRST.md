# 먼저 읽기 — 로봇 경기 타겟 검출 프로젝트 (DGX → GB10 이전본)

이 압축 파일 하나에 프로젝트 전체가 들어 있다: 원본 데이터, 라벨, 학습 데이터셋, 가중치와 학습 결과, 코드, 문서.
DGX는 2026-10-05(월)까지만 쓸 수 있으므로 **그 뒤로는 이 폴더가 유일한 원본**이다.

## 1. 풀기와 확인

```bash
cd ~/<내 폴더>                                   # 공용 계정이므로 개인 폴더 안에서만 작업
sha256sum -c yolo_target_full.tar.zst.sha256     # 전송 중 손상 확인
tar --zstd -xf yolo_target_full.tar.zst          # zstd 옵션이 안 되면: zstd -dc yolo_target_full.tar.zst | tar -xf -
cd YOLO
sha256sum -c --quiet SHA256SUMS && echo "전 파일 OK"
```

## 2. 환경 세팅

```bash
python3 -m venv venv
source scripts/env.sh                            # venv 활성화 + 캐시·설정을 이 폴더 안으로 (공용 ~/.config 오염 방지)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130     # torch 를 반드시 먼저
pip install ultralytics pandas                   # pandas 는 ultralytics 가 같이 설치하지 않는다
bash scripts/setup_gb10.sh                       # 무결성 + 데이터셋 장수 + GPU 동작 확인
```

막히면 [docs/GB10_CHECKLIST.md](docs/GB10_CHECKLIST.md)의 항목별 통과 기준과 대처를 본다.

## 3. 동작 확인 (2분)

```bash
source scripts/env.sh
yolo detect val model=runs/r3/weights/best.pt data=handover/dataset/data.yaml imgsz=1280 batch=16 project=runs name=check_val
python tools/target_pipeline.py --config tools/pipeline_config.json --image test/enemy_000624.jpg
```

- 첫 줄: `val: ... 228 images, 0 backgrounds, 0 corrupt`가 나오고 mAP50 ≈ 0.99면 정상.
- 둘째 줄: 박스 1개와 `"action": "hold"`가 나오면 정상이다(색 기준 파일이 아직 없어서 보류하는 것이 맞다).

## 4. 그다음

- **[RESUME.md](RESUME.md)** — 과제, 데이터, 확정 측정값, 결정 사항, DGX에서 끝낸 학습 결과, **다음 할 일**. 클로드 코드에 넘길 때는 "RESUME.md를 읽고 '다음 할 일'부터 이어서 해줘"라고 하면 된다.
- [docs/HANDOVER_GB10.md](docs/HANDOVER_GB10.md) — 폴더 지도와 각 산출물을 다시 만드는 명령.
- [docs/GB10_CHECKLIST.md](docs/GB10_CHECKLIST.md) — GB10 환경 체크리스트.

문서에 나오는 `/data/YOLO`는 DGX에서의 프로젝트 경로이고, 여기서는 방금 푼 `YOLO/` 폴더다. 도구와 스크립트는 자기 위치를 기준으로 경로를 잡으므로 폴더를 어디에 두어도 동작한다.
