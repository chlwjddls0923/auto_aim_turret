# YOLO — 타겟 검출기 학습

라즈베리파이 포탑이 **멀리 있는 작은 타겟(공 + 막대 + 받침)** 을 찾도록 YOLO 검출기를 학습하는 폴더다.
학습은 GPU 서버(DGX, GB10)에서 하고, 결과 모델만 파이로 옮겨 [PI/](../PI/) 쪽 코드가 실전에서 쓴다.

```
[파이] 사진 촬영 ──▶ [GPU 서버] 라벨 → 데이터셋 → 학습 → NCNN 변환 → 평가 ──▶ [파이] 검출 → 색 판정 → 조준 → 발사
 PI/05_*.py           YOLO/tools, YOLO/scripts                                  PI/pipeline/tools, PI/07_aim_calib.py
```

- 클래스는 1개(`tar`)뿐이다. **아군/적군은 YOLO가 구분하지 않는다.** YOLO는 "타겟이 어디 있는지"만 찾고, 아군/적군은 박스 안의 색(초록 / 마젠타)으로 따로 판정한다.
- 현재 파이에서 쓰는 모델은 **r4**다([모델 이력](#7-모델-이력과-성능)).

---

## 목차

1. [얼마나 작은 타겟인가](#1-얼마나-작은-타겟인가)
2. [폴더 구성과 git에 없는 것](#2-폴더-구성과-git에-없는-것)
3. [학습 환경 준비](#3-학습-환경-준비)
4. [학습 순서 (처음부터 끝까지)](#4-학습-순서-처음부터-끝까지)
5. [학습 설정값 설명](#5-학습-설정값-설명)
6. [실제 상황에 적용하기 (파이)](#6-실제-상황에-적용하기-파이)
7. [모델 이력과 성능](#7-모델-이력과-성능)
8. [주의할 점](#8-주의할-점)
9. [더 자세한 문서](#9-더-자세한-문서)

---

## 1. 얼마나 작은 타겟인가

**결론: 매우 작은 타겟이 맞다.** r4 학습 데이터셋(`handover/dataset_r4`)의 라벨 박스를 전부 재 보면 다음과 같다. 사진은 모두 1280×960이다.

| | 박스 수 | 가로 (중앙값, 범위) | 세로 (중앙값, 범위) | 화면 면적 대비 |
|---|---|---|---|---|
| train | 3,559 | **18 px** (9~35) | **33 px** (22~58) | 약 0.05% |
| val | 440 | 19 px (12~26) | 35 px (21~43) | 약 0.05% |
| test (사람 라벨) | 98 | 16 px (10~20) | 25 px (20~31) | 약 0.03% |

- 1280×960 화면에서 **손톱만 한 크기**다. 화면 전체 면적의 2,000분의 1 정도다.
- 객체 검출에서 흔히 쓰는 COCO 기준(32×32 px 미만이면 "작은 물체")으로 보면 train 박스의 **86%**, val의 98%, test의 100%가 "작은 물체"다. "큰 물체"는 하나도 없다.
- 거리가 멀수록 작아진다. 세로 길이는 대략 **h ≈ 59.1 ÷ 거리(m)** px이다(색칠 전 측정). 1.5 m에서 약 39 px, 2.0 m에서 약 30 px, 2.4 m에서 약 26 px이다.

### 그래서 학습할 때 무엇을 다르게 했나

1. **입력 해상도를 줄이지 않는다 (`imgsz=1280`).** YOLO는 보통 640으로 줄여서 학습하는데, 그러면 박스가 9×16 px 정도로 뭉개진다. 입력 크기별로 학습해 본 결과(val mAP50-95)가 이것을 보여 준다.

   | 입력 크기 | 160 | 224 | 320 | 416 | 512 | 640 |
   |---|---|---|---|---|---|---|
   | mAP50-95 | 0.343 | 0.597 | 0.690 | 0.732 | 0.752 | 0.765 |

   해상도를 올릴수록 계속 좋아져서, 최종 모델(r3 이후)은 원본 크기 그대로 **1280**으로 학습했다. 기록은 `runs/sweep*/`에 있다.
2. **실전에서도 원본 해상도로 본다.** 처음 찾을 때는 전체 프레임(1280×960)을 쓰고, 한 번 찾은 뒤에는 타겟 주변 **384×384 영역을 원본 해상도 그대로 잘라서** 다시 검출한다(ROI 추적). 줄이지 않고 잘라 내므로 작은 타겟이 더 작아지지 않고, 속도도 약 9배 빠르다.
3. **평가 지표를 작은 물체에 맞췄다.** 박스가 작아서 1~2 px만 어긋나도 조준이 틀어진다. 그래서 mAP 대신 **박스 중심 오차(px)** 와 **높이 오차(px)** 를 주 지표로 본다(`tools/eval_cross_color.py`). 높이는 거리를 추정하는 데 쓰인다.

### 확인해 둘 한계

- 모델 구조는 기본 YOLO11n이다. 작은 물체 전용 구조(P2 검출 헤드)나 사진을 잘게 나눠 검출하는 방식(SAHI 타일링)은 **쓰지 않았다.** 지금 데이터에서는 성능이 충분해서 필요 없었다.
- 가장 작은 검출 격자(stride 8) 기준으로 중앙값 박스는 가로 약 2칸 × 세로 약 4칸에 걸친다. 검출은 되지만 여유가 크지 않은 크기다. **2.4 m보다 멀어지거나 해상도를 낮추면 성능이 급격히 떨어질 수 있다.**
- 학습 증강의 `scale=0.5`는 사진을 최대 절반까지 줄인다. 그래서 학습 중에는 9×16 px 정도의 더 작은 타겟도 보게 된다. 이것은 의도된 동작이다.
- "작음" 구간(세로 29 px 미만)의 새 test 사진은 마젠타 27장뿐이다. **먼 거리의 초록 타겟은 평가 근거가 부족하다**([COMPARE_r4.md](pi_bundle_r4/COMPARE_r4.md)).

---

## 2. 폴더 구성과 git에 없는 것

git에는 **코드, 문서, 학습 기록(숫자)** 만 있다. 사진, 라벨, 가중치는 용량이 커서 git에 넣지 않고, 학습 서버의 원본 폴더와 압축본(`yolo_target_full.tar.zst`)으로 따로 보관한다.

| 경로 | 내용 | git |
|---|---|---|
| `tools/` | 학습용 파이썬 도구(데이터셋 만들기, 라벨, 평가, 변환) | ✅ |
| `scripts/` | 학습 실행 셸 스크립트 | ✅ |
| `runs/<이름>/` | 학습마다의 설정(`args.yaml`)과 epoch별 결과(`results.csv`), 평가 csv | ✅ (가중치 `weights/*.pt`는 ❌) |
| `weights/ncnn*/`, `ncnn_r3_coco/`, `pi_bundle_r4/ncnn_r4/` | 파이용 NCNN 모델. git에는 `metadata.yaml`, `model_ncnn.py`, 체크섬만 있다 | 일부 (`.bin`, `.param` ❌) |
| `pi_bundle_r4/` | r4 비교표(`COMPARE_r4.md`)와 세부 표 | ✅ |
| `docs/` | GB10 체크리스트, 폴더 지도, 이전 데이터셋 목록 | ✅ |
| `README_FIRST.md`, `RESUME.md`, `TRAINING_DESIGN.md` | 서버 이전 안내, 인계 브리프, 초기 설계 | ✅ |
| `data/`, `data2/`, `finetune_data(10.08)/` | 원본 사진 | ❌ |
| `data2_labels/`, `friend_pseudo/`, `finetune_labels/` | 라벨 | ❌ |
| `handover/dataset/`, `handover/dataset_r4/` | 학습에 넣는 최종 데이터셋 | ❌ |
| `logs/`, `.cache/`, `venv/` | 로그, 캐시, 가상환경 | ❌ |

> 학습을 다시 하려면 git만으로는 안 되고, 데이터가 들어 있는 원본 폴더(압축본)가 필요하다.

---

## 3. 학습 환경 준비

GB10(aarch64, CUDA 13) 기준이다. 공용 계정이면 **반드시 개인 폴더 안에서만** 작업한다.

```bash
cd ~/<내 폴더>
sha256sum -c yolo_target_full.tar.zst.sha256      # 압축본이 손상되지 않았는지
tar --zstd -xf yolo_target_full.tar.zst           # → YOLO/ 폴더가 생긴다
cd YOLO

python3 -m venv venv
source scripts/env.sh                             # venv를 켜고, 캐시·설정을 이 폴더 안에 둔다
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # torch를 반드시 먼저
pip install ultralytics pandas                    # pandas는 따로 설치해야 한다
pip install ncnn pnnx                             # NCNN 변환용 (안 되면 x86 PC에서 변환)
bash scripts/setup_gb10.sh                        # 파일 무결성 + 데이터셋 장수 + GPU 동작 확인
```

- **명령을 실행하기 전에는 항상 `source scripts/env.sh`부터** 한다. 이것을 빼먹으면 venv가 꺼져 있거나 공용 `~/.config`에 설정 파일이 생긴다.
- 막히면 [docs/GB10_CHECKLIST.md](docs/GB10_CHECKLIST.md)를 본다.

---

## 4. 학습 순서 (처음부터 끝까지)

아래는 r4를 만든 실제 순서다. 새 사진을 찍어 다시 학습할 때도 같은 순서를 따르면 된다(이름만 `r5`처럼 바꾼다).

### 4-1. 사진 찍기 — 파이에서

파이에서 [PI/05_sweep_capture.py](../PI/05_sweep_capture.py)(서보 자동 스윕)나 [PI/05_manual_capture.py](../PI/05_manual_capture.py)(키보드 수동 조준)로 찍는다.

- 해상도는 **1280×960, 회전 없음**이다. 학습과 실전이 같은 조건이어야 한다.
- 결과 폴더에 `images/`, `labels.csv`(사진마다 라벨·거리 메모), `sessions.jsonl`이 생긴다. **세 가지를 모두 학습 서버로 옮긴다.**
- 메모(note)에는 거리와 배치를 정해진 형식으로 적는다(예: `both_1.5m`). 도구가 메모에서 거리를 읽는다.

### 4-2. train / val / test 나누기

```bash
source scripts/env.sh
python tools/split_finetune.py --src "finetune_data(10.08)" --dry-run    # 먼저 결과만 본다
python tools/split_finetune.py --src "finetune_data(10.08)"
```

- 세션마다 **촬영 시간순**으로 마지막 10%를 test, 그 앞 10%를 val로 쓴다. 비슷한 연속 사진이 train과 test에 섞여 점수가 부풀려지는 것을 막기 위해서다.
- test 사진은 `<폴더>_test/`로 따로 옮겨진다. **test는 마지막에 한 번만 평가한다.**

### 4-3. 라벨 만들기 — 기존 모델로 자동으로 그리고, 사람이 확인

박스를 사람이 수천 장 그리는 대신, 이미 학습된 모델로 먼저 그리고(사전 라벨) 사람은 확인만 한다.

```bash
# (1) 기존 모델로 새 사진 전체를 검출해서 집계
python tools/det_census.py --model runs/r3_coco/weights/best.pt --tag r3_coco
# (2) 장면마다 색별로 가장 확실한 박스 1개를 타겟으로 보고, 나머지(여분) 박스를 분류·시각화
python tools/extra_box_sheet.py --tag r3_coco
# (3) 사전 라벨 생성 (YOLO txt + LabelImg용 XML) + 검수 목록
python tools/finetune_prelabel.py --census runs/finetune_census/r3_coco --out finetune_labels
# (4) 박스 3개 장면에서 초록 1 + 마젠타 1 외의 여분 박스 자동 삭제 (원본은 orig/에 보존)
python tools/finetune_autoclean.py --labels finetune_labels --dry-run
python tools/finetune_autoclean.py --labels finetune_labels
```

- (2)의 `sheet_*.jpg`를 열어 보면 여분 박스가 무엇인지(붉은 물체, 도어 스토퍼 등) 눈으로 확인할 수 있다.
- 색 판정 규칙: 초록 a\* < −15, 마젠타 a\* > +30, 그 밖은 "어느 쪽도 아님"이다.
- **(5) 사람 검수**: [LabelImg](https://github.com/HumanSignal/labelImg)로 이미지 폴더와 `xml/` 폴더를 열고 `review_list.csv`에 있는 사진을 확인·수정한다. 수정한 XML은 다음 명령으로 txt 라벨에 반영한다.

  ```bash
  python tools/xml2yolo.py --xml <세션>/xml --labels <세션>/labels --only <검수 목록 csv> --dry-run
  python tools/xml2yolo.py --xml <세션>/xml --labels <세션>/labels --only <검수 목록 csv>
  ```

- 오검출된 박스(붉은 물체, 도어 스토퍼)를 지운 사진은 **hard negative**가 된다. 모델이 "이건 타겟이 아니다"를 배우는 데 쓰인다. r4에는 64장이 들어갔다.

### 4-4. 학습 데이터셋 만들기

```bash
python tools/build_dataset_r4.py --out handover/dataset_r4
```

- 기존 데이터셋(`handover/dataset`, train 1,423 / val 228)에 새 사진(train 1,068 / val 106)을 더해 **train 2,491 / val 334**를 만든다.
- 이미지는 복사하지 않고 링크로 연결한다(용량 절약). 라벨은 복사한다.
- 결과 폴더의 `data.yaml`이 학습에 넣는 설정 파일이다.

> 처음부터 데이터셋을 다시 만들 때는 `tools/package_dataset.py`를 쓴다(b1·b2·아군 원본 → `handover/dataset`).

### 4-5. 학습하기

```bash
EPOCHS=2 NAME=r4_probe bash scripts/train_r4.sh     # ① 먼저 2 epoch만 돌려 메모리와 epoch당 시간을 본다
bash scripts/train_r4.sh                            # ② 본 학습 (최대 100 epoch)
```

- ①에서 메모리가 부족하면 `BATCH=16 bash scripts/train_r4.sh`처럼 배치를 줄인다. `batch=-1`(자동)은 GB10의 통합 메모리에서 과하게 잡힐 수 있어 쓰지 않는다.
- 학습은 오래 걸리므로 터미널이 끊겨도 계속 돌게 하려면 다음처럼 실행한다.

  ```bash
  setsid nohup bash scripts/train_r4.sh > logs/r4.out 2>&1 &
  tail -f logs/r4.log          # 진행 상황 보기
  ```

- 30 epoch 동안 val 성능이 나아지지 않으면 자동으로 멈춘다(`patience=30`). r4는 74 epoch째가 가장 좋았다.
- 결과는 `runs/r4/`에 생긴다. `weights/best.pt`가 가장 좋은 모델이다.

**결과 읽는 법** — `runs/r4/results.csv`의 마지막 열들을 본다.

| 열 | 뜻 | r4 최고값 |
|---|---|---|
| `metrics/precision(B)` | 검출한 것 중 맞은 비율 | 0.9999 |
| `metrics/recall(B)` | 실제 타겟 중 찾은 비율 | 1.0 |
| `metrics/mAP50(B)` | 박스가 절반 이상 겹치면 정답으로 본 점수 | 0.995 |
| `metrics/mAP50-95(B)` | 겹침 기준을 엄격하게 올려 가며 잰 점수(위치 정확도) | **0.904** |

mAP50은 거의 항상 0.99라 비교에 쓸모가 없다. **모델끼리는 mAP50-95와 4-7의 평가표로 비교한다.**

### 4-6. 파이용 NCNN 모델로 변환

파이는 PyTorch 모델(`.pt`)을 직접 돌리기엔 느리므로, 가벼운 추론 엔진인 **NCNN** 형식으로 바꾼다.

```bash
# ⚠️ 반드시 PI 쪽 export_ncnn.py를 쓴다 (아래 설명)
python ../PI/pipeline/tools/export_ncnn.py --model runs/r4/weights/best.pt --out weights/ncnn_r4 --name r4 \
       --check-image handover/dataset_r4/val/images/<아무 사진>.jpg
```

- 입력 크기별로 4개가 만들어진다. NCNN은 입력 크기가 변환할 때 고정되기 때문이다.

  | 폴더 | 용도 |
  |---|---|
  | `960x1280_ncnn_model` | **전체 프레임에서 처음 찾기** |
  | `384x384_ncnn_model` | **찾은 뒤 주변만 추적(ROI)** |
  | `736x960_ncnn_model`, `480x640_ncnn_model` | 속도 비교용 |

- `--check-image`를 주면 `.pt`와 NCNN의 검출 박스를 나란히 출력한다. 좌표가 1 px 안쪽으로 같으면 변환이 정상이다.
- **⚠️ `YOLO/tools/export_ncnn.py`는 쓰지 않는다.** 이 파일은 폴더를 예전 이름(`ncnn_960x1280`)으로 만든다. ultralytics는 폴더 이름이 `_ncnn_model`로 끝나야 NCNN 모델로 인식하므로, 파이 설정에서 그대로 쓸 수 없다. 이름을 고친 것은 PI 쪽 버전([PI/pipeline/tools/export_ncnn.py](../PI/pipeline/tools/export_ncnn.py))이고, `weights/ncnn_r4`도 그 버전으로 만들었다.
- `pnnx`가 설치되지 않는 장비라면 x86 PC에서 변환한다. 결과 파일(`.param`, `.bin`)은 어느 장비에서나 그대로 쓸 수 있다.

### 4-7. 평가 — 기존 모델과 비교

```bash
python tools/eval_r4.py --set newval  --models r3_coco=ncnn_r3_coco/960x1280_ncnn_model r4=weights/ncnn_r4/960x1280_ncnn_model
python tools/eval_r4.py --set oldtest --models <위와 같음>
python tools/eval_r4.py --set newtest --models <위와 같음> --conf-op 0.25 --final     # 마지막에 단 한 번
```

- `--models`에는 `이름=960x1280 NCNN 폴더`를 비교할 모델 수만큼 적는다. 폴더 이름은 `_ncnn_model`로 끝나야 한다. r2(`weights/ncnn/ncnn_960x1280`)처럼 예전 이름의 폴더는 `960x1280_ncnn_model`로 이름을 바꾼 복사본을 넣는다.

- **newval**: 새 val로 운용 임계값(conf)을 정한다. r4는 0.10~0.50 어디서나 결과가 같아서 0.25로 정했다.
- **oldtest**: 사람이 직접 그린 라벨 98장으로 박스 위치 정확도(중심·높이 오차 px)를 잰다.
- **newtest**: 4-2에서 떼어 둔 test로 **딱 한 번만** 평가한다. 한 번 돌리면 잠금 파일이 생겨 다시 돌릴 수 없다. test 점수를 보고 설정을 고치면 test가 test 역할을 못 하기 때문이다.
- 판정 기준: 각 장면에서 "초록 1개 + 마젠타 1개"를 모두 찾고 여분 박스가 없으면 정답 프레임이다.
- 결과는 [pi_bundle_r4/COMPARE_r4.md](pi_bundle_r4/COMPARE_r4.md)처럼 표로 정리해 둔다.

---

## 5. 학습 설정값 설명

[scripts/train_r4.sh](scripts/train_r4.sh)의 값이다. 왜 이 값인지 함께 적었다.

| 설정 | 값 | 뜻과 이유 |
|---|---|---|
| `model` | `weights/yolo11n.pt` | COCO로 미리 학습된 가장 작은 YOLO11(nano). 파이4에서 돌려야 해서 가장 가벼운 모델을 쓴다 |
| `imgsz` | **1280** | 원본 해상도. 타겟이 작아서 줄이면 안 된다(§1) |
| `epochs` / `patience` | 100 / 30 | 최대 100번 반복하되, 30번 연속 나아지지 않으면 멈춘다 |
| `batch` | 32 | 한 번에 보는 사진 수. 메모리가 부족하면 줄인다 |
| `cache` | `ram` | 사진을 메모리에 올려 학습을 빠르게 한다(약 6 GB 필요) |
| `hsv_h` | **0.3** | 색상을 크게(±108°) 무작위로 바꾼다. 검출기가 **색에 의존하지 않게** 하려는 것이다. 색지를 바꿔도 다시 학습할 필요가 없도록 |
| `hsv_s`, `hsv_v` | 0.7, 0.4 | 채도·밝기를 무작위로 바꾼다. 조명 변화에 강하게 |
| `scale` | 0.5 | 사진을 0.5~1.5배로 무작위 확대·축소한다. 거리 변화에 강하게 |
| `translate`, `fliplr` | 0.1, 0.5 | 위치 이동, 좌우 반전 |
| `mosaic` / `close_mosaic` | 1.0 / 15 | 4장을 이어 붙여 학습하다가 마지막 15 epoch는 끈다 |
| `seed` | 0 | 같은 결과를 다시 얻을 수 있게 고정 |

> **왜 `hsv_h=0.3`인가**: ultralytics의 `hsv_h`는 색상을 더하는 방식이라 0.015면 ±5°, 0.3이면 ±108°만큼 돌린다. 아군·적군 색의 색상 차이가 약 85°라서, 0.015로는 "다른 색 타겟"을 본 적이 없는 모델이 된다. 이를 확인하려고 한 것이 3-arm 실험([scripts/train_3arm.sh](scripts/train_3arm.sh))이다.

---

## 6. 실제 상황에 적용하기 (파이)

### 6-1. 실전에서 모델이 어떻게 쓰이나

[PI/pipeline/tools/target_pipeline.py](../PI/pipeline/tools/target_pipeline.py)가 매 프레임 아래 순서로 동작한다.

1. **검출(처음)**: 전체 프레임 1280×960을 `960x1280_ncnn_model`로 검출한다. 파이4 실측 **약 1.2초**(r3_coco 기준, §7).
2. **검출(추적)**: 타겟을 찾았으면 다음부터는 그 주변 384×384만 잘라 `384x384_ncnn_model`로 검출한다. **약 0.14초**. 잘라 낸 좌표는 원본 좌표로 되돌린다.
3. **색 판정**: 박스 안에서 채도가 높은 픽셀만 골라 색(Lab)을 재고, 기준색(`color_refs.json`)과 비교해 적군 / 아군 / 판정 불가(unknown)로 나눈다.
4. **조준**: 박스 중심 x로 좌우(yaw), 박스 높이로 거리를 추정하고 조준점 보정표(`aim_table.json`)로 상하(pitch)를 정한다.
5. **결정**: 적군이 있으면 `engage`(가장 가까운 = 박스가 가장 큰 적군), 판정 불가인 타겟이 하나라도 있으면 `hold`(안전을 위해 보류), 없거나 아군뿐이면 `no_target`.
   - 화면 위아래 끝(y 352~672 밖)의 박스는 신뢰도를 절반으로 깎는다. 타겟이 그 높이에 나타날 수 없기 때문이다.

### 6-2. 새 모델을 파이로 옮기기

```bash
# 파이에서 (주소·계정·폴더는 실제 값으로)
scp -r <계정>@<GB10주소>:~/<폴더>/YOLO/weights/ncnn_r4 ~/src/pipeline/weights/
ls ~/src/pipeline/weights/ncnn_r4
# → 960x1280_ncnn_model 등 4개 폴더. 각 폴더에 model.ncnn.param / model.ncnn.bin / metadata.yaml 이 있어야 한다
```

옮긴 뒤 `weights/ncnn_r4/SHA256SUMS`로 파일이 깨지지 않았는지 확인한다.

```bash
cd ~/src/pipeline/weights/ncnn_r4 && sha256sum -c --quiet SHA256SUMS && echo OK
```

### 6-3. 설정 파일에서 모델 바꾸기

`~/src/pipeline/tools/pipeline_config.json`의 `detector` 두 줄을 바꾼다.

```json
"model": "../weights/ncnn_r4/960x1280_ncnn_model",
"roi_model": "../weights/ncnn_r4/384x384_ncnn_model",
```

- 경로는 이 json 파일 위치 기준이다.
- **폴더 이름의 `_ncnn_model`은 바꾸면 안 된다.**
- `conf`는 0.25 그대로 둔다(§4-7에서 정한 값).
- 예전 모델로 되돌릴 수 있게, 원래 경로는 `_model_r3_coco`, `_roi_model_r3_coco`처럼 `_`로 시작하는 키로 남겨 둔다. `_`로 시작하는 키는 무시된다.

> 현재 파이와 [PI/pipeline/tools/pipeline_config.json](../PI/pipeline/tools/pipeline_config.json)은 **r4**를 쓴다. r3_coco로 되돌리려면 `model`, `roi_model` 값을 `_model_r3_coco`, `_roi_model_r3_coco`의 값으로 바꾼다.

### 6-4. 동작 확인 → 보정 → 실전

모든 명령은 파이에서 `~/src`(=`PI/`) 기준이다. 카메라는 한 번에 한 프로그램만 쓸 수 있다.

```bash
# ① 사진 1장으로 검출 확인 (카메라 없이)
cd ~/src/pipeline && venv/bin/python tools/target_pipeline.py --config tools/pipeline_config.json --image samples/enemy_roomA.jpg

# ② 카메라로 검출 + 아군/적군 판정이 맞는지 화면으로 확인 (서보·발사 없음)
cd ~/src && pipeline/venv/bin/python 08_detect_check.py

# ③ 거리별로 실제로 쏴 보며 조준점 보정표 만들기
cd ~/src && pipeline/venv/bin/python 07_aim_calib.py --distance 1.8

# ④ 대회: 모델을 미리 올려 두고 SPACE/Enter로 시작
cd ~/src/pipeline && venv/bin/python tools/run_match.py
```

| 단계 | 무엇을 보나 |
|---|---|
| ① | JSON에 박스 좌표와 conf가 나오는지. stdout에는 JSON만 나온다 |
| ② | 화면에 적군은 빨강, 아군은 초록, 판정 불가는 노랑으로 표시된다. `g`로 그 순간을 `08_checks/`에 저장할 수 있다 |
| ③ | `e`로 실사격 위치를 기록하면, 종료할 때 `calib/aim_table.json`(박스 높이 → 조준점)이 만들어진다. `p`는 적군만 자동 조준·발사 |
| ④ | 매 프레임 결과가 jsonl 로그로 남는다. 카메라 없이 시험하려면 `--fake samples/enemy_roomA.jpg --start-now --max-frames 5` |

- 색 기준값(`color_refs.json`)은 **실제 카메라 설정에서 잰 값**이다. 노출이나 화이트밸런스를 바꾸면 [PI/pipeline/tools/build_color_refs.py](../PI/pipeline/tools/build_color_refs.py)로 다시 만든다.
- 카메라 위치가 바뀌거나 타겟 외형이 바뀌면 ③의 조준점 보정표도 다시 만든다.
- 파이 설치 방법은 [PI/pipeline/PI_SETUP.md](../PI/pipeline/PI_SETUP.md)를 본다.

---

## 7. 모델 이력과 성능

| 모델 | 시작점 | 학습 데이터 | 입력 | 상태 |
|---|---|---|---|---|
| r2 | — | b1 138장 + b2 494장 | 640 | 사전 라벨을 만든 모델. NCNN: `weights/ncnn/` |
| r3 | r2 | 전체 1,423장 | 1280 | 비교용 |
| r3_coco | COCO `yolo11n.pt` | 전체 1,423장 | 1280 | 이전 배포 모델. NCNN: `ncnn_r3_coco/` |
| **r4** | COCO `yolo11n.pt` | 1,423장 + 새 촬영분 1,068장(색칠한 타겟, 초록·마젠타 동시 배치) | 1280 | **현재 파이 배포 모델.** NCNN: `weights/ncnn_r4/` |

r2 → r3_coco → r4 비교(NCNN 960×1280, conf 0.25, 자세한 표는 [COMPARE_r4.md](pi_bundle_r4/COMPARE_r4.md)):

| | r2 | r3_coco | **r4** |
|---|---|---|---|
| 새 val 정답 프레임 (106장) | 36.8% | 97.2% | **100%** |
| 새 test 정답 프레임 (136장, 1회 평가) | 51.5% | 96.3% | **99.3%** |
| 기존 test recall (사람 라벨 98장) | 0.990 | 0.990 | **1.000** |
| 기존 test 오검출 / 프레임 | 0.061 | 0 | 0 |
| 박스 중심 오차 중앙값 | 0.75 px | 0.56 px | 0.71 px |

- r4의 새 test 실패 1장은 검출기 문제가 아니었다. 타겟이 화면 왼쪽 끝에 있어 색이 흐려져 **색 판정**이 "어느 쪽도 아님"이 된 경우다.
- **라즈베리파이4 4GB 실측 검출 시간** (2026-10-04, r3_coco NCNN FP32, `pi_bench.py --runs 50`, 성능 제한 없음):

  | 입력 크기 | p50 | p95 |
  |---|---|---|
  | 960×1280 (전체 프레임) | **1,224 ms** | 1,286 ms |
  | 736×960 | 694 ms | 724 ms |
  | 480×640 | 300 ms | 349 ms |
  | 384×384 (ROI 추적) | **142 ms** | 157 ms |

  전체 1회 + ROI 5회 = 1.93초, + ROI 10회 = 2.64초(추론만). r4는 같은 YOLO11n 구조·입력 크기라 비슷할 것으로 예상하지만 아직 재지 않았다. 자세한 내용은 [PI README §6-2](../PI/README.md#6-2-검출-시간-라즈베리파이4-4gb-실측).

---

## 8. 주의할 점

1. **test는 낙관적일 수 있다.** 새 test는 train과 같은 날(10/08), 같은 장소에서 찍은 사진이다. 다른 장소에서도 이만큼 나오는지는 아직 모른다.
2. **오검출은 장소에 따라 다르다.** 지금까지의 오검출은 문 아래 도어 스토퍼와 화면 오른쪽 끝의 붉은 물체였다. 대회장에서 새로 찍어 hard negative로 넣는 것이 가장 확실하다.
3. **먼 거리 데이터가 적다.** 세로 29 px 미만(먼 거리) 구간의 test는 마젠타 27장뿐이고 초록은 없다.
4. **화면 왼쪽 끝은 색이 약하다.** 렌즈 가장자리라 고채도 픽셀이 중앙보다 약 35% 적다. 검출은 되지만 색 판정이 보류될 수 있다.
5. **val 일부는 사람이 아니라 모델이 만든 라벨(의사 라벨)이다.** val 점수는 모델끼리 상대 비교용으로만 본다. 사람이 그린 라벨로 잰 것은 기존 test 98장뿐이다.
6. **`tools/`의 원본은 파이 쪽이다.** 공통 도구(`colorlib.py`, `target_pipeline.py` 등)는 [PI/pipeline/tools/](../PI/pipeline/tools/)가 최신이다. 학습 서버의 `tools/`는 고치지 말고, 파이에서 고친 뒤 가져온다([PI_SETUP.md §9](../PI/pipeline/PI_SETUP.md)). `pipeline_config.json`은 장비마다 다르므로 덮어쓰지 않는다.
7. **`Yolo_Training.ipynb`는 초기 예시**다(Roboflow 데이터, 640 학습). 지금 학습 방식과 다르니 참고만 한다.

---

## 9. 더 자세한 문서

| 문서 | 내용 |
|---|---|
| [README_FIRST.md](README_FIRST.md) | 압축본을 새 서버에 풀고 확인하는 순서 |
| [RESUME.md](RESUME.md) | 과제, 데이터, 측정값, 결정 사항 전체 (10/3 기준) |
| [pi_bundle_r4/COMPARE_r4.md](pi_bundle_r4/COMPARE_r4.md) | r4 비교표와 해석 (10/11) |
| [docs/HANDOVER_GB10.md](docs/HANDOVER_GB10.md) | 폴더 지도, 데이터셋 구성, 재현 명령 |
| [docs/GB10_CHECKLIST.md](docs/GB10_CHECKLIST.md) | GB10 환경 체크리스트 |
| [TRAINING_DESIGN.md](TRAINING_DESIGN.md) | 초기 학습 설계 (RESUME.md와 다르면 RESUME.md가 맞다) |
| [PI/pipeline/PI_SETUP.md](../PI/pipeline/PI_SETUP.md) | 파이 설치와 운영 |
