# colorlab — 아군/적군 표지판 색 선정 도구

라즈베리파이 + 파이카메라로 **현재 환경(배경)에 어떤 색이 많은지**, **후보 색 표지판이 실제로
어떤 값으로 찍히는지**를 측정해서 표지판에 쓸 두 가지 색을 고르기 위한 도구입니다.

- 확인된 환경: Raspberry Pi 4 Model B / Raspberry Pi OS Bookworm 64-bit / Camera Module v1 (ov5647)
- 화면: VNC(wayvnc)로 데스크톱 접속 → OpenCV 창. `04_scene_capture.py`는 화면 없이 SSH만으로 동작

---

## 1. 설치

picamera2는 **apt로 설치된 시스템 패키지**를 씁니다. `pip install picamera2`는 하지 마세요.

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-pil
```

선택: 화면 오버레이를 한글로 보고 싶으면 (없으면 영문으로 자동 표시)

```bash
sudo apt install -y fonts-nanum
```

설치 확인:

```bash
cd ~/src
python3 02_check_colorspace.py        # 색 변환 수식 검사 (카메라 불필요) -> ALL PASS
```

---

## 2. 폴더 구조

```
~/src/
├─ colorlab/                  # 공용 모듈 (01~04에서 사용)
│   ├─ camera.py              #   카메라 제어, AWB/AE(/AF) 고정, config 저장·불러오기
│   ├─ color.py               #   색공간 변환(RGB/HSV/Lab/Y), ROI 측정, 포화/암부 판정
│   ├─ overlay.py             #   화면 텍스트
│   └─ store.py               #   측정값 CSV + 스냅샷 저장
├─ 01_test_camera.py          # [처음 1회] 카메라 동작 + 채널 순서 확인
├─ 01_test_camera/            #   └ 채널 확인용 이미지 2장
├─ 02_check_colorspace.py     # 색 변환 수식 검사 (+ --camera 로 중앙 영역 실측) - 생성 폴더 없음
├─ 03_preview.py              # 실시간 화면 + 마우스 영역 색 측정 + 설정 고정
├─ 03_config/                 #   └ camera_config.json (고정된 카메라 설정, 04~05도 읽음)
├─ 03_logs/                   #   └ measurements.csv (측정값 누적)
├─ 03_captures/               #   └ 측정 스냅샷
├─ 04_scene_capture.py        # 배경 전체 색 분포 수집 (화면 없이)
├─ 04_scene_captures/         #   └ 결과 (촬영 1회 = 폴더 1개)
├─ 05_sweep_capture.py        # 서보 자동 스윕 촬영 (학습 데이터, 키보드 1/2/3 라벨)
├─ 05_dataset/                #   └ enemy/ friend/ both/ 라벨별 폴더
├─ 99_fake_camera.py          # 카메라 없이 위 스크립트 테스트
└─ Lecture/                   # 교수님 교안과 코드 (참고용)
    ├─ 06_data_collect.py     # 키보드 서보 조작 + 연속 촬영 (학습 데이터)
    ├─ 06_data_yolo/          #   └ images/ + data_angles.csv
    ├─ 07_inference.py        # YOLO 탐지 + CNN 요 예측 → 조준·발사 - 생성 폴더 없음
    ├─ preprocessing_data_make_model(colab_pytorch).ipynb   # Colab 학습 노트북
    ├─ yolo_target.pt
    └─ best_yaw_cnn.pth
```

---

## 3. 사용 순서 (권장)

> **카메라는 한 번에 한 프로그램만 쓸 수 있습니다.** `03_preview.py`, `04_scene_capture.py`,
> `05_sweep_capture.py`, `06_data_collect.py`, `07_inference.py`, `rpicam-hello`는 동시에 실행하지 마세요.
> 사용 중인지 확인: `fuser -v /dev/video0`

### 3-1. 채널 순서 확인 (처음 1회)

빨간 물체를 화면 가운데에 꽉 차게 대고:

```bash
python3 01_test_camera.py --truth red
```

`가설 A 확정`이 나오면 정상입니다. `가설 B`가 나오면 이후 모든 명령에 `--no-swap`을 붙이세요.

### 3-2. 실제 환경에서 카메라 설정 고정

카메라를 **실전과 같은 위치·조명**에 설치한 뒤, VNC 데스크톱의 터미널에서:

```bash
python3 03_preview.py
```

`f` 키를 누르면 자동 모드로 되돌려 노출/화이트밸런스가 **안정될 때까지 기다린 뒤**(최소 3초, 최대 10초)
고정하고 `03_config/camera_config.json`에 저장합니다. 10초 안에 안정되지 않으면 경고가 나오니
조명이 안정된 뒤 `f`를 다시 누르세요.

이 config에는 고정값과 함께 **해상도·센서 모드·화각(ScalerCrop)**이 저장됩니다. 나중에 다른
해상도로 불러오면 경고가 나옵니다. 데이터 수집과 실전(`06_data_collect.py`, `07_inference.py`)도
**같은 640x480**과 이 config를 써야 측정값이 의미가 있습니다.

### 3-3. 배경 색 분포 수집 → 비어 있는 색 찾기

→ [5. 04_scene_capture.py](#5-scene_capturepy--배경-색-분포-수집) 참고.
결과의 `summary.txt`에 있는 **"비어 있는 Hue 구간"**이 표지판 색 후보입니다.

### 3-4. 후보 색 표지판 측정

실제 표지판 재료(색지·테이프 등)를 표지판이 놓일 위치에 두고:

```bash
python3 03_preview.py --load
```

영역을 드래그 → `SPACE`로 측정 → `s`로 이름을 붙여 저장합니다.
이름 예: `후보_파랑_근거리`, `후보_파랑_좌측`, `배경_벽`.
같은 색을 위치를 바꿔 3~4번씩 측정해 두면 변동 폭을 볼 수 있습니다.

---

## 4. 03_preview.py — 실시간 화면 + 영역 측정

```bash
python3 03_preview.py                 # 자동 모드로 시작
python3 03_preview.py --load          # 저장된 config를 불러와 고정한 채로 시작
python3 03_preview.py --view-fps 15   # VNC가 버벅일 때 (창 갱신만 줄임, 측정은 그대로)
```

VNC 화면의 `ColorLab` 창을 클릭해 포커스를 준 뒤 키를 누릅니다.

| 키 | 동작 |
|---|---|
| 마우스 드래그 | 측정할 영역(ROI) 선택 |
| `SPACE` / `c` | 영역 측정 (최근 N프레임 평균, 기본 10) |
| `s` | 방금 측정값을 이름 붙여 저장 — **이름은 터미널에 입력** |
| `f` | 자동으로 되돌려 안정화 → AWB/AE 고정 + config 저장 |
| `a` | 자동 모드로 되돌리기 |
| `l` | config 불러와 다시 고정 |
| `[` / `]` | 평균낼 프레임 수 줄이기 / 늘리기 |
| `r` | 영역을 화면 중앙으로 초기화 |
| `h` | 도움말 표시 켜기/끄기 |
| `p` | 현재 카메라 메타데이터를 터미널에 출력 |
| `q` / `ESC` | 종료 |

> **`s`로 이름을 입력하는 동안에는 OpenCV 창이 멈춘 것처럼 보입니다. 정상입니다.**
> 터미널에서 이름을 입력하고 Enter를 누르면 다시 움직입니다. (그냥 Enter = 저장 취소)

**측정 품질 경고** — 측정할 때 다음 경우 터미널과 화면에 경고가 나오고, CSV에도 기록됩니다.

- 포화: 영역 픽셀 중 어느 채널이든 250 이상인 비율이 1% 초과 → 색이 잘려 있음
- 암부: 모든 채널이 5 이하인 비율이 1% 초과 → 너무 어두움
- 자동 모드(AWB/AE 켜짐)에서 측정 → 실전 조건과 다를 수 있음

**저장 결과** — `03_logs/measurements.csv`에 누적되고 스냅샷은 `03_captures/`에 저장됩니다.
측정값은 **측정한 순간의** 고정 상태로 기록됩니다(측정 후 `a`를 눌러도 기록은 바뀌지 않음).
주요 컬럼: `R,G,B`(0~255), `H_deg`(0~360), `S_pct/V_pct`(0~100), `L`(0~100), `a,b`,
`Y`(밝기 0~255), `clip_ratio`, `dark_ratio`, `awb_ae_locked`, `exposure_us`, `analogue_gain`.

---

## 5. 04_scene_capture.py — 배경 색 분포 수집

고정 설치된 카메라로 **화면 전체** 배경의 색 분포를 CSV와 PNG로 저장합니다.
화면 없이 SSH에서 실행합니다. 결과 폴더 하나만 보내면 외부에서도 분석할 수 있게
`meta.json`에 촬영 조건과 모든 컬럼 정의가 들어 있습니다.

### 실행

```bash
# 처음 1회: PNG에 저장되는 색 순서 검증
python3 04_scene_capture.py --self-test

# 1회 캡처
python3 04_scene_capture.py --label 오전_맑음

# 10분 간격 12회 (조명 변화 기록) - grid-step 4면 20MB를 넘으므로 5로
python3 04_scene_capture.py --label 하루_맑음 --interval 600 --count 12 --grid-step 5
```

- **config가 반드시 필요합니다.** 없으면 오류로 종료합니다(→ 3-2에서 `f`로 먼저 만드세요).
  `--allow-auto`를 주면 자동 모드로 찍지만, 폴더 이름·라벨(`[AUTO]`)·meta·summary에 자동 모드로
  표시되고 실전 값과 비교할 수 없습니다.
- 캡처 1회 = 연속 15프레임의 **픽셀별 중앙값** 1장 (노이즈·지나가는 물체 제거).
- `pixels.csv` 예상 크기가 20MB를 넘으면 권장 `--grid-step`을 알려주고 **촬영하지 않고 종료**합니다.
  그래도 진행하려면 `--force`.
- **Ctrl+C 한 번**: 진행 중인 캡처까지 저장하고 종료 / **두 번**: 즉시 종료.
  결과는 캡처 단위로 기록되므로 중단해도 그때까지의 파일은 온전합니다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--label` | (필수) | 촬영 조건 라벨. 폴더명·CSV에 기록 |
| `--count` / `--interval` | 1 / 600 | 캡처 횟수 / 캡처 시작 간격(초) |
| `--frames` | 15 | 캡처당 합성할 연속 프레임 수 |
| `--grid-step` | 4 | `pixels.csv` 격자 간격(px) |
| `--sat-min` / `--val-min` | 0.25 / 0.15 | Hue 분석에 넣을 최소 채도·명도 (0~1) |
| `--config` | 03_config/camera_config.json | 불러올 카메라 설정 |

### 결과 폴더

```
04_scene_captures/<날짜시간>_<label>/
  ├─ meta.json                 # 촬영 조건, 고정값, 카메라/센서 정보, 각 파일·컬럼 정의
  ├─ capture_XX.png            # 중앙값 합성 원본 (무손실 PNG, 표준 RGB)
  ├─ capture_XX_clipmap.png    # 포화=빨강 / 암부=파랑 표시
  ├─ pixels.csv                # 격자 샘플 (블록 평균 RGB/HSV/Lab), 전체 캡처 누적
  ├─ hist_lab.csv              # 전체 픽셀 3D Lab 히스토그램, 전체 캡처 누적
  ├─ hist_hue.csv              # 유채색 픽셀 Hue 히스토그램 (캡처마다 36행)
  └─ summary.txt               # 사람이 읽는 요약 — 먼저 이것부터 보세요
```

값 범위: `R,G,B` 0~255 / `H` 0~360 / `S,V` **0~1** / `L` 0~100 / `a,b` 약 -128~127 (표준 CIELAB).
자세한 정의는 각 폴더의 `meta.json` → `"files"`를 보세요.

### 권장 절차

1. `python3 04_scene_capture.py --self-test` → `PASS` 확인
2. 실전 위치·조명에서 `03_preview.py`로 `f` 고정 (config 생성)
3. **표지판·사람·움직이는 물체가 없는 상태**로 배경만 촬영
4. 조명이 바뀌는 환경이면 시간대를 나눠 여러 라벨로 촬영 (예: `오전_맑음`, `오후_흐림`)
5. 각 폴더의 `summary.txt`에서
   - **포화 경고**가 있으면 → 노출을 낮춰 config를 다시 만들고 재촬영
   - **"비어 있는 Hue 구간"** 중 모든 캡처에서 공통으로 비어 있는 구간 → 표지판 색 후보
6. 후보 색 표지판을 만들어 `03_preview.py`로 측정 (3-4)

---

## 6. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `Camera __init__ sequence did not complete` / `Device or resource busy` | 다른 프로그램이 카메라 사용 중. `fuser -v /dev/video0`로 확인 후 종료 |
| 화면이 거의 검정 | 렌즈 보호 비닐/캡, 조명 확인 |
| VNC 화면이 끊김 | 네트워크 전송 문제(측정값엔 영향 없음). `--view-fps 15`, VNC Viewer 화질 Low, 가능하면 유선 LAN |
| 실행할 때마다 `QStandardPaths: wrong permissions` | Qt 경고일 뿐, 동작에 영향 없음 |
| `--load` 때 "저장 당시와 조건이 다릅니다" | config를 만든 해상도와 다르게 실행함. 같은 `--main-width/--main-height`로 실행하거나 `f`로 다시 고정 |
| 게인이 높다는 경고 (7.9 이상) | 조명이 어두워 노이즈가 큼. 가능하면 조명을 밝게 하고 config 재생성 |

참고: Camera Module v1(ov5647)은 오토포커스가 없어 초점 고정은 자동으로 건너뜁니다.
Camera Module 3처럼 AF가 있는 카메라에서는 `f`로 고정할 때 초점(LensPosition)도 함께 고정·저장됩니다.

---

## 7. 아직 구현하지 않은 것

- 후보 색 비교 리포트 (후보끼리 ΔE2000, 배경 대비 ΔE, 조건별 변동 폭, L\* 차이 → 미정)
- `03_preview.py` 안에서의 히스토그램 이미지 저장 (현재는 `04_scene_capture.py`가 CSV로 대신함)
