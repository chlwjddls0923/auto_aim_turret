# 파이 설치와 운영 안내 (`pi_bundle.tar.gz`)

라즈베리파이4에서 **검출 → 색 판정 → 조준값 계산**을 돌리기 위한 묶음이다. 학습 데이터와 PyTorch 가중치(`.pt`)는 들어 있지 않다. 파이는 NCNN 모델만 쓴다.
2026-10-04에 DGX(x86)에서 만들었다. **이 묶음을 실제 파이에서 설치·실행해 본 적은 없다.** x86에서 NCNN 추론으로 동작과 결과를 확인한 것까지가 검증 범위다. "미검증"이라고 적은 곳은 파이에서 처음 확인해야 한다.

프로젝트 전체 맥락(과제, 측정값, 결정, 다음 할 일)은 [RESUME.md](RESUME.md)에 있다.

## 1. 구성

풀면 `pipeline/` 폴더가 나온다. `~/src/`에 풀어 `~/src/pipeline/`로 쓰는 것을 기준으로 적는다.

| 경로 | 내용 |
|---|---|
| `tools/` | 실행 코드 전부(아래 표). `pipeline_config.json`은 **파이용(NCNN)** 으로 맞춰져 있다 |
| `weights/ncnn_r3_coco/` | **배포 모델** r3_coco의 NCNN FP32 4종: `960x1280_ncnn_model`(전체 프레임), `736x960_…`, `480x640_…`, `384x384_ncnn_model`(ROI 추적) |
| `weights/ncnn_r3/` | 비교용 r3의 NCNN 4종 |
| `wheels/aarch64/` | ncnn(cp39~cp313)·pnnx의 aarch64 휠. PyPI가 안 될 때의 보험 |
| `samples/` | 동작 확인용 프레임 3장(1280×960). 기대 출력은 §5 |
| `docs/` | 폴더 지도(HANDOVER_GB10), GB10 체크리스트, 초기 설계 문서(TRAINING_DESIGN — RESUME.md와 다르면 RESUME.md가 맞다) |
| `RESUME.md` | 프로젝트 인계 브리프. GB10용으로 쓴 문서라 `README_FIRST.md`, `runs/`, `handover/` 등 이 묶음에 없는 경로도 나온다 |
| `SHA256SUMS` | 전 파일 해시 |

`tools/`에서 파이가 실제로 쓰는 것:

| 파일 | 용도 | 추가로 필요한 패키지 |
|---|---|---|
| `target_pipeline.py` + `pipeline_config.json` | 실전 파이프라인: `Detector` → `ColorClassifier` → `Aimer` → `decide()` | ultralytics, ncnn |
| `colorlib.py` | 색 함수: Lab 변환, CIEDE2000, 고채도 픽셀 선택(`patch_stats`), 판정(`classify`), 분리도(`separation_report`) | numpy, opencv만 |
| `color_separation.py` | 촬영분에서 색 분리(안전비)를 재고 `color_refs.json`을 만든다 | + pandas |
| `pi_bench.py` | NCNN 4종의 지연 실측 | ultralytics, ncnn |
| `seed_label.py` | 촬영분에 사전 라벨 + 미검출 집계(새 색지에서 검출이 버티는지 확인) | + pandas |

나머지(`package_dataset.py`, `eval_cross_color.py`, `compare_models.py`, `make_gray.py`, `export_ncnn.py`, `xml2yolo.py`, `pseudo_bias_check.py`)는 학습 장비용이다. 원본을 한곳에 두려고 같이 넣었다.

## 2. 풀기와 확인

```bash
cd ~/src
sha256sum -c pi_bundle.tar.gz.sha256
tar -xzf pi_bundle.tar.gz            # → ~/src/pipeline/
cd pipeline && sha256sum -c --quiet SHA256SUMS && echo "전 파일 OK"
```

## 3. 파이 환경 확인

```bash
uname -m                       # aarch64 여야 한다
getconf LONG_BIT               # 64 여야 한다
dpkg --print-architecture      # arm64 여야 한다 (armhf 면 커널만 64비트이고 사용자 영역은 32비트)
grep -E "PRETTY_NAME|VERSION_CODENAME" /etc/os-release
python3 --version
```

셋 중 하나라도 32비트로 나오면 이 묶음의 휠과 PyTorch를 설치할 수 없다. 64비트 라즈베리파이 OS가 필요하다.
PyTorch 최신 휠은 Python 3.10 이상을 요구한다(Bookworm의 3.11이면 된다. Bullseye의 3.9면 pip이 옛 버전을 고른다 — 미검증).

## 4. 설치

**torch가 필요한가: 필요하다.** `tools/`는 추론을 `ultralytics.YOLO`로 하는데, ultralytics는 import할 때 torch를 불러온다. NCNN은 추론 엔진만 대신할 뿐이다. 그래서 **CPU용 torch가 설치돼 있어야 한다**(GPU·CUDA는 필요 없다). `pnnx`는 `.pt`를 NCNN으로 바꾸는 변환기라 **파이에서는 필요 없다**(export는 GB10이나 x86 PC에서 한다).
torch 없이 `ncnn` 파이썬 패키지만으로 추론하는 것도 원리상 가능하지만, 전처리와 후처리(NMS)를 직접 써야 하고 이 묶음에는 그 코드가 없다.

```bash
cd ~/src/pipeline
python3 -m venv --system-site-packages venv      # apt 로 설치된 picamera2 를 그대로 쓰기 위해 시스템 패키지를 보이게 한다
source venv/bin/activate
pip install --upgrade pip
pip install ncnn || pip install --no-index --find-links wheels/aarch64 ncnn      # 1) NCNN 런타임
pip install ultralytics pandas "numpy<2"                                          # 2) ultralytics (+ CPU torch 가 따라온다)
python -c "import numpy, cv2, torch, ncnn, ultralytics; print(numpy.__version__, cv2.__version__, torch.__version__, ultralytics.__version__)"
python -c "import picamera2; print('picamera2 OK')"                               # 카메라 쪽이 깨지지 않았는지
```

### 4-1. 실제 파이에서 설치한 순서 (2026-10-04, 검증됨)

위 명령 대신 실제로는 아래 순서로 설치했고, §5의 확인을 모두 통과했다. numpy는 **2.2.6**으로 맞췄다(기존 `~/venv`에서 picamera2 동작이 확인된 버전. 아래 "numpy<2" 권장은 따르지 않았다).

```bash
cd ~/src/pipeline
python3 -m venv --system-site-packages venv
venv/bin/pip install --upgrade pip
# 1) CPU torch (aarch64 CPU 휠)
venv/bin/pip install torch torchvision "numpy==2.2.6" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# 2) ultralytics (검증 버전 고정)
venv/bin/pip install "ultralytics==8.4.171" "numpy==2.2.6"
# 3) ncnn·pnnx: 묶음의 휠을 쓰되 의존성(portalocker)은 PyPI에서 받는다
venv/bin/pip install --find-links wheels/aarch64 "ncnn==1.0.20260526" "pnnx==20260526" "numpy==2.2.6"
# 4) pandas
venv/bin/pip install pandas "numpy==2.2.6"
# 5) picamera2용 simplejpeg (numpy 2 빌드) — 이 단계가 없으면 picamera2 import 가 깨진다
venv/bin/pip install "simplejpeg==1.9.0" "numpy==2.2.6"
venv/bin/python -c "import picamera2, numpy; print('picamera2 OK, numpy', numpy.__version__)"
```

- **portalocker**: ncnn 휠이 의존한다. `wheels/aarch64/`에 없어서 `--no-index`(오프라인) 설치는 `No matching distribution found for portalocker`로 실패한다. 오프라인으로 설치하려면 portalocker 휠도 함께 챙긴다.
- **simplejpeg**: picamera2가 import할 때 시스템(apt)의 simplejpeg를 불러오는데, 이것은 numpy 1.x용으로 빌드되어 있어 numpy 2.2.6에서 `ValueError: numpy.dtype size changed`로 실패한다. venv에 `simplejpeg==1.9.0`(numpy 2 빌드)을 설치하면 해결된다(기존 `~/venv`도 같은 방식).
- pnnx는 파이에서 쓰지 않지만 함께 설치했다.
- 설치된 조합: torch 2.14.1+cpu, torchvision 0.29.1+cpu, ultralytics 8.4.171, ncnn 1.0.20260526, opencv-python 5.0.0, numpy 2.2.6, pandas 3.0.6, simplejpeg 1.9.0.
- 설치 직후 venv 크기 약 0.8 GB(torch 포함).

설치할 때 알아 둘 것(전부 미검증, PyPI 조사에 근거):

- **`"numpy<2"`를 같은 명령에 넣는 이유**: 최신 `opencv-python`(5.x)은 numpy 2 이상을 요구한다. numpy가 2로 올라가면 apt로 설치된 picamera2 쪽(시스템 numpy 1.x에 맞춰 빌드됨)이 깨질 수 있다. 핀을 같이 주면 pip이 numpy 1.x와 맞는 opencv-python 4.11을 고른다. 설치 뒤 위의 `import picamera2`가 통과하는지 꼭 본다.
- **torch 휠 크기**: PyPI의 aarch64 torch 휠은 약 450 MB다(2.14.1 기준). 내려받기와 SD카드 여유 공간을 확인한다. 더 작은 휠이 필요하면 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`를 먼저 시도하고 그다음 ultralytics를 설치한다.
- **`libGL` 에러**(`import cv2`에서): 데스크톱이 없는 OS(Lite)다. `pip uninstall -y opencv-python && pip install "opencv-python-headless<4.12"`.
- DGX에서 확인한 조합은 ultralytics 8.4.171, ncnn 1.0.20260526, numpy 1.26.4, opencv 4.11.0이다(`docs/dgx_pip_freeze.txt`). 파이에서는 더 새 버전이 깔릴 것이다. 이 도구들은 "export된 모델은 export 때의 입력 크기를 그대로 쓴다"는 ultralytics 동작에 기대고 있고, 이 동작은 8.4.171에서 확인했다.
- 처음 `import ultralytics`를 하면 `~/.config/Ultralytics/`에 설정 파일이 생긴다. 정상이다.

## 5. 동작 확인

### 5-1. 이미지 1장으로 파이프라인

```bash
cd ~/src/pipeline && source venv/bin/activate
python tools/target_pipeline.py --config tools/pipeline_config.json --image samples/enemy_roomA.jpg
python tools/target_pipeline.py --config tools/pipeline_config.json --image samples/enemy_roomA.jpg --roi-center 412,477     # ROI 384 추적 모드
python tools/target_pipeline.py --config tools/pipeline_config.json --image samples/friend_hallB_1.5m.jpg --save            # 결과 그림 저장
```

stdout에는 JSON만 나온다(모델 로딩 로그와 경고는 stderr). 기대 출력(x86 NCNN에서 얻은 값. 파이에서는 좌표 ±1 px, conf ±0.02 안이면 정상):

| 이미지 | 박스 (x1, y1, x2, y2) | conf |
|---|---|---|
| `samples/enemy_roomA.jpg` (전체 프레임) | 406.4, 466.5, 418.4, 487.2 | 0.84 |
| `samples/enemy_roomA.jpg` `--roi-center 412,477` | 406.5, 466.5, 418.2, 487.5 | 0.83 |
| `samples/enemy_hallB_2.4m.jpg` | 363.2, 392.7, 376.8, 417.8 | 0.90 |
| `samples/friend_hallB_1.5m.jpg` | 853.9, 449.0, 878.2, 487.5 | 0.89 |

세 장 모두 `"action": "hold"`, `"reason": "unknown 1개 (no_refs)"`가 나와야 한다. **색 기준 파일(`tools/color_refs.json`)이 아직 없어서 보류하는 것이 정상이다.** `pitch_table 없음` 경고도 정상이다(피치 보정표를 아직 재지 않았다).
`enemy_roomA.jpg`의 사람 라벨은 (404, 465, 419, 486)이다.

### 5-2. 지연 실측

```bash
python tools/pi_bench.py --models weights/ncnn_r3_coco --image samples/enemy_roomA.jpg --runs 50
```

출력 형식(숫자는 **추정치**이고, 이 실측으로 바꿔야 하는 값이다):

```
arch aarch64 | temp temp=5x.x'C | throttled throttled=0x0 | cpus 4
960x1280_ncnn_model  p50  ~1300-1500 ms  p95 ...
736x960_ncnn_model   p50  ...
480x640_ncnn_model   p50  ...
384x384_ncnn_model   p50  ~150 ms
닫힌 루프 추정: 전체 1회 + ROI 5회 = ... s   (전체만 5회 = ... s)
닫힌 루프 추정: 전체 1회 + ROI 10회 = ... s
throttled(끝) throttled=0x0
```

- 추정 근거: x86에서 잰 해상도 간 비율(640×480 = 1.0, 1280×960 = 4.88×, 384² ROI = 0.47×)에 Ultralytics 문서의 Pi4 수치를 곱한 것이다.
- 목표는 조준 완료 10 s 이내. 계획은 전체 프레임 1회 + ROI 5~10회다.
- `throttled`가 `0x0`이 아니면 전원이나 방열 문제라 수치를 믿을 수 없다.
- 실측값을 사용자에게 알리고 RESUME.md §4의 추정치를 바꾼다.

## 6. 모델 경로 설정

`tools/pipeline_config.json`의 `detector` 부분:

```json
"model": "../weights/ncnn_r3_coco/960x1280_ncnn_model",
"roi_model": "../weights/ncnn_r3_coco/384x384_ncnn_model",
```

- **상대 경로는 이 json 파일의 위치가 기준**이다. `tools/`와 `weights/`가 `~/src/pipeline/` 아래에 나란히 있으면 고칠 것이 없다.
- 다른 곳의 모델을 쓰려면 절대 경로나 `~`로 적는다: `"model": "~/src/pipeline/weights/ncnn_r3_coco/960x1280_ncnn_model"`.
- NCNN은 입력 크기가 export 때 고정된다. 그래서 전체 프레임용(960x1280)과 ROI용(384x384)을 **따로** 지정한다. `roi_model`을 비워도 동작은 하지만, 384 크롭을 960×1280으로 키워서 전체 프레임 모델에 넣게 되어 전체 프레임만큼 느리고(ROI의 이점이 없다) 학습 때와 다른 배율이 된다. 반드시 384 모델을 지정한다.
- **폴더 이름의 `_ncnn_model`은 바꾸면 안 된다.** ultralytics가 폴더 이름으로 형식을 판별한다.
- 비교용 r3로 바꾸려면 두 줄의 `ncnn_r3_coco`를 `ncnn_r3`로 바꾼다.
- 새 모델(r4 등)이 오면 `weights/ncnn_r4/`에 풀고 두 줄을 그 폴더로 바꾼다.

같은 파일의 나머지 값:

| 키 | 지금 값 | 뜻 |
|---|---|---|
| `detector.conf` | 0.25 | 검출 임계값(밴드 감점 적용 뒤 기준) |
| `detector.band`, `band_penalty` | [352, 672], 0.5 | 박스 중심 y가 이 범위 밖이면 conf × 0.5 |
| `color.refs` | `color_refs.json` | 기준색 파일(§8). 없으면 전부 unknown |
| `color.min_safety_ratio` | 3.0 | 기준색 파일의 안전비가 이보다 낮으면 전부 unknown |
| `aim.yaw_calib` | `coef [0, 64]`, `sign -1` | **임시값.** yaw_deg = sign × 64 × (cx − 640)/640. b2 촬영분에서 d(cx)/d(yaw_cmd) = +10.3 px/°였던 것에서 나왔다 |
| `aim.pitch_table` | null | 박스 높이 h(px) → 피치 각도 보정표. 재기 전까지 pitch 0 |

## 7. 파이에 이미 있는 코드와의 관계

**DGX 세션은 파이의 코드를 본 적이 없다.** 아래는 촬영 결과물(`labels.csv`, `sessions.jsonl`)에서 알 수 있는 것과 사용자가 정한 역할 분담이다. 실제 코드와 다르면 실제 코드가 맞다.

| | 담당 | 파일 |
|---|---|---|
| 파이 쪽 기존 코드 | **카메라 제어와 촬영** — 미리보기, 노출·화이트밸런스 설정과 잠금, 서보 구동, 세션 단위 저장 | `03_preview.py`, `05_manual_capture.py`, `colorlab/`, `03_config/camera_config.json` |
| 이 묶음 | **검출·색 판정·조준값 계산** | `tools/target_pipeline.py`, `tools/colorlib.py`, `tools/color_separation.py` |

두 쪽이 만나는 곳:

1. **촬영 폴더 → 색 측정.** `05_manual_capture.py`가 만든 폴더(`images/*.jpg`, `labels.csv`, `sessions.jsonl`)를 `color_separation.py --dir`에 그대로 넣는다. `labels.csv`의 `label`(enemy / friend / both / negative)과 `note`를 읽는다. 거리는 note의 `1.5m` 같은 표기에서 뽑고, both 세션은 note의 `EL`(적군이 왼쪽) / `FL`(아군이 왼쪽)로 좌우를 배정한다. note 이름 규칙은 RESUME.md §5-8.
2. **프레임 → 파이프라인.** 실전 코드에서는 모델을 한 번만 올리고 프레임마다 `run()`을 부른다.

   ```python
   import sys; sys.path.insert(0, "/home/<사용자>/src/pipeline/tools")
   from target_pipeline import Pipeline
   pipe = Pipeline("/home/<사용자>/src/pipeline/tools/pipeline_config.json")   # 한 번만 (모델 로딩에 수 초)
   out = pipe.run(frame_bgr)                           # 획득: 전체 프레임 1280×960
   if out["decision"]["action"] == "engage":
       t = out["decision"]["target"]                   # 선택된 적군 박스 (원본 좌표)
       yaw, pitch = out["aim"]["yaw_deg"], out["aim"]["pitch_deg"]
       out = pipe.run(frame_bgr, roi_center=(t.cx, t.cy))   # 추적: 타겟 주변 384×384 만 검출
   ```

   - `frame_bgr`는 **`cv2.imread`로 읽은 것과 같은 BGR 순서의 uint8 1280×960**이어야 한다. picamera2에서 배열을 직접 받으면 포맷 이름과 실제 채널 순서가 헷갈리기 쉽다. R과 B가 뒤바뀌면 검출은 되는데 **색 판정이 틀린다.** 처음에 `cv2.imwrite`로 한 장 저장해서 색이 맞는지 눈으로 확인한다.
   - 기준색을 잴 때와 실전에서 **같은 경로로 얻은 프레임**을 쓴다(JPEG로 저장한 것을 읽어 잰 기준색을 배열로 직접 받은 프레임에 쓰는 식으로 섞지 않는다).
   - `action`은 `engage`(적군 선택됨) / `hold`(판정 불가인 타겟이 하나라도 있음) / `no_target`(검출 없음 또는 아군뿐).
   - `yaw_deg`는 "타겟을 화면 중앙에 두려면 카메라 서보 yaw 명령을 얼마나 바꿔야 하는가"다(`sign: -1`: 화면 오른쪽 타겟이면 음수). 발사부의 축과 방향 규약은 **실기에서 확인해야 한다.**
3. **`colorlab/`과 `tools/colorlib.py`.** 이 묶음의 색 코드는 `tools/colorlib.py`와 `target_pipeline.ColorClassifier`다. **`RefPatches`, `classify_relative`라는 이름은 이 묶음에 없다.** 파이의 `colorlab/`에 있거나 앞으로 파이에서 추가할 것으로 이해했다. 합칠 때 지켜야 하는 것은 RESUME.md §5의 결정이다: 판정 불가(unknown)가 하나라도 있으면 보류, 기준색은 카메라로 측정한 값, 안전비 3.0 미만이면 판정하지 않음.

## 8. 색 기준 만들기 (새 색지 + 노출 확정 뒤)

```bash
python tools/color_separation.py --dir <P0 촬영 폴더> --model weights/ncnn_r3_coco/960x1280_ncnn_model --out color_report --ball-frac 0.45
cat color_report/report.md
```

- `--ball-frac 0.45`는 **공에만 색지가 있을 때** 쓴다(박스 위쪽 45%만 본다). 받침·막대까지 색칠한 뒤에는 뺀다.
- 통과 기준: 안전비(클래스 간 dE2000 ÷ 클래스 내 흩어짐) **3.0 이상**. 지금까지의 색지는 1.51이었다.
- 통과하면 `cp color_report/color_refs.json tools/color_refs.json`. 그때부터 파이프라인이 enemy / friend를 판정한다.
- 기준색은 **확정한 카메라 설정에서 잰 값**이다. 노출이나 화이트밸런스를 바꾸면 다시 잰다(RESUME.md §5-10).
- 새 색지에서 검출 자체가 버티는지는 `python tools/seed_label.py --images <폴더>/images --model weights/ncnn_r3_coco/960x1280_ncnn_model --out seed_out --device cpu --max-expected 2`의 `status` 집계(none = 미검출)와 `seed_out/sheet_*.jpg`로 본다. **`--device cpu`를 꼭 붙인다**(기본값이 GPU 0번이다).
- `color_separation.py`의 `--model` 경로는 x86에서 NCNN 모델로 16장을 돌려 `.pt`와 같은 결과가 나오는 것까지 확인했다. both 세션의 EL/FL 배정은 **해당 데이터가 없어 돌려 본 적이 없다** — 처음 쓸 때 `color_report/samples.csv`의 `cls` 열을 눈으로 확인한다.

## 9. 동기화 규칙

1. **`tools/`의 원본은 앞으로 파이다**(`~/src/pipeline/tools/`). 색 판정 로직을 포함한 수정은 파이에서 한다. 수정하면 `tools/CHANGELOG.md`에 날짜와 내용을 한 줄 적는다.
2. **GB10·DGX에서는 `tools/`를 수정하지 않는다.** 거기서는 학습과 export만 하고, 파이는 **학습 결과만** 받는다: NCNN 폴더(`weights/ncnn_<이름>/`)와 평가 결과.
3. 학습 장비에서 `tools/`를 고쳐야 할 일이 생기면(예: 학습용 스크립트의 버그) 거기서 고치지 말고 파이에서 고친 뒤 아래 절차로 가져간다.

**GB10 → 파이: 새 모델 받기**

```bash
# 파이에서 (GB10 주소·계정·폴더는 실제 값으로)
scp -r <계정>@<GB10주소>:~/<폴더>/YOLO/weights/ncnn_r4 ~/src/pipeline/weights/
ls ~/src/pipeline/weights/ncnn_r4        # 960x1280_ncnn_model 등 4개 폴더, 각 폴더에 model.ncnn.param / model.ncnn.bin / metadata.yaml
# tools/pipeline_config.json 의 model, roi_model 두 줄을 ncnn_r4 로 바꾼 뒤 §5 로 확인
```

**파이 → GB10: 수정한 `tools/` 가져가기**

```bash
# 파이에서 보내는 경우
rsync -av --exclude __pycache__ --exclude pipeline_config.json --exclude color_refs.json \
      ~/src/pipeline/tools/ <계정>@<GB10주소>:~/<폴더>/YOLO/tools/

# GB10에서 당겨오는 경우 (rsync 가 없으면 scp 로 받아서 덮어쓴다)
cd ~/<폴더>/YOLO && cp -r tools tools.bak_$(date +%m%d)
scp -r <파이계정>@<파이주소>:~/src/pipeline/tools ./tools_from_pi
diff -rq tools tools_from_pi | grep -v __pycache__        # 무엇이 바뀌었는지 먼저 본다
cp tools_from_pi/*.py tools_from_pi/CHANGELOG.md tools/ && rm -r tools_from_pi
```

- **`pipeline_config.json`은 장비마다 다르므로 덮어쓰지 않는다.** 파이의 것은 NCNN 폴더를, GB10의 것은 `.pt`(`../runs/r3_coco/weights/best.pt`)를 가리킨다. 키를 추가했으면 양쪽에 손으로 반영한다.
- `color_refs.json`은 파이의 카메라로 잰 값이다. GB10에서 분석에 필요하면 따로 복사한다(`scp … tools/color_refs.json`). GB10에서 만든 값으로 파이의 것을 덮어쓰지 않는다.
- 가져간 뒤 GB10에서 `python tools/target_pipeline.py --help`와 `python tools/package_dataset.py --help`가 뜨는지 확인한다.

## 10. 처음 할 일 순서

1. §2~§5: 풀기, 환경 확인, 설치, 샘플 3장 확인, `pi_bench.py` 실측.
2. 실측 지연과 설치 중 막힌 점을 사용자에게 알린다.
3. 새 색지와 노출을 확정한 뒤 P0 촬영 → §8.
4. 실기에서 yaw 부호 확인, 피치 보정표(h → 각도) 측정 → `pipeline_config.json`의 `aim`에 입력. 받침·막대를 색칠한 뒤에는 **반드시 다시 잰다.**
