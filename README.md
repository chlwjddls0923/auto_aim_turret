# auto_aim_turret — 고무줄총 자동 조준 터렛

라즈베리파이 4 + 카메라 + SG90 서보 3개로, 1.5~2.0 m에 놓인 타겟을 찾아 **색으로 아군/적군을 구분하고 적군만** 조준·발사한다. 하드웨어는 수업 키트 그대로이고 소프트웨어만으로 푼다.

대회 결과: 약 1.8~2.0 m에서 6발 중 4발 명중, 아군 오사격 0.

```
카메라 1280×960 → YOLO11n(NCNN) 검출 → 박스 안 Lab 색 판정 → 조준점 오차(px) → 서보 이동 → 다시 검출 … 오차 ≤ 6 px이면 발사
```

## 폴더

| 폴더 | 내용 | 자세한 설명 |
|---|---|---|
| `PI/` | 파이 실행 코드. 번호가 실행 순서. `07_aim_calib.py`가 대회 실행 프로그램 | [PI/README.md](PI/README.md) |
| `PI/pipeline/tools/` | 검출·색 판정·조준 공용 로직(`target_pipeline.py`), 색 기준값(`color_refs.json`), 설정(`pipeline_config.json`) | |
| `PI/pipeline/weights/ncnn_r4/` | 배포 모델 r4의 NCNN 변환본 4종(960×1280, 736×960, 480×640, 384×384) + SHA256 | |
| `YOLO/` | 학습·평가 코드, 학습 기록(`runs/`), 설계 문서(`TRAINING_DESIGN.md`) | [YOLO/README.md](YOLO/README.md) |
| `PI/tests/`, `PI/pipeline/tests/` | 카메라·모델 없이 도는 단위 시험 (`pytest`) | |

## 최종 구성

| 단계 | 방법 | 설정 |
|---|---|---|
| 검출 | YOLO11n 1클래스 검출기 r4. COCO 사전학습에서 2,491장으로 파인튜닝, 입력 1280, hsv_h 0.3. NCNN FP32 | 처음 찾기 960×1280, 조준 중 384×384 잘라내기 추적, conf 0.25 |
| 아군/적군 | 박스 안 고채도 픽셀의 Lab a\*b\* 평균을 현장에서 만든 기준색과 비교 | 안전비 3.0 미만 기준값은 사용 금지, 판정 불가 상태 있음 |
| 조준 | 실사격으로 잰 화면 조준점(머리 기준) + 카메라를 피드백 센서로 쓰는 닫힌 루프 | 두 축 오차 ±6 px 안에서만 발사 |
| 안전 | 시작 후 5 s 발사 잠금, 이동 중 색 재판정(아군이 되면 정지) | |

## 실행 순서 (파이)

```bash
python3 01_test_camera.py --truth red          # 카메라 채널 순서 확인 (처음 1회)
python3 03_preview.py                           # 카메라 설정 고정·저장 (장소·조명 바뀔 때)
python3 98_servo_center.py                      # 서보 중앙값 (재조립 때)
pipeline/venv/bin/python pipeline/tools/build_color_refs.py ...   # 현장 사진 6장으로 색 기준값 생성
pipeline/venv/bin/python 08_detect_check.py    # 검출·판정 확인 (서보 없음)
pipeline/venv/bin/python 07_aim_calib.py       # 장전 후 p: 자율 조준·발사, e: 명중 위치 기록
```

학습과 NCNN 변환은 [YOLO/README.md](YOLO/README.md)의 순서를 따른다.

## 버전 메모

- 레포의 `07_aim_calib.py`는 10/11 이전 버전(적응 걸음 제어)이다. 대회에 쓴 축별 한 방향 접근 버전은 파이에만 있었고, 파이를 초기화하면서 올리지 못했다. 동작 설명은 보고서 5.2절에 있다.
- `PI/pipeline/RESUME.md`의 파이 검출 지연(960×1280 1,224 ms, 384×384 142 ms)은 r3_coco NCNN으로 잰 값이다. r4는 같은 구조·같은 입력 크기라 지연이 같다.

## 보고서

`docs/고무줄총_중간보고서.docx` — 방법 논의·시도·검증과 최종 모델. 본 README의 수치는 모두 보고서와 같다.
