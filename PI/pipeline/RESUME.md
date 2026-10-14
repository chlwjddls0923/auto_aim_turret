# RESUME — 로봇 경기 타겟 검출 프로젝트 인계 브리프

2026-10-03, DGX 마지막 작업 세션에서 작성. **DGX는 2026-10-05(월)까지만 쓸 수 있고, 그 뒤로는 GB10과 이 폴더가 전부다.**
이 문서 하나로 맥락 없이 이어받을 수 있게 썼다. 폴더 지도는 [docs/HANDOVER_GB10.md](docs/HANDOVER_GB10.md), 환경 세팅은 [README_FIRST.md](README_FIRST.md)와 [docs/GB10_CHECKLIST.md](docs/GB10_CHECKLIST.md).
경로는 프로젝트 폴더(`YOLO/`) 기준이다(DGX에서는 `/data/YOLO`였다).

## 0. 장비

- **일정**: 10-04부터는 색지 교체와 노출 수정 작업이고, **그 이후의 모든 학습은 GB10에서 한다.** DGX에서 할 수 있는 학습(r3, 3-arm, COCO 시작 비교)은 10-03에 끝냈다.
- **GB10**(aarch64, CUDA 13.0 드라이버 580, 통합 메모리 121 GiB): 10/5 이후의 유일한 작업 장비. 공용 계정이라 개인 폴더와 그 안의 venv만 쓴다. 이 프로젝트를 GB10에서 돌려 본 적은 아직 없다.
- **라즈베리파이4 4GB**: 배포. 조준 완료까지 10 s 이내면 된다.
- DGX(x86, A100 MIG 10 GB): 10/5까지. r3 학습, 3-arm 실험, NCNN export를 여기서 끝냈다.
- **`tools/`의 원본은 10-04부터 파이다**(`~/src/pipeline/tools/`). 색 판정 로직을 포함한 수정은 파이에서 하고, **GB10에서는 `tools/`를 수정하지 않는다.** GB10은 학습과 export만 하고 파이에는 학습 결과(NCNN 폴더)만 보낸다. 파이에서 고친 `tools/`를 GB10으로 가져오는 절차와 파이 설치 방법은 [docs/PI_SETUP.md](PI_SETUP.md). `tools/pipeline_config.json`은 장비별 파일이라 덮어쓰지 않는다(GB10 = `.pt`, 파이 = `tools/pipeline_config.pi.json`에서 만든 NCNN 설정).
- NCNN export에는 `pnnx`가 필요하다. PyPI에 aarch64 휠이 있는 것은 확인했고 `wheels/aarch64/`에도 넣어 두었지만(pnnx 20260526, ncnn 1.0.20260526), **aarch64에서 실제로 설치·실행해 본 적은 없다.** GB10에서 **가장 먼저 확인**한다(§7-A). 안 되면 x86 PC(Windows 가능)에서 export한다.

## 1. 과제

타겟(공 + 막대 + 받침, 현재는 공에만 색지)을 검출한다(YOLO11n, 클래스 1개 `tar`) → 색으로 아군(청록 계열) / 적군(보라 계열)을 판정한다(색지는 §5-9) → 박스 중심 x로 yaw, 박스 높이 h로 피치 보정표를 읽는다. 교전 거리 1.5~2.0 m, 타겟은 정지.

파이프라인 규약: `detect()`는 입력이 전체 프레임이든 ROI 크롭이든 **원본 1280×960 좌표** 박스를 반환한다(ultralytics는 넣은 이미지 좌표로 돌려주므로 ROI는 크롭 오프셋만 더한다). 색·yaw·pitch는 원본 프레임과 원본 좌표만 쓴다.

## 2. 데이터 (전부 1280×960)

| 묶음 | 경로 | 장수 | 라벨 | 비고 |
|---|---|---|---|---|
| b1 | `data/images` + `labellmg/*.xml` | 180 | 사람(LabelImg) | 방 A, 비디오 프레임 296~554. 데이터셋에서는 전부 train |
| b1x | `data/images(1)` + `labellmg/*.xml` | 100 | 사람 98 | 방 A, 프레임 473·537~674. **10-03에 발견. 학습에 쓴 적 없음 → test 전용**. 673·674는 라벨 없음 |
| b2 적군 | `data2/enemy(26.10.02.1754)/{images,labels.csv,sessions.jsonl}` | 830 | 의사 809 → `data2_labels/labels` | 홀 B, 19 세션, center/right × 1.2/1.5/1.8/2.1/2.4 m, 서보 2° 스텝 정지 촬영, yaw_cmd 65~125. 제외 21 = 미검출 18(타겟 존재 미확인) + 손으로 놓는 장면 3 |
| 아군 | `data2/friend(26.10.02.1839)/*.jpg` | 662 | 의사 662 → `friend_pseudo/labels` | **labels.csv·sessions.jsonl 없음(사용자 업로드 대기)**. 파일 순서로 보면 idx 1~220 중앙, 221~418 좌, 419~662 우, 각 5거리 |
| test | `test/` | 8 | b1x에 사람 라벨 있음 | 방 A. b1x와 같은 이미지 |

- 학습 데이터셋 `handover/dataset`: train 1,423 / val 228 / test 98. 구성과 리스트는 [docs/HANDOVER_GB10.md](docs/HANDOVER_GB10.md).
- 의사 라벨 육안 확인: b2는 review 49장 전수 + 무작위 35장, 아군은 confident 48장 + review 41장 → 모두 정확.
- **아군 none 25장은 전부 타겟이 있다**(10-03 확인). 640으로는 놓쳤지만 r2를 입력 1280으로 돌리면 25장 모두 conf .73~.85로 잡힌다. 원거리 중앙, 갈색 문 앞이라 대비가 낮은 장면이다. 이 라벨을 `friend_pseudo/labels`·`xml`에 넣었고 train에 포함했다. 2장(198, 205)은 밴드 밖에 오검출이 하나씩 더 있어 최고 conf 박스만 남겼다. 기록은 `friend_pseudo/none_recheck/`.
- **사람 검수 대기**: `friend_pseudo/review_list.csv` 149장(위 25장 포함). `inventory.csv`의 `review` 열에 `pending`으로 표시돼 있다. 검수 반영 방법은 §7-C.
- **val은 의사 라벨이다.** 의사 라벨을 만든 모델 계열로 평가하면 부풀려지므로 val 수치는 run 간 상대 비교용이다. 사람 라벨로 잰 수치는 test(b1x) 98장뿐이고, 이것은 방 A·적군색·원거리에 한정된다.
- **아군 val 분할은 세션 단위로 깔끔하지 않다.** 메타가 없어 박스 높이·위치로 추정한 strata로 나눴고, val 91장 중 8장쯤은 다른 세션에서 왔으며 val 세션으로 보이는 구간의 19장쯤은 train에 남아 있다. 메타가 오면 다시 패키징한다(§7-D).
- 파일명 충돌: b1 `enemy_000296.jpg`와 b2 `enemy_000296.jpg`는 다른 이미지다. 데이터셋에서는 uid(`b1_`, `b1x_`, `b2_`, `fr_`)로 구분한다.

## 3. 모델

| 모델 | 학습 | 용도 |
|---|---|---|
| `runs/r2/weights/best.pt` | b1 138장 + b2 494장, 입력 640, 74 epoch | 의사 라벨을 만든 모델. NCNN은 `weights/ncnn/` |
| `runs/r3/weights/best.pt` | r2에서 시작, 전체 train 1,423장, 입력 1280, `hsv_h=0.3`, 100 epoch | 이전 배포 후보. 비교용으로 보관. NCNN은 `weights/ncnn_r3/` |
| **`runs/r3_coco/weights/best.pt`** | **COCO 가중치(yolo11n.pt)에서 시작**, 나머지는 r3와 같음 | **현재 배포 후보**(§6의 비교로 r3를 대체). NCNN은 `weights/ncnn_r3_coco/` |
| `runs/3arm/<arm>_<색>/weights/best.pt` | yolo11n.pt(COCO)에서 시작, 한 색만 학습 | 색 불변성 실험(§6) |

NCNN 폴더는 `<세로>x<가로>_ncnn_model` 4종(960x1280 전체 프레임, 736x960, 480x640, 384x384 ROI)이다. **폴더 이름에 `_ncnn_model`이 있어야 ultralytics가 NCNN으로 로드한다**(예전 이름 `ncnn_960x1280`은 로드되지 않아 10-03에 고쳤다). r2·r3 모두 .pt와 NCNN 960×1280의 검출 박스가 일치하는 것을 확인했다.

예전 run(det640, crop640, 해상도 스윕)은 `results.csv`와 `args.yaml`만 남겼다. 밴드 크롭(crop640) 전략은 폐기했다.

## 4. 확정 측정값

**박스와 기하**
- b1 박스 중앙값 25×43 px. b2 거리별 h 중앙값: 1.2 m 47.5 / 1.5 m 38.7 / 1.8 m 32.8 / 2.1 m 29.3 / 2.4 m 26.0 px → **h ≈ 59.1/d(m)**, 잔차 중앙값 1.5 px. 절편을 허용하면 h = 53.7/d + 3.6. 2.1~2.4 m에서 +1.3~1.6 px이 남는 원인은 미확정(받침의 고정 픽셀인지 의사 라벨 편향인지 → 사람 라벨로 판별).
- 배치(center/right)에 따른 h 차이는 2.3 px 이하 → cx 보정 불필요. 두 묶음 합산 타겟 y 범위 389~632.
- **yaw**: b2에서 d(cx)/d(yaw_cmd) = **+10.3 px/°**(세션별 7.8~11.6). yaw_cmd가 커지면 타겟이 화면 오른쪽으로 간다. `tools/pipeline_config.json`은 `coef [0, 64]`, **`sign: -1`**(카메라 서보 명령 기준: 화면 오른쪽 타겟을 중앙에 두려면 yaw_cmd를 줄인다). 발사부의 축과 방향 규약은 실기에서 확인해야 한다.

**색 (현재 노출, 공 영역의 채도 > 0.24 픽셀)**
- 적군: hue 234°, S .44, **V .27**, Lab (22.5, +6.7, −5.1). 아군: hue 74°, S .39, **V .26**, Lab (27.6, −1.6, +7.1).
- 클래스 간 dE2000 16.1, 클래스 내 RMS 10.7 / 10.2 → **안전비 1.51 (기준 3.0 미달)**. 지금 색지(`#0E7C7B` / `#5B3FA8`)의 공칭색끼리는 dE 35.3인데 카메라에서는 16.1로 줄어든 것이다. 2.4 m에서 고채도 픽셀은 적군 27개 / 아군 63개뿐이다.
- 지금 노출에서는 색 판정이 작동하지 않는다. 모의 판정은 100% unknown(= 전부 보류)이다.

**검출기와 색**
- ultralytics 8.4의 `hsv_h`는 **덧셈**이다: `lut_hue = (x + r·180) % 180`, r ∈ [−h, h] → 0.015 = ±5.4°, 0.1 = ±36°, 0.3 = ±108°. 색지의 색상 차는 지금 색지가 약 77°, 새 후보가 약 95°라 0.1로는 못 덮는다.
- **r2는 색맹이 아니다.** 아군 637장 recall이 컬러 .986 → **흑백 .854**, 적군 b2는 흑백에서 .906이다(입력 640). 색상(hue)만 돌린 시험에서는 검출률이 유지됐는데(회전 없음 94.9%, +44° 95.9%, +86° 77.9%), 채도와 명도가 그대로였기 때문으로 보인다(해석). 즉 **색지를 바꾸면 검출 성능이 떨어질 수 있다.** 색 분리가 실패해 색을 바꿔야 할 가능성이 있으므로 이것은 직접적인 리스크이고, 3-arm 실험(§6)의 목적이 여기에 있다.
- r2의 b2 개선(미검출 336 → 18)은 `hsv_h`가 아니라 b2 494장을 학습에 넣었기 때문이다.

**해상도와 지연**
- b1 해상도 스윕(val mAP50-95): 160 .343 / 224 .597 / 320 .690 / 416 .732 / 512 .752 / 640 .765.
- 지연 비율(x86 NCNN FP32 4스레드): 640×480 29 ms = 1.0 / 960×736 2.59× / **1280×960 4.88×** / 384² ROI 0.47×.
- **Pi4 실측 (2026-10-04, `pi_bench.py --runs 50`, r3_coco NCNN FP32, ultralytics 8.4.171, ncnn 1.0.20260526, throttled=0x0):**

  | 모델 | p50 | p95 |
  |---|---|---|
  | 960x1280 (전체 프레임) | **1224 ms** | 1286 ms |
  | 736x960 | 694 ms | 724 ms |
  | 480x640 | 300 ms | 349 ms |
  | 384x384 (ROI) | **142 ms** | 157 ms |

  닫힌 루프(추론만): 전체 1회 + ROI 5회 = **1.93 s**, + ROI 10회 = **2.64 s** (전체만 5회 6.12 s / 10회 12.24 s). 서보 이동·흔들림 대기·캡처 시간은 빠져 있다. 측정 중 CPU 온도 50 → 74 ℃(Pi4는 80 ℃부터 속도를 낮춘다) → 연속 운용에는 방열판/팬 권장.
  같은 파이에서 x86 기대 박스·conf가 샘플 3장 모두 소수 첫째 자리까지 일치했다. CLI 1회 실행은 11~14 s인데 대부분 torch·ultralytics import와 모델 로딩이다 → 실전에서는 `Pipeline`을 한 번만 만든다.

## 5. 확정 결정

1. 입력 = **전체 1280×960**(획득) + **원본 해상도 ROI 384×384**(추적, 같은 모델). 닫힌 루프 5~10프레임 → 전체 1회 + ROI 9회 ≈ 2.8 s(추정). 밴드 크롭·폴백·dual 학습은 폐기. 밴드 y 352~672는 conf 감점(×0.5)으로만 쓴다.
2. ~~(10-04 재설계로 11번이 대체, 절대값 폴백으로만 남음)~~ 색 판정 = 박스 안 **고채도 픽셀**의 Lab 평균 → 기준색과 dE2000. 게이트: 픽셀 수 < min_px(임시 20) → unknown(`few_px`), |dE_적 − dE_아| < margin → unknown(`ambiguous`). **unknown이 하나라도 있으면 보류**, 적군이 여러 개면 **h가 가장 큰(가까운) 것**. 기준색·margin·min_px는 색칠 후 `tools/color_separation.py`가 만든 `color_refs.json`으로만 정한다.
3. (폴백 경로에 계속 적용) **색 안전 게이트**: `ColorClassifier`는 `color_refs.json`이 없거나 그 안의 안전비가 3.0 미만이면 기준색을 쓰지 않고 전부 unknown으로 둔다. 시운전 산출물(안전비 1.51)이 실수로 연결되는 것을 막는다.
4. (10-04: 기본은 WB 고정 + AE 켜짐, 노출 루프를 켜면 AE 끄고 ExposureTime 직접 조정 — 11번) 노출: **노출 시간 우선**(정지 타겟이라 블러 비용이 없다), 게인은 노출을 더 못 올릴 때만. 조건 = 타겟 패치 V ≥ 0.5(S ≥ 0.4). 벽 포화는 허용하되 포화 영역이 30%를 넘으면 경고.
5. 라벨: b2 809장과 아군 662장의 의사 라벨을 학습에 쓴다. 아군은 `review_list.csv` 149장을 사람이 검수한다. val/test로 쓸 세션은 LabelImg 전수 검수 후에 쓴다. test 후보: center_1.8m + **right_1.5m**(교전 거리 기준) + 아군·both의 같은 거리대.
6. 1차 지표: 박스 **중심 오차 px**(yaw) + **높이 오차 px**(피치) + recall@conf .25 + 음성 FP/프레임. mAP50은 포화라 쓰지 않는다. `tools/eval_cross_color.py`가 이 지표를 낸다.
7. 3-arm 실험: `hsv_h` 0.015 / 0.3 / 흑백 × (적군만 학습 → 아군 검출률, 아군만 학습 → 적군 검출률). 색지를 바꿔도 재학습이 필요 없는 설정을 고른다. 최종 검출기는 색맹이어야 한다.
8. 촬영 계획(색칠·노출 수정 후, 하루 1회, note 이름 고정):
   - P0: `painted_both_EL-FR_{1.5,1.75,2.0}m` + `painted_both_FL-ER_{…}` 각 40장 + `painted_negative_bin` 10장 → `color_separation.py` 통과 여부 확인
   - P1: `painted_{left,center,right}_{1.5,1.75,2.0}m` 적군·아군 각 9 × 40장
   - P2: `painted_both_Enear1.5-Ffar2.0_{left,right}` / `Fnear…` 4 × 40장, `painted_negative_{empty,people,chairs}` 3 × 30장, `painted_static_{enemy,friend}_1.75m` 15장
   - **labels.csv와 sessions.jsonl을 함께 올린다.**
9. **색지를 밝은 색으로 교체한다**(10-03 결정, 사용자).
   - 지금까지의 색지: 아군 `#0E7C7B`(L\* 47.1) / 적군 `#5B3FA8`(L\* 35.3). 카메라에서 V 0.26~0.27로 어둡게 찍혀 색 판정이 안 된다(안전비 1.51).
   - 새 후보: 아군 `#2FD6CE`(L\* 78.0, Lab 78, −42, −8, hue 177°) / 적군 **`#C98CFF`**(L\* 68.3, Lab 68, +44, −49, hue 272°, S 0.45). 공칭색끼리의 거리는 dE76 95.9, dE2000 42.7(지금 색지는 dE2000 35.3).
   - 근거: 공칭 L\*가 카메라에서 약 18 떨어진다(측정: 아군 47.1 → 27.6, 적군 35.3 → 22.5). 타겟이 V ≥ 0.5로 찍히려면 **공칭 L\* 70 이상**이 필요하다.
   - 적군 후보는 처음 `#B573F0`(L\* 60.4)였다가 기준에 못 미쳐 `#C98CFF`로 바꿨다(10-03, 사용자). 기준 70에 근접하고, 더 밝히면 채도가 떨어져 색상이 약해지므로 여기가 균형점이다. 실제로 충분한지는 P0에서 V와 안전비로 확인한다.
   - 예전 문서(TRAINING_DESIGN.md 등)에 나오는 `#14A3A1` / `#7E4FC4`는 처음에 추천했던 색이고 지금 계획이 아니다.
10. ~~(10-04 재설계로 11번이 대체: 기준은 프레임마다 화면 안 기준 조각에서 읽는다)~~ **기준색은 공칭값이 아니라, 최종 카메라 설정에서 카메라로 측정한 Lab 값이다**(`color_separation.py`가 만드는 `color_refs.json`). 화이트밸런스나 노출 때문에 색감이 공칭과 달라져도 상관없다. 두 색의 **상대 거리(안전비 3.0 이상)** 만 본다. 카메라 설정은 한 번 확정하면 **데이터 수집과 실전 내내 잠근다**(설정이 바뀌면 기준색을 다시 재야 한다).
11. **색 판정 재설계 (10-04)** — 고정 기준색 대신, 총(카메라 화면) 안에 붙인 **기준 조각 3개(아군·적군·흰색)** 를 프레임마다 읽어 비교한다.
    - 1차 = 상대 판정(`colorlib.classify_relative`): 흰 조각으로 화면 안 WB 게인을 구해 적용 → 타겟과 두 색 조각의 **a\*b\* 거리**(L\* 제외) 비교. 차이 < `margin_ab` → unknown(`ambiguous`).
    - 기준 조각이 포화·어두움·픽셀 부족이면 그 프레임은 `color_refs.json` 절대값으로 폴백, 그것도 없거나 안전비 미달·JPEG 산출물이면 unknown(`ref_<조각>_<상태>`).
    - 설정: `pipeline_config.json`의 `color.mode/ref_patches/margin_ab/...`, `exposure_loop`(기본 꺼짐). 조각 좌표와 `margin_ab`는 실측 후 입력(§3 측정: 총 몸체 조각 vs 2 m 타겟의 a\*b\* 차이).
    - 색 쌍 선택은 `tools/swatch_measure.py` → `swatch_db.csv` → `tools/pair_select.py`(safety_worst ≥ 5, 2 m에서 C\* ≥ 25·픽셀 ≥ 40, 배경 거리 ≥ 3×spread). 9번 색은 후보 중 하나일 뿐이다.
    - **색 측정은 PNG만**(03_preview `g` 키). JPEG는 `--allow-jpeg` 시운전에서만 쓰고 결과에 `jpeg_input=true`가 붙는다.
    - 실전 실행 = `tools/run_match.py`(모델 1회 로드·예열 후 키보드 SPACE로 시작). 테스트 직전 점검 = 03_preview `m` 키.

## 6. DGX에서 끝낸 것 (2026-10-03)

- 데이터셋 패키징과 5 epoch 검증(r2에서 시작, 1280, batch 8): 스캔 1,423 + 228장·배경 0·손상 0, box loss .92 → .60, val mAP50 .995, mAP50-95 .78 → .82, GPU_mem 4.5~4.7 G, 약 41 s/epoch.
- `tools/target_pipeline.py` 작성과 확인: ROI 모드와 전체 프레임의 박스 중심 차이 1.3 px 이하, 밴드 감점·색 게이트·교전 결정 동작 확인.
- `color_separation.py`(`--labels` 경로)와 `pseudo_bias_check.py`(단일 세트) 시운전: §4의 수치를 재현했다.

<!-- RESULTS:BEGIN -->
### r3 학습 결과

`runs/r3`: r2에서 시작, train 1,423장, 입력 1280, batch 8, `hsv_h=0.3`. 100 epoch를 다 돌았고(69분, early stop 없음) best는 95 epoch. val(의사 라벨) mAP50 .995, mAP50-95 .864.

1차 지표(conf .25, IoU .5, `tools/eval_cross_color.py`):

| 평가 세트 | 모델·입력 | recall | 평균 conf | 중심 오차 중앙값 / p95 (px) | 높이 오차 평균 (px) | FP/프레임 |
|---|---|---|---|---|---|---|
| **test 98장** (사람 라벨, 방 A, 적군색, h 20~31 px) | r2 @640 | 1.000 | .80 | 0.90 / 1.93 | −0.28 | 0 |
| | r2 @1280 | .990 | .81 | 0.79 / 1.60 | −0.30 | .061 |
| | **r3 @1280** | **.980** | .85 | **0.56 / 1.28** | −0.97 | 0 |
| val 적군 137장 (의사 라벨) | r2 @1280 | 1.000 | .78 | 0.84 / 1.92 | −1.25 | .131 |
| | r3 @1280 | 1.000 | .90 | 0.46 / 1.14 | 0.00 | .015 |
| val 아군 91장 (의사 라벨) | r2 @1280 | 1.000 | .78 | 0.99 / 1.85 | −1.68 | .066 |
| | r3 @1280 | 1.000 | .91 | 0.47 / 1.17 | +0.12 | 0 |

읽는 법과 주의:
- 배포 입력(1280)에서 r3는 r2보다 중심이 정확하고 오검출이 훨씬 적다. conf 임계값을 .10까지 내려도 r3의 FP는 0~.015/프레임이고 r2@1280은 .18~.21/프레임이다. 중심 오차 0.56 px는 yaw 약 0.05°다.
- **r3는 사람 라벨 test에서 2장을 놓쳤다**(r2@640은 둘 다 검출). `b1x_enemy_000598`은 타겟이 로봇 자체의 앞부분에 반쯤 가려진 장면이고, `b1x_enemy_000651`은 타겟이 완전히 보이는데 conf가 .126이다(임계값을 .10으로 내리면 잡힌다). 방 A의 원거리 타겟에서는 r3가 r2보다 낫다고 할 수 없다. 표본은 98장이다.
- **r3의 박스는 사람 라벨보다 평균 1 px쯤 짧다**(h 오차 −0.97). 피치 보정표(h → 각도)는 실제 배포할 모델의 박스로 재야 한다.
- val 행은 r2@640이 만든 의사 라벨이 기준이라 r2@640 자신은 오차가 0에 가깝게 나온다(표에서 뺐다). val의 높이 오차는 "r2@640 라벨과의 차이"로만 읽는다.
- r3의 NCNN 4종(`weights/ncnn_r3/`)은 .pt와 검출 박스가 일치한다.

### r3_coco: COCO 가중치에서 시작한 재학습과 교체 결정

`runs/r3_coco`: `weights/yolo11n.pt`(COCO)에서 시작, 나머지 설정은 r3와 같다(train 1,423장, 입력 1280, batch 8, `hsv_h=0.3`, epochs 100, patience 30). 100 epoch를 다 돌았고(69분) best는 99 epoch, val(의사 라벨) mAP50-95 .866(r3는 .864).

**사람 라벨 test 98장(1차 기준)**, 입력 1280, `tools/compare_models.py` → `runs/compare_test_human.md`:

| 지표 | r3 | **r3_coco** |
|---|---|---|
| recall @conf .25 / .15 / .10 | .980 / .980 / .990 | **.990 / 1.000 / 1.000** |
| 평균 conf | .846 | .854 |
| 중심 오차 중앙값 / p95 (px) | 0.560 / 1.284 | 0.561 / 1.279 |
| 높이 오차 평균 (px) | −0.97 | **−0.82** |
| \|높이 오차\| 중앙값 (px) | 1.23 | **1.08** |
| FP/프레임 @.25 / .15 / .10 | 0 / 0 / .010 | 0 / 0 / 0 |

- 둘 다 검출한 96장의 짝비교: 중심 오차 차이 중앙값 −0.03 px(Wilcoxon p = .19, **차이 없음**), |높이 오차| 차이 중앙값 −0.12 px(p < .001, **r3_coco가 더 정확**).
- r3가 놓친 2장: `b1x_enemy_000651`(완전히 보이는 타겟)은 r3 conf .126 → **r3_coco .772로 검출**. `b1x_enemy_000598`(로봇 앞부분에 반쯤 가려짐)은 r3는 박스 없음 → r3_coco conf .244로 임계값 .25에 살짝 못 미친다(.15에서는 잡힌다).
- val(의사 라벨): 적군 137장은 중심 오차 0.43 vs r3 0.46 px(짝비교 p = .03), FP 0 vs .015/프레임. 아군 91장은 차이 없음(0.45 vs 0.47 px, p = .94). recall은 둘 다 1.000.
- **결정: r3_coco로 교체.** 판단 기준은 "recall 몇 장 차이보다 중심·높이 오차가 우선, r3의 중심 오차 수준을 유지하면서 recall이 같거나 나으면 교체"였고, 중심 오차는 같고 높이 오차·recall·오검출이 같거나 낫다. `tools/pipeline_config.json`의 기본 모델을 r3_coco로 바꿨고, NCNN 4종(`weights/ncnn_r3_coco/`)은 25장에서 .pt와 검출 박스가 일치한다(ROI 384 NCNN은 전체 프레임 대비 중심 차이 중앙값 0.15 px, 최대 0.89 px).
- r3_coco의 박스도 사람 라벨보다 평균 0.8 px쯤 짧다. 피치 보정표는 r3_coco의 박스로 잰다.
- 이후 학습의 시작점은 COCO 가중치다(`scripts/train_r3.sh`의 기본값을 `weights/yolo11n.pt`로 바꿨다). r2에서 이어 학습할 이유는 없다.
- 표본은 98장이고 방 A·적군색·원거리에 한정된다. 두 모델의 차이는 작으므로, P0 촬영분에서 둘을 다시 비교할 수 있게 r3도 보관한다.

### 3-arm 결과

6 run 모두 끝났다(yolo11n.pt COCO 가중치에서 시작, 입력 1280, 최대 60 epoch, patience 20. `h0015_friend`는 54, `gray_friend`는 57 epoch에서 early stop). conf .25 기준이고 원본은 `runs/3arm/table.txt`, `summary.csv`, 각 run의 `cross_boxes.csv`.

| arm | 학습에 쓴 색 | 반대색 전체 recall | 놓친 장수 | 사람 라벨 test recall | 반대색 중심 오차 중앙값 (px) | 반대색 FP/프레임 |
|---|---|---|---|---|---|---|
| `hsv_h=0.015` | 적군만 (852장) | .997 (아군 662) | 2 | 1.000 | 0.57 | .005 |
| `hsv_h=0.015` | 아군만 (571장) | 1.000 (적군 b2 809) | 0 | .990 | 0.80 | .007 |
| **`hsv_h=0.3`** | 적군만 | **1.000** | 0 | **1.000** | 0.63 | 0 |
| **`hsv_h=0.3`** | 아군만 | **1.000** | 0 | **1.000** | 0.69 | .002 |
| 흑백 | 적군만 | .988 | 8 | .990 | 0.59 | 0 |
| 흑백 | 아군만 | 1.000 | 0 | .990 | 0.82 | .001 |

해석:
- 세 arm 모두 **한 색만 보고 학습해도 반대색을 98.8~100% 잡는다.** 입력 1280으로 학습한 검출기는 주로 모양을 본다. 아군만 본 모델이 한 번도 보지 못한 방 A의 사람 라벨 test도 99~100% 잡는다.
- 네 칸이 모두 1.000인 것은 `hsv_h=0.3`뿐이다. arm 간 차이는 2~8장이라 결정적이지는 않지만 순서는 일관된다(0.3 ≥ 0.015 ≥ 흑백). **학습 설정은 `hsv_h=0.3`을 유지한다**(r3가 이미 이 설정).
- 흑백 arm은 색 정보가 전혀 없는데도 98.8~100%다. 놓친 8장은 모두 2.1~2.4 m 중앙 세션의 아군 타겟이다(idx 171~214, 640에서 놓쳤던 25장과 같은 저대비 구간). 흑백 입력으로 배포할 이유는 없다.
- **한계**: 지금 노출에서는 두 색이 모두 어둡게(V ≈ .26) 찍혀 서로 비슷하다. 이 실험이 보인 것은 "어두운 두 색 사이의 불변성"이고, **노출을 고치고 색칠한 뒤의 선명한 색에도 버티는지는 아직 모른다.** P0 촬영분이 오면 재학습 없이 이 6개 모델과 r3로 검출률부터 본다(§7-E).
- 부수 관찰: `hsv_h=0.3` 모델은 적군만 본 것도 아군만 본 것도 사람 라벨 test 98장을 모두 잡았는데, r2 가중치에서 시작한 r3는 96장이었다. 그래서 전체 데이터를 COCO 가중치에서 다시 학습해 비교했다(아래 r3_coco).
<!-- RESULTS:END -->

## 7. 다음 할 일

### A. GB10 환경 (클로드 코드)
1. [README_FIRST.md](README_FIRST.md) 순서로 풀고 세팅한 뒤 `bash scripts/setup_gb10.sh`.
2. `pip install ncnn pnnx`가 되는지 확인하고(안 되면 `pip install --no-index --find-links wheels/aarch64 ncnn pnnx`), `python tools/export_ncnn.py --model runs/r3_coco/weights/best.pt --out /tmp/ncnn_test --name t --check-image test/enemy_000624.jpg`로 실제 export가 되는지 본다. 안 되면 새 모델의 export는 x86 PC에서 해야 하므로 사용자에게 바로 알린다([wheels/README.md](wheels/README.md)).
3. `EPOCHS=2 NAME=probe bash scripts/train_r3.sh`로 GB10의 GPU_mem과 s/epoch를 잰다.

### B. 사용자가 실물로 해야 하는 것 (이 순서)
1. **색지를 새 후보로 교체**(§5-9)하고 **노출을 수정**(§5-4)한다. 참고: 지금 색지에서 아군의 측정 색상각(Lab 103°)이 공칭(196°)과 달랐는데, V 0.26에서는 색상이 의미가 없기 때문으로 본다. 새 색지에서는 측정 색상각이 공칭 방향(아군 ≈ 191°, 적군 ≈ 312°)과 대략 맞는지 P0 결과의 `samples.csv`로 확인한다.
2. 카메라 설정 확정·잠금(§5-10) → 받침·막대 색칠 → P0 촬영(§5-8).
3. 아군 세션 메타(labels.csv, sessions.jsonl) 업로드.
4. 파이4에 `pi_bundle.tar.gz`를 풀고 [docs/PI_SETUP.md](PI_SETUP.md) 순서로 설치한 뒤 `tools/pi_bench.py`로 지연 실측. 묶음은 `bash scripts/make_pi_bundle.sh`로 다시 만들 수 있다(새 모델을 export한 뒤 파이로 보낼 때는 NCNN 폴더만 보내도 된다).
5. 실기에서 yaw 부호와 피치 보정표(h → 각도) 측정. 받침·막대를 색칠하면 외형이 바뀌므로 **h ≈ 59.1/d 계수와 피치 보정표는 색칠 후에 반드시 다시 잰다**(배포할 모델의 박스로). 조준점(카메라–총구 어긋남) 은 `~/src/07_aim_calib.py` 로 거리별(1.5~2.0 m) 실사격 기록 → `pipeline/calib/aim_table.json` 이 생기면 `Aimer` 가 자동으로 읽는다(10-04 추가, 실기 기록은 아직 없음).

### C. 사람 검수 반영
1. LabelImg로 `data2/friend(26.10.02.1839)`(이미지)와 `friend_pseudo/xml`(XML)을 열고 `friend_pseudo/review_list.csv`의 149장을 본다. 198·205번은 박스가 하나만 남아 있는지 확인한다.
2. `python tools/xml2yolo.py --xml friend_pseudo/xml --labels friend_pseudo/labels --only friend_pseudo/review_list.csv` (먼저 `--dry-run`으로 바뀐 장수를 본다).
3. b2의 val 세션(center_2.4m 2개, right_1.5m)과 test 후보 세션도 같은 방식으로 `data2_labels/xml` → `data2_labels/labels`.
4. `python tools/package_dataset.py` → sanity 통과 확인 → `NAME=r4 bash scripts/train_r3.sh`(COCO 가중치에서 시작) → `python tools/compare_models.py --list handover/dataset/list_test_human.txt --models r3_coco=runs/r3_coco/weights/best.pt r4=runs/r4/weights/best.pt`로 교체 여부 판단(중심·높이 오차 우선).
5. 사람 라벨이 생기면 `tools/pseudo_bias_check.py --set pseudo:… --set human:…`로 §4의 원거리 +1.3~1.6 px이 라벨 편향인지 판별한다(이 비교 경로는 아직 돌려 본 적이 없다).

### D. 아군 메타가 오면
`python tools/package_dataset.py --friend-meta "data2/friend(26.10.02.1839)/labels.csv" --friend-val-notes <val로 쓸 note들>` → uid가 `fr_<세션>_<idx>`로 바뀌고 val이 실제 세션 단위가 된다. note 이름을 보고 `--friend-val-notes`를 정한다(기본값 `right_1.5m left_2.4m`은 추측이다).

### E. P0 촬영분이 오면
1. **재학습하기 전에 먼저 현재 배포 후보 모델로 검출을 확인한다**: `python tools/seed_label.py --images <dir> --model runs/r3_coco/weights/best.pt --out <out> --imgsz 1280 --max-expected 2`. r3와 r3_coco의 차이가 작았으므로 `runs/r3/weights/best.pt`로도 같이 돌려 비교한다. 출력의 `status` 집계에서 none(미검출) 비율을 보고 대조 시트(`sheet_*.jpg`)를 눈으로 확인한다. 3-arm에서 한 색만 학습해도 반대색을 98.8~100% 잡았으므로, 선명한 색에서도 버티면 r4 학습 부담이 크게 줄어든다. **여기서 검출이 떨어지면 그때 r4를 학습한다**(새 데이터를 라벨·검수 → `package_dataset.py`에 묶음 추가 → `NAME=r4 bash scripts/train_r3.sh`).
   같은 명령을 `runs/3arm/*/weights/best.pt` 6개로도 돌려 출력의 `status` 집계(none = 미검출)를 비교한다. 색칠 후 외형에 어느 학습 설정이 강한지 재학습 없이 알 수 있다.
2. `python tools/color_separation.py --dir <dir> --model runs/r3_coco/weights/best.pt --out color_report` → 안전비 ≥ 3.0이면 `color_report/color_refs.json`을 `tools/color_refs.json`으로 복사. `--model` 자동 검출 경로와 both 세션의 EL/FL 좌우 배정은 **아직 돌려 본 적이 없으니** 결과의 `samples.csv`를 눈으로 확인한다.
3. 미달이면 색을 바꾸거나 노출을 다시 조정한다. 색을 바꾸면 3-arm 결과에 따라 검출기 재학습 여부를 정한다.

## 8. 리스크

1. **색 판정은 노출 수정 전까지 작동하지 않는다**(안전비 1.51). 노출 수정 → 색칠 → P0 통과가 선결.
2. **새 색지가 카메라에서 충분히 밝고 서로 멀게 찍히는지 미확인**이다. 적군 후보 `#C98CFF`는 공칭 L\* 68.3으로 기준(70)에 조금 못 미치고 채도가 0.45로 낮은 편이다(§5-9).
3. **선명한 새 색에서 검출기가 버티는지 미확인**이다. r2는 색에 일부 의존했고(§4), 1280으로 학습한 모델들은 지금의 어두운 두 색 사이에서는 불변이었다(§6). P0에서 먼저 확인한다(§7-E).
4. 받침·막대 색칠 후 외형이 바뀐다 → h 계수와 피치 보정표 재측정, 색칠 후 데이터로 검출기 재확인(라벨 규약은 공 + 막대 + 받침 유지).
5. val이 의사 라벨이고, 사람 라벨 test는 방 A·적군색뿐이다. 홀 B와 아군에 대한 사람 라벨 수치는 아직 없다.
6. 아군 메타 부재(세션·거리가 추정치), 우측 2.4 m는 11장뿐이다.
7. GB10: 공용 계정의 설정·캐시 오염과 타인 프로세스 종료 사고, pnnx aarch64 휠 유무 미확인, 이 프로젝트를 GB10에서 실제로 돌려 본 적 없음.

## 9. 환경 메모

- GB10: torch를 먼저 설치하고 그다음 ultralytics(순서가 바뀌면 torch가 갈아엎어진다). PyTorch cu130 인덱스에 aarch64 휠이 있는 것은 10-03에 확인했다(torch 2.9~2.14, cp310~cp315). DGX는 torch 2.7.1이었으므로 버전이 다르다 — DGX 환경 전체 버전은 `docs/dgx_pip_freeze.txt`. **pandas는 따로 설치한다**(ultralytics 8.4가 더 이상 같이 설치하지 않는다). `scripts/env.sh`가 캐시와 ultralytics 설정을 프로젝트 폴더 안으로 돌린다.
- 학습 설정 참고(DGX 실측): 1280 입력 batch 8에서 GPU_mem 4.5~4.7 G, `cache=ram` 약 5.7 GB. ultralytics `optimizer=auto`는 이 규모에서 AdamW lr 0.002를 고른다.
- DGX에서 썼던 환경: python 3.11, torch 2.7.1+cu128, ultralytics 8.4.171, opencv-python-headless 4.11, numpy 1.26, pandas 2.0, ncnn·pnnx 20260526.
