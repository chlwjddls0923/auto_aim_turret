# RESUME — 로봇 경기 타겟 검출 프로젝트 인계 브리프

2026-10-03, DGX 마지막 작업 세션에서 작성. **DGX는 2026-10-05(월)까지만 쓸 수 있고, 그 뒤로는 GB10과 이 폴더가 전부다.**
이 문서 하나로 맥락 없이 이어받을 수 있게 썼다. 폴더 지도는 [docs/HANDOVER_GB10.md](docs/HANDOVER_GB10.md), 환경 세팅은 [README_FIRST.md](README_FIRST.md)와 [docs/GB10_CHECKLIST.md](docs/GB10_CHECKLIST.md).
경로는 프로젝트 폴더(`YOLO/`) 기준이다(DGX에서는 `/data/YOLO`였다).

## 0. 장비

- **GB10**(aarch64, CUDA 13.0 드라이버 580, 통합 메모리 121 GiB): 10/5 이후의 유일한 작업 장비. 공용 계정이라 개인 폴더와 그 안의 venv만 쓴다. 이 프로젝트를 GB10에서 돌려 본 적은 아직 없다.
- **라즈베리파이4 4GB**: 배포. 조준 완료까지 10 s 이내면 된다.
- DGX(x86, A100 MIG 10 GB): 10/5까지. r3 학습, 3-arm 실험, NCNN export를 여기서 끝냈다.
- NCNN export에는 `pnnx`가 필요하다. aarch64 휠이 없으면 GB10에서는 새 모델을 export할 수 없으니 **가장 먼저 확인**한다(§7-A).

## 1. 과제

타겟(공 + 막대 + 받침, 현재는 공에만 색지)을 검출한다(YOLO11n, 클래스 1개 `tar`) → 색으로 아군(계획: 청록 #14A3A1) / 적군(보라 #7E4FC4)을 판정한다 → 박스 중심 x로 yaw, 박스 높이 h로 피치 보정표를 읽는다. 교전 거리 1.5~2.0 m, 타겟은 정지.

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
| `runs/r3/weights/best.pt` | r2에서 시작, 전체 train 1,423장, 입력 1280, `hsv_h=0.3` | **현재 배포 후보**. NCNN은 `weights/ncnn_r3/` |
| `runs/3arm/<arm>_<색>/weights/best.pt` | yolo11n.pt(COCO)에서 시작, 한 색만 학습 | 색 불변성 실험(§6) |

예전 run(det640, crop640, 해상도 스윕)은 `results.csv`와 `args.yaml`만 남겼다. 밴드 크롭(crop640) 전략은 폐기했다.

## 4. 확정 측정값

**박스와 기하**
- b1 박스 중앙값 25×43 px. b2 거리별 h 중앙값: 1.2 m 47.5 / 1.5 m 38.7 / 1.8 m 32.8 / 2.1 m 29.3 / 2.4 m 26.0 px → **h ≈ 59.1/d(m)**, 잔차 중앙값 1.5 px. 절편을 허용하면 h = 53.7/d + 3.6. 2.1~2.4 m에서 +1.3~1.6 px이 남는 원인은 미확정(받침의 고정 픽셀인지 의사 라벨 편향인지 → 사람 라벨로 판별).
- 배치(center/right)에 따른 h 차이는 2.3 px 이하 → cx 보정 불필요. 두 묶음 합산 타겟 y 범위 389~632.
- **yaw**: b2에서 d(cx)/d(yaw_cmd) = **+10.3 px/°**(세션별 7.8~11.6). yaw_cmd가 커지면 타겟이 화면 오른쪽으로 간다. `tools/pipeline_config.json`은 `coef [0, 64]`, **`sign: -1`**(카메라 서보 명령 기준: 화면 오른쪽 타겟을 중앙에 두려면 yaw_cmd를 줄인다). 발사부의 축과 방향 규약은 실기에서 확인해야 한다.

**색 (현재 노출, 공 영역의 채도 > 0.24 픽셀)**
- 적군: hue 234°, S .44, **V .27**, Lab (22.5, +6.7, −5.1). 아군: hue 74°, S .39, **V .26**, Lab (27.6, −1.6, +7.1).
- 클래스 간 dE2000 16.1, 클래스 내 RMS 10.7 / 10.2 → **안전비 1.51 (기준 3.0 미달)**. 색지 공칭색끼리의 dE는 41.0이다. 2.4 m에서 고채도 픽셀은 적군 27개 / 아군 63개뿐이다.
- 지금 노출에서는 색 판정이 작동하지 않는다. 모의 판정은 100% unknown(= 전부 보류)이다.

**검출기와 색**
- ultralytics 8.4의 `hsv_h`는 **덧셈**이다: `lut_hue = (x + r·180) % 180`, r ∈ [−h, h] → 0.015 = ±5.4°, 0.1 = ±36°, 0.3 = ±108°. 공칭색의 색상 차는 약 85°라 0.1로는 못 덮는다.
- **r2는 색맹이 아니다.** 아군 637장 recall이 컬러 .986 → **흑백 .854**, 적군 b2는 흑백에서 .906이다(입력 640). 색상(hue)만 돌린 시험에서는 검출률이 유지됐는데(회전 없음 94.9%, +44° 95.9%, +86° 77.9%), 채도와 명도가 그대로였기 때문으로 보인다(해석). 즉 **색지를 바꾸면 검출 성능이 떨어질 수 있다.** 색 분리가 실패해 색을 바꿔야 할 가능성이 있으므로 이것은 직접적인 리스크이고, 3-arm 실험(§6)의 목적이 여기에 있다.
- r2의 b2 개선(미검출 336 → 18)은 `hsv_h`가 아니라 b2 494장을 학습에 넣었기 때문이다.

**해상도와 지연**
- b1 해상도 스윕(val mAP50-95): 160 .343 / 224 .597 / 320 .690 / 416 .732 / 512 .752 / 640 .765.
- 지연 비율(x86 NCNN FP32 4스레드): 640×480 29 ms = 1.0 / 960×736 2.59× / **1280×960 4.88×** / 384² ROI 0.47×. Pi4 추정: 1280×960 ≈ 1.3~1.5 s, ROI ≈ 0.15 s. **파이 실측 전까지 추정이다.**

## 5. 확정 결정

1. 입력 = **전체 1280×960**(획득) + **원본 해상도 ROI 384×384**(추적, 같은 모델). 닫힌 루프 5~10프레임 → 전체 1회 + ROI 9회 ≈ 2.8 s(추정). 밴드 크롭·폴백·dual 학습은 폐기. 밴드 y 352~672는 conf 감점(×0.5)으로만 쓴다.
2. 색 판정 = 박스 안 **고채도 픽셀**의 Lab 평균 → 기준색과 dE2000. 게이트: 픽셀 수 < min_px(임시 20) → unknown(`few_px`), |dE_적 − dE_아| < margin → unknown(`ambiguous`). **unknown이 하나라도 있으면 보류**, 적군이 여러 개면 **h가 가장 큰(가까운) 것**. 기준색·margin·min_px는 색칠 후 `tools/color_separation.py`가 만든 `color_refs.json`으로만 정한다.
3. **색 안전 게이트**: `ColorClassifier`는 `color_refs.json`이 없거나 그 안의 안전비가 3.0 미만이면 기준색을 쓰지 않고 전부 unknown으로 둔다. 시운전 산출물(안전비 1.51)이 실수로 연결되는 것을 막는다.
4. 노출: **노출 시간 우선**(정지 타겟이라 블러 비용이 없다), 게인은 노출을 더 못 올릴 때만. 조건 = 타겟 패치 V ≥ 0.5(S ≥ 0.4). 벽 포화는 허용하되 포화 영역이 30%를 넘으면 경고.
5. 라벨: b2 809장과 아군 662장의 의사 라벨을 학습에 쓴다. 아군은 `review_list.csv` 149장을 사람이 검수한다. val/test로 쓸 세션은 LabelImg 전수 검수 후에 쓴다. test 후보: center_1.8m + **right_1.5m**(교전 거리 기준) + 아군·both의 같은 거리대.
6. 1차 지표: 박스 **중심 오차 px**(yaw) + **높이 오차 px**(피치) + recall@conf .25 + 음성 FP/프레임. mAP50은 포화라 쓰지 않는다. `tools/eval_cross_color.py`가 이 지표를 낸다.
7. 3-arm 실험: `hsv_h` 0.015 / 0.3 / 흑백 × (적군만 학습 → 아군 검출률, 아군만 학습 → 적군 검출률). 색지를 바꿔도 재학습이 필요 없는 설정을 고른다. 최종 검출기는 색맹이어야 한다.
8. 촬영 계획(색칠·노출 수정 후, 하루 1회, note 이름 고정):
   - P0: `painted_both_EL-FR_{1.5,1.75,2.0}m` + `painted_both_FL-ER_{…}` 각 40장 + `painted_negative_bin` 10장 → `color_separation.py` 통과 여부 확인
   - P1: `painted_{left,center,right}_{1.5,1.75,2.0}m` 적군·아군 각 9 × 40장
   - P2: `painted_both_Enear1.5-Ffar2.0_{left,right}` / `Fnear…` 4 × 40장, `painted_negative_{empty,people,chairs}` 3 × 30장, `painted_static_{enemy,friend}_1.75m` 15장
   - **labels.csv와 sessions.jsonl을 함께 올린다.**

## 6. DGX에서 끝낸 것 (2026-10-03)

- 데이터셋 패키징과 5 epoch 검증(r2에서 시작, 1280, batch 8): 스캔 1,423 + 228장·배경 0·손상 0, box loss .92 → .60, val mAP50 .995, mAP50-95 .78 → .82, GPU_mem 4.5~4.7 G, 약 41 s/epoch.
- `tools/target_pipeline.py` 작성과 확인: ROI 모드와 전체 프레임의 박스 중심 차이 1.3 px 이하, 밴드 감점·색 게이트·교전 결정 동작 확인.
- `color_separation.py`(`--labels` 경로)와 `pseudo_bias_check.py`(단일 세트) 시운전: §4의 수치를 재현했다.

<!-- RESULTS:BEGIN -->
### 학습 결과

r3와 3-arm은 이 문서를 쓰는 시점에 DGX에서 학습 중이다. 결과가 이 압축본에 들어 있는지는 `runs/r3/eval_summary.csv`와 `runs/3arm/table.txt`가 있는지로 확인한다.
<!-- RESULTS:END -->

## 7. 다음 할 일

### A. GB10 환경 (클로드 코드)
1. [README_FIRST.md](README_FIRST.md) 순서로 풀고 세팅한 뒤 `bash scripts/setup_gb10.sh`.
2. `pip install ncnn pnnx`가 되는지 확인한다. 안 되면 새 모델의 NCNN export 수단이 없다는 뜻이므로 사용자에게 바로 알린다(x86 PC에서 export하는 방법이 대안).
3. `EPOCHS=2 NAME=probe bash scripts/train_r3.sh`로 GB10의 GPU_mem과 s/epoch를 잰다.

### B. 사용자가 실물로 해야 하는 것 (이 순서)
1. **아군 색지가 실제로 무슨 색인지 확인.** 계획은 청록 #14A3A1(Lab 60.8, −34.0, −9.0, hue 179°)인데 측정값은 hue 74°, Lab (27.6, −1.6, +7.1)이다. V가 0.26이라 색상을 믿을 수 없는 상태이긴 하지만, a\*·b\*의 방향(Lab 색상각 103°)이 청록(195°)과 다르다. 적군은 측정 322°, 공칭 보라 309°로 방향이 맞는다. 화면에서 아군 타겟은 거의 검정으로 보인다.
2. 노출 수정(§5-4) → 받침·막대 색칠 → P0 촬영(§5-8).
3. 아군 세션 메타(labels.csv, sessions.jsonl) 업로드.
4. 파이4에서 `tools/pi_bench.py`로 지연 실측(`weights/ncnn_r3/`와 프레임 1장을 파이로 복사).
5. 실기에서 yaw 부호와 피치 보정표(h → 각도) 측정. 색칠하면 외형이 바뀌므로 **h ≈ 59.1/d 계수와 피치 보정표는 색칠 후에 다시 잰다.**

### C. 사람 검수 반영
1. LabelImg로 `data2/friend(26.10.02.1839)`(이미지)와 `friend_pseudo/xml`(XML)을 열고 `friend_pseudo/review_list.csv`의 149장을 본다. 198·205번은 박스가 하나만 남아 있는지 확인한다.
2. `python tools/xml2yolo.py --xml friend_pseudo/xml --labels friend_pseudo/labels --only friend_pseudo/review_list.csv` (먼저 `--dry-run`으로 바뀐 장수를 본다).
3. b2의 val 세션(center_2.4m 2개, right_1.5m)과 test 후보 세션도 같은 방식으로 `data2_labels/xml` → `data2_labels/labels`.
4. `python tools/package_dataset.py` → sanity 통과 확인 → `NAME=r4 bash scripts/train_r3.sh`.
5. 사람 라벨이 생기면 `tools/pseudo_bias_check.py --set pseudo:… --set human:…`로 §4의 원거리 +1.3~1.6 px이 라벨 편향인지 판별한다(이 비교 경로는 아직 돌려 본 적이 없다).

### D. 아군 메타가 오면
`python tools/package_dataset.py --friend-meta "data2/friend(26.10.02.1839)/labels.csv" --friend-val-notes <val로 쓸 note들>` → uid가 `fr_<세션>_<idx>`로 바뀌고 val이 실제 세션 단위가 된다. note 이름을 보고 `--friend-val-notes`를 정한다(기본값 `right_1.5m left_2.4m`은 추측이다).

### E. P0 촬영분이 오면
1. `python tools/seed_label.py --images <dir> --model runs/r3/weights/best.pt --out <out> --imgsz 1280 --max-expected 2` 로 사전 라벨.
2. `python tools/color_separation.py --dir <dir> --model runs/r3/weights/best.pt --out color_report` → 안전비 ≥ 3.0이면 `color_report/color_refs.json`을 `tools/color_refs.json`으로 복사. `--model` 자동 검출 경로와 both 세션의 EL/FL 좌우 배정은 **아직 돌려 본 적이 없으니** 결과의 `samples.csv`를 눈으로 확인한다.
3. 미달이면 색을 바꾸거나 노출을 다시 조정한다. 색을 바꾸면 3-arm 결과에 따라 검출기 재학습 여부를 정한다.

## 8. 리스크

1. **색 판정은 노출 수정 전까지 작동하지 않는다**(안전비 1.51). 노출 수정 → 색칠 → P0 통과가 선결.
2. **아군 색지의 실제 색이 미확인**이다(§7-B-1).
3. **검출기가 색에 의존한다**(§4). 색지를 바꾸면 검출 성능이 떨어질 수 있다. 3-arm 결과로 대응 설정을 고른다.
4. 받침·막대 색칠 후 외형이 바뀐다 → h 계수와 피치 보정표 재측정, 색칠 후 데이터로 검출기 재확인(라벨 규약은 공 + 막대 + 받침 유지).
5. val이 의사 라벨이고, 사람 라벨 test는 방 A·적군색뿐이다. 홀 B와 아군에 대한 사람 라벨 수치는 아직 없다.
6. 아군 메타 부재(세션·거리가 추정치), 우측 2.4 m는 11장뿐이다.
7. GB10: 공용 계정의 설정·캐시 오염과 타인 프로세스 종료 사고, pnnx aarch64 휠 유무 미확인, 이 프로젝트를 GB10에서 실제로 돌려 본 적 없음.

## 9. 환경 메모

- GB10: torch를 먼저 설치하고 그다음 ultralytics(순서가 바뀌면 torch가 갈아엎어진다). **pandas는 따로 설치한다**(ultralytics 8.4가 더 이상 같이 설치하지 않는다). `scripts/env.sh`가 캐시와 ultralytics 설정을 프로젝트 폴더 안으로 돌린다.
- 학습 설정 참고(DGX 실측): 1280 입력 batch 8에서 GPU_mem 4.5~4.7 G, `cache=ram` 약 5.7 GB. ultralytics `optimizer=auto`는 이 규모에서 AdamW lr 0.002를 고른다.
- DGX에서 썼던 환경: python 3.11, torch 2.7.1+cu128, ultralytics 8.4.171, opencv-python-headless 4.11, numpy 1.26, pandas 2.0, ncnn·pnnx 20260526.
