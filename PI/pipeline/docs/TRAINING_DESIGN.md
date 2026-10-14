# 로봇 경기 타겟 검출기 최종 학습 설계 문서 (기존 180장 + 신규 830장 기준, 나머지 ~1170장 소재 확인 중)

> 작성 기준: ultralytics 8.4.171, A100 MIG 1g.10gb(학습), Raspberry Pi 4 4GB Cortex-A72 x4(배포). 모든 수치는 이 세션 측정값(FACTS, results.csv, sweep.log, `/data/YOLO/data2/enemy(26.10.02.1754)/labels.csv`·`sessions.jsonl`) 또는 리서치 출처에서 가져왔으며, 측정되지 않은 값은 **(추정)** 또는 **(가정)** 으로 표시한다. 신규 830장은 박스 GT가 없으므로 det640 best.pt(CPU, conf 0.1) **의사박스 기준 수치는 "참고치(FP 오염 가능)"** 로 표기하고 GT 라벨 후 재측정한다.
>
> 데이터 배치 표기: **b1** = 기존 180장(비디오 프레임, /data/YOLO/data) + 테스트 8장(/data/YOLO/test), **b2** = 신규 830장(서보 스텝 정지 촬영, /data/YOLO/data2/enemy(26.10.02.1754)). 사용자가 말한 "약 2000장" 중 서버에서 확인된 것은 b2 830장뿐이며 나머지 ~1170장의 소재는 확인 필요(K1).

## ⚠️ 최신 사실 반영 (2026-10-02, 이 블록이 본문보다 우선)

본문은 설계 워크플로 시점의 정보로 작성됐다. 그 뒤 data2 자동 라벨링 과정에서 아래가 확인되어 본문 일부 전제가 바뀌었다.

| 본문 전제 | 실제 (확인 방법) | 영향 |
|---|---|---|
| b2 830장은 박스 라벨 없음 | **809장 라벨 완료** (`/data/YOLO/data2_labels/labels`, YOLO txt + LabelImg xml). 2라운드 자기학습: det640 → b2 확실 검출 494장 추가 학습(r2) → 재라벨. 검토 필요 49장 **전수 육안 확인**, 신규 확실 무작위 35장 육안 확인, 모두 정확 | C-5의 "R0 수동 ≈490장 / ≈8h"는 **불필요**. 대신 809장을 LabelImg로 세션별 샘플 검수(≈1~2h 추정) |
| idx 88~357의 파란 물체 정체 미확인(K5) | **파란 공 + 파란 받침이 b2의 타겟** (확대 육안 확인). b1 타겟은 어두운/보라 공 + 나무 받침 | K5 해소. 클래스 'tar' 하나로 두 외형을 모두 학습. 색 차이 때문에 `hsv_h=0.015`로는 부족 → r2는 `hsv_h=0.05`로 학습해 검출 없음 336→18 |
| b2는 b1과 같은 방 | **다른 장소** (육안: b2는 양문·좌석 공간이 있는 홀, b1은 단문·자판기가 있는 방). 153623 세션의 분홍/보라 색감은 WB 실패 | "방 간 일반화 미측정"이 아니라 **측정 가능**: b1(방 A) ↔ b2(홀 B) 교차 평가를 F에 추가 |
| 밴드 y[384:704]는 b1 한정, 재산정 필요 | 합산 라벨 기준 박스 y **389~632** (b1 448~632, b2 389~577), pitch 96°에서 y2 최대 577 | 밴드 **y[352:672]** (높이 320 유지, 상하 여유 37/40px)로 확정. 본문 D의 예상과 일치 |
| 학습 서버 = A100 MIG 10GB | GB10(통합 메모리 121GiB, aarch64, CUDA 13) 사용 예정 (교수님 허락 대기) | E의 시간 추정은 MIG 기준. GB10에서 E0 에폭 시간 실측 후 재추정. `cache=ram`, batch 확대 여유 |

b2 라벨 거리별 박스 높이 중앙값(px): 1.2m 47.5 / 1.5m 38.7 / 1.8m 32.8 / 2.1m 29.3 / 2.4m 26.0.
b2 직접 확인 필요 21장: `data2_labels/check_list.csv` (검출 없음 18장 — 타겟 화면 밖이면 네거티브로 사용, 손으로 타겟 놓는 장면 3장 — 제외).
**주의**: 809장은 사람 라벨이 아니라 검증된 의사 라벨이다. F의 "val/test는 사람 라벨만" 원칙에 따라 val/test로 쓸 세션(본문 제안: center_2.4m·right_1.8m / center_1.8m·right_2.1m)은 LabelImg로 **전수 검수**한 뒤 사용한다.

---

## 핵심 결정 (10줄)

1. **입력 = 밴드 크롭** `imgsz=640 rect=True`(실제 640x160). 근거: crop640 mAP50-95 .802, CPU 31.3ms vs full 640² 73.2ms. 단 "밴드가 더 정밀"은 미확정(det640 ep161 조기중단; crop640 ep≤160 최고 .7748 ≈ det640 학습 중 best .7738) → 확정 이점은 **속도 2~3배**. 밴드 범위 `y[384:704]`는 b1 한정 사실이며 b2 의사박스 y1 최소 388.9 → **두 배치 합산 GT로 재산정 후 확정**(예상 y≈[352:672] 또는 더 넓게, D 참조).
2. **레짐 고정**: `optimizer=AdamW lr0=0.002 warmup_bias_lr=0.0 patience=0`(측정 run의 실제 레짐을 명시). b1+b2 ≈730 train에서도 auto는 AdamW 유지(iterations ≈2400 <10000).
3. **1차 지표 교체**: 원본 좌표 중심 오차 px(median/p95, x·y 분리) + recall@운용 conf + 네거티브 FP/프레임 + **b2 세션 내 cx~yaw_cmd 선형성 잔차(px)**. mAP50은 .995 포화라 미사용, mAP50-95는 보조. yaw 라벨 q=2°라 end-to-end yaw MAE로는 3px급 개선이 보이지 않음(1px≈0.1° 실측).
4. **분할**: 단위 계층 room/day(1) > camera_config(3) > placement×distance(11) > capture session(19). val/test는 **placement×distance 단위 held-out**, 사람 라벨만, 테스트 봉인(체크섬, ≤3회). b1은 전부 train(기존 42 val·테스트 8장 승격 폐기), "방 간 일반화 미측정" 보고서 명시.
5. **파일명 전역 유일화**: b2는 enemy_000001~000830으로 b1(296~554, 624~647)과 이름이 충돌(md5 상이) → 병합 전 `b1_<old>`, `b2_<session>_<idx>` 접두사 부여, inventory 키 = (batch, session, idx).
6. **라벨 규약 고정** + 50장 2인 재라벨로 라벨 노이즈 하한 측정. presence는 b2 기하 예측 + 사람 확인으로 채움(labels.csv 830장 전부 label=enemy이나 화각 밖 프레임 포함). 네거티브 10~15%(presence 확인된 프레임만).
7. **yaw**: YawCNN 폐기 → 박스 중심→기하 모델/단조 LUT, **세션 고정효과 b_s + 공통 광학 모델 f(cx)** 로 적합(b2는 세션마다 타겟 bearing이 미지 상수). τ(동기 지연)는 b1 비디오 배치에만 해당, b2는 정지 촬영이라 서보 백래시·명령각-실제각 오차가 지배. 역방향 촬영 요청은 b2가 양방향 스텝을 포함하므로 **불필요**.
8. **Pi4**: 학습 동일 기하 → NCNN FP32 `imgsz=160,640` → 역변환. FP16 이득 없음, INT8은 수동·조건부. 패리티는 자체 스크립트. 카메라는 Picamera2/libcamera CSI, camera_config.json 잠금(노출 15.1~15.5ms 고정).
9. **실험 ~12 run 상한**, 한 변수씩, 상위만 3 seed, 채택 ≥2σ. P2·yolo11s는 Pi4 실측 전 배포 후보 제외.
10. **224 리사이즈 후 YOLO 경로 완전 폐기**.

---

## (A) 목표·제약·성공 기준

| 항목 | 내용 |
|---|---|
| 최종 목표 | Pi4에서 타겟(b1: 보라 공+나무받침 / b2: 파란 공+파란 받침) 검출 → 박스 중심 → yaw 각도. 최종 지표는 **yaw 오차** |
| 학습 제약 | A100 MIG 1g.10gb 1장, 순차 실행. 데이터: b1 180장(박스 GT 있음) + b2 830장(의사 라벨 809장 완료 — 최신 사실 블록 참조, yaw_cmd/pitch_cmd 있음) + 소재 미확인 ~1170장 |
| 배포 제약 | Pi4 4GB, Cortex-A72 x4 1.5GHz, NPU/GPU 없음, FEAT_FP16·SDOT 없음 |
| 데이터 특성 b1(측정) | 객체 w 15~32(med 25), h 26~53(med 43)px @1280x960. 박스 y 448~632, cx 1~822. 이미지당 객체 1개, 배경-only 0장 |
| 데이터 특성 b2(측정) | 830장 1280x960, 19 캡처 세션, 같은 방·같은 날(2026-10-02 15:36~17:40)·같은 마운트. 배치×거리 11종(center/right × 1.2/1.5/1.8/2.1/2.4m), yaw_cmd 65~125(중심 95, 2° 스텝, 31개 고유값), pitch_cmd 90~98(≠90인 42장: 3세션). 카메라 설정 3종(colour_gains 1.29/1.63·1.03/1.76·1.17/1.76). 의사박스 참고치: cy 403~556(med 424), right_* 세션 cx 807~1266(cx>822 217건), 거리별 h med 1.2m≈46~48 / 1.5m≈39 / 1.8m≈32~33 / 2.1m≈28~29 / 2.4m≈26px(w≈14) |

**성공 기준(수치)** — 모든 값은 봉인 테스트(사람 라벨)에서 측정:

| 지표 | 목표 | 근거/비고 |
|---|---|---|
| 중심 오차 median (1280 좌표, cx) | ≤ 3px | 현재 테스트 8장 ≈5px은 근사 GT 대비 상한. 25x43 박스 1px=IoU .883 → 라벨 노이즈 하한(2인 재라벨) 이하로는 불가 |
| 중심 오차 p95 | ≤ 8px | (추정 목표) |
| recall@conf 0.25 (+기하 게이트) | ≥ 0.98, **2.4m 세션 단독 recall도 ≥ 0.95** | 파이프라인은 미검출 시 샘플 폐기. 2.4m 타겟은 0.5배 입력에서 7x13px(의사박스 참고치) |
| 네거티브 프레임 FP/프레임 @conf 0.25 | ≤ 0.02 | 기준선 미측정. b2 화각 밖 프레임(presence=none 확인분)이 첫 네거티브 셋 |
| export 모델 vs PyTorch 패리티 | 중심 오차 차 <0.5px@640x160, 미검출 증가 0건 | |
| b2 세션 내 cx~yaw_cmd 선형성 잔차 | median ≤ 3px(≈0.3°) | 1차 yaw 품질 지표. 실측 dyaw/dx ≈10px/°(right_1.5~2.4m 의사박스 선형 적합, 잔차 7~10px — GT 후 재측정) |
| end-to-end yaw MAE | 라벨 양자화 q=2° → 하한 ≈ q/4 = 0.5°. 목표는 "하한 + 서보 명령각 오차(가정 ±1~2°)" 수준으로만 서술 | 명령값-실제각 오차는 미측정(K2). b1만 1프레임 동기 오차 ≈44.6px ≈4.5°(실측 10px/° 적용) 리스크 |
| Pi4 지연 | 목표 FPS는 사용자 확인 필요(K 참조). 640x160 NCNN FP32 ≈100~150ms **(추정)** | 공식 Pi4 YOLOv8n 640² NCNN 414.73ms를 픽셀 비례 환산 |

---

## (B) 현재 상태 요약 (실측)

### B-1. 학습 결과 (yolo11n.pt, batch 16, 실제 레짐 AdamW 0.002)

| run | 입력 | epochs/patience | 완료 ep | best ep | mAP50 | mAP50-95(best) | 마지막 50ep std | A100 ms/img | x86 CPU ms |
|---|---|---|---|---|---|---|---|---|---|
| sweep160 | full→160 | 150/40 | 150 | 146 | .734 | .343 (R .719) | .023 | – | – |
| sweep224 | full→224 | 150/40 | 150 | 136 | .995 | .597 | .016 | – | 23.5 |
| sweep320 | full→320 | 150/40 | **102(조기중단)** | 62 | .995 | .690 | .024 | – | 35.3 |
| sweep416 | full→416 | 150/40 | 148 | 108 | .995 | .732 | .014 | – | 45.5 |
| sweep512 | full→512 | 150/40 | 150 | 124 | .995 | .752 | .012 | – | 59.1 |
| det640 | full→640 | 300/60 | **161(조기중단)** | 101 | .995 | **.765 (별도 yolo val, FACTS)** / 학습 중 EMA val best .7738, last .7458 | .015 | 11.2 | 73.2 |
| crop640 | 밴드 1280x320→640x160 | 300/60 | 300 | 245 | .995 | .802 (last .799) | .009 | 3.8 | 31.3 |

- det640 두 값의 출처: `.765`는 FACTS의 별도 `yolo val` 측정값(이 문서의 기준값), `.7738`은 results.csv 학습 중 val best. 이하 본문에서 det640 비교 기준은 **.765**를 쓴다.
- crop640 궤적: ep100 .719 → ep150 .759 → ep200 .768 → ep245 .802. **ep≤160 최고 .7748**(det640 학습 중 best .7738과 동일) → .802 vs .765 차이는 "저-lr 구간 완주 여부"로 교란됨.
- 해상도 결론: **160 부근 절벽(recall .719)**, 224 이상은 단조 증가 경향. 512→640 증가폭 .013(기울기 둔화). 점별 차이는 예산(150 vs 300ep)·조기중단으로 오염되어 과해석 금지.
- args.yaml에는 `optimizer: auto, lr0: 0.01, warmup_bias_lr: 0.1`로 기록되지만 실제 적용은 AdamW lr0=0.002(results.csv lr/pg0 max .00198), warmup_bias_lr=0.0(trainer가 강제).
- crop640은 `rect: true` → **mosaic/mixup/cutmix 0으로 학습됨**(dataset.py). 밴드는 모두 같은 shape라 shuffle은 유지.
- 에폭 시간(results.csv time 열): crop640 median **1.48s/epoch**(총 8.5분/300ep, 9 iters), det640 2.1s/epoch.
- val 42장: 모든 설정 mAP50 .995 포화, run 내 mAP50-95 변동 σ .009~.024 → 단일 run 비교는 노이즈.

### B-2. 데이터·파이프라인(측정)

| 항목 | 값 |
|---|---|
| b1 라벨 데이터 | 180장 1280x960 VOC, 프레임 296~554, 연속 세그먼트 23(FACTS)/25(재집계 기준 차이), 길이 1~17 |
| b1 인접 프레임 중심 이동 | med 44.6px, p90 50.6, max 134.9. **Δcx 155/155 양수**(단방향 등속 회전), 세그먼트 내 cy 범위 ≤26px(1개 예외). τ 식별 불가는 **b1 한정** |
| b1 밝기 | 이미지 평균 103.5~137.1, 타겟 패치 70~105(항상 ≥24 어두움) |
| b1 h/w | min 1.47, p5 1.53, med 1.68, p95 2.04, max 2.58 |
| 테스트 8장(b1) | enemy_000624~647, 같은 방, 근사 GT만. det640/crop640 8/8 검출, conf .63~.92 |
| **b2 신규 데이터** | `/data/YOLO/data2/enemy(26.10.02.1754)/images/*.jpg` 830장(전부 1280x960) + `labels.csv`(filename,label,session,mode,note,yaw_cmd,pitch_cmd,timestamp,clip_ratio,dark_ratio,mean_bgr,config_locked) + `sessions.jsonl`(19 세션 start/end, step_deg 2.0, settle_sec 0.4, locked_settings). **박스 라벨 없음**, label 전부 'enemy', presence 태그 없음 |
| b2 촬영 방식 | SG90 팬틸트 서보 2° 스텝 → settle 0.4s → 정지 촬영. `note_on_angles: "All saved angles are command values. The SG90 cannot read its real angle."` 세션 안에서 타겟 고정, 카메라만 회전. 모든 세션에 +/− 스텝 혼재(예 170657 pos 14/neg 30, right_* 세션 ≈20/20) |
| b2 세션 구성 | center_1.2(5세션 220장, 그중 153623은 WB 실패 핑크 캐스트 mean_bgr≈199·clip 0.036, 163106은 gains 1.03/1.76 mean≈177), center_1.5/1.8/2.1/2.4(각 2세션 92/108/98/103장), right_1.2(2세션 43장, 173136은 4장 중단), right_1.5/1.8/2.1/2.4(각 1세션 41/42/41/42장). 161753 세션은 0장 |
| b2 밝기 | mean_bgr 124.8~200.7(정상 세션 med 128~140), 노출 고정(exposure_time 15.1~15.5ms, analogue_gain 2.0~2.25, lens_position null) |
| b2 의사박스 참고치(det640, conf 0.1) | 검출 585/830(conf≥0.25는 494). 세션 170657~171540(idx 88~357, 270장)에서 14/45, 28/45, 22/45, 5/45, 0/45, 1/45로 급락 — 샘플 이미지 100·290의 바닥 물체가 보라가 아닌 **파란 계열**로 보임(370·10·800은 보라). 같은 yaw_cmd의 상승/하강 cx 차 median ≈5.6px(백래시 참고치). 박스 ≥2개 프레임 20/830(배경 사람·보라색 의류 포함) |
| 파일명 충돌 | b2 enemy_000296/000400/000554/000624/000647은 b1 동명 파일과 **md5 상이**(다른 이미지). 번호 겹침은 같은 비디오의 증거가 아님 |
| 기존 모델 yolo_target.pt | 224 경로 recall 0/180, 640 0/180, 1280 29/180 → 폐기 |
| YawCNN | 1280x960→224x224 비등방 리사이즈, np.rint 마스크, 4-conv 228,353 파라미터(≈180M MAC), Sigmoid+min-max |
| 환경 | ultralytics 8.4.171: `yolo11-p2.yaml` 없음, NCNN은 INT8_FORMATS에 없음, train imgsz는 int만, fraction은 정렬 앞부분 절단, patience=0이 정식 비활성 |

---

## (C) 데이터 계획

### C-1. 인벤토리(모든 작업의 선행 조건, CPU 스크립트)

입력: b1 `/data/YOLO/data/images + labellmg/*.xml`, b1 테스트 8장, b2 `labels.csv + sessions.jsonl`. 출력: `inventory.csv` (uid, batch{b1/b2}, orig_filename, session, idx, placement{center/right}, distance_m, camera_config_id{1.29/1.63, 1.03/1.76, 1.17/1.76}, yaw_cmd, pitch_cmd, timestamp, mean_bgr, clip_ratio, 해상도, label_source{human/pseudo/none}, presence{target/none/unknown}, cx, cy, w, h).

규칙:
- **전역 유일 이름**: 병합 전 `b1_enemy_000296.jpg`, `b2_171112_000100.jpg`(= b2_<session 시각 6자리>_<idx>)로 복사. inventory 키 = (batch, session, idx). 원본 파일명은 orig_filename 열에만 보존. ds2_crop/ds2_full 생성 스크립트는 uid만 사용(동명 덮어쓰기·라벨 짝 뒤바뀜 방지).
- b2 session/note 파싱: note `center_1.2`/`center_1.2m` 표기 혼재 → `m` 제거 후 placement·distance 분리. sessions.jsonl의 locked_settings.colour_gains로 camera_config_id 부여. 173136(4장)·161753(0장)은 train 전용/제외로 표기.
- b2 presence 초기값: (a) 세션별 bearing 추정(의사박스가 있는 프레임의 cx~yaw_cmd 선형 적합 → 화각 내/외 예측) → (b) 사람 1차 확인(C-5 R0에 포함). 사람 확인 전에는 unknown 유지, 확인 후에만 none에 빈 .txt 부여.
- b2 label_source = none으로 시작, det640 의사박스는 별도 `pseudo_cx/cy/w/h/conf` 열(QA·presence 예측용, 학습 라벨 아님).
- "같은 비디오 여부" 분기(b1 프레임 번호 기반)는 **b1에만** 적용. b2는 번호가 독립(1~830)이라 적용 금지. 추가 ~1170장이 들어오면 배치 b3로 같은 절차.
- 박스 y 분포 재산정: 두 배치 합산 GT **p0.5~p99.5 ±48px**, 높이 32 배수(D 참조).

파일에서 이미 답한 항목 vs 여전히 확인 필요한 항목은 K1·K2에 분리.

### C-2. 촬영/선별 (추가 녹화가 가능한 경우)

| 축 | 목표 | 비고 |
|---|---|---|
| 세션 | 방 ≥3곳(대회장 유사 바닥 포함), 조명 ≥3조건, 날짜 ≥2 | b2는 방 1·날짜 1·마운트 1이라 다양성 축이 카메라 설정 3종·배치×거리 11종뿐. 세션당 상한 180장 |
| 거리 | **요구 교전 거리(K3) + 20%까지**. b2는 1.2~2.4m(h≈26~48px 참고치)만 있음 → 2.7/3.0m 세션 추가, 원거리(h<30) 15% 내외 | h<20px(640에서 10px 미만)은 정밀도 바닥 영역 → 소량 5% 검출 여부만. 2.4m 초과 데이터 현재 0장 |
| 밴드 밖 | 유도 세션(피치·단상)은 train 전용. **자연 주행 세션 2개를 따로 지정해 밴드 밖 발생률 측정**. b2 pitch_cmd 90~98 세션은 유도 세션으로 분류 | 유도 세션에서 재면 발생률이 아님 |
| 네거티브 | 전체의 10~15%, 그중 ≥50%는 혼동물체(빗자루 손잡이·파란 분리수거함·자판기·아군 색 공·**사람·보라색 의류·가방**) 앞 타겟 없는 영상 | 타겟 치우고 같은 궤적 재녹화, 빈 .txt. b2 화각 밖 프레임(presence=none 확인분)은 비용 0 네거티브 |
| 다중 타겟 | ~10% | 최고 conf 1개 규칙의 실패 모드 확인용 |
| 가로 위치 | b2 right_* 세션으로 cx 807~1266(참고치)이 이미 확보 → 추가 유도 불필요 | fliplr=0.5가 좌우를 이미 보완하므로 근거는 "라벨 예산 절약" |
| 촬영 방식 | b2 방식(서보 스텝 정지 촬영)은 모션 블러 0 → 배포(로봇 회전 중 15ms 노출)와 다름. **회전 중 연속 촬영 세션 1~2개**(타임스탬프 포함) 추가 권장 | 블러량은 회전 속도(°/s, K4)에 좌우 |

프레임 선별(b1·비디오 배치만): 원본 fps 확인 후 stride 결정(미확인). 2차 규칙(중심 이동 <25px AND 높이 변화 <15% AND dHash 해밍 <6 → 버림), 세그먼트당 ≤40장, 세션당 ≤180장 균등 stride. b2는 2° 스텝(≈20px 이동, 참고치)이라 선별 없이 전량 사용.

### C-3. 라벨링 규약 (문서 1장, 이후 변경 금지)

- 박스 = **공 + 막대 + 나무받침 전체**, 바닥 반사 제외, 받침 하단 = 바닥 접선, 공 상단 = 공 끝. 근거: 현재 180장이 이 규약(h/w med 1.68)이고 하류 분포를 흔들지 않음.
- 잘림: 보이는 부분만 + truncated=1. 블러: 윤곽 보이면 라벨 + difficult=1, 사람이 못 알아보면 프레임 폐기(가짜 네거티브 방지).
- 아군 색 공은 라벨 없음(클래스 'tar' 1개 유지). **b2 idx 88~357의 파란 계열 물체는 K5 답변 전까지 '보류' 태그**(적 타겟이면 라벨, 아군/다른 물체면 네거티브 후보, 조명/WB 차이면 라벨). 포맷 LabelImg VOC, 변환 스크립트 1개만 사용.
- presence 열: 라벨러가 프레임마다 target/none을 명시(b2 화각 밖 프레임 포함). 빈 .txt는 presence=none 사람 확인분만.
- **(선택, yaw 단계용)** 공 중심 클릭 1회 포인트 라벨을 전량 부여(이미지당 1클릭, 1~2h 추정).

### C-4. 라벨 QA

| 규칙 | 임계 |
|---|---|
| 자동 | h/w ∉ [1.3, 2.6], sqrt(면적) ∉ [10, 100], 이미지-XML 짝 불일치, size 태그 ≠ 1280x960, presence=target 파일에 박스 0개 |
| 궤적(b1 비디오) | 프레임 간격 정규화 px/frame > 60 플래그(절대 150px 규칙 폐기) |
| 궤적(b2 서보 스텝) | 세션별 cx~yaw_cmd **단조(isotonic) 적합 잔차 >15px(≈1.5°) → 재검 큐**, 단조성 위반 프레임 플래그. 같은 yaw_cmd 재촬영 쌍(상승/하강)의 cx 차는 백래시 추정에 사용(참고치 ≈5.6px) |
| 모델 보조 | det640 best.pt(full 640x480, 좌표 ×2) 예측과 중심차 >8px 또는 IoU<0.5 → 재검 큐. 의사검출 실패 구간(idx 88~357)은 이 규칙 적용 불가 → 궤적(b2) 규칙만 |
| 사람 | 20장 파일럿으로 라벨러 간 분포 측정 후 임계 확정. 목표 IoU ≥0.70, **중심차 med ≤3px**. 50장 2인 독립 재라벨(b1 20 + b2 30, 거리 균등) → inter-annotator 중심 오차를 모델 목표의 하한으로 보고. b2 같은 yaw_cmd 재촬영 쌍은 inter-annotator 측정의 보조 자료 |

### C-5. 라벨링 순서 (b2 830장) — ⚠️ 2라운드 자기학습으로 809장 라벨 완료. 아래 수동 계획은 val/test 세션 전수 검수와 앞으로 추가될 ~1170장에만 적용

R0 수동 = **세션별 균등 ≈150장 + 의사검출 실패 구간 전량(idx 88~357, ≈340장) + presence 1차 확인 전량(830장 target/none 클릭)** ≈490장 박스 수동 → R0 모델(E0 전 단계, b1 180 + R0)로 미라벨 예측(conf 0.05) → R1 액티브 러닝 ≈200+랜덤 50: 우선순위 (1) 최고 conf<0.25 (2) conf>0.25 박스 ≥2 (3) 최고 conf 0.25~0.6 (4) b2 cx~yaw_cmd 잔차 상위 → R2 pre-label 승인/수정(잔여 ≈90). **pseudo-label은 train 전용**, `inventory.csv`의 label_source에 기록. pre-label은 **full 640x480(det640 경로)** 으로 생성해 밴드 의존을 제거하고 좌표만 ×2(1280 추론 금지; 밴드 상단 여유 ≈5px이라 밴드 크롭 pre-label은 b2에서 추가 손실). 라벨 비용 0 네거티브: (1) presence=none 확인 프레임, (2) 타겟 없는 다른 y 밴드(예 y[64:384])를 같은 1280x320로 잘라 train에만 추가.

시간 재추정(830장 기준, 추정): presence 클릭 830장 ≈40분, 수동 박스 ≈490장×45s ≈6h, AL/승인 ≈340장×15s ≈1.5h → **합계 ≈8h**(2000장 전제의 "수동 8h vs 보조 4~5h"를 대체). 추가 ~1170장이 오면 b3로 같은 비율 적용.

학습 곡선이 필요하면 `fraction` 금지 → placement×distance 층화 랜덤 `train_025.txt / train_050.txt` 목록으로.

### C-6. 분할 프로토콜

| 셋 | 규칙 |
|---|---|
| 단위(계층) | room/day(현재 1) > camera_config(3) > **placement×distance(11)** > capture session(19). 같은 배치·같은 세션 프레임은 절대 분리 안 함(인접 2° 스텝 ≈20px 이동이라 세션 내 분할은 누수). val/test는 placement×distance 단위로 **통째로** held-out → "거리 보간" 성능 측정 가능. 비디오 배치(b1)는 시간 블록(≥300프레임) + 경계 양쪽 ≥30프레임 버퍼 |
| train/val/test(제안) | val = {center_2.4m 2세션(103), right_1.8m(42)} ≈145장(선택용, 2.4m recall 측정) / test = {center_1.8m 2세션(108), right_2.1m(41)} ≈149장(봉인) / train = 나머지 b2 ≈536 + b1 180. 173136(4장)·153623(WB 실패)·163106(gains 변경)은 train 전용. 네거티브는 presence=none 확인분을 같은 단위로 배분. **보고서에 "방 간·날짜 간 일반화 미측정" 명시** |
| 테스트 봉인 | 디렉터리 읽기 전용 + 체크섬, data.yaml 어디에도 넣지 않음, 전체 프로젝트에서 **최대 3회** 평가, 2인 교차 라벨 |
| 레거시 | 기존 42 val은 "같은 녹화 인접 구간"이라 변별력 없음 → 폐기(b1 180장 전부 train). 테스트 8장(b1 624~647)은 b1 ±30프레임을 어느 split에도 넣지 않고 스모크 전용(**b1 한정 규칙**) |
| 재현 | `split.json` 단일 매핑(uid 기준), split seed 고정(0), 모델 seed만 0/1/2 변경 |
| 누수 검사 | (1) 같은 (batch, session)이 두 split에 걸치면 실패, (2) b1만 train↔val/test 최소 프레임 거리 ≥30, (3) uid 중복 0 |

---

## (D) 입력 전략

| 선택지 | 객체 스케일 | 픽셀 수 | 측정/추정 | 결정 |
|---|---|---|---|---|
| **A. 밴드 640x160** (기본선) | 0.5배(12x21px; 2.4m에서 7x13px 참고치) | 102k | mAP50-95 .802, CPU 31.3ms, A100 3.8ms | 기본 채택. 2.4m recall이 E0에서 <.95면 B로 상향 |
| B. 밴드 960x256(rect pad로 240→256) | 0.75배 | 246k | 미측정 | E1 ablation |
| C. 밴드 1280x320(원본 1:1) | 1.0배(25x43px) | 410k | 미측정, CPU 47.6ms(full 640² 73.2보다 빠름) | E1 정밀도 상한. Pi4 지연 허용 시만 |
| D. full 640x480 | 0.5배 | 307k | det640 .765, CPU 73.2ms | 폴백(획득 단계) 전용 |
| E. 밴드 512x128 | 0.4배(=full 512 스케일, .752) | 66k | 미측정 | Pi4 FPS 미달 시 후퇴 후보(benchncnn 후 결정) |

- **밴드 재산정(즉시, GT 라벨 후 확정)**: 두 배치 합산 GT y의 p0.5~p99.5 ±48px, 높이 32 배수. b1 y 448~632(밴드 384:704은 상 64/하 72 여유)는 **b1 한정 사실**. b2 의사박스 y1 최소 388.9, cy median 424 → 384:704에서 상단 여유 ≈5px, b1 대비 median ≈50px 위(마운트/피치 차이 추정). 의사박스 기준 예상 밴드 **y≈[352:672](높이 320 유지)** 또는 [320:704](높이 384, 640x192). pitch_cmd≠90 세션(42장)은 별도 확인.
- **피치 연동 밴드(선택지)**: 운용 밴드를 `y0 = y0_ref + k·(pitch_cmd − 90)`로 정의(k는 b2 pitch 90/92/…/98 세션의 GT cy 회귀로 추정, 단위 px/°). 배포 측이 pitch 명령을 알고 있으므로 비용 0. 학습 샘플은 오프셋 ±64 랜덤 생성으로 동적 밴드와 분포 일치.
- **학습=배포 해상도 일치**: 학습 ds2_crop(1280x320 원본 저장, ultralytics 내부 INTER_LINEAR 0.5배) = 배포(슬라이스 → `cv2.resize(…, (640,160), INTER_LINEAR)`; 정확히 0.5배에서는 INTER_AREA와 동일). 960/1280 변형은 사전 리사이즈된 데이터셋을 디스크에 저장해 보간 통제.
- **밴드 이탈 fallback**: 선택지 A와 D는 같은 0.5배 스케일이므로 한 가중치로 운용 가능. (1) 획득 단계/5연속 미검출 → full 640x480 (별도 export imgsz=480,640); (2) 트래킹 중에는 동적 밴드 ±64px 이동(피치 연동 포함); (3) 주기적 full 재확인은 Pi4 실측 전 수치 금지. 폴백 채택 여부는 **자연 주행 세션의 밴드 밖 발생률**로: <2% 밴드 전용, 2~10% 밴드 확대(예 y[320:768]), >10% dual(E12) 학습. 두 배치 간 50px 오프셋이 실측됐으므로 "<2%" 판정은 밴드 재산정 **이후** 데이터로만.
- 금지: 224 리사이즈 경로, 320 타일 학습 주력(유일한 측정이 반대 방향, val이 320x80으로 축소되는 best.pt 선택 모순) — 3-way ablation의 한 arm으로만 강등.

---

## (E) 모델·학습 레시피

### E-1. 기본선 E0 (측정 레짐 재현 + b1/b2 병합)

```bash
yolo detect train model=/data/yolo11n.pt data=/data/YOLO/ds2_crop/data.yaml \
  imgsz=640 rect=True batch=16 epochs=200 patience=0 \
  optimizer=AdamW lr0=0.002 lrf=0.01 momentum=0.9 weight_decay=0.0005 \
  warmup_epochs=3 warmup_bias_lr=0.0 cos_lr=False \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=0 translate=0.1 scale=0.5 shear=0 perspective=0 \
  fliplr=0.5 flipud=0 mosaic=1.0 close_mosaic=20 mixup=0 copy_paste=0 \
  cache=ram workers=8 seed=0 deterministic=True iou=0.7 \
  project=/data/YOLO/runs2 name=E0_band640_s0
```

인자 해석 주석(소스 확인됨): `rect=True`면 mosaic/mixup/cutmix 자동 0(인자를 적어도 적용 안 됨, close_mosaic 무의미); 밴드가 모두 같은 shape라 shuffle 유지; `copy_paste`는 segments 없으면 no-op; `erasing`은 classify 전용; `warmup_bias_lr=0.0`은 auto 경로가 강제하던 값이므로 필수; `patience=0`이 8.4.171 정식 비활성(기본 100); train ≈730장×200ep에서도 auto는 AdamW(iterations=ceil(730/64)×200=2400)지만 명시로 재현성 확보; hue 0.015는 덧셈이라 ±2.7 단위≈±5.4°(b2 153623 핑크 캐스트는 이 범위 밖 → 학습 범위 외 조건으로 F-5에 등재); hsv_v=0.4는 노출 고정(camera_config.json 잠금) 배포 대비 과할 수 있어 E5에서 0.2 비교.

변경점(측정 run 대비): epochs 300→200(train 730장이면 iters ≈5배), patience 60→0, cache ram, workers 8. 베이스라인 행으로 **crop640 best.pt를 새 val(placement×distance held-out, 사람 라벨)에 먼저 재평가**해 summary.csv 첫 행에 기록.

### E-2. 실험 매트릭스 (한 번에 한 변수, 1 seed 스크리닝)

| ID | 변수 | 값 | 통과 기준(새 val, 사람 라벨) | 추정 시간/run |
|---|---|---|---|---|
| E0 | 기본선 | 위 명령 | crop640 재평가 대비 중심 오차 median 악화 없음, recall ≥ .98, **2.4m 세션 recall ≥ .95** | 35~45분 (추정: crop640 1.48s/epoch@9 iters ≈0.16s/iter → 730장 ≈46 iters ≈ 8~10s/epoch; 830장 단독이면 ≈52 iters) |
| E1a | 밴드 해상도 | imgsz=960 (실제 960x256) | mAP50-95 ≥ E0+2σ 또는 중심 오차 p95 개선 또는 2.4m recall 개선 | 1~1.5h (추정) |
| E1b | 밴드 해상도 | imgsz=1280 batch=8 (1280x320) | 동일 (정밀도 상한 기록용) | 1.5~2h (추정) |
| E3a | rect/mosaic | rect=False mosaic=0 (shuffle만 변동) | 3-arm 비교: 중심 오차·네거티브 FP | 1~1.5h (640² 패딩, 추정) |
| E3b | rect/mosaic | rect=False mosaic=1.0 close_mosaic=20 | 네거티브 FP 감소 & 중심 오차 악화 없음 | 1~1.5h |
| E4 | scale | 0.25, 0.4 | 거리 분포가 1.2~2.4m로 좁으면 0.25 우세 예상(추정) | 각 35~45분 |
| E5 | 색 증강 | hsv_v=0.2 (노출 고정 반영) / hsv_s=0.5 | 중심 오차·FP 악화 없음이면 채택(배포 분포 근접) | 35~45분 |
| E10 | 손실 가중 | box=10 dfl=2.0 | 중심 오차 median/p95 개선 ≥2σ | 35~45분 |
| E8 | fliplr | 0.0 | 구조물 비대칭 영향 확인 | 35~45분 |
| E12 | dual(선택) | train=ds2_crop+ds2_full(640x480) | 밴드 정밀도 E0 대비 유지 & full val(det640 .765 기준) | 1.5~2h |
| (선택) | 용량 상한 | yolo11s.pt, 커스텀 P2 1회 | 배포 후보 아님. FLOPs 3.3배/+72%. 예산 남을 때만 | – |

순서: E0 → E1a/b → E3a/b → E4 → E5 → E10 → E8 → (E12). 스크리닝 ≈11 run ≈ 9~13h(추정) → **상위 2~3개 조합을 seed 0/1/2**(≈4~6h) → 채택 규칙 "3 seed 평균 차 ≥ 2×표준편차 AND 중심 오차 p95 악화 없음". 전체 **1일 순차**(추정). 모든 run의 지표를 `/data/YOLO/runs2/summary.csv`(run, seed, 인자 diff, best_epoch, mAP50-95, 중심 오차 med/p95/x/y, recall@0.25 전체/2.4m, FP/프레임, b2 선형성 잔차 med)에 누적.

최종 재학습: train+val 합쳐 `patience=0 epochs=(fold best_epoch 중앙값×1.1)`, val은 train 소수 샘플 지정, **last.pt 채택**, 봉인 테스트는 data.yaml에 넣지 않음.

금지(must_drop): optimizer=SGD를 "재현성"으로 포장, patience=50+cos_lr 기본선, scale=0.25 기본값 변경, degrees≥3/shear/perspective, mixup=0.1, 40-run 5-fold 매트릭스, teacher 앙상블 라벨 정제(train 불일치 플래그→사람 재검토까지만 허용), val iou=0.6.

---

## (F) 평가 프로토콜

1. **raw 저장 1회**: 각 모델로 val/held-out/네거티브에 `conf=0.001` 예측을 저장하고 게이트·임계·운용점 곡선은 오프라인 재집계(GPU 재추론 없음).
2. **중심 오차 스크립트**(yolo val 대체): `model.predict(img_band, imgsz=(160,640), rect=True, conf=0.25)` 또는 `LetterBox((160,640), auto=False)`로 **배포와 같은 입력**을 넣고 최고 conf 1개 → `(x×2, y×2+y0)` 역변환 → GT 중심(사람 라벨, 가능하면 클릭 포인트도)과 |Δx|, |Δy|, L2의 median/p95/max. 주의: `yolo val`의 PyTorch rect는 pad=0.5로 672x192라 배포 기하와 다름 → mAP는 참고치만.
3. **per-condition 표**: placement×distance 11종(거리 버킷 1.2/1.5/1.8/2.1/2.4m) / camera_config 3종 / 조명 버킷(두 배치 합산 mean_bgr 분위수 재정의: b1 103~137, b2 정상 세션 125~146이므로 **<128, 128~138, >138** + 별도 "WB 실패 153623(≈199)"·"gains 변경 163106(≈177)") / 크기 버킷(sqrt 면적 <30, ≥30) / cx 빈(160px 8빈; b2 right_* 세션으로 오른쪽 빈 채워짐, GT 후 확인) / 밴드 안·밖 / pitch_cmd≠90 / 가림 / 다중 타겟 / 세션별.
4. **네거티브**: FP/프레임 @conf 0.25(+게이트), 혼동물체 영역 박스 수, **사람 영역 FP 열**(배경 사람·보라색 의류, b2 박스 ≥2개 프레임 20/830 참고치) 별도.
5. **교란 스위트**(라벨 추가 없이 변별력): 학습 증강 범위 **밖**만 — 밝기 ×0.35/×0.5, 감마 2.2, 블러 σ3, **수평 모션 블러 커널 5/11/21px**(배포는 회전 중 15ms 노출, b2는 정지 촬영이라 블러 0), 노이즈 σ20, 수직 시프트 ±120px(근거: 두 배치 간 cy median 오프셋 ≈50px 실측 + 여유; 고정 밴드 vs 동적/피치 연동 밴드+폴백 2가지), 가림 30/50%, **WB 캐스트(153623 세션 실측 핑크 캐스트 재현: colour_gains 1.29/1.63 적용 또는 해당 세션 그대로)**. 표에 "학습 범위 내/외" 열 구분. ×0.6/×1.4는 hsv_v 범위 내라 사용 안 함.
6. **통계**: 세션(b2)/세그먼트(b1) 단위 페어드 부트스트랩(1000회) 95% CI. 이미지 단위 CI 금지(인접 프레임·인접 스텝 상관).
7. **선택 vs 보고 분리**: val(선택용) / placement×distance held-out 테스트(보고용, ≤3회). val_hard를 조기종료·선택·보고에 동시에 쓰지 않음.
8. **end-to-end yaw**: (G)의 OOF 프로토콜로, cx 빈별·세션별 MAE/p95 + **세션 내 선형성 잔차(px)** 를 1차로.
9. 스모크: b1 테스트 8장 8/8 검출, 근사 GT 중심 오차 ≤5px(상한), b2 right_2.4m 1세션 cx~yaw_cmd 단조성.

---

## (G) end-to-end yaw 단계

**권고: YawCNN 폐기 → 기하 모델/LUT.** 근거(노트북 확인): 마스크 정보량은 정수 4개뿐, 1280→224 비등방 리사이즈(x 0.175배)+np.rint로 ±2.9px@1280 양자화, Sigmoid+min-max는 학습 범위(cx 10~822) 밖 외삽 불가, 228,353 파라미터 vs 180장, Pi4에서 torch 의존.

**데이터 전제(b2)**: 세션 안에서 타겟은 고정되고 카메라만 2° 스텝으로 회전 → yaw_cmd는 "카메라 팬 명령값"이고 타겟 bearing은 세션별 미지 상수. 따라서 **yaw_cmd = b_s + f(cx)** (b_s 세션 고정효과, f 공통 광학 모델)로만 적합 가능하며, 배포에서 쓰는 것은 f(cx)(타겟 상대 각)이다. b1(비디오)은 data_angles.csv 정의·동기 확인 전까지 적합에서 제외.

| 후보 | 식 (모두 b_s 포함) | 파라미터 |
|---|---|---|
| M1 핀홀 | yaw = b_s + a·atan(k·u), u=(cx−640)/640 | 2 + S |
| M2 등거리 어안 | yaw = b_s + a·u | 1 + S |
| M3 | M2 + c·u³ | 2 + S |
| M4 시차 | M3 + d·u·v (v=(cy−cy0)/…) | 3 + S — d가 유의할 때만 채택(세션 내 cy가 거의 상수라 b_s와 혼동) |
| M5 LUT | 1280 엔트리 단조(isotonic) + 선형 보간, b_s는 세션별 상수 | – |
| (선택) 백래시 항 | 위 모델 + e·dir (dir=스텝 방향 ±1) | +1 — 같은 yaw_cmd 상승/하강 cx 차 참고치 ≈5.6px(≈0.5°); e 유의 시만 |

프로토콜:
- 두 단계 **공통 분할**(split.json). YOLO fold 모델로 **out-of-fold 박스** 생성(fold는 1~2개로 축소, 5-fold 전량 금지) → yaw 모델은 OOF 박스로 적합·평가. YOLO 실패 샘플은 버리지 않고 GT 중심점으로 yaw 모델에 사용.
- 손실 Huber(δ=1°), 선택 기준 **세션 내 잔차**(세션 GroupKFold OOF; b_s는 held-out 세션에서 그 세션 데이터로 재추정하므로 f의 일반화만 평가), 동률이면 파라미터 적은 쪽. 산출물 `yaw_calib.json {model, params, band_y0, scale, lut[1280], backlash_e}`.
- **라벨 진단(모델 개선보다 먼저)**: q=2°(labels.csv 측정, 31개 고유값) → MAE 하한 ≈ q/4 = 0.5°, 3px 중심 오차 ≈0.3°는 하한 아래라 **end-to-end yaw MAE로는 중심 정밀도 개선이 보이지 않음 → 세션 내 선형성 잔차(px)를 1차 지표로**. 세션 내 단조성 위반 횟수, **방향별 잔차(백래시) 추정**(상승/하강 스텝 분리 적합), 세션별 b_s 분포. **명령각 vs 실제각 오차(가정 ±1~2°) 측정 방법**: 체커보드를 고정 위치에 두고 같은 yaw_cmd 시퀀스를 2회(상승/하강) 재촬영해 코너 변위로 실제 회전각 추정, 또는 정지 재촬영 반복으로 반복 정밀도 측정. **τ(동기 지연) 적합은 b1 비디오 배치에만 해당**(Δcx 155/155 양수·속도 상수라 식별 불가 → 타임스탬프 확보 전 적합 금지); b2는 settle 0.4s 후 정지 촬영이라 τ 없음.
- **오차 전파 검증표**: 측정 중심 오차 px × dyaw/dx(세션별 선형 적합: right_1.5~2.4m **≈10 px/°**, 즉 1px≈0.1°; 1.2m 세션은 비선형/FP 오염으로 GT 후 재측정) = 예측 yaw 오차 vs 실측 세션 내 잔차. 일치하면 라벨(명령각) 노이즈 작음, 실측이 훨씬 크면 서보 명령-실제각 오차·백래시 문제. b1 1프레임 지연 44.6px ≈ **4.5°**(실측 10px/° 적용; 종전 "4~6°(가정 FOV)" 대체).
- **대조군**: 기존 YawCNN을 같은 fold·OOF 박스 마스크로 1회만 학습해 MAE 비교, 폐기 근거 기록. YawCNN을 유지해야 한다면 ONNX(onnxruntime aarch64)로만 실행하고 좌표 환산 `x·224/1280, y·224/960` 규칙을 노트북과 일치시킬 것.
- 선택: 박스 ROI HSV 블롭 밝기가중 centroid로 서브픽셀 정제(**박스 내부 ROI로만 제한** — 배경 보라색 의류 오염 방지; 블롭 면적 0.3~0.9×박스, 박스 중심과 6px 이내일 때만; 실패 시 박스 중심). 조명 취약하므로 기본 off, ablation으로 중심 오차 비교.
- 모든 yaw 출력에 플래그(conf, centroid 채택, 외삽 여부) + CSV 로그(cx_orig, conf, yaw).

---

## (H) Pi4 배포

**파이프라인**: Picamera2/libcamera CSI 1280x960 캡처(camera_config.json 잠금 유지: colour_gains/exposure_time/analogue_gain 고정, 캡처 스레드, 최신 프레임만 유지 큐) → `frame[y0:y0+320]` 슬라이스(0비용, y0는 피치 연동 선택 시 pitch_cmd로 갱신) → `cv2.resize((640,160), INTER_LINEAR)` → NCNN FP32 batch=1 → 기하 게이트(밴드 내 cy, h/w 1.0~3.0 느슨하게, 크기는 트랙 대비 상대 변화만) → 최고 conf 1개(conf 0.25) → `(x×2, y×2+y0)` → yaw_calib(µs) → 플래그+CSV 로그.

**export**
```bash
# 배포 기본(FP32, 고정 160x640; quantize 미지정)
yolo export model=/data/YOLO/runs2/final/weights/last.pt format=ncnn imgsz=160,640
# 폴백 full 모드(같은 가중치, 별도 디렉터리)
yolo export model=/data/YOLO/runs2/final/weights/last.pt format=ncnn imgsz=480,640
# 패리티/ORT 벤치용
yolo export model=/data/YOLO/runs2/final/weights/last.pt format=onnx imgsz=160,640 opset=12 simplify=True dynamic=False
```

| 항목 | 결정 | 근거 |
|---|---|---|
| 런타임 | NCNN 1순위 | 공식 Pi4 YOLOv8n 640² FP32: NCNN 414.73ms > ONNX 560.04 > OpenVINO 534.93 > PyTorch 1068.42. Pi5에서 YOLO11n NCNN ≈ YOLOv8n(88~94 vs 94ms) |
| FP16 | `quantize=16` 미사용 | A72(ARMv8.0-A)는 FEAT_FP16 없음 → 가중치 크기만 반감, 속도 이득 거의 없음(추정) |
| INT8 | ultralytics 경로 불가(NCNN은 INT8_FORMATS에 없음, quantize=8 assert) → `ncnn2table`(placement×distance 균등 ≥300장 보정 리스트) + `ncnn2int8` 수동. 채택 조건: 중심 오차 증가 <0.5px@640x160 & 미검출 증가 0(2.4m 포함) & 교란 스위트 재측정 | ncnn 벤치 A72 INT8 1.2~1.8배(mobilenet 계열), YOLO 미측정; SDOT 없어 실패 가능 |
| 스레드 | NCNN `net.opt.num_threads` 3~4 실측 결정(ultralytics NCNNBackend는 미설정), `use_fp16_arithmetic=False` | libcamera ISP/캡처 스레드의 코어 점유는 미측정(가정 1코어) |
| 최종 러너 | ncnn 바인딩 + 단일 클래스 argmax 디코드(파이썬 래퍼 오버헤드 제거) | |
| 지연 예산(추정) | 640x160 NCNN FP32 ≈100~150ms, 512x128 ≈70~100ms, 1280x320 ≈400ms급. 캡처 15~25ms(병렬), 전처리 ~2ms, yaw µs | 모두 **benchncnn 실측으로 치환** 전까지 설계 결정 보류(512 후퇴 사전 결정 금지) |

**패리티**: `yolo val`은 non-pt 모델에 rect=False·정사각 강제라 불가 → 자체 스크립트(같은 640x160 입력, LetterBox auto=False)로 PyTorch vs NCNN vs (INT8) 중심 오차 차·미검출 건수·conf 차 100장(거리 균등).

**Pi4 측정 체크리스트**(모델 선택보다 앞, warmup 20 후 200프레임 p50/p95):
1. `uname -m`=aarch64, `vcgencmd get_throttled`(0x0 유지)/measure_temp, 방열 확인
2. `benchncnn` 640x160/512x128/768x192/640x480, threads 1~4, powersave 0
3. ultralytics 래퍼 포함 지연 vs benchncnn 차이(파이썬 오버헤드)
4. Picamera2 1280x960 캡처 fps·프레임 간격 지터, camera_config.json 잠금 상태(`config_locked`), 캡처 스레드 코어 점유
5. 전처리(슬라이스+리사이즈+normalize) ms
6. glass-to-yaw end-to-end p50/p95, 10분 지속 FPS 드리프트
7. RSS <1GB, 스왑 0
8. 정확도 패리티 100장(위)
9. 폴백 모드 지연·재획득 성공률(타겟을 밴드 밖으로 이동시키는 시나리오)
10. 런타임 로그(폴백 진입 횟수, 유령 트랙, 평균 conf), 회전 중 프레임의 모션 블러량(회전 속도 °/s × 15ms 노출)

---

## (I) 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| 데이터 다양성 부족(b2는 방 1·날짜 1·마운트 1) | 방 간 일반화 미보증, 대회장 성능 미측정 | 보고서에 명시, 새 방·조명·날짜 세션 녹화 1순위, 나머지 ~1170장 소재 확인(K1) |
| 파일명 충돌(b1/b2 동명 다른 내용) | ds2_crop 덮어쓰기·라벨 짝 뒤바뀜 | 배치 접두사 uid, inventory 키 (batch, session, idx), 누수 검사에 uid 중복 0 |
| b2 idx 88~357 타겟 색상 불일치(파란 계열, 의사검출 거의 0) | 아군/다른 물체면 라벨 규약 충돌, 적이면 recall 파괴 구간 | K5 확인 전 '보류' 태그, R0 수동 전량 포함, 색 원인(공/조명/WB) 확인 |
| 라벨 규약 드리프트/라벨러 간 지터 | 중심 오차 하한 상승(1px=IoU .883) | 규약 문서 고정, 20장 파일럿, 50장 2인 재라벨, pre-label 편향(박스 체계적 축소) 사람 교정 |
| 서보 명령각 vs 실제각 오차·백래시(b2) | yaw 라벨 바이어스가 중심 정밀도 개선을 가림 | 세션 고정효과 + 방향별 잔차 추정, 체커보드 재촬영으로 명령-실제각 측정, 오차 전파표 |
| yaw 라벨 동기 오차(b1 비디오만) | 1프레임 지연 ≈ 44.6px ≈ 4.5°(실측 10px/°) | b1은 타임스탬프 확보 전 yaw 적합 제외. τ 적합 금지 |
| 밴드 가정 붕괴(피치·마운트 변경; b1↔b2 cy 오프셋 ≈50px 실측) | recall 0 | 합산 GT로 밴드 재산정, 피치 연동 밴드, 동적 밴드 ±64, full 폴백(같은 가중치), 수직 시프트 ±120 교란 |
| 네거티브 마이닝 오용 | 타겟 있는 프레임을 빈 라벨로 넣어 recall 파괴 | presence 사람 확인 프레임만 빈 .txt(화각 밖 기하 예측은 후보 선정에만) |
| WB 드리프트(153623 핑크 캐스트 실측) / 모션 블러(배포만) | 학습 범위 밖 분포 | 교란 스위트 "학습 범위 외" 등재, camera_config.json 잠금 유지 확인(K4), 회전 중 촬영 세션 추가 |
| 원거리(2.4m 초과) 데이터 0장 | 요구 거리 > 2.4m이면 160 절벽 영역 진입 가능 | K3 요구 거리 확인, 2.7/3.0m 세션 추가, E1a/b 통과 기준에 2.4m recall |
| Pi4 지연 미측정 | 640x160로 목표 FPS 미달 가능 | benchncnn 먼저, 512x128/INT8은 실측 후 |
| 양자화로 12x21px(2.4m 7x13px) 객체 중심 열화 | yaw 바이어스 | 패리티 <0.5px 조건, 교란 스위트 재측정 |
| val 포화/선택 편향 | 잘못된 설정 채택 | 사람 라벨 val, placement×distance held-out 보고 전용, 3 seed ≥2σ |
| 오른쪽(cx>822) 영역 검증 | b1만 0장이었음 → b2 right_* 의사박스 cx 807~1266(217건) | 리스크 하향: **박스 GT 라벨링 후 확인**으로 대체, 추가 촬영 불필요 |
| MIG 1장 순차 예산 | 매트릭스 ≈1일(추정) | 12 run 상한, 1 fold 스크리닝 |

---

## (J) 우선순위 로드맵 (주 단위)

| 주 | 작업 | 산출물/통과 기준 |
|---|---|---|
| 1주 | b1+b2 인벤토리(labels.csv/sessions.jsonl 파싱, uid 부여, presence 기하 예측), 규약 문서, 20장 파일럿 QA, split.json(placement×distance), 테스트 봉인, presence 830장 사람 확인, 밴드 재산정 1차(의사박스), ds2_crop/ds2_full 생성 스크립트, crop640 best.pt 새 val 재평가(베이스라인 행), b2 yaw 라벨 진단(q=2°, 세션 내 단조성, 방향별 잔차). 나머지 ~1170장 소재·idx 88~357 색상·요구 거리 확인(K). 필요 시 새 방·네거티브·회전 중·2.7/3.0m 녹화 시작 | inventory.csv, summary.csv 1행, presence 열 완성 |
| 2주 | R0 ≈490장 수동(세션 균등 + 실패 구간 전량) → R0 모델 → AL R1 → 밴드 재산정 확정(GT) → E0 → E1a/b, E3a/b 스크리닝. 중심 오차 스크립트·raw 저장 파이프라인 완성 | E0 통과(recall ≥.98, 2.4m ≥.95, 중심 오차 median ≤ 베이스라인) |
| 3주 | R2 pre-label 승인, E4/E5/E10/E8/(E12), 상위 3개 3 seed, 50장 2인 재라벨 결과 보고. OOF 박스 → yaw 기하 모델(세션 고정효과) 적합, 백래시 항 검정, YawCNN 대조군 1회, 오차 전파표(10px/° 실측 갱신), 체커보드 명령-실제각 측정 | 레시피 확정(≥2σ), yaw_calib.json |
| 4주 | 최종 재학습(train+val, last.pt), export(ncnn 160x640 / 480x640 / onnx), 패리티 스크립트, Pi4 체크리스트 1~10, INT8 수동 시도(조건부), 교란 스위트(모션 블러·WB 캐스트 포함) | 봉인 테스트 1회(최대 3회 중), Pi4 실측표 |
| 5주 | 현장 CSV 로그 기반 재캘리브레이션(b_s 재추정), 폴백/동적·피치 연동 밴드 튜닝(실측 FPS 기준), 잔여 라벨링(b3 포함) | 운용 conf/게이트 확정, 최종 보고 |

---

## (K) 사용자 확인 필요 질문

**파일에서 확인됨(질문 불필요)**: b2 830장 전부 1280x960·같은 카메라·같은 방·같은 날(2026-10-02); 박스 라벨 없음; yaw 라벨은 labels.csv의 yaw_cmd(서보 팬 명령값, 2° 양자화, pitch_cmd 포함, timestamp 초 단위) — 엔코더/IMU 아님; 촬영은 settle 0.4s 후 정지 촬영이라 프레임-각도 동기 문제 없음; 카메라는 Picamera2/libcamera CSI, camera_config.json 잠금(노출·게인·WB 고정); 모든 세션에 양방향 스텝 포함; right_* 세션으로 오른쪽 영역 확보.

**여전히 확인 필요**:
1. "약 2000장" 중 b2 830장 외 **나머지 ~1170장의 소재**(추가 배치 폴더 예정? 다른 방/날짜/마운트인가? 같은 labels.csv 포맷인가?).
2. yaw 정의: 배포 시 yaw 출력은 **서보 팬 명령값 기준인가, 로봇 차체 회전 명령인가**? SG90 명령각 vs 실제각 오차(스펙/실측)와 백래시를 측정한 적 있는가? b1 data_angles.csv의 yaw_offset 정의·측정 방법·타임스탬프 동기 여부(b1 yaw 적합 포함 여부 결정용)?
3. **대회 최대 교전 거리(m)** 와 요구 FPS·허용 glass-to-yaw 지연(ms), 제어 루프 주기, 동시 실행 프로세스의 CPU 점유?
4. 배포 시 **camera_config.json 잠금 유지 여부**(AWB/AE 자동 복귀 가능성), 로봇 회전 속도(°/s)와 15ms 노출의 모션 블러량, 체커보드 촬영 가능 여부(FOV/어안 보정·명령-실제각 측정용)?
5. **b2 idx 88~357(세션 170657~171540, center_1.2~2.4m)의 파란 계열 물체가 적 타겟인가?** 색상 차이 원인(다른 공/조명/WB)? 아군 타겟(다른 색 공+같은 받침)이 존재하는가? 한 화면에 타겟 2개 이상이 보일 때 어느 것을 골라야 하는가?
6. 운용 중 타겟이 밴드 밖으로 나가는 상황(피치·경사·근접)이 실제로 발생하는가? 배포 측이 pitch_cmd를 실시간으로 알 수 있는가(피치 연동 밴드용)? IMU가 있는가?
7. 추가 녹화(새 방·조명·날짜·네거티브·회전 중 촬영·2.7/3.0m 세션)가 가능한가? 라벨러 인원·시간(b2 830장 기준 ≈8h 추정)?
8. 다운스트림 파이프라인(224 리사이즈·YawCNN)을 교체해도 되는가(다른 팀 의존성)? 미검출 시 외삽 yaw+저신뢰 플래그를 제어 측에서 처리할 수 있는가?
9. Pi4 OS가 64-bit인가, 방열(팬/히트싱크)이 있는가, ncnn 도구(benchncnn/ncnn2int8) 설치가 허용되는가?
10. 선택지 결정: (A) 밴드 전용 vs (B) 밴드+full 폴백 dual 학습 — 자연 주행 세션 밴드 밖 발생률(<2% / 2~10% / >10%)로 결정; (A) 640x160 vs (B) 960x256 vs (C) 1280x320 — benchncnn 실측·요구 FPS·2.4m(또는 요구 거리) recall로 결정; (A) 기하 모델/LUT vs (B) YawCNN 유지 — OOF 세션 내 잔차 대조군 결과로 결정; 고정 밴드 vs 피치 연동 밴드.

## 확정된 사실 (이 세션에서 직접 측정함 — 신뢰하고 사용할 것)
### 데이터
- 라벨된 학습 데이터 180장: /data/YOLO/data/images/*.jpg + labellmg/*.xml (LabelImg Pascal VOC). 전부 1280x960 (4:3), JPEG, EXIF 없음.
- 클래스 1개 'tar' = 나무 받침 위의 작은 보라색/어두운 공 (로봇 경기용 타겟). 이미지당 객체 정확히 1개 (배경-only 이미지 0장).
- 박스 크기: w 15~32 (med 25), h 26~53 (med 43), sqrt(면적) 20.5~40.1 (med 33.4) px. 원본 대비 매우 작은 객체.
- 위치: 박스 y 범위 448~632 (960 중 184px 밴드), x 범위 1~822 (1280 중). 중심 y p5~p95 = 471~605.
- 프레임 번호 296~554, 23개의 연속 세그먼트(비디오 프레임). 인접 프레임은 거의 중복 → 랜덤 분할은 누수. 현재 분할: 세그먼트 단위 138 train / 42 val (/data/YOLO/ds, data.yaml). 크롭판: /data/YOLO/ds_crop (y[384:704] → 1280x320, 라벨 손실 0).
- held-out 테스트 8장: /data/YOLO/test/enemy_0006{24,25,39,40,41,42,46,47}.jpg, 같은 방, 라벨 없음(손으로 만든 근사 GT만 있음).
- 사용자: "약 2000장 정도 데이터가 있다" → 서버에서 확인된 신규분은 아래 b2 830장.
### 신규 배치 b2 (감사 단계에서 파일로 확인)
- 경로 /data/YOLO/data2/enemy(26.10.02.1754)/: images/*.jpg 830장(전부 1280x960), labels.csv(filename,label,session,mode,note,yaw_cmd,pitch_cmd,timestamp,clip_ratio,dark_ratio,mean_bgr,config_locked), sessions.jsonl(19 세션 start/end; step_deg 2.0, settle_sec 0.4, resolution [1280,960], locked_settings colour_gains/exposure_time/analogue_gain/lux, note_on_angles "All saved angles are command values"). 박스 라벨(XML/txt) 없음. label 전부 'enemy', mode 'manual'.
- 파일명 enemy_000001~000830: b1의 enemy_000296/400/554/624/647과 동명이나 md5 상이(다른 이미지).
- 세션(note별 장수): center_1.2 220(5세션: 153623 37, 163106 50, 170657 45, 170948 45, 171757 43), center_1.5 92, center_1.8 108, center_2.1 98, center_2.4 103, right_1.2 43(173136은 4장), right_1.5 41, right_1.8 42, right_2.1 41, right_2.4 42. 161753 세션은 0장.
- yaw_cmd 65~125(중심 95, 2° 스텝, 31개 고유값), pitch_cmd 90~98(≠90 42장: 153623·163106·173136). 모든 세션에 +/− 스텝 혼재(예 170657 pos 14/neg 30, right_* ≈20/20).
- 카메라 설정 3종: colour_gains [1.29,1.63]·exposure 15464·gain 2.25(153623, 161753) / [1.03,1.76]·15124·2.0(163106) / [1.17,1.76]·15124·2.0(나머지 16세션). mean_bgr 124.8~200.7(153623 세션 med 199.4, 163106 177.3, 정상 세션 125~140). clip_ratio med 0.005, max 0.050.
- 촬영 시각 2026-10-02 15:36:28~17:40:30(같은 방·같은 날·같은 마운트).
- det640 best.pt(CPU, conf 0.1) 의사박스 참고치(GT 아님, FP 오염 가능): 검출 585/830(conf≥0.25 494); 세션 170657~171540에서 14/45, 28/45, 22/45, 5/45, 0/45, 1/45; cy 403~556(med 424), y1 최소 388.9; right_* cx 807~1266(cx>822 217, >900 165); dyaw/dx ≈10px/°(right_1.5~2.4m 선형 적합, 잔차 7~10px); 거리별 h med 1.2m≈46~48, 1.5m≈39, 1.8m≈32~33, 2.1m≈28~29, 2.4m≈26(w≈14); 같은 yaw_cmd 상승/하강 cx 차 med ≈5.6px; 박스 ≥2개 프레임 20/830. 샘플 이미지 100·290의 바닥 물체는 파란 계열, 370·10·800은 보라. 이미지 800 우측 테이블에 사람, 10 우측 가장자리에 보라색 옷 사람.
### 장면
- 실내(대리석 바닥, 타일 벽, 문), 로봇에 낮게 장착된 광각/어안 카메라. 오른쪽 하단에 로봇 자체의 나무 구조물이 항상 가림. 혼동 물체: 파란 라벨 분리수거함, 빗자루 손잡이(이전 모델이 여기에 오탐), 코카콜라 자판기, (b2) 배경 사람·보라색 의류.
### 다운스트림 파이프라인 (preprocessing_data_make_model(colab_pytorch).ipynb)
- 이미지 → 224x224 리사이즈 → YOLO(conf 0.5) → 최고 conf 박스 1개 → 224x224 흰 사각형 마스크 → YawCNN(4-conv, 224 그레이스케일 입력, Sigmoid 출력, MSE) 로 yaw_offset 회귀. 라벨: data_angles.csv (filename, yaw_offset). YOLO의 seg 마스크는 전혀 안 씀 — 박스만 씀. YOLO가 못 찾으면 그 샘플은 버려짐(failed).
- 즉 최종 목표 지표는 yaw 각도 오차이며, YOLO 박스의 '중심 위치 정밀도'가 직접 영향.
### 기존 모델 yolo_target.pt (yolo11n-seg, Roboflow 데이터로 imgsz 640 학습, 체크포인트 기재 mAP50 0.995)
- 이 방 데이터 180장에서: 224 파이프라인 recall 0/180, imgsz 640 recall 0/180, imgsz 1280 recall 29/180. 사실상 무용. 살릴 가중치 없음.
### 이번 세션에서 새로 학습 (yolo11n.pt COCO 사전학습 detection, epochs 300 patience 60, batch 16, hsv_h.015 s.7 v.4, translate.1, scale.5, fliplr.5, mosaic 1.0, close_mosaic 20, seed 0)
- det640 (전체 이미지 → 640): val P .998 R 1.000 mAP50 .995 mAP50-95 .765(별도 yolo val; results.csv 학습 중 best .7738, last .7458). A100 추론 11.2ms/img. 2.1s/epoch.
- crop640 (밴드 1280x320, imgsz 640 rect → 실제 640x160): val mAP50 .995 mAP50-95 .802. 3.8ms/img. 1.48s/epoch(총 8.5분/300ep). 박스 더 정밀하고 3배 빠름.
- 해상도 스윕 (전체 이미지, 150 epochs): imgsz 160: R .719 mAP50 .734 mAP50-95 .343 / 224: .597 / 320: .690(102ep 조기중단) / 416: .732 / 512: .752. 교훈: 같은 방·같은 배경에서는 224도 찾기는 하지만 mAP50-95가 크게 떨어짐(박스 정밀도 손실 → yaw 오차로 직결). 절벽은 160 부근.
- 테스트 8장: det640/crop640 모두 8/8 검출, conf .63~.92, 중심 오차 ~5px (근사 GT 기준).
- val 셋(42장)이 포화(mAP50 .995)되어 설정 간 변별력 낮음 → 평가 프로토콜 개선 필요.
### CPU 지연시간 (x86 4스레드 PyTorch, 절대값은 Pi4보다 훨씬 빠름 — 비율만 사용)
- full 224²: 23.5ms / 320²: 35.3 / 416²: 45.5 / 512²: 59.1 / 640²: 73.2 / 960²: 134.7
- band 640x160: 31.3 / band 1280x320: 47.6 (크롭은 같은 픽셀 수의 정사각형보다 빠름)
### 하드웨어/환경
- 학습: A100 MIG 1g.10gb (10GB). 배포: Raspberry Pi 4 4GB (Cortex-A72 x4, NPU 없음). 파이5가 아님. 카메라: Picamera2/libcamera CSI, camera_config.json 잠금.
- python: /data/envs/aimers/bin/python (ultralytics 8.4.171, torch 2.7.1+cu128). 결과: /data/YOLO/runs/{det640,crop640,sweep*}/results.csv
- 제약: 학습 실행·pip 설치 금지(스윕이 GPU 사용 중). 읽기 전용 확인(ls/grep/Read/짧은 python 통계)은 허용.