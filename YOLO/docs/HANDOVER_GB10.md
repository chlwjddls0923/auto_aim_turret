# 폴더 지도와 재현 방법

프로젝트 내용(과제, 측정값, 결정, 다음 할 일)은 [RESUME.md](../RESUME.md)에 있다. 이 문서는 **무엇이 어디에 있고, 무엇으로 만들었는지**만 적는다.
경로는 프로젝트 폴더(`YOLO/`, DGX에서는 `/data/YOLO`) 기준이다.

## 폴더 지도

| 경로 | 내용 | 만든 것 |
|---|---|---|
| `data/images/` + `labellmg/` | b1: 방 A 비디오 프레임 180장 + 사람 라벨(LabelImg XML) | 사용자 |
| `data/images(1)/` + `labellmg/` | b1x: 방 A 프레임 100장, 그중 98장에 사람 라벨. **학습에 쓴 적 없음 → test 전용** | 사용자 |
| `data2/enemy(26.10.02.1754)/` | b2 적군 830장 + `labels.csv` + `sessions.jsonl` (홀 B, 19 세션) | 사용자 |
| `data2/friend(26.10.02.1839)/` | 아군 662장. 세션 메타(labels.csv·sessions.jsonl)는 **아직 없음** | 사용자 |
| `test/` | 방 A 8장(b1x에도 같은 이미지와 라벨이 있다) | 사용자 |
| `data2_labels/` | b2 최종 의사 라벨 809장(`labels/`, `xml/`), `all_predictions.csv`, 제외 21장 `check_list.csv` | r2 자기학습 2라운드 |
| `data2_pseudo/`, `data2_pseudo_r2/` | b2 의사 라벨의 1·2라운드 중간 결과(기록용) | 〃 |
| `friend_pseudo/` | 아군 사전 라벨 637장, `images.csv`(상태·추정 strata), **`review_list.csv`(사람 검수 대상 149장)**, 대조 시트 | `tools/seed_label.py` |
| `friend_pseudo/none_recheck/` | 640에서 놓친 아군 25장을 1280으로 다시 잡은 라벨·XML·확대 시트 | r2 @1280 |
| `handover/dataset/` | **학습 데이터셋**: train 1,423 / val 228 / test 98, `data*.yaml`, `eval_*.yaml`, `list_*.txt`, `inventory.csv`, `split.json` | `tools/package_dataset.py` |
| `runs/r2/` | 라벨링에 쓴 모델(640 학습) + 기준선 평가 `eval_summary.csv` | 이전 세션 |
| `runs/r3/` | 전체 데이터 1280 학습 결과: `weights/best.pt`, `results.csv`, `args.yaml`, 평가 CSV | `scripts/train_r3.sh` |
| `runs/3arm/` | 색 불변성 실험 6 run + `summary.csv` + `table.txt` | `scripts/train_3arm.sh` |
| `runs/<기타>/` | 예전 run(det640, crop640, 해상도 스윕, pkgcheck)의 `results.csv`·`args.yaml`만 | — |
| `weights/ncnn/` | r2의 NCNN FP32 4종(960×1280, 736×960, 480×640, 384×384) | 이전 세션 |
| `weights/ncnn_r3/` | r3의 NCNN FP32 4종 | `tools/export_ncnn.py` |
| `weights/yolo11n.pt`, `yolo26n.pt` | COCO 사전학습(3-arm 시작점)과 ultralytics AMP 검사용 | Ultralytics |
| `tools/` | 아래 표 | — |
| `scripts/` | `env.sh`, `setup_gb10.sh`, `train_r3.sh`, `train_3arm.sh`, `run_all_training.sh`, `make_full_archive.sh` | — |
| `docs/provenance/` | 예전 학습 세트(`ds`, `ds_r2`)의 파일 목록 | — |
| `logs/` | 학습 로그 | — |
| `TRAINING_DESIGN.md`, `*.ipynb` | 초기 설계 문서와 원래 노트북(기록용. **RESUME.md와 다르면 RESUME.md가 맞다**) | 이전 세션 |

압축에 넣지 않은 것: 중간 데이터셋 `ds/`·`ds_r2/`·`ds_crop/`(원본에서 다시 만들 수 있음, 목록은 `docs/provenance/`), 흑백 데이터셋 `handover/dataset_gray/`(`tools/make_gray.py`로 재생성), `last.pt`와 학습 그림, 폐기한 예전 모델 `test/yolo_target.pt`, 아군 원본의 중복 zip.

## 데이터셋 구성

| 소스 | uid 형식 | train | val | test | 라벨 |
|---|---|---|---|---|---|
| b1 (방 A, 적군색) | `b1_enemy_000296` | 180 | 0 | 0 | 사람 |
| b1x (방 A, 적군색) | `b1x_enemy_000537` | 0 | 0 | 98 | 사람 |
| b2 (홀 B, 적군색) | `b2_<세션 시각>_<idx>` | 672 | 137 | 0 | 의사 |
| 아군 (홀 B) | `fr_na_<idx>` | 571 | 91 | 0 | 의사 |

- val = b2 세션 note ∈ {`center_2.4m`, `right_1.5m`} + 아군 추정 strata ∈ {`1.5_right3`, `2.4_left3`}.
- test의 `source`는 둘로 나뉜다: `b1x_after554`(83장, b1 학습 프레임보다 뒤)와 `b1x_adjacent`(15장, b1 학습 프레임과 시간상 이웃).
- `inventory.csv`의 `review` 열: `pending` = 사람 검수 대상(아군 149장), `pending:2nd_box_removed` = 재확인 때 두 번째 박스를 지운 2장(198, 205).
- 리스트: `list_train_enemy_only`(852) / `list_train_friend_only`(571) / `list_val_enemy`(137) / `list_val_friend`(91) / `list_eval_friend_all`(662) / `list_eval_enemy_b2_all`(809) / `list_test_human`(98).
- yaml에는 `path:` 키가 없다. ultralytics가 yaml이 있는 폴더를 데이터셋 루트로 쓰므로 폴더를 옮겨도 그대로 동작한다.

## 도구

| 파일 | 용도 | 실행 확인 |
|---|---|---|
| `tools/package_dataset.py` | 원본 + 라벨 → `handover/dataset` (sanity 포함). 아군 메타가 오면 `--friend-meta` | ✅ |
| `tools/seed_label.py` | 검출기로 사전 라벨 + 층화 검수 목록 + 대조 시트 (`--max-expected 2`는 both 세션) | ✅ |
| `tools/eval_cross_color.py` | recall, 중심 오차 px, 높이 오차 px, FP/프레임 — 리스트 단위 평가 | ✅ |
| `tools/make_gray.py` | 흑백 arm용 데이터셋 미러 | ✅ |
| `tools/export_ncnn.py` | .pt → NCNN 4종 | ✅ (x86) |
| `tools/colorlib.py` | Lab, CIEDE2000, 고채도 픽셀 선택, 색 판정 | ✅ |
| `tools/color_separation.py` | 색 분리 검증 → `color_refs.json` | `--labels` 경로만 ✅. `--model` 자동 검출과 both 세션 EL/FL 배정은 **미시운전** |
| `tools/pseudo_bias_check.py` | 박스 높이 편향 검사(h ≈ k/d) | 단일 세트만 ✅. 사람/의사 비교는 **미시운전** |
| `tools/target_pipeline.py` + `pipeline_config.json` | 검출 → 색 판정 → 조준값 → 교전 결정 | ✅ (x86 CPU) |
| `tools/pi_bench.py` | 파이4에서 NCNN 지연 실측 | 파이에서 **미실행** |

## 재현 명령

```bash
source scripts/env.sh
python tools/package_dataset.py                               # 데이터셋 다시 만들기
BATCH=8 WORKERS=8 bash scripts/train_r3.sh                    # r3 (DGX 설정. GB10은 기본 BATCH=32)
BATCH=8 WORKERS=8 bash scripts/train_3arm.sh                  # 3-arm 6 run
python tools/export_ncnn.py --model runs/r3/weights/best.pt --out weights/ncnn_r3 --name r3
python tools/eval_cross_color.py --model runs/r3/weights/best.pt --list handover/dataset/list_test_human.txt --imgsz 1280
bash scripts/make_full_archive.sh                             # 전체를 다시 한 파일로
```
