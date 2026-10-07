# tools/ 변경 기록

`tools/`의 원본은 파이(`~/src/pipeline/tools/`)다. 수정하면 여기에 날짜와 내용을 한 줄 적는다. 동기화 절차는 PI_SETUP.md §9.

- 2026-10-04 (DGX): 파이용 묶음 최초 배포. `target_pipeline.py`의 설정 경로에서 `~`와 환경변수를 풀도록 하고, CLI의 stdout에 JSON만 나오게 함(모델 로딩 로그는 stderr). `pipeline_config.json`은 장비별 파일(파이 = NCNN 폴더, GB10 = `.pt`).
