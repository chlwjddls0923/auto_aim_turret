"""
수집 데이터(05_dataset/<라벨>/labels.csv)의 열 구성을 한 곳에서 정한다.

05_sweep_capture.py와 05_manual_capture.py가 같은 폴더에 섞어 저장하므로 두 스크립트가
반드시 같은 열을 써야 한다. 열을 바꿀 때는 여기만 고친다.

    mode : 어느 스크립트로 찍었는지 ("sweep" 또는 "manual")
    yaw_cmd, pitch_cmd : 찍을 때의 서보 명령 각도 (SG90은 실제 각도를 읽을 수 없다)
"""

CSV_COLUMNS = [
    "filename", "label", "session", "mode", "note",
    "yaw_cmd", "pitch_cmd", "timestamp",
    "clip_ratio", "dark_ratio", "mean_bgr", "config_locked",
]
