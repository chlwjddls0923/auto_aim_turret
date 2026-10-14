#!/usr/bin/env python
"""pi_bench.py — 라즈베리파이4에서 NCNN 모델 해상도별 지연시간 실측 (설계 §H 체크리스트 2~3번)
준비: weights/ncnn/ 폴더(ncnn_960x1280, ncnn_736x960, ncnn_480x640, ncnn_384x384)와 실제 프레임 1장을 파이로 복사, `pip install ultralytics` (ncnn 은 자동 설치)
사용: python pi_bench.py --models weights/ncnn --image frame.jpg [--runs 50]
출력: 모델별 p50/p95 ms (ultralytics predict 전체 = 전처리+추론+후처리), 온도/스로틀 상태, 닫힌 루프 추정(전체 1회 + ROI N회)
"""
import argparse, os, subprocess, time, cv2, numpy as np
from ultralytics import YOLO
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True).strip()
    except Exception: return 'n/a'
ap = argparse.ArgumentParser(); ap.add_argument('--models', required=True); ap.add_argument('--image', required=True); ap.add_argument('--runs', type=int, default=50); a = ap.parse_args()
im = cv2.imread(a.image); H, W = im.shape[:2]; roi = im[max(0, H // 2 - 192):H // 2 + 192, max(0, W // 2 - 192):W // 2 + 192]
print('arch', sh('uname -m'), '| temp', sh('vcgencmd measure_temp'), '| throttled', sh('vcgencmd get_throttled'), '| cpus', os.cpu_count())
res = {}
for name, src, sz in [('ncnn_960x1280', im, (960, 1280)), ('ncnn_736x960', im, (736, 960)), ('ncnn_480x640', im, (480, 640)), ('ncnn_384x384', roi, 384)]:
    p = os.path.join(a.models, name)
    if not os.path.isdir(p): print(name, '없음'); continue
    m = YOLO(p, task='detect')
    for _ in range(10): m.predict(src, imgsz=sz, verbose=False)
    t = []
    for _ in range(a.runs):
        t0 = time.perf_counter(); m.predict(src, imgsz=sz, verbose=False); t.append(time.perf_counter() - t0)
    res[name] = (np.median(t) * 1000, np.percentile(t, 95) * 1000); print(f'{name:<16s} p50 {res[name][0]:7.1f} ms  p95 {res[name][1]:7.1f} ms   | temp {sh("vcgencmd measure_temp")}')
if 'ncnn_960x1280' in res and 'ncnn_384x384' in res:
    for n in (5, 10): print(f'닫힌 루프 추정: 전체 1회 + ROI {n}회 = {(res["ncnn_960x1280"][0] + n * res["ncnn_384x384"][0]) / 1000:.2f} s   (전체만 {n}회 = {n * res["ncnn_960x1280"][0] / 1000:.2f} s)')
print('throttled(끝)', sh('vcgencmd get_throttled'), ' ← 0x0 이 아니면 방열/전원 문제, 수치 신뢰 불가')
