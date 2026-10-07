#!/usr/bin/env python
"""compare_models.py — 모델 교체 판단용 비교표. 같은 리스트에서 여러 모델을 돌려 나란히 비교한다. 첫 모델이 기준(baseline).

지표 (IoU ≥ --iou 로 GT 와 매칭):
  recall·FP/프레임 : --confs 의 임계값마다
  평균 conf, 중심 오차 중앙값/p95 (yaw 정밀도), 높이 오차 평균·|오차| 중앙값 (피치 정밀도) : 첫 임계값(기본 .25)에서
  짝비교 : 기준 모델과 둘 다 검출한 이미지에서 이미지별 중심 오차·|높이 오차| 차이 (중앙값, 더 좋은/나쁜 장수, Wilcoxon p)
  --watch : 지정한 uid 에서 각 모델의 최고 conf 박스 (conf 0.01 까지 내려서 확인)
판단 원칙(RESUME.md): recall 몇 장 차이보다 중심·높이 오차가 우선. 사람 라벨 리스트(list_test_human.txt)를 1차 기준으로 쓴다.

사용:
  python tools/compare_models.py --list handover/dataset/list_test_human.txt \
      --models r3=runs/r3/weights/best.pt r3_coco=runs/r3_coco/weights/best.pt --watch b1x_enemy_000598 b1x_enemy_000651 --out runs/compare_test_human.md
"""
import argparse, os, sys, cv2, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cross_color import load_list, gt_boxes, iou

def predict_all(model_path, paths, imgsz, conf, device, batch):
    """{uid: [(xyxy, conf), ...] conf 내림차순}"""
    from ultralytics import YOLO
    m = YOLO(model_path); out = {}
    for k in range(0, len(paths), batch):
        chunk = paths[k:k + batch]
        for p, r in zip(chunk, m.predict([cv2.imread(p) for p in chunk], imgsz=imgsz, conf=conf, device=device, verbose=False)):
            out[Path(p).stem] = [] if r.boxes is None else sorted(zip(r.boxes.xyxy.cpu().numpy().astype(float).tolist(), r.boxes.conf.cpu().numpy().astype(float).tolist()), key=lambda t: -t[1])
    return out

def match(preds, gts, conf, iou_thr):
    """이미지 1장: (tp 목록[dict], fp 수, fn 수) — conf 높은 예측부터 IoU 최대인 미매칭 GT 에 배정"""
    used, tp, fp = set(), [], 0
    for b, c in preds:
        if c < conf: continue
        cand = [(iou(b, g), j) for j, g in enumerate(gts) if j not in used]; best = max(cand) if cand else (0, -1)
        if best[0] >= iou_thr:
            g = gts[best[1]]; used.add(best[1])
            tp.append(dict(conf=c, center_err=float(np.hypot((b[0] + b[2] - g[0] - g[2]) / 2, (b[1] + b[3] - g[1] - g[3]) / 2)), h_err=(b[3] - b[1]) - (g[3] - g[1])))
        else: fp += 1
    return tp, fp, len(gts) - len(used)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True); ap.add_argument('--models', nargs='+', required=True, help='이름=경로 ... (첫 번째가 기준)')
    ap.add_argument('--imgsz', type=int, default=1280); ap.add_argument('--confs', type=float, nargs='+', default=[0.25, 0.15, 0.10]); ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='0'); ap.add_argument('--batch', type=int, default=16); ap.add_argument('--watch', nargs='*', default=[]); ap.add_argument('--out', default=None)
    a = ap.parse_args(); dev = a.device if a.device == 'cpu' else int(a.device)
    paths = load_list(a.list); uids = [Path(p).stem for p in paths]; gts = {}
    for p in paths:
        H, W = cv2.imread(p).shape[:2]; gts[Path(p).stem] = gt_boxes(p, W, H)
    n_gt = sum(len(g) for g in gts.values()); models = [m.split('=', 1) for m in a.models]
    low = min(min(a.confs), 0.01 if a.watch else 1.0); rows, per_img, watch, missed = [], {}, [], {}
    for name, path in models:
        P = predict_all(path, paths, a.imgsz, low, dev, a.batch); row = dict(model=name)
        for c in a.confs:
            res = {u: match(P[u], gts[u], c, a.iou) for u in uids}
            row[f'recall@{c:g}'] = sum(len(r[0]) for r in res.values()) / n_gt; row[f'FP/frame@{c:g}'] = sum(r[1] for r in res.values()) / len(uids)
            if c == a.confs[0]:
                tp = pd.DataFrame([dict(uid=u, **t) for u, r in res.items() for t in r[0]]); per_img[name] = tp.groupby('uid').first() if len(tp) else tp
                missed[name] = [u for u, r in res.items() if r[2]]
                row.update(mean_conf=tp.conf.mean(), center_med=tp.center_err.median(), center_p95=tp.center_err.quantile(.95), h_err_mean=tp.h_err.mean(), h_abs_med=tp.h_err.abs().median())
        rows.append(row)
        for u in a.watch:
            best = max(((iou(b, g), c, b) for b, c in P.get(u, []) for g in gts.get(u, [])), default=None, key=lambda t: (t[0] >= a.iou, t[1]))
            watch.append(dict(uid=u, model=name, conf=None if best is None or best[0] < a.iou else round(best[1], 3), iou=None if best is None else round(best[0], 2)))
    T = pd.DataFrame(rows).set_index('model'); c0 = a.confs[0]
    cols = [f'recall@{c:g}' for c in a.confs] + ['mean_conf', 'center_med', 'center_p95', 'h_err_mean', 'h_abs_med'] + [f'FP/frame@{c:g}' for c in a.confs]
    L = [f'# 모델 비교 — {Path(a.list).name} ({len(uids)}장, GT {n_gt}개), imgsz {a.imgsz}, IoU {a.iou}', '', f'정밀도 지표(mean_conf, center, h)는 conf {c0:g} 기준. 단위 px.', '', '```', T[cols].round(3).to_string(), '```', '']
    L += [f'conf {c0:g} 에서 놓친 이미지: ' + ', '.join(f'{k}: {v if v else "없음"}' for k, v in missed.items()), '']
    base = models[0][0]
    for name, _ in models[1:]:
        j = per_img[base].join(per_img[name], lsuffix='_a', rsuffix='_b', how='inner'); L.append(f'## 짝비교 {name} − {base} (둘 다 검출한 {len(j)}장)')
        for label, d in [('중심 오차', j.center_err_b - j.center_err_a), ('|높이 오차|', j.h_err_b.abs() - j.h_err_a.abs())]:
            try:
                from scipy.stats import wilcoxon; p = f'{wilcoxon(d).pvalue:.3f}'
            except Exception: p = 'n/a'
            L.append(f'- {label}: 차이 중앙값 {d.median():+.3f} px, 평균 {d.mean():+.3f} px | {name} 가 더 좋은 장수 {(d < 0).sum()} / 더 나쁜 장수 {(d > 0).sum()} | Wilcoxon p = {p}')
        L.append('')
    if watch: L += ['## 지정 이미지 (conf 0.01 까지 내려서 GT 와 IoU ≥ 기준인 박스의 conf. None = 그런 박스 없음)', '```', pd.DataFrame(watch).pivot(index='uid', columns='model', values='conf')[[m[0] for m in models]].to_string(), '```']
    txt = '\n'.join(L); print(txt)
    if a.out: open(a.out, 'w').write(txt + '\n')

if __name__ == '__main__': main()
