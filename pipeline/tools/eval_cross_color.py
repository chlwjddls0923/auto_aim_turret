#!/usr/bin/env python
"""eval_cross_color.py — 교차 색 평가 (인계 브리프 §5-5·6, §6 A-5).
'적군만 학습한 모델이 아군을 얼마나 찾는가'(또는 반대)를 재는 스크립트. mAP 대신 1차 지표를 직접 계산한다:
  recall@IoU(conf ≥ --conf), 매칭 박스 평균 conf, 박스 중심 오차 px(yaw), 높이 오차 px(피치), FP/프레임 — 전체 + 소스별(+거리별).

입력: --list 는 패키지의 list_*.txt (줄 형식 ./train/images/<uid>.jpg, 리스트 파일 위치 기준). 라벨은 images→labels 치환 경로.
      inventory.csv(리스트와 같은 폴더)가 있으면 source/distance_m 로 나눠서 보고.
사용 예:
  python tools/eval_cross_color.py --model runs/3arm/h03_enemy/weights/best.pt --list dataset/list_eval_friend_all.txt --imgsz 1280 \
      --tag h03_enemy --append-csv runs/3arm/summary.csv
주의: 흑백 arm 모델은 흑백 데이터셋(make_gray.py 출력)의 리스트로 평가한다.
"""
import argparse, os, cv2, numpy as np, pandas as pd
from pathlib import Path

def load_list(p):
    parent = str(Path(p).resolve().parent) + os.sep
    return [x.replace('./', parent, 1) if x.startswith('./') else x for x in open(p).read().split('\n') if x.strip()]

def gt_boxes(img_path, W, H):
    s = os.sep; t = (s + 'labels' + s).join(img_path.rsplit(s + 'images' + s, 1)); t = os.path.splitext(t)[0] + '.txt'
    out = []
    for ln in open(t):
        p = ln.split()
        if len(p) >= 5:
            xc, yc, w, h = map(float, p[1:5]); out.append(((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H))
    return out

def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    return iw * ih / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih)

def summarize(g):
    tp = g[g.kind == 'tp']; n_gt = (g.kind != 'fp').sum(); n_img = g.uid.nunique()
    return dict(images=n_img, gt=int(n_gt), recall=len(tp) / n_gt if n_gt else np.nan, mean_conf=tp.conf.mean(), center_err_med=tp.center_err.median(), center_err_p95=tp.center_err.quantile(.95),
                h_err_mean=tp.h_err.mean(), h_abs_err_med=tp.h_err.abs().median(), fp_per_frame=(g.kind == 'fp').sum() / n_img if n_img else np.nan)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--list', required=True)
    ap.add_argument('--imgsz', type=int, default=1280); ap.add_argument('--conf', type=float, default=0.25); ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='0'); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--inventory', default=None, help='기본: 리스트와 같은 폴더의 inventory.csv')
    ap.add_argument('--tag', default=None); ap.add_argument('--append-csv', default=None, help='요약 1행(전체)을 이 CSV 에 추가'); ap.add_argument('--out', default=None, help='박스별 상세 CSV')
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.model); dev = a.device if a.device == 'cpu' else int(a.device); paths = load_list(a.list); rows = []
    for k in range(0, len(paths), a.batch):
        chunk = paths[k:k + a.batch]; ims = [cv2.imread(p) for p in chunk]
        for p, im, r in zip(chunk, ims, m.predict(ims, imgsz=a.imgsz, conf=a.conf, device=dev, verbose=False)):
            H, W = im.shape[:2]; uid = Path(p).stem; gts = gt_boxes(p, W, H); used = set()
            preds = [] if r.boxes is None else sorted(zip(r.boxes.xyxy.cpu().numpy().astype(float).tolist(), r.boxes.conf.cpu().numpy().astype(float).tolist()), key=lambda t: -t[1])
            for b, c in preds:   # conf 높은 예측부터 IoU 최대인 미매칭 GT 에 배정
                cand = [(iou(b, g), j) for j, g in enumerate(gts) if j not in used]; best = max(cand) if cand else (0, -1)
                if best[0] >= a.iou:
                    g = gts[best[1]]; used.add(best[1])
                    rows.append(dict(uid=uid, kind='tp', conf=c, iou=best[0], center_err=float(np.hypot((b[0] + b[2] - g[0] - g[2]) / 2, (b[1] + b[3] - g[1] - g[3]) / 2)), h_err=(b[3] - b[1]) - (g[3] - g[1])))
                else: rows.append(dict(uid=uid, kind='fp', conf=c, iou=best[0]))
            rows += [dict(uid=uid, kind='fn') for j in range(len(gts)) if j not in used]
            if not gts and not preds: rows.append(dict(uid=uid, kind='empty'))
    D = pd.DataFrame(rows, columns=['uid', 'kind', 'conf', 'iou', 'center_err', 'h_err']); D = D[D.kind != 'empty']   # 검출 0건이어도 열 유지
    inv_p = a.inventory or str(Path(a.list).resolve().parent / 'inventory.csv')
    if os.path.exists(inv_p): D = D.merge(pd.read_csv(inv_p)[['uid', 'source', 'distance_m', 'label_source']], on='uid', how='left')
    else: D['source'] = 'all'
    if a.out: D.to_csv(a.out, index=False)
    allrow = summarize(D); tab = {'ALL': allrow} | {f'source={s}': summarize(g) for s, g in D.groupby('source')}
    if 'distance_m' in D and D.distance_m.notna().any(): tab |= {f'{s} {d}m': summarize(g) for (s, d), g in D[D.distance_m.notna()].groupby(['source', 'distance_m'])}
    T = pd.DataFrame(tab).T; T[['images', 'gt']] = T[['images', 'gt']].astype(int)
    print(f'model={a.model}\nlist={a.list} ({len(paths)}장) imgsz={a.imgsz} conf={a.conf} IoU={a.iou}')
    print(T.round(3).to_string())
    if 'label_source' in D and (D.label_source.dropna() != 'human_labelimg').any(): print('※ GT 에 의사 라벨 포함 — 수치는 상대 비교용(자기 평가 편향).')
    if a.append_csv:
        row = pd.DataFrame([dict(tag=a.tag or Path(a.model).parent.parent.name, model=a.model, list=Path(a.list).name, imgsz=a.imgsz, conf=a.conf, **allrow)])
        os.makedirs(os.path.dirname(os.path.abspath(a.append_csv)), exist_ok=True); row.to_csv(a.append_csv, mode='a', index=False, header=not os.path.exists(a.append_csv))

if __name__ == '__main__': main()
