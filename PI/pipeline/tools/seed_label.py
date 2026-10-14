#!/usr/bin/env python
"""seed_label.py — 검출기로 사전 라벨(의사 라벨) 생성 + 층화 검수 목록 + 대조 시트. **다중 박스(both 세션) 지원.**

사용 예:
  python tools/seed_label.py --images "data2/friend(26.10.02.1839)" --model runs/r2/weights/best.pt --out friend_pseudo \
      --imgsz 640 --hue-shift-fallback 22 --review-n 150 [--meta path/labels.csv] [--max-expected 1|2]
출력(out/):
  labels/<stem>.txt   YOLO 형식, 이미지의 **모든** 박스 (검출 없음 → 빈 파일)
  xml/<stem>.xml      LabelImg 형식 (검수용)
  predictions.csv     박스 1개 = 1행 (filename, conf, x1..y2, used_fallback)
  images.csv          이미지 1장 = 1행 (status: confident / review / none, n_boxes, best_conf, strata)
  review_list.csv     사람 검수 목록 (층화 추출; review·none 우선, 나머지 confident에서 비례 추출)
  sheet_<status>.jpg  대조 시트 (박스 주변 확대 crop)
status 규칙: none = 유지 임계(conf-keep) 이상 박스 없음 / review = 박스 수 > max-expected 또는 conf < conf-confident 또는 색상 회전 fallback으로만 검출 / confident = 그 외
"""
import argparse, os, cv2, numpy as np, pandas as pd
from pathlib import Path

def shift_hue(im, s):
    if not s: return im
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); hsv[..., 0] = ((hsv[..., 0].astype(np.int32) + s) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def extract(r, keep):
    if r.boxes is None or len(r.boxes) == 0: return []
    xy = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
    out = [(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(cc)) for a, cc in zip(xy, c) if cc >= keep]
    return sorted(out, key=lambda t: -t[4])

def xml_doc(fname, W, H, boxes, cls):
    objs = ''.join(f'<object><name>{cls}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>'
                   f'<bndbox><xmin>{int(round(b[0]))}</xmin><ymin>{int(round(b[1]))}</ymin><xmax>{int(round(b[2]))}</xmax><ymax>{int(round(b[3]))}</ymax></bndbox></object>' for b in boxes)
    return (f'<annotation><folder>images</folder><filename>{fname}</filename><size><width>{W}</width><height>{H}</height><depth>3</depth></size>'
            f'<segmented>0</segmented>{objs}</annotation>\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--meta', default=None, help='labels.csv (filename, session, note ...) — 있으면 session으로 층화')
    ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--device', default='0')
    ap.add_argument('--conf-low', type=float, default=0.1); ap.add_argument('--conf-keep', type=float, default=0.25); ap.add_argument('--conf-confident', type=float, default=0.6)
    ap.add_argument('--hue-shift-fallback', type=int, default=0, help='OpenCV hue 단위(0~179). 미검출 이미지에만 색상 회전 후 재시도. 0=끔')
    ap.add_argument('--max-expected', type=int, default=1, help='이미지당 기대 객체 수 (both 세션=2). 초과 시 review')
    ap.add_argument('--review-n', type=int, default=150); ap.add_argument('--class-name', default='tar'); ap.add_argument('--sheet-max', type=int, default=48)
    ap.add_argument('--h-to-dist', type=float, default=59.1, help='거리 추정 상수: d(m)=k/h(px). 메타 없을 때 층화에 사용')
    a = ap.parse_args()
    from ultralytics import YOLO
    imgs = sorted(Path(a.images).rglob('*.jpg')); out = Path(a.out)
    for d in ('labels', 'xml'): (out / d).mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(a.meta) if a.meta else None
    m = YOLO(a.model); dev = a.device if a.device == 'cpu' else int(a.device)
    rows, imrows = [], []
    for k in range(0, len(imgs), 16):
        chunk = imgs[k:k + 16]; ims = [cv2.imread(str(p)) for p in chunk]
        rs = m.predict(ims, imgsz=a.imgsz, conf=a.conf_low, device=dev, verbose=False)
        boxes_all = [extract(r, a.conf_keep) for r in rs]
        need = [i for i, b in enumerate(boxes_all) if not b]
        fb = {}
        if a.hue_shift_fallback and need:
            fr = m.predict([shift_hue(ims[i], a.hue_shift_fallback) for i in need], imgsz=a.imgsz, conf=a.conf_low, device=dev, verbose=False)
            fb = {i: extract(r, a.conf_keep) for i, r in zip(need, fr)}
        for i, p in enumerate(chunk):
            used_fb = False; boxes = boxes_all[i]
            if not boxes and fb.get(i): boxes = fb[i]; used_fb = True
            H, W = ims[i].shape[:2]; stem = p.stem
            if not boxes: status = 'none'
            elif len(boxes) > a.max_expected or min(b[4] for b in boxes) < a.conf_confident or used_fb: status = 'review'
            else: status = 'confident'
            with open(out / 'labels' / f'{stem}.txt', 'w') as f:
                for b in boxes: f.write(f'0 {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}\n')
            open(out / 'xml' / f'{stem}.xml', 'w').write(xml_doc(p.name, W, H, boxes, a.class_name))
            for j, b in enumerate(boxes): rows.append(dict(filename=p.name, box_id=j, conf=round(b[4], 4), x1=round(b[0], 1), y1=round(b[1], 1), x2=round(b[2], 1), y2=round(b[3], 1), used_fallback=used_fb))
            best = boxes[0] if boxes else None
            imrows.append(dict(filename=p.name, path=str(p), status=status, n_boxes=len(boxes), best_conf=round(best[4], 4) if best else 0.0,
                               h=round(best[3]-best[1], 1) if best else None, cx=round((best[0]+best[2])/2, 1) if best else None, used_fallback=used_fb))
    pred = pd.DataFrame(rows); im = pd.DataFrame(imrows)
    # 층화 키
    if meta is not None and 'session' in meta.columns:
        im = im.merge(meta[['filename', 'session'] + (['note'] if 'note' in meta.columns else [])], on='filename', how='left'); im['strata'] = im['session'].astype(str)
    else:
        bk = np.array([1.2, 1.5, 1.8, 2.1, 2.4]); d_est = a.h_to_dist / im.h.astype(float)
        im['dist_est'] = np.where(im.h.notna(), bk[np.abs(d_est.fillna(1.8).values[:, None] - bk).argmin(1)], np.nan)
        im['third'] = pd.cut(im.cx.astype(float), [0, 427, 853, 1280], labels=['left3', 'center3', 'right3']).astype(str)
        im['strata'] = np.where(im.status == 'none', 'none', im.dist_est.astype(str) + '_' + im.third)
    pred.to_csv(out / 'predictions.csv', index=False); im.to_csv(out / 'images.csv', index=False)
    # 검수 목록: review/none 전부 + confident에서 strata 비례
    must = im[im.status != 'confident']; rest = im[im.status == 'confident']; n_rest = max(0, a.review_n - len(must))
    picks = [must]
    if n_rest and len(rest):
        q = (rest.groupby('strata').size() / len(rest) * n_rest).round().astype(int).clip(lower=1)
        picks += [g.sample(min(len(g), int(q.get(s, 1))), random_state=0) for s, g in rest.groupby('strata')]
    review = pd.concat(picks).drop_duplicates('filename'); review.to_csv(out / 'review_list.csv', index=False)
    # 대조 시트
    for status, g in im.groupby('status'):
        g = g.sample(min(len(g), a.sheet_max), random_state=0).sort_values('filename'); tiles = []
        for _, r in g.iterrows():
            img = cv2.imread(r.path)
            if status == 'none':
                t = cv2.resize(img, (320, 240)); cv2.putText(t, r.filename[-10:-4], (3, 14), 0, 0.45, (0, 255, 255), 1); tiles.append(t); continue
            bx = pred[pred.filename == r.filename]
            for _, b in bx.iterrows():
                cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2; x0, y0 = int(np.clip(cx - 60, 0, 1160)), int(np.clip(cy - 45, 0, 870))
                t = cv2.resize(img[y0:y0 + 90, x0:x0 + 120], (320, 240), interpolation=cv2.INTER_CUBIC); s = 320 / 120
                cv2.rectangle(t, (int((b.x1 - x0) * s), int((b.y1 - y0) * s)), (int((b.x2 - x0) * s), int((b.y2 - y0) * s)), (0, 0, 255), 1)
                cv2.putText(t, f'{r.filename[-10:-4]} {b.conf:.2f} n{r.n_boxes}{" fb" if r.used_fallback else ""}', (3, 14), 0, 0.45, (0, 255, 255), 1); tiles.append(t)
        tiles = tiles[:a.sheet_max]
        if not tiles: continue
        while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f'sheet_{status}.jpg'), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('status:', im.status.value_counts().to_dict(), '| boxes:', len(pred), '| fallback 사용:', int(im.used_fallback.sum()))
    print('strata x status:'); print(pd.crosstab(im.strata, im.status).to_string())
    print(f'review_list: {len(review)}장 (review/none {len(must)} + confident 표본 {len(review)-len(must)}) -> {out/"review_list.csv"}')

if __name__ == '__main__': main()
