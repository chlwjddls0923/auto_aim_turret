#!/usr/bin/env python
"""color_separation.py — 아군/적군 색 분리 검증 (설계 §1-4). 다른 모든 작업보다 먼저 통과해야 하는 게이트.

입력 폴더 규약 (05_manual_capture.py 출력 그대로, --dir 에 여러 폴더를 줄 수 있다 — 폴더마다 labels.csv):
  <dir>/images/*.png (또는 평평하게), <dir>/labels.csv  열: filename,label(enemy|friend|both|negative),note,...
  색 측정은 PNG/raw 만 쓴다 (10/4 재설계 §7). JPEG 는 --allow-jpeg 를 줄 때만 받고(시운전),
  report.md 와 color_refs.json 에 jpeg_input=true 를 남긴다.  --exclude-note test 처럼 note 로 세션을 뺄 수 있다.
  note 예: painted_center_1.5m / painted_left_2.0m / painted_both_EL-FR_1.75m (EL=enemy left, FL=friend left)  — 거리는 note의 숫자(m)에서 파싱
박스 소스:
  --model <pt|ncnn dir>  : 검출기로 자동 (기본)          --labels <dir> : YOLO txt 라벨 사용 (사람 라벨 있을 때)
추가 음성 색 샘플(분리수거함 파란 라벨 등): --extra-rois rois.csv  열: filename,x1,y1,x2,y2,label
색 픽셀 선택: 박스 안 채도 ≥ --sat-min(기본 60/255) [+ --top-frac 상위 비율] [+ --ball-frac 상단 비율: 색칠 전 데이터 0.45, 색칠 후 생략]

사용 예:
  python tools/color_separation.py --dir data3/painted_test --model runs/r3/weights/best.pt --out color_report --extra-rois bin_rois.csv
출력: out/samples.csv(박스별 Lab/HSV/n_px), out/report.md, out/ab_scatter.png, out/color_refs.json(파이프라인 설정에 그대로 사용)
판정 기준: 클래스 간 dE2000 ≥ 3 × max(클래스 내 RMS 흩어짐) → 통과.  제안값: margin = max(p95 흩어짐), min_px = 가장 먼 거리에서 n_px p05
"""
import argparse, json, os, re, sys, cv2, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import colorlib as C

def tbl(df):
    """DataFrame → markdown 표. tabulate 가 없으면 고정폭 텍스트(코드 블록)로 폴백."""
    try: return df.to_markdown()
    except ImportError: return '```\n' + df.to_string() + '\n```'

def parse_dist(note):
    m = re.search(r'(\d+(?:\.\d+)?)\s*m', str(note)); return float(m.group(1)) if m else np.nan

def boxes_from_labels(txt, W, H):
    if not os.path.exists(txt): return []
    out = []
    for ln in open(txt):
        p = ln.split()
        if len(p) < 5: continue
        xc, yc, w, h = map(float, p[1:5]); out.append((((xc - w / 2) * W), ((yc - h / 2) * H), ((xc + w / 2) * W), ((yc + h / 2) * H), 1.0))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--allow-jpeg', action='store_true', help='JPEG 입력 허용 (시운전 전용, 결과에 jpeg_input=true)')
    ap.add_argument('--exclude-note', nargs='*', default=[], help='이 note 의 행은 뺀다 (예: test)')
    ap.add_argument('--model', default=None); ap.add_argument('--labels', default=None); ap.add_argument('--imgsz', type=int, default=1280); ap.add_argument('--device', default='cpu')
    ap.add_argument('--extra-rois', default=None); ap.add_argument('--sat-min', type=int, default=60); ap.add_argument('--top-frac', type=float, default=None); ap.add_argument('--ball-frac', type=float, default=None)
    ap.add_argument('--min-px-floor', type=int, default=20); ap.add_argument('--ratio-pass', type=float, default=3.0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    metas, imgs = [], {}
    for d in map(Path, a.dir):
        for lc in d.rglob('labels.csv'): metas.append(pd.read_csv(lc))
        for p in d.rglob('*'):
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg'): imgs[p.name] = p
    meta = pd.concat(metas, ignore_index=True)
    if a.exclude_note: meta = meta[~meta['note'].astype(str).isin(a.exclude_note)]
    meta = meta[meta.filename.isin(imgs)]
    jpeg_input = bool(any(imgs[f].suffix.lower() in ('.jpg', '.jpeg') for f in meta.filename))
    if jpeg_input and not a.allow_jpeg:
        sys.exit('JPEG 입력입니다. 색 측정은 PNG/raw 로만 합니다 (시운전이면 --allow-jpeg).')
    print(f'입력: {len(meta)}장 ({", ".join(map(str, a.dir))})' + (' — jpeg_input=true (시운전)' if jpeg_input else ''), file=sys.stderr)
    model = None
    if a.model and not a.labels:
        from ultralytics import YOLO; model = YOLO(a.model)
    rows = []
    for _, r in meta.iterrows():
        if r.filename not in imgs: continue
        im = cv2.imread(str(imgs[r.filename])); H, W = im.shape[:2]; lab = str(r.label).lower(); note = str(r.get('note', ''))
        if lab == 'negative': continue
        if a.labels: boxes = boxes_from_labels(os.path.join(a.labels, Path(r.filename).stem + '.txt'), W, H)
        else:
            res = model.predict(im, imgsz=a.imgsz, conf=0.25, device=a.device if a.device == 'cpu' else int(a.device), verbose=False)[0]
            boxes = [] if res.boxes is None else [(*map(float, b), float(c)) for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
        boxes = sorted(boxes, key=lambda b: b[0])  # 왼쪽부터
        # both 세션: note의 EL/FL 로 좌우 배정. 표기 없으면 참조색 산출에서 제외(cls='both?')
        if lab == 'both':
            if len(boxes) != 2: cls_list = ['both?'] * len(boxes)
            elif 'EL' in note.upper(): cls_list = ['enemy', 'friend']
            elif 'FL' in note.upper(): cls_list = ['friend', 'enemy']
            else: cls_list = ['both?', 'both?']
        else: cls_list = [lab] * len(boxes)
        for (x1, y1, x2, y2, c), cls in zip(boxes, cls_list):
            st = C.patch_stats(im, (x1, y1, x2, y2), sat_min=a.sat_min, top_frac=a.top_frac, ball_frac=a.ball_frac)
            cx = (x1 + x2) / 2; pos = 'left' if cx < W / 3 else ('right' if cx > 2 * W / 3 else 'center')
            rows.append(dict(filename=r.filename, cls=cls, note=note, dist=parse_dist(note), pos=pos, conf=round(c, 3), x1=x1, y1=y1, x2=x2, y2=y2, h=y2 - y1,
                             n_px=st['n_px'], hue=st['hue_deg'], S=st['S'], V=st['V'], L=None if st['lab'] is None else st['lab'][0], a=None if st['lab'] is None else st['lab'][1], b=None if st['lab'] is None else st['lab'][2]))
    if a.extra_rois:
        for _, r in pd.read_csv(a.extra_rois).iterrows():
            if r.filename not in imgs: continue
            im = cv2.imread(str(imgs[r.filename])); st = C.patch_stats(im, (r.x1, r.y1, r.x2, r.y2), sat_min=a.sat_min, top_frac=a.top_frac)
            rows.append(dict(filename=r.filename, cls=str(r.label), note='extra', dist=np.nan, pos='-', conf=1.0, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, h=r.y2 - r.y1, n_px=st['n_px'], hue=st['hue_deg'], S=st['S'], V=st['V'],
                             L=None if st['lab'] is None else st['lab'][0], a=None if st['lab'] is None else st['lab'][1], b=None if st['lab'] is None else st['lab'][2]))
    S = pd.DataFrame(rows); S.to_csv(out / 'samples.csv', index=False)
    ok = S[(S.n_px >= 5) & S.L.notna() & S.cls.isin(['enemy', 'friend'])]
    rep = C.separation_report({c: g[['L', 'a', 'b']].values.astype(float) for c, g in S[(S.n_px >= 5) & S.L.notna()].groupby('cls')})
    far = ok.dist.max(); n_far = ok[ok.dist == far].groupby('cls').n_px.quantile(0.05) if not np.isnan(far) else ok.groupby('cls').n_px.quantile(0.05)
    min_px = int(max(a.min_px_floor, np.floor(n_far.min()))) if len(n_far) else a.min_px_floor
    # 모의 판정(leave-one-out 생략: refs=평균)
    refs = {k: rep['mean'][k].tolist() for k in ('enemy', 'friend') if k in rep['mean']}
    if len(refs) == 2:
        S['pred'] = [C.classify(dict(n_px=r.n_px, lab=None if pd.isna(r.L) else np.array([r.L, r.a, r.b])), refs, min_px=min_px, margin=rep['suggested_margin'])['side'] for _, r in S.iterrows()]
    md = [f"# 색 분리 검증 보고 ({', '.join(map(str, a.dir))})", ''] + (['**jpeg_input=true — JPEG 시운전 결과다. 색 선택·기준색으로 쓰지 않는다.**', ''] if jpeg_input else []) + [f"샘플: {S.cls.value_counts().to_dict()} | 픽셀 선택: sat_min={a.sat_min}, top_frac={a.top_frac}, ball_frac={a.ball_frac}", '',
          '## 클래스별 평균 (고채도 픽셀)', tbl(ok.groupby('cls').agg(n=('n_px', 'size'), n_px_med=('n_px', 'median'), hue=('hue', 'median'), S=('S', 'median'), V=('V', 'median'), L=('L', 'mean'), a=('a', 'mean'), b=('b', 'mean')).round(2)), '',
          '## 거리 × 위치별 (n_px 중앙값 / V 중앙값 / hue 중앙값)', tbl(ok.pivot_table(index=['cls', 'dist'], columns='pos', values=['n_px', 'V', 'hue'], aggfunc='median').round(2)), '',
          '## 분리도', f"- 클래스 간 dE2000: **{rep['between'].get('enemy|friend', float('nan')):.1f}**",
          f"- 클래스 내 흩어짐 RMS: enemy {rep['spread_rms'].get('enemy', float('nan')):.1f} / friend {rep['spread_rms'].get('friend', float('nan')):.1f} (p95 {rep['spread_p95'].get('enemy', float('nan')):.1f} / {rep['spread_p95'].get('friend', float('nan')):.1f})",
          f"- 안전비 = {rep.get('safety_ratio', float('nan')):.2f} (기준 {a.ratio_pass}) → **{'통과' if rep.get('safety_ratio', 0) >= a.ratio_pass else '미달 — 색 변경 또는 노출 재조정 필요'}**",
          f"- 제안: margin = {rep.get('suggested_margin', float('nan')):.1f} (p95 흩어짐), min_px = {min_px} (최원거리 n_px p05, 하한 {a.min_px_floor})"]
    others = [k for k in rep['mean'] if k not in ('enemy', 'friend')]
    if others: md += ['', '## 음성 색 샘플과의 거리 (dE2000)'] + [f"- {k}: enemy까지 {rep['between'].get(f'enemy|{k}', rep['between'].get(f'{k}|enemy', float('nan'))):.1f}, friend까지 {rep['between'].get(f'friend|{k}', rep['between'].get(f'{k}|friend', float('nan'))):.1f}" for k in others]
    if 'pred' in S: md += ['', '## 모의 판정 (refs=측정 평균)', tbl(pd.crosstab(S.cls, S.pred, normalize='index').round(3)), '', '### 거리별 오판정/판정불가', tbl(S.assign(wrong=((S.cls == 'enemy') & (S.pred == 'friend')) | ((S.cls == 'friend') & (S.pred == 'enemy')), unk=S.pred == 'unknown').pivot_table(index='dist', columns='cls', values=['wrong', 'unk'], aggfunc='mean').round(3))]
    open(out / 'report.md', 'w').write('\n'.join(md) + '\n')
    json.dump(dict(refs=refs, min_px=min_px, margin=float(rep.get('suggested_margin', 5.0)), sat_min=a.sat_min, top_frac=a.top_frac, ball_frac=a.ball_frac, safety_ratio=float(rep.get('safety_ratio', 0)), source=[str(x) for x in a.dir], jpeg_input=jpeg_input), open(out / 'color_refs.json', 'w'), indent=1)
    # a*b* 산점도 (cv2)
    img = np.full((600, 600, 3), 255, np.uint8); col = {'enemy': (200, 60, 160), 'friend': (160, 160, 20), 'both?': (0, 0, 0)}
    cv2.line(img, (300, 0), (300, 600), (200, 200, 200), 1); cv2.line(img, (0, 300), (600, 300), (200, 200, 200), 1)
    for _, r in S[S.L.notna()].iterrows(): cv2.circle(img, (int(300 + r.a * 4), int(300 - r.b * 4)), 3, col.get(r.cls, (0, 0, 255)), -1)
    cv2.putText(img, 'a* ->', (540, 315), 0, 0.5, (0, 0, 0), 1); cv2.putText(img, 'b*', (305, 15), 0, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(out / 'ab_scatter.png'), img)
    print('\n'.join(md))

if __name__ == '__main__': main()
