#!/usr/bin/env python
"""pseudo_bias_check.py — 의사 라벨 박스 크기(높이 h) 편향 검사 (설계 §2-1).
h 는 피치 보정표의 입력이므로 체계적으로 작거나 크면 조준이 틀어진다.
검사: (1) h ≈ k / 거리 가 원점을 지나는 직선인지 (거리별 잔차), (2) 사람 라벨 vs 의사 라벨의 k 와 거리별 h 중앙값 비교, (3) h 의 화면 위치(cx) 의존성.
입력: --set 이름:라벨dir:labels.csv [--set ...]  (라벨 YOLO txt, labels.csv 의 note 에서 거리 파싱; 파일명 stem 으로 매칭)
사용 예:
  python tools/pseudo_bias_check.py --set pseudo:data2_labels/labels:"data2/enemy(26.10.02.1754)/labels.csv" --set human:review/labels:"data2/enemy(26.10.02.1754)/labels.csv"
"""
import argparse, os, re, glob, numpy as np, pandas as pd
W, H = 1280, 960
def parse_dist(note):
    m = re.search(r'(\d+(?:\.\d+)?)\s*m?', str(note)); return float(m.group(1)) if m else np.nan
def load(name, ldir, csv):
    meta = pd.read_csv(csv); rows = []
    for _, r in meta.iterrows():
        t = os.path.join(ldir, os.path.splitext(r.filename)[0] + '.txt')
        if not os.path.exists(t): continue
        for ln in open(t):
            p = ln.split()
            if len(p) < 5: continue
            xc, yc, w, h = map(float, p[1:5]); rows.append(dict(set=name, filename=r.filename, note=r.get('note', ''), dist=parse_dist(r.get('note', '')), h=h * H, w=w * W, cx=xc * W, cy=yc * H))
    return pd.DataFrame(rows)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--set', action='append', required=True); a = ap.parse_args()
    D = pd.concat([load(*s.split(':', 2)) for s in a.set]); D = D[D.dist.notna()]
    print(f"박스 {len(D)}개, 세트: {D.set.value_counts().to_dict()}\n")
    ks = {}
    for s, g in D.groupby('set'):
        x = 1 / g.dist.values; y = g.h.values; k = (x * y).sum() / (x * x).sum(); ks[s] = k
        r = y - k * x; A = np.vstack([x, np.ones_like(x)]).T; k2, c2 = np.linalg.lstsq(A, y, rcond=None)[0]
        print(f"[{s}] h ≈ {k:.1f}/d  | 잔차 med {np.median(np.abs(r)):.2f}px | 절편 허용 적합: h = {k2:.1f}/d + ({c2:+.1f})  (절편 |c| > 2px 이면 '원점 통과' 가정 위반)")
        print('   거리별 h 중앙값 / 평균잔차(px):', g.assign(r=r).groupby('dist').agg(h=('h', 'median'), res=('r', 'mean'), n=('h', 'size')).round(2).T.to_string().replace('\n', '\n   '))
        # cx 의존성: 좌/중/우 3등분 h 중앙값 (같은 거리 내)
        g2 = g.assign(third=pd.cut(g.cx, [0, 427, 853, 1280], labels=['L', 'C', 'R']))
        print('   화면 위치별 h 중앙값:'); print('   ' + g2.pivot_table(index='dist', columns='third', values='h', aggfunc='median').round(1).to_string().replace('\n', '\n   '))
        print()
    if len(ks) >= 2:
        names = list(ks); print('세트 간 k 비교:', {n: round(ks[n], 1) for n in names})
        base = [n for n in names if 'human' in n.lower()] or names[:1]; b = base[0]
        for n in names:
            if n == b: continue
            ratio = ks[n] / ks[b]; print(f"  {n}/{b} = {ratio:.3f}  →  {'보정 불필요 (±2% 이내)' if abs(ratio-1) < 0.02 else f'보정 계수 {1/ratio:.3f} 를 {n} h 에 곱하거나 해당 구간 사람 라벨로 교체'}")
        # 공통 이미지에서 직접 비교
        piv = D.groupby(['filename', 'set']).h.median().unstack()
        if piv.shape[1] >= 2 and piv.dropna().shape[0] >= 5:
            both = piv.dropna(); d = both.iloc[:, 0] - both.iloc[:, 1]
            print(f"  공통 이미지 {len(both)}장: h 차이({both.columns[0]}−{both.columns[1]}) 평균 {d.mean():+.2f}px, 중앙값 {d.median():+.2f}px, |차|p95 {d.abs().quantile(.95):.2f}px")
if __name__ == '__main__': main()
