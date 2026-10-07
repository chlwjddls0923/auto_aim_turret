#!/usr/bin/env python
"""xml2yolo.py — 사람이 LabelImg 로 검수한 XML 을 YOLO txt 라벨로 반영하고, 무엇이 바뀌었는지 보고한다.
검수 흐름: LabelImg 에서 이미지 폴더와 XML 폴더(friend_pseudo/xml, none 25장은 friend_pseudo/none_recheck/xml)를 열어 박스를 고치고 저장
          → 이 스크립트로 txt 반영 → python tools/package_dataset.py 로 데이터셋 재생성.
클래스명은 'tar' 만 허용. 박스가 없는 XML 은 빈 txt 가 된다(타겟 없음으로 확정한 경우).

사용:
  python tools/xml2yolo.py --xml friend_pseudo/xml --labels friend_pseudo/labels [--only friend_pseudo/review_list.csv] [--dry-run]
  --only : filename 열이 있는 CSV. 그 목록의 파일만 반영(나머지 XML 은 건드리지 않음)
"""
import argparse, os
import xml.etree.ElementTree as ET
from pathlib import Path

def read_xml(p):
    r = ET.parse(p).getroot(); W, H = float(r.find('size/width').text), float(r.find('size/height').text); out = []
    for o in r.iter('object'):
        if o.find('name').text != 'tar': raise SystemExit(f'{p}: 클래스 {o.find("name").text!r} (tar 만 허용)')
        b = o.find('bndbox'); x1, y1, x2, y2 = (float(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        x1, x2 = max(0, x1), min(W, x2); y1, y2 = max(0, y1), min(H, y2); out.append((x1, y1, x2, y2))
    return out, W, H

def read_txt(p, W, H):
    if not os.path.exists(p): return None
    out = []
    for ln in open(p):
        q = ln.split()
        if len(q) >= 5: xc, yc, w, h = map(float, q[1:5]); out.append(((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--xml', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--only', default=None); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--tol', type=float, default=1.5, help='이 px 이하의 차이는 XML 정수 반올림으로 보고 바뀐 것으로 치지 않는다')
    a = ap.parse_args(); only = None
    if a.only:
        import csv; only = {Path(r['filename']).stem for r in csv.DictReader(open(a.only))}
    n = dict(same=0, moved=0, count_changed=0, new=0); changed = []
    for x in sorted(Path(a.xml).glob('*.xml')):
        if only is not None and x.stem not in only: continue
        boxes, W, H = read_xml(x); t = Path(a.labels) / f'{x.stem}.txt'; old = read_txt(t, W, H)
        if old is None: kind = 'new'
        elif len(old) != len(boxes): kind = 'count_changed'
        else:
            d = max((max(abs(p - q) for p, q in zip(b, o)) for b, o in zip(sorted(boxes), sorted(old))), default=0); kind = 'same' if d <= a.tol else 'moved'
        n[kind] += 1
        if kind == 'same': continue   # 반올림 차이뿐이면 원래 txt(소수 좌표)를 유지
        changed.append(f'{x.stem}: {kind} ({0 if old is None else len(old)} → {len(boxes)} 박스)')
        if not a.dry_run:
            open(t, 'w').write(''.join(f'0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n' for b in boxes))
    print(f"XML {sum(n.values())}개: 변화 없음 {n['same']} / 박스 이동 {n['moved']} / 박스 수 변경 {n['count_changed']} / 새 라벨 {n['new']}" + (' (dry-run: 쓰지 않음)' if a.dry_run else ''))
    print('\n'.join('  ' + c for c in changed[:60]))

if __name__ == '__main__': main()
