#!/usr/bin/env python
"""package_dataset.py — GB10 전송용 데이터셋 패키징 (인계 브리프 §6 A-1).

입력(전부 1280×960):
  b1   data/images/*.jpg + data/images/labellmg/*.xml (사람 라벨)              → uid b1_<원본stem>, 전부 train
  b2   data2/enemy(...)/{images,labels.csv} + data2_labels/labels/*.txt (의사)  → uid b2_<session 시각 6자리>_<idx 6자리>
  아군 data2/friend(...)/*.jpg + friend_pseudo/{labels,images.csv}              → uid fr_na_<idx 6자리>
       (--friend-meta labels.csv 가 있으면 fr_<session 시각 6자리>_<idx>, val 은 --friend-val-notes 의 note 로 결정)
       640 에서 놓친 none 25장도 포함: r2 @1280 재확인 라벨이 friend_pseudo/labels 에 들어 있다(기록은 none_recheck/). 사람 검수 대기.
       images.csv 의 status 가 none 인데 라벨이 비어 있으면(검수에서 타겟 없음으로 확정) 그 이미지는 제외
  b1x  data/images(1)/*.jpg + labellmg/*.xml (사람 라벨 98장, 학습에 쓴 적 없음)  → uid b1x_<원본stem>, 전부 test (학습·val 미포함)
val: b2 note ∈ {center_2.4m, right_1.5m} + 아군 strata ∈ {1.5_right3, 2.4_left3} (아군은 박스 기반 추정)

출력(out/): train|val|test/{images,labels}, data.yaml + data_{enemy,friend}_only.yaml + eval_{friend,enemy}_all.yaml + eval_test_human.yaml
           (path 키 없음 → ultralytics 가 yaml 이 있는 폴더를 데이터셋 루트로 사용: 폴더를 옮겨도 그대로 동작), list_*.txt, split.json, inventory.csv
inventory 의 review 열: 'pending' = friend_pseudo/review_list.csv 의 사람 검수 대상(149장), 'pending:2nd_box_removed' = 재확인에서 2번째 박스를 지운 2장
이미지는 복사(symlink 금지). sanity 실패 시 종료 코드 1.

사용: python tools/package_dataset.py [--out handover/dataset] [--friend-meta <labels.csv>]
"""
import argparse, hashlib, json, re, shutil, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 960
B1_IMG, B1_XML = ROOT / 'data/images', ROOT / 'data/images/labellmg'
B1X_IMG, B1X_XML = ROOT / 'data/images(1)', ROOT / 'data/images(1)/labellmg'
B2_DIR, B2_LBL = ROOT / 'data2/enemy(26.10.02.1754)', ROOT / 'data2_labels'
FR_DIR, FR_PSEUDO = ROOT / 'data2/friend(26.10.02.1839)', ROOT / 'friend_pseudo'
VAL_B2_NOTES = ['center_2.4m', 'right_1.5m']
VAL_FR_STRATA = ['1.5_right3', '2.4_left3']
SPLITS = ('train', 'val', 'test')
COLS = ['uid', 'split', 'source', 'orig_filename', 'session', 'note', 'distance_m', 'placement', 'label_source', 'status', 'n_boxes', 'review']

def idx_of(name): return int(re.search(r'(\d+)\.jpg$', name).group(1))
def dist_of(note):
    m = re.search(r'(\d+(?:\.\d+)?)', str(note)); return float(m.group(1)) if m else ''

def xml_to_yolo(p):
    """LabelImg xml → YOLO 줄 목록. 클래스명은 'tar' 하나만 허용."""
    out = []
    for o in ET.parse(p).getroot().iter('object'):
        assert o.find('name').text == 'tar', f'{p}: 클래스 {o.find("name").text}'
        b = o.find('bndbox'); x1, y1, x2, y2 = (float(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        x1, x2 = max(0, x1), min(W, x2); y1, y2 = max(0, y1), min(H, y2)
        out.append(f'0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
    return out

def collect(a):
    """(src_image, label_lines, inventory_row) 목록"""
    items = []
    # b1 — 사람 라벨, 전부 train
    for p in sorted(B1_IMG.glob('*.jpg')):
        lines = xml_to_yolo(B1_XML / f'{p.stem}.xml')
        items.append((p, lines, dict(uid=f'b1_{p.stem}', split='train', source='b1', orig_filename=p.name, session='b1_roomA', note='roomA_video',
                                     distance_m='', placement='', label_source='human_labelimg', status='human', n_boxes=len(lines))))
    # b1x — 사람 라벨, 학습에 쓴 적 없는 방 A 프레임 → test 전용. b1 마지막 프레임(554) 이전 것은 b1 학습 프레임과 시간상 이웃
    for p in sorted(B1X_IMG.glob('*.jpg')):
        if not (B1X_XML / f'{p.stem}.xml').exists(): continue   # 라벨 없는 2장 제외
        lines = xml_to_yolo(B1X_XML / f'{p.stem}.xml')
        items.append((p, lines, dict(uid=f'b1x_{p.stem}', split='test', source='b1x_adjacent' if idx_of(p.name) <= 554 else 'b1x_after554', orig_filename=p.name, session='b1x_roomA',
                                     note='roomA_video', distance_m='', placement='', label_source='human_labelimg', status='human', n_boxes=len(lines))))
    # b2 — 의사 라벨 있는 것만 (제외 21장 = 라벨 파일 없음)
    meta = pd.read_csv(B2_DIR / 'labels.csv', dtype={'session': str}); pred = pd.read_csv(B2_LBL / 'all_predictions.csv').set_index('filename')
    for r in meta.itertuples():
        t = B2_LBL / 'labels' / f'{Path(r.filename).stem}.txt'
        if not t.exists(): continue
        lines = [ln.strip() for ln in open(t) if ln.strip()]
        items.append((B2_DIR / 'images' / r.filename, lines,
                      dict(uid=f'b2_{r.session[-6:]}_{idx_of(r.filename):06d}', split='val' if r.note in VAL_B2_NOTES else 'train', source='b2_enemy', orig_filename=r.filename,
                           session=r.session, note=r.note, distance_m=dist_of(r.note), placement=r.note.split('_')[0], label_source='pseudo_r2', status=pred.status[r.filename], n_boxes=len(lines))))
    # 아군 — confident + review + none 재확인(r2 @1280, 25장 전부 타겟 있음)
    fr = pd.read_csv(FR_PSEUDO / 'images.csv'); fmeta = None
    review = set(pd.read_csv(FR_PSEUDO / 'review_list.csv').filename)
    rc = pd.read_csv(FR_PSEUDO / 'none_recheck/predictions.csv'); top1_kept = set(rc[rc.box_id > 0].filename)
    bk = [1.2, 1.5, 1.8, 2.1, 2.4]
    if a.friend_meta:
        fmeta = pd.read_csv(a.friend_meta, dtype={'session': str}).set_index('filename')
    for r in fr.itertuples():
        recheck = r.status == 'none'   # 640 사전 라벨에서 미검출이던 이미지
        lines = [ln.strip() for ln in open(FR_PSEUDO / 'labels' / f'{Path(r.filename).stem}.txt') if ln.strip()]
        if recheck:
            if not lines: continue   # 라벨이 비어 있음 = 타겟 없음 → 제외
            xc, _, _, h = map(float, lines[0].split()[1:5])   # strata 를 박스(높이→거리, 중심→좌/중/우)로 추정
            d = min(bk, key=lambda x: abs(x - 59.1 / (h * H))); third = 'left3' if xc * W < 427 else ('right3' if xc * W > 853 else 'center3')
            r = r._replace(dist_est=d, third=third, strata=f'{d}_{third}', status='none_recheck')
        if fmeta is not None:
            m = fmeta.loc[r.filename]; note = str(m.note)
            row = dict(uid=f'fr_{m.session[-6:]}_{idx_of(r.filename):06d}', split='val' if note in a.friend_val_notes else 'train', session=m.session, note=note,
                       distance_m=dist_of(note), placement=note.replace('painted_', '').split('_')[0])
        else:  # 메타 없음: 박스 높이·위치로 추정한 strata 사용
            row = dict(uid=f'fr_na_{idx_of(r.filename):06d}', split='val' if r.strata in VAL_FR_STRATA else 'train', session='na', note=f'est:{r.strata}',
                       distance_m=r.dist_est, placement=f'est:{r.third}')
        items.append((FR_DIR / r.filename, lines, dict(row, source='friend', orig_filename=r.filename, label_source='pseudo_r2_1280_recheck' if recheck else 'pseudo_r2_seed', status=r.status, n_boxes=len(lines),
                                                         review='pending:2nd_box_removed' if r.filename in top1_kept else ('pending' if r.filename in review else ''))))
    return items

def sanity(out, inv):
    """브리프 A-1 sanity 항목. 문제 목록 반환(비어 있으면 통과)."""
    bad = []
    dup = inv.uid[inv.uid.duplicated()].tolist()
    if dup: bad.append(f'uid 중복 {len(dup)}: {dup[:5]}')
    hs, multi, digests = [], [], {}
    for r in inv.itertuples():
        img = out / r.split / 'images' / f'{r.uid}.jpg'; lbl = out / r.split / 'labels' / f'{r.uid}.txt'
        if img.is_symlink() or not img.is_file(): bad.append(f'{r.uid}: 이미지 없음/symlink'); continue
        with Image.open(img) as im:
            if im.size != (W, H): bad.append(f'{r.uid}: 크기 {im.size}')
        digests.setdefault(hashlib.sha256(img.read_bytes()).hexdigest(), []).append(r.uid)
        rows = [ln.split() for ln in open(lbl) if ln.strip()]
        if not rows: bad.append(f'{r.uid}: 빈 라벨')
        if len(rows) >= 2: multi.append(r.uid)
        for p in rows:
            if len(p) != 5: bad.append(f'{r.uid}: 열 수 {len(p)}'); continue
            if p[0] != '0': bad.append(f'{r.uid}: class {p[0]}')
            xc, yc, w, h = map(float, p[1:])
            if not all(0 <= v <= 1 for v in (xc, yc, w, h)) or w <= 0 or h <= 0: bad.append(f'{r.uid}: 좌표 범위 {p[1:]}')
            if xc - w / 2 < -1e-6 or xc + w / 2 > 1 + 1e-6 or yc - h / 2 < -1e-6 or yc + h / 2 > 1 + 1e-6: bad.append(f'{r.uid}: 박스가 이미지 밖 {p[1:]}')
            hs.append(h * H)
            if not 8 <= h * H <= 90: bad.append(f'{r.uid}: h {h * H:.1f}px (허용 8~90)')
    same = [v for v in digests.values() if len(v) > 1]
    if same: bad.append(f'내용이 같은 이미지 {len(same)}쌍: {same[:3]}')
    if multi: bad.append(f'박스 2개 이상 이미지 {len(multi)}: {multi[:5]}')
    n_files = {s: (len(list((out / s / "images").glob("*.jpg"))), len(list((out / s / "labels").glob("*.txt")))) for s in SPLITS}
    for s, (ni, nl) in n_files.items():
        if ni != nl or ni != (inv.split == s).sum(): bad.append(f'{s}: images {ni} / labels {nl} / inventory {(inv.split == s).sum()} 불일치')
    hs = pd.Series(hs)
    print(f'sanity: 이미지 {len(inv)} (train {n_files["train"][0]} / val {n_files["val"][0]} / test {n_files["test"][0]}), 박스 {len(hs)}, h px min {hs.min():.1f} / med {hs.median():.1f} / max {hs.max():.1f}, 박스≥2 이미지 {len(multi)}, uid 중복 {len(dup)}')
    return bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'handover/dataset'))
    ap.add_argument('--friend-meta', default=None, help='아군 labels.csv (도착 시). 있으면 uid=fr_<session>_<idx>, val 은 --friend-val-notes')
    ap.add_argument('--friend-val-notes', nargs='*', default=['right_1.5m', 'left_2.4m'])
    a = ap.parse_args(); out = Path(a.out)
    if out.exists(): shutil.rmtree(out)
    for s in SPLITS:
        for d in ('images', 'labels'): (out / s / d).mkdir(parents=True)
    items = collect(a)
    for src, lines, row in items:
        shutil.copyfile(src, out / row['split'] / 'images' / f"{row['uid']}.jpg")
        open(out / row['split'] / 'labels' / f"{row['uid']}.txt", 'w').write('\n'.join(lines) + '\n')
    inv = pd.DataFrame([r for _, _, r in items]).reindex(columns=COLS).fillna({'review': ''}); inv.to_csv(out / 'inventory.csv', index=False)

    rel = lambda r: f'./{r.split}/images/{r.uid}.jpg'
    enemy = inv.source.isin(['b1', 'b2_enemy']); tr = inv.split == 'train'; va = inv.split == 'val'; te = inv.split == 'test'
    lists = {'list_train_enemy_only.txt': inv[enemy & tr], 'list_train_friend_only.txt': inv[(inv.source == 'friend') & tr],
             'list_val_enemy.txt': inv[(inv.source == 'b2_enemy') & va], 'list_val_friend.txt': inv[(inv.source == 'friend') & va],
             'list_eval_friend_all.txt': inv[inv.source == 'friend'], 'list_eval_enemy_b2_all.txt': inv[inv.source == 'b2_enemy'], 'list_test_human.txt': inv[te]}
    for name, g in lists.items(): open(out / name, 'w').write('\n'.join(rel(r) for r in g.itertuples()) + '\n')
    yamls = {'data.yaml': ('train/images', 'val/images'), 'data_enemy_only.yaml': ('list_train_enemy_only.txt', 'list_val_enemy.txt'),
             'data_friend_only.yaml': ('list_train_friend_only.txt', 'list_val_friend.txt'),
             'eval_friend_all.yaml': ('list_eval_friend_all.txt', 'list_eval_friend_all.txt'), 'eval_enemy_all.yaml': ('list_eval_enemy_b2_all.txt', 'list_eval_enemy_b2_all.txt'),
             'eval_test_human.yaml': ('list_test_human.txt', 'list_test_human.txt')}
    for name, (t, v) in yamls.items(): open(out / name, 'w').write(f'# path 키 없음: ultralytics 는 이 yaml 이 있는 폴더를 데이터셋 루트로 쓴다 (폴더째 옮겨도 동작)\ntrain: {t}\nval: {v}\nnames:\n  0: tar\n')
    json.dump(dict(rules=dict(b1='전부 train (사람 라벨, 방 A)', b1x='data/images(1) 사람 라벨 98장 — 전부 test (학습·val 미포함). source=b1x_adjacent 는 b1 학습 프레임과 시간상 이웃', b2_val_notes=VAL_B2_NOTES,
                              friend_val=dict(notes=a.friend_val_notes) if a.friend_meta else dict(strata=VAL_FR_STRATA, caveat='세션 메타 없음 — 박스 높이/위치로 추정한 strata. 메타 도착 시 실제 세션으로 교체'),
                              excluded='b2 21장(none 18 + 손으로 놓는 장면 3), b1x 라벨 없는 2장', friend_none_recheck='아군 none 25장은 r2 @1280 재확인 라벨로 train 에 포함 (inventory status=none_recheck, review=pending)'),
                   counts={k: int(len(g)) for k, g in lists.items()} | {'train': int(tr.sum()), 'val': int(va.sum()), 'test': int(te.sum())},
                   train=inv.uid[tr].tolist(), val=inv.uid[va].tolist(), test=inv.uid[te].tolist()), open(out / 'split.json', 'w'), ensure_ascii=False, indent=1)

    print(pd.crosstab(inv.source, inv.split, margins=True).to_string())
    print({k: len(g) for k, g in lists.items()})
    bad = sanity(out, inv)
    if bad:
        print(f'\n❌ sanity 실패 {len(bad)}건:'); print('\n'.join('  ' + b for b in bad[:40])); sys.exit(1)
    print(f'✅ sanity 통과 → {out}')

if __name__ == '__main__': main()
