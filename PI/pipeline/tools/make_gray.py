#!/usr/bin/env python
"""make_gray.py — 흑백 arm 용 데이터셋 미러 (인계 브리프 §5-6, §6 A-5).
src 데이터셋(package_dataset.py 출력)을 dst 로 복제하되 이미지는 흑백으로 바꿔 3채널에 같은 값을 넣는다(모델 입력 형식 유지).
라벨·리스트·inventory·split·yaml 은 그대로 복사한다(yaml 은 path 키가 없어 자기 폴더를 루트로 쓴다. path 줄이 있으면 dst 로 바꿔 씀).
흑백 이미지에서는 hsv_h/hsv_s 증강이 아무 효과가 없으므로(채도 0) 이 arm 은 색 정보를 전혀 못 쓴다. 평가도 dst 의 리스트로 한다.

사용: python tools/make_gray.py --src handover/dataset --dst handover/dataset_gray [--workers 8]
"""
import argparse, re, shutil, cv2
from multiprocessing import Pool
from pathlib import Path

def convert(job):
    src, dst = job
    im = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if im is None: return str(src)
    cv2.imwrite(str(dst), cv2.merge([im, im, im]), [cv2.IMWRITE_JPEG_QUALITY, 95]); return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--dst', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); src, dst = Path(a.src).resolve(), Path(a.dst).resolve(); jobs = []
    for s in ('train', 'val', 'test'):
        if not (src / s / 'images').is_dir(): continue
        (dst / s / 'images').mkdir(parents=True, exist_ok=True)
        if (dst / s / 'labels').exists(): shutil.rmtree(dst / s / 'labels')
        shutil.copytree(src / s / 'labels', dst / s / 'labels')
        jobs += [(p, dst / s / 'images' / p.name) for p in sorted((src / s / 'images').glob('*.jpg'))]
    for p in list(src.glob('*.txt')) + list(src.glob('*.csv')) + list(src.glob('*.json')): shutil.copyfile(p, dst / p.name)
    for p in src.glob('*.yaml'): (dst / p.name).write_text(re.sub(r'(?m)^path:.*$', f'path: {dst}', p.read_text()))
    for c in dst.rglob('*.cache'): c.unlink()   # 라벨 캐시는 이미지 해시가 달라 재생성돼야 함
    with Pool(a.workers) as pool: failed = [f for f in pool.imap_unordered(convert, jobs, chunksize=16) if f]
    if failed: raise SystemExit(f'읽기 실패 {len(failed)}장: {failed[:5]}')
    print(f'흑백 미러 완료: {len(jobs)}장 → {dst}')

if __name__ == '__main__': main()
