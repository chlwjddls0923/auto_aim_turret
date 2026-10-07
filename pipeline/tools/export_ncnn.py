#!/usr/bin/env python
"""export_ncnn.py — .pt 검출기를 파이4용 NCNN FP32 4종으로 export (weights/ncnn/ 와 같은 구성).
  960x1280_ncnn_model (전체 프레임) / 736x960_ncnn_model / 480x640_ncnn_model / 384x384_ncnn_model (ROI 추적)
NCNN 은 입력 크기가 export 시점에 고정된다. pnnx 패키지가 필요하다(x86 은 pip 로 설치됨. aarch64 는 휠 유무 확인 — 없으면 x86 에서 export.
결과물 param/bin 은 플랫폼 독립이라 그대로 파이로 복사하면 된다).

사용: python tools/export_ncnn.py --model runs/r3/weights/best.pt --out weights/ncnn_r3 --name r3 [--check-image <jpg>]
"""
import argparse, shutil, tempfile
from pathlib import Path
SIZES = {'960x1280_ncnn_model': (960, 1280), '736x960_ncnn_model': (736, 960), '480x640_ncnn_model': (480, 640), '384x384_ncnn_model': (384, 384)}   # 폴더 이름에 _ncnn_model 이 있어야 ultralytics 가 NCNN 으로 인식

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True); ap.add_argument('--name', default='model')
    ap.add_argument('--check-image', default=None, help='주면 .pt 와 NCNN 960x1280 의 검출 박스를 나란히 출력(패리티 확인)')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    shutil.copyfile(a.model, out / f'yolo11n_{a.name}.pt')
    for name, sz in SIZES.items():
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / 'model.pt'; shutil.copyfile(a.model, w)
            res = YOLO(str(w)).export(format='ncnn', imgsz=list(sz), device='cpu')
            if (out / name).exists(): shutil.rmtree(out / name)
            shutil.move(res, out / name)
        print('exported', out / name)
    if a.check_image:
        fmt = lambda r: [[round(float(v), 1) for v in b] + [round(float(c), 3)] for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        print('pt  :', fmt(YOLO(a.model).predict(a.check_image, imgsz=(960, 1280), conf=0.25, device='cpu', verbose=False)[0]))
        print('ncnn:', fmt(YOLO(str(out / '960x1280_ncnn_model'), task='detect').predict(a.check_image, imgsz=(960, 1280), conf=0.25, verbose=False)[0]))

if __name__ == '__main__': main()
