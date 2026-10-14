"""새 촬영분(both) 검출 집계: 장면마다 박스 수와 색(초록/마젠타)을 센다. test 는 split.csv 로 제외.
  python tools/det_census.py --model runs/r2/weights/best.pt --tag r2 [--splits train val] [--conf 0.10]
출력: runs/finetune_census/<tag>/boxes.csv (박스별), summary.csv (세션×conf 별 표)
색: colorlib.patch_stats(top_frac=0.5) 고채도 픽셀 a* 부호. n_px < 20 → unknown.
"""
import argparse, os
import numpy as np, pandas as pd, cv2
from ultralytics import YOLO
from colorlib import patch_stats

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--src", default="finetune_data(10.08)")
ap.add_argument("--splits", nargs="+", default=["train", "val"])
ap.add_argument("--conf", type=float, default=0.10)
ap.add_argument("--report-confs", type=float, nargs="+", default=[0.10, 0.25])
ap.add_argument("--imgsz", type=int, nargs="+", default=[1280])
a = ap.parse_args()

sp = pd.read_csv(f"{a.src}/split.csv")
assert "test" not in a.splits
sp = sp[sp.split.isin(a.splits)]
out = f"runs/finetune_census/{a.tag}"; os.makedirs(out, exist_ok=True)
m = YOLO(a.model)

def color_of(st):
    if st["n_px"] < 20: return "unknown"
    return "green" if st["lab"][1] < 0 else "magenta"

rows = []
for sess, g in sp.groupby("session_dir"):
    fs = [f"{a.src}/{sess}/images/{f}" for f in g.filename]
    splits = dict(zip(g.filename, g.split))
    for i in range(0, len(fs), 32):
        for f, r in zip(fs[i:i + 32], m.predict(fs[i:i + 32], imgsz=a.imgsz if len(a.imgsz) > 1 else a.imgsz[0], conf=a.conf, verbose=False)):
            fn = os.path.basename(f)
            if len(r.boxes) == 0:
                rows.append(dict(session=sess, filename=fn, split=splits[fn], conf=np.nan)); continue
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                st = patch_stats(r.orig_img, (x1, y1, x2, y2), top_frac=0.5)
                rows.append(dict(session=sess, filename=fn, split=splits[fn], conf=float(c), x1=x1, y1=y1, x2=x2, y2=y2,
                                 h=y2 - y1, w=x2 - x1, color=color_of(st), n_px=st["n_px"],
                                 a_star=None if st["lab"] is None else float(st["lab"][1])))
bx = pd.DataFrame(rows); bx.to_csv(f"{out}/boxes.csv", index=False)

summ = []
for cf in a.report_confs:
    for sess, g in bx.groupby("session"):
        per = []
        for fn, h in g.groupby("filename"):
            h = h[h.conf >= cf]; cols = h.color.tolist()
            ng, nm = cols.count("green"), cols.count("magenta")
            per.append(dict(n=len(h), both=ng >= 1 and nm >= 1, only_m=nm >= 1 and ng == 0, only_g=ng >= 1 and nm == 0,
                            none=len(h) == 0, many=len(h) >= 3, exact=len(h) == 2 and ng == 1 and nm == 1))
        p = pd.DataFrame(per)
        summ.append(dict(model=a.tag, conf=cf, session=sess[:16], frames=len(p), 둘다=p.both.mean(), 마젠타만=p.only_m.mean(),
                         초록만=p.only_g.mean(), 없음=p.none.mean(), 셋이상=p.many.mean(), 정확히_초1마1=p.exact.mean()))
s = pd.DataFrame(summ); s.to_csv(f"{out}/summary.csv", index=False)
pd.set_option("display.width", 200)
print(s.to_string(index=False, float_format=lambda v: f"{v:.1%}" if v <= 1 else f"{v:g}"))
