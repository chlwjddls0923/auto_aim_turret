"""finetune_data 사전 라벨(r3_coco 검출, det_census 의 boxes.csv 재사용) + 자동 통과 판정 + 검수 목록. test 는 다루지 않는다.
  python tools/finetune_prelabel.py --census runs/finetune_census/r3_coco --out finetune_labels [--dry-run]
색 규칙(2026-10-11 결정): 초록 a* < -15, 마젠타 a* > +30, 그 외 neither. colorlib.patch_stats(top_frac=0.5).
자동 통과: 저장 박스(conf>=0.10, 최대 3개)가 정확히 2개, 초록 1·마젠타 1, 둘 다 conf>=0.5.
검수: 그 외 전부 + 자동 통과 중 무작위 5%(seed 0).
출력(out/<세션>/): labels/*.txt, xml/*.xml (LabelImg), 그리고 out/images.csv, out/review_list.csv, out/boxes_lab.csv
"""
import argparse, os
import numpy as np, pandas as pd, cv2
from colorlib import patch_stats

G_MAX, M_MIN = -15.0, 30.0
ap = argparse.ArgumentParser()
ap.add_argument("--census", required=True)
ap.add_argument("--src", default="finetune_data(10.08)")
ap.add_argument("--out", default="finetune_labels")
ap.add_argument("--conf-keep", type=float, default=0.10)
ap.add_argument("--conf-pass", type=float, default=0.5)
ap.add_argument("--max-boxes", type=int, default=3)
ap.add_argument("--audit-frac", type=float, default=0.05)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

def col(astar):
    return "green" if astar < G_MAX else ("magenta" if astar > M_MIN else "neither")

def xml_doc(fname, W, H, boxes):
    objs = "".join(f"<object><name>tar</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>"
                   f"<bndbox><xmin>{round(x1)}</xmin><ymin>{round(y1)}</ymin><xmax>{round(x2)}</xmax><ymax>{round(y2)}</ymax></bndbox></object>"
                   for x1, y1, x2, y2 in boxes)
    return (f"<annotation><folder>images</folder><filename>{fname}</filename><size><width>{W}</width><height>{H}</height>"
            f"<depth>3</depth></size><segmented>0</segmented>{objs}</annotation>")

sp = pd.read_csv(f"{a.src}/split.csv"); sp = sp[sp.split.isin(["train", "val"])]
bx = pd.read_csv(f"{a.census}/boxes.csv")
bx = bx[bx.split.isin(["train", "val"])]
assert set(bx.filename.astype(str) + bx.session).issubset(set(sp.filename + sp.session_dir))

ims, lab_rows = [], []
for (sess, fn), g in sp.groupby(["session_dir", "filename"]):
    b = bx[(bx.session == sess) & (bx.filename == fn) & (bx.conf >= a.conf_keep)].sort_values("conf", ascending=False).head(a.max_boxes)
    frame = cv2.imread(f"{a.src}/{sess}/images/{fn}"); H, W = frame.shape[:2]
    cols = []
    for _, r in b.iterrows():
        st = patch_stats(frame, (r.x1, r.y1, r.x2, r.y2), top_frac=0.5)
        L, A, B = st["lab"] if st["lab"] is not None else (np.nan, np.nan, np.nan)
        c = col(A) if st["n_px"] >= 20 else "neither"; cols.append(c)
        lab_rows.append(dict(session=sess, filename=fn, conf=r.conf, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, h=r.y2 - r.y1,
                             n_px=st["n_px"], L=L, a_star=A, b_star=B, color=c))
    ok = len(b) == 2 and sorted(cols) == ["green", "magenta"] and b.conf.min() >= a.conf_pass
    why = "auto" if ok else ("n=%d" % len(b) if len(b) != 2 else ("color:" + "+".join(sorted(cols)) if sorted(cols) != ["green", "magenta"] else "low_conf"))
    ims.append(dict(session=sess, filename=fn, split=sp[(sp.session_dir == sess) & (sp.filename == fn)].split.iloc[0],
                    n_boxes=len(b), colors="+".join(sorted(cols)), min_conf=b.conf.min() if len(b) else np.nan,
                    status="auto" if ok else "review", reason=why))
    if not a.dry_run:
        d = f"{a.out}/{sess}"; os.makedirs(f"{d}/labels", exist_ok=True); os.makedirs(f"{d}/xml", exist_ok=True)
        stem = os.path.splitext(fn)[0]
        with open(f"{d}/labels/{stem}.txt", "w") as f:
            for _, r in b.iterrows():
                f.write(f"0 {(r.x1 + r.x2) / 2 / W:.6f} {(r.y1 + r.y2) / 2 / H:.6f} {(r.x2 - r.x1) / W:.6f} {(r.y2 - r.y1) / H:.6f}\n")
        open(f"{d}/xml/{stem}.xml", "w").write(xml_doc(fn, W, H, list(zip(b.x1, b.y1, b.x2, b.y2))))

im = pd.DataFrame(ims); lb = pd.DataFrame(lab_rows)
auto = im[im.status == "auto"]
audit = auto.sample(n=int(np.ceil(a.audit_frac * len(auto))), random_state=0)
im.loc[audit.index, "status"] = "audit"
review = im[im.status != "auto"]
print(im.groupby(["session", "split"]).status.value_counts().unstack(fill_value=0))
print("검수 사유:", im[im.status == "review"].reason.value_counts().to_dict())
print(f"검수 합계 {len(review)}장 (review {int((im.status=='review').sum())} + audit {len(audit)}) / 전체 {len(im)}")
if not a.dry_run:
    os.makedirs(a.out, exist_ok=True)
    im.to_csv(f"{a.out}/images.csv", index=False); lb.to_csv(f"{a.out}/boxes_lab.csv", index=False)
    review.to_csv(f"{a.out}/review_list.csv", index=False)
    print("저장:", a.out)
