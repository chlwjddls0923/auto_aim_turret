"""finetune_prelabel 결과에서 박스 3개 장면의 여분('neither') 박스를 자동 삭제하고, 눈 확인용 썸네일 시트를 만든다.
  python tools/finetune_autoclean.py --labels finetune_labels [--dry-run]
삭제 조건: n_boxes == 3 이고 초록 정확히 1·마젠타 정확히 1 → neither 박스 삭제. 아니면 손대지 않고 leftover 로 남김.
원본 txt/xml 은 <세션>/orig/ 에 보존. 출력: deleted_boxes.csv, autoclean.csv, check_sheets/*.jpg
시트 표시: 초록=초록 실선, 마젠타=분홍 실선, neither(유지)=노랑 실선, 삭제=빨강 점선.
"""
import argparse, os, shutil
import numpy as np, pandas as pd, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="finetune_labels")
ap.add_argument("--src", default="finetune_data(10.08)")
ap.add_argument("--per-sheet", type=int, default=25)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
L = a.labels
im = pd.read_csv(f"{L}/images.csv"); lb = pd.read_csv(f"{L}/boxes_lab.csv")

def xml_doc(fname, W, H, boxes):
    objs = "".join(f"<object><name>tar</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>"
                   f"<bndbox><xmin>{round(x1)}</xmin><ymin>{round(y1)}</ymin><xmax>{round(x2)}</xmax><ymax>{round(y2)}</ymax></bndbox></object>"
                   for x1, y1, x2, y2 in boxes)
    return (f"<annotation><folder>images</folder><filename>{fname}</filename><size><width>{W}</width><height>{H}</height>"
            f"<depth>3</depth></size><segmented>0</segmented>{objs}</annotation>")

rows, deleted = [], []
for _, r in im[im.status == "review"].iterrows():
    b = lb[(lb.session == r.session) & (lb.filename == r.filename)]
    ng, nm = (b.color == "green").sum(), (b.color == "magenta").sum()
    if r.n_boxes == 3 and ng == 1 and nm == 1:
        act = "auto_deleted"
        drop = b[b.color == "neither"]; keep = b[b.color != "neither"]
        deleted.append(drop)
        if not a.dry_run:
            d = f"{L}/{r.session}"; stem = os.path.splitext(r.filename)[0]
            os.makedirs(f"{d}/orig", exist_ok=True)
            for ext, sub in (("txt", "labels"), ("xml", "xml")):
                if not os.path.exists(f"{d}/orig/{stem}.{ext}"): shutil.copy(f"{d}/{sub}/{stem}.{ext}", f"{d}/orig/{stem}.{ext}")
            W, H = 1280, 960
            with open(f"{d}/labels/{stem}.txt", "w") as f:
                for _, k in keep.iterrows():
                    f.write(f"0 {(k.x1 + k.x2) / 2 / W:.6f} {(k.y1 + k.y2) / 2 / H:.6f} {(k.x2 - k.x1) / W:.6f} {(k.y2 - k.y1) / H:.6f}\n")
            open(f"{d}/xml/{stem}.xml", "w").write(xml_doc(r.filename, W, H, list(zip(keep.x1, keep.y1, keep.x2, keep.y2))))
    elif r.reason == "low_conf": act = "low_conf"
    elif r.reason.startswith("color:") and r.n_boxes == 2 and ng == 0 and nm == 1: act = "faded_green_kept"
    else: act = "leftover"
    rows.append(dict(session=r.session, filename=r.filename, split=r.split, action=act, n_green=ng, n_magenta=nm, n_boxes=r.n_boxes))
ac = pd.DataFrame(rows); dl = pd.concat(deleted) if deleted else pd.DataFrame(columns=lb.columns)
print(ac.action.value_counts().to_dict(), "| 삭제 박스", len(dl), "| 삭제 박스 유형", dl.get("type", pd.Series()).value_counts().to_dict())
if a.dry_run: raise SystemExit
ac.to_csv(f"{L}/autoclean.csv", index=False); dl.to_csv(f"{L}/deleted_boxes.csv", index=False)

# ---------- 시트 ----------
COL = {"green": (0, 190, 0), "magenta": (230, 80, 230), "neither": (0, 220, 255)}
def dashed(img, p1, p2, color, t=2, dash=6):
    (x1, y1), (x2, y2) = p1, p2
    for xa, ya, xb, yb in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(1, int(np.hypot(xb - xa, yb - ya) // dash))
        for i in range(0, n, 2):
            s, e = i / n, min(1, (i + 1) / n)
            cv2.line(img, (int(xa + (xb - xa) * s), int(ya + (yb - ya) * s)), (int(xa + (xb - xa) * e), int(ya + (yb - ya) * e)), color, t)

TW = 480
def thumb(sess, fn, tag):
    img = cv2.imread(f"{a.src}/{sess}/images/{fn}")
    cur = lb[(lb.session == sess) & (lb.filename == fn)]
    gone = dl[(dl.session == sess) & (dl.filename == fn)] if len(dl) else dl
    gk = set(zip(gone.x1.round(2), gone.y1.round(2))) if len(gone) else set()
    boxes = [(r, (round(r.x1, 2), round(r.y1, 2)) in gk) for _, r in cur.iterrows()]
    big = img.copy()
    for r, isdel in boxes:
        p1, p2 = (int(r.x1) - 2, int(r.y1) - 2), (int(r.x2) + 2, int(r.y2) + 2)
        if isdel: dashed(big, p1, p2, (0, 0, 255), 2, 5)
        else: cv2.rectangle(big, p1, p2, COL[r.color], 2)
    # 위: 박스 주변 가로 띠 개요 / 아래: 박스별 확대
    ys = [r.y1 for r, _ in boxes] + [r.y2 for r, _ in boxes]
    cy = int(np.mean(ys)) if ys else 480; y0 = max(0, min(960 - 200, cy - 100))
    ov = cv2.resize(big[y0:y0 + 200], (TW, 75))
    zs = []
    for r, isdel in sorted(boxes, key=lambda t: t[0].x1)[:3]:
        cx, cyy = int((r.x1 + r.x2) / 2), int((r.y1 + r.y2) / 2); hf = int(max(45, 1.6 * r.h))
        x0, yy0 = max(0, cx - hf), max(0, cyy - hf)
        z = cv2.resize(big[yy0:cyy + hf, x0:cx + hf], (TW // 3, TW // 3), interpolation=cv2.INTER_NEAREST)
        lbl = ("DEL " if isdel else "") + f"{r.color[:3]} {r.conf:.2f}"
        cv2.putText(z, lbl, (3, 15), 0, 0.45, (0, 0, 255) if isdel else (0, 0, 0), 1, cv2.LINE_AA); zs.append(z)
    zs += [np.full((TW // 3, TW // 3, 3), 255, np.uint8)] * (3 - len(zs))
    head = np.full((22, TW, 3), 255, np.uint8)
    cv2.putText(head, f"{'1.5m' if sess.startswith('both(1.5') else '2.0m+'}/{fn}  [{tag}]", (4, 16), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    t = np.vstack([head, ov, np.hstack(zs)])
    return cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(120, 120, 120))

groups = [("A_auto_deleted", ac[ac.action == "auto_deleted"][["session", "filename"]].assign(tag="auto-del")),
          ("B_auto_pass_sample", im[im.status == "audit"][["session", "filename"]].assign(tag="auto-pass")),
          ("C_left_lowconf_faded", ac[ac.action != "auto_deleted"][["session", "filename", "action"]].rename(columns={"action": "tag"}))]
os.makedirs(f"{L}/check_sheets", exist_ok=True)
for name, g in groups:
    g = g.reset_index(drop=True); n = len(g)
    if n == 0: continue
    k = int(np.ceil(n / a.per_sheet)); parts = np.array_split(np.arange(n), k)
    for j, idx in enumerate(parts):
        ts = [thumb(g.session[i], g.filename[i], g.tag[i]) for i in idx]
        ts += [np.full_like(ts[0], 255)] * (-len(ts) % 5)
        sheet = np.vstack([np.hstack(ts[i:i + 5]) for i in range(0, len(ts), 5)])
        p = f"{L}/check_sheets/{name}_{j + 1}of{k}.jpg"; cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(p, len(idx), "장")
