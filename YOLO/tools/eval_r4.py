"""r2 / r3_coco / r4 비교 평가 (모두 NCNN 960x1280, 같은 경로).
  python tools/eval_r4.py --set newval  --models r2=<dir> r3_coco=<dir> r4=<dir>
  python tools/eval_r4.py --set oldtest --models ...
  python tools/eval_r4.py --set newtest --models ... --conf-op 0.xx --final      # 1회만. 잠금 파일이 있으면 거부
새 val/test (라벨 없음, 모든 장면 = 초록 1 + 마젠타 1 로 간주):
  색: colorlib.patch_stats(top_frac=0.5) 고채도 픽셀 a*: 초록 < -15, 마젠타 > +30, 그 외(또는 n_px<20) neither
  프레임별 색당 최고 conf 박스 1개 = 타겟, 나머지 = 여분. 여분 유형: x2>1270 붉은 물체(오른쪽 끝) / h<=30 도어 스토퍼 추정 / 기타
  높이 구간: finetune_labels train+val 라벨 높이의 3분위. 놓친 색은 같은 프레임 반대색 박스 높이로 대신.
기존 test 98장(사람 라벨): IoU>=0.5 매칭 (eval_cross_color.py 와 같은 정의).
"""
import argparse, glob, os
import numpy as np, pandas as pd, cv2
from ultralytics import YOLO
from colorlib import patch_stats

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["newval", "oldtest", "newtest"])
ap.add_argument("--models", nargs="+", required=True, help="tag=ncnn_model_dir")
ap.add_argument("--confs", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
ap.add_argument("--conf-op", type=float, default=None, help="newtest: 4번에서 정한 운용 conf (이 값만 보고)")
ap.add_argument("--final", action="store_true")
ap.add_argument("--out", default="runs/r4_eval")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LOCK = f"{a.out}/NEWTEST_DONE.lock"
if a.set == "newtest":
    assert a.final and a.conf_op is not None, "newtest 는 --final 과 --conf-op 가 필요"
    assert not os.path.exists(LOCK), f"새 test 는 이미 평가됨 ({LOCK}) — 다시 돌리지 않는다"
    a.confs = [a.conf_op]

# ---------- 프레임 목록 ----------
SRC = "finetune_data(10.08)"
if a.set == "newval":
    sp = pd.read_csv(f"{SRC}/split.csv"); sp = sp[sp.split == "val"]
    frames = [(f"{SRC}/{r.session_dir}/images/{r.filename}", r.session_dir) for _, r in sp.iterrows()]
elif a.set == "newtest":
    frames = [(p, p.split("/")[1]) for p in sorted(glob.glob(f"{SRC}_test/*/images/*.jpg"))]
else:
    frames = [(p, "oldtest") for p in sorted(glob.glob("handover/dataset_r4/test/images/*.jpg"))]

# 높이 구간 경계 (train+val 검수 라벨)
hs = [float(l.split()[4]) * 960 for f in glob.glob("finetune_labels/*/labels/*.txt") for l in open(f) if l.strip()]
E = np.quantile(hs, [1 / 3, 2 / 3]).round(1)
binof = lambda h: "small" if h < E[0] else ("mid" if h < E[1] else "large")

def colr(st):
    if st["n_px"] < 20 or st["lab"] is None: return "neither"
    A = st["lab"][1]
    return "green" if A < -15 else ("magenta" if A > 30 else "neither")

# ---------- 예측 (conf 0.05 로 1회, 캐시) ----------
def predict(tag, mdir):
    cp = f"{a.out}/pred_{a.set}_{tag}.csv"
    if os.path.exists(cp): return pd.read_csv(cp)
    m = YOLO(mdir, task="detect"); rows = []
    for k in range(0, len(frames), 16):
        ch = frames[k:k + 16]
        for (p, s), r in zip(ch, m.predict([p for p, _ in ch], imgsz=[960, 1280], conf=0.05, verbose=False)):
            fid = s + "/" + os.path.basename(p)
            rows.append(dict(fid=fid, conf=np.nan))
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                st = patch_stats(r.orig_img, (x1, y1, x2, y2), top_frac=0.5)
                rows.append(dict(fid=fid, conf=float(c), x1=x1, y1=y1, x2=x2, y2=y2, h=y2 - y1, color=colr(st), n_px=st["n_px"],
                                 a_star=np.nan if st["lab"] is None else st["lab"][1], b_star=np.nan if st["lab"] is None else st["lab"][2]))
    d = pd.DataFrame(rows); d.to_csv(cp, index=False); return d

# ---------- 라벨 없는 지표 ----------
def label_free(d, cf, tag, fail_dir=None):
    per, ex_rows = [], []
    for fid in sorted(set(d.fid)):
        g = d[(d.fid == fid) & (d.conf >= cf)].sort_values("conf", ascending=False)
        tg = {c: g[g.color == c].head(1) for c in ("green", "magenta")}
        keep = pd.concat([t for t in tg.values() if len(t)]) if any(len(t) for t in tg.values()) else g.head(0)
        ex = g.drop(keep.index)
        hg = tg["green"].h.iloc[0] if len(tg["green"]) else np.nan
        hm = tg["magenta"].h.iloc[0] if len(tg["magenta"]) else np.nan
        hg, hm = (hg if hg == hg else hm), (hm if hm == hm else hg)
        fh = np.nanmean([hg, hm]) if (hg == hg or hm == hm) else np.nan
        for _, e in ex.iterrows():
            ty = "붉은 물체(오른쪽 끝)" if e.x2 > 1270 else ("도어 스토퍼 추정(h<=30)" if e.h <= 30 else "기타")
            ex_rows.append(dict(e.to_dict(), type=ty))
        per.append(dict(fid=fid, n=len(g), g_miss=len(tg["green"]) == 0, m_miss=len(tg["magenta"]) == 0, n_extra=len(ex),
                        ex_mag=int((ex.color == "magenta").sum()), dup=((g.color == "green").sum() >= 2) or ((g.color == "magenta").sum() >= 2),
                        exact=len(g) == 2 and len(tg["green"]) == 1 and len(tg["magenta"]) == 1,
                        bin_g=binof(hg) if hg == hg else "none", bin_m=binof(hm) if hm == hm else "none", bin_f=binof(fh) if fh == fh else "none"))
        if fail_dir and not per[-1]["exact"]:
            p = f"{SRC}_test/{fid.split('/')[0]}/images/{fid.split('/')[1]}"
            im = cv2.imread(p)
            for _, b in g.iterrows():
                col = (0, 190, 0) if b.color == "green" else ((230, 80, 230) if b.color == "magenta" else (0, 0, 255))
                cv2.rectangle(im, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), col, 2)
                cv2.putText(im, f"{b.color[:3]} {b.conf:.2f}", (int(b.x1), int(b.y1) - 4), 0, 0.5, col, 1, cv2.LINE_AA)
            cv2.putText(im, f"{tag} conf>={cf} {fid}", (10, 30), 0, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            os.makedirs(f"{fail_dir}/{tag}", exist_ok=True)
            cv2.imwrite(f"{fail_dir}/{tag}/{fid.replace('/', '__')}", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    P = pd.DataFrame(per); X = pd.DataFrame(ex_rows)
    tot = dict(model=tag, conf=cf, frames=len(P), 정답프레임=P.exact.mean(), 초록놓침=P.g_miss.mean(), 마젠타놓침=P.m_miss.mean(),
               여분_프레임당=P.n_extra.mean(), 여분합계=int(P.n_extra.sum()), 여분중_마젠타판정=int(P.ex_mag.sum()), 중복프레임=P.dup.mean())
    for t in ("붉은 물체(오른쪽 끝)", "도어 스토퍼 추정(h<=30)", "기타"):
        tot["여분:" + t] = int((X.type == t).sum()) if len(X) else 0
    bins = []
    for b in ("small", "mid", "large"):
        bins.append(dict(model=tag, conf=cf, 구간=b, 초록프레임=int((P.bin_g == b).sum()), 초록놓침=P[P.bin_g == b].g_miss.mean(),
                         마젠타프레임=int((P.bin_m == b).sum()), 마젠타놓침=P[P.bin_m == b].m_miss.mean(),
                         프레임=int((P.bin_f == b).sum()), 여분_프레임당=P[P.bin_f == b].n_extra.mean()))
    return tot, bins, X

# ---------- 기존 test (사람 라벨) ----------
def gt(p):
    t = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    return [(lambda xc, yc, w, h: ((xc - w / 2) * 1280, (yc - h / 2) * 960, (xc + w / 2) * 1280, (yc + h / 2) * 960))(*map(float, l.split()[1:5]))
            for l in open(t) if l.strip()]
def iou(p, q):
    iw = min(p[2], q[2]) - max(p[0], q[0]); ih = min(p[3], q[3]) - max(p[1], q[1])
    if iw <= 0 or ih <= 0: return 0.0
    i = iw * ih; return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i)
def old_metrics(d, cf, tag):
    tp, fp, fn, ce, he = 0, 0, 0, [], []
    for p, s in frames:
        fid = s + "/" + os.path.basename(p); G = gt(p); used = set()
        g = d[(d.fid == fid) & (d.conf >= cf)].sort_values("conf", ascending=False)
        for _, b in g.iterrows():
            bb = (b.x1, b.y1, b.x2, b.y2); cand = [(iou(bb, q), j) for j, q in enumerate(G) if j not in used]
            best = max(cand) if cand else (0, -1)
            if best[0] >= 0.5:
                q = G[best[1]]; used.add(best[1]); tp += 1
                ce.append(np.hypot((bb[0] + bb[2] - q[0] - q[2]) / 2, (bb[1] + bb[3] - q[1] - q[3]) / 2)); he.append((bb[3] - bb[1]) - (q[3] - q[1]))
            else: fp += 1
        fn += len(G) - len(used)
    return dict(model=tag, conf=cf, recall=tp / (tp + fn), FP_프레임당=fp / len(frames), FP합계=fp, 중심오차_med=np.median(ce), 중심오차_p95=np.percentile(ce, 95),
                높이오차_평균=np.mean(he), 높이오차_abs_med=np.median(np.abs(he)))

# ---------- 실행 ----------
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
T, B, XS = [], [], []
for spec in a.models:
    tag, mdir = spec.split("=", 1)
    d = predict(tag, mdir)
    for cf in a.confs:
        if a.set == "oldtest": T.append(old_metrics(d, cf, tag))
        else:
            t, b, x = label_free(d, cf, tag, fail_dir=f"{a.out}/eval_test_failures" if a.set == "newtest" else None)
            T.append(t); B += b
            if len(x): XS.append(x.assign(model=tag, conf_th=cf))
T = pd.DataFrame(T); T.to_csv(f"{a.out}/{a.set}_summary.csv", index=False)
print(f"[{a.set}] 프레임 {len(frames)} | 높이 구간 경계 small < {E[0]} px <= mid < {E[1]} px <= large")
print(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
if B:
    B = pd.DataFrame(B); B.to_csv(f"{a.out}/{a.set}_by_height.csv", index=False)
    print(B.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
if XS:
    X = pd.concat(XS); X.to_csv(f"{a.out}/{a.set}_extras.csv", index=False)
if a.set == "newtest":
    open(LOCK, "w").write(f"conf_op={a.conf_op} models={' '.join(a.models)}\n")
