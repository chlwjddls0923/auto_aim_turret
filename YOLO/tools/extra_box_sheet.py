"""det_census 결과에서 여분 박스(장면당 색별 최고 conf 1개를 타겟으로 보고 나머지)를 뽑아 위치 관계를 분류하고 크롭 대조 시트를 만든다.
  python tools/extra_box_sheet.py --tag r3_coco [--conf 0.10]
출력: runs/finetune_census/<tag>/extras.csv, sheet_XX.jpg (빨강=여분, 초록/자홍=타겟), h_bins.json
"""
import argparse, json, os
import numpy as np, pandas as pd, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--src", default="finetune_data(10.08)")
ap.add_argument("--conf", type=float, default=0.10)
a = ap.parse_args()
out = f"runs/finetune_census/{a.tag}"
bx = pd.read_csv(f"{out}/boxes.csv")
bx = bx[bx.conf >= a.conf].copy()

def overlap(b, t):  # b 와 t 의 교집합 / b 면적
    ix = max(0, min(b.x2, t.x2) - max(b.x1, t.x1)); iy = max(0, min(b.y2, t.y2) - max(b.y1, t.y1))
    return ix * iy / max(1e-6, (b.x2 - b.x1) * (b.y2 - b.y1))

tg_rows, ex_rows = [], []
for (s, f), g in bx.groupby(["session", "filename"]):
    g = g.sort_values("conf", ascending=False)
    tg = [g[g.color == c].head(1) for c in ("green", "magenta")]
    tg = pd.concat([t for t in tg if len(t)])
    tg_rows.append(tg)
    for _, b in g.drop(tg.index).iterrows():
        rel, near_h, near_d = "far", np.nan, np.nan
        if len(tg):
            cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
            d = [np.hypot(cx - (t.x1 + t.x2) / 2, cy - (t.y1 + t.y2) / 2) / t.h for _, t in tg.iterrows()]
            k = int(np.argmin(d)); t = tg.iloc[k]; near_h, near_d = t.h, d[k]
            ov = [overlap(b, tt) for _, tt in tg.iterrows()]
            cov = [overlap(tt, b) for _, tt in tg.iterrows()]   # 타겟이 여분 박스 안에 들어간 비율
            if len(tg) == 2 and min(cov) > 0.5: rel = "two_targets_merged"
            elif max(cov) > 0.7 and b.h > 1.3 * t.h: rel = "enclosing_target"
            elif max(ov) > 0.5: rel = "part_of_target"          # 타겟을 쪼개 잡음 / 중복
            elif near_d < 3: rel = "beside_target(<3h)"
            else: rel = "far"
        ex_rows.append(dict(b.to_dict(), relation=rel, near_target_h=near_h, near_dist_h=near_d))
ex = pd.DataFrame(ex_rows); tg = pd.concat(tg_rows)

# 높이 구간: 타겟 박스 높이(train+val) 3분위
q = np.quantile(tg.h, [1 / 3, 2 / 3]).round(1).tolist()
json.dump(dict(edges_px=q, n_targets=len(tg), source=f"{a.tag} conf>={a.conf} 타겟 박스(색별 최고 conf)"),
          open(f"{out}/h_bins.json", "w"), ensure_ascii=False, indent=1)
binof = lambda h: "small" if h < q[0] else ("mid" if h < q[1] else "large")
if len(ex):
    ex["near_bin"] = ex.near_target_h.map(lambda h: binof(h) if h == h else "none")
ex.to_csv(f"{out}/extras.csv", index=False)

print(f"타겟 박스 높이: n={len(tg)}, 경계 {q[0]} / {q[1]} px, 범위 {tg.h.min():.1f}~{tg.h.max():.1f}, 중앙 {tg.h.median():.1f}")
print(f"여분 박스 {len(ex)}개 (장면 {ex.filename.nunique() if len(ex) else 0})")
if len(ex):
    print(ex.groupby("relation").agg(n=("conf", "size"), conf_med=("conf", "median"), conf_max=("conf", "max"), h_med=("h", "median")).round(3))
    print("conf 구간:", pd.cut(ex.conf, [0.1, 0.15, 0.2, 0.25, 0.5, 1.0], right=False).value_counts().sort_index().to_dict())
    print("가장 가까운 타겟의 높이 구간별 여분 수:", ex.near_bin.value_counts().to_dict(),
          "| 타겟 전체 구간 비율:", tg.h.map(binof).value_counts(normalize=True).round(2).to_dict())
    print("색:", ex.color.value_counts().to_dict())

    # 크롭 시트: 여분 박스 중심, 한 변 = max(200, 6*h), 160x160 타일, 40개씩
    tiles = []
    for _, e in ex.sort_values(["relation", "conf"]).iterrows():
        im = cv2.imread(f"{a.src}/{e.session}/images/{e.filename}")
        for _, t in bx[(bx.session == e.session) & (bx.filename == e.filename)].iterrows():
            is_ex = abs(t.x1 - e.x1) < 1e-3 and abs(t.y1 - e.y1) < 1e-3
            col = (0, 0, 255) if is_ex else ((0, 200, 0) if t.color == "green" else (200, 0, 200) if t.color == "magenta" else (0, 200, 255))
            cv2.rectangle(im, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), col, 1)
        cx, cy, half = int((e.x1 + e.x2) / 2), int((e.y1 + e.y2) / 2), int(max(100, 3 * e.h))
        H, W = im.shape[:2]
        crop = im[max(0, cy - half):min(H, cy + half), max(0, cx - half):min(W, cx + half)]
        tile = cv2.resize(crop, (240, 240))
        cv2.putText(tile, f"{e.relation[:10]} {e.conf:.2f}", (3, 14), 0, 0.45, (0, 0, 255), 1)
        cv2.putText(tile, f"{e.session[5:9]} {e.filename[5:11]} h{e.h:.0f}", (3, 234), 0, 0.42, (0, 0, 0), 1)
        tiles.append(tile)
    for k in range(0, len(tiles), 30):
        t = tiles[k:k + 30] + [np.full((240, 240, 3), 255, np.uint8)] * (-len(tiles[k:k + 30]) % 6)
        cv2.imwrite(f"{out}/sheet_{k // 30:02d}.jpg", np.vstack([np.hstack(t[i:i + 6]) for i in range(0, len(t), 6)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("시트:", sorted(f for f in os.listdir(out) if f.startswith("sheet_")))
