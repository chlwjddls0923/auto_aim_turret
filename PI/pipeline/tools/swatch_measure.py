#!/usr/bin/env python
"""swatch_measure.py — 후보 색지들의 '카메라에서의 색'을 측정해 swatch_db.csv에 누적한다 (색 재설계 10/4 §1-1).

입력 (색 측정은 PNG/raw만. JPEG는 색차 서브샘플링으로 작은 패치 색이 희석되므로 거부한다):
  1) 한 장 + 수동 ROI 목록
       python tools/swatch_measure.py --image f.png --rois rois.csv --time noon --dist 1.75 --pos center
       rois.csv 열: name,x1,y1,x2,y2[,kind]   kind = swatch(기본) | background
  2) 한 장 + 고채도 덩어리 자동 분리 (이름은 auto_1.. → 나중에 rois.csv로 옮겨 이름을 붙인다)
       python tools/swatch_measure.py --image f.png --auto --time noon --dist 1.75 --pos center
  3) 촬영 폴더 + 검출 박스 (시운전용. color_separation.py 의 samples.csv 박스를 다시 쓴다)
       python tools/swatch_measure.py --boxes color_report/samples.csv --dataset-dirs D1 D2 --ball-frac 0.45 --allow-jpeg
       후보 이름 = 박스의 cls(enemy/friend). 조건 태그는 note(거리·위치)와 촬영 시각(시간대)에서 뽑는다.

번호 이미지: --image 모드는 ROI 박스와 이름(auto 는 번호)을 그린 PNG 를 저장한다 (--annot, 기본 = db 옆 <사진 이름>_rois.png).
  초록 = 측정됨, 빨강 = 픽셀 없음. auto_N 이 어느 색지인지 사람이 이걸 보고 맞춘다.

출력: swatch_db.csv 에 한 줄 = (사진 1장 × 후보 1개)
  n_px(고채도 픽셀 수), L,a,b(평균), C(a*b* 채도), ab_rms(패치 안 a*b* 흩어짐), hue_deg,S,V(참고), 조건 태그, jpeg_input
  background 행은 채도 선택 없이 ROI 전체 픽셀을 쓴다 (배경은 원래 채도가 낮다).
"""
import argparse, csv, json, os, re, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import colorlib as C

DB_COLUMNS = ["measured_at", "source", "jpeg_input", "candidate", "kind", "time", "dist", "pos",
              "n_px", "L", "a", "b", "C", "ab_rms", "hue_deg", "S", "V", "x1", "y1", "x2", "y2"]
JPEG_EXT = {".jpg", ".jpeg"}


def is_jpeg(path) -> bool:
    return Path(path).suffix.lower() in JPEG_EXT


def check_input(path, allow_jpeg: bool) -> None:
    """JPEG는 --allow-jpeg 를 준 경우(시운전)에만 받는다."""
    if is_jpeg(path) and not allow_jpeg:
        sys.exit(f"[ERROR] {path} is JPEG. Color must be measured on PNG/raw. "
                 "(Use --allow-jpeg only for a dry run; rows are marked jpeg_input=true.)")


def measure_region(frame_bgr, box, kind="swatch", sat_min=60, ball_frac=None):
    """ROI 하나를 측정한다. swatch 는 고채도 픽셀만, background 는 전체 픽셀.
    반환 dict 또는 None(픽셀 없음)."""
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    if ball_frac:
        y2 = y1 + max(3, int((y2 - y1) * ball_frac))
    crop = frame_bgr[y1:y2, x1:x2]
    if kind == "background":
        px = crop.reshape(-1, 3)
    else:
        px = crop[C.select_target_pixels(crop, sat_min=sat_min)]
    if len(px) == 0:
        return None
    lab = C.bgr_to_lab(px)
    mu = lab.mean(0)
    ab_rms = float(np.sqrt(((lab[:, 1:] - mu[1:]) ** 2).sum(1).mean()))
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    return dict(n_px=int(len(px)), L=float(mu[0]), a=float(mu[1]), b=float(mu[2]),
                C=float(np.hypot(mu[1], mu[2])), ab_rms=ab_rms,
                hue_deg=C.circular_mean_deg(hsv[:, 0] * 2), S=float(hsv[:, 1].mean() / 255),
                V=float(hsv[:, 2].mean() / 255), x1=x1, y1=y1, x2=x2, y2=y2)


def auto_rois(frame_bgr, sat_min=90, min_area=150, max_blobs=20):
    """고채도 덩어리를 자동으로 찾아 박스 목록을 돌려준다 (큰 것부터)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] >= sat_min).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = [(stats[i, 4], stats[i, 0], stats[i, 1], stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3])
             for i in range(1, n) if stats[i, 4] >= min_area]
    boxes.sort(reverse=True)
    return [(f"auto_{k + 1}", b[1], b[2], b[3], b[4], "swatch") for k, b in enumerate(boxes[:max_blobs])]


def save_annotated(frame_bgr, rois, measured, out_path):
    """ROI 박스와 이름을 그려 저장한다. 이름이 auto_N 이면 N 만 쓴다."""
    img = frame_bgr.copy()
    for name, x1, y1, x2, y2, _ in rois:
        p1, p2 = (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1)
        cv2.rectangle(img, p1, p2, (0, 255, 0) if name in measured else (0, 0, 255), 1)
        label = name[5:] if name.startswith("auto_") else name
        org = (int(x1) + 2, int(y1) + 14)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def time_tag_from_iso(ts: str) -> str:
    """촬영 시각 → 시간대 태그. 11시 전 morning / 16시 전 noon / 그 뒤 evening."""
    try:
        h = int(ts[11:13])
    except (ValueError, IndexError):
        return "unknown"
    return "morning" if h < 11 else ("noon" if h < 16 else "evening")


def parse_note(note: str):
    """note 에서 거리(m)와 위치(left/center/right)를 뽑는다. 'center_1.2' 처럼 m 가 빠진 것도 받는다."""
    note = str(note)
    m = re.search(r"(\d+(?:\.\d+)?)\s*m?", note.replace("_", " "))
    dist = float(m.group(1)) if m else float("nan")
    pos = next((p for p in ("left", "center", "right") if p in note.lower()), "")
    return dist, pos


def append_rows(db_path: Path, rows):
    new = not db_path.exists() or db_path.stat().st_size == 0
    with db_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DB_COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in DB_COLUMNS})


def run_single(a):
    check_input(a.image, a.allow_jpeg)
    frame = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if frame is None:
        sys.exit(f"[ERROR] cannot read {a.image}")
    if a.auto:
        rois = auto_rois(frame, sat_min=a.auto_sat)
        print(f"[AUTO] {len(rois)} blobs. Copy to a rois.csv and rename them:")
        print("name,x1,y1,x2,y2,kind")
        for r in rois:
            print(",".join(str(v) for v in r))
    else:
        rois = []
        with open(a.rois, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rois.append((r["name"], float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                             (r.get("kind") or "swatch").strip()))
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for name, x1, y1, x2, y2, kind in rois:
        m = measure_region(frame, (x1, y1, x2, y2), kind=kind, sat_min=a.sat_min)
        if m is None:
            print(f"[WARN] {name}: no pixels selected")
            continue
        rows.append(dict(measured_at=now, source=a.image, jpeg_input=is_jpeg(a.image), candidate=name, kind=kind,
                         time=a.time, dist=a.dist, pos=a.pos, **m))
        print(f"  {name:12s} {kind:10s} n_px {m['n_px']:6d}  Lab ({m['L']:5.1f},{m['a']:+6.1f},{m['b']:+6.1f})  "
              f"C* {m['C']:5.1f}  ab_rms {m['ab_rms']:5.1f}")
    annot = Path(a.annot) if a.annot else Path(a.db).resolve().parent / f"{Path(a.image).stem}_rois.png"
    save_annotated(frame, rois, {r["candidate"] for r in rows}, annot)
    print(f"[ANNOT] {annot}")
    return rows


def run_dataset(a):
    """color_separation.py 의 samples.csv 박스를 다시 써서 촬영 폴더를 측정한다 (검출을 다시 하지 않는다)."""
    images, meta = {}, {}
    for d in a.dataset_dirs:
        d = Path(os.path.expanduser(d))
        for p in d.rglob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                images[p.name] = p
        with open(d / "labels.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                meta[r["filename"]] = r
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rows, skipped = [], 0
    with open(a.boxes, encoding="utf-8") as f:
        boxes = list(csv.DictReader(f))
    cache = {}
    for i, bx in enumerate(boxes):
        fn, cls = bx["filename"], bx["cls"]
        if cls not in ("enemy", "friend") or fn not in images or fn not in meta:
            skipped += 1
            continue
        p = images[fn]
        check_input(p, a.allow_jpeg)
        if fn not in cache:
            cache.clear()
            cache[fn] = cv2.imread(str(p), cv2.IMREAD_COLOR)
        m = measure_region(cache[fn], (float(bx["x1"]), float(bx["y1"]), float(bx["x2"]), float(bx["y2"])),
                           sat_min=a.sat_min, ball_frac=a.ball_frac)
        if m is None:
            skipped += 1
            continue
        r = meta[fn]
        dist, pos = parse_note(r.get("note", ""))
        rows.append(dict(measured_at=now, source=str(p), jpeg_input=is_jpeg(p), candidate=cls, kind="swatch",
                         time=time_tag_from_iso(r.get("timestamp", "")), dist=dist, pos=pos or bx.get("pos", ""), **m))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(boxes)} boxes", flush=True)
    print(f"[DATASET] {len(rows)} rows, {skipped} boxes skipped (other class / missing image / no pixels)")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Measure candidate swatch colors into swatch_db.csv")
    ap.add_argument("--image", help="PNG (or raw) frame with the candidate swatches")
    ap.add_argument("--rois", help="CSV: name,x1,y1,x2,y2[,kind]  (kind: swatch | background)")
    ap.add_argument("--auto", action="store_true", help="find high-saturation blobs automatically")
    ap.add_argument("--auto-sat", type=int, default=90, help="saturation threshold for --auto (0-255)")
    ap.add_argument("--boxes", help="samples.csv from color_separation.py (dataset mode)")
    ap.add_argument("--dataset-dirs", nargs="+", help="capture folders with images/ and labels.csv (dataset mode)")
    ap.add_argument("--time", default="unknown", help="morning | noon | evening")
    ap.add_argument("--dist", type=float, default=float("nan"), help="distance in m, e.g. 1.75")
    ap.add_argument("--pos", default="", help="center | left | right")
    ap.add_argument("--sat-min", type=int, default=60, help="high-saturation pixel threshold (OpenCV S 0-255)")
    ap.add_argument("--ball-frac", type=float, default=None, help="dataset mode: use only the top part of the box")
    ap.add_argument("--allow-jpeg", action="store_true", help="accept JPEG input (dry run only, rows get jpeg_input=true)")
    ap.add_argument("--db", default="swatch_db.csv", help="output CSV (rows are appended)")
    ap.add_argument("--annot", help="--image mode: numbered ROI image (default: <db dir>/<image stem>_rois.png)")
    a = ap.parse_args()

    if a.boxes:
        if not a.dataset_dirs:
            sys.exit("[ERROR] --boxes needs --dataset-dirs")
        rows = run_dataset(a)
    elif a.image and (a.rois or a.auto):
        rows = run_single(a)
    else:
        ap.error("give --image with --rois or --auto, or --boxes with --dataset-dirs")
    if rows:
        append_rows(Path(a.db), rows)
        n_jpeg = sum(1 for r in rows if r["jpeg_input"])
        print(f"[SAVED] {len(rows)} rows -> {a.db}" + (f"  (jpeg_input=true: {n_jpeg})" if n_jpeg else ""))


if __name__ == "__main__":
    main()
