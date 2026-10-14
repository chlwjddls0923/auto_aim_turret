#!/usr/bin/env python
"""build_color_refs.py — 새 타겟 색(적군 B04 마젠타, 아군 B06 초록)의 기준 Lab 을 실사진에서 재서 color_refs.json 을 만든다 (10/4).

입력: 아군·적군이 한 프레임에 나란히 있는 PNG 여러 장 (파일 또는 폴더, 폴더는 하위 폴더까지). JPEG 는 받지 않는다.
처리:
  1) pipeline_config.json 의 검출기(r3_coco)로 박스 검출. 정확히 2개가 아니면 그 사진은 건너뛰고 경고.
  2) 박스 안 고채도 픽셀(colorlib.patch_stats, sat_min)의 Lab 평균.
  3) a* 부호로 분류: a* > 0 → 적군(마젠타), a* < 0 → 아군(초록). 두 박스가 같은 부호면 건너뛰고 경고.
  4) 거리: --dist-map CSV(filename,dist[,pos]) > 파일명 > 상위 폴더 이름('1.5m', '2.0m') > 옆 JSON 의 note > --dist.
     배치(아군이 왼쪽/오른쪽)는 박스 위치로 자동. 촬영 설정은 PNG 옆 JSON(03_preview.py 가 저장)에서 읽는다.
계산 (전부 a*b* 평면, L* 제외):
  기준색      = 사진별 평균의 전체 평균
  이동 RMS    = 사진별 평균이 기준색에서 퍼진 RMS  (사진 간 흔들림 = 판정이 실제로 겪는 흩어짐)
  안전비      = 두 기준색 거리 / 두 이동 RMS 의 평균.  사진이 --min-images(3) 장 미만이면 0 으로 저장(게이트가 막음)
  margin 제안 = max(0.2 × 두 기준색 거리, 3 × 큰 쪽 이동 RMS)
  min_px 제안 = max(20, ⌊0.5 × 가장 먼 거리 사진들의 최소 고채도 픽셀 수⌋)
출력: color_refs.json (metric='ab' → target_pipeline 이 a*b* 거리로 판정), 같은 폴더에 color_refs_samples.csv(사진별 표).
  기존 color_refs.json 은 color_refs.json.<시각>.bak 으로 남긴다.

검증 (--verify): 같은 PNG 로 target_pipeline.Pipeline(설정 그대로)을 돌려 박스마다
  예상(a* 부호) / 판정 / d_friend / d_enemy / margin 여유 를 표로 내고, 오판 수와 unknown 비율을 요약한다. 오판이 있으면 종료 코드 1.

사용 (파이, ~/src/pipeline 에서):
  venv/bin/python tools/build_color_refs.py ../03_captures/png/target_1004 [--dist-map dist.csv] [--save-dir out/]
  venv/bin/python tools/build_color_refs.py ../03_captures/png/target_1004 --verify
"""
import argparse, contextlib, csv, json, os, re, shutil, sys, time, warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colorlib as C  # noqa: E402
import target_pipeline as T  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "pipeline_config.json"
DIST_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*m(?![a-z])", re.I)
SIDES = ("friend", "enemy")


# ---------- 입력 ----------
def collect_pngs(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() == ".png")
        elif p.suffix.lower() in (".jpg", ".jpeg"):
            sys.exit(f"[ERROR] {p} is JPEG. Color references must be built from PNG.")
        elif p.is_file():
            out.append(p)
        else:
            sys.exit(f"[ERROR] not found: {p}")
    return sorted(set(out))


def read_meta(png: Path) -> dict:
    j = png.with_suffix(".json")
    try:
        return json.loads(j.read_text(encoding="utf-8")) if j.is_file() else {}
    except (OSError, ValueError):
        return {}


def parse_dist(text):
    m = DIST_RE.search(str(text))
    return float(m.group(1)) if m else None


def load_dist_map(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["filename"].strip(): (float(r["dist"]) if r.get("dist") else None, (r.get("pos") or "").strip())
                for r in csv.DictReader(f)}


def dist_of(png: Path, meta: dict, dist_map: dict, default):
    """(거리, 출처). 우선순위: dist-map > 파일명 > 상위 폴더 > JSON note > --dist."""
    if png.name in dist_map and dist_map[png.name][0] is not None:
        return dist_map[png.name][0], "map"
    for src, text in (("name", png.stem), ("dir", png.parent.name), ("note", meta.get("note", ""))):
        d = parse_dist(text)
        if d is not None:
            return d, src
    return default, ("arg" if default is not None else "-")


def camera_conditions(meta: dict) -> dict:
    lk, ac = meta.get("locked", {}), meta.get("actual", {})
    return dict(ev=lk.get("exposure_value"), colour_gains=lk.get("colour_gains"), auto_exposure=lk.get("auto_exposure"),
                exposure_us=ac.get("ExposureTime"), analogue_gain=ac.get("AnalogueGain"), lux=ac.get("Lux"),
                colour_temp=ac.get("ColourTemperature"))


# ---------- 분류·요약 (모델 없이 시험 가능) ----------
def split_by_a_sign(measured):
    """measured: [(det, stats), (det, stats)] → ({'friend': (det, stats), 'enemy': (det, stats)}, None) 또는 (None, 이유)."""
    if any(st["lab"] is None for _, st in measured):
        return None, "no high-saturation pixels in a box"
    pos = [m for m in measured if m[1]["lab"][1] > 0]
    neg = [m for m in measured if m[1]["lab"][1] < 0]
    if len(pos) != 1 or len(neg) != 1:
        a = ", ".join(f"{st['lab'][1]:+.1f}" for _, st in measured)
        return None, f"both boxes have the same a* sign (a* = {a})"
    return dict(friend=neg[0], enemy=pos[0]), None


def summarize(samples, min_images=3):
    """samples: [{'friend': [L,a,b], 'enemy': [L,a,b], ...}, ...] → 기준색·이동 RMS·안전비·margin 제안."""
    out = dict(n=len(samples), mean={}, rms_ab={}, rms_L={})
    for s in SIDES:
        lab = np.array([r[s] for r in samples], float)
        mu = lab.mean(0)
        out["mean"][s] = mu
        out["rms_ab"][s] = float(np.sqrt(((lab[:, 1:] - mu[1:]) ** 2).sum(1).mean()))
        out["rms_L"][s] = float(np.sqrt(((lab[:, 0] - mu[0]) ** 2).mean()))
    sep = C.ab_distance(out["mean"]["friend"], out["mean"]["enemy"])
    spread = (out["rms_ab"]["friend"] + out["rms_ab"]["enemy"]) / 2
    out["sep_ab"], out["spread"] = sep, spread
    out["enough"] = len(samples) >= min_images
    out["safety_ratio"] = (sep / spread if spread > 0 else float("inf")) if out["enough"] else 0.0
    out["margin"] = float(max(0.2 * sep, 3 * max(out["rms_ab"].values())))
    out["margin_basis"] = "0.2*sep" if 0.2 * sep >= 3 * max(out["rms_ab"].values()) else "3*rms"
    return out


def suggest_min_px(samples, floor=20):
    known = [r for r in samples if r.get("dist") is not None]
    pool = [r for r in known if r["dist"] == max(k["dist"] for k in known)] if known else samples
    return int(max(floor, np.floor(0.5 * min(min(r["friend_npx"], r["enemy_npx"]) for r in pool))))


def decompose(samples, summ, need=3.0, expect_sep=91.4, expect_spread=7.2):
    """안전비 미달 원인: 거리 부족인지 이동이 큰지. 반대쪽 값을 기대치(색지 실측)로 바꿨을 때의 안전비로 가른다."""
    sep, spread = summ["sep_ab"], summ["spread"]
    lines = [f"  needed sep_ab for {need:.1f}: {need * spread:.1f}   actual {sep:.1f}   "
             f"(swatch sheet at 1.75 m: {expect_sep:.1f})",
             f"  spread {spread:.1f} = mean of friend RMS {summ['rms_ab']['friend']:.1f} / enemy RMS {summ['rms_ab']['enemy']:.1f}   "
             f"(swatch pixel spread: {expect_spread:.1f})",
             f"  if spread were {expect_spread:.1f}: safety {sep / expect_spread:.2f}   "
             f"if sep were {expect_sep:.1f}: safety {expect_sep / spread if spread else float('inf'):.2f}"]
    short_sep, big_spread = sep / expect_spread < need, expect_sep / spread < need if spread else False
    cause = ("both: separation too small AND image-to-image drift too large" if short_sep and big_spread else
             "separation too small (colors closer on camera than on the swatch sheet)" if short_sep else
             "drift too large (colors move between photos)" if big_spread else
             "neither alone — both are a bit worse than the swatch measurement")
    lines.append(f"  -> cause: {cause}")
    # 어느 조건이 이동을 만드는지: 조건(거리 × 배치)별 평균이 전체 기준색에서 얼마나 떨어졌나
    groups = {}
    for r in samples:
        groups.setdefault((r.get("dist"), r["layout"]), []).append(r)
    lines.append("  per condition: offset of the condition mean from the overall reference (a*b*)")
    for (d, lay), g in sorted(groups.items(), key=lambda kv: (kv[0][0] is None, kv[0][0] or 0, kv[0][1])):
        offs = {s: C.ab_distance(np.mean([r[s] for r in g], 0), summ["mean"][s]) for s in SIDES}
        lines.append(f"    {str(d) + ' m' if d is not None else '? m':>7s} {lay:8s} n={len(g)}  friend {offs['friend']:5.1f}  "
                     f"enemy {offs['enemy']:5.1f}")
    return lines


# ---------- 실행 ----------
def quiet(fn, *args):
    """ultralytics 로딩 로그를 stderr 로 돌린다."""
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args)


def save_check_image(frame, measured_or_dets, path, labels):
    img = frame.copy()
    for (x1, y1, x2, y2), lab in zip(measured_or_dets, labels):
        col = (204, 102, 255) if lab.startswith("E") else (90, 217, 79) if lab.startswith("F") else (0, 200, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, lab, (int(x1), max(14, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def build(a, pngs, cfg):
    sat_min = a.sat_min if a.sat_min is not None else cfg.get("color", {}).get("sat_min", 60)
    det = quiet(T.load_detector, a.config)
    dist_map = load_dist_map(a.dist_map)
    samples, skipped, conds = [], [], []
    print(f"{'file':28s} {'dist':>5s} {'layout':8s} {'friend L*  a*    b*':>21s} {'n_px':>6s} "
          f"{'enemy L*  a*    b*':>21s} {'n_px':>6s} {'ab_dist':>7s}")
    for p in pngs:
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            skipped.append((p.name, "cannot read")); continue
        meta = read_meta(p)
        dist, dsrc = dist_of(p, meta, dist_map, a.dist)
        dets = det.detect(frame)
        if len(dets) != 2:
            skipped.append((p.name, f"{len(dets)} boxes (need 2)"))
            if a.save_dir:
                save_check_image(frame, [d.xyxy for d in dets], Path(a.save_dir) / f"{p.stem}_SKIP.png", ["?"] * len(dets))
            continue
        measured = [(d, C.patch_stats(frame, d.xyxy, sat_min=sat_min)) for d in dets]
        pair, why = split_by_a_sign(measured)
        if pair is None:
            skipped.append((p.name, why)); continue
        (fd, fs), (ed, es) = pair["friend"], pair["enemy"]
        layout = "F-left" if fd.cx < ed.cx else "F-right"
        pos = dist_map.get(p.name, (None, ""))[1] or layout
        r = dict(file=p.name, dist=dist, dist_src=dsrc, layout=pos, friend=fs["lab"], enemy=es["lab"],
                 friend_npx=fs["n_px"], enemy_npx=es["n_px"], ab_dist=C.ab_distance(fs["lab"], es["lab"]),
                 friend_box=[round(v, 1) for v in fd.xyxy], enemy_box=[round(v, 1) for v in ed.xyxy],
                 weak_a=min(abs(fs["lab"][1]), abs(es["lab"][1])) < 10)
        samples.append(r)
        conds.append(camera_conditions(meta))
        f, e = r["friend"], r["enemy"]
        print(f"{p.name:28s} {dist if dist is not None else '?':>5} {pos:8s} {f[0]:6.1f} {f[1]:+6.1f} {f[2]:+6.1f} "
              f"{r['friend_npx']:6d} {e[0]:6.1f} {e[1]:+6.1f} {e[2]:+6.1f} {r['enemy_npx']:6d} {r['ab_dist']:7.1f}"
              + ("  |a*|<10" if r["weak_a"] else ""))
        if a.save_dir:
            save_check_image(frame, [fd.xyxy, ed.xyxy], Path(a.save_dir) / f"{p.stem}_check.png",
                             [f"F a*{f[1]:+.0f}", f"E a*{e[1]:+.0f}"])
    for name, why in skipped:
        print(f"[WARN] skipped {name}: {why}")
    nodist = [r["file"] for r in samples if r["dist"] is None]
    if nodist:
        print(f"[WARN] no distance for {len(nodist)} image(s) (use a '1.5m' folder name, --dist-map or --dist): "
              + ", ".join(nodist[:5]) + (" ..." if len(nodist) > 5 else ""))
    if not samples:
        sys.exit("[ERROR] no usable image (need exactly 2 boxes with opposite a* signs)")
    return samples, skipped, conds, sat_min


def report_and_save(a, samples, skipped, conds, sat_min, cfg, out_path):
    summ = summarize(samples, a.min_images)
    min_px = suggest_min_px(samples)
    print(f"\n[REFERENCE] {summ['n']} images used, {len(skipped)} skipped   (a*b* only, L* not used)")
    for s in SIDES:
        L, aa, bb = summ["mean"][s]
        print(f"  {s:6s}  Lab ({L:5.1f}, {aa:+6.1f}, {bb:+6.1f})  C* {np.hypot(aa, bb):5.1f}   "
              f"drift RMS a*b* {summ['rms_ab'][s]:5.2f}   (L* RMS {summ['rms_L'][s]:4.1f}, not used)")
    print(f"  sep_ab {summ['sep_ab']:.1f}   spread {summ['spread']:.2f}   "
          f"safety {summ['safety_ratio']:.2f}" + ("" if summ["enough"] else f"  (fewer than {a.min_images} images -> saved as 0)"))
    print(f"  margin {summ['margin']:.1f} ({summ['margin_basis']})   min_px {min_px}")
    need = cfg.get("color", {}).get("min_safety_ratio", 3.0)
    if summ["margin"] >= summ["sep_ab"]:
        print("[WARN] margin >= sep_ab: every box would be 'ambiguous'.")
    if summ["safety_ratio"] < need:
        print(f"\n[SAFETY] {summ['safety_ratio']:.2f} < {need} -> ColorClassifier will NOT use these refs (all unknown).")
        if summ["enough"]:
            for line in decompose(samples, summ, need, a.expect_sep, a.expect_spread):
                print(line)
    evs = sorted({c["ev"] for c in conds if c["ev"] is not None})
    gains = sorted({tuple(round(g, 3) for g in c["colour_gains"]) for c in conds if c["colour_gains"]})
    if len(evs) > 1 or len(gains) > 1:
        print(f"[WARN] camera settings differ between images: EV {evs}, colour gains {gains}")
    dists = sorted({r["dist"] for r in samples if r["dist"] is not None})
    vals = lambda k: [c[k] for c in conds if c[k] is not None]
    rng = lambda k: [round(min(vals(k)), 1), round(max(vals(k)), 1)] if vals(k) else None

    refs = dict(
        refs={s: [round(float(v), 3) for v in summ["mean"][s]] for s in SIDES},
        metric="ab", margin=round(summ["margin"], 2), margin_basis=summ["margin_basis"], min_px=min_px,
        sat_min=sat_min, top_frac=None, ball_frac=None,
        safety_ratio=round(float(summ["safety_ratio"]), 3) if np.isfinite(summ["safety_ratio"]) else 999.0,
        sep_ab=round(summ["sep_ab"], 2), drift_rms_ab={s: round(summ["rms_ab"][s], 3) for s in SIDES},
        n_images=summ["n"], jpeg_input=False,
        conditions=dict(distances_m=dists, layouts=dict(Counter(r["layout"] for r in samples)), ev=evs,
                        colour_gains=[list(g) for g in gains], exposure_us=rng("exposure_us"),
                        analogue_gain=rng("analogue_gain"), lux=rng("lux"), colour_temp=rng("colour_temp")),
        colors=dict(enemy="B04 magenta #FF66CC", friend="B06 green #4FD95A"),
        source=sorted(r["file"] for r in samples), skipped=[f"{n}: {w}" for n, w in skipped],
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"), created_by="build_color_refs.py")
    if out_path.exists():
        bak = out_path.with_name(out_path.name + time.strftime(".%Y%m%d_%H%M%S.bak"))
        shutil.copy2(out_path, bak)
        print(f"[BACKUP] {bak}")
    out_path.write_text(json.dumps(refs, indent=1, ensure_ascii=False), encoding="utf-8")
    csv_path = out_path.with_name(out_path.stem + "_samples.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "dist", "dist_src", "layout", "friend_L", "friend_a", "friend_b", "friend_npx",
                    "enemy_L", "enemy_a", "enemy_b", "enemy_npx", "ab_dist", "friend_box", "enemy_box"])
        for r in samples:
            w.writerow([r["file"], r["dist"], r["dist_src"], r["layout"], *np.round(r["friend"], 2), r["friend_npx"],
                        *np.round(r["enemy"], 2), r["enemy_npx"], round(r["ab_dist"], 2),
                        " ".join(map(str, r["friend_box"])), " ".join(map(str, r["enemy_box"]))])
    print(f"[SAVED] {out_path}\n[SAVED] {csv_path}")


def verify(a, pngs):
    pipe = quiet(T.Pipeline, a.config)
    cc = pipe.color
    print(f"[VERIFY] color mode={cc.mode}  refs={'loaded' if cc.cfg else 'NOT loaded'}"
          + (f" (metric={cc.cfg.get('metric', 'dE2000')}, margin={cc.cfg.get('margin')})" if cc.cfg else "")
          + f"  relative={'on' if cc.ref is not None else 'off'}")
    print(f"{'file':28s} {'#':>1s} {'cx':>5s} {'h':>4s} {'a*':>6s} {'expect':7s} {'result':7s} {'reason':10s} "
          f"{'mode':8s} {'d_fr':>6s} {'d_en':>6s} {'slack':>6s}")
    n_box, wrong, unk, odd, decisions = 0, [], Counter(), [], Counter()
    for p in pngs:
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        out = pipe.run(frame)
        decisions[out["decision"]["action"]] += 1
        if len(out["dets"]) != 2:
            odd.append(f"{p.name}: {len(out['dets'])} boxes")
        for i, d in enumerate(out["dets"]):
            n_box += 1
            st = C.patch_stats(frame, d.xyxy, sat_min=cc.sat_min)
            a_star = None if st["lab"] is None else float(st["lab"][1])
            expect = "-" if a_star is None else ("enemy" if a_star > 0 else "friend")
            if d.mode == "a_sign":
                slack = abs(d.a_star) - cc.a_sign_thr if d.a_star is not None else None
            elif d.d_friend_ab is not None:
                m = cc.margin_ab if d.mode == "relative" else cc.cfg.get("margin", 5.0)
                slack = abs(d.d_friend_ab - d.d_enemy_ab) - m
            else:
                slack = None
            if d.side == "unknown":
                unk[d.reason] += 1
            elif expect != "-" and d.side != expect:
                wrong.append(f"{p.name} box{i}: expected {expect}, got {d.side}")
            fmt = lambda v: f"{v:6.1f}" if v is not None else f"{'-':>6s}"
            print(f"{p.name:28s} {i:1d} {d.cx:5.0f} {d.h:4.0f} {fmt(a_star)} {expect:7s} {d.side:7s} {d.reason:10s} "
                  f"{str(d.mode):8s} {fmt(d.d_friend_ab)} {fmt(d.d_enemy_ab)} {fmt(slack)}"
                  + ("  <-- WRONG" if d.side not in ("unknown", expect) and expect != "-" else ""))
    n_unk = sum(unk.values())
    print(f"\n[SUMMARY] {len(pngs)} images, {n_box} boxes")
    print(f"  misjudged: {len(wrong)}" + ("" if not wrong else "   <-- " + "; ".join(wrong)))
    print(f"  unknown  : {n_unk}/{n_box} = {100 * n_unk / n_box if n_box else 0:.1f}%"
          + (f"   ({', '.join(f'{k} {v}' for k, v in unk.items())})" if unk else ""))
    print(f"  decisions: {dict(decisions)}")
    if odd:
        print(f"  box count != 2 in {len(odd)} image(s): " + "; ".join(odd))
    print("  (expected = a* sign of the box; verifying on the same photos used to build the refs is in-sample)")
    return 1 if wrong else 0


def main():
    ap = argparse.ArgumentParser(description="Build color_refs.json for the new target colors (or --verify it)")
    ap.add_argument("inputs", nargs="+", help="PNG files or folders (folders are searched recursively)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", help="output color_refs.json (default: color.refs in the config)")
    ap.add_argument("--dist", type=float, default=None, help="distance (m) for images without one in name/folder/map")
    ap.add_argument("--dist-map", help="CSV with columns filename,dist[,pos]")
    ap.add_argument("--sat-min", type=int, default=None, help="high-saturation threshold (default: color.sat_min)")
    ap.add_argument("--min-images", type=int, default=3, help="fewer usable images -> safety_ratio saved as 0")
    ap.add_argument("--save-dir", help="write check images with F/E boxes here")
    ap.add_argument("--expect-sep", type=float, default=91.4, help="swatch a*b* distance for the cause breakdown")
    ap.add_argument("--expect-spread", type=float, default=7.2, help="swatch pixel spread for the cause breakdown")
    ap.add_argument("--verify", action="store_true", help="run target_pipeline on the images and tabulate decisions")
    a = ap.parse_args()

    pngs = collect_pngs(a.inputs)
    if not pngs:
        sys.exit("[ERROR] no PNG found")
    if a.verify:
        sys.exit(verify(a, pngs))
    cfg, rel = T.read_config(a.config)
    out_path = Path(a.out) if a.out else Path(rel(cfg.get("color", {}).get("refs") or "color_refs.json"))
    if a.save_dir:
        Path(a.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"[BUILD] {len(pngs)} PNG -> {out_path}")
    samples, skipped, conds, sat_min = build(a, pngs, cfg)
    report_and_save(a, samples, skipped, conds, sat_min, cfg, out_path)


if __name__ == "__main__":
    main()
