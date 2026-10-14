#!/usr/bin/env python
"""pair_select.py — swatch_db.csv 의 모든 후보 쌍을 평가해 아군/적군 색 쌍을 고른다 (색 재설계 10/4 §1-2).

지표 (전부 a*b* 평면, L* 제외):
  sep_ab        두 후보 평균색 사이의 거리
  spread        두 후보 흩어짐의 평균. 후보의 흩어짐 = 사진별 평균색이 후보 평균에서 퍼진 RMS
                (판정은 박스 평균색으로 하므로 픽셀 잡음이 아니라 평균색의 흔들림을 본다).
                사진이 3장 미만이면 추정할 수 없어 패치 안 픽셀 흩어짐(ab_rms)의 평균을 대신 쓴다 (보수적, 표에 'px').
  safety        sep_ab / spread  (모든 조건을 풀링)
  safety_worst  조건(시간대 x 거리 x 위치)마다 같은 계산을 해서 가장 나쁜 값
  drift         후보별로 시간대마다 평균색을 내서 그 사이 최대 이동량, 쌍은 둘 중 큰 값

통과 기준 (전부 만족):
  safety_worst >= --safety-min (5)
  각 후보 C* >= --c-min (25) at --check-dist (2.0 m)
  각 후보 고채도 픽셀 중앙값 >= --npx-min (40) at --check-dist
  배경 샘플과의 a*b* 거리 >= --bg-mult (3) x spread  (배경 행이 없으면 '미검증'으로 실패 처리)
통과 쌍 중 drift 가 가장 작은 쌍을 추천. 표는 상위 --top (5) 쌍.
--allow-untested: --check-dist 행이 없거나 배경 행이 없어 검사 자체를 못 하는 경우만 실패 대신 경고 (한 조건만 찍은 예비 실행용).
  값이 있는데 기준 미달이면 그대로 실패. 2 m·배경 촬영이 들어오면 이 옵션 없이 다시 돌린다.

사용:
  python tools/pair_select.py --db swatch_db.csv [--check-dist 2.0] [--out pairs.csv]
"""
import argparse, itertools, sys
import numpy as np
import pandas as pd

MIN_SAMPLES = 3


def spread_of(g: pd.DataFrame):
    """후보 하나(여러 행)의 흩어짐과 그 근거('samples' | 'px')."""
    ab = g[["a", "b"]].to_numpy(float)
    if len(ab) >= MIN_SAMPLES:
        mu = ab.mean(0)
        return float(np.sqrt(((ab - mu) ** 2).sum(1).mean())), "samples"
    return float(g["ab_rms"].mean()), "px"


def ab_mean(g: pd.DataFrame) -> np.ndarray:
    return g[["a", "b"]].to_numpy(float).mean(0)


def pair_metrics(ga: pd.DataFrame, gb: pd.DataFrame):
    sa, ka = spread_of(ga)
    sb, kb = spread_of(gb)
    sep = float(np.linalg.norm(ab_mean(ga) - ab_mean(gb)))
    spread = (sa + sb) / 2
    return sep, spread, (sep / spread if spread > 0 else float("inf")), ("px" if "px" in (ka, kb) else "samples")


def drift_of(g: pd.DataFrame) -> float:
    means = [ab_mean(t) for _, t in g.groupby("time")]
    if len(means) < 2:
        return 0.0
    return float(max(np.linalg.norm(p - q) for p, q in itertools.combinations(means, 2)))


def main():
    ap = argparse.ArgumentParser(description="Evaluate all candidate pairs in swatch_db.csv")
    ap.add_argument("--db", default="swatch_db.csv")
    ap.add_argument("--check-dist", type=float, default=2.0, help="distance (m) for the C* and pixel-count checks")
    ap.add_argument("--dist-tol", type=float, default=0.15)
    ap.add_argument("--safety-min", type=float, default=5.0)
    ap.add_argument("--c-min", type=float, default=25.0)
    ap.add_argument("--npx-min", type=float, default=40.0)
    ap.add_argument("--bg-mult", type=float, default=3.0)
    ap.add_argument("--allow-untested", action="store_true",
                    help="warn instead of fail when there are no rows at --check-dist or no background rows")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="pairs.csv")
    a = ap.parse_args()

    db = pd.read_csv(a.db)
    db["dist"] = pd.to_numeric(db["dist"], errors="coerce")
    for col in ("time", "pos"):
        db[col] = db[col].fillna("").astype(str)
    sw = db[db["kind"].fillna("swatch") == "swatch"]
    bg = db[db["kind"] == "background"]
    cands = sorted(sw["candidate"].unique())
    if len(cands) < 2:
        sys.exit("[ERROR] need at least 2 candidates in the db")
    jpeg_any = bool(sw["jpeg_input"].astype(str).str.lower().eq("true").any())

    # 후보별 요약
    near = sw[(sw["dist"] - a.check_dist).abs() <= a.dist_tol]
    summ = {}
    for c, g in sw.groupby("candidate"):
        gn = near[near["candidate"] == c]
        sp, kind = spread_of(g)
        summ[c] = dict(n_rows=len(g), C_all=float(g["C"].mean()), spread=sp, spread_kind=kind, drift=drift_of(g),
                       C_check=float(gn["C"].mean()) if len(gn) else float("nan"),
                       npx_check=float(gn["n_px"].median()) if len(gn) else float("nan"),
                       times=sorted(g["time"].unique()))
    print(f"[CANDIDATES] {len(cands)}   check distance {a.check_dist} m (+/-{a.dist_tol})"
          + ("   ** includes JPEG input (dry run) **" if jpeg_any else ""))
    print(f"  {'candidate':14s} {'rows':>5s} {'C*all':>6s} {'C*@d':>6s} {'npx@d':>6s} {'spread':>7s} {'drift':>6s}  times")
    for c in cands:
        s = summ[c]
        print(f"  {c:14s} {s['n_rows']:5d} {s['C_all']:6.1f} {s['C_check']:6.1f} {s['npx_check']:6.0f} "
              f"{s['spread']:6.1f}{'*' if s['spread_kind'] == 'px' else ' '} {s['drift']:6.1f}  {','.join(s['times'])}")

    rows = []
    for ca, cb in itertools.combinations(cands, 2):
        ga, gb = sw[sw["candidate"] == ca], sw[sw["candidate"] == cb]
        sep, spread, safety, kind = pair_metrics(ga, gb)
        # 조건별 최악
        worst, worst_cond = float("inf"), "-"
        for cond, gc in sw[sw["candidate"].isin([ca, cb])].groupby(["time", "dist", "pos"], dropna=False):
            gca, gcb = gc[gc["candidate"] == ca], gc[gc["candidate"] == cb]
            if len(gca) == 0 or len(gcb) == 0:
                continue
            s_c = pair_metrics(gca, gcb)[2]
            if s_c < worst:
                worst, worst_cond = s_c, "/".join(str(v) for v in cond)
        # 배경 거리
        if len(bg):
            bg_ab = bg[["a", "b"]].to_numpy(float)
            d_bg = min(float(np.linalg.norm(bg_ab - ab_mean(g), axis=1).min()) for g in (ga, gb))
            bg_ratio = d_bg / spread if spread > 0 else float("inf")
        else:
            bg_ratio = float("nan")
        sa, sb = summ[ca], summ[cb]
        fails, warns = [], []
        if not worst >= a.safety_min:
            fails.append("safety_worst" if worst != float("inf") else "no_shared_condition")
        if a.allow_untested and np.isnan([sa["C_check"], sb["C_check"]]).any():
            warns.append("C*_untested")
        elif not (sa["C_check"] >= a.c_min and sb["C_check"] >= a.c_min):
            fails.append("C*")
        if a.allow_untested and np.isnan([sa["npx_check"], sb["npx_check"]]).any():
            warns.append("n_px_untested")
        elif not (sa["npx_check"] >= a.npx_min and sb["npx_check"] >= a.npx_min):
            fails.append("n_px")
        if np.isnan(bg_ratio):
            (warns if a.allow_untested else fails).append("bg_untested")
        elif bg_ratio < a.bg_mult:
            fails.append("bg")
        rows.append(dict(pair=f"{ca} + {cb}", sep_ab=sep, spread=spread, spread_kind=kind, safety=safety,
                         safety_worst=worst, worst_cond=worst_cond, C_a=sa["C_check"], C_b=sb["C_check"],
                         npx_a=sa["npx_check"], npx_b=sb["npx_check"], bg_ratio=bg_ratio,
                         drift=max(sa["drift"], sb["drift"]), passed=not fails, fails=",".join(fails), warns=",".join(warns),
                         jpeg_input=jpeg_any))

    res = pd.DataFrame(rows)
    res["_order"] = np.where(res["passed"], 0, 1)
    res = res.sort_values(["_order", "drift", "safety_worst"], ascending=[True, True, False]) if res["passed"].any() \
        else res.sort_values("safety_worst", ascending=False)
    res.drop(columns="_order").to_csv(a.out, index=False)

    print(f"\n[PAIRS] {len(res)} pairs, {int(res['passed'].sum())} passed   (top {a.top})")
    print(f"  {'pair':30s} {'sep_ab':>6s} {'spread':>7s} {'safety':>6s} {'worst':>6s} {'C*a':>5s} {'C*b':>5s} "
          f"{'npx_a':>5s} {'npx_b':>5s} {'bg':>5s} {'drift':>5s}  result")
    for _, r in res.head(a.top).iterrows():
        print(f"  {r['pair']:30s} {r['sep_ab']:6.1f} {r['spread']:6.1f}{'*' if r['spread_kind'] == 'px' else ' '} "
              f"{r['safety']:6.2f} {r['safety_worst']:6.2f} {r['C_a']:5.1f} {r['C_b']:5.1f} "
              f"{r['npx_a']:5.0f} {r['npx_b']:5.0f} {r['bg_ratio']:5.1f} {r['drift']:5.1f}  "
              f"{'PASS' if r['passed'] else 'fail: ' + r['fails']}"
              f"{'  (warn: ' + r['warns'] + ')' if r['warns'] else ''}")
    if (res.head(a.top)["spread_kind"] == "px").any():
        print("  (* spread from pixel noise because a condition had fewer than 3 photos)")
    if res["passed"].any():
        best = res[res["passed"]].iloc[0]
        print(f"\n[RECOMMEND] {best['pair']}  (smallest time drift {best['drift']:.1f} among passing pairs)")
    else:
        print("\n[RECOMMEND] none passed")
    if a.allow_untested and (res["warns"] != "").any():
        print(f"[WARN] --allow-untested: pairs passed without the {a.check_dist} m C*/n_px check and/or the background "
              "check (no data). Re-run without --allow-untested once those photos are in the db.")
    if jpeg_any:
        print("[NOTE] jpeg_input=true: JPEG rows were used (dry run). Do not use these numbers to choose colors.")
    print(f"[SAVED] {a.out}")


if __name__ == "__main__":
    main()
