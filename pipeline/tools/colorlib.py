"""colorlib — 타겟 색 측정·판정 공용 함수 (의존성: numpy, opencv-python 만. 파이4에서도 그대로 사용)

규칙 (설계 문서 §색 판정):
  * 색은 항상 **원본 프레임 + 원본 좌표 박스**에서 읽는다 (검출 입력이 무엇이든 무관).
  * 박스 안에서 **고채도 픽셀만** 골라 Lab 평균을 낸다 (침식 박스 평균은 폐기 — 바닥·검은 막대가 섞임).
  * 고채도 픽셀 수가 min_px 미만이면 색을 판정하지 않는다 ("few_px").
  * 두 기준색과의 dE2000 차이가 margin 미만이면 "ambiguous" → 발사하지 않는다.
"""
import numpy as np
import cv2

# ---------- 색 공간 ----------
def bgr_to_lab(bgr_u8):
    """uint8 BGR (N,3)/(H,W,3) -> float CIELAB (L 0..100, a,b ≈ -128..127). OpenCV sRGB/D65 변환을 실수 스케일로 보정."""
    arr = np.ascontiguousarray(np.asarray(bgr_u8, dtype=np.uint8).reshape(-1, 1, 3))
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1:] -= 128.0
    return lab

def ciede2000(lab1, lab2):
    """CIEDE2000 색차 (Sharma 2005). 입력 (...,3) 브로드캐스트 가능. kL=kC=kH=1."""
    lab1 = np.asarray(lab1, float); lab2 = np.asarray(lab2, float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2); Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.where(C1p == 0, 0.0, np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = np.where(C2p == 0, 0.0, np.degrees(np.arctan2(b2, a2p)) % 360)
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dhp = np.where(C1p * C2p == 0, 0.0, np.where(np.abs(dh) <= 180, dh, np.where(dh > 180, dh - 360, dh + 360)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hsum,
          np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-((hbp - 275) / 25) ** 2)
    RC = 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    return np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))

def circular_mean_deg(deg):
    deg = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(deg).mean(), np.cos(deg).mean())) % 360)

# ---------- 픽셀 선택 ----------
def select_target_pixels(bgr_box, sat_min=60, top_frac=None):
    """박스 crop(BGR uint8)에서 고채도 픽셀 마스크.
    sat_min : OpenCV S(0~255) 하한 (60 ≈ 0.24).  top_frac: 지정 시 채도 상위 비율만 (예 0.5) — 색칠 후 데이터에 권장.
    반환 mask (H,W) bool"""
    hsv = cv2.cvtColor(bgr_box, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    thr = sat_min
    if top_frac is not None and sat.size:
        thr = max(sat_min, float(np.percentile(sat, 100 * (1 - top_frac))))
    return sat >= thr

def patch_stats(frame_bgr, box_xyxy, sat_min=60, top_frac=None, ball_frac=None):
    """원본 프레임 + 원본 좌표 박스 -> 색 통계.
    ball_frac: 지정 시 박스 상단 비율만 사용 (색칠 전 데이터: 공만 색이 있으므로 0.45 권장). 색칠 후에는 None.
    반환 dict(n_px, lab(평균 L,a,b), hue_deg, S, V, rgb, box_px)"""
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return dict(n_px=0, lab=None, hue_deg=None, S=None, V=None, rgb=None, box_px=0)
    if ball_frac:
        y2 = y1 + max(3, int((y2 - y1) * ball_frac))
    crop = frame_bgr[y1:y2, x1:x2]
    mask = select_target_pixels(crop, sat_min=sat_min, top_frac=top_frac)
    n = int(mask.sum())
    if n == 0:
        return dict(n_px=0, lab=None, hue_deg=None, S=None, V=None, rgb=None, box_px=crop.shape[0] * crop.shape[1])
    px = crop[mask]                                   # (n,3) BGR
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    lab = bgr_to_lab(px).mean(0)
    return dict(n_px=n, lab=lab, hue_deg=circular_mean_deg(hsv[:, 0] * 2), S=float(hsv[:, 1].mean() / 255),
                V=float(hsv[:, 2].mean() / 255), rgb=px[:, ::-1].mean(0), box_px=crop.shape[0] * crop.shape[1])

# ---------- 판정 ----------
def classify(stats, refs, min_px=20, margin=5.0):
    """stats: patch_stats 결과. refs: {'enemy': [L,a,b], 'friend': [L,a,b]}.
    반환 dict(side='enemy'|'friend'|'unknown', dE_enemy, dE_friend, reason)
    reason: 'ok' | 'few_px' | 'ambiguous'"""
    if stats['lab'] is None or stats['n_px'] < min_px:
        return dict(side='unknown', dE_enemy=None, dE_friend=None, reason='few_px', n_px=stats['n_px'])
    dE = {k: float(ciede2000(stats['lab'], np.asarray(v, float))) for k, v in refs.items()}
    order = sorted(dE, key=dE.get)
    if dE[order[1]] - dE[order[0]] < margin:
        return dict(side='unknown', dE_enemy=dE.get('enemy'), dE_friend=dE.get('friend'), reason='ambiguous', n_px=stats['n_px'])
    return dict(side=order[0], dE_enemy=dE.get('enemy'), dE_friend=dE.get('friend'), reason='ok', n_px=stats['n_px'])

def separation_report(lab_by_class):
    """lab_by_class: {'enemy': (N,3) Lab 샘플, 'friend': (M,3), ... 'bin_blue': ...}
    반환 dict: 클래스 평균, 클래스 내 흩어짐(각 샘플→자기 평균 dE2000의 RMS/p95), 클래스 간 평균 dE, 안전비(간/내), 추천 margin"""
    out = {'mean': {}, 'spread_rms': {}, 'spread_p95': {}, 'n': {}}
    for k, arr in lab_by_class.items():
        arr = np.asarray(arr, float)
        if len(arr) == 0: continue
        mu = arr.mean(0); d = ciede2000(arr, mu)
        out['mean'][k] = mu; out['spread_rms'][k] = float(np.sqrt((d**2).mean())); out['spread_p95'][k] = float(np.percentile(d, 95)); out['n'][k] = len(arr)
    keys = list(out['mean'])
    out['between'] = {f'{a}|{b}': float(ciede2000(out['mean'][a], out['mean'][b])) for i, a in enumerate(keys) for b in keys[i+1:]}
    if 'enemy' in out['mean'] and 'friend' in out['mean']:
        between = out['between']['enemy|friend']
        spread = max(out['spread_rms']['enemy'], out['spread_rms']['friend'])
        out['safety_ratio'] = between / spread if spread > 0 else float('inf')
        # 판정 불가 margin: 두 클래스 분포가 겹치지 않는 최소 간격의 절반 수준 — 보수적으로 p95 흩어짐을 사용
        out['suggested_margin'] = float(max(out['spread_p95']['enemy'], out['spread_p95']['friend']))
        out['pass'] = out['safety_ratio'] >= 3.0
    return out
