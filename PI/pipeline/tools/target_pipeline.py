#!/usr/bin/env python
"""target_pipeline.py — 검출 → 색 판정 → 조준값 계산 → 교전 결정 (인계 브리프 §5·§6 A-5). 파이4에서도 그대로 사용.

규약:
  * Detector.detect() 는 입력이 전체 프레임이든 ROI 크롭이든 **원본 프레임(1280×960) 좌표** 박스를 반환한다.
    ultralytics 는 넣은 이미지 좌표로 박스를 돌려주므로 ROI 는 크롭 오프셋만 더한다(배율 없음 — ROI 는 원본 해상도 크롭).
  * 색·yaw·pitch 는 원본 프레임 + 원본 좌표만 사용한다.
  * 밴드(y 352~672) 밖 박스는 버리지 않고 conf × band_penalty 로 감점만 한다.
  * 색 판정 (10/4 재설계): color.mode='relative' 이고 ref_patches 좌표가 있으면 1층 = 프레임 안 기준 조각과의 상대 판정
    (a*b* 거리, 흰 조각으로 프레임 내 WB). 기준 조각이 불량(포화·어두움·픽셀 부족)인 프레임은 color_refs.json 절대값으로 폴백하고,
    그것도 없거나 안전비 미달이면 unknown. 피치 LUT 가 없으면 pitch 0 + 경고.
    color_refs.json 에 metric='ab' 가 있으면(build_color_refs.py 산출물) 절대값 판정도 a*b* 거리(L* 제외)로 한다. 없으면 dE2000(기존).
    color.mode='a_sign' 은 폴백: 기준값 없이 a* > +a_sign_thr → enemy, a* < −a_sign_thr → friend, 사이 → unknown
    (적군 마젠타 / 아군 초록 전용. color_refs.json 을 쓰지 않으므로 안전비 게이트도 거치지 않는다).
  * 노출 소프트 루프(ExposureLoop, 기본 off): 흰 조각 L* 가 목표값이 되도록 ExposureTime 을 천천히 바꾼 값을 제안만 한다.
    카메라에 적용하는 것은 실행기(run_match.py 등)의 몫이다.
  * decide(): unknown 이 하나라도 있으면 hold. 적군이 여러 개면 박스 높이 h 가 가장 큰(가장 가까운) 것.
  * 조준(Aimer): 조준점 = aim.aim_table(07_aim_calib.py 가 실사격으로 만든 calib/aim_table.json, h → (cx, cy)) 이 있으면 그 값,
    없으면 화면 중앙. yaw/pitch 는 "타겟을 조준점까지 옮기는 서보 보정각". 카메라가 총구와 어긋나 있어도 표가 흡수한다.

사용:
  python tools/target_pipeline.py --config tools/pipeline_config.json --image frame.jpg [--roi-center 640,480] [--save [out.jpg]]
"""
import argparse, contextlib, json, os, sys, warnings
from dataclasses import dataclass, asdict
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import colorlib as C

@dataclass
class Det:
    x1: float; y1: float; x2: float; y2: float
    conf: float                     # 밴드 감점 적용 후
    raw_conf: float                 # 모델 출력 그대로
    side: str = 'unknown'           # 'enemy' | 'friend' | 'unknown'
    reason: str = 'unclassified'    # 'ok' | 'few_px' | 'ambiguous' | 'no_refs' | 'ref_<조각>_<상태>' | 'unclassified'
    n_px: int = 0
    dE_enemy: float = None          # 절대값 모드: 기준색과의 dE2000
    dE_friend: float = None
    mode: str = None                # 'relative' | 'absolute' | None(판정 안 함)
    d_enemy_ab: float = None        # 상대 모드: 이 프레임 적군 조각과의 a*b* 거리
    d_friend_ab: float = None
    a_star: float = None            # 박스 고채도 픽셀 평균 a* (색을 잰 경우)
    @property
    def xyxy(self): return (self.x1, self.y1, self.x2, self.y2)
    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2
    @property
    def w(self): return self.x2 - self.x1
    @property
    def h(self): return self.y2 - self.y1

class Detector:
    def __init__(self, model, roi_model=None, imgsz_full=(960, 1280), roi=384, conf=0.25, band=(352, 672), band_penalty=0.5, device='cpu'):
        """model: 전체 프레임용 (.pt 또는 NCNN 폴더). roi_model: ROI 용 — None 이면 model 을 같이 쓴다
        (.pt 는 입력 크기가 자유, NCNN 은 export 크기 고정이라 384×384 export 를 따로 지정)."""
        from ultralytics import YOLO
        self.model = YOLO(model, task='detect'); self.roi_model = YOLO(roi_model, task='detect') if roi_model else self.model
        self.imgsz_full, self.roi, self.conf, self.band, self.band_penalty, self.device = tuple(imgsz_full), int(roi), conf, band, band_penalty, device

    def detect(self, frame, roi_center=None):
        """frame: 원본 BGR. roi_center=(x, y) 를 주면 그 주변 roi×roi 원본 해상도 크롭에서 검출. 반환: 원본 좌표 Det 목록(conf 내림차순)."""
        H, W = frame.shape[:2]; ox = oy = 0; src, m, sz = frame, self.model, self.imgsz_full
        if roi_center is not None:
            r = self.roi; ox = int(np.clip(round(roi_center[0] - r / 2), 0, max(0, W - r))); oy = int(np.clip(round(roi_center[1] - r / 2), 0, max(0, H - r)))
            src, m, sz = frame[oy:oy + r, ox:ox + r], self.roi_model, r
        res = m.predict(src, imgsz=sz, conf=self.conf, device=self.device, verbose=False)[0]
        dets = []
        if res.boxes is not None:
            for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy().astype(float), res.boxes.conf.cpu().numpy().astype(float)):
                d = Det(x1 + ox, y1 + oy, x2 + ox, y2 + oy, conf=c, raw_conf=c)
                if self.band and not (self.band[0] <= d.cy <= self.band[1]): d.conf = c * self.band_penalty
                if d.conf >= self.conf: dets.append(d)
        return sorted(dets, key=lambda d: -d.conf)

class ColorClassifier:
    def __init__(self, refs_path=None, min_safety_ratio=3.0, mode='absolute', ref_patches=None, margin_ab=5.0,
                 min_px=20, sat_min=60, ref_sat_thr=250, ref_sat_frac=0.01, ref_v_min=0.15, wb_in_frame=True, a_sign_thr=10.0):
        """mode='relative': ref_patches(friend/enemy/white 좌표)로 프레임마다 기준을 읽어 상대 판정, 불량 프레임은 절대값 폴백.
        mode='absolute'  : 기존 방식 (color_refs.json 만).
        mode='a_sign'    : 폴백. a* 부호만 (a_sign_thr). 기준 조각·color_refs.json 을 읽지 않는다.
        refs_path: color_separation.py 가 만든 color_refs.json (절대값/폴백용). 없거나 안전비 미달이면 폴백 불가."""
        self.cfg = None; self.ref = None; self.mode = mode
        self.margin_ab, self.min_px, self.sat_min, self.a_sign_thr = margin_ab, min_px, sat_min, a_sign_thr
        if mode == 'a_sign':
            warnings.warn(f'color.mode=a_sign — a* 부호 폴백 판정(±{a_sign_thr}). color_refs.json·기준 조각을 쓰지 않습니다')
            return
        if mode == 'relative':
            try:
                self.ref = C.RefPatches(ref_patches, sat_thr=ref_sat_thr, sat_frac_max=ref_sat_frac, v_min=ref_v_min,
                                        sat_min=sat_min, min_px=min_px, wb_in_frame=wb_in_frame)
            except ValueError as e:
                warnings.warn(f'상대 판정 불가 — {e} (pipeline_config.json 의 color.ref_patches). 절대값만 사용합니다')
        if not refs_path or not os.path.isfile(refs_path):
            if self.ref is None:
                warnings.warn(f'color_refs 없음({refs_path}) — 모든 타겟을 unknown 으로 판정합니다')
            return
        cfg = json.load(open(refs_path))
        if set(cfg.get('refs', {})) != {'enemy', 'friend'}:
            warnings.warn(f'{refs_path}: refs 에 enemy/friend 가 모두 있어야 함 — 절대값 판정 안 함'); return
        if cfg.get('safety_ratio', 0) < min_safety_ratio:
            warnings.warn(f"{refs_path}: 안전비 {cfg.get('safety_ratio', 0):.2f} < {min_safety_ratio} — 색 분리 미통과 기준색은 쓰지 않습니다"); return
        if cfg.get('jpeg_input'):
            warnings.warn(f'{refs_path}: jpeg_input=true (시운전 산출물) — 절대값 판정에 쓰지 않습니다'); return
        self.cfg = cfg

    def read_refs(self, frame):
        """프레임마다 한 번 부른다. 상대 모드가 아니면 None."""
        return self.ref.read(frame) if self.ref is not None else None

    def classify(self, frame, det, reading=None):
        """det.side/reason/n_px/거리 값을 채우고 det 를 반환. reading = read_refs(frame) 결과."""
        if self.mode == 'a_sign':
            st = C.patch_stats(frame, det.xyxy, sat_min=self.sat_min)
            det.mode, det.n_px = 'a_sign', st['n_px']
            if st['lab'] is None or st['n_px'] < self.min_px:
                det.side, det.reason = 'unknown', 'few_px'; return det
            r = C.classify_a_sign(st['lab'], thr=self.a_sign_thr)
            det.side, det.reason, det.a_star = r['side'], r['reason'], r['a_star']
            return det
        if reading is not None and reading['ok']:
            st = C.patch_stats(frame, det.xyxy, sat_min=self.sat_min, gains=reading['gains'])
            det.mode, det.n_px = 'relative', st['n_px']
            if st['lab'] is None or st['n_px'] < self.min_px:
                det.side, det.reason = 'unknown', 'few_px'; return det
            det.a_star = float(st['lab'][1])
            r = C.classify_relative(st['lab'], reading['lab']['friend'], reading['lab']['enemy'], margin=self.margin_ab)
            det.side, det.reason, det.d_friend_ab, det.d_enemy_ab = r['side'], r['reason'], r['d_friend'], r['d_enemy']
            return det
        if self.cfg is None:   # 상대 판정 불가 + 절대값 폴백도 없음
            det.side = 'unknown'
            det.reason = (C.ref_problem(reading) or 'no_refs') if reading is not None else 'no_refs'
            return det
        c = self.cfg
        st = C.patch_stats(frame, det.xyxy, sat_min=c.get('sat_min', 60), top_frac=c.get('top_frac'), ball_frac=c.get('ball_frac'))
        if st['lab'] is not None:
            det.a_star = float(st['lab'][1])
        if c.get('metric') == 'ab':   # build_color_refs.py 산출물: a*b* 거리, margin 도 a*b* 단위
            det.mode, det.n_px = 'absolute', st['n_px']
            if st['lab'] is None or st['n_px'] < c.get('min_px', 20):
                det.side, det.reason = 'unknown', 'few_px'; return det
            r = C.classify_relative(st['lab'], c['refs']['friend'], c['refs']['enemy'], margin=c.get('margin', 5.0))
            det.side, det.reason, det.d_friend_ab, det.d_enemy_ab = r['side'], r['reason'], r['d_friend'], r['d_enemy']
            return det
        r = C.classify(st, c['refs'], min_px=c.get('min_px', 20), margin=c.get('margin', 5.0))
        det.mode = 'absolute'
        det.side, det.reason, det.n_px, det.dE_enemy, det.dE_friend = r['side'], r['reason'], r['n_px'], r['dE_enemy'], r['dE_friend']
        return det


class ExposureLoop:
    """노출 소프트 루프 (선택, 기본 off). 흰 기준 조각의 L* 가 target_L 이 되도록 ExposureTime 을 천천히 바꾼 값을 제안한다.
    켜면 실행기는 카메라의 자동 노출을 끄고(색 게인은 그대로) 제안값을 ExposureTime 으로 적용한다.
    벽 포화는 허용한다 (검출기는 hsv_v=0.4 로 학습되어 노출 변동을 버틴다)."""

    def __init__(self, enabled=False, target_L=75.0, deadband=3.0, max_step=0.15, min_us=200, max_us=33000):
        self.enabled, self.target_L, self.deadband = enabled, target_L, deadband
        self.max_step, self.min_us, self.max_us = max_step, min_us, max_us

    def propose(self, reading, exposure_us):
        """reading: ColorClassifier.read_refs() 결과. 반환: 새 ExposureTime(us) 또는 None(그대로)."""
        if not self.enabled or reading is None:
            return None
        st = reading['status'].get('white')
        if st == 'saturated':
            ratio = 1.0 - self.max_step                     # 흰 조각이 포화 → 무조건 조금 줄인다
        elif st == 'dark':
            ratio = 1.0 + self.max_step
        elif reading['lab'].get('white') is None:
            return None
        else:
            L = float(reading['lab']['white'][0])
            if abs(L - self.target_L) <= self.deadband:
                return None
            # L* 는 대략 노출의 1/3 제곱에 비례 → 노출 비율 ≈ (목표/현재)^3, 한 번에 max_step 까지만
            ratio = float(np.clip((self.target_L / max(L, 1.0)) ** 3, 1 - self.max_step, 1 + self.max_step))
        new = int(np.clip(round(exposure_us * ratio), self.min_us, self.max_us))
        return new if new != int(exposure_us) else None


class Aimer:
    def __init__(self, yaw_calib=None, pitch_table=None, frame_w=1280, aim_table=None, pitch_calib=None, frame_h=960):
        """yaw_calib:   {'type': 'poly', 'coef': [c0, c1, ...], 'sign': ±1}  yaw = sign · Σ c_i · xn^i,  xn = (cx − aim_cx)/(W/2)
                      {'type': 'lut', 'px': [...], 'deg': [...]}            cx(px) → 각도 선형 보간, f(cx) − f(aim_cx) (옛 키 'cx' 도 됨)
        pitch_calib: yaw_calib 과 같은 형식, yn = (cy − aim_cy)/(H/2). 있으면 pitch_table 보다 우선.
        pitch_table: {'h_px': [...], 'pitch_deg': [...]}                  박스 높이 h(px) → 피치 각도 선형 보간 (pitch_calib 이 없을 때)
        aim_table:   07_aim_calib.py 가 만든 calib/aim_table.json 의 경로 또는 그 dict {'h_px': [...], 'cx': [...], 'cy': [...]}.
                     h(px) → 조준점 (cx, cy) 선형 보간(범위 밖은 끝값) = "박스 높이가 h 이면 타겟을 화면의 (cx, cy) 에 두면 맞는다".
                     없으면 조준점 = 화면 중앙 (W/2, H/2). 카메라 위치가 바뀌거나 받침·막대를 색칠하면 표를 다시 만든다."""
        self.yaw, self.W, self.H = yaw_calib, frame_w, frame_h; self.pitch = None; self.pitch_calib = pitch_calib; self.table = None
        if isinstance(aim_table, (str, os.PathLike)):
            if os.path.isfile(aim_table):
                with open(aim_table, encoding='utf-8') as f: aim_table = json.load(f)
            else: warnings.warn(f'aim_table 없음({aim_table}) — 조준점 = 화면 중앙. 07_aim_calib.py 로 측정하면 생긴다'); aim_table = None
        if aim_table:
            hp, cx, cy = (np.asarray(aim_table.get(k, []), float) for k in ('h_px', 'cx', 'cy'))
            if len(hp) >= 1 and len(hp) == len(cx) == len(cy):
                o = np.argsort(hp); self.table = (hp[o], cx[o], cy[o])
            else: warnings.warn('aim_table 형식 오류(h_px/cx/cy 길이가 다르거나 비어 있음) — 무시, 조준점 = 화면 중앙')
        if pitch_table and len(pitch_table.get('h_px', [])) >= 2:
            o = np.argsort(pitch_table['h_px']); self.pitch = (np.asarray(pitch_table['h_px'], float)[o], np.asarray(pitch_table['pitch_deg'], float)[o])
        elif not pitch_calib: warnings.warn('pitch_calib/pitch_table 없음 — pitch 0 을 반환합니다(피치 보정 측정 후 pipeline_config.json 에 입력)')
        if not yaw_calib: warnings.warn('yaw_calib 없음 — yaw 0 을 반환합니다')

    def aim_point(self, h):
        """h(px) → (aim_cx, aim_cy, from_table). 표가 없으면 화면 중앙."""
        if self.table is None: return self.W / 2, self.H / 2, False
        hp, cx, cy = self.table
        return float(np.interp(h, hp, cx)), float(np.interp(h, hp, cy)), True

    @staticmethod
    def _angle(calib, px, aim_px, half):
        """calib(poly | lut) 로 '타겟 px 를 조준점 aim_px 로 옮기는 각도'를 계산한다."""
        if calib.get('type') == 'lut':
            xs = calib.get('px', calib.get('cx', calib.get('cy'))); f = lambda v: float(np.interp(v, xs, calib['deg']))
            return float(calib.get('sign', 1) * (f(px) - f(aim_px)))
        n = (px - aim_px) / half
        return float(calib.get('sign', 1) * sum(c * n ** i for i, c in enumerate(calib['coef'])))

    def aim(self, det):
        """반환 dict(yaw_deg, pitch_deg, pitch_valid, aim_cx, aim_cy, dx_px, dy_px, aim_table):
        yaw/pitch 는 타겟을 조준점으로 옮기는 보정각(해당 보정이 없으면 0), dx/dy = 타겟 중심 − 조준점(px), aim_table = 표에서 온 조준점인지."""
        acx, acy, from_table = self.aim_point(det.h)
        yaw = self._angle(self.yaw, det.cx, acx, self.W / 2) if self.yaw else 0.0
        if self.pitch_calib: pitch, valid = self._angle(self.pitch_calib, det.cy, acy, self.H / 2), True
        elif self.pitch is not None: pitch, valid = float(np.interp(det.h, *self.pitch)), True
        else: pitch, valid = 0.0, False
        return dict(yaw_deg=yaw, pitch_deg=pitch, pitch_valid=valid, aim_cx=acx, aim_cy=acy,
                    dx_px=float(det.cx - acx), dy_px=float(det.cy - acy), aim_table=from_table)

def decide(dets):
    """반환 dict(action, target, reason). action: 'engage'(target=선택된 적군) | 'hold' | 'no_target'."""
    if not dets: return dict(action='no_target', target=None, reason='no_detection')
    unk = [d for d in dets if d.side == 'unknown']
    if unk: return dict(action='hold', target=None, reason=f"unknown {len(unk)}개 ({', '.join(sorted({d.reason for d in unk}))})")
    enemies = [d for d in dets if d.side == 'enemy']
    if not enemies: return dict(action='no_target', target=None, reason='friend_only')
    return dict(action='engage', target=max(enemies, key=lambda d: d.h), reason=f'enemy {len(enemies)}개 중 h 최대')

def read_config(config_path):
    """pipeline_config.json 을 읽어 (cfg, rel) 을 돌려준다. rel(p): 상대 경로를 config 파일 위치 기준 절대 경로로 (~ 와 $HOME 같은 환경변수도 풂)."""
    cfg = json.load(open(config_path)); base = os.path.dirname(os.path.abspath(config_path))
    def rel(p):
        if p is None: return None
        p = os.path.expandvars(os.path.expanduser(p)); return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
    return cfg, rel

def make_detector(cfg, rel):
    d = {k: v for k, v in cfg['detector'].items() if not k.startswith('_')}
    d['model'] = rel(d['model']); d['roi_model'] = rel(d.get('roi_model'))
    return Detector(**d)

def load_detector(config_path):
    """config 의 detector 블록만으로 Detector 를 만든다 — 색 판정·조준 없이 검출만 쓰는 도구용(07_aim_calib.py)."""
    return make_detector(*read_config(config_path))

class Pipeline:
    def __init__(self, config_path):
        cfg, rel = read_config(config_path)
        self.detector = make_detector(cfg, rel)
        col = {k: v for k, v in cfg.get('color', {}).items() if not k.startswith('_')}
        refs = rel(col.pop('refs', None))
        self.color = ColorClassifier(refs, **col)
        exp = {k: v for k, v in cfg.get('exposure_loop', {}).items() if not k.startswith('_')}
        self.exposure_loop = ExposureLoop(**exp)
        aim = cfg.get('aim', {})
        self.aimer = Aimer(aim.get('yaw_calib'), aim.get('pitch_table'), aim.get('frame_w', 1280),
                           aim_table=rel(aim.get('aim_table')), pitch_calib=aim.get('pitch_calib'), frame_h=aim.get('frame_h', 960))

    def run(self, frame, roi_center=None):
        reading = self.color.read_refs(frame)            # 기준 조각은 프레임당 한 번만 읽는다
        dets = [self.color.classify(frame, d, reading) for d in self.detector.detect(frame, roi_center)]
        dec = decide(dets)
        return dict(dets=dets, decision=dec, aim=self.aimer.aim(dec['target']) if dec['target'] is not None else None,
                    refs=reading)

def draw(frame, out):
    col = {'enemy': (196, 79, 126), 'friend': (161, 163, 20), 'unknown': (0, 200, 255)}; im = frame.copy()
    for d in out['dets']:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)); cv2.rectangle(im, p1, p2, col[d.side], 3 if d is out['decision']['target'] else 1)
        cv2.putText(im, f'{d.side} {d.conf:.2f} h{d.h:.0f}', (p1[0], max(12, p1[1] - 5)), 0, 0.5, col[d.side], 1)
    a = out['aim']; txt = out['decision']['action'] + (f" yaw {a['yaw_deg']:+.1f} pitch {a['pitch_deg']:+.1f}{'' if a['pitch_valid'] else '(LUT 없음)'} dx {a['dx_px']:+.0f} dy {a['dy_px']:+.0f}px" if a else '') + f" | {out['decision']['reason']}"
    if a:   # 조준점: 표에서 왔으면 노란 십자, 화면 중앙(표 없음)이면 회색
        cv2.drawMarker(im, (int(round(a['aim_cx'])), int(round(a['aim_cy']))), (0, 255, 255) if a['aim_table'] else (160, 160, 160), cv2.MARKER_CROSS, 28, 1)
    cv2.putText(im, txt.encode('ascii', 'replace').decode(), (10, 24), 0, 0.7, (0, 0, 255), 2)
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--image', required=True)
    ap.add_argument('--roi-center', default=None, help='x,y (원본 좌표). 주면 ROI 384×384 추적 모드')
    ap.add_argument('--save', nargs='?', const='', default=None, help='결과 그림 저장 (경로 생략 시 <image stem>_pipeline.jpg)')
    a = ap.parse_args()
    frame = cv2.imread(a.image)
    if frame is None: sys.exit(f'이미지를 읽을 수 없음: {a.image}')
    with contextlib.redirect_stdout(sys.stderr):   # ultralytics 의 로딩 로그는 stderr 로 → stdout 에는 JSON 만 남는다
        out = Pipeline(a.config).run(frame, tuple(float(v) for v in a.roi_center.split(',')) if a.roi_center else None)
    rnd = lambda d: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(d).items()}
    t = out['decision']['target']
    r = out['refs']
    refs = None if r is None else dict(ok=r['ok'], status=r['status'], n_px=r['n_px'],
                                       gains=None if r['gains'] is None else [round(g, 3) for g in r['gains']],
                                       lab={k: None if v is None else [round(float(x), 1) for x in v] for k, v in r['lab'].items()})
    print(json.dumps(dict(image=a.image, roi_center=a.roi_center, dets=[rnd(d) for d in out['dets']],
                          decision=dict(action=out['decision']['action'], reason=out['decision']['reason'], target=rnd(t) if t else None),
                          aim=out['aim'], refs=refs), ensure_ascii=False, indent=1))
    if a.save is not None:
        p = a.save or os.path.splitext(os.path.basename(a.image))[0] + '_pipeline.jpg'; cv2.imwrite(p, draw(frame, out)); print('saved', p, file=sys.stderr)

if __name__ == '__main__': main()
