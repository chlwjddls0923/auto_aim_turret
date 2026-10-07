#!/usr/bin/env python
"""target_pipeline.py — 검출 → 색 판정 → 조준값 계산 → 교전 결정 (인계 브리프 §5·§6 A-5). 파이4에서도 그대로 사용.

규약:
  * Detector.detect() 는 입력이 전체 프레임이든 ROI 크롭이든 **원본 프레임(1280×960) 좌표** 박스를 반환한다.
    ultralytics 는 넣은 이미지 좌표로 박스를 돌려주므로 ROI 는 크롭 오프셋만 더한다(배율 없음 — ROI 는 원본 해상도 크롭).
  * 색·yaw·pitch 는 원본 프레임 + 원본 좌표만 사용한다.
  * 밴드(y 352~672) 밖 박스는 버리지 않고 conf × band_penalty 로 감점만 한다.
  * 색 기준(color_refs.json)이 없거나 안전비 미달이면 전부 unknown → 결정은 hold. 피치 LUT 가 없으면 pitch 0 + 경고.
  * decide(): unknown 이 하나라도 있으면 hold. 적군이 여러 개면 박스 높이 h 가 가장 큰(가장 가까운) 것.

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
    reason: str = 'unclassified'    # 'ok' | 'few_px' | 'ambiguous' | 'no_refs' | 'unclassified'
    n_px: int = 0
    dE_enemy: float = None
    dE_friend: float = None
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
    def __init__(self, refs_path=None, min_safety_ratio=3.0):
        """refs_path: color_separation.py 가 만든 color_refs.json. 없거나 안전비가 기준 미달이면 전부 unknown 을 반환하고 계속한다."""
        self.cfg = None
        if not refs_path or not os.path.isfile(refs_path):
            warnings.warn(f'color_refs 없음({refs_path}) — 모든 타겟을 unknown 으로 판정합니다'); return
        cfg = json.load(open(refs_path))
        if set(cfg.get('refs', {})) != {'enemy', 'friend'}:
            warnings.warn(f'{refs_path}: refs 에 enemy/friend 가 모두 있어야 함 — 모든 타겟을 unknown 으로 판정합니다'); return
        if cfg.get('safety_ratio', 0) < min_safety_ratio:
            warnings.warn(f"{refs_path}: 안전비 {cfg.get('safety_ratio', 0):.2f} < {min_safety_ratio} — 색 분리 미통과 기준색은 쓰지 않습니다(전부 unknown)"); return
        self.cfg = cfg

    def classify(self, frame, det):
        """det.side/reason/n_px/dE_* 를 채우고 det 를 반환."""
        if self.cfg is None:
            det.side, det.reason = 'unknown', 'no_refs'; return det
        c = self.cfg
        st = C.patch_stats(frame, det.xyxy, sat_min=c.get('sat_min', 60), top_frac=c.get('top_frac'), ball_frac=c.get('ball_frac'))
        r = C.classify(st, c['refs'], min_px=c.get('min_px', 20), margin=c.get('margin', 5.0))
        det.side, det.reason, det.n_px, det.dE_enemy, det.dE_friend = r['side'], r['reason'], r['n_px'], r['dE_enemy'], r['dE_friend']
        return det

class Aimer:
    def __init__(self, yaw_calib=None, pitch_table=None, frame_w=1280):
        """yaw_calib: {'type': 'poly', 'coef': [c0, c1, ...], 'sign': ±1}  yaw = sign · Σ c_i · xn^i,  xn = (cx − W/2)/(W/2)
                      {'type': 'lut', 'cx': [...], 'deg': [...]}           cx(px) → 각도 선형 보간
        pitch_table: {'h_px': [...], 'pitch_deg': [...]}                  박스 높이 h(px) → 피치 각도 선형 보간"""
        self.yaw, self.W = yaw_calib, frame_w; self.pitch = None
        if pitch_table and len(pitch_table.get('h_px', [])) >= 2:
            o = np.argsort(pitch_table['h_px']); self.pitch = (np.asarray(pitch_table['h_px'], float)[o], np.asarray(pitch_table['pitch_deg'], float)[o])
        else: warnings.warn('pitch_table 없음 — pitch 0 을 반환합니다(피치 보정표 측정 후 pipeline_config.json 에 입력)')
        if not yaw_calib: warnings.warn('yaw_calib 없음 — yaw 0 을 반환합니다')

    def aim(self, det):
        """반환 dict(yaw_deg, pitch_deg, pitch_valid): yaw 는 화면 중심 기준 보정각, pitch 는 LUT 값(없으면 0)."""
        yaw = 0.0
        if self.yaw:
            if self.yaw.get('type') == 'lut': yaw = float(np.interp(det.cx, self.yaw['cx'], self.yaw['deg']))
            else:
                xn = (det.cx - self.W / 2) / (self.W / 2)
                yaw = float(self.yaw.get('sign', 1) * sum(c * xn ** i for i, c in enumerate(self.yaw['coef'])))
        pitch = float(np.interp(det.h, *self.pitch)) if self.pitch is not None else 0.0
        return dict(yaw_deg=yaw, pitch_deg=pitch, pitch_valid=self.pitch is not None)

def decide(dets):
    """반환 dict(action, target, reason). action: 'engage'(target=선택된 적군) | 'hold' | 'no_target'."""
    if not dets: return dict(action='no_target', target=None, reason='no_detection')
    unk = [d for d in dets if d.side == 'unknown']
    if unk: return dict(action='hold', target=None, reason=f"unknown {len(unk)}개 ({', '.join(sorted({d.reason for d in unk}))})")
    enemies = [d for d in dets if d.side == 'enemy']
    if not enemies: return dict(action='no_target', target=None, reason='friend_only')
    return dict(action='engage', target=max(enemies, key=lambda d: d.h), reason=f'enemy {len(enemies)}개 중 h 최대')

class Pipeline:
    def __init__(self, config_path):
        cfg = json.load(open(config_path)); base = os.path.dirname(os.path.abspath(config_path))
        def rel(p):   # 상대 경로는 config 파일 위치 기준. ~ 와 $HOME 같은 환경변수도 풀어 준다
            if p is None: return None
            p = os.path.expandvars(os.path.expanduser(p)); return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
        d = {k: v for k, v in cfg['detector'].items() if not k.startswith('_')}
        d['model'] = rel(d['model']); d['roi_model'] = rel(d.get('roi_model'))
        self.detector = Detector(**d)
        self.color = ColorClassifier(rel(cfg.get('color', {}).get('refs')), cfg.get('color', {}).get('min_safety_ratio', 3.0))
        aim = cfg.get('aim', {}); self.aimer = Aimer(aim.get('yaw_calib'), aim.get('pitch_table'), aim.get('frame_w', 1280))

    def run(self, frame, roi_center=None):
        dets = [self.color.classify(frame, d) for d in self.detector.detect(frame, roi_center)]
        dec = decide(dets)
        return dict(dets=dets, decision=dec, aim=self.aimer.aim(dec['target']) if dec['target'] is not None else None)

def draw(frame, out):
    col = {'enemy': (196, 79, 126), 'friend': (161, 163, 20), 'unknown': (0, 200, 255)}; im = frame.copy()
    for d in out['dets']:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)); cv2.rectangle(im, p1, p2, col[d.side], 3 if d is out['decision']['target'] else 1)
        cv2.putText(im, f'{d.side} {d.conf:.2f} h{d.h:.0f}', (p1[0], max(12, p1[1] - 5)), 0, 0.5, col[d.side], 1)
    a = out['aim']; txt = out['decision']['action'] + (f" yaw {a['yaw_deg']:+.1f} pitch {a['pitch_deg']:+.1f}{'' if a['pitch_valid'] else '(LUT 없음)'}" if a else '') + f" | {out['decision']['reason']}"
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
    print(json.dumps(dict(image=a.image, roi_center=a.roi_center, dets=[rnd(d) for d in out['dets']],
                          decision=dict(action=out['decision']['action'], reason=out['decision']['reason'], target=rnd(t) if t else None), aim=out['aim']), ensure_ascii=False, indent=1))
    if a.save is not None:
        p = a.save or os.path.splitext(os.path.basename(a.image))[0] + '_pipeline.jpg'; cv2.imwrite(p, draw(frame, out)); print('saved', p, file=sys.stderr)

if __name__ == '__main__': main()
