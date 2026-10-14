#!/usr/bin/env python
"""run_match.py — 대회용 상주 실행기. 모델을 미리 올려 두고 키보드 신호를 기다렸다가 판정 루프를 돈다.

순서:
  1. Pipeline 을 한 번만 만든다 (NCNN 모델 로드, 약 10초).
  2. 빈 프레임으로 전체·ROI 추론을 두 번씩 돌려 둔다 (첫 추론이 느린 것을 시작 전에 치른다).
  3. 카메라를 열고 03_config/camera_config.json 을 적용한다 (색 게인 고정).
     exposure_loop.enabled 이면 AE 를 끄고 ExposureTime 을 직접 조정한다 (지금 AE 값에서 출발).
  4. 터미널에서 키를 기다린다:  SPACE/Enter = 시작,  q = 끝내기
     도는 중에는:               SPACE = 일시정지(다시 대기),  q = 끝내기
  5. 루프: 프레임 → pipeline.run → 결정(engage/hold/no_target) → 조준값.
     추적: engage 이면 다음 프레임은 그 타겟 중심의 ROI(384)로, --full-every 프레임마다 전체 프레임으로.
     노출 루프: 흰 조각 L* 를 보고 ExposureTime 을 한 단계씩 조정한다.
  매 프레임 결과는 --log-dir 의 jsonl 로 남긴다.

서보 구동과 발사는 아직 연결하지 않았다 (조준값은 로그와 화면 출력만 한다).
  - yaw_calib 이 없으면 yaw 는 0, pitch_table 이 없으면 pitch 는 0 이다.

사용:
  cd ~/src/pipeline
  venv/bin/python tools/run_match.py                                  # 카메라
  venv/bin/python tools/run_match.py --fake samples/enemy_roomA.jpg --start-now --max-frames 5   # 카메라 없이 시험
"""
import argparse
import contextlib
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from target_pipeline import Pipeline  # noqa: E402


class Keys:
    """터미널 키를 Enter 없이 읽는다. 터미널이 아니면(파이프 등) 키 입력 없이 동작한다."""

    def __init__(self):
        self.tty = sys.stdin.isatty()
        self.old = None

    def __enter__(self):
        if self.tty:
            self.old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)

    def get(self, timeout=0.0):
        """눌린 키 하나(문자) 또는 None."""
        if not self.tty:
            if timeout:
                time.sleep(timeout)
            return None
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None


class FakeCamera:
    """카메라 없이 시험할 때: 사진 한 장을 계속 돌려준다 (cv2.imread 와 같은 진짜 BGR)."""

    def __init__(self, path):
        import cv2
        self.img = cv2.imread(path)
        if self.img is None:
            sys.exit(f"[ERROR] cannot read {path}")
        self.exposure_us = 10000

    def frame(self):
        return self.img.copy()

    def exposure(self):
        return self.exposure_us

    def set_exposure(self, us):
        self.exposure_us = int(us)

    def close(self):
        pass


class RealCamera:
    """colorlab ColorCamera 를 감싼다. frame() 은 항상 진짜 BGR 을 돌려준다."""

    def __init__(self, src_dir, cam_config, size, manual_exposure):
        sys.path.insert(0, str(src_dir))
        from colorlab.camera import ColorCamera
        self.cam = ColorCamera(main_size=size)
        self.cam.start()
        s = self.cam.load_config(Path(cam_config))
        for w in self.cam.load_warnings:
            print(f"[WARNING] {w}")
        print(f"[CAMERA] {cam_config}  colour_gains ({s.colour_gains[0]:.3f}, {s.colour_gains[1]:.3f})  "
              f"{'AE on, EV %+.2f' % s.exposure_value if s.auto_exposure else 'exposure %d us' % s.exposure_time}")
        if manual_exposure and self.cam.locked.auto_exposure:
            time.sleep(1.0)    # AE 가 자리 잡을 시간을 준 뒤, 지금 값으로 노출을 고정한다
            s = self.cam.lock_now(note="run_match: AE off for exposure loop", auto_exposure=False)
            print(f"[CAMERA] exposure loop on -> AE off, start at {s.exposure_time} us, gain {s.analogue_gain:.2f}")

    def frame(self):
        b = self.cam.read_bgr()
        return np.ascontiguousarray(self.cam.to_rgb(b)[:, :, ::-1])

    def exposure(self):
        return self.cam.locked.exposure_time

    def set_exposure(self, us):
        self.cam.adjust(exposure_time=us, note="run_match exposure loop")

    def close(self):
        self.cam.close()


def warm_up(pipe, size):
    """빈 프레임으로 전체·ROI 추론을 미리 돌린다. 반환: 마지막 전체/ROI 추론 시간(ms)."""
    w, h = size
    blank = np.full((h, w, 3), 128, np.uint8)
    t_full = t_roi = 0.0
    for _ in range(2):
        t = time.perf_counter(); pipe.run(blank); t_full = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); pipe.run(blank, (w / 2, h / 2)); t_roi = (time.perf_counter() - t) * 1000
    return t_full, t_roi


def status_of(refs):
    if refs is None:
        return "refs off"
    return "refs " + ("ok" if refs["ok"] else ",".join(f"{k}:{v}" for k, v in refs["status"].items() if v != "ok"))


def main():
    ap = argparse.ArgumentParser(description="Resident match runner: load once, wait for a key, then run")
    ap.add_argument("--config", default=str(HERE / "pipeline_config.json"))
    ap.add_argument("--src", default=str(Path("~/src").expanduser()), help="folder that has colorlab/")
    ap.add_argument("--camera-config", default=str(Path("~/src/03_config/camera_config.json").expanduser()))
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--full-every", type=int, default=6, help="while tracking, run a full frame every N frames")
    ap.add_argument("--log-dir", default=str(HERE.parent / "match_logs"))
    ap.add_argument("--fake", default=None, help="image file instead of the camera (test)")
    ap.add_argument("--start-now", action="store_true", help="do not wait for a key (test)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.lower().split("x"))

    # 1. 모델 로드 (한 번만)
    t0 = time.perf_counter()
    print("[LOAD] pipeline ...", flush=True)
    with contextlib.redirect_stdout(sys.stderr):
        pipe = Pipeline(a.config)
    print(f"[LOAD] done in {time.perf_counter() - t0:.1f} s   color mode={pipe.color.mode}"
          f"  ref_patches={'set' if pipe.color.ref is not None else 'NOT SET'}"
          f"  abs_refs={'yes' if pipe.color.cfg is not None else 'no'}"
          f"  exposure_loop={'on' if pipe.exposure_loop.enabled else 'off'}")

    # 2. 예열
    with contextlib.redirect_stdout(sys.stderr):
        tf, tr = warm_up(pipe, size)
    print(f"[WARMUP] full {tf:.0f} ms, roi {tr:.0f} ms")

    # 3. 카메라
    cam = FakeCamera(a.fake) if a.fake else RealCamera(a.src, a.camera_config, size, pipe.exposure_loop.enabled)

    os.makedirs(a.log_dir, exist_ok=True)
    log_path = Path(a.log_dir) / (time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    log = open(log_path, "a", encoding="utf-8")
    print(f"[LOG] {log_path}")

    n = 0
    try:
        with Keys() as keys:
            running = a.start_now
            while True:
                # 4. 대기
                if not running:
                    print("[READY] Press SPACE or Enter to start, q to quit.", flush=True)
                    while True:
                        k = keys.get(0.2)
                        if k in (" ", "\n", "\r"):
                            break
                        if k == "q":
                            return 0
                    print("[START]", time.strftime("%H:%M:%S"), flush=True)
                    running = True

                # 5. 루프
                roi_center, since_full = None, 0
                while running:
                    k = keys.get(0.0)
                    if k == "q":
                        return 0
                    if k == " ":
                        print("[PAUSE] back to waiting", flush=True)
                        running = False
                        break
                    t = time.perf_counter()
                    frame = cam.frame()
                    use_roi = roi_center if (roi_center is not None and since_full < a.full_every - 1) else None
                    out = pipe.run(frame, use_roi)
                    since_full = since_full + 1 if use_roi is not None else 0
                    dec, aim, refs = out["decision"], out["aim"], out["refs"]
                    tgt = dec["target"]
                    roi_center = (tgt.cx, tgt.cy) if tgt is not None else None
                    # 노출 루프 (꺼져 있으면 None)
                    exp_now = cam.exposure()
                    new_exp = pipe.exposure_loop.propose(refs, exp_now) if refs is not None else None
                    if new_exp is not None and new_exp != exp_now:
                        cam.set_exposure(new_exp)
                    ms = (time.perf_counter() - t) * 1000
                    n += 1
                    sides = [d.side for d in out["dets"]]
                    aim_txt = f" yaw {aim['yaw_deg']:+.1f} pitch {aim['pitch_deg']:+.1f}" if aim else ""
                    print(f"#{n:5d} {'roi ' if use_roi else 'full'} {ms:5.0f} ms  {dec['action']:9s}{aim_txt}  "
                          f"dets {len(sides)} (e{sides.count('enemy')} f{sides.count('friend')} u{sides.count('unknown')})  "
                          f"{status_of(refs)}" + (f"  exp {exp_now}->{new_exp}" if new_exp else ""), flush=True)
                    log.write(json.dumps(dict(
                        t=time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}", n=n, ms=round(ms, 1),
                        roi=None if use_roi is None else [round(v, 1) for v in use_roi],
                        action=dec["action"], aim=aim,
                        target=None if tgt is None else [round(v, 1) for v in tgt.xyxy],
                        dets=[dict(side=d.side, mode=d.mode, reason=d.reason, n_px=d.n_px, box=[round(v, 1) for v in d.xyxy],
                                   d_f=None if d.d_friend_ab is None else round(d.d_friend_ab, 2),
                                   d_e=None if d.d_enemy_ab is None else round(d.d_enemy_ab, 2)) for d in out["dets"]],
                        refs=None if refs is None else dict(ok=refs["ok"], status=refs["status"]),
                        exposure_us=exp_now, exposure_new=new_exp), ensure_ascii=True) + "\n")
                    if a.max_frames and n >= a.max_frames:
                        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        log.close()
        cam.close()
        print(f"[EXIT] {n} frames, log {log_path}")


if __name__ == "__main__":
    sys.exit(main())
