#!/usr/bin/env python3
"""
99_fake_camera.py - Run the scripts without a real camera.

It puts a fake 'picamera2' module in place of the real one. The fake camera
sends the same picture again and again (with a little noise), so you can test
the screens, measuring and saving without any camera connected.

Usage:
    python3 99_fake_camera.py 03_preview.py
    python3 99_fake_camera.py 03_preview.py --load
    python3 99_fake_camera.py --image some_photo.png 03_preview.py
    python3 99_fake_camera.py 01_test_camera.py --truth red
    python3 99_fake_camera.py 02_check_colorspace.py --camera
    python3 99_fake_camera.py 04_scene_capture.py --label fake_test

Picture:
    Default is a color chart (3 x 3 color patches, RED in the center).
    Use --image to send your own PNG/JPG instead.

    Chart colors (true R,G,B):
        orange (230,120,30)   green (40,160,60)   blue (40,60,190)
        yellow (220,200,40)   RED   (200,40,40)   cyan (40,180,190)
        white  (235,235,235)  magenta (190,50,160) black (20,20,20)
    Background gray is (120,120,120).
    If you measure a patch, you should get about these values.

Where results go:
    Everything is saved in 99_fake_camera/, not in the real folders.
    (99_fake_camera/03_config, 03_logs, 03_captures, 04_scene_captures, ...)
    So fake results never mix with real data, and the real camera config
    03_config/camera_config.json is never changed.
    The real config is copied into 99_fake_camera/03_config/ once, so --load works.

Not supported: Lecture/06_data_collect.py and Lecture/07_inference.py (they need servos
and AI models). For 05 use --no-servo.
"""

from __future__ import annotations

import shutil
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SANDBOX = HERE / "99_fake_camera"

FPS = 30.0
NOISE_SD = 2.0  # sensor noise, in 0~255 units

CHART = [
    [("orange", (230, 120, 30)), ("green", (40, 160, 60)), ("blue", (40, 60, 190))],
    [("yellow", (220, 200, 40)), ("RED", (200, 40, 40)), ("cyan", (40, 180, 190))],
    [("white", (235, 235, 235)), ("magenta", (190, 50, 160)), ("black", (20, 20, 20))],
]
BG = (120, 120, 120)


def make_chart(w: int, h: int) -> np.ndarray:
    """Color chart as an RGB image. Each patch fills most of its grid cell."""
    rgb = np.full((h, w, 3), BG, dtype=np.uint8)
    cw, ch = w / 3, h / 3
    pad = 0.06
    for r, row in enumerate(CHART):
        for c, (name, color) in enumerate(row):
            x0, y0 = int(c * cw + pad * cw), int(r * ch + pad * ch)
            x1, y1 = int((c + 1) * cw - pad * cw), int((r + 1) * ch - pad * ch)
            rgb[y0:y1, x0:x1] = color
            # small label in the corner, so it does not disturb measuring the middle
            txt = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
            cv2.putText(rgb, name, (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt, 1, cv2.LINE_AA)
    return rgb


# ---------------------------------------------------------------------------
# fake picamera2
# ---------------------------------------------------------------------------
class FakePicamera2:
    """Supports only the parts of Picamera2 that these scripts use."""

    source_rgb: np.ndarray | None = None  # set by main(); None = color chart

    def __init__(self, camera_num: int = 0):
        self.sensor_modes = [
            {"size": (640, 480), "bit_depth": 10, "fps": 58.92},
            {"size": (1296, 972), "bit_depth": 10, "fps": 43.25},
            {"size": (2592, 1944), "bit_depth": 10, "fps": 15.63},
        ]
        self.camera_properties = {"Model": "ov5647"}
        self.camera_controls = {  # ov5647 has no autofocus, so no AfMode/LensPosition
            "AwbEnable": (False, True, True), "AeEnable": (False, True, True),
            "ColourGains": (0.0, 32.0, None), "ExposureTime": (130, 3066985, None),
            "AnalogueGain": (1.0, 63.9, None),
        }
        self._config: dict | None = None
        self._frame_rgb: np.ndarray | None = None
        self._last = 0.0
        self._rng = np.random.default_rng()
        self._controls = {"AwbEnable": True, "AeEnable": True}
        self._meta = {
            "ExposureTime": 29968, "AnalogueGain": 2.25, "DigitalGain": 1.0,
            "ColourGains": (1.1489, 1.9246), "Lux": 538.3, "ColourTemperature": 3095,
            "ScalerCrop": (16, 0, 2560, 1920), "FrameDuration": int(1e6 / FPS),
        }
        # Reference values: at these settings the picture looks exactly like the chart.
        # Changing exposure / gain / colour gains changes the picture like a real camera
        # (brightness ~ exposure * gain, red ~ red gain, blue ~ blue gain).
        self._ref = (self._meta["ExposureTime"] * self._meta["AnalogueGain"],
                     self._meta["ColourGains"][0], self._meta["ColourGains"][1])
        print("[FAKE] Using the fake camera (no real camera).")

    # ---- configuration -------------------------------------------------
    def _make_config(self, main=None, **_):
        main = dict(main or {})
        main.setdefault("size", (640, 480))
        main.setdefault("format", "XBGR8888")
        main["size"] = tuple(main["size"])
        return {"main": main, "sensor": {"output_size": main["size"], "bit_depth": 10}}

    create_preview_configuration = _make_config
    create_video_configuration = _make_config
    create_still_configuration = _make_config

    def configure(self, cfg: dict) -> None:
        self._config = cfg
        w, h = cfg["main"]["size"]
        src = self.source_rgb if self.source_rgb is not None else make_chart(w, h)
        if src.shape[1] != w or src.shape[0] != h:
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
        self._frame_rgb = src

    def camera_configuration(self) -> dict:
        return self._config

    def start(self) -> None:
        if self._config is None:
            self.configure(self._make_config())

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    # ---- frames --------------------------------------------------------
    def _wait_frame(self) -> None:
        """Act like a real camera: at most FPS frames per second."""
        gap = 1.0 / FPS - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()

    def capture_array(self, name: str = "main") -> np.ndarray:
        self._wait_frame()
        noise = self._rng.normal(0, NOISE_SD, self._frame_rgb.shape)
        m = self._meta
        bright = m["ExposureTime"] * m["AnalogueGain"] / self._ref[0]
        gains = np.array([m["ColourGains"][0] / self._ref[1], 1.0, m["ColourGains"][1] / self._ref[2]])
        rgb = np.clip(np.rint(self._frame_rgb * bright * gains + noise), 0, 255).astype(np.uint8)
        # Same memory order as the real picamera2:
        #   RGB888 -> B,G,R    BGR888 -> R,G,B
        #   XBGR8888 -> R,G,B,255    XRGB8888 -> B,G,R,255
        fmt = self._config["main"]["format"]
        if fmt == "RGB888":
            return np.ascontiguousarray(rgb[:, :, ::-1])
        if fmt == "BGR888":
            return rgb
        alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
        if fmt == "XRGB8888":
            return np.concatenate([rgb[:, :, ::-1], alpha], axis=2)
        return np.concatenate([rgb, alpha], axis=2)

    def capture_metadata(self) -> dict:
        self._wait_frame()
        meta = dict(self._meta)
        meta["SensorTimestamp"] = time.monotonic_ns()
        return meta

    def set_controls(self, ctrls: dict) -> None:
        self._controls.update(ctrls)
        for k in ("ColourGains", "ExposureTime", "AnalogueGain"):
            if k in ctrls:
                self._meta[k] = tuple(ctrls[k]) if k == "ColourGains" else ctrls[k]

    @staticmethod
    def global_camera_info() -> list[dict]:
        return [{"Model": "ov5647", "Num": 0}]


# ---------------------------------------------------------------------------
def install_fake_modules() -> None:
    mod = types.ModuleType("picamera2")
    mod.Picamera2 = FakePicamera2
    sys.modules["picamera2"] = mod


def redirect_outputs() -> None:
    """Send colorlab outputs (03_*, 04_*) to the sandbox folder."""
    import colorlab

    colorlab.ROOT = SANDBOX
    colorlab.CAPTURES = SANDBOX / "03_captures"
    colorlab.LOGS = SANDBOX / "03_logs"
    colorlab.CONFIG = SANDBOX / "03_config"
    # import these only after the paths above are changed
    import colorlab.camera  # noqa: F401
    import colorlab.store  # noqa: F401

    real_cfg = HERE / "03_config" / "camera_config.json"
    fake_cfg = colorlab.CONFIG / "camera_config.json"
    if real_cfg.exists() and not fake_cfg.exists():
        fake_cfg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(real_cfg, fake_cfg)
        print(f"[FAKE] Copied the real config to {fake_cfg.relative_to(HERE)}")


def main() -> int:
    args = sys.argv[1:]
    image = None
    if len(args) >= 2 and args[0] == "--image":
        image, args = Path(args[1]), args[2:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    script = (HERE / args[0]) if not Path(args[0]).is_absolute() else Path(args[0])
    if not script.exists():
        print(f"[ERROR] Script not found: {args[0]}")
        return 1

    if image is not None:
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[ERROR] Cannot read the image: {image}")
            return 1
        FakePicamera2.source_rgb = bgr[:, :, ::-1].copy()
        print(f"[FAKE] Picture: {image}")
    else:
        print("[FAKE] Picture: color chart (RED in the center)")

    SANDBOX.mkdir(exist_ok=True)
    install_fake_modules()
    sys.path.insert(0, str(HERE))
    redirect_outputs()
    print(f"[FAKE] Results are saved in {SANDBOX.relative_to(HERE)}/\n")

    # Run the script as if it were in the sandbox folder, so paths made from
    # __file__ (01, 05, 06) also go to the sandbox. Error messages still show
    # the real file and line number.
    sys.argv = [str(script)] + args[1:]
    code = compile(script.read_text(encoding="utf-8"), str(script), "exec")
    g = {"__name__": "__main__", "__file__": str(SANDBOX / script.name), "__builtins__": __builtins__}
    exec(code, g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
