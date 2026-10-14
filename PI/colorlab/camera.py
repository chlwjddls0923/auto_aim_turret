"""
카메라 제어 모듈.

채널 순서에 대해 (가장 중요):
  picamera2에서 format="RGB888"을 쓰면 실제 메모리 버퍼는 **BGR 순서**이다.
  (libcamera/picamera2의 포맷 이름은 little-endian 32비트 워드를 기준으로 하므로
  이름과 메모리 순서가 반대가 된다.)

  그래서 이 모듈은 둘을 구분해 다룬다:
    * read_bgr()  -> 버퍼를 그대로 준다. cv2.imshow / cv2.imwrite에 바로 넘기면
                     색이 올바르게 나온다.
    * read_rgb()  -> 채널을 뒤집어 준다. 측정 / 기록 / 표시하는 모든 숫자는 이것을 쓴다.

  01_test_camera.py가 "Case B"라고 했다면 channel_swap=False로 카메라를 만든다.

잠금에 대해:
  lock_from_current()는 항상 먼저 자동 모드로 돌아가 값이 안정될 때까지 기다린 뒤
  잠근다 (그래서 잠긴 상태에서 다시 불러도 예전 값을 그대로 유지하지 않는다).
  안정 대기 시간은 고정이 아니다: 최소 settle_sec만큼 기다리고, 그다음 노출 /
  gain / ColourGains가 window 동안 계속 tol 안에 머물 때까지 기다린다.

  lock_now()는 카메라가 지금 쓰고 있는 값을 기다리지 않고 바로 잠근다.
  adjust()는 잠긴 값 중 일부만 바꾼다 (03_preview.py의 수동 노출 / gain / 화이트
  밸런스 조정). 둘 다 카메라가 시작된 상태여야 한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np

from . import CONFIG

DEFAULT_CONFIG_PATH = CONFIG / "camera_config.json"

# 모든 스크립트가 쓰는 표준 출력 크기. ov5647에서는 1296x972 sensor mode가
# 선택되며, 이 모드는 예전 640x480 모드와 화각이 같다.
DEFAULT_MAIN_SIZE = (1280, 960)

# libcamera AfMode 값 (controls.AfModeEnum과 같다)
AF_MODE_MANUAL = 0
AF_MODE_CONTINUOUS = 2

# 불러올 때 현재 카메라와 비교하는 config "camera" 블록의 항목
_COMPARE_KEYS = {
    "camera_model": "Camera model",
    "main_size": "Output resolution",
    "sensor_output_size": "Sensor mode size",
    "sensor_bit_depth": "Sensor bit depth",
    "scaler_crop": "ScalerCrop (field of view)",
    "channel_swap": "Channel swap",
}


@dataclass
class LockedSettings:
    """AWB/AE(와 초점)를 끄고 잠근 카메라 설정."""

    colour_gains: tuple[float, float]  # (빨강 gain, 파랑 gain)
    exposure_time: int  # 마이크로초
    analogue_gain: float
    lens_position: float | None = None  # AF 카메라 전용. None = 초점을 잠그지 않는다
    # 참고용 metadata (잠금에는 쓰지 않고, 잠글 때의 조명 상태를 기록한다)
    lux: float = 0.0
    colour_temperature: int = 0
    locked_at: str = ""
    converged: bool = True  # 잠그기 전에 자동 값이 안정되었으면 True
    settle_seconds: float = 0.0  # 잠그기 전에 실제로 기다린 시간
    note: str = ""
    # True면 화이트밸런스(ColourGains)만 고정하고 노출은 카메라가 자동으로 맞춘다 (AE 켜짐).
    # 이때 exposure_time / analogue_gain은 잠글 때의 참고값일 뿐 적용하지 않는다.
    auto_exposure: bool = False
    exposure_value: float = 0.0  # 자동 노출일 때 밝기 보정 (EV, -면 어둡게 / +면 밝게)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["colour_gains"] = list(self.colour_gains)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LockedSettings":
        cg = d.get("colour_gains", [1.0, 1.0])
        lp = d.get("lens_position")
        return cls(
            colour_gains=(float(cg[0]), float(cg[1])),
            exposure_time=int(d.get("exposure_time", 10000)),
            analogue_gain=float(d.get("analogue_gain", 1.0)),
            lens_position=None if lp is None else float(lp),
            lux=float(d.get("lux", 0.0)),
            colour_temperature=int(d.get("colour_temperature", 0)),
            locked_at=str(d.get("locked_at", "")),
            converged=bool(d.get("converged", True)),
            settle_seconds=float(d.get("settle_seconds", 0.0)),
            note=str(d.get("note", "")),
            auto_exposure=bool(d.get("auto_exposure", False)),  # 예전 config에는 없다 -> 노출 고정
            exposure_value=float(d.get("exposure_value", 0.0)),
        )

    def summary_lines(self) -> list[str]:
        lines = [f"ColourGains  = (R {self.colour_gains[0]:.4f}, B {self.colour_gains[1]:.4f})  (white balance fixed)"]
        if self.auto_exposure:
            lines.append(f"Exposure     = AUTO  (EV {self.exposure_value:+.2f})")
        else:
            lines += [
                f"ExposureTime = {self.exposure_time} us  ({self.exposure_time/1000.0:.2f} ms)  (fixed)",
                f"AnalogueGain = {self.analogue_gain:.3f}  (fixed)",
            ]
        if self.lens_position is not None:
            lines.append(f"LensPosition = {self.lens_position:.3f}  (focus locked)")
        lines += [
            f"(at lock time: Lux={self.lux:.2f}, ColourTemperature={self.colour_temperature}K)",
            f"Locked at: {self.locked_at}  (settled {self.settle_seconds:.1f} s, "
            f"{'converged' if self.converged else 'NOT converged!'})",
        ]
        if self.note:
            lines.append(f"Note: {self.note}")
        return lines


def read_config_camera(path: Path) -> dict | None:
    """저장된 config의 "camera" 블록(해상도, sensor mode 등)을 돌려준다. 없으면 None."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("camera")
    except (OSError, ValueError):
        return None


class ColorCamera:
    """색 측정용 Picamera2 래퍼."""

    def __init__(
        self,
        main_size: tuple[int, int] = DEFAULT_MAIN_SIZE,
        channel_swap: bool = True,
    ):
        self.main_size = tuple(main_size)
        # True = 버퍼를 BGR로 보고 뒤집어 RGB를 얻는다 (picamera2 기본값)
        self.channel_swap = channel_swap
        self.picam2 = None
        self.locked: LockedSettings | None = None
        # 마지막 load_config()에서 찾은 해상도 / sensor mode 불일치 경고
        self.load_warnings: list[str] = []
        # 마지막으로 불러온 config의 "camera" 블록 (불러온 적이 없으면 None)
        self.saved_camera: dict | None = None

    # ---- 수명 주기 ---------------------------------------------------
    def start(self) -> None:
        from picamera2 import Picamera2

        # libcamera가 카메라를 하나도 찾지 못하면 Picamera2는 현재 __init__에서
        # 원인을 알기 어려운 IndexError를 낸다. 호출하는 쪽이 쓸모 있는
        # 하드웨어/설정 오류를 받도록 먼저 확인한다.
        cameras = Picamera2.global_camera_info()
        if not cameras:
            raise RuntimeError(
                "No cameras detected by libcamera. Check the camera cable and "
                "connector, then verify detection with `rpicam-hello --list-cameras` "
                "(or `libcamera-hello --list-cameras` on older Raspberry Pi OS)."
            )
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": self.main_size, "format": "RGB888"}
        )
        self.picam2.configure(cfg)
        self.picam2.start()

    def close(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            finally:
                self.picam2.close()
            self.picam2 = None

    def __enter__(self) -> "ColorCamera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 프레임 -------------------------------------------------------
    def read_bgr(self) -> np.ndarray:
        """cv2에 바로 넘길 수 있는 BGR 배열(원본 버퍼)을 돌려준다."""
        return self.picam2.capture_array("main")[:, :, :3]

    def read_rgb(self) -> np.ndarray:
        """실제 RGB 순서의 배열을 돌려준다. 측정 / 표시하는 모든 숫자는 이것을 기준으로 한다."""
        buf = self.read_bgr()
        return buf[:, :, ::-1] if self.channel_swap else buf

    def to_rgb(self, bgr_buf: np.ndarray) -> np.ndarray:
        """이미 읽은 버퍼를 RGB로 바꾼다 (두 번 캡처하지 않도록 한다)."""
        return bgr_buf[:, :, ::-1] if self.channel_swap else bgr_buf

    def grab_rgb_frames(self, n: int) -> list[np.ndarray]:
        """n개의 프레임을 연속으로 RGB로 가져온다. 노이즈를 평균으로 줄이는 데 쓴다."""
        return [self.read_rgb() for _ in range(max(1, n))]

    def metadata(self) -> dict:
        return self.picam2.capture_metadata()

    # ---- 카메라 정보 --------------------------------------------------
    @property
    def has_af(self) -> bool:
        cc = self.picam2.camera_controls
        return "AfMode" in cc and "LensPosition" in cc

    def model(self) -> str:
        return str(self.picam2.camera_properties.get("Model", "unknown"))

    def sensor_info(self) -> dict:
        """해상도 / sensor mode / 화각을 돌려준다. config에 저장되며, 불일치 검사와 scene_capture meta에 쓴다."""
        cfg = self.picam2.camera_configuration()
        sensor = cfg.get("sensor") or {}
        out_size = sensor.get("output_size")
        meta = self.metadata()
        crop = meta.get("ScalerCrop")
        return {
            "camera_model": self.model(),
            "main_size": list(cfg["main"]["size"]),
            "main_format": cfg["main"]["format"],
            "sensor_output_size": list(out_size) if out_size else None,
            "sensor_bit_depth": sensor.get("bit_depth"),
            "scaler_crop": list(crop) if crop else None,
            "channel_swap": self.channel_swap,
            "has_af": self.has_af,
        }

    def control_limits(self, name: str, default: tuple[float, float]) -> tuple[float, float]:
        """카메라가 알려 주는 libcamera 컨트롤의 (min, max)를 돌려준다. 없으면 `default`."""
        try:
            lo, hi = self.picam2.camera_controls[name][:2]
            lo = float(lo if not isinstance(lo, (tuple, list)) else lo[0])
            hi = float(hi if not isinstance(hi, (tuple, list)) else hi[0])
            if hi > lo:
                return lo, hi
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        return default

    # ---- AWB / AE / AF 잠금 -------------------------------------------
    def _enable_auto(self) -> None:
        ctrls = {"AwbEnable": True, "AeEnable": True}
        if "ExposureValue" in self.picam2.camera_controls:
            ctrls["ExposureValue"] = 0.0  # 자동 노출 밝기 보정도 기본값으로 되돌린다
        if self.has_af:
            ctrls["AfMode"] = AF_MODE_CONTINUOUS
        self.picam2.set_controls(ctrls)

    def set_auto(self) -> None:
        """자동 모드로 돌아간다. 잠긴 값도 지운다."""
        self._enable_auto()
        self.locked = None

    def _settings_from_meta(self, meta: dict, converged: bool, settle: float, note: str,
                            auto_exposure: bool = False) -> LockedSettings:
        cg = meta.get("ColourGains", (1.0, 1.0))
        lp = meta.get("LensPosition") if self.has_af else None
        return LockedSettings(
            colour_gains=(float(cg[0]), float(cg[1])),
            exposure_time=int(meta.get("ExposureTime", 10000)),
            analogue_gain=float(meta.get("AnalogueGain", 1.0)),
            lens_position=None if lp is None else float(lp),
            lux=float(meta.get("Lux", 0.0)),
            colour_temperature=int(meta.get("ColourTemperature", 0)),
            locked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            converged=converged,
            settle_seconds=round(settle, 2),
            note=note,
            auto_exposure=auto_exposure,
        )

    def lock_from_current(
        self,
        settle_sec: float = 3.0,
        max_wait: float = 10.0,
        tol: float = 0.02,
        window: float = 0.5,
        note: str = "",
        auto_exposure: bool = False,
    ) -> LockedSettings:
        """
        자동 모드로 돌아가 값이 수렴할 때까지 기다린 뒤, 그 시점의 값을 잠근다.

        settle_sec : 최소 대기 시간
        max_wait   : 최대 대기 시간 (settle_sec 포함). 이 시간이 지나면 수렴하지 않아도 잠근다
        tol        : 노출 / gain / ColourGains 각각에 허용하는 상대 변화량 (0.02 = 2%)
        window     : 값이 이 시간 동안 tol 안에 머물러야 수렴한 것으로 본다
        auto_exposure : True면 화이트밸런스만 잠그고 노출은 자동으로 둔다
        """
        self._enable_auto()
        self.locked = None
        t0 = time.monotonic()
        time.sleep(settle_sec)

        hist: list[tuple[float, tuple[float, ...], dict]] = []
        converged = False
        while True:
            meta = self.metadata()
            now = time.monotonic()
            cg = meta.get("ColourGains", (1.0, 1.0))
            vals = (
                float(meta.get("ExposureTime", 0)),
                float(meta.get("AnalogueGain", 0)),
                float(cg[0]),
                float(cg[1]),
            )
            hist.append((now, vals, meta))
            recent = [v for (t, v, _) in hist if now - t <= window]
            if now - hist[0][0] >= window and len(recent) >= 3:
                last = recent[-1]
                if all(
                    abs(v[i] - last[i]) <= tol * max(abs(last[i]), 1e-9)
                    for v in recent
                    for i in range(4)
                ):
                    converged = True
                    break
            if now - t0 >= max(max_wait, settle_sec):
                break

        locked = self._settings_from_meta(hist[-1][2], converged, time.monotonic() - t0, note,
                                          auto_exposure=auto_exposure)
        self.apply(locked)
        return locked

    def lock_now(self, note: str = "locked as is", auto_exposure: bool = False) -> LockedSettings:
        """카메라가 지금(자동 모드에서) 쓰고 있는 값을 기다리지 않고 잠근다."""
        locked = self._settings_from_meta(self.metadata(), converged=False, settle=0.0, note=note,
                                          auto_exposure=auto_exposure)
        self.apply(locked)
        return locked

    def adjust(
        self,
        exposure_time: float | None = None,
        analogue_gain: float | None = None,
        colour_gains: tuple[float, float] | None = None,
        exposure_value: float | None = None,
        note: str = "manual adjust",
    ) -> LockedSettings:
        """
        잠긴 값 중 일부만 바꿔서 적용한다. 카메라가 잠긴 상태여야 한다.
        값은 카메라가 알려 주는 범위로 제한한다. 노출은 현재 프레임 시간보다
        짧게 유지해서 프레임 속도가 떨어지지 않게 한다.
        """
        if self.locked is None:
            raise RuntimeError("Camera is not locked. Lock it first (f in 03_preview.py).")
        s = self.locked
        if exposure_time is not None:
            lo, hi = self.control_limits("ExposureTime", (100.0, 66000.0))
            frame_us = self.metadata().get("FrameDuration")
            if frame_us:
                hi = min(hi, float(frame_us))
            s = replace(s, exposure_time=int(round(min(max(exposure_time, lo), hi))))
        if analogue_gain is not None:
            lo, hi = self.control_limits("AnalogueGain", (1.0, 16.0))
            s = replace(s, analogue_gain=float(min(max(analogue_gain, lo), hi)))
        if colour_gains is not None:
            lo, hi = self.control_limits("ColourGains", (0.0, 32.0))
            lo = max(lo, 0.1)
            r, b = (float(min(max(g, lo), hi)) for g in colour_gains)
            s = replace(s, colour_gains=(r, b))
        if exposure_value is not None:
            lo, hi = self.control_limits("ExposureValue", (-8.0, 8.0))
            s = replace(s, exposure_value=float(min(max(exposure_value, lo), hi)))
        s = replace(s, locked_at=time.strftime("%Y-%m-%d %H:%M:%S"), note=note)
        self.apply(s)
        return s

    def apply(self, s: LockedSettings) -> None:
        """
        저장된 설정을 적용한다. AWB는 항상 끄고 ColourGains를 고정한다.
        auto_exposure가 False면 AE도 끄고 노출 / gain을 고정하고,
        True면 AE를 켜 두고 (카메라가 지원하면) ExposureValue로 밝기만 보정한다.
        """
        ctrls = {
            "AwbEnable": False,
            "ColourGains": (s.colour_gains[0], s.colour_gains[1]),
        }
        if s.auto_exposure:
            ctrls["AeEnable"] = True
            if "ExposureValue" in self.picam2.camera_controls:
                ctrls["ExposureValue"] = float(s.exposure_value)
        else:
            ctrls["AeEnable"] = False
            ctrls["ExposureTime"] = int(s.exposure_time)
            ctrls["AnalogueGain"] = float(s.analogue_gain)
        if self.has_af and s.lens_position is not None:
            ctrls["AfMode"] = AF_MODE_MANUAL
            ctrls["LensPosition"] = float(s.lens_position)
        self.picam2.set_controls(ctrls)
        # 컨트롤이 실제 프레임에 반영될 때까지 몇 프레임을 버린다.
        for _ in range(5):
            self.picam2.capture_metadata()
        self.locked = s

    @property
    def is_locked(self) -> bool:
        return self.locked is not None

    # ---- config 파일 -----------------------------------------------------
    def save_config(self, path: Path = DEFAULT_CONFIG_PATH) -> Path:
        """잠긴 값과 함께 해상도, sensor mode, ScalerCrop을 저장한다."""
        if self.locked is None:
            raise RuntimeError("No locked settings. Call lock_from_current() first.")
        data = self.locked.as_dict()
        data["camera"] = self.sensor_info()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    def load_config(self, path: Path = DEFAULT_CONFIG_PATH) -> LockedSettings:
        """
        설정을 불러와 적용한다. 해상도 / sensor mode / 화각이 저장할 때와 다르면
        self.load_warnings에 경고를 추가한다 (설정은 그래도 적용한다).
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        s = LockedSettings.from_dict(data)
        self.apply(s)
        self.saved_camera = data.get("camera")
        self.load_warnings = self._compare_camera(self.saved_camera, s)
        return s

    def resolution_hint(self) -> str | None:
        """불러온 config가 다른 해상도에서 만들어졌다면, 그에 맞춰 실행하는 방법을 알려 준다."""
        saved = self.saved_camera or {}
        size = saved.get("main_size")
        if size and list(size) != list(self.main_size):
            return (f"The config was saved at {size[0]}x{size[1]}, but the camera now runs at "
                    f"{self.main_size[0]}x{self.main_size[1]}. Run again with that size "
                    f"(03_preview.py: --main-width {size[0]} --main-height {size[1]}, "
                    f"other scripts: --width {size[0]} --height {size[1]}), "
                    "or lock again at the new size (f in 03_preview.py).")
        return None

    def _compare_camera(self, saved: dict | None, s: LockedSettings) -> list[str]:
        if not saved:
            return ["Config has no resolution/sensor mode info (old config version). Press f to lock and save again."]
        now = self.sensor_info()
        warns = []
        for key, label in _COMPARE_KEYS.items():
            if saved.get(key) != now.get(key):
                warns.append(f"{label}: saved={saved.get(key)}  now={now.get(key)}")
        if s.lens_position is not None and not now["has_af"]:
            warns.append("Config has a focus value (LensPosition), but this camera has no AF, so it is ignored.")
        if s.lens_position is None and now["has_af"]:
            warns.append("This camera has AF, but config has no focus value, so focus is not locked.")
        return warns
