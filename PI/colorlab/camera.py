"""
카메라 제어 모듈.

채널 순서에 대해 (가장 중요):
  picamera2에서 format="RGB888"로 설정하면 실제 메모리 버퍼는 **BGR 순서**다.
  (libcamera/picamera2의 포맷 이름은 little-endian 32비트 워드 기준이라서
  이름과 메모리 순서가 반대가 된다.)

  그래서 이 모듈은 둘을 명확히 구분한다:
    * read_bgr()  -> 버퍼 그대로. cv2.imshow / cv2.imwrite에 바로 넣으면
                     색이 올바르게 나온다.
    * read_rgb()  -> 채널을 뒤집은 것. 측정/로그/표시용 수치는 모두 이것을 쓴다.

  01_test_camera.py 확인 결과가 "Case B"였다면 channel_swap=False로 생성한다.

잠금에 대해:
  lock_from_current()는 항상 먼저 auto 모드로 돌아가 안정될 때까지 기다린 뒤 잠근다
  (그래서 잠긴 상태에서 다시 호출해도 예전 잠금 값이 남지 않는다). 안정화는
  고정 시간이 아니다: 최소 settle_sec만큼 기다린 뒤, 노출/게인/ColourGains가
  window 시간 동안 tol 범위 안에 머물 때까지 기다린다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import CONFIG

DEFAULT_CONFIG_PATH = CONFIG / "camera_config.json"

# libcamera AfMode 값 (controls.AfModeEnum과 동일)
AF_MODE_MANUAL = 0
AF_MODE_CONTINUOUS = 2

# 로드할 때 현재 값과 비교할 config "camera" 블록의 항목
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
    """AWB/AE(및 초점)를 끄고 잠근 카메라 설정."""

    colour_gains: tuple[float, float]  # (red 게인, blue 게인)
    exposure_time: int  # 마이크로초
    analogue_gain: float
    lens_position: float | None = None  # AF 카메라 전용. None이면 초점을 잠그지 않는다
    # 참고용 메타데이터 (잠금에는 쓰지 않지만, 어떤 조명에서 잠갔는지 기록한다)
    lux: float = 0.0
    colour_temperature: int = 0
    locked_at: str = ""
    converged: bool = True  # 안정화가 수렴한 뒤에 잠갔는지 여부
    settle_seconds: float = 0.0  # 잠그기 전에 실제로 기다린 시간
    note: str = ""

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
        )

    def summary_lines(self) -> list[str]:
        lines = [
            f"ColourGains  = (R {self.colour_gains[0]:.4f}, B {self.colour_gains[1]:.4f})",
            f"ExposureTime = {self.exposure_time} us  ({self.exposure_time/1000.0:.2f} ms)",
            f"AnalogueGain = {self.analogue_gain:.3f}",
        ]
        if self.lens_position is not None:
            lines.append(f"LensPosition = {self.lens_position:.3f}  (focus locked)")
        lines += [
            f"(at lock time: Lux={self.lux:.2f}, ColourTemperature={self.colour_temperature}K)",
            f"Locked at: {self.locked_at}  (settled {self.settle_seconds:.1f} s, "
            f"{'converged' if self.converged else 'NOT converged!'})",
        ]
        return lines


class ColorCamera:
    """색상 측정용 Picamera2 래퍼."""

    def __init__(
        self,
        main_size: tuple[int, int] = (640, 480),
        channel_swap: bool = True,
    ):
        self.main_size = main_size
        # True면 버퍼를 BGR로 보고 뒤집어서 RGB로 만든다 (picamera2 기본 동작)
        self.channel_swap = channel_swap
        self.picam2 = None
        self.locked: LockedSettings | None = None
        # 마지막 load_config()에서 발견한 해상도/센서 모드 불일치 경고
        self.load_warnings: list[str] = []

    # ---- 생명주기 ----------------------------------------------------
    def start(self) -> None:
        from picamera2 import Picamera2

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

    # ---- 프레임 ------------------------------------------------------
    def read_bgr(self) -> np.ndarray:
        """cv2에 바로 넘길 수 있는 BGR 배열 (원본 버퍼)."""
        return self.picam2.capture_array("main")[:, :, :3]

    def read_rgb(self) -> np.ndarray:
        """실제 RGB 순서의 배열. 측정/표시되는 모든 수치는 이것을 기준으로 한다."""
        buf = self.read_bgr()
        return buf[:, :, ::-1] if self.channel_swap else buf

    def to_rgb(self, bgr_buf: np.ndarray) -> np.ndarray:
        """이미 읽은 버퍼를 RGB로 변환한다 (두 번 캡처하지 않도록)."""
        return bgr_buf[:, :, ::-1] if self.channel_swap else bgr_buf

    def grab_rgb_frames(self, n: int) -> list[np.ndarray]:
        """n개 프레임을 연속으로 RGB로 가져온다. 노이즈를 평균으로 줄이는 데 쓴다."""
        return [self.read_rgb() for _ in range(max(1, n))]

    def metadata(self) -> dict:
        return self.picam2.capture_metadata()

    # ---- 카메라 정보 -------------------------------------------------
    @property
    def has_af(self) -> bool:
        cc = self.picam2.camera_controls
        return "AfMode" in cc and "LensPosition" in cc

    def model(self) -> str:
        return str(self.picam2.camera_properties.get("Model", "unknown"))

    def sensor_info(self) -> dict:
        """해상도/센서 모드/화각. config 저장, 불일치 확인, scene_capture 메타에 쓴다."""
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

    # ---- AWB / AE / AF 잠금 ------------------------------------------
    def _enable_auto(self) -> None:
        ctrls = {"AwbEnable": True, "AeEnable": True}
        if self.has_af:
            ctrls["AfMode"] = AF_MODE_CONTINUOUS
        self.picam2.set_controls(ctrls)

    def set_auto(self) -> None:
        """auto 모드로 돌아간다. 저장된 잠금 값도 지운다."""
        self._enable_auto()
        self.locked = None

    def lock_from_current(
        self,
        settle_sec: float = 3.0,
        max_wait: float = 10.0,
        tol: float = 0.02,
        window: float = 0.5,
        note: str = "",
    ) -> LockedSettings:
        """
        auto 모드로 돌아가 값이 수렴할 때까지 기다린 뒤, 그 시점의 값으로 잠근다.

        settle_sec : 최소 대기 시간
        max_wait   : 최대 대기 시간 (settle_sec 포함). 이 시간이 지나면 수렴하지 않아도 잠근다
        tol        : 노출/게인/ColourGains 각각에 허용하는 상대 변화량 (0.02 = 2%)
        window     : 값이 이 시간 동안 tol 안에 머물러야 수렴한 것으로 본다
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

        meta = hist[-1][2]
        cg = meta.get("ColourGains", (1.0, 1.0))
        lp = meta.get("LensPosition") if self.has_af else None
        locked = LockedSettings(
            colour_gains=(float(cg[0]), float(cg[1])),
            exposure_time=int(meta.get("ExposureTime", 10000)),
            analogue_gain=float(meta.get("AnalogueGain", 1.0)),
            lens_position=None if lp is None else float(lp),
            lux=float(meta.get("Lux", 0.0)),
            colour_temperature=int(meta.get("ColourTemperature", 0)),
            locked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            converged=converged,
            settle_seconds=round(time.monotonic() - t0, 2),
            note=note,
        )
        self.apply(locked)
        return locked

    def apply(self, s: LockedSettings) -> None:
        """저장된 설정을 적용하고 AWB/AE(및 AF)를 끈다."""
        ctrls = {
            "AwbEnable": False,
            "ColourGains": (s.colour_gains[0], s.colour_gains[1]),
            "AeEnable": False,
            "ExposureTime": int(s.exposure_time),
            "AnalogueGain": float(s.analogue_gain),
        }
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

    # ---- config 파일 ----------------------------------------------------
    def save_config(self, path: Path = DEFAULT_CONFIG_PATH) -> Path:
        if self.locked is None:
            raise RuntimeError("No locked settings. Call lock_from_current() first.")
        data = self.locked.as_dict()
        data["camera"] = self.sensor_info()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_config(self, path: Path = DEFAULT_CONFIG_PATH) -> LockedSettings:
        """
        설정을 불러와 적용한다. 해상도/센서 모드/화각이 저장할 때와 다르면
        self.load_warnings에 경고를 추가한다 (설정은 그래도 적용된다).
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        s = LockedSettings.from_dict(data)
        self.apply(s)
        self.load_warnings = self._compare_camera(data.get("camera"), s)
        return s

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
