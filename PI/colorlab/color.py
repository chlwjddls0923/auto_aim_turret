"""
색 공간 변환과 측정.

중요 - 값 범위:
  OpenCV는 입력 dtype에 따라 출력 범위가 다르다.
    * uint8 입력: H=0~179, S=0~255, V=0~255 / L=0~255, a=0~255(+128 오프셋), b=0~255
    * float32 입력 (0~1): H=0~360, S=0~1, V=0~1 / L=0~100, a=-127~127, b=-127~127
  이 모듈은 항상 float32 경로를 쓴다. 그래야 사람이 읽는 표준 범위를 얻고,
  uint8 반올림으로 인한 정밀도 손실(H가 2도 단위)도 없다.

측정 규칙:
  ROI의 평균 색은 "RGB로 평균한 다음 변환"한다.
  Hue는 원형 값이라서 hue 값을 그냥 평균하면 틀린다.
  (예: 350도와 10도의 평균은 180도가 아니라 0도여야 한다)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

# BT.601 luma 가중치. 흑백 카메라 / grayscale 변환의 표준이다.
_Y_COEF = np.array([0.299, 0.587, 0.114], dtype=np.float64)

# 측정 품질 검사용 기준값 (0~255 RGB)
CLIP_THR = 250  # 한 채널이라도 이 값 이상이면 클리핑(포화)된 픽셀이다
DARK_THR = 5  # 모든 채널이 이 값 이하면 어두운 픽셀이다
QUALITY_WARN_RATIO = 0.01  # ROI 안의 클리핑/어두운 픽셀 비율이 이 값을 넘으면 경고한다


def clip_dark_masks(rgb: np.ndarray, clip_thr: int = CLIP_THR, dark_thr: int = DARK_THR):
    """uint8 RGB(...,3) -> (클리핑 마스크, 어두운 픽셀 마스크). 둘 다 (...,) bool이다."""
    return (rgb >= clip_thr).any(axis=-1), (rgb <= dark_thr).all(axis=-1)


def _as_float_patch(rgb: np.ndarray) -> np.ndarray:
    """(...,3) RGB(0~255)를 cvtColor용 float32 1x1 또는 HxWx3 패치로 만든다."""
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, 3)
    return np.ascontiguousarray(arr)


def rgb_to_hsv360(rgb: np.ndarray) -> np.ndarray:
    """RGB(0~255) -> HSV. H=0~360 (deg), S=0~100 (%), V=0~100 (%)."""
    hsv = cv2.cvtColor(_as_float_patch(rgb), cv2.COLOR_RGB2HSV)
    out = hsv.reshape(-1, 3).astype(np.float64)
    # 빨간색 근처에서 OpenCV는 359.99 같은 값을 돌려준다. 정확히 360.0이 나오면
    # 0.0과 같은 색이지만 [0,360) 범위를 벗어나 히스토그램 bin 계산이 깨지므로
    # 다시 0 쪽으로 감싼다.
    out[:, 0] = np.mod(out[:, 0], 360.0)
    out[:, 1] *= 100.0
    out[:, 2] *= 100.0
    return out.reshape(np.shape(rgb))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB(0~255, sRGB) -> CIELAB. L*=0~100, a*/b* = 약 -128~127."""
    lab = cv2.cvtColor(_as_float_patch(rgb), cv2.COLOR_RGB2Lab)
    return lab.reshape(np.shape(rgb)).astype(np.float64)


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    """RGB(0~255) -> 밝기 Y (BT.601, 0~255)."""
    arr = np.asarray(rgb, dtype=np.float64)
    return arr @ _Y_COEF


def hue_name(h_deg: float, sat_pct: float, val_pct: float) -> str:
    """hue 각도를 사람이 읽을 수 있는 영어 색 이름으로 바꾼다.
    색조가 없는 색(흰색/회색/검은색)은 채도와 명도로 판단한다."""
    if val_pct < 8:
        return "black"
    if sat_pct < 12:
        return "white" if val_pct > 75 else ("gray" if val_pct > 25 else "black")
    h = h_deg % 360.0
    table = [
        (15, "red"), (45, "orange"), (70, "yellow"), (160, "green"),
        (200, "cyan"), (255, "blue"), (290, "purple"), (330, "pink"), (360, "red"),
    ]
    for edge, name in table:
        if h < edge:
            return name
    return "red"


@dataclass
class ColorSample:
    """ROI 1회 측정 결과."""

    # RGB (0~255) - 실제 R,G,B 순서
    r: float
    g: float
    b: float
    # 공간 표준편차: ROI 안의 픽셀이 얼마나 고른지 (표면 질감/얼룩/그림자)
    r_std: float
    g_std: float
    b_std: float
    # 시간 표준편차: 프레임마다 값이 얼마나 변하는지 (센서 노이즈/조명 깜빡임)
    r_tstd: float
    g_tstd: float
    b_tstd: float

    h: float  # 0~360 도
    s: float  # 0~100 %
    v: float  # 0~100 %
    L: float  # 0~100
    a: float  # -128~127
    bb: float  # -128~127
    y: float  # 0~255 밝기

    n_frames: int = 0
    n_pixels: int = 0
    color_name: str = ""
    hex: str = ""
    # 측정 품질: N개 프레임 전체의 원본 픽셀 기준 클리핑/어두운 픽셀 비율 (0~1)
    clip_ratio: float = 0.0
    dark_ratio: float = 0.0

    def quality_warnings(self, locked: bool) -> list[str]:
        """측정 품질 경고. 문제가 없으면 빈 리스트다."""
        w = []
        if self.clip_ratio > QUALITY_WARN_RATIO:
            w.append(f"Clipped pixels {self.clip_ratio*100:.1f}% (>= {CLIP_THR}) - lower the exposure or move the ROI")
        if self.dark_ratio > QUALITY_WARN_RATIO:
            w.append(f"Dark pixels {self.dark_ratio*100:.1f}% (<= {DARK_THR}) - the ROI includes areas that are too dark")
        if not locked:
            w.append("Measured in auto mode (AWB/AE on) - conditions may differ from real use (press f to lock)")
        return w

    def as_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self, en: bool = False) -> list[str]:
        if en:  # 화면 오버레이용 짧은 버전
            return [
                f"RGB  {self.r:6.1f},{self.g:6.1f},{self.b:6.1f}  {self.hex}",
                f" sd  {self.r_std:6.1f},{self.g_std:6.1f},{self.b_std:6.1f} (spatial)",
                f" tsd {self.r_tstd:6.2f},{self.g_tstd:6.2f},{self.b_tstd:6.2f} (frames)",
                f"HSV  H={self.h:5.1f}  S={self.s:5.1f}%  V={self.v:5.1f}%",
                f"LAB  L={self.L:6.2f} a={self.a:7.2f} b={self.bb:7.2f}",
                f"Y    {self.y:6.1f} / 255",
            ]
        return [
            f"RGB    {self.r:6.1f}, {self.g:6.1f}, {self.b:6.1f}   {self.hex}  ({self.color_name})",
            f"  spatial sd  {self.r_std:6.1f}, {self.g_std:6.1f}, {self.b_std:6.1f}  (evenness inside ROI)",
            f"  temporal sd {self.r_tstd:6.2f}, {self.g_tstd:6.2f}, {self.b_tstd:6.2f}  (change between frames)",
            f"HSV    H={self.h:6.1f}deg  S={self.s:5.1f}%  V={self.v:5.1f}%",
            f"LAB    L*={self.L:6.2f}  a*={self.a:7.2f}  b*={self.bb:7.2f}",
            f"Luma   Y={self.y:6.1f} / 255",
        ]


def quick_mean_rgb(frame_rgb: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[float, float, float]:
    """실시간 표시용: ROI 평균 RGB만 빠르게 구한다. (표준편차 / 색 공간 변환 없음)"""
    x, y, w, h = roi
    m = cv2.mean(np.ascontiguousarray(frame_rgb[y : y + h, x : x + w, :3]))
    return m[0], m[1], m[2]


def measure_roi(frames: list[np.ndarray], roi: tuple[int, int, int, int]) -> ColorSample:
    """
    여러 프레임에서 같은 ROI를 측정해 ColorSample 하나를 만든다.

    frames : RGB 순서(실제 R,G,B)의 uint8 배열 리스트. 모두 같은 크기여야 한다.
    roi    : (x, y, w, h) - frames 좌표 기준
    """
    if not frames:
        raise ValueError("frames is empty")

    x, y, w, h = roi
    # read_rgb()는 [:, :, ::-1] view(음수 stride)라서 cv2가 받지 못한다 -> ROI만 연속 메모리로 복사한다
    crops = [np.ascontiguousarray(f[y : y + h, x : x + w, :3]) for f in frames]
    crops = [c for c in crops if c.size > 0]
    if not crops:
        raise ValueError("ROI is outside the frame or its size is 0")

    # 프레임별 ROI 평균 -> (N, 3).  cv2.mean이 numpy float64로 변환하는 것보다 몇 배 빠르다.
    per_frame_mean = np.array([cv2.mean(c)[:3] for c in crops])

    # 최종 색: 전체 프레임의 평균
    mean_rgb = per_frame_mean.mean(axis=0)

    # 시간 표준편차: 프레임마다 평균이 얼마나 변했는지
    t_std = per_frame_mean.std(axis=0) if len(crops) > 1 else np.zeros(3)

    # 공간 표준편차: 프레임 평균 이미지에서 픽셀이 얼마나 퍼져 있는지
    if len(crops) == 1:
        avg_img = crops[0]
    else:
        avg_img = np.zeros(crops[0].shape, np.float32)
        for c in crops:
            cv2.accumulate(c, avg_img)
        avg_img /= len(crops)
    s_std = cv2.meanStdDev(avg_img)[1].reshape(3)

    # 품질: 평균을 내면 순간적인 클리핑이 가려지므로 N개 프레임의 원본 픽셀 전체로 센다
    n_total = sum(c.shape[0] * c.shape[1] for c in crops)
    n_clip = n_dark = 0
    for c in crops:
        cm, dm = clip_dark_masks(c)
        n_clip += int(cm.sum())
        n_dark += int(dm.sum())

    hsv = rgb_to_hsv360(mean_rgb).reshape(3)
    lab = rgb_to_lab(mean_rgb).reshape(3)
    yy = float(rgb_to_luma(mean_rgb))

    ri, gi, bi = (int(round(float(np.clip(c, 0, 255)))) for c in mean_rgb)

    return ColorSample(
        r=float(mean_rgb[0]), g=float(mean_rgb[1]), b=float(mean_rgb[2]),
        r_std=float(s_std[0]), g_std=float(s_std[1]), b_std=float(s_std[2]),
        r_tstd=float(t_std[0]), g_tstd=float(t_std[1]), b_tstd=float(t_std[2]),
        h=float(hsv[0]), s=float(hsv[1]), v=float(hsv[2]),
        L=float(lab[0]), a=float(lab[1]), bb=float(lab[2]),
        y=yy,
        n_frames=len(crops),
        n_pixels=int(crops[0].shape[0] * crops[0].shape[1]),
        color_name=hue_name(float(hsv[0]), float(hsv[1]), float(hsv[2])),
        hex=f"#{ri:02X}{gi:02X}{bi:02X}",
        clip_ratio=n_clip / n_total,
        dark_ratio=n_dark / n_total,
    )
