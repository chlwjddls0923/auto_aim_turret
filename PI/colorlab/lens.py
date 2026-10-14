"""
06_calibrate_lens.py에서 쓰는 pinhole 렌즈 계산.

    focal_px     = pixel_height * distance / object_height
    pixel_height = focal_px * object_height / distance

focal_px는 출력 해상도에 따라 달라지므로(1280x960에서는 640x480 값의 두 배),
항상 측정했을 때의 해상도와 함께 사용한다.

이 공식은 이미지 중심 근처에서만 정확하다. 광각 렌즈는 가장자리를 휘게 하므로
(barrel distortion), 가장자리 근처의 물체는 예측보다 작게 보인다.
"""

from __future__ import annotations

import math

SIGN_SIZES_M = (0.15, 0.20, 0.25, 0.30, 0.35)
DISTANCES_M = (1.2, 1.5, 1.75, 2.0, 2.3)
LOW_PX = 40.0  # 이보다 작으면 경고한다 (검출 / 색 판별을 믿기에는 너무 작다)
OK_PX = 60.0  # 이 값 이상이면 OK


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not value > 0:
        raise ValueError(f"{name} must be greater than 0 (got {value})")
    return value


def focal_from_measure(pixel_height: float, distance_m: float, object_height_m: float) -> float:
    """측정 한 번으로 픽셀 단위 focal length를 구한다."""
    return (_positive("pixel_height", pixel_height) * _positive("distance_m", distance_m)
            / _positive("object_height_m", object_height_m))


def pixel_height(focal_px: float, object_height_m: float, distance_m: float) -> float:
    """이 크기의 물체가 이 거리에서 몇 픽셀 높이로 보이는지 구한다."""
    return (_positive("focal_px", focal_px) * _positive("object_height_m", object_height_m)
            / _positive("distance_m", distance_m))


def median(values) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("no values")
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def px_per_degree(focal_px: float) -> float:
    """이미지 중심 근처에서 1도당 픽셀 수 (focal_px * pi / 180)."""
    return _positive("focal_px", focal_px) * math.pi / 180.0


def fov_deg(size_px: float, focal_px: float) -> float:
    """`size_px` 픽셀에 걸친 화각을 도(deg) 단위로 구한다."""
    return math.degrees(2.0 * math.atan(_positive("size_px", size_px) / (2.0 * _positive("focal_px", focal_px))))


def mark(px: float) -> str:
    """40 px 미만이면 LOW, 60 px 이상이면 OK, 그 사이면 '--'를 돌려준다."""
    if px < LOW_PX:
        return "LOW"
    if px >= OK_PX:
        return "OK"
    return "--"


def size_table(focal_px: float, sizes_m=SIGN_SIZES_M, distances_m=DISTANCES_M) -> list[list[float]]:
    """행 = 표지판 크기, 열 = 거리, 칸 = 픽셀 높이."""
    return [[pixel_height(focal_px, s, d) for d in distances_m] for s in sizes_m]


def format_size_table(focal_px: float, sizes_m=SIGN_SIZES_M, distances_m=DISTANCES_M) -> list[str]:
    """터미널에 출력할 텍스트 표를 만든다."""
    head = "sign size |" + "".join(f"{d:>11.2f} m" for d in distances_m)
    lines = [head, "-" * len(head)]
    for s, row in zip(sizes_m, size_table(focal_px, sizes_m, distances_m)):
        cells = "".join(f"{px:>8.0f}px {mark(px):>3}" for px in row)
        lines.append(f"  {s * 100:4.0f} cm |{cells}")
    lines.append(f"(LOW = under {LOW_PX:.0f} px, OK = {OK_PX:.0f} px or more, "
                 "values are for the image center)")
    return lines
