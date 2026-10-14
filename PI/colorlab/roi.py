"""03_preview.py와 06_calibrate_lens.py가 함께 쓰는 마우스 ROI 선택 기능."""

from __future__ import annotations

import cv2


class RoiSelector:
    """마우스 드래그로 사각형을 고른다. 좌표는 화면에 표시된 이미지 기준이다."""

    def __init__(self):
        self.dragging = False
        self.start = (0, 0)
        self.cur = (0, 0)
        self.rect: tuple[int, int, int, int] | None = None  # 표시 좌표 기준 (x, y, w, h)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.cur = (x, y)
            x0, x1 = sorted((self.start[0], x))
            y0, y1 = sorted((self.start[1], y))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                self.rect = (x0, y0, x1 - x0, y1 - y0)

    def live_rect(self) -> tuple[int, int, int, int] | None:
        """지금 드래그 중인 사각형을, 없으면 마지막으로 완료된 사각형을 돌려준다."""
        if self.dragging:
            x0, x1 = sorted((self.start[0], self.cur[0]))
            y0, y1 = sorted((self.start[1], self.cur[1]))
            return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        return self.rect


def clamp_roi(roi, w: int, h: int) -> tuple[int, int, int, int]:
    """(x, y, w, h)를 w x h 이미지 안에 들어가게 하고, 최소 2x2 픽셀로 유지한다."""
    x, y, rw, rh = roi
    x = max(0, min(int(x), w - 2))
    y = max(0, min(int(y), h - 2))
    rw = max(2, min(int(rw), w - x))
    rh = max(2, min(int(rh), h - y))
    return (x, y, rw, rh)


def display_to_main(rect, scale: float, w: int, h: int) -> tuple[int, int, int, int]:
    """표시 좌표의 사각형을 main 이미지 좌표로 변환한다."""
    return clamp_roi((rect[0] / scale, rect[1] / scale, rect[2] / scale, rect[3] / scale), w, h)
