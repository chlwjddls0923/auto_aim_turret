"""
화면 오버레이 텍스트 그리기.

cv2.putText는 ASCII 텍스트만 그릴 수 있다 (다른 문자는 빈 네모나 깨진 글자로 나온다).
시스템에서 한글도 지원하는 TTF 폰트를 찾으면 PIL로 텍스트를 그린다
(그래서 사용자가 입력한 한글 이름도 보인다). 없으면 cv2.putText로 대체하고
ASCII가 아닌 문자는 '?'로 바꾼다.

속도 참고:
  PIL을 쓰면 매 프레임 numpy<->PIL 변환에 시간이 든다. 그래서 텍스트를 항목마다
  따로 그리지 않는다. begin()/text()...../end()로 모아서 프레임당
  한 번만 변환한다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

# 우선순위: 나눔고딕 -> Noto -> 은글꼴(UnFonts)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
]


def find_korean_font() -> str | None:
    """시스템에서 한글을 지원하는 TTF 폰트 경로를 찾는다. 없으면 None."""
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    # fc-list로 한 번 더 찾아본다
    try:
        out = subprocess.run(
            ["fc-list", ":lang=ko", "file"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            path = line.split(":")[0].strip()
            if path and Path(path).exists():
                return path
    except Exception:
        pass
    return None


class Overlay:
    """여러 줄의 텍스트를 모아서 한 프레임에 그린다."""

    def __init__(self, font_path: str | None = None, default_size: int = 15):
        self.font_path = font_path if font_path is not None else find_korean_font()
        self.default_size = default_size
        self._fonts: dict[int, object] = {}
        self._pending: list[tuple[int, int, str, tuple[int, int, int], int]] = []
        self._img: np.ndarray | None = None

        self.korean_ok = False
        if self.font_path:
            try:
                from PIL import ImageFont  # noqa: F401

                self._get_font(default_size)
                self.korean_ok = True
            except Exception:
                self.korean_ok = False

    # ---- 내부 -------------------------------------------------------
    def _get_font(self, size: int):
        from PIL import ImageFont

        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    # ---- 공개 API ---------------------------------------------------
    def begin(self, img_bgr: np.ndarray) -> None:
        """
        img_bgr를 복사하지 않고 참조만 보관한다. 그래서 begin과 end 사이에 numpy/cv2로
        그린 패널, 사각형, 색 견본도 end()에 그대로 나타나고, 텍스트는
        항상 그 위에 그려진다.
        (주의: 이것을 복사하도록 바꾸면 그 도형들이 사라진다)
        """
        self._img = img_bgr
        self._pending.clear()

    def text(
        self,
        x: int,
        y: int,
        s: str,
        color: tuple[int, int, int] = (255, 255, 255),
        size: int | None = None,
    ) -> None:
        """(x, y)는 텍스트의 왼쪽 위 좌표다. color는 BGR이다."""
        self._pending.append((x, y, s, color, size or self.default_size))

    def line_height(self, size: int | None = None) -> int:
        return int((size or self.default_size) * 1.45)

    def end(self) -> np.ndarray:
        """모아 둔 텍스트를 실제로 그리고 이미지를 반환한다."""
        img = self._img
        if img is None:
            raise RuntimeError("Call begin() first")
        if not self._pending:
            return img

        if self.korean_ok:
            from PIL import Image, ImageDraw

            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for x, y, s, color, size in self._pending:
                # color는 BGR로 들어오므로 PIL용으로 RGB로 뒤집는다
                draw.text((x, y), s, font=self._get_font(size), fill=(color[2], color[1], color[0]))
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            for x, y, s, color, size in self._pending:
                scale = size / 30.0
                # cv2는 baseline 기준이므로 size만큼 내려서 왼쪽 위 위치에 맞춘다
                cv2.putText(
                    img, _ascii_only(s), (x, y + size),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA,
                )

        self._pending.clear()
        self._img = img
        return img


def _ascii_only(s: str) -> str:
    """한글 폰트가 없을 때 글자 깨짐을 막기 위해 쓴다: ASCII가 아닌 문자를 '?'로 바꾼다."""
    return "".join(ch if ord(ch) < 128 else "?" for ch in s)


def panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    """텍스트가 잘 보이도록 반투명 검은 패널을 제자리(in place)에 그린다."""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = cv2.convertScaleAbs(roi, alpha=1 - alpha)
