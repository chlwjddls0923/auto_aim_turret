"""
출력 폴더에 다른 카메라 설정으로 찍은 데이터가 이미 들어 있지 않은지 확인한다.

해상도나 회전이 다른 사진을 한 폴더에 섞으면 안 되므로, 수집 스크립트는 멈추고
(스스로 아무것도 옮기거나 이름을 바꾸지 않는다) 사용자에게 예전 폴더의 이름을 바꾸라고 알린다.
"""

from __future__ import annotations

import json
from pathlib import Path


def _size_text(size) -> str:
    return f"{size[0]}x{size[1]}" if size else "unknown size"


def dataset_mismatches(root: Path, size: tuple[int, int], rotate: int) -> list[str]:
    """
    <root>/<label>/sessions.jsonl (05_sweep_capture.py / 05_manual_capture.py)을 살펴본다.
    다른 해상도나 회전으로 찍은 세션이 있는 label 폴더마다 한 줄씩 돌려준다.
    """
    found = []
    if not root.is_dir():
        return found
    for sessions in sorted(root.glob("*/sessions.jsonl")):
        seen = set()
        for line in sessions.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("event") != "start":
                continue
            res = tuple(rec.get("resolution") or ())
            rot = rec.get("rotate")
            if res != tuple(size) or rot != rotate:
                seen.add((res, rot))
        for res, rot in sorted(seen, key=str):
            found.append(f"{sessions.parent.name}/: {_size_text(res)}, rotate {rot}")
    return found


def scene_mismatches(root: Path, size: tuple[int, int]) -> list[str]:
    """
    <root>/<run>/meta.json (04_scene_capture.py)을 살펴본다. "_"로 시작하는 폴더
    (self test)는 건너뛴다. 다른 해상도로 찍은 run마다 한 줄씩 돌려준다.
    """
    found = []
    if not root.is_dir():
        return found
    for meta_path in sorted(root.glob("*/meta.json")):
        if meta_path.parent.name.startswith("_"):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        saved = tuple((meta.get("camera") or {}).get("main_size") or ())
        if saved != tuple(size):
            found.append(f"{meta_path.parent.name}/: {_size_text(saved)}")
    return found


def print_old_data_stop(root: Path, found: list[str], size: tuple[int, int], extra: str = "") -> None:
    """멈추는 이유와 예전 데이터를 보존하는 방법을 출력한다."""
    new_name = f"{root.name}_old_{found[0].split(': ')[-1].split(',')[0]}" if found else f"{root.name}_old"
    print(f"[STOP] {root} already has data taken with other camera settings:")
    for f in found[:10]:
        print(f"         {f}")
    if len(found) > 10:
        print(f"         ... and {len(found) - 10} more")
    print(f"       This run uses {_size_text(size)}{extra}. Photos with different settings must not be mixed.")
    print("       Nothing was changed. Keep the old data by renaming the folder, then run again:")
    print(f"         mv {root} {root.parent / new_name}")
