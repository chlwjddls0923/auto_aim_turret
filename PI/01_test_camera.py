#!/usr/bin/env python3
"""
[1단계] 최소 카메라 테스트 - 동작 확인 + 채널 순서 확인

목표:
  1) picamera2가 실제로 프레임을 전달하는지 확인한다.
  2) picamera2의 "RGB888" 포맷이 메모리에 BGR 순서로 저장된다는 점을
     실측으로 확인한다.

사용법:
  빨간 물체(빨간 종이, 빨간 테이프 등)를 카메라 화면 중앙에 두어
  중앙을 가득 채운 뒤 다음을 실행한다:

      python3 01_test_camera.py --truth red

  파란 물체라면 --truth blue, 초록 물체라면 --truth green 을 사용한다.
  카메라 동작만 확인하려면(물체 없이) --truth 를 생략한다.
"""

import argparse
import time
import sys

import numpy as np
import cv2

# 이 스크립트는 /home/pa8/src 에서 실행하는 것을 전제로 한다.
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURES = HERE / "01_test_camera"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--truth",
        choices=["red", "green", "blue"],
        default=None,
        help="The real color of the object you hold in front of the camera (used to find the channel order)",
    )
    ap.add_argument("--width", type=int, default=1296)
    ap.add_argument("--height", type=int, default=972)
    ap.add_argument("--warmup", type=float, default=3.0, help="Time to wait for AWB/AE to settle (seconds)")
    args = ap.parse_args()

    try:
        from picamera2 import Picamera2
    except ImportError as e:
        print(f"[FAIL] Cannot import picamera2: {e}")
        print("      Check that you use the system package installed with apt (sudo apt install python3-picamera2)")
        return 1

    CAPTURES.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print(" Step 1: camera works + channel order test")
    print("=" * 62)

    # ---- 카메라 열기 ---------------------------------------------------
    picam2 = Picamera2()

    print("\n[Sensor info]")
    for i, m in enumerate(picam2.sensor_modes):
        print(f"  mode {i}: size={m.get('size')} format={m.get('format')} bit={m.get('bit_depth')}")

    cfg = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    picam2.configure(cfg)
    picam2.start()
    print(f"\n[Wait] Letting AWB/AE settle for {args.warmup:.1f} s...")
    time.sleep(args.warmup)

    # ---- 프레임 획득 ---------------------------------------------------
    arr = picam2.capture_array("main")
    meta = picam2.capture_metadata()

    print(f"\n[Frame] shape={arr.shape} dtype={arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] < 3:
        print("[FAIL] The image does not have 3 channels.")
        picam2.stop()
        return 1

    # ---- 중앙 ROI (이미지 중앙 30%) ------------------------------------
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    ry, rx = int(h * 0.15), int(w * 0.15)
    roi = arr[cy - ry : cy + ry, cx - rx : cx + rx, :3].astype(np.float64)

    # 메모리에 있는 순서 그대로 채널 평균을 구한다. 어느 채널이 R인지는 아직 모른다고 가정한다.
    ch0, ch1, ch2 = roi[:, :, 0].mean(), roi[:, :, 1].mean(), roi[:, :, 2].mean()

    print(f"\n[Center ROI {2*rx}x{2*ry}px] Channel averages, in memory order")
    print(f"  channel[0] = {ch0:7.2f}")
    print(f"  channel[1] = {ch1:7.2f}")
    print(f"  channel[2] = {ch2:7.2f}")

    # ---- 채널 순서 판별 -----------------------------------------------
    # Case A (가설 A): picamera2 "RGB888" = 메모리상 BGR  -> R은 channel[2]
    # Case B (가설 B): 이름 그대로 RGB                    -> R은 channel[0]
    print("\n[Channel order check]")
    print("  Case A: 'RGB888' is BGR in memory  -> R=ch2, G=ch1, B=ch0   (official picamera2 behavior)")
    print("  Case B: RGB, as the name says      -> R=ch0, G=ch1, B=ch2")

    verdict = None
    if args.truth is None:
        print("\n  --truth was not given, so the automatic check is skipped.")
        print("  Hold a red object in front of the camera and run again: 'python3 01_test_camera.py --truth red'")
    else:
        vals = [ch0, ch1, ch2]
        dominant = int(np.argmax(vals))  # 값이 가장 큰 채널의 인덱스
        # Case A / Case B 각각에서 실제 색이 위치해야 할 채널 인덱스
        idx_under_A = {"blue": 0, "green": 1, "red": 2}[args.truth]
        idx_under_B = {"red": 0, "green": 1, "blue": 2}[args.truth]

        print(f"\n  Real object color = {args.truth.upper()}")
        print(f"  Channel with the largest value = channel[{dominant}]")

        if args.truth == "green":
            print("  ! Green is ch1 in both cases, so it cannot tell them apart. Test again with red or blue.")
        elif dominant == idx_under_A and dominant != idx_under_B:
            verdict = "A"
            print("  => Result: Case A confirmed. The 'RGB888' buffer is in BGR order in memory.")
            print("     So to get RGB for display or logs, you must flip it with arr[:, :, ::-1].")
        elif dominant == idx_under_B and dominant != idx_under_A:
            verdict = "B"
            print("  => Result: Case B. The buffer is RGB, as the name says. (Not expected - the code must be changed)")
        else:
            print("  => Cannot decide. The object may not fill the center of the image, or the light may be too weak.")
            print("     Hold a strongly colored object closer to the center and try again.")

    # ---- 확인용 이미지 2장 저장 ----------------------------------------
    # cv2.imwrite 는 입력을 BGR로 취급한다.
    #   - arr 가 실제로 BGR이라면(Case A), 그대로 저장한 파일이 올바른 색으로 보인다.
    p_a = CAPTURES / "channel_test_A_assume_bgr.jpg"
    p_b = CAPTURES / "channel_test_B_assume_rgb.jpg"
    cv2.imwrite(str(p_a), arr[:, :, :3])
    cv2.imwrite(str(p_b), arr[:, :, 2::-1])
    print("\n[Saved check images] The one that shows the correct colors is the right answer.")
    print(f"  Looks correct if Case A is true: {p_a}")
    print(f"  Looks correct if Case B is true: {p_b}")
    print("  To view on a PC:  scp pa8@<IP of this Raspberry Pi>:~/src/01_test_camera/channel_test_*.jpg .")

    # ---- 현재 카메라 제어값 --------------------------------------------
    print("\n[Current auto control values] (Step 2 will lock these values)")
    for key in ("ColourGains", "ExposureTime", "AnalogueGain", "DigitalGain", "Lux", "ColourTemperature"):
        if key in meta:
            print(f"  {key:18s} = {meta[key]}")

    picam2.stop()
    picam2.close()

    print("\n" + "=" * 62)
    if verdict == "A":
        print(" Result: OK. Channel order = BGR (as expected). You can go to the next step.")
    elif verdict == "B":
        print(" Result: The channel order is not what we expected. Please report it - the code will be fixed to match.")
    else:
        print(" Result: The camera works. Check the channel order again with --truth red.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
