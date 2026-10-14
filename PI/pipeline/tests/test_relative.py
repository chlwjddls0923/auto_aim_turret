"""
색 재설계(10/4) 단위 테스트 — 합성 프레임만 쓴다 (카메라·검출 모델이 필요 없다).

    cd ~/src/pipeline
    venv/bin/python -m unittest discover -s tests -v

(a) 화면 전체 색이 일정량 밀려도(조명 변화) 상대 판정이 유지되는지
(b) 기준 조각이 포화되면 unknown(판정 보류)이 나오는지
(c) a*b*만 쓰므로 L*(밝기)가 바뀌어도 판정과 거리가 거의 그대로인지
그 밖에: 상대 판정 실패 시 절대값 폴백, 노출 루프 제안 방향, 픽셀 부족·애매함
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import colorlib as C  # noqa: E402
import target_pipeline as T  # noqa: E402

# 합성 장면의 색 (BGR)
WALL = (150, 165, 175)
WHITE = (200, 200, 200)    # 치우침(+30)을 더해도 포화(250)에 닿지 않게 여유를 둔다. 포화는 (b)에서 따로 본다
FRIEND = (60, 200, 90)     # 연두 계열
ENEMY = (170, 70, 210)     # 분홍 계열
ROIS = {"friend": [1100, 100, 60, 60], "enemy": [1100, 200, 60, 60], "white": [1100, 300, 60, 60]}
TARGET_BOX = (500, 400, 540, 470)


def make_frame(target_bgr, cast=(0, 0, 0), scale=1.0, seed=0, white=WHITE):
    """벽 + 기준 조각 3개 + 타겟 하나. cast = 화면 전체에 더하는 색 치우침(BGR), scale = 밝기 배율."""
    rng = np.random.default_rng(seed)
    img = np.empty((960, 1280, 3), np.float32)
    img[:] = WALL
    for name, bgr in (("friend", FRIEND), ("enemy", ENEMY), ("white", white)):
        x, y, w, h = ROIS[name]
        img[y:y + h, x:x + w] = bgr
    x1, y1, x2, y2 = TARGET_BOX
    img[y1:y2, x1:x2] = target_bgr
    img = (img + np.asarray(cast, np.float32)) * scale + rng.normal(0, 2.0, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def classify(frame, margin=5.0, refs_path=None):
    cc = T.ColorClassifier(refs_path, mode="relative", ref_patches=ROIS, margin_ab=margin)
    reading = cc.read_refs(frame)
    det = T.Det(*TARGET_BOX, conf=0.9, raw_conf=0.9)
    return cc.classify(frame, det, reading), reading


class TestRelative(unittest.TestCase):
    def test_basic(self):
        d, r = classify(make_frame(FRIEND))
        self.assertTrue(r["ok"])
        self.assertEqual((d.side, d.mode), ("friend", "relative"))
        d, _ = classify(make_frame(ENEMY))
        self.assertEqual(d.side, "enemy")

    def test_a_color_cast(self):
        """(a) 저녁 조명처럼 화면 전체가 붉게/푸르게 밀려도 상대 판정은 유지된다."""
        for cast in ((-25, -5, 30), (30, 5, -25), (0, 20, 0)):
            for target, want in ((FRIEND, "friend"), (ENEMY, "enemy")):
                d, r = classify(make_frame(target, cast=cast))
                self.assertTrue(r["ok"], cast)
                self.assertEqual(d.side, want, f"cast {cast}")

    def test_a_absolute_would_fail(self):
        """(a) 비교: 치우침 없는 장면에서 잰 '고정 기준색'은 강한 치우침에서 틀릴 수 있다 → 상대 판정이 필요한 이유."""
        base = C.patch_stats(make_frame(FRIEND), TARGET_BOX)["lab"]
        shifted = C.patch_stats(make_frame(FRIEND, cast=(-40, -10, 45)), TARGET_BOX)["lab"]
        self.assertGreater(C.ab_distance(base, shifted), 10.0)

    def test_b_saturated_ref(self):
        """(b) 흰 조각이 포화되면 그 프레임은 기준 불량 → unknown."""
        d, r = classify(make_frame(FRIEND, white=(255, 255, 255)))
        self.assertFalse(r["ok"])
        self.assertEqual(r["status"]["white"], "saturated")
        self.assertEqual(d.side, "unknown")
        self.assertIn("ref_white_saturated", d.reason)

    def test_b_dark_ref(self):
        d, r = classify(make_frame(FRIEND, scale=0.12))
        self.assertFalse(r["ok"])
        self.assertEqual(d.side, "unknown")
        self.assertIn("dark", d.reason)

    def test_c_brightness_invariant(self):
        """(c) 밝기만 바뀌면 판정은 그대로이고, a*b* 거리 차이(판정 여유)도 크게 바뀌지 않는다."""
        out = []
        for scale in (0.8, 1.0, 1.1):
            d, r = classify(make_frame(ENEMY, scale=scale))
            self.assertTrue(r["ok"], scale)
            self.assertEqual(d.side, "enemy", scale)
            out.append(d.d_friend_ab - d.d_enemy_ab)
        self.assertGreater(min(out), 0.5 * max(out))

    def test_ab_ignores_L(self):
        self.assertAlmostEqual(C.ab_distance([30, 10, -5], [80, 10, -5]), 0.0)
        self.assertAlmostEqual(C.ab_distance([50, 0, 0], [50, 3, 4]), 5.0)

    def test_ambiguous(self):
        mid = tuple(int(round((f + e) / 2)) for f, e in zip(FRIEND, ENEMY))
        d, _ = classify(make_frame(mid), margin=1000.0)
        self.assertEqual((d.side, d.reason), ("unknown", "ambiguous"))

    def test_few_px(self):
        d, _ = classify(make_frame(WALL))     # 타겟이 벽색 → 고채도 픽셀 없음
        self.assertEqual((d.side, d.reason), ("unknown", "few_px"))

    def test_white_gains_neutralize(self):
        g = C.white_gains((180, 200, 230))
        out = C.apply_gains(np.array([[180, 200, 230]], np.uint8), g)[0]
        self.assertLessEqual(int(out.max()) - int(out.min()), 1)

    def test_fallback_needs_refs(self):
        """기준 불량 + 절대값 기준 없음 → unknown, reason 에 불량 조각이 남는다."""
        d, _ = classify(make_frame(FRIEND, white=(255, 255, 255)), refs_path=None)
        self.assertEqual(d.mode, None)
        self.assertTrue(d.reason.startswith("ref_"))

    def test_decide_holds_on_unknown(self):
        a = T.Det(0, 0, 10, 20, 0.9, 0.9, side="enemy", reason="ok")
        b = T.Det(0, 0, 10, 20, 0.9, 0.9, side="unknown", reason="ref_white_saturated")
        self.assertEqual(T.decide([a, b])["action"], "hold")


class TestExposureLoop(unittest.TestCase):
    def reading(self, L=None, status="ok"):
        return dict(status={"white": status}, lab={"white": None if L is None else [L, 0, 0]})

    def test_off(self):
        self.assertIsNone(T.ExposureLoop(enabled=False).propose(self.reading(40), 10000))

    def test_direction_and_limit(self):
        loop = T.ExposureLoop(enabled=True, target_L=75, max_step=0.15)
        self.assertEqual(loop.propose(self.reading(50), 10000), 11500)        # 어두우면 늘린다 (최대 15%)
        self.assertEqual(loop.propose(self.reading(95), 10000), 8500)         # 밝으면 줄인다
        self.assertIsNone(loop.propose(self.reading(76), 10000))              # 목표 근처면 그대로
        self.assertEqual(loop.propose(self.reading(status="saturated"), 10000), 8500)


if __name__ == "__main__":
    unittest.main()
