"""
새 타겟 색(10/4: 적군 B04 마젠타, 아군 B06 초록) 기준값 도구·판정 단위 테스트 — 합성 데이터만 쓴다 (검출 모델 불필요).

    cd ~/src/pipeline
    venv/bin/python -m unittest discover -s tests -v

(a) a* 부호 폴백 판정(classify_a_sign, ColorClassifier mode='a_sign')
(b) metric='ab' color_refs.json → 절대값 판정이 a*b* 거리로, 안전비 게이트는 그대로
(c) build_color_refs 의 a* 부호 분류·요약(이동 RMS, 안전비, margin)·거리 파싱
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import build_color_refs as B  # noqa: E402
import colorlib as C  # noqa: E402
import target_pipeline as T  # noqa: E402

GREEN = (90, 217, 79)      # B06 #4FD95A (BGR)
MAGENTA = (204, 102, 255)  # B04 #FF66CC (BGR)
BOX = (500, 400, 540, 470)


def frame_with(target_bgr, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((960, 1280, 3), (150, 165, 175), np.float32)
    x1, y1, x2, y2 = BOX
    img[y1:y2, x1:x2] = target_bgr
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def lab_of(bgr):
    return C.bgr_to_lab(np.array([bgr], np.uint8))[0]


def det():
    return T.Det(*BOX, conf=0.9, raw_conf=0.9)


class TestASign(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(C.classify_a_sign([50, 30, 0])["side"], "enemy")
        self.assertEqual(C.classify_a_sign([50, -30, 0])["side"], "friend")
        self.assertEqual(C.classify_a_sign([50, 9.9, 0])["side"], "unknown")
        self.assertEqual(C.classify_a_sign([50, -10.0, 0])["side"], "unknown")   # 경계는 unknown
        self.assertEqual(C.classify_a_sign([50, 15, 0], thr=20)["side"], "unknown")

    def test_classifier_mode(self):
        cc = T.ColorClassifier(None, mode="a_sign")
        self.assertIsNone(cc.cfg)
        e = cc.classify(frame_with(MAGENTA), det())
        f = cc.classify(frame_with(GREEN), det())
        self.assertEqual((e.side, e.mode), ("enemy", "a_sign"))
        self.assertEqual((f.side, f.mode), ("friend", "a_sign"))
        self.assertGreater(e.a_star, 10)
        self.assertLess(f.a_star, -10)

    def test_few_px(self):
        cc = T.ColorClassifier(None, mode="a_sign")
        d = cc.classify(frame_with((150, 150, 150)), det())   # 회색 → 고채도 픽셀 없음
        self.assertEqual((d.side, d.reason), ("unknown", "few_px"))


class TestAbRefs(unittest.TestCase):
    def write_refs(self, safety, margin=10.0):
        g, m = lab_of(GREEN), lab_of(MAGENTA)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(dict(refs=dict(friend=g.tolist(), enemy=m.tolist()), metric="ab", margin=margin, min_px=20,
                       sat_min=60, safety_ratio=safety, jpeg_input=False), tmp)
        tmp.close()
        return tmp.name

    def test_ab_metric(self):
        cc = T.ColorClassifier(self.write_refs(10.0), mode="absolute")
        e = cc.classify(frame_with(MAGENTA), det())
        f = cc.classify(frame_with(GREEN), det())
        self.assertEqual((e.side, e.mode, e.reason), ("enemy", "absolute", "ok"))
        self.assertEqual(f.side, "friend")
        self.assertIsNotNone(e.d_friend_ab)
        self.assertLess(e.d_enemy_ab, e.d_friend_ab)

    def test_ab_ignores_L(self):
        """같은 색을 어둡게 해도(L* 변화) a*b* 판정은 유지."""
        cc = T.ColorClassifier(self.write_refs(10.0), mode="absolute")
        dark = tuple(int(v * 0.6) for v in MAGENTA)
        self.assertEqual(cc.classify(frame_with(dark), det()).side, "enemy")

    def test_ambiguous_large_margin(self):
        """margin 이 두 기준색 거리보다 크면 기준색 그 자체도 ambiguous."""
        cc = T.ColorClassifier(self.write_refs(10.0, margin=200.0), mode="absolute")
        self.assertEqual(cc.classify(frame_with(MAGENTA), det()).reason, "ambiguous")

    def test_safety_gate_kept(self):
        cc = T.ColorClassifier(self.write_refs(2.9), mode="absolute")
        self.assertIsNone(cc.cfg)
        d = cc.classify(frame_with(MAGENTA), det())
        self.assertEqual((d.side, d.reason), ("unknown", "no_refs"))


class TestBuild(unittest.TestCase):
    def test_split_by_a_sign(self):
        d1, d2 = det(), det()
        e = dict(lab=np.array([50, 40, -5.0]), n_px=500)
        f = dict(lab=np.array([55, -30, 25.0]), n_px=400)
        pair, why = B.split_by_a_sign([(d1, e), (d2, f)])
        self.assertIsNone(why)
        self.assertIs(pair["enemy"][1], e)
        self.assertIs(pair["friend"][1], f)
        pair, why = B.split_by_a_sign([(d1, e), (d2, dict(lab=np.array([50, 5.0, 0]), n_px=10))])
        self.assertIsNone(pair)
        self.assertIn("same a* sign", why)

    def test_summarize(self):
        # 아군 a*b* 가 (−35,26) 주위 ±2, 적군 (50,−7) 주위 ±3 으로 흔들림 (L* 는 크게 흔들려도 무관)
        fr = [[57 + dl, -35 + da, 26] for dl, da in ((-10, 2), (0, -2), (10, 2), (5, -2))]
        en = [[50 + dl, 50, -7 + db] for dl, db in ((8, 3), (-8, -3), (0, 3), (3, -3))]
        s = B.summarize([dict(friend=f, enemy=e) for f, e in zip(fr, en)])
        self.assertAlmostEqual(s["rms_ab"]["friend"], 2.0, places=6)
        self.assertAlmostEqual(s["rms_ab"]["enemy"], 3.0, places=6)
        sep = float(np.hypot(85, 33))
        self.assertAlmostEqual(s["sep_ab"], sep, places=6)
        self.assertAlmostEqual(s["safety_ratio"], sep / 2.5, places=6)
        self.assertAlmostEqual(s["margin"], max(0.2 * sep, 9.0), places=6)

    def test_too_few_images_blocks_gate(self):
        s = B.summarize([dict(friend=[57, -35, 26], enemy=[50, 50, -7])] * 2)
        self.assertEqual(s["safety_ratio"], 0.0)

    def test_parse_dist(self):
        self.assertEqual(B.parse_dist("1.5m"), 1.5)
        self.assertEqual(B.parse_dist("target_2.0m_left"), 2.0)
        self.assertEqual(B.parse_dist("1.75 m"), 1.75)
        self.assertIsNone(B.parse_dist("20261004_173459_104"))
        self.assertIsNone(B.parse_dist("png"))

    def test_dist_priority(self):
        p = Path("/x/1.5m/20261004_180000_001.png")
        self.assertEqual(B.dist_of(p, {}, {}, None), (1.5, "dir"))
        self.assertEqual(B.dist_of(p, {}, {p.name: (2.0, "")}, None), (2.0, "map"))
        self.assertEqual(B.dist_of(Path("/x/y/a.png"), {"note": "1.75m"}, {}, None), (1.75, "note"))
        self.assertEqual(B.dist_of(Path("/x/y/a.png"), {}, {}, 2.0), (2.0, "arg"))


if __name__ == "__main__":
    unittest.main()
