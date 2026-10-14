"""
조준점 보정(07_aim_calib.py) + Aimer(aim_table) 단위 테스트 — 카메라·검출 모델이 필요 없다.

    cd ~/src/pipeline
    venv/bin/python -m unittest discover -s tests -v

(a) 박스가 여러 개일 때 선택 거리의 기대 h(59.1/d)에 맞는 박스를 고르고, 비슷하면 '애매'로 넘기는지
(b) CSV 행 -> 거리별 요약(평균·표준편차·경고) -> aim_table.json 형식(h 오름차순)이 맞는지
(c) Aimer가 aim_table을 읽어 조준점을 보간하고, yaw를 화면 중앙 대신 조준점 기준으로 내는지 (표가 없으면 예전과 같은 값)
(d) AimStore가 PNG와 CSV를 쓰고 다시 읽는지, 창 없이 record/on_click 경로가 동작하는지
"""
import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # ~/src/pipeline
sys.path.insert(0, str(ROOT / "tools"))
import target_pipeline as T  # noqa: E402

_spec = importlib.util.spec_from_file_location("aim_calib", ROOT.parent / "07_aim_calib.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def det(cx, cy, h, w=None, conf=0.9):
    w = h * 0.6 if w is None else w
    return T.Det(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf=conf, raw_conf=conf)


class ChooseBox(unittest.TestCase):
    def test_single_box_is_clear(self):
        d = det(600, 480, 30)
        self.assertEqual(A.choose_box([d], 2.0), (d, False))

    def test_picks_box_matching_distance(self):
        near, far = det(400, 480, 39.0), det(800, 470, 29.5)      # 1.5 m 와 2.0 m 에 해당하는 h
        self.assertEqual(A.choose_box([near, far], 1.5), (near, False))
        self.assertEqual(A.choose_box([far, near], 2.0), (far, False))

    def test_similar_heights_are_ambiguous(self):
        a, b = det(400, 480, 38.0), det(800, 480, 40.0)
        chosen, ambiguous = A.choose_box([a, b], 1.5)
        self.assertTrue(ambiguous)
        self.assertIn(chosen, (a, b))

    def test_no_match_is_ambiguous(self):
        a, b = det(400, 480, 80.0), det(800, 480, 15.0)           # 둘 다 1.5 m 의 39.4 px 와 멀다
        self.assertTrue(A.choose_box([a, b], 1.5)[1])

    def test_nearest_box(self):
        a, b = det(400, 480, 38.0), det(800, 480, 40.0)
        self.assertIs(A.nearest_box([a, b], 790, 470), b)
        self.assertIs(A.nearest_box([a, b], 100, 100), a)


def row(d, cx, cy, h=None, source="detect", png="x.png"):
    return dict(timestamp="t", distance_m=A.dist_key(d), cx=cx, cy=cy, h_px=h, w_px=None if h is None else h * 0.6,
                conf=None if h is None else 0.9, yaw_cmd=95.0, pitch_cmd=90.0, source=source, png=png)


class Summary(unittest.TestCase):
    ROWS = [row(1.5, 660, 470, 38.0), row(1.5, 664, 474, 39.0), row(1.5, 662, 472, 38.5),
            row(2.0, 650, 440, 29.0), row(2.0, 652, 442, 30.0),
            row(1.8, 655, 455, None, source="click")]

    def test_per_distance_stats(self):
        s = A.summarize(self.ROWS)
        self.assertEqual(list(s), [1.5, 1.8, 2.0])
        self.assertEqual((s[1.5]["n"], s[1.5]["n_detect"], s[1.5]["n_click"]), (3, 3, 0))
        self.assertAlmostEqual(s[1.5]["cx"], 662.0); self.assertAlmostEqual(s[1.5]["cy"], 472.0)
        self.assertAlmostEqual(s[1.5]["h"], 38.5); self.assertAlmostEqual(s[1.5]["cx_std"], 2.0)
        self.assertFalse(s[1.5]["warn"])
        self.assertEqual(s[1.8]["h_src"], "expected"); self.assertAlmostEqual(s[1.8]["h"], 59.1 / 1.8)
        self.assertEqual(s[1.8]["cx_std"], 0.0)       # 기록 1개 → 표준편차 0

    def test_std_warning(self):
        rows = [row(1.6, 600, 480, 36), row(1.6, 640, 480, 36)]   # cx 가 40 px 벌어짐 → sd 28
        self.assertTrue(A.summarize(rows)[1.6]["warn"])

    def test_table_is_sorted_by_h(self):
        t = A.build_aim_table(A.summarize(self.ROWS), (1280, 960), "aim_points.csv")
        self.assertEqual(t["h_px"], sorted(t["h_px"]))
        self.assertEqual(len(t["h_px"]), len(t["cx"]), len(t["cy"]))
        self.assertEqual(t["n_points"], 6)
        self.assertEqual([p["distance_m"] for p in t["per_distance"]], [2.0, 1.8, 1.5])   # h 오름차순 = 거리 내림차순
        self.assertEqual(t["frame_size"], [1280, 960])


class AimerTable(unittest.TestCase):
    YAW = {"type": "poly", "coef": [0, 64], "sign": -1}
    TABLE = {"h_px": [29.5, 38.5], "cx": [650.0, 662.0], "cy": [441.0, 472.0]}

    def test_without_table_is_center(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = T.Aimer(self.YAW, None, 1280)
        r = a.aim(det(704, 480, 38.5))
        self.assertAlmostEqual(r["yaw_deg"], -6.4)                 # 예전과 같은 값: -64·(64/640)
        self.assertEqual((r["aim_cx"], r["aim_cy"], r["aim_table"]), (640.0, 480.0, False))
        self.assertAlmostEqual(r["dx_px"], 64.0); self.assertAlmostEqual(r["dy_px"], 0.0)
        self.assertFalse(r["pitch_valid"])

    def test_table_shifts_yaw_reference(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = T.Aimer(self.YAW, None, 1280, aim_table=self.TABLE)
        r = a.aim(det(662, 472, 38.5))                              # 조준점 위에 있으면 보정 0
        self.assertAlmostEqual(r["yaw_deg"], 0.0); self.assertAlmostEqual(r["dx_px"], 0.0); self.assertTrue(r["aim_table"])
        r = a.aim(det(726, 472, 38.5))                              # 조준점에서 +64 px → -6.4 deg
        self.assertAlmostEqual(r["yaw_deg"], -6.4)
        mid = a.aim(det(640, 480, 34.0))                            # 두 점 사이: 선형 보간
        self.assertAlmostEqual(mid["aim_cx"], 656.0); self.assertAlmostEqual(mid["aim_cy"], 456.5)
        self.assertAlmostEqual(a.aim(det(640, 480, 10.0))["aim_cx"], 650.0)   # 범위 밖은 끝값
        self.assertAlmostEqual(a.aim(det(640, 480, 90.0))["aim_cx"], 662.0)

    def test_pitch_calib_uses_aim_cy(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = T.Aimer(self.YAW, None, 1280, aim_table=self.TABLE, pitch_calib={"type": "poly", "coef": [0, 46.6], "sign": 1}, frame_h=960)
        r = a.aim(det(662, 472 + 48, 38.5))
        self.assertTrue(r["pitch_valid"]); self.assertAlmostEqual(r["pitch_deg"], 4.66); self.assertAlmostEqual(r["dy_px"], 48.0)

    def test_table_from_file_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "aim_table.json"
            p.write_text(json.dumps(self.TABLE))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = T.Aimer(self.YAW, None, 1280, aim_table=str(p))
            self.assertAlmostEqual(a.aim(det(640, 480, 38.5))["aim_cx"], 662.0)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                b = T.Aimer(self.YAW, None, 1280, aim_table=str(Path(tmp) / "none.json"))
            self.assertTrue(any("aim_table" in str(x.message) for x in w))
            self.assertFalse(b.aim(det(640, 480, 38.5))["aim_table"])

    def test_lut_yaw_is_relative_to_aim_point(self):
        lut = {"type": "lut", "px": [0, 1280], "deg": [-64, 64]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = T.Aimer(lut, None, 1280, aim_table=self.TABLE)
        self.assertAlmostEqual(a.aim(det(662 + 64, 472, 38.5))["yaw_deg"], 6.4)


class StoreAndApp(unittest.TestCase):
    def make_app(self, tmp, detector=None, live=False):
        args = SimpleNamespace(distance=1.5, step=2.0, yaw_limit=30.0, center_yaw=95.0, center_pitch=90.0,
                               width=1280, height=960, live_detect=live, view_scale=1.0)
        store = A.AimStore(Path(tmp) / "calib")
        frame = np.full((960, 1280, 3), 120, np.uint8)
        cam = SimpleNamespace(frame=lambda: frame.copy(), label="TEST")
        servos = A.Servos(enabled=False, settle=0.0)
        servos.center(95.0, 90.0)
        app = A.AimCalib(args, cam, servos, detector, store, window=False)
        return app, store, frame

    def test_store_roundtrip_and_click_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, store, frame = self.make_app(tmp)
            self.assertEqual(app.begin_record(frame), "click")          # 검출기 없음 → 클릭 대기
            self.assertFalse(app.on_click(10, 10) is False)             # 대기 중 클릭 → 기록
            self.assertEqual(store.count(1.5), 1)
            r = store.rows[0]
            self.assertEqual((r["cx"], r["cy"], r["h_px"], r["source"], r["png"]), (10.0, 10.0, None, "click", "1.5m_001.png"))
            png = store.shots / "1.5m_001.png"
            self.assertTrue(png.exists())
            import cv2
            self.assertTrue(np.array_equal(cv2.imread(str(png)), frame))   # 오버레이 없는 원본
            # 검출 결과 기록 (record 직접 호출) + 번호 증가
            app.record(frame, 662.3, 471.9, 38.5, 24.0, 0.89, "detect")
            self.assertEqual(store.rows[-1]["png"], "1.5m_002.png")
            app.set_distance(+0.1); app.set_distance(+0.1)
            self.assertEqual(app.distance, 1.7)
            app.record(frame, 600.0, 450.0, 34.0, 20.0, 0.8, "detect")
            self.assertEqual(store.rows[-1]["png"], "1.7m_001.png")
            # 다시 읽기
            again = A.AimStore(Path(tmp) / "calib")
            self.assertEqual(len(again.rows), 3)
            self.assertEqual(again.next_index(1.5), 3); self.assertEqual(again.next_index(1.7), 2); self.assertEqual(again.next_index(2.0), 1)
            self.assertEqual(again.rows[1]["h_px"], 38.5); self.assertEqual(again.rows[1]["conf"], 0.89)
            # 종료 요약 → aim_table.json
            table = app.finish()
            self.assertTrue(store.table_path.exists())
            self.assertEqual(table["h_px"], sorted(table["h_px"]))
            self.assertEqual(table["n_points"], 3)
            # 렌더는 원본을 건드리지 않고 같은 크기 이미지를 돌려준다
            img = app.render(frame, None)
            self.assertEqual(img.shape, frame.shape)
            self.assertTrue(np.array_equal(frame, np.full((960, 1280, 3), 120, np.uint8)))

    def test_pick_mode_with_fake_detector(self):
        a, b = det(400, 480, 38.0), det(800, 480, 40.0)
        fake = SimpleNamespace(detect=lambda frame, roi=None: [a, b])
        with tempfile.TemporaryDirectory() as tmp:
            app, store, frame = self.make_app(tmp, detector=fake)
            self.assertIsNone(app.live)                                   # 기본: 실시간 검출 없음
            self.assertEqual(app.begin_record(frame), "pick")
            self.assertEqual(len(app.last_e["dets"]), 2)
            app.on_click(805, 485)
            self.assertEqual(store.rows[-1]["cx"], 800.0)
            self.assertEqual(store.rows[-1]["source"], "detect")
            self.assertIs(app.last_e["chosen"], b)
            img = app.render(frame, None)                                 # e 결과 박스를 그리는 경로가 돈다
            self.assertEqual(img.shape, frame.shape)

    def test_single_box_records_without_live(self):
        calls = []
        one = det(600, 470, 39.0)
        fake = SimpleNamespace(detect=lambda frame, roi=None: calls.append(roi) or [one])
        with tempfile.TemporaryDirectory() as tmp:
            app, store, frame = self.make_app(tmp, detector=fake)
            self.assertEqual(app.begin_record(frame), "recorded")
            self.assertEqual(calls, [None])                               # e 한 번 = 전체 프레임 검출 한 번
            self.assertEqual((store.rows[-1]["cx"], store.rows[-1]["h_px"]), (600.0, 39.0))
            self.assertIs(app.last_e["chosen"], one)

    def test_live_detect_option_creates_thread_and_uses_lock(self):
        one = det(600, 470, 39.0)
        fake = SimpleNamespace(detect=lambda frame, roi=None: [one])
        with tempfile.TemporaryDirectory() as tmp:
            app, store, frame = self.make_app(tmp, detector=fake, live=True)
            self.assertIsInstance(app.live, A.LiveDetector)
            self.assertFalse(app.live.is_alive())                         # run() 에서만 시작한다
            self.assertEqual(app.begin_record(frame), "recorded")
            self.assertFalse(app.live.pause.is_set())

    def test_fire_arming_and_auto_return(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            app, _, _ = self.make_app(tmp)
            servos = app.servos
            app.fire_duration = 0.2
            seen = []
            orig = servos.write
            servos.write = lambda pin, angle: (seen.append((pin, round(angle, 2))), orig(pin, angle))[1]
            servos.fire_ready()
            self.assertFalse(app.fire())                               # 시작 직후 5초: 막힘
            self.assertTrue(app.fire_state().startswith("ARMING"))
            app.fire_armed_at = time.monotonic() - 1
            self.assertEqual(app.fire_state(), "READY")
            seen.clear()
            t0 = time.monotonic()
            self.assertTrue(app.fire())
            self.assertTrue(app.fire_busy())
            self.assertFalse(app.fire())                               # 발사 중 재발사 막힘
            app.move(2.0, 2.0)                                         # 발사 중 yaw/pitch 명령 안 나감
            app.center()
            app.fire_thread.join(5)
            took = time.monotonic() - t0
            fire_steps = [a for p, a in seen if p == A.FIRE_PIN]
            self.assertFalse([p for p, a in seen if p != A.FIRE_PIN])
            up = fire_steps[:fire_steps.index(A.FIRE_SHOOT) + 1]
            self.assertEqual(len(up), round((A.FIRE_SHOOT - A.FIRE_READY) / A.FIRE_STEP_DEG))   # 0.5도씩 140칸
            self.assertTrue(all(abs(b - a) <= A.FIRE_STEP_DEG + 1e-6 for a, b in zip([A.FIRE_READY] + fire_steps, fire_steps)))
            self.assertEqual(fire_steps[-1], A.FIRE_READY)
            self.assertGreater(took, 2 * 0.2 + A.FIRE_HOLD_SEC - 0.05)   # 접근 + 유지 + 복귀
            self.assertEqual(app.fire_state(), "WAIT")                  # 끝난 뒤 0.3초 쿨다운
            self.assertTrue(app.fire_busy())
            time.sleep(A.FIRE_COOLDOWN_SEC + 0.05)
            self.assertFalse(app.fire_busy())
            self.assertEqual(app.fire_state(), "READY")
            self.assertTrue(app.fire())                                # 종료 중 발사: 멈추고 천천히 READY
            time.sleep(0.1)
            app.fire_safe()
            self.assertEqual(servos.get(A.FIRE_PIN, None), A.FIRE_READY)
            self.assertEqual(A.Servos.angle_to_us(A.FIRE_SHOOT), round(500 + 160 / 180 * 2000))

    def test_marks_persist_and_e_saves_last(self):
        one = det(600, 470, 39.0)
        fake = SimpleNamespace(detect=lambda frame, roi=None: [one])
        with tempfile.TemporaryDirectory() as tmp:
            app, store, frame = self.make_app(tmp, detector=fake)
            for xy in ((100, 100), (300, 200), (602, 468)):
                self.assertFalse(app.on_click(*xy))
            self.assertEqual(app.marks, [(100, 100), (300, 200), (602, 468)])
            app.render(frame, None)                                     # 표시를 그리는 경로가 돈다
            self.assertEqual(app.begin_record(frame), "recorded")
            r = store.rows[-1]
            self.assertEqual((r["cx"], r["cy"], r["source"], r["h_px"]), (602.0, 468.0, "click", 39.0))   # 마지막 표시 + 감싸는 박스의 h
            self.assertEqual(app.marks, [])                             # 저장 후 지움
            app.on_click(50, 50)
            self.assertEqual(app.begin_record(frame), "recorded")
            self.assertEqual((store.rows[-1]["cx"], store.rows[-1]["h_px"]), (50.0, None))                # 박스 밖 → h 비움
            self.assertEqual(store.count(), 2)

    def test_distance_clamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _, _ = self.make_app(tmp)
            for _ in range(10):
                app.set_distance(+0.1)
            self.assertEqual(app.distance, 2.0)
            for _ in range(10):
                app.set_distance(-0.1)
            self.assertEqual(app.distance, 1.5)


class LivePlan(unittest.TestCase):
    def test_full_only_every_interval(self):
        plan = A.LiveDetector.plan
        self.assertEqual(plan(None, -1e9, 100.0, 2.0), "full")           # 처음: 전체
        self.assertIsNone(plan(None, 100.0, 100.5, 2.0))                 # 타겟 없음 + 2초 전 → 쉼
        self.assertEqual(plan((640, 480), 100.0, 100.5, 2.0), "roi")     # 타겟 있음 → ROI 추적
        self.assertEqual(plan((640, 480), 100.0, 101.9, 2.0), "roi")
        self.assertEqual(plan((640, 480), 100.0, 102.0, 2.0), "full")    # 2초 지남 → 전체 1회
        self.assertEqual(plan(None, 100.0, 102.1, 2.0), "full")


if __name__ == "__main__":
    unittest.main()


# 10-06 실제 조준 기록 8줄 (시험은 실제 calib/ 파일이 바뀌어도 흔들리지 않게 이 고정본을 쓴다)
AIM_ROWS_1006 = """timestamp,distance_m,cx,cy,h_px,w_px,conf,yaw_cmd,pitch_cmd,source,png
2026-10-04T16:28:27,1.5,603.0,611.0,34.4,29.3,0.268,93.5,103.8,click,1.5m_005.png
2026-10-06T14:21:40,1.5,1027.0,623.0,,,,89.0,108.0,click,1.5m_006.png
2026-10-06T14:35:27,1.5,914.0,407.0,,,,93.0,102.0,click,1.5m_007.png
2026-10-06T14:38:42,1.5,620.0,604.0,,,,91.0,104.0,click,1.5m_008.png
2026-10-06T14:40:55,1.5,618.0,605.0,,,,93.0,104.0,click,1.5m_009.png
2026-10-06T15:26:10,1.5,651.0,619.0,,,,97.0,106.0,click,1.5m_010.png
2026-10-06T15:49:39,1.5,615.0,625.0,,,,95.0,106.0,click,1.5m_011.png
2026-10-06T15:50:39,1.5,618.0,608.0,,,,101.0,106.0,click,1.5m_012.png
"""


class FakeColor:
    """가짜 판정기: side_of(x, y) -> 'enemy' | 'friend' | 'unknown'. ColorClassifier와 같은 모양(read_refs, classify, mode, cfg, ref)."""

    def __init__(self, side_of):
        self.side_of, self.mode, self.cfg, self.ref = side_of, "absolute", {"safety_ratio": 15.3}, None
        self.calls = 0

    def read_refs(self, frame):
        return None

    def classify(self, frame, d, reading=None):
        self.calls += 1
        d.side = self.side_of(d.cx, d.cy)
        d.reason = "ok" if d.side != "unknown" else "few_px"
        d.mode = "absolute"
        return d


class AutoAim(unittest.TestCase):
    """p 적군 자동 조준: 서보 각도에 따라 화면이 밀리는 가상 카메라 + 가짜 판정기로 끝까지 돌려 본다."""

    K_TRUE = 12.0             # 가상 카메라: 서보 1도 = 화면 12 px

    def test_helpers(self):
        rows = [dict(distance_m=1.5, cx=cx, cy=cy, h_px=h) for cx, cy, h in
                ((603, 611, 34.4), (1027, 623, None), (914, 407, None), (620, 604, None), (618, 605, None),
                 (651, 619, None), (615, 625, None), (618, 608, None))]          # 10-06 실제 기록 (잘못 찍은 2개 포함)
        ax, ay, d, n = A.aim_point_for(rows, 1.52)
        self.assertEqual((ax, ay, d, n), (619.0, 609.5, 1.5, 8))                # 중앙값 -> 잘못 찍은 기록에 흔들리지 않음
        self.assertIsNone(A.aim_point_for(rows, 1.8))                          # 기록 거리와 0.2 m 넘게 차이
        self.assertIsNone(A.aim_point_for([], 1.5))
        self.assertEqual(A.h_coef_from_rows(rows), (34.4 * 1.5, "measured x1"))
        self.assertEqual(A.h_coef_from_rows([dict(distance_m=1.5, h_px=None)]), (A.H_COEF, "default"))
        self.assertAlmostEqual(A.servo_step(120, 12), 8.0)                     # 최대 8도
        self.assertAlmostEqual(A.servo_step(-24, 12), -2.0)
        self.assertAlmostEqual(A.servo_step(8, 12), A.MIN_STEP_DEG)            # 최소 폭(기본 0.9도)보다 작으면 최소 폭만큼
        self.assertEqual(A.servo_step(4, 12, tol=6), 0.0)                      # 허용 오차 안이면 그 축은 안 움직인다
        self.assertAlmostEqual(A.servo_step(4, 12), A.MIN_STEP_DEG)
        self.assertAlmostEqual(A.servo_step(4, 12, min_deg=0.225), 4 / 12)     # 넘어선 뒤 줄어든 최소 폭
        self.assertAlmostEqual(A.learn_gain(22, 2.0, 24.0), 12.0)
        self.assertAlmostEqual(A.learn_gain(22, 2.0, -24.0), -12.0)            # 방향이 반대여도 배운다
        self.assertEqual(A.learn_gain(22, 0.5, 24.0), 22)                      # 너무 작은 이동은 무시
        self.assertEqual(A.learn_gain(12, 0.9, 17.0), 12)                      # 0.9도(최소 한 칸) 이동으로는 다시 재지 않는다
        self.assertEqual(A.learn_gain(22, 2.0, 400.0), 22)                     # 이상한 값은 무시
        e, f = det(1, 1, 30), det(5, 5, 30)
        e.side, f.side, f.reason = "enemy", "unknown", "few_px"
        self.assertEqual(A.judge_text([e, f]), "enemy 1, unknown 1 (few_px)")
        self.assertEqual(A.judge_text([]), "no box")
        self.assertIn("no friend check", A.color_status(None))
        self.assertIn("NO REFS", A.color_status(SimpleNamespace(mode="relative", ref=None, cfg=None)))
        self.assertIn("safety 15.3", A.color_status(FakeColor(lambda x, y: "enemy")))

    def make_world(self, tmp, k_yaw=12.0, k_pitch=12.0, px_per_deg=22.0, sides=("enemy",), side_fn=None, color=True):
        """세계 좌표에 타겟 여러 개 (sides 순서로 왼쪽부터 200 px 간격). 판정기는 세계 좌표로 진짜 side를 돌려준다."""
        import cv2
        calib = Path(tmp) / "calib"
        calib.mkdir()
        (calib / "aim_points.csv").write_text(AIM_ROWS_1006, encoding="utf-8")   # 고정 기록 -> 조준점 (619, 609.5)
        rng = np.random.default_rng(0)
        world = cv2.GaussianBlur(rng.integers(0, 255, (1960, 2280, 3), dtype=np.uint8), (0, 0), 3)
        targets = [(1300 + 200 * i, 950, side) for i, side in enumerate(sides)]
        for TX, TY, _ in targets:
            cv2.rectangle(world, (TX - 12, TY - 17), (TX + 12, TY + 17), (200, 60, 230), -1)
        servos = A.Servos(enabled=False, settle=0.0)
        servos.center(95.0, 90.0)

        def origin():   # yaw +1도 -> 화면 내용이 +k_yaw px(오른쪽), pitch +1도 -> +k_pitch px(아래)
            return (500 - k_yaw * (servos.get(A.YAW_PIN, 95) - 95), 500 - k_pitch * (servos.get(A.PITCH_PIN, 90) - 90))

        def frame():
            x0, y0 = origin()
            return cv2.warpAffine(world, np.float32([[1, 0, -x0], [0, 1, -y0]]), (1280, 960))

        def detect(img, roi=None):
            x0, y0 = origin()
            out = [det(TX - x0, TY - y0, 34.4, 24.0) for TX, TY, _ in targets]
            if roi is not None:      # ROI 검출: 384 px 창 안의 것만
                out = [d for d in out if abs(d.cx - roi[0]) <= 192 and abs(d.cy - roi[1]) <= 192]
            return out

        def true_side(x, y):
            x0, y0 = origin()
            TX, TY, side = min(targets, key=lambda t: (t[0] - x0 - x) ** 2 + (t[1] - y0 - y) ** 2)
            return side_fn(side) if side_fn else side

        args = SimpleNamespace(distance=1.5, step=2.0, yaw_limit=30.0, center_yaw=95.0, center_pitch=90.0,
                               width=1280, height=960, live_detect=False, view_scale=1.0, px_per_deg=px_per_deg)
        store = A.AimStore(calib)
        cam = SimpleNamespace(frame=frame, label="SIM")
        fc = FakeColor(true_side) if color else None
        app = A.AimCalib(args, cam, servos, SimpleNamespace(detect=detect), store, window=False, color=fc)
        app.fire_duration = 0.05
        app.fire_armed_at = -1e9
        pos_of = lambda i: (targets[i][0] - origin()[0], targets[i][1] - origin()[1])   # noqa: E731
        return app, pos_of

    def run_auto(self, app):
        import time
        old = A.AUTO_SETTLE_SEC
        A.AUTO_SETTLE_SEC = 0.0
        try:
            res = app.auto_aim(app.cam.frame())
        finally:
            A.AUTO_SETTLE_SEC = old
        if app.fire_thread is not None:
            app.fire_thread.join(5)
        time.sleep(A.FIRE_COOLDOWN_SEC + 0.05)
        return res

    def last_log(self, tmp):
        import csv as _csv
        return list(_csv.DictReader(open(Path(tmp) / "calib" / "auto_shots.csv")))[-1]

    def assert_on_aim(self, app, xy, slack=None):
        slack = app.auto_tol + A.MIN_STEP_DEG * self.K_TRUE / 2 + 1 if slack is None else slack
        ax, ay, _, _ = A.aim_point_for(app.store.rows, app.distance, float("inf"))
        self.assertLessEqual(abs(xy[0] - ax), slack)
        self.assertLessEqual(abs(xy[1] - ay), slack)

    def test_arming_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp)
            app.fire_armed_at = float("inf")
            self.assertEqual(app.auto_aim(app.cam.frame()), "arming")
            self.assertEqual(app.servos.get(A.YAW_PIN, None), 95.0)            # 잠금 중엔 서보도 안 움직인다

    def test_enemy_converges_and_fires(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp, px_per_deg=22.0)
            self.assertEqual(self.run_auto(app), "fired")
            self.assert_on_aim(app, pos_of(0))
            self.assertAlmostEqual(app.k_yaw, self.K_TRUE, delta=1.0)          # 22 가정 -> 실제 12를 배움
            self.assertEqual(app.servos.get(A.FIRE_PIN, None), A.FIRE_READY)   # 발사 후 READY 복귀
            self.assertFalse(app.fire_busy())                                  # 평소 동작으로 돌아옴
            log = self.last_log(tmp)
            self.assertTrue(log["result"].startswith("fired"))
            self.assertEqual((log["judge"], log["dist_est"]), ("enemy 1", "1.50"))

    def test_shoots_enemy_not_friend(self):
        """적군(왼쪽) + 아군(오른쪽, 더 가운데 쪽이 아님): 적군에 맞춰 쏜다."""
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp, sides=("enemy", "friend"))
            self.assertEqual(self.run_auto(app), "fired")
            self.assert_on_aim(app, pos_of(0))
            self.assertGreater(abs(pos_of(1)[0] - 619.0), 150)                 # 아군은 조준점에서 멀다
            self.assertEqual(self.last_log(tmp)["judge"], "enemy 1, friend 1")

    def test_friend_only_does_not_move_or_fire(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp, sides=("friend",))
            self.assertEqual(app.auto_aim(app.cam.frame()), "no_enemy")
            self.assertIsNone(app.fire_thread)
            self.assertEqual((app.yaw(), app.pitch()), (95.0, 90.0))

    def test_unknown_is_shot(self):
        """07 규칙: 아군이 아니면 쏜다 -> 판정 못 한(unknown, few_px) 적군도 쏜다."""
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp, sides=("unknown",))
            self.assertEqual(self.run_auto(app), "fired")
            self.assert_on_aim(app, pos_of(0))

    def test_unknown_and_friend_shoots_unknown(self):
        """화면 그대로: 적군이 few_px(unknown) + 아군 FRIEND -> unknown 쪽을 쏜다."""
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp, sides=("unknown", "friend"))
            self.assertEqual(self.run_auto(app), "fired")
            self.assert_on_aim(app, pos_of(0))
            self.assertGreater(abs(pos_of(1)[0] - 619.0), 150)
            self.assertEqual(self.last_log(tmp)["judge"], "friend 1, unknown 1 (few_px)")

    def test_no_color_classifier_shoots_any(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp, color=False)
            self.assertEqual(self.run_auto(app), "fired")                       # 판정 안 된 박스 = unknown = 아군 아님

    def test_target_turns_friend_while_tracking(self):
        """처음엔 적군으로 보였는데 가까이 가서 다시 보니 아군 -> 멈추고 안 쏜다."""
        with tempfile.TemporaryDirectory() as tmp:
            seen = {"n": 0}

            def flip(side):
                seen["n"] += 1
                return side if seen["n"] <= 1 else "friend"
            app, _ = self.make_world(tmp, side_fn=flip)
            self.assertEqual(self.run_auto(app), "not_enemy")
            self.assertIsNone(app.fire_thread)

    def test_reversed_pitch_servo_learns(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp, k_pitch=-self.K_TRUE, px_per_deg=12.0)
            self.assertEqual(self.run_auto(app), "fired")
            self.assert_on_aim(app, pos_of(0), slack=20)
            self.assertLess(app.k_pitch, 0)

    def test_far_distance_does_not_fire(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp)
            far = det(800, 450, 22.0)                                          # 51.6/22 = 2.35 m
            app.detector = SimpleNamespace(detect=lambda img, roi=None: [far])
            app.check_distance = True
            self.assertEqual(app.auto_aim(app.cam.frame()), "no_aim_point")
            self.assertIsNone(app.fire_thread)

    def test_default_ignores_distance(self):
        """기본(거리 확인 안 함): 박스가 작아(2.35 m로 계산) 보여도 적군이면 선택 거리 조준점으로 쏜다."""
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp)
            self.assertFalse(app.check_distance)
            inner = app.detector.detect
            app.detector = SimpleNamespace(detect=lambda img, roi=None: [
                det(d.cx, d.cy, 22.0) for d in inner(img, roi)])
            self.assertEqual(self.run_auto(app), "fired")
            self.assert_on_aim(app, pos_of(0))
            self.assertIn("not checked", self.last_log(tmp)["note"])

    def test_servo_limit_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp)
            app.yaw_lo = app.yaw_hi = 95.0                                     # yaw를 못 움직이게
            app.args.center_pitch = 90.0
            old = A.AUTO_PITCH_LIMIT
            A.AUTO_PITCH_LIMIT = 0.0                                           # pitch도
            try:
                self.assertEqual(self.run_auto(app), "limit")
            finally:
                A.AUTO_PITCH_LIMIT = old
            self.assertIsNone(app.fire_thread)

    def test_h_coef_option(self):
        """기록에서 나온 계수(51.6)로는 h 49.6 px가 1.04 m -> 거부. --h-coef 74.4면 1.5 m -> 쏜다."""
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp)
            big = det(800, 450, 49.6)
            app.check_distance = True
            app.detector = SimpleNamespace(detect=lambda img, roi=None: [big])
            self.assertEqual(app.auto_aim(app.cam.frame()), "no_aim_point")
            app.h_coef = 49.6 * 1.5
            app.detector = SimpleNamespace(detect=lambda img, roi=None: [det(big.cx, big.cy, 49.6)])
            res = app.auto_aim(app.cam.frame())
            self.assertNotEqual(res, "no_aim_point")
            self.assertEqual(self.last_log(tmp)["dist_est"], "1.50")


    def test_e_after_auto_shot_saves_source_auto(self):
        """p로 쏜 뒤 e -> 발사 순간의 박스 중앙·h·서보 각도·프레임이 source=auto로 저장된다."""
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp)
            self.assertEqual(self.run_auto(app), "fired")
            a = app.last_auto
            self.assertIsNotNone(a)
            yaw, pitch = app.yaw(), app.pitch()
            # 맞아서 타겟이 사라진 상황: e 시점엔 검출이 안 된다 -> 그래도 발사 순간 값으로 저장
            calls = []
            app.detector = SimpleNamespace(detect=lambda img, roi=None: calls.append(1) or [])
            n0 = app.store.count()
            self.assertEqual(app.begin_record(app.cam.frame()), "recorded")
            self.assertEqual(calls, [])                                        # e에서 다시 검출하지 않는다
            r = app.store.rows[-1]
            self.assertEqual(app.store.count(), n0 + 1)
            self.assertEqual(r["source"], "auto")
            self.assertAlmostEqual(r["cx"], round(a["cx"], 1)); self.assertAlmostEqual(r["cy"], round(a["cy"], 1))
            self.assertEqual((r["h_px"], r["yaw_cmd"], r["pitch_cmd"]), (34.4, round(yaw, 1), round(pitch, 1)))
            png = app.store.shots / r["png"]
            self.assertTrue(np.array_equal(cv2.imread(str(png)), a["frame"]))  # 발사 순간 원본 프레임
            self.assertIsNone(app.last_auto)                                   # 한 번만 저장
            self.assertEqual(app.begin_record(app.cam.frame()), "click")       # 다시 e -> 평소 동작 (박스 없음 -> 클릭 대기)
            app.cancel_pending()

    def test_auto_record_cleared_by_move_and_marks_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp)
            self.assertEqual(self.run_auto(app), "fired")
            app.on_click(100, 100)                                             # 클릭 표시가 있으면 클릭 기록이 우선
            self.assertEqual(app.begin_record(app.cam.frame()), "recorded")
            self.assertEqual(app.store.rows[-1]["source"], "click")
            self.assertEqual(self.run_auto(app), "fired")
            app.move(2.0, 0.0)                                                 # 서보가 움직이면 auto 대기 사라짐
            self.assertIsNone(app.last_auto)
            self.assertEqual(app.begin_record(app.cam.frame()), "recorded")
            self.assertEqual(app.store.rows[-1]["source"], "detect")           # 평소 e (지금 프레임 검출)

    def test_no_auto_record_when_not_fired(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp, sides=("friend",))
            self.assertEqual(app.auto_aim(app.cam.frame()), "no_enemy")
            self.assertIsNone(app.last_auto)

    def test_click_records_are_converted_to_box_center(self):
        """클릭(머리 자리) 기록은 박스 중앙 기준으로 바뀐다: + 0.35 x h 아래. detect/auto 기록은 그대로."""
        rows = [dict(distance_m=1.5, cx=650.0, cy=564.0, h_px=None, source="click"),
                dict(distance_m=1.5, cx=650.0, cy=564.0, h_px=40.0, source="click"),
                dict(distance_m=1.5, cx=650.0, cy=564.0, h_px=40.0, source="auto")]
        ax, ay, _, _ = A.aim_point_for(rows, 1.5)
        self.assertEqual(ax, 650.0)
        self.assertAlmostEqual(ay, 564.0 + 0.35 * 40.0)           # 중앙값 (564+14, 564+14, 564)
        only_click = [dict(distance_m=1.5, cx=650.0, cy=564.0, h_px=None, source="click")]
        self.assertAlmostEqual(A.aim_point_for(only_click, 1.5)[1], 564.0 + 0.35 * A.expected_h(1.5))

    def test_e_skips_friend_box(self):
        """e: 박스가 여러 개면 아군 박스는 고르지 않는다 (10-06에 아군 박스가 명중 기록으로 저장됨)."""
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp, sides=("friend", "enemy"))
            self.assertEqual(app.begin_record(app.cam.frame()), "recorded")
            r = app.store.rows[-1]
            self.assertEqual(r["source"], "detect")
            self.assertAlmostEqual(r["cx"], round(pos_of(1)[0], 1), delta=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp, sides=("friend",))
            self.assertEqual(app.begin_record(app.cam.frame()), "click")       # 아군만 -> 저장 안 하고 클릭 대기
            app.cancel_pending()

    def test_keeps_moving_until_within_tol_on_coarse_servo(self):
        """서보가 거칠게(0.3도 격자) 움직이고 px/도가 가정(12)과 다른(16.7) 경우: 넘어설 때마다 폭을 줄여
        6 px 안에 들 때까지 계속 움직인 뒤 쏜다 (횟수 제한 없음)."""
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp, k_yaw=16.7, k_pitch=16.7, px_per_deg=12.0)
            orig = app.servos.goto
            app.servos.goto = lambda pin, angle, settle=None: orig(pin, round(angle / 0.3) * 0.3, settle)
            self.assertEqual(self.run_auto(app), "fired")
            log = self.last_log(tmp)
            self.assertEqual(log["result"], "fired_converged")
            self.assertLessEqual(abs(float(log["err_x"])), app.auto_tol)
            self.assertLessEqual(abs(float(log["err_y"])), app.auto_tol)

    def test_overshoot_halves_step_until_within_tol(self):
        """px/도를 실제(12)의 절반(6)으로 잘못 알고 학습도 안 되면 매번 2배로 넘어선다 -> 넘어설 때마다 폭을 줄여 결국 6 px 안."""
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp, k_yaw=12.0, k_pitch=12.0, px_per_deg=6.0)
            old = A.learn_gain
            A.learn_gain = lambda k, *a: k                                     # 학습 끔
            try:
                self.assertEqual(self.run_auto(app), "fired")
            finally:
                A.learn_gain = old
            log = self.last_log(tmp)
            self.assertLessEqual(abs(float(log["err_x"])), app.auto_tol)
            self.assertLessEqual(abs(float(log["err_y"])), app.auto_tol)
            self.assertGreater(int(log["iters"]), 2)                           # 오가면서 좁혀 들어갔다


    def test_flaky_detection_retries_and_fires(self):
        """먼 타겟처럼 검출이 가끔 빠져도(호출 3번 중 1번 빈 결과) 다시 찾아서 끝까지 조준하고 쏜다."""
        with tempfile.TemporaryDirectory() as tmp:
            app, pos_of = self.make_world(tmp)
            inner, n = app.detector.detect, {"i": 0}

            def flaky(img, roi=None):
                n["i"] += 1
                return [] if n["i"] % 3 == 0 else inner(img, roi)
            app.detector = SimpleNamespace(detect=flaky)
            self.assertEqual(self.run_auto(app), "fired")
            self.assert_on_aim(app, pos_of(0))

    def test_lost_target_does_not_fire(self):
        """이동 뒤 타겟이 계속 안 보이면(3번 다시 찾아도 없음) 쏘지 않는다."""
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = self.make_world(tmp)
            inner, n = app.detector.detect, {"i": 0}

            def gone(img, roi=None):
                n["i"] += 1
                return inner(img, roi) if n["i"] == 1 else []          # 첫 전체 검출만 보이고 그 뒤로 사라짐
            app.detector = SimpleNamespace(detect=gone)
            self.assertEqual(self.run_auto(app), "not_enemy")
            self.assertIsNone(app.fire_thread)
            self.assertIn("lost", self.last_log(tmp)["note"])
