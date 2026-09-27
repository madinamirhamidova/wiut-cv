"""Sanity tests for rules on synthetic tracks (no video / YOLO needed).
Run: python -m pytest tests  (or: python tests/test_rules_synthetic.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import postprocess  # noqa: E402
from src.config import load_params  # noqa: E402
from src.kinematics import add_kinematics, mark_riders  # noqa: E402
from src.rules import RULES  # noqa: E402
from src.rules.base import Context  # noqa: E402
from src.scene import Scene  # noqa: E402

DT = 0.08  # 12.5 processed fps
SCENE = Scene({
    "road": [[0, 0], [1000, 0], [1000, 600], [0, 600]],
    "lanes": [
        {"polygon": [[0, 300], [1000, 300], [1000, 600], [0, 600]], "direction": [[0, 450], [100, 450]]},
        {"polygon": [[0, 0], [1000, 0], [1000, 300], [0, 300]], "direction": [[100, 150], [0, 150]]},
    ],
    "crosswalks": [[[700, 0], [760, 0], [760, 600], [700, 600]]],
    "stop_line": [[600, 300], [600, 600]],
    "intersection": [[620, 300], [900, 300], [900, 600], [620, 600]],
    "light_box": [0, 0, 5, 5],
    "solid_lines": [[[0, 300], [400, 300]]],
})


def track(tid, cls, t0, t1, fx, fy, w=60, h=40):
    t = np.arange(t0, t1, DT)
    x, y = fx(t - t0), fy(t - t0)
    return pd.DataFrame({"frame": np.round(t / 0.04).astype(int), "t": t, "tid": tid, "cls": cls, "conf": 0.9,
                         "x1": x - w / 2, "y1": y - h, "x2": x + w / 2, "y2": y})


def run(tracks, lights, duration=120.0):
    params = load_params()
    k = mark_riders(add_kinematics(pd.concat(tracks, ignore_index=True)))
    ctx = Context(k, lights, SCENE, {"duration": duration, "fps": 25}, params)
    raw = []
    for name in params["classes_enabled"]:
        raw += RULES[name].detect(ctx, params["rules"][name])
    return postprocess.clean(raw, duration, params["postprocess"])


def labels(events):
    return {e[2] for e in events}


LIGHTS_RED = pd.DataFrame({"frame": [0, 1500], "t": [0.0, 60.0], "state": ["red", "green"]})
const = lambda v: (lambda t: np.full_like(t, v, dtype=float))


def test_stopped_vehicle():
    ev = run([track(1, "car", 0, 20, const(300), const(450))], LIGHTS_RED)
    assert "stopped_vehicle" in labels(ev), ev


def test_wrong_way():
    ev = run([track(2, "car", 70, 78, lambda t: 500 - 50 * t, const(500))], LIGHTS_RED)
    assert "wrong_way" in labels(ev), ev


def test_jaywalking_and_not_on_crosswalk():
    ev = run([track(3, "person", 70, 75, lambda t: 200 + 10 * t, const(150), 20, 50)], LIGHTS_RED)
    assert "jaywalking" in labels(ev), ev
    ev = run([track(3, "person", 70, 75, const(730), lambda t: 100 + 20 * t, 20, 50)], LIGHTS_RED)
    assert "jaywalking" not in labels(ev), ev


def test_red_light():
    ev = run([track(4, "car", 30, 40, lambda t: 300 + 60 * t, const(400))], LIGHTS_RED)
    reds = [e for e in ev if e[2] == "red_light"]
    assert reds and 34 < reds[0][0] < 36, ev          # crosses x=600 at t=35


def test_no_red_light_on_green():
    ev = run([track(4, "car", 70, 80, lambda t: 300 + 60 * t, const(400))], LIGHTS_RED)
    assert "red_light" not in labels(ev), ev


def test_stop_line():
    fx = lambda t: np.where(t < 3, 450 + 50 * t, 605)  # stops just past the line
    ev = run([track(5, "car", 10, 30, fx, const(450))], LIGHTS_RED)
    assert "stop_line" in labels(ev) and "red_light" not in labels(ev), ev


def test_solid_line_crossing():
    ev = run([track(6, "car", 70, 76, lambda t: 100 + 30 * t, lambda t: 250 + 20 * t)], LIGHTS_RED)
    assert "solid_line_crossing" in labels(ev), ev


def test_quiet_traffic_has_no_events():
    ev = run([track(7, "car", 70, 80, lambda t: 50 + 80 * t, const(450))], LIGHTS_RED)
    assert ev == [], ev


def test_accident():
    a = track(8, "car", 70, 85, lambda t: np.where(t < 5, 300 + 60 * t, 600), const(450))   # rams B
    b = track(9, "car", 70, 85, lambda t: np.where(t < 5, 640.0, 640 + 3 * np.minimum(t - 5, 2)), const(452))
    ev = run([a, b], LIGHTS_RED)
    acc = [e for e in ev if e[2] == "accident"]
    assert acc and 73 < acc[0][0] < 76, ev


def test_light_flicker_is_debounced():
    from src.light_state import light_series
    t = np.arange(0, 10, 0.1)
    st = np.where(t < 5, "green", "red").astype(object)
    st[20:23] = "red"                                      # 0.3 s blip inside the green phase
    _, out = light_series(pd.DataFrame({"t": t, "state": st}), min_phase_s=2.0)
    assert (out[t < 5] == "green").all() and (out[t >= 5] == "red").all()


def test_scene_scales_to_video_size():
    sc = Scene({"frame_size": [1000, 600], "light_box": [100, 60, 200, 120],
                "lanes": [{"polygon": [[0, 0], [1000, 600], [0, 600]], "direction": [[0, 0], [10, 0]], "group": "a"}]})
    half = sc.for_size(500, 300)
    assert half.light_box == [50, 30, 100, 60]
    assert half.lanes[0].polygon.tolist() == [[0, 0], [500, 300], [0, 300]] and half.lanes[0].group == "a"
    assert sc.for_size(1000, 600) is sc and SCENE.for_size(10, 10) is SCENE     # same size / no frame_size


def test_risk_ttc_rises_before_collision():
    from src.risk import RiskEstimator
    r = RiskEstimator.__new__(RiskEstimator)
    r.params = load_params(); r.cfg = r.params["risk"]
    from collections import defaultdict, deque
    r.hist = defaultdict(lambda: deque(maxlen=64))
    for t in np.arange(0, 1.0, DT):                      # head-on, 2 s apart
        r.hist[1].append((t, 300 + 100 * t, 450, 40, "car"))
        r.hist[2].append((t, 700 - 100 * t, 450, 40, "car"))
        r.hist[3].append((t, 100 + 100 * t, 150, 40, "car"))   # parallel, no risk
    assert r._ttc_risk(0.96) > 0.3
    del r.hist[1]
    assert r._ttc_risk(0.96) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
