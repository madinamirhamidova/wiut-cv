"""Part B: causal accident-risk estimator based on time-to-collision (TTC).

Sees frames one at a time. Runs its own detector+tracker (it may NOT reuse Part A output).
"""
from collections import defaultdict, deque
from itertools import combinations

import numpy as np

from .config import load_params, set_seed
from .detection import COCO, VEHICLES, load_model, resolve_device


class RiskEstimator:
    def __init__(self):
        self.params = load_params()
        self.cfg = self.params["risk"]
        self.model = None

    def reset(self, meta: dict) -> None:
        set_seed(self.params["seed"])
        self.model = load_model(self.params["detector"])   # fresh model = fresh tracker state
        self.stride = int(self.cfg["stride"])
        self.i = 0
        self.score = 0.0
        self.hist = defaultdict(lambda: deque(maxlen=64))   # tid -> (t, gx, gy, h, cls)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        i, self.i = self.i, self.i + 1
        if i % self.stride:
            return self.score
        det = self.params["detector"]
        res = self.model.track(frame, persist=True, tracker=det["tracker"], classes=list(COCO),
                               conf=det["conf"], imgsz=self.cfg["imgsz"], device=resolve_device(det.get("device")),
                               verbose=False)[0]
        b = res.boxes
        if b is not None and b.id is not None:
            for (x1, y1, x2, y2), tid, c in zip(b.xyxy.cpu().numpy(), b.id.int().cpu().numpy(),
                                                b.cls.int().cpu().numpy()):
                self.hist[int(tid)].append((t_sec, (x1 + x2) / 2, y2, max(1.0, y2 - y1), COCO[int(c)]))
        raw = self._ttc_risk(t_sec)
        a = self.cfg["ema"]
        self.score = float(np.clip(a * raw + (1 - a) * self.score, 0.0, 1.0))
        return self.score

    def _states(self, now):
        """Position/velocity for objects seen in the last second (linear fit over history_s)."""
        out = []
        for tid in list(self.hist):
            h = [r for r in self.hist[tid] if now - r[0] <= self.cfg["history_s"]]
            if not h:
                if now - self.hist[tid][-1][0] > 2.0:
                    del self.hist[tid]
                continue
            if len(h) < 3 or h[-1][0] < now - 0.3:
                continue
            t = np.array([r[0] for r in h])
            x = np.array([r[1] for r in h])
            y = np.array([r[2] for r in h])
            vx, vy = np.polyfit(t, x, 1)[0], np.polyfit(t, y, 1)[0]
            out.append((np.array([x[-1], y[-1]]), np.array([vx, vy]), h[-1][3], h[-1][4]))
        return out

    def _ttc_risk(self, now) -> float:
        H, tau = self.cfg["horizon"], self.cfg["tau"]
        w = np.array([1.0, self.cfg["depth_weight"]])   # image y ~ depth: a small y gap is a large real gap
        risk = 0.0
        for (pa, va, ha, ca), (pb, vb, hb, cb) in combinations(self._states(now), 2):
            if ca not in VEHICLES and cb not in VEHICLES:
                continue                      # person-person is irrelevant
            scale = (ha + hb) / 2
            p, v = (pb - pa) * w, (vb - va) * w
            vv = float(v @ v)
            if np.sqrt(vv) / scale < self.cfg["min_rel_speed"]:
                continue
            t_star = -float(p @ v) / vv          # time of closest approach
            if not 0 < t_star <= H:
                continue
            d_min = np.linalg.norm(p + v * t_star)
            r = self.cfg["collision_radius"] * scale
            if d_min < r:
                risk = max(risk, np.exp(-t_star / tau) * (0.5 + 0.5 * (1 - d_min / r)))
        return risk
