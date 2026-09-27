"""Vehicle moving against the direction of the lane it is in."""
import numpy as np

from ..geometry import points_in_poly
from ..kinematics import bool_runs, vehicles
from .base import ev

LABEL = "wrong_way"


def detect(ctx, p):
    lanes = ctx.scene.lanes
    if not lanes:
        return []
    out = []
    for _, g in vehicles(ctx.tracks).groupby("tid"):
        vel = np.stack([g.vx.values, g.vy.values], 1)
        unit = vel / np.linalg.norm(vel, axis=1, keepdims=True).clip(min=1e-9)
        wrong = np.zeros(len(g), bool)
        for lane in lanes:
            inside = points_in_poly(lane.polygon, g.gx, g.gy)
            wrong |= inside & (unit @ lane.direction < p["max_cos"])
        wrong &= g.rspeed.values > p["min_rspeed"]
        for s, e in bool_runs(g.t.values, wrong, p["max_gap"]):
            if e - s >= p["min_duration"]:
                out.append(ev(s, e, LABEL))
    return out
