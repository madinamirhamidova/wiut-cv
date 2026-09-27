"""Vehicle drives through a crossing while a pedestrian is on it."""
import numpy as np

from ..geometry import points_in_poly
from ..kinematics import bool_runs, pedestrians, vehicles
from .base import ev

LABEL = "failure_to_yield"


def detect(ctx, p):
    out = []
    peds = pedestrians(ctx.tracks)
    for cw in ctx.scene.polys("crosswalks"):
        ped_t = np.sort(peds.t.values[points_in_poly(cw, peds.gx, peds.gy)])
        if not len(ped_t):
            continue
        for _, g in vehicles(ctx.tracks).groupby("tid"):
            mask = points_in_poly(cw, g.gx, g.gy) & (g.rspeed.values > 0.2)
            for s, e in bool_runs(g.t.values, mask, p["max_gap"]):
                lo, hi = np.searchsorted(ped_t, [s - 0.5, e])
                if hi > lo:
                    out.append(ev(s, max(e, s + 0.5), LABEL))
    return out
