"""Vehicle stops past the stop line on red without entering the intersection."""
import numpy as np

from ..geometry import points_in_poly
from ..kinematics import bool_runs, vehicles
from ._stopline import crossings_towards_intersection
from .base import ev

LABEL = "stop_line"


def detect(ctx, p):
    line, inter = ctx.scene.stop_line, ctx.scene.poly("intersection")
    if line is None or inter is None or ctx.scene.light_box is None:
        return []
    out = []
    for _, g in vehicles(ctx.tracks).groupby("tid"):
        t = g.t.values
        inside = points_in_poly(inter, g.sx, g.sy)
        for i in crossings_towards_intersection(g, line, inter):
            if ctx.light(t[i]) != "red":
                continue
            stopped = (g.rspeed.values[i:] < p["max_rspeed"]) & ~inside[i:]
            runs = bool_runs(t[i:], stopped, 0.5)
            runs = [r for r in runs if r[1] - r[0] >= p["min_stop"]]
            if runs:
                s = runs[0][0]
                out.append(ev(s, ctx.next_green_after(s), LABEL))
    return out
