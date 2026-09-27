"""Vehicle crosses the stop line on red and enters the intersection."""
import numpy as np

from ..geometry import points_in_poly
from ..kinematics import vehicles
from ._stopline import crossings_towards_intersection
from .base import ev

LABEL = "red_light"


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
            after = np.where(inside[i:])[0]
            if not len(after):              # never entered -> stop_line case, not red_light
                continue
            j = i + after[0]
            left = np.where(~inside[j:])[0]
            end = t[j + left[0]] if len(left) else t[-1]
            out.append(ev(t[i], max(end, t[i] + 0.5), LABEL))
    return out
