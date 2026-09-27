"""Vehicle trajectory crosses a solid lane marking."""
from ..geometry import segments_intersect
from ..kinematics import vehicles
from .base import ev

LABEL = "solid_line_crossing"


def detect(ctx, p):
    lines = ctx.scene.lines("solid_lines")
    if not lines:
        return []
    out = []
    for _, g in vehicles(ctx.tracks).groupby("tid"):
        xs, ys, t, rs = g.sx.values, g.sy.values, g.t.values, g.rspeed.values
        for i in range(1, len(g)):
            if rs[i] < p["min_rspeed"]:
                continue
            for a, b in lines:
                if segments_intersect((xs[i - 1], ys[i - 1]), (xs[i], ys[i]), a, b):
                    out.append(ev(max(0, t[i] - p["pre_s"]), t[i] + p["post_s"], LABEL))
    return out
