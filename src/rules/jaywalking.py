"""Pedestrian on the carriageway outside a crossing or a refuge island."""
from ..geometry import expand_poly, points_in_any, points_in_poly
from ..kinematics import bool_runs, pedestrians
from .base import ev

LABEL = "jaywalking"


def detect(ctx, p):
    road = ctx.scene.poly("road")
    if road is None:
        return []
    crossings = [expand_poly(c, p["crosswalk_margin_px"])
                 for c in ctx.scene.polys("crosswalks") + ctx.scene.polys("islands")]
    out = []
    for _, g in pedestrians(ctx.tracks).groupby("tid"):
        mask = points_in_poly(road, g.gx, g.gy) & ~points_in_any(crossings, g.gx, g.gy)
        for s, e in bool_runs(g.t.values, mask, p["max_gap"]):
            if e - s >= p["min_duration"]:
                out.append(ev(s, e, LABEL))
    return out
