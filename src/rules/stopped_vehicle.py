"""Vehicle stationary on the carriageway >= 10 s, not queueing at a red signal."""
from ..geometry import points_in_any, points_in_poly
from ..kinematics import bool_runs, vehicles
from .base import ev

LABEL = "stopped_vehicle"


def detect(ctx, p):
    road = ctx.scene.poly("road")
    queues = ctx.scene.polys("queue_zones")
    out = []
    for _, g in vehicles(ctx.tracks).groupby("tid"):
        in_road = points_in_poly(road, g.gx, g.gy) if road is not None else True
        queued = points_in_any(queues, g.gx, g.gy) & (ctx.light(g.t.values) != "green")
        mask = (g.rspeed.values < p["max_rspeed"]) & in_road & ~queued
        for s, e in bool_runs(g.t.values, mask, p["max_gap"]):
            if e - s >= p["min_duration"]:
                out.append(ev(s, e, LABEL))
    return out
