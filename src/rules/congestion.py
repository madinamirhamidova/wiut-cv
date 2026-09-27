"""Traffic standstill/crawling across all lanes of a direction (lane 'group')."""
import numpy as np

from ..geometry import points_in_poly
from ..kinematics import bool_runs, vehicles
from .base import ev

LABEL = "congestion"


def detect(ctx, p):
    v = vehicles(ctx.tracks)
    if v.empty:
        return []
    groups = {}
    for lane in ctx.scene.lanes:
        groups.setdefault(lane.group, []).append(lane.polygon)
    if not groups and ctx.scene.poly("road") is not None:
        groups = {"road": [ctx.scene.poly("road")]}
    out = []
    bins = np.floor(v.t.values / p["bin_s"]).astype(int)
    for polys in groups.values():
        inside = np.zeros(len(v), bool)
        for poly in polys:
            inside |= points_in_poly(poly, v.gx, v.gy)
        sub = v[inside].assign(b=bins[inside])
        if sub.empty:
            continue
        agg = sub.groupby("b").agg(n=("tid", "nunique"), med=("rspeed", "median"))
        all_bins = np.arange(0, int(ctx.duration / p["bin_s"]) + 1)
        agg = agg.reindex(all_bins, fill_value=0)
        mask = (agg.n.values >= p["min_vehicles"]) & (agg.med.values < p["max_rspeed"])
        t = all_bins * p["bin_s"]
        for s, e in bool_runs(t, mask, p["max_gap"]):
            if e - s >= p["min_duration"]:
                out.append(ev(s, e + p["bin_s"], LABEL))
    return out
