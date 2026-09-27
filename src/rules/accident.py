"""Heuristic collision: two road users overlap at the same depth while closing in on each other,
then both come to a stop (or leave the frame) shortly after."""
import numpy as np

from ..geometry import box_iou, points_in_any
from ..kinematics import vehicles
from .base import ev

LABEL = "accident"


def detect(ctx, p):
    v = vehicles(ctx.tracks)
    if v.empty:
        return []
    by_tid = {tid: g for tid, g in v.groupby("tid")}
    queues = ctx.scene.polys("queue_zones")
    seen, out = set(), []
    for _, f in v.groupby("frame"):
        rows = f.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                pair = (min(a["tid"], b["tid"]), max(a["tid"], b["tid"]))
                if pair in seen:
                    continue
                depth = abs(a["gy"] - b["gy"]) / max(1.0, (a["h"] + b["h"]) / 2)
                if depth > p["max_depth_ratio"]:
                    continue       # different distance from camera -> just occlusion
                iou = box_iou((a["x1"], a["y1"], a["x2"], a["y2"]), (b["x1"], b["y1"], b["x2"], b["y2"]))
                if iou < p["min_iou"]:
                    continue
                t0 = a["t"]
                ga, gb = by_tid[a["tid"]], by_tid[b["tid"]]
                before = lambda g: g.rspeed[(g.t >= t0 - 1.0) & (g.t <= t0)].max()
                if max(before(ga), before(gb)) < p["min_rspeed_before"]:
                    continue
                if _closing_speed(ga, gb, t0, (a["h"] + b["h"]) / 2) < p["min_closing_rspeed"]:
                    continue       # moving together: duplicate detection of one vehicle (bus halves)
                if queues and ctx.light(t0) != "green" and points_in_any(queues, [a["gx"]], [a["gy"]])[0]:
                    continue       # pulling up to a queue at the signal
                ends = [_end_after(g, t0, p, ctx.meta) for g in (ga, gb)]
                if any(e is None for e in ends) or not any(still for _, still in ends):
                    continue       # someone kept driving, or both just drove out of view -> not a crash
                seen.add(pair)
                out.append(ev(t0, max(max(t for t, _ in ends), t0 + 1.0), LABEL))
    return out


def _closing_speed(ga, gb, t0, h):
    """How fast the two ground points approached each other over the last second (box heights / s)."""
    def pos(g, t):
        return np.array([np.interp(t, g.t, g.sx), np.interp(t, g.t, g.sy)])
    t1 = max(t0 - 1.0, ga.t.iloc[0], gb.t.iloc[0])
    if t0 - t1 < 0.2:
        return 0.0
    d0 = np.linalg.norm(pos(ga, t1) - pos(gb, t1))
    d1 = np.linalg.norm(pos(ga, t0) - pos(gb, t0))
    return (d0 - d1) / (t0 - t1) / max(h, 1.0)


def _end_after(g, t0, p, meta):
    """(time, stopped?) when the object stays stationary or leaves the frame within max_stop_delay, else None."""
    after = g[(g.t > t0) & (g.t <= t0 + p["max_stop_delay"])]
    stopped = after.t[after.rspeed < p["stop_rspeed"]]
    if len(stopped):
        ts = float(stopped.iloc[0])
        still = g[(g.t >= ts) & (g.t <= ts + p["min_still_s"])]
        # crashed vehicles stay put; traffic that merely slowed down moves on
        if still.t.iloc[-1] - ts >= p["min_still_s"] - 0.2 and (still.rspeed < p["stop_rspeed"]).all():
            return ts, True
        return None
    if g.t.iloc[-1] <= t0 + p["max_stop_delay"] and _at_border(g.iloc[-1], meta, p["border_frac"]):
        return float(g.t.iloc[-1]), False      # left the frame (a track ending mid-frame is an ID switch)
    return None


def _at_border(r, meta, frac):
    w, h = meta.get("width"), meta.get("height")
    if not w or not h:
        return True
    mx, my = frac * w, frac * h
    return r.x1 < mx or r.y1 < my or r.x2 > w - mx or r.y2 > h - my
