"""U-turn inside a zone where U-turns are prohibited (heading change >= ~150 deg)."""
import numpy as np

from ..geometry import points_in_poly
from ..kinematics import vehicles
from .base import ev

LABEL = "illegal_u_turn"


def detect(ctx, p):
    zones = ctx.scene.polys("no_u_turn_zones")
    if not zones:
        return []
    out = []
    for _, g in vehicles(ctx.tracks).groupby("tid"):
        for z in zones:
            m = points_in_poly(z, g.sx, g.sy) & (g.rspeed.values > p["min_rspeed"])
            if m.sum() < 5:
                continue
            t = g.t.values[m]
            heading = np.unwrap(np.arctan2(g.vy.values[m], g.vx.values[m]))
            turn = np.degrees(np.abs(heading - heading[0]))
            if turn.max() < p["min_turn_deg"]:
                continue
            i0 = np.argmax(turn >= p["start_deg"])
            i1 = np.argmax(turn >= 0.95 * turn.max())
            if t[i1] > t[i0]:
                out.append(ev(t[i0], t[i1], LABEL))
    return out
