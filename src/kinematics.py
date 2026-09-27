"""Step 2: turn raw tracks into smoothed positions and velocities."""
import numpy as np
import pandas as pd

from .detection import TWO_WHEELERS, VEHICLES
from .geometry import containment


def add_kinematics(df: pd.DataFrame, smooth_s: float = 0.6, min_points: int = 5) -> pd.DataFrame:
    """Adds gx, gy (ground point = bottom-centre), h, sx, sy, vx, vy, speed, rspeed.

    rspeed = speed / box height  -> "box heights per second", roughly perspective independent.
    """
    cols = ["gx", "gy", "h", "sx", "sy", "vx", "vy", "speed", "rspeed"]
    if df.empty:
        return df.assign(**{c: [] for c in cols})
    df = df.sort_values(["tid", "t"]).reset_index(drop=True)
    df["gx"] = (df.x1 + df.x2) / 2
    df["gy"] = df.y2.astype(float)
    df["h"] = (df.y2 - df.y1).clip(lower=1.0)
    parts = []
    for _, g in df.groupby("tid", sort=False):
        if len(g) < min_points:
            continue
        g = g.copy()
        dt = float(np.median(np.diff(g.t.values)))
        k = max(1, int(round(smooth_s / max(dt, 1e-6))))
        roll = lambda s: s.rolling(k, center=True, min_periods=1).mean()
        g["sx"], g["sy"], hs = roll(g.gx), roll(g.gy), roll(g.h)
        t = g.t.values
        g["vx"] = np.gradient(g.sx.values, t)
        g["vy"] = np.gradient(g.sy.values, t)
        g["speed"] = np.hypot(g.vx, g.vy)
        g["rspeed"] = g.speed / hs
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].assign(**{c: [] for c in cols})


def mark_riders(df: pd.DataFrame, min_overlap: float = 0.3) -> pd.DataFrame:
    """Flag persons sitting on bicycles/motorcycles (they are not pedestrians)."""
    df = df.copy()
    df["rider"] = False
    if df.empty:
        return df
    two = df[df.cls.isin(TWO_WHEELERS)]
    if two.empty:
        return df
    boxes_by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in two.groupby("frame")}
    persons = df.index[df.cls == "person"]
    for i in persons:
        r = df.loc[i]
        for b in boxes_by_frame.get(r.frame, []):
            # rider box overlaps the bike box (bike roughly under the person)
            if containment(b, (r.x1, r.y1, r.x2, r.y2 + 0.5 * (r.y2 - r.y1))) > min_overlap:
                df.at[i, "rider"] = True
                break
    return df


def vehicles(df):
    return df[df.cls.isin(VEHICLES)]


def pedestrians(df):
    return df[(df.cls == "person") & (~df.get("rider", False))]


def bool_runs(t, mask, max_gap: float = 0.5):
    """Contiguous True runs in a time series -> [(t_start, t_end)].
    False samples shorter than max_gap inside a run are tolerated."""
    runs, start, last = [], None, None
    for ti, m in zip(t, mask):
        if m:
            if start is None:
                start = ti
            elif ti - last > max_gap:
                runs.append((start, last))
                start = ti
            last = ti
        elif start is not None and ti - last > max_gap:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    return runs
