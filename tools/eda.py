"""EDA of the sample videos (uses the cached tracks; runs detection first if needed).

  python tools/eda.py            # -> outputs/eda/*.png, *_counts.csv, summary.json

Per video: resolution/fps/duration/codec, brightness over time, objects in view by class,
motion heatmap, trajectories, lane-direction flow field, traffic-light cycle, speed by class.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, sample_videos  # noqa: E402
from src.detection import VEHICLES, video_meta  # noqa: E402
from src.pipeline import build_context  # noqa: E402
from src.scene import draw_scene  # noqa: E402

OUT = ROOT / "outputs/eda"
FLOW_CELL = 160          # px, grid for the lane-direction field
MOVING_RSPEED = 0.5      # box heights / s


def frame_at(video, t):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    cap.release()
    return fr


def brightness(video, every_s=10):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vals = []
    for f in range(0, n, int(fps * every_s)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if ok:
            vals.append((round(f / fps, 1), round(float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean()), 1)))
    cap.release()
    return vals


def codec(video):
    cap = cv2.VideoCapture(str(video))
    c = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    cap.release()
    return c


def light_phases(ctx):
    """Durations of contiguous red / green phases (first and last phase are cut by the clip, dropped)."""
    t, st = ctx.light_timeline()
    runs, start = [], 0
    for i in range(1, len(st) + 1):
        if i == len(st) or st[i] != st[start]:
            runs.append((st[start], float(t[min(i, len(t) - 1)] - t[start])))
            start = i
    runs = runs[1:-1]
    return {s: [round(d, 1) for c, d in runs if c == s] for s in ("red", "green")}


def plot_counts(tracks, stem):
    per_sec = (tracks.assign(sec=tracks.t.astype(int)).groupby(["sec", "cls"]).tid.nunique()
               .unstack(fill_value=0))
    per_sec.to_csv(OUT / f"{stem}_counts.csv")
    fig, ax = plt.subplots(figsize=(10, 3))
    per_sec.rolling(5, min_periods=1).mean().plot(ax=ax)
    ax.set(title=f"{stem}: objects in view by class (5 s mean)", xlabel="s", ylabel="count")
    fig.tight_layout(); fig.savefig(OUT / f"{stem}_counts.png", dpi=110); plt.close(fig)
    return per_sec


def plot_motion(k, bg, stem):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax in axes:
        ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)); ax.axis("off")
    moving = k[k.rspeed > MOVING_RSPEED]
    axes[0].hexbin(moving.gx, moving.gy, gridsize=60, cmap="inferno", alpha=0.6, mincnt=1)
    for _, g in k.groupby("tid"):
        axes[1].plot(g.sx, g.sy, lw=0.5, alpha=0.5)
    axes[0].set_title("motion heatmap (moving objects)"); axes[1].set_title("trajectories")
    fig.tight_layout(); fig.savefig(OUT / f"{stem}_motion.png", dpi=110); plt.close(fig)


def plot_flow(k, bg, scene, stem):
    """Mean heading of moving vehicles per grid cell: shows the lane directions we put in scene.yaml."""
    img = draw_scene(bg.copy(), scene)
    m = k[k.cls.isin(VEHICLES) & (k.rspeed > MOVING_RSPEED)].copy()
    m["ux"], m["uy"] = m.vx / m.speed, m.vy / m.speed
    for (cx, cy), g in m.groupby([m.gx // FLOW_CELL, m.gy // FLOW_CELL]):
        if g.tid.nunique() < 5:
            continue
        u = np.array([g.ux.mean(), g.uy.mean()])
        if np.linalg.norm(u) < 0.5:             # mixed directions in this cell
            continue
        c = np.array([cx + 0.5, cy + 0.5]) * FLOW_CELL
        cv2.arrowedLine(img, tuple(np.int32(c)), tuple(np.int32(c + u / np.linalg.norm(u) * 70)),
                        (0, 0, 255), 6, tipLength=0.4)
    cv2.imwrite(str(OUT / f"{stem}_flow.jpg"), cv2.resize(img, (1280, int(1280 * img.shape[0] / img.shape[1]))))


def plot_light_and_speed(ctx, stem):
    t, st = ctx.light_timeline()
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    colors = {"red": "tab:red", "green": "tab:green", "unknown": "lightgray"}
    for i in range(len(t) - 1):
        axes[0].axvspan(t[i], t[i + 1], color=colors[st[i]], lw=0)
    axes[0].set(title="traffic light state (light_box)", xlabel="s", yticks=[])
    v = ctx.tracks[ctx.tracks.rspeed > 0.1]
    for cls, g in v.groupby("cls"):
        axes[1].hist(g.rspeed.clip(upper=6), bins=60, histtype="step", density=True, label=cls)
    axes[1].set(title="speed of moving objects", xlabel="box heights / s")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f"{stem}_light_speed.png", dpi=110); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for v in sample_videos():
        stem, meta = v.stem, video_meta(str(v))
        ctx = build_context(str(v))
        k, bg = ctx.tracks, frame_at(v, 5.0)
        per_sec = plot_counts(k, stem)
        plot_motion(k, bg, stem)
        plot_flow(k, bg, ctx.scene, stem)
        plot_light_and_speed(ctx, stem)
        veh = per_sec[[c for c in per_sec.columns if c in VEHICLES]].sum(axis=1)
        summary[v.name] = {
            **{key: meta[key] for key in ("width", "height", "fps", "duration")},
            "codec": codec(v),
            "brightness": brightness(v),
            "n_tracks_by_class": k.groupby("cls").tid.nunique().to_dict(),
            "vehicles_in_view": {"mean": round(float(veh.mean()), 1), "max": int(veh.max())},
            "light_phases_s": light_phases(ctx),
        }
        print("EDA done:", v.name)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("EDA written to", OUT)


if __name__ == "__main__":
    main()
