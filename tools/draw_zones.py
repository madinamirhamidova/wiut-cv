"""Click zones on a video frame and save them to configs/scene.yaml.

Usage:
  python tools/draw_zones.py --video data/samples/a.mp4 --key road
  python tools/draw_zones.py --video data/samples/a.mp4 --key lanes      # polygon, ENTER, then 2-click arrow, ENTER
  python tools/draw_zones.py --video data/samples/a.mp4 --show           # just view everything
Controls: left-click = add point, u = undo, ENTER = finish, ESC = cancel, c (with --key) clears that key first.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CONFIG_DIR  # noqa: E402
from src.scene import Scene, draw_scene  # noqa: E402

SCHEMA = {  # key: (kind, is_list)
    "road": ("polygon", False), "intersection": ("polygon", False),
    "lanes": ("lane", True), "crosswalks": ("polygon", True), "queue_zones": ("polygon", True),
    "no_u_turn_zones": ("polygon", True), "islands": ("polygon", True), "solid_lines": ("line", True),
    "stop_line": ("line", False), "light_box": ("box", False),
}
SCENE_PATH = CONFIG_DIR / "scene.yaml"
WIN = "draw_zones"


def grab_frame(video, t):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"cannot read frame at {t}s from {video}")
    return frame


def collect(base, prompt, n_max=None):
    """Collect clicks until ENTER (or n_max points). Returns list of [x, y] or None on ESC."""
    pts = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and (n_max is None or len(pts) < n_max):
            pts.append([int(x), int(y)])

    cv2.setMouseCallback(WIN, on_click)
    while True:
        img = base.copy()
        for p in pts:
            cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.polylines(img, [np.int32(pts)], n_max is None, (0, 0, 255), 2)
        cv2.putText(img, prompt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(WIN, img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            return None
        if k == ord("u") and pts:
            pts.pop()
        if k in (13, 10):
            return pts


def _to_frame_size(v, kx, ky):
    """Clicked pixels of this video -> frame_size pixels."""
    if isinstance(v, dict):
        return {key: _to_frame_size(x, kx, ky) for key, x in v.items()}
    if v and all(isinstance(x, (int, float)) for x in v):
        return [int(round(x * (kx, ky)[i % 2])) for i, x in enumerate(v)]
    return [_to_frame_size(x, kx, ky) for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--t", type=float, default=5.0, help="timestamp of frame to draw on")
    ap.add_argument("--key", choices=list(SCHEMA))
    ap.add_argument("--clear", action="store_true", help="delete existing value of --key first")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    scene_d = yaml.safe_load(SCENE_PATH.read_text()) or {}
    if a.key and a.clear:
        scene_d.pop(a.key, None)
    frame = grab_frame(a.video, a.t)
    h, w = frame.shape[:2]
    fs = scene_d.setdefault("frame_size", [w, h])        # zones are stored in this resolution
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    base = draw_scene(frame.copy(), Scene(scene_d).for_size(w, h))

    if a.show or not a.key:
        cv2.imshow(WIN, base)
        cv2.waitKey(0)
        return

    kind, is_list = SCHEMA[a.key]
    if kind == "polygon":
        v = collect(base, f"{a.key}: click polygon, ENTER")
        ok = v and len(v) >= 3
    elif kind == "line":
        v = collect(base, f"{a.key}: click 2 points, ENTER", 2)
        ok = v and len(v) == 2
    elif kind == "box":
        v = collect(base, f"{a.key}: click 2 corners (tight around the light), ENTER", 2)
        ok = v and len(v) == 2
        if ok:
            (x1, y1), (x2, y2) = v
            v = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    else:  # lane
        poly = collect(base, "lane: click polygon, ENTER")
        arrow = collect(base, "lane: click 2 points = traffic direction (tail -> head), ENTER", 2) if poly else None
        ok = poly and len(poly) >= 3 and arrow and len(arrow) == 2
        v = {"polygon": poly, "direction": arrow} if ok else None
    if not ok:
        sys.exit("cancelled / not enough points - nothing saved")
    v = _to_frame_size(v, fs[0] / w, fs[1] / h)

    if is_list:
        scene_d.setdefault(a.key, []).append(v)
    else:
        scene_d[a.key] = v
    SCENE_PATH.write_text(yaml.safe_dump(scene_d, default_flow_style=None, sort_keys=False))
    print(f"saved {a.key} -> {SCENE_PATH}")


if __name__ == "__main__":
    main()
