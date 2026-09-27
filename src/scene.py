"""Scene configuration: the hand-drawn zones for this fixed camera."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CONFIG_DIR, load_scene_dict


@dataclass
class Lane:
    polygon: np.ndarray
    direction: np.ndarray  # unit vector, image coords
    group: str


class Scene:
    def __init__(self, d: dict):
        self.d = d or {}

    @classmethod
    def load(cls, path: Path = CONFIG_DIR / "scene.yaml") -> "Scene":
        return cls(load_scene_dict(path))

    def for_size(self, width: int, height: int) -> "Scene":
        """Copy with every coordinate rescaled from `frame_size` (the drawing resolution) to width x height."""
        fs = self.d.get("frame_size")
        if not fs or list(fs) == [width, height]:
            return self
        k = (width / fs[0], height / fs[1])

        def scale(v):
            if isinstance(v, dict):
                return {key: scale(x) for key, x in v.items()}
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                return [x * k[i % 2] for i, x in enumerate(v)]      # [x, y] points and [x1, y1, x2, y2] boxes
            if isinstance(v, list):
                return [scale(x) for x in v]
            return v
        return Scene({**scale({key: v for key, v in self.d.items() if key != "frame_size"}),
                      "frame_size": [width, height]})

    def poly(self, key):
        v = self.d.get(key)
        return None if not v else np.asarray(v, dtype=float)

    def polys(self, key):
        return [np.asarray(p, dtype=float) for p in self.d.get(key, []) or []]

    def lines(self, key):
        return [np.asarray(l, dtype=float) for l in self.d.get(key, []) or []]

    @property
    def lanes(self):
        out = []
        for i, l in enumerate(self.d.get("lanes", []) or []):
            (x1, y1), (x2, y2) = l["direction"]
            v = np.array([x2 - x1, y2 - y1], dtype=float)
            v /= max(np.linalg.norm(v), 1e-9)
            out.append(Lane(np.asarray(l["polygon"], float), v, str(l.get("group", f"lane{i}"))))
        return out

    @property
    def stop_line(self):
        return self.poly("stop_line")

    @property
    def light_box(self):
        v = self.d.get("light_box")
        return None if not v else [int(x) for x in v]


COLORS = {
    "road": (80, 80, 80), "intersection": (0, 165, 255), "crosswalks": (255, 255, 255),
    "queue_zones": (255, 0, 255), "no_u_turn_zones": (0, 0, 255), "islands": (203, 192, 255), "lanes": (0, 255, 0),
    "solid_lines": (0, 255, 255), "stop_line": (0, 0, 255), "light_box": (255, 255, 0),
}


def draw_scene(img, scene: Scene):
    """Overlay all zones on a BGR image (in place)."""
    import cv2
    d = scene.d
    for key in ("road", "intersection"):
        if d.get(key):
            cv2.polylines(img, [np.int32(d[key])], True, COLORS[key], 2)
    for key in ("crosswalks", "queue_zones", "no_u_turn_zones", "islands"):
        for p in d.get(key, []) or []:
            cv2.polylines(img, [np.int32(p)], True, COLORS[key], 2)
    for lane in scene.lanes:
        cv2.polylines(img, [np.int32(lane.polygon)], True, COLORS["lanes"], 1)
        c = lane.polygon.mean(axis=0)
        cv2.arrowedLine(img, tuple(np.int32(c)), tuple(np.int32(c + lane.direction * 60)),
                        COLORS["lanes"], 3, tipLength=0.35)
    for l in d.get("solid_lines", []) or []:
        cv2.line(img, tuple(np.int32(l[0])), tuple(np.int32(l[1])), COLORS["solid_lines"], 2)
    if d.get("stop_line"):
        l = d["stop_line"]
        cv2.line(img, tuple(np.int32(l[0])), tuple(np.int32(l[1])), COLORS["stop_line"], 3)
    if d.get("light_box"):
        x1, y1, x2, y2 = map(int, d["light_box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["light_box"], 2)
    return img
