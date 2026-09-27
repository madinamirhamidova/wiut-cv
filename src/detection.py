"""Step 1 of the pipeline: YOLO detection + ByteTrack tracking, cached to disk.

Output tables (pandas):
  tracks: frame, t, tid, cls, conf, x1, y1, x2, y2
  lights: frame, t, state
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .light_state import LightReader

# COCO ids used by pretrained YOLO
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLES = {"car", "motorcycle", "bus", "truck", "bicycle"}
TWO_WHEELERS = {"bicycle", "motorcycle"}

CACHE_DIR = ROOT / "cache"


def video_meta(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    meta = {
        "fps": float(fps),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    meta["duration"] = meta["n_frames"] / meta["fps"]
    return meta


def resolve_device(device):
    """None = auto: CUDA if available, else Apple MPS, else CPU (Ultralytics' own auto skips MPS)."""
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(det_cfg: dict):
    from ultralytics import YOLO
    return YOLO(str(ROOT / det_cfg["weights"]))


def cache_paths(video_path: str, det_cfg: dict, light_box=None):
    """Cache files for this video *and* these settings: changing imgsz, stride, the light box, ...
    gives a new key instead of silently reusing stale tracks."""
    key = hashlib.sha1(json.dumps([det_cfg, light_box], sort_keys=True).encode()).hexdigest()[:10]
    stem = f"{Path(video_path).stem}_{key}"
    return (CACHE_DIR / f"{stem}_tracks.csv.gz",
            CACHE_DIR / f"{stem}_lights.csv.gz",
            CACHE_DIR / f"{stem}_meta.json")


def track_video(video_path: str, det_cfg: dict, light_box=None, use_cache: bool = True, on_progress=None):
    """Run detector+tracker over the video. Returns (tracks_df, lights_df, meta).
    on_progress(fraction) is called about once per second of video."""
    import cv2
    stride = int(det_cfg.get("stride", 2))
    p_tracks, p_lights, p_meta = cache_paths(video_path, det_cfg, light_box)
    if use_cache and p_tracks.exists() and p_lights.exists() and p_meta.exists():
        return (pd.read_csv(p_tracks), pd.read_csv(p_lights),
                json.loads(p_meta.read_text()))

    model = load_model(det_cfg)
    reader = LightReader(light_box, **det_cfg.get("light", {}))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    rows, lights = [], []
    idx = 0
    while True:
        if idx % stride:                      # skip frame cheaply (no decode)
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        res = model.track(frame, persist=True, tracker=det_cfg.get("tracker", "bytetrack.yaml"),
                          classes=list(COCO), conf=det_cfg.get("conf", 0.25),
                          imgsz=det_cfg.get("imgsz", 960), device=resolve_device(det_cfg.get("device")),
                          verbose=False)[0]
        b = res.boxes
        if b is not None and b.id is not None:
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.int().cpu().numpy()
            cls = b.cls.int().cpu().numpy()
            conf = b.conf.cpu().numpy()
            for (x1, y1, x2, y2), i, c, s in zip(xyxy, ids, cls, conf):
                rows.append((idx, t, int(i), COCO[int(c)], float(s), x1, y1, x2, y2))
        lights.append((idx, t, reader.read(frame)))
        if on_progress and idx % max(1, round(fps)) < stride:
            on_progress(min(idx / n, 1.0))
        idx += 1
    cap.release()

    tracks = pd.DataFrame(rows, columns=["frame", "t", "tid", "cls", "conf", "x1", "y1", "x2", "y2"])
    lights_df = pd.DataFrame(lights, columns=["frame", "t", "state"])
    meta = video_meta(video_path)
    meta["duration"] = idx / fps if idx else meta["duration"]   # trust frames actually read
    meta["stride"] = stride

    CACHE_DIR.mkdir(exist_ok=True)
    tracks.to_csv(p_tracks, index=False)
    lights_df.to_csv(p_lights, index=False)
    p_meta.write_text(json.dumps(meta))
    return tracks, lights_df, meta
