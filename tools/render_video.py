"""Draw tracks, zones, light state and active events onto a video (for the website/debugging).

  python tools/render_video.py --video data/samples/a.mp4 --pred predictions_samples.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_params  # noqa: E402
from src.detection import track_video  # noqa: E402
from src.scene import Scene, draw_scene  # noqa: E402

CLS_COLOR = {"person": (0, 200, 255), "car": (255, 128, 0), "bus": (255, 0, 128), "truck": (128, 0, 255),
             "motorcycle": (0, 255, 128), "bicycle": (0, 255, 255)}


def annotate(frame, scene, boxes, k, header, alert=False):
    """Draw zones, tracked boxes (original-resolution coords scaled by k) and a header bar, in place."""
    draw_scene(frame, scene)
    if boxes is not None:
        for r in boxes.itertuples():
            c = CLS_COLOR.get(r.cls, (255, 255, 255))
            x1, y1, x2, y2 = (int(v * k) for v in (r.x1, r.y1, r.x2, r.y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{r.cls}#{r.tid}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if alert else (255, 255, 255), 2)
    return frame


def _open_writer(out_path, fps, size):
    """H.264 plays in browsers (Gradio, website); fall back to mp4v where OpenCV has no H.264 encoder."""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"cannot open video writer for {out_path}")


def render(video, events, out_path, risk=None, max_width=None, every=1, on_progress=None, detector=None):
    """max_width: downscale output to this width; every: keep every Nth frame (faster, smaller preview);
    on_progress: optional callback(fraction_done); detector: same overrides as pipeline.detect_events.
    Tracks come from the detection cache (computed first if missing)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w0, h0 = int(cap.get(3)), int(cap.get(4))
    n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    det = {**load_params()["detector"], **(detector or {})}
    tracks, lights_df, _ = track_video(str(video), det, Scene.load().for_size(w0, h0).light_box)
    lights = lights_df.set_index("frame")["state"].to_dict()
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    k = min(1.0, max_width / w0) if max_width else 1.0
    w, h = int(w0 * k) // 2 * 2, int(h0 * k) // 2 * 2
    scene = Scene.load().for_size(w, h)
    writer = _open_writer(out_path, fps / every, (w, h))
    idx, last, light = 0, None, "unknown"
    while True:
        t = idx / fps
        last = by_frame.get(idx, last)
        light = lights.get(idx, light)
        if idx % every:                       # skipped frame: advance without decoding to BGR
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if k < 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        active = [e[2] for e in events if e[0] <= t <= e[1]]
        txt = f"t={t:6.1f}s  light={light}  " + (" | ".join(active) if active else "")
        if risk is not None:
            txt += f"  risk={risk.get(idx, 0):.2f}"
        writer.write(annotate(frame, scene, last, k, txt, bool(active)))
        if on_progress and idx % int(fps * 10) == 0:     # every ~10 s of video
            on_progress(min(1.0, idx / n_frames))
        idx += 1
    cap.release()
    writer.release()
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pred", default=str(ROOT / "predictions_samples.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-width", type=int, default=None, help="downscale, e.g. 1280")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    a = ap.parse_args()
    pred = json.loads(Path(a.pred).read_text())["videos"].get(Path(a.video).name, {})
    risk = {i: s for i, (_, s) in enumerate(pred.get("risk", []))} or None
    out = Path(a.out or ROOT / "outputs" / f"{Path(a.video).stem}_annotated.mp4")
    out.parent.mkdir(exist_ok=True)
    print("wrote", render(a.video, pred.get("events", []), out, risk, a.max_width, a.every))
