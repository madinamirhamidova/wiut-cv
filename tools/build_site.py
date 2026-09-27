"""Build the website assets from predictions_samples.json (run after run_submission.py).

  python tools/build_site.py            # -> app/assets/{eda,results}/, app/assets/example_clip.mp4

For every sample video: annotated preview video, events + risk curve as JSON, one example frame
per detected class. Also runs the EDA and cuts a short clip for the demo's example button.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_params, sample_videos  # noqa: E402
from src.detection import track_video  # noqa: E402
from src.scene import Scene  # noqa: E402
from tools import eda  # noqa: E402
from tools.render_video import _open_writer, annotate, render  # noqa: E402

ASSETS = ROOT / "app" / "assets"
PREVIEW_WIDTH = 960       # px, annotated preview videos on the site
PREVIEW_EVERY = 2         # keep every 2nd frame
RISK_HZ = 4               # risk curve samples per second on the site
EXAMPLES_PER_CLASS = 2
CLIP_SECONDS = 30         # demo example clip
CLIP_WIDTH = 1280


def risk_curve(risk, hz):
    """Harness curve [[t, score], ...] per frame -> ~hz samples/s (max within each bin keeps the peaks)."""
    out = {}
    for t, s in risk:
        b = round(int(t * hz) / hz, 3)
        out[b] = max(out.get(b, 0.0), s)
    return [[t, round(s, 3)] for t, s in sorted(out.items())]


def examples(video, events, out_dir):
    """One annotated frame at the middle of the first events of each class."""
    cap = cv2.VideoCapture(str(video))
    w0, h0 = int(cap.get(3)), int(cap.get(4))
    k = min(1.0, PREVIEW_WIDTH / w0)
    w, h = int(w0 * k) // 2 * 2, int(h0 * k) // 2 * 2
    scene = Scene.load().for_size(w, h)
    tracks, _, _ = track_video(str(video), load_params()["detector"], Scene.load().for_size(w0, h0).light_box)
    fps, made, per_cls = cap.get(cv2.CAP_PROP_FPS) or 25, [], {}
    for s, e, lab in events:
        if per_cls.get(lab, 0) >= EXAMPLES_PER_CLASS:
            continue
        per_cls[lab] = per_cls.get(lab, 0) + 1
        t = (s + e) / 2
        f = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        near = tracks[(tracks.frame - f).abs() == (tracks.frame - f).abs().min()]
        annotate(frame, scene, near, k, f"{lab}  [{s:.1f}-{e:.1f}s]", True)
        name = f"{video.stem}_{lab}_{per_cls[lab]}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        made.append({"file": name, "label": lab, "start": s, "end": e})
    cap.release()
    return made


def example_clip(video, out_path):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w0, h0 = int(cap.get(3)), int(cap.get(4))
    k = min(1.0, CLIP_WIDTH / w0)
    size = (int(w0 * k) // 2 * 2, int(h0 * k) // 2 * 2)
    writer = _open_writer(out_path, fps, size)
    for _ in range(int(CLIP_SECONDS * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "predictions_samples.json"))
    ap.add_argument("--skip-videos", action="store_true", help="only JSON, examples and EDA")
    a = ap.parse_args()
    pred = json.loads(Path(a.pred).read_text())["videos"]

    res = ASSETS / "results"
    res.mkdir(parents=True, exist_ok=True)
    index = []
    for video in sample_videos():
        p = pred.get(video.name)
        if p is None:
            print(f"skip {video.name}: not in {a.pred} (run run_submission.py first)")
            continue
        cap = cv2.VideoCapture(str(video))
        duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 25)
        cap.release()
        events = sorted(p["events"])
        info = {"video": video.name, "duration": round(duration, 2), "events": events,
                "risk": risk_curve(p.get("risk", []), RISK_HZ),
                "examples": examples(video, events, res)}
        (res / f"{video.stem}.json").write_text(json.dumps(info))
        if not a.skip_videos:
            print("rendering", video.name)
            risk = {i: s for i, (_, s) in enumerate(p.get("risk", []))} or None
            render(str(video), events, res / f"{video.stem}.mp4", risk, PREVIEW_WIDTH, PREVIEW_EVERY)
        index.append(video.stem)
    (res / "index.json").write_text(json.dumps(index))

    eda.main()
    shutil.copytree(eda.OUT, ASSETS / "eda", dirs_exist_ok=True)
    if sample_videos() and not a.skip_videos:
        example_clip(sample_videos()[0], ASSETS / "example_clip.mp4")
    print("site assets in", ASSETS)


if __name__ == "__main__":
    main()
