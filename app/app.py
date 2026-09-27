"""Team website + live demo in one Gradio app (runs locally or as a Hugging Face Space).

  pip install -r app/requirements.txt
  python tools/build_site.py          # once, after run_submission.py: EDA + sample results -> app/assets/
  python app/app.py                   # http://127.0.0.1:7860
  python tools/deploy_space.py <hf-user>/<space-name>
"""
import html
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd
import plotly.graph_objects as go

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT))
from solution import CLASSES, RiskEstimator  # noqa: E402
from src.pipeline import detect_events  # noqa: E402
from tools.render_video import render  # noqa: E402

ASSETS, CONTENT = APP / "assets", APP / "content"
ON_SPACE = "SPACE_ID" in os.environ
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", 120 if ON_SPACE else 600))
MAX_FILE = os.environ.get("MAX_FILE", "500mb" if ON_SPACE else "8gb")
FAST_DETECTOR = {"imgsz": 640, "stride": 3}     # demo setting when there is no GPU
COLORS = dict(zip(CLASSES, ["#d62728", "#ff7f0e", "#e377c2", "#9467bd", "#8c564b", "#7f7f7f", "#17becf",
                            "#bcbd22", "#1f77b4", "#2ca02c", "#aa3377", "#4477aa", "#66ccee", "#ee6677"]))

HEAD = """<script>
document.addEventListener('click', (e) => {            // click an event -> jump the video to it
  const el = e.target.closest('[data-seek]');
  if (!el) return;
  const v = document.querySelector('#' + el.dataset.target + ' video');
  if (!v) return;
  v.currentTime = parseFloat(el.dataset.seek);
  v.play();
  v.scrollIntoView({behavior: 'smooth', block: 'center'});
});
</script>"""
CSS = """
.timeline {width: 100%; height: auto; font: 12px sans-serif}
.timeline text {fill: currentColor}
.timeline .grid {stroke: currentColor; opacity: .15}
[data-seek] {cursor: pointer}
.timeline rect[data-seek]:hover {opacity: .75}
.events {width: 100%; border-collapse: collapse; font-size: 14px}
.events td, .events th {padding: 4px 8px; border-bottom: 1px solid rgba(128,128,128,.25); text-align: left}
.events tr[data-seek]:hover {background: rgba(128,128,128,.12)}
.swatch {display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px}
.pipeline {width: 100%; max-width: 980px; height: auto; font: 13px sans-serif}
.pipeline text {fill: currentColor}
.pipeline rect {fill: none; stroke: currentColor; stroke-width: 1.5}
.pipeline path {stroke: currentColor; stroke-width: 1.5; fill: none}
"""

PIPELINE_SVG = """<svg class="pipeline" viewBox="0 0 980 230" role="img" aria-label="pipeline diagram">
<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10" style="fill:currentColor;stroke:none"/></marker></defs>
<g text-anchor="middle">
<rect x="5" y="30" width="110" height="54" rx="8"/><text x="60" y="54">video</text><text x="60" y="72">.mp4, 4K</text>
<rect x="150" y="30" width="170" height="54" rx="8"/><text x="235" y="52">YOLOv8s + ByteTrack</text>
<text x="235" y="70">every 2nd frame · cached</text>
<rect x="355" y="30" width="170" height="54" rx="8"/><text x="440" y="52">trajectories</text>
<text x="440" y="70">ground point · box-heights/s</text>
<rect x="560" y="30" width="170" height="54" rx="8"/><text x="645" y="52">one rule per class</text>
<text x="645" y="70">+ scene map + signal state</text>
<rect x="765" y="30" width="205" height="54" rx="8"/><text x="867" y="52">post-process</text>
<text x="867" y="70">merge · drop blips → events</text>
<rect x="355" y="120" width="170" height="44" rx="8"/><text x="440" y="147">scene map (drawn once)</text>
<rect x="560" y="120" width="170" height="44" rx="8"/><text x="645" y="147">traffic light (HSV)</text>
<rect x="5" y="180" width="110" height="44" rx="8"/><text x="60" y="206">frames, causal</text>
<rect x="150" y="180" width="170" height="44" rx="8"/><text x="235" y="206">YOLOv8s + ByteTrack</text>
<rect x="355" y="180" width="170" height="44" rx="8"/><text x="440" y="206">time-to-collision</text>
<rect x="560" y="180" width="170" height="44" rx="8"/><text x="645" y="206">risk score (Part B)</text>
<text x="490" y="16">Part A: event detection</text>
</g>
<path d="M115,57 H148" marker-end="url(#arr)"/><path d="M320,57 H353" marker-end="url(#arr)"/>
<path d="M525,57 H558" marker-end="url(#arr)"/><path d="M730,57 H763" marker-end="url(#arr)"/>
<path d="M525,142 C545,142 545,90 600,86" marker-end="url(#arr)"/><path d="M645,120 V86" marker-end="url(#arr)"/>
<path d="M115,202 H148" marker-end="url(#arr)"/><path d="M320,202 H353" marker-end="url(#arr)"/>
<path d="M525,202 H558" marker-end="url(#arr)"/>
</svg>"""


def md(name):
    p = CONTENT / name
    return p.read_text() if p.exists() else f"*{name} missing*"


# ---------- shared visualisations ----------

def timeline_html(events, duration, target):
    """Clickable SVG timeline: one row per class, click a segment to play the video from there."""
    labs = [c for c in CLASSES if any(e[2] == c for e in events)]
    if not labs:
        return "<p>No events detected.</p>"
    W, L, R, RH = 1000, 150, 12, 26
    H = RH * len(labs) + 30
    x = lambda t: L + (W - L - R) * t / max(duration, 1e-6)  # noqa: E731
    step = next(s for s in (5, 10, 30, 60, 120, 300) if duration / s <= 12)
    out = [f'<svg class="timeline" viewBox="0 0 {W} {H}" role="img" aria-label="event timeline">']
    for t in range(0, int(duration) + 1, step):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{H - 18}"/>'
                   f'<text x="{x(t):.1f}" y="{H - 4}" text-anchor="middle">{t}s</text>')
    for i, lab in enumerate(labs):
        y = i * RH
        out.append(f'<text x="{L - 8}" y="{y + RH * 0.65:.1f}" text-anchor="end">{lab}</text>')
        for s, e, l in events:
            if l == lab:
                out.append(f'<rect x="{x(s):.1f}" y="{y + 4}" width="{max(3.0, x(e) - x(s)):.1f}" height="{RH - 8}" '
                           f'rx="3" fill="{COLORS[lab]}" data-seek="{s:.2f}" data-target="{target}">'
                           f'<title>{lab}: {s:.1f}-{e:.1f} s (click to play)</title></rect>')
    out.append("</svg>")
    return "".join(out)


def events_html(events, target):
    if not events:
        return ""
    rows = "".join(
        f'<tr data-seek="{s:.2f}" data-target="{target}" title="click to play">'
        f'<td><span class="swatch" style="background:{COLORS[lab]}"></span>{html.escape(lab)}</td>'
        f'<td>{s:.2f}</td><td>{e:.2f}</td><td>{e - s:.1f}</td></tr>'
        for s, e, lab in sorted(events))
    return ('<table class="events"><thead><tr><th>event</th><th>start s</th><th>end s</th><th>length s</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')


def risk_fig(curve):
    if not curve:
        return None
    t, s = zip(*curve)
    fig = go.Figure(go.Scatter(x=t, y=s, mode="lines", line=dict(width=1.5), name="risk"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#d62728", annotation_text="alarm threshold 0.5")
    fig.update_layout(height=240, margin=dict(l=40, r=10, t=10, b=35), yaxis=dict(range=[0, 1], title="P(accident in 5 s)"),
                      xaxis_title="seconds", showlegend=False)
    return fig


# ---------- results on the sample videos ----------

def results_index():
    p = ASSETS / "results" / "index.json"
    return json.loads(p.read_text()) if p.exists() else []


def load_result(stem):
    return json.loads((ASSETS / "results" / f"{stem}.json").read_text())


def show_result(stem):
    if not stem:
        return None, "<p>No results built yet: run <code>python tools/build_site.py</code>.</p>", "", None, []
    r = load_result(stem)
    vid = ASSETS / "results" / f"{stem}.mp4"
    gallery = [(str(ASSETS / "results" / x["file"]), f'{x["label"]} {x["start"]:.1f}-{x["end"]:.1f}s')
               for x in r["examples"]]
    return (str(vid) if vid.exists() else None, timeline_html(r["events"], r["duration"], "res_video"),
            events_html(r["events"], "res_video"), risk_fig(r["risk"]), gallery)


def failure_cases():
    d = CONTENT / "failures"
    caps = json.loads((d / "captions.json").read_text()) if (d / "captions.json").exists() else {}
    return [(str(d / f), cap) for f, cap in caps.items() if (d / f).exists()]


def dashboard_fig():
    rows = [(stem, e[2], e[1] - e[0]) for stem in results_index() for e in load_result(stem)["events"]]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["video", "label", "length"])
    counts = df.groupby(["label", "video"]).size().unstack(fill_value=0)
    fig = go.Figure([go.Bar(name=v, x=counts.index, y=counts[v]) for v in counts.columns])
    fig.update_layout(barmode="stack", height=300, margin=dict(l=40, r=10, t=10, b=80),
                      yaxis_title="events", legend_title="video")
    return fig


# ---------- EDA ----------

def eda_table():
    p = ASSETS / "eda" / "summary.json"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for name, s in json.loads(p.read_text()).items():
        b = [v for _, v in s["brightness"]]
        ph = s["light_phases_s"]
        med = lambda xs: f"{sorted(xs)[len(xs) // 2]:.0f}" if xs else "-"  # noqa: E731
        rows.append({"video": name, "resolution": f'{s["width"]}x{s["height"]}', "fps": round(s["fps"], 2),
                     "duration s": round(s["duration"], 1), "codec": s["codec"],
                     "brightness (0-255)": f"{min(b):.0f}-{max(b):.0f}" if b else "-",
                     "vehicles in view (mean/max)": f'{s["vehicles_in_view"]["mean"]}/{s["vehicles_in_view"]["max"]}',
                     "green / red phase s": f'{med(ph["green"])} / {med(ph["red"])}',
                     **{f"tracks: {k}": v for k, v in sorted(s["n_tracks_by_class"].items())}})
    return pd.DataFrame(rows)


def eda_counts_fig(stem):
    p = ASSETS / "eda" / f"{stem}_counts.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0).rolling(5, min_periods=1).mean()
    fig = go.Figure([go.Scatter(x=df.index, y=df[c], mode="lines", name=c) for c in df.columns])
    fig.update_layout(height=260, margin=dict(l=40, r=10, t=10, b=35), xaxis_title="seconds",
                      yaxis_title="objects in view (5 s mean)")
    return fig


def eda_view(stem):
    img = lambda suffix: str(p) if (p := ASSETS / "eda" / f"{stem}{suffix}").exists() else None  # noqa: E731
    return eda_counts_fig(stem), img("_flow.jpg"), img("_motion.png"), img("_light_speed.png")


# ---------- live demo ----------

def fast_detector():
    import torch
    return None if torch.cuda.is_available() or torch.backends.mps.is_available() else FAST_DETECTOR


def risk_series(video, on_progress):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    est = RiskEstimator()
    est.reset({"video_id": Path(video).name, "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)),
               "n_frames": n})
    curve, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        curve.append((i / fps, est.step(fr, i / fps)))
        if i % int(fps * 5) == 0:
            on_progress(i / n)
        i += 1
    cap.release()
    return curve


def analyze(video, with_risk, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload an .mp4 first")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    cap.release()
    if duration <= 0:
        raise gr.Error("Could not read this video")
    if duration > MAX_SECONDS:
        raise gr.Error(f"Video is {duration:.0f} s; this demo accepts up to {MAX_SECONDS} s")
    # keep the original name so re-uploads of the same file hit the detection cache
    tmp = Path(tempfile.mkdtemp()) / (Path(video).stem + ".mp4")
    try:
        tmp.symlink_to(Path(video).resolve())   # 4K camera files are several GB: link, don't copy
    except OSError:
        shutil.copy(video, tmp)
    det = fast_detector()
    cpu = " · CPU mode: 640 px, every 3rd frame" if det else ""
    det_end = 0.5 if with_risk else 0.8            # share of the bar spent on detection + tracking
    progress(0.02, desc="Detecting & tracking (the longest step)")
    events = detect_events(str(tmp), detector=det,
                           on_progress=lambda f: progress(0.02 + (det_end - 0.02) * f,
                                                          desc="Detecting & tracking (the longest step)"))
    found = f"**{len(events)} events** in {duration:.0f} s of video{cpu}"
    # show the events now: rendering the annotated video still takes a while
    yield (found + " · still preparing the annotated video…", gr.skip(),
           timeline_html(events, duration, "demo_video"), events_html(events, "demo_video"), gr.skip())
    curve, risk = None, None
    if with_risk:
        curve = risk_series(str(tmp), lambda f: progress(det_end + (0.85 - det_end) * f, desc="Computing risk curve"))
        risk = {i: s for i, (_, s) in enumerate(curve)}
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(value=risk_fig(curve), visible=True)
    out = render(str(tmp), events, tmp.with_name("annotated.mp4"), risk, max_width=960, every=2,
                 on_progress=lambda f: progress(0.85 + 0.15 * f, desc="Rendering video"), detector=det)
    yield found, str(out), gr.skip(), gr.skip(), gr.skip()


# ---------- page ----------

def build():
    stems = results_index()
    with gr.Blocks(title="Traffic event detection · WIUT Hackathon 2026") as demo:
        gr.Markdown("# Traffic event detection from a fixed road camera\n"
                    "WIUT Hackathon 2026 · Computer Vision track. Detector + tracker + rules on a hand-drawn "
                    "scene map, with a causal time-to-collision risk score.")
        with gr.Tabs():
            with gr.Tab("Approach"):
                gr.HTML(PIPELINE_SVG)
                gr.Markdown(md("approach.md"))
            with gr.Tab("EDA"):
                gr.Markdown("### The sample videos")
                gr.Dataframe(eda_table(), wrap=True)
                gr.Markdown(md("eda.md"))
                eda_pick = gr.Dropdown(stems, value=stems[0] if stems else None, label="Video")
                eda_counts = gr.Plot(label="Objects in view by class (interactive)")
                with gr.Row():
                    eda_flow = gr.Image(label="Lane directions: mean heading of moving vehicles", type="filepath")
                    eda_motion = gr.Image(label="Motion heatmap and trajectories", type="filepath")
                eda_light = gr.Image(label="Traffic-light state over time, speed by class", type="filepath")
                eda_pick.change(eda_view, eda_pick, [eda_counts, eda_flow, eda_motion, eda_light])
                demo.load(eda_view, eda_pick, [eda_counts, eda_flow, eda_motion, eda_light])
            with gr.Tab("Results"):
                gr.Markdown("Annotated sample videos. **Click an event** in the timeline or table to jump to it.")
                gr.Plot(dashboard_fig(), label="Events per class, all sample videos")
                res_pick = gr.Dropdown(stems, value=stems[0] if stems else None, label="Sample video")
                res_video = gr.Video(elem_id="res_video", label="Annotated video", interactive=False)
                res_tl = gr.HTML()
                res_risk = gr.Plot(label="Accident risk (Part B)")
                res_table = gr.HTML()
                res_gallery = gr.Gallery(label="One example per detected class", columns=3, height="auto")
                gr.Markdown("### Honest failure cases")
                gr.Gallery(failure_cases(), label="What went wrong", columns=1, height="auto")
                outs = [res_video, res_tl, res_table, res_risk, res_gallery]
                res_pick.change(show_result, res_pick, outs)
                demo.load(show_result, res_pick, outs)
            with gr.Tab("Live demo"):
                gr.Markdown(f"Upload an **.mp4 from this camera**, up to **{MAX_SECONDS} s** and {MAX_FILE} "
                            "(1080p or smaller uploads fastest). Other cameras will not work: the rules use "
                            "zones drawn for this view.")
                with gr.Row():
                    inp = gr.Video(label="Input video", sources=["upload"])
                    out_vid = gr.Video(elem_id="demo_video", label="Annotated", interactive=False)
                with_risk = gr.Checkbox(label="Also compute the accident-risk curve (about 2× slower)")
                btn = gr.Button("Analyze", variant="primary")
                status = gr.Markdown("Upload a video and press **Analyze**. The event timeline and table appear "
                                     "here as soon as detection finishes, before the annotated video is ready.")
                tl = gr.HTML()
                risk = gr.Plot(label="Accident risk", visible=False)
                table = gr.HTML()
                if (ASSETS / "example_clip.mp4").exists():
                    gr.Examples([[str(ASSETS / "example_clip.mp4"), False]], [inp, with_risk],
                                label="Or try a 30 s clip from a sample video")
                btn.click(lambda: ("Analyzing…", None, "", "", gr.update(value=None, visible=False)),
                          None, [status, out_vid, tl, table, risk], show_progress="hidden").then(
                    analyze, [inp, with_risk], [status, out_vid, tl, table, risk], show_progress_on=out_vid)
            with gr.Tab("Report"):
                gr.Markdown(md("report.md"))
            with gr.Tab("Team"):
                gr.Markdown(md("team.md"))
            with gr.Tab("Links"):
                gr.Markdown(md("links.md"))
    return demo


if __name__ == "__main__":
    build().launch(theme=gr.themes.Soft(), css=CSS, head=HEAD, allowed_paths=[str(ASSETS), str(CONTENT)],
                   max_file_size=MAX_FILE)
