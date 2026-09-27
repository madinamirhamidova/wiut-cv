# WIUT Hackathon 2026 — Traffic Event Detection (CV track)

Pretrained detector + tracker + one rule per event class on a hand-drawn scene map of this fixed camera,
plus a causal time-to-collision accident-risk score. No training needed.

**Website & live demo:** *TODO: Hugging Face Space URL* · **Predictions on the samples:** [`predictions_samples.json`](predictions_samples.json)

## Install & run (what the organizers run)

```bash
pip install -r requirements.txt               # Python 3.10 - 3.13, pinned versions
bash weights/download.sh                      # once, with internet (YOLOv8s, 22 MB)
python run_submission.py --videos /data/test --out predictions.json
```

`run_submission.py` and `evaluate.py` are the starter kit's files, unchanged. Nothing else is needed: no
manual steps, no network at inference. On a minimal Linux image without OpenGL, `import cv2` needs
`apt-get install -y libgl1`.

**Runtime:** Part A + Part B take 422 s for the 340 s sample on an Apple M5 laptop: 1.24× the video length, 41 % of the 3× budget.
`notebooks/colab_clean_check.ipynb` repeats the organizers' run on a fresh Colab T4.

## Reproduce `predictions_samples.json`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && bash weights/download.sh
# put the sample videos (C3896.MP4, C3897.MP4, C3902.MP4, C3905.MP4) into data/samples/
python run_submission.py --videos data/samples --out predictions_samples.json
python evaluate.py --pred predictions_samples.json --validate-only
```

## Approach

| Stage | Method | Learned / rule-based |
|---|---|---|
| Detection | YOLOv8s, COCO-pretrained (person, bicycle, car, motorcycle, bus, truck), every 2nd frame at 960 px | learned (pretrained, not fine-tuned) |
| Tracking | ByteTrack (Ultralytics implementation) | algorithmic |
| Kinematics | ground point = bottom-centre of the box, 0.6 s smoothing; speed in *box heights / s* (≈ perspective independent) | rule |
| Traffic light | HSV colour count in a fixed box around the signal head | rule |
| Scene map | `configs/scene.yaml`: road, lanes + direction, crosswalks, pedestrian islands, stop line, intersection, queue zones, light box; drawn once for this camera, rescaled to any input resolution | hand-made |
| Events | one module per class in `src/rules/` on trajectories + scene map, then `src/postprocess.py` (merge gaps, drop blips, no same-class overlap) | rule |
| Part B | `src/risk.py`: own detector + tracker run on the frames received so far; pairwise time-to-collision, depth-weighted, EMA-smoothed | rule |

Predicted classes: `stopped_vehicle, congestion, wrong_way, jaywalking, red_light, stop_line, failure_to_yield,
solid_line_crossing, illegal_u_turn, accident` (list in `configs/params.yaml → classes_enabled`).
Not predicted: `near_miss, illegal_turn, road_obstacle, fire_smoke`.

```
video ─► src/detection.py  YOLOv8s + ByteTrack, cached in cache/ per video *and* settings
      ─► src/kinematics.py smoothed ground points, velocities, rider flag
      ─► src/rules/<class>.py  one pure function per class (scene map + light state)
      ─► src/postprocess.py    clean segments  ─► [[start, end, label], ...]
frames ─► src/risk.py          causal TTC risk ─► P(accident within 5 s)
```

All thresholds live in `configs/params.yaml`, and all zones in `configs/scene.yaml`.

## Models, data, licences

- **YOLOv8s** weights: Ultralytics, AGPL-3.0, pretrained on COCO (CC BY 4.0). Downloaded by `weights/download.sh`.
- **Ultralytics** library (detector + ByteTrack tracker): AGPL-3.0.
- **Data:** only the provided sample videos, with our own annotations in `data/labels/dev_gt.json`. No other datasets.
- `run_submission.py`, `evaluate.py`, `examples/`: WIUT Hackathon starter kit, unchanged.

## Reproducibility

- Seed 42 for Python, NumPy and torch (`src/config.set_seed`), `cudnn.deterministic = True`, no cuDNN benchmark.
- Fixed frame sampling (`stride`), fixed thresholds, pinned package versions.
- The detection cache key includes every detector setting and the light box, so a changed setting is
  never served from a stale cache.
- Non-determinism: results computed on a different GPU type (or CPU / Apple MPS) can differ slightly in
  detector floating point. Two runs on the same machine give identical `predictions.json`.

## Development workflow

```bash
python -m pytest tests                                   # synthetic tests for every rule
python tools/make_gt_template.py                         # data/labels/dev_gt.json, then label by hand
python evaluate.py --pred predictions_samples.json --gt data/labels/dev_gt.json --per-video
python tools/draw_zones.py --video data/samples/C3896.MP4 --show          # check the scene map
python tools/draw_zones.py --video data/samples/C3896.MP4 --key crosswalks --clear   # redraw one zone
python tools/render_video.py --video data/samples/C3896.MP4 --max-width 1280 --every 2
```

Label with the start/end conventions in the task brief (Event classes table). Tune thresholds in
`configs/params.yaml`, and keep only classes that score on the dev labels in `classes_enabled`: a
predicted class that never occurs in the test set adds a 0 to the macro F1.

## Website

One Gradio app (`app/app.py`) holds the whole website: approach, EDA, annotated results for every sample
video with clickable timelines, live demo, report, team and links.

```bash
pip install -r app/requirements.txt
python tools/build_site.py                  # after run_submission.py: EDA + annotated samples -> app/assets/
python app/app.py                           # http://127.0.0.1:7860
python tools/deploy_space.py <hf-user>/<space-name>     # publish as a Hugging Face Space
```

## Repository layout

```
solution.py            interface imported by the harness (Part A + Part B)
run_submission.py      organizers' harness (unchanged)      evaluate.py   organizers' metric (unchanged)
src/                   pipeline: detection, kinematics, scene, light state, rules/, postprocess, risk
configs/               params.yaml (all thresholds), scene.yaml (zones for this camera)
tools/                 draw_zones, render_video, eda, make_gt_template, build_site, deploy_space
tests/                 synthetic tests of every rule and the risk score
app/                   website + live demo (Gradio), content/ = page texts
notebooks/             colab_clean_check.ipynb: clean-machine + time-budget check on a T4
weights/download.sh    fetches YOLOv8s
```

## Team

| Member | Links |   
| Madina Mirxamitova | [GitHub](https://github.com/madinamirhamidova) · [LinkedIn](https://www.linkedin.com/in/madina-mirhamidova) |    
| Odina Mustafaeva | [GitHub](https://github.com/OdinaMustafaeva) · [LinkedIn](https://www.linkedin.com/in/odina-m-3a6b0a275?utm_source=share_via&utm_content=profile&utm_medium=member_ios) |   
| Kamola Komilova | [GitHub](https://github.com/kamolak07) · [LinkedIn](https://www.linkedin.com/in/kamolakomilova?utm_source=share_via&utm_content=profile&utm_medium=member_android) |