## Problem

A fixed CCTV camera looks at a signalised intersection in Tashkent. For every video we must

- **Part A:** list every traffic event as `[start_sec, end_sec, label]` (14 official classes, scored by
  macro F1 at temporal IoU 0.3 / 0.5 / 0.7), and
- **Part B:** at every frame, output the probability that an accident starts within the next 5 s,
  using only the frames seen so far.

## Approach

We use no end-to-end event model. A pretrained detector and tracker turn the video into trajectories,
and one small, readable rule per event class works on those trajectories plus a hand-drawn map of the
scene (zones for this camera).

| Stage | Method | Learned or rule-based |
|---|---|---|
| Detection | YOLOv8s, COCO-pretrained (person, bicycle, car, motorcycle, bus, truck), every 2nd frame, 960 px | learned (pretrained, not fine-tuned) |
| Tracking | ByteTrack (Ultralytics) | algorithmic |
| Kinematics | ground point = bottom-centre of the box, 0.6 s smoothing, speed in *box heights per second* (roughly independent of distance to the camera) | rule |
| Traffic light | HSV colour count inside a fixed box around the signal head | rule |
| Scene map | road, lanes + direction, crosswalks, pedestrian islands, stop line, intersection, queue zones: drawn once for this camera, scaled to any resolution | hand-made |
| Events | one rule per class on trajectories + scene map, then segment clean-up (merge gaps, drop blips, no same-class overlap) | rule |
| Part B | pairwise time-to-collision between tracked road users, depth-weighted, EMA-smoothed; its own detector run, strictly causal | rule |

### Why this design

- **No labels.** We get a few unlabeled sample videos. A rule on a trajectory can be checked by
  watching the video, whereas a learned event classifier would need data we don't have.
- **One fixed camera.** A scene map drawn once (stop line, crosswalks, lane directions) gives most
  classes their meaning. Red-light running is "crossed *this* line while *this* light was red".
- **Boundaries matter** (IoU up to 0.7), and trajectories give frame-accurate start/end times that
  follow the annotation conventions of the brief.

### Classes

Predicted: `stopped_vehicle`, `congestion`, `wrong_way`, `jaywalking`, `red_light`, `stop_line`,
`failure_to_yield`, `solid_line_crossing`, `illegal_u_turn`, `accident`.
Not predicted (no reliable signal yet): `near_miss`, `illegal_turn`, `road_obstacle`, `fire_smoke`.
A class we predict that never occurs in the test set costs a zero in the macro average, so the final
list is chosen on our own dev labels.
