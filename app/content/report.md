## Technical report

### What we built

A detector → tracker → rules pipeline (see *Approach*): YOLOv8s + ByteTrack give trajectories. A
hand-drawn scene map for this camera gives them meaning, and one rule per class turns them into
time segments. Part B is a causal time-to-collision risk score with its own detector run.
Everything is deterministic (fixed seeds, fixed frame sampling), and detections are cached per video
and per setting.

### What worked

- **Speed in box heights per second.** Pixel speeds depend on the distance to the camera; dividing
  by the box height makes one threshold work across the whole image.
- **Reading the signal from pixels.** The first version never saw the light: its brightness
  threshold (HSV value > 150) was too strict for this footage. After measuring the lit lamp (value 60–150 in daylight), it
  reads red and green correctly over the whole clip, with a cycle of 36 s green and 39 s red. Red blips from occlusion are ignored (phases under 5 s).
- **Lane directions from data.** The mean heading of moving vehicles per grid cell (see *EDA*)
  showed the two carriageways and confirmed the lane arrows in the scene map.
- **Runtime.** Part A + Part B take 1.24× the video length on an Apple M5 laptop (41 % of the
  3× budget); Part A and Part B take about half each.

### What did not work (yet)

- **Accident heuristic.** "Two boxes overlap, then both stop" fired **37 times** in a normal
  5-minute clip. The causes were cars pulling up to a red-light queue, articulated buses detected as
  two buses, one truck detected as both *truck* and *car*, and tracker ID switches read as "left the
  frame". Requiring a real closing speed, a border exit, 5 s of standstill afterwards, and no queue
  at a red light brought it down to **1** false alarm. A heuristic this fragile is still the weakest
  part.
- **Risk score too eager.** The first TTC version was ≥ 0.5 on 15 % of all frames (47 alarms in
  5.7 min) because perspective makes cars in neighbouring lanes look close. Weighting vertical image
  distance ×3 (it is depth), a tighter collision radius and a faster decay: 0.8 % of frames, 8 alarms.
  Without any real accident in the samples, this is calibrated on negatives only.
- **Dense traffic.** Pedestrians partly hidden behind cars, and long queues, break tracks. Jaywalking
  and stopped-vehicle segments get too long when a track jumps between people or vehicles.

### Evaluation

<!-- Fill in after labelling the samples: python evaluate.py --pred predictions_samples.json --gt data/labels/dev_gt.json --per-video -->
*Dev-set scores on our own labels of the sample videos go here (per class F1 at IoU 0.3 / 0.5 / 0.7).*

### What we would do next

1. Label all sample videos and tune each rule's thresholds against F1 at IoU 0.7.
2. Replace the accident heuristic with a small learned classifier on track-pair features, trained on a
   public crash dataset (e.g. CCD / DoTA) and verified on our clips.
3. Add `near_miss` (sharp deceleration or swerve while TTC < 1 s) and `illegal_turn` (turn
   movements between lane zones).
4. Fine-tune the detector on frames from this camera, to separate articulated buses and reduce missed pedestrians.
