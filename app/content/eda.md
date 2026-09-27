### What the data told us, and what we changed because of it

<!-- TODO: update the numbers once all sample videos are in (python tools/build_site.py re-runs the EDA). -->

- **4K at 29.97 fps, about 5–6 min per clip.** Detection runs every 2nd frame at 960 px. That is
  enough for the rules (events last seconds) and keeps Part A + B at 1.24× real time on a laptop GPU (budget: 3×).
- **Two carriageways with opposite directions** (flow field below). The lower one queues at the signal
  and runs towards the bottom-right; the upper one runs towards the top-left. These directions became the
  lanes in the scene map (used by `wrong_way` and `congestion`).
- **The signal cycle is 36 s green and 39 s red.** The lit lamp is small and dim in daylight
  (HSV value 60–150), so the reader's thresholds were measured on these frames and not guessed.
- **Speeds span a large range of pixel sizes.** A car at the bottom of the frame is about 5× taller than
  at the top. All speed thresholds are in *box heights per second*, which makes one threshold work
  everywhere.
- **Dense traffic and articulated buses.** Many overlapping boxes, and buses detected as two halves.
  These caused most false accident alarms, and led to the closing-speed and standstill checks.
- **Pedestrians cross in groups**, often partly hidden by queued cars. Tracks break, which makes
  `jaywalking` and `failure_to_yield` the hardest boundaries.
