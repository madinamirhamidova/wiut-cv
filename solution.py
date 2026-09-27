"""WIUT Hackathon 2026 - CV track. Entry point imported by run_submission.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import detect_events as _detect_events  # noqa: E402
from src.risk import RiskEstimator as _RiskEstimator      # noqa: E402

CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]


def detect_events(video_path: str) -> list[list]:
    """Part A. Return [[start_sec, end_sec, label], ...] for one .mp4."""
    return [e for e in _detect_events(video_path) if e[2] in CLASSES]


class RiskEstimator(_RiskEstimator):
    """Part B. Causal: step() sees frames in order and nothing else."""

    def reset(self, meta: dict) -> None:
        super().reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return super().step(frame, t_sec)
