"""Load params.yaml / scene.yaml and fix seeds."""
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


VIDEO_EXTS = (".mp4", ".MP4")


def sample_videos(folder: Path = ROOT / "data" / "samples") -> list:
    return sorted(p for p in Path(folder).iterdir() if p.suffix in VIDEO_EXTS)


def load_params(path: Path = CONFIG_DIR / "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_scene_dict(path: Path = CONFIG_DIR / "scene.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
