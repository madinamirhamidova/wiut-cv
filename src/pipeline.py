"""Part A orchestration: video -> tracks -> rules -> clean events."""
import logging

from . import postprocess
from .config import load_params, set_seed
from .detection import track_video, video_meta
from .kinematics import add_kinematics, mark_riders
from .rules import RULES
from .rules.base import Context
from .scene import Scene

log = logging.getLogger(__name__)


def build_context(video_path: str, use_cache: bool = True, detector: dict | None = None, on_progress=None):
    """detector: optional overrides of params["detector"] (e.g. a faster setting for the CPU demo).
    on_progress(fraction): called during detection + tracking."""
    params = load_params()
    set_seed(params["seed"])
    params["detector"] = {**params["detector"], **(detector or {})}
    m = video_meta(video_path)
    scene = Scene.load().for_size(m["width"], m["height"])
    tracks, lights, meta = track_video(video_path, params["detector"], scene.light_box, use_cache, on_progress)
    tracks = add_kinematics(tracks, min_points=params["min_track_points"])
    tracks = mark_riders(tracks)
    return Context(tracks=tracks, lights=lights, scene=scene, meta=meta, params=params)


def run_rules(ctx: Context):
    raw = []
    for name in ctx.params["classes_enabled"]:
        rule = RULES.get(name)
        if rule is None:
            log.warning("no rule for %s", name)
            continue
        try:
            raw += rule.detect(ctx, ctx.params["rules"].get(name, {}))
        except Exception:                           # one broken rule must not kill the video
            log.exception("rule %s failed", name)
    return postprocess.clean(raw, ctx.duration, ctx.params["postprocess"])


def detect_events(video_path: str, use_cache: bool = True, detector: dict | None = None, on_progress=None):
    return run_rules(build_context(video_path, use_cache, detector, on_progress))
