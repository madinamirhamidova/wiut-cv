"""Shared context passed to every rule."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..light_state import light_series, state_at
from ..scene import Scene


@dataclass
class Context:
    tracks: pd.DataFrame      # with kinematics + rider flag
    lights: pd.DataFrame
    scene: Scene
    meta: dict
    params: dict              # params["rules"][<rule>] is injected per rule as `p`
    _light: tuple = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return float(self.meta["duration"])

    def light_timeline(self):
        """(times, states) of the signal, debounced by params["light_min_phase_s"]."""
        if self._light is None:
            self._light = light_series(self.lights, self.params.get("light_min_phase_s", 0.0))
        return self._light

    def light(self, t):
        return state_at(*self.light_timeline(), t)

    def next_green_after(self, t0: float) -> float:
        times, states = self.light_timeline()
        idx = np.where((times > t0) & (states == "green"))[0]
        return float(times[idx[0]]) if len(idx) else self.duration


def ev(s, e, label):
    return [float(s), float(e), label]
