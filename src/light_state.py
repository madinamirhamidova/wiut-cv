"""Traffic-light state from a small image box via HSV colour counting."""
import numpy as np


class LightReader:
    def __init__(self, box, min_frac: float = 0.02, min_sat: int = 90, min_val: int = 150):
        self.box = box          # [x1, y1, x2, y2] or None
        self.min_frac = min_frac
        self.min_sat = min_sat
        self.min_val = min_val

    def read(self, frame) -> str:
        """Return 'red', 'green' or 'unknown'."""
        if self.box is None:
            return "unknown"
        import cv2
        x1, y1, x2, y2 = self.box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lit = (s > self.min_sat) & (v > self.min_val)
        red = lit & ((h < 10) | (h > 165))
        green = lit & (h > 40) & (h < 95)
        n = crop.shape[0] * crop.shape[1]
        r, g = red.sum() / n, green.sum() / n
        if max(r, g) < self.min_frac:
            return "unknown"
        return "red" if r > g else "green"


def light_series(lights_df, min_phase_s: float = 0.0):
    """Return (times, states) arrays; forward-fill 'unknown' with the last known state, then absorb
    phases shorter than min_phase_s into the previous phase (flicker from occlusion / reflections)."""
    if lights_df is None or len(lights_df) == 0:
        return np.array([0.0]), np.array(["unknown"])
    t = lights_df["t"].to_numpy(float)
    st = lights_df["state"].to_numpy(object).copy()
    last = "unknown"
    for i, s in enumerate(st):
        if s == "unknown":
            st[i] = last
        else:
            last = s
    start = 0
    for i in range(1, len(st) + 1):
        if i == len(st) or st[i] != st[start]:
            if 0 < start and i < len(st) and t[i] - t[start] < min_phase_s:   # last phase runs on: keep
                st[start:i] = st[start - 1]
            start = i
    return t, st


def state_at(times, states, t):
    """Vectorised lookup of light state at time(s) t."""
    idx = np.clip(np.searchsorted(times, np.asarray(t), side="right") - 1, 0, len(times) - 1)
    return states[idx]
