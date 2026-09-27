"""Pure-numpy geometry helpers (no OpenCV needed)."""
import numpy as np


def points_in_poly(poly, xs, ys) -> np.ndarray:
    """Vectorised ray casting. poly: (N,2); xs, ys: arrays. Returns bool array."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if poly is None:
        return np.zeros(xs.shape, dtype=bool)
    poly = np.asarray(poly, dtype=float)
    inside = np.zeros(xs.shape, dtype=bool)
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        crosses = ((y1 > ys) != (y0 > ys)) & (
            xs < (x0 - x1) * (ys - y1) / np.where(y0 - y1 == 0, 1e-9, y0 - y1) + x1
        )
        inside ^= crosses
        x0, y0 = x1, y1
    return inside


def points_in_any(polys, xs, ys) -> np.ndarray:
    out = np.zeros(np.asarray(xs).shape, dtype=bool)
    for p in polys:
        out |= points_in_poly(p, xs, ys)
    return out


def expand_poly(poly, margin: float):
    """Push polygon vertices away from the centroid by `margin` px (rough buffer)."""
    poly = np.asarray(poly, dtype=float)
    c = poly.mean(axis=0)
    d = poly - c
    n = np.linalg.norm(d, axis=1, keepdims=True).clip(min=1e-9)
    return poly + d / n * margin


def side_of_line(line, xs, ys) -> np.ndarray:
    """Sign of cross product: which side of the infinite line each point is on."""
    (ax, ay), (bx, by) = np.asarray(line, dtype=float)
    return np.sign((bx - ax) * (np.asarray(ys) - ay) - (by - ay) * (np.asarray(xs) - ax))


def segments_intersect(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return o1 != o2 and o3 != o4 and 0 not in (o1, o2, o3, o4)


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def containment(inner, outer) -> float:
    """Fraction of `inner` box covered by `outer` box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area
