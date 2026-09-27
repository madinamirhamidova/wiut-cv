"""Shared helper: find when a vehicle crosses the stop line towards the intersection."""
import numpy as np

from ..geometry import side_of_line


def crossings_towards_intersection(g, stop_line, intersection):
    """Yield row positions i where the ground point moves from approach side to intersection side."""
    c = intersection.mean(axis=0)
    int_side = side_of_line(stop_line, [c[0]], [c[1]])[0]
    sides = side_of_line(stop_line, g.sx.values, g.sy.values)
    a, b = stop_line
    seg = b - a
    seg_len2 = float(seg @ seg)
    for i in range(1, len(g)):
        if sides[i - 1] == -int_side and sides[i] == int_side:
            pt = np.array([g.sx.values[i], g.sy.values[i]])
            u = float((pt - a) @ seg) / seg_len2      # projection along the line
            if -0.1 <= u <= 1.1:                        # crossed within the painted line
                yield i
