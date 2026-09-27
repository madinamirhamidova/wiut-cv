"""Step 4: clean raw rule events so they match annotation conventions."""
from collections import defaultdict


def clean(events, duration: float, cfg: dict):
    """Per class: clip to [0, duration], merge segments closer than merge_gap,
    drop segments shorter than min_len. Guarantees no same-class overlaps."""
    by_cls = defaultdict(list)
    for s, e, lab in events:
        s, e = max(0.0, float(s)), min(float(duration), float(e))
        if e > s:
            by_cls[lab].append([s, e])
    out = []
    for lab, segs in by_cls.items():
        p = {**cfg.get("default", {}), **cfg.get(lab, {})}
        segs.sort()
        merged = [segs[0]]
        for s, e in segs[1:]:
            if s - merged[-1][1] <= p.get("merge_gap", 1.0):
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out += [[round(s, 2), round(e, 2), lab] for s, e in merged if e - s >= p.get("min_len", 0.5)]
    return sorted(out)
