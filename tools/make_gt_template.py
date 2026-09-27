"""Create data/labels/dev_gt.json with duration/fps for every sample video and empty event lists.
Then fill "events" by hand while watching the videos: [start_sec, end_sec, label]."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import sample_videos  # noqa: E402
from src.detection import video_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data/labels/dev_gt.json"
existing = json.loads(out.read_text()) if out.exists() else {}
for v in sample_videos():
    m = video_meta(str(v))
    existing.setdefault(v.name, {"duration": round(m["duration"], 2), "fps": m["fps"], "events": []})
out.write_text(json.dumps(existing, indent=2))
print(f"wrote {out} ({len(existing)} videos)")
