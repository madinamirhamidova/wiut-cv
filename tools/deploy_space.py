"""Publish the website + live demo as a Hugging Face Space (Gradio SDK, free CPU hardware is fine).

  pip install huggingface_hub
  hf auth login                                  # once, with a *write* token from huggingface.co/settings/tokens
  python tools/build_site.py                     # assets must exist first
  python tools/deploy_space.py <hf-user>/<space-name>

Creates the Space if needed and uploads app/, src/, configs/, tools/, solution.py and the weights.
The Space installs the CPU build of torch (much smaller); the demo switches to a faster detector
setting automatically when there is no GPU.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import gradio

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ["app", "src", "configs", "tools"]
FILES = ["solution.py", "weights/yolov8s.pt"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")

SPACE_README = """---
title: Traffic Event Detection
emoji: 🚦
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: {gradio}
python_version: "3.11"
app_file: app/app.py
pinned: false
short_description: Traffic events + accident risk from a fixed road camera
---

WIUT Hackathon 2026, Computer Vision track: website and live demo. Code: see the Links tab.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="<hf-user>/<space-name>")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    if not (ROOT / "app/assets/results/index.json").exists():
        sys.exit("app/assets is empty: run  python tools/build_site.py  first")
    for f in FILES:
        if not (ROOT / f).exists():
            sys.exit(f"missing {f} (weights: bash weights/download.sh)")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for d in FOLDERS:
            shutil.copytree(ROOT / d, stage / d, ignore=IGNORE)
        for f in FILES:
            (stage / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, stage / f)
        reqs = (ROOT / "requirements.txt").read_text()
        (stage / "requirements.txt").write_text(
            "--extra-index-url https://download.pytorch.org/whl/cpu\n" + reqs + "plotly==7.1.0\n")
        (stage / "README.md").write_text(SPACE_README.format(gradio=gradio.__version__))

        api = HfApi()
        api.create_repo(a.repo_id, repo_type="space", space_sdk="gradio", private=a.private, exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=a.repo_id, repo_type="space",
                          commit_message="Deploy website + demo")
    print(f"deployed: https://huggingface.co/spaces/{a.repo_id}  (first build takes a few minutes)")


if __name__ == "__main__":
    main()
