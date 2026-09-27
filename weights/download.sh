#!/usr/bin/env bash
# Run once, with internet, before evaluation.
set -e
cd "$(dirname "$0")"
[ -f yolov8s.pt ] || curl -L -o yolov8s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
echo "weights ready"
