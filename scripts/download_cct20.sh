#!/usr/bin/env bash
# Download the CCT20 benchmark subset of Caltech Camera Traps (LILA BC).
# Images: ~6 GB (resized to max 1024 px), annotations: ~3 MB.
# License: Community Data License Agreement (permissive).
set -euo pipefail

RAW=data/raw
BASE=https://storage.googleapis.com/public-datasets-lila/caltechcameratraps
mkdir -p "$RAW"

echo "Downloading annotations..."
curl -L -o "$RAW/eccv_18_annotations.tar.gz" "$BASE/eccv_18_annotations.tar.gz"
tar -xzf "$RAW/eccv_18_annotations.tar.gz" -C "$RAW"

if [[ "${1:-}" != "--annotations-only" ]]; then
  echo "Downloading images (~6 GB)..."
  curl -L -C - -o "$RAW/eccv_18_all_images_sm.tar.gz" "$BASE/eccv_18_all_images_sm.tar.gz"
  tar -xzf "$RAW/eccv_18_all_images_sm.tar.gz" -C "$RAW"
fi

echo "Done. Contents of $RAW:"
ls "$RAW"
