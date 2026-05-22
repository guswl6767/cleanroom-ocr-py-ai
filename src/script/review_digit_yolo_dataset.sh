#!/bin/bash

set -e

PYTHON=$(command -v python || command -v python3)
if [ -z "$PYTHON" ]; then
  echo "python not found"
  exit 1
fi

BASE_DIR="/root/Projects/cleanroom-ocr-py-ai"
DATASET_DIR="/root/Storage/cleanroom-ocr-py-ai/Data/digit_yolo_dataset"
OUT_DIR="/root/Storage/cleanroom-ocr-py-ai/Data/digit_yolo_dataset_review"

mkdir -p "$OUT_DIR"

$PYTHON "$BASE_DIR/src/script/review_digit_yolo_dataset.py" \
  --dataset_dir "$DATASET_DIR" \
  --out_dir "$OUT_DIR"

echo "Review previews saved to: $OUT_DIR"
