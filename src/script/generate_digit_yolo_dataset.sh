#!/bin/bash

set -e

PYTHON=$(command -v python || command -v python3)
if [ -z "$PYTHON" ]; then
  echo "python not found"
  exit 1
fi

BASE_DIR="/root/Projects/cleanroom-ocr-py-ai"
INPUT_DIR="/root/Storage/cleanroom-ocr-py-ai/Data/origin/numbers"
OUT_DIR="/root/Storage/cleanroom-ocr-py-ai/Data/digit_yolo_dataset"
POSE_MODEL="yolo26m-pose.pt"
NUMBER_MODEL="/root/Storage/cleanroom-ocr-py-ai/Models/yolo_detector/back_num/yolov8n_back_number/weights/best.pt"
OCR_BACKEND="easyocr"
OCR_MODEL=""
DEVICE="0"

mkdir -p "$OUT_DIR"

$PYTHON "$BASE_DIR/src/script/generate_digit_yolo_dataset.py" \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_DIR" \
  --pose_model "$POSE_MODEL" \
  --number_detector "$NUMBER_MODEL" \
  --ocr_backend "$OCR_BACKEND" \
  --ocr_lang "en" \
  --ocr_gpu \
  --device "$DEVICE" \
  --trim \
  --ultra_aggressive \
  --save_debug \
  --train_ratio 0.85 \
  --seed 42

echo "Dataset generation finished: $OUT_DIR"
