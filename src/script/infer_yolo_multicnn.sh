#!/bin/bash

VIDEO_NAME="107-Clip-20260416_095515_59"

python /root/Projects/cleanroom-ocr-py-ai/src/infer/infer_yolo_multicnn.py \
  --video /root/Storage/cleanroom-ocr-py-ai/Data/origin/test/${VIDEO_NAME}.mp4 \
  --ckpt /root/Storage/cleanroom-ocr-py-ai/Models/ocr_train_v4/best_digit_cnn.pt \
  --pose_model yolo26m-pose.pt \
  --number_detector /root/Storage/cleanroom-ocr-py-ai/Models/yolo_detector/back_num/yolov8n_back_number/weights/best.pt \
  --out /root/Storage/cleanroom-ocr-py-ai/Data/output/yolo_multicnn/${VIDEO_NAME}_result.mp4 \
  --csv_out /root/Storage/cleanroom-ocr-py-ai/Data/output/yolo_multicnn/${VIDEO_NAME}_result.csv \
  --device cuda \
  --number_conf_th 0.65 \
  --frame_step 3 \
  --number_history 20 \
  --number_min_hits 5 \
  --number_lock_ratio 0.55