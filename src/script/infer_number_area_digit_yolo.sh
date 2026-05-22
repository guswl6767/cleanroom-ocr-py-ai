# python /root/Projects/cleanroom-ocr-py-ai/src/infer/infer_number_area_digit_yolo.py \
#   --video /root/Storage/cleanroom-ocr-py-ai/Data/origin/test/107-Clip-20260416_095515_50.mp4 \
#   --pose_model yolo26m-pose.pt \
#   --number_detector /root/Storage/cleanroom-ocr-py-ai/Models/yolo_detector/back_num/yolov8n_back_number/weights/best.pt \
#   --digit_detector /root/Storage/cleanroom-ocr-py-ai/Models/digit_yolo26n/digit_yolo26n_v1-2/weights/best.pt \
#   --out_dir /root/Storage/cleanroom-ocr-py-ai/Data/output/digit \
#   --device 0 \
#   --frame_step 1 \
#   --pad_x 0.0 \
#   --pad_y 0.0 \
#   --digit_conf 0.15 \
#   --digit_imgsz 640 \
#   --use_trim \
#   --save_crops \
#   --save_raw_trim_pair

python /root/Projects/cleanroom-ocr-py-ai/src/infer/infer_number_area_digit_yolo_v2.py \
  --video /root/Storage/cleanroom-ocr-py-ai/Data/origin/test/107-Clip-20260416_095515_59.mp4 \
  --pose_model yolo26m-pose.pt \
  --number_detector /root/Storage/cleanroom-ocr-py-ai/Models/yolo_detector/back_num/yolov8n_back_number/weights/best.pt \
  --digit_detector /root/Storage/cleanroom-ocr-py-ai/Models/digit_yolo26n/digit_yolo26n_v2-2/weights/best.pt \
  --out_dir /root/Storage/cleanroom-ocr-py-ai/Data/output/digit \
  --device 0 \
  --frame_step 1 \
  --pad_x 0.0 \
  --pad_y 0.0 \
  --digit_conf 0.15 \
  --digit_imgsz 640 \
  --use_trim \
  --ultra_aggressive \
  --trim_pad_x 0.06 \
  --trim_pad_y 0.08 \
  --dark_thresh 190 \
  --min_dark_area 10 \
  --edge_margin_x 0.01 \
  --edge_margin_y 0.03 \
  --save_raw_trim_pair