# python /root/Projects/cleanroom-ocr-py-ai/src/infer/infer_number_area_contour.py \
#   --video /root/Storage/cleanroom-ocr-py-ai/Data/origin/test/107-Clip-20260416_095515_50.mp4 \
#   --pose_model yolo26m-pose.pt \
#   --number_detector /root/Storage/cleanroom-ocr-py-ai/Models/yolo_detector/back_num/yolov8n_back_number/weights/best.pt \
#   --digit_detector /root/Storage/cleanroom-ocr-py-ai/Models/digit_yolo26n/digit_yolo26n_v1-2/weights/best.pt \
#   --out_dir /root/Storage/cleanroom-ocr-py-ai/Data/output/digit \
#   --device cuda \
#   --frame_step 1 \
#   --pad_x 0.0 \
#   --pad_y 0.0 \
#   --use_trim \
#   --digit_conf 0.15 \
#   --digit_imgsz 640 \
#   --confirm_frames 5 \
#   --confirm_ratio 0.6 \
#   --draw_pending \
#   --save_debug_csv

python /root/Projects/cleanroom-ocr-py-ai/src/infer/infer_number_area_contour.py \
  --video /root/Storage/cleanroom-ocr-py-ai/Data/origin/test/107-Clip-20260416_095515_50.mp4 \
  --pose_model yolo26m-pose.pt \
  --number_detector /root/Storage/cleanroom-ocr-py-ai/Models/yolo_detector/back_num/yolov8n_back_number/weights/best.pt \
  --digit_detector /root/Storage/cleanroom-ocr-py-ai/Models/digit_yolo26n/digit_yolo26n_v1-2/weights/best.pt \
  --out_dir /root/Storage/cleanroom-ocr-py-ai/Data/output/digit \
  --device cuda \
  --frame_step 1 \
  --use_trim \
  --save_crops \
  --save_raw_trim_pair \
  --save_debug_csv \
  --draw_pending