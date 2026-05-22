python /root/Projects/cleanroom-ocr-py-ai/src/infer/infer_detector_pipeline.py \
    --video /root/Storage/cleanroom-ocr-py-ai/Data/origin/test/107-Clip-20260416_102609_17.mp4 \
    --pose_model yolov8n-pose.pt \
    --number_detector /root/Storage/cleanroom-ocr-py-ai/Models/yolo_detector/back_num/yolov8n_back_number/weights/best.pt \
    --recognizer_ckpt /root/Storage/cleanroom-ocr-py-ai/Models/ocr_train_v4/best_digit_cnn.pt \
    --out_dir /root/Storage/cleanroom-ocr-py-ai/Data/output/infer_detector_pipeline \
    --device cuda