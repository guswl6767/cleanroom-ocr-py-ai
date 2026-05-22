python /root/Projects/cleanroom-ocr-py-ai/src/infer/infer_yolo_pose_pipeline.py \
    --video /root/Storage/cleanroom-ocr-py-ai/Data/origin/test/107-Clip-20260416_102609_17.mp4 \
    --ckpt /root/Storage/cleanroom-ocr-py-ai/Models/ocr_train_v4/best_digit_cnn.pt \
    --pose_model yolov8n-pose.pt \
    --out_dir /root/Storage/cleanroom-ocr-py-ai/Data/output/cnn_train_v4 \
    --device cuda