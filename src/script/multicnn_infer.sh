python multicnn_infer_v2.py \
  --video /root/Storage/cleanroom-ocr-py-ai/Data/origin/test/107-Clip-20260416_102609_17.mp4 \
  --ckpt /root/Storage/cleanroom-ocr-py-ai/Models/ocr_train_v4/best_digit_cnn.pt \
  --pose_model yolov8n-pose.pt \
  --out /root/Storage/cleanroom-ocr-py-ai/Data/output/ocr_train_v4/107-Clip-20260416_102609_17_result.mp4 \
  --csv_out /root/Storage/cleanroom-ocr-py-ai/Data/output/ocr_train_v4/107-Clip-20260416_102609_17_result.csv \
  --roi_mode person \
  --device cuda \
  --number_conf_th 0.65 \
  --number_history 20 \
  --number_min_hits 5 \
  --number_lock_ratio 0.55

# python train_crnn_ctc.py \
#     --csv /root/.../train_person_only.csv,/root/.../synthetic/labels.csv \
#     --out_dir ./crnn_runs \
#     --epochs 80 \
#     --batch_size 32 \
#     --img_w 128 \
#     --img_h 256 \
#     --lr 5e-4 \
#     --device cuda