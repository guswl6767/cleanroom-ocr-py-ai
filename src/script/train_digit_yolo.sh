python /root/Projects/cleanroom-ocr-py-ai/src/train/train_digit_yolo.py \
  --data /root/Storage/cleanroom-ocr-py-ai/Data/merged_digit_dataset/data.yaml \
  --model yolo26n.pt \
  --imgsz 640 \
  --epochs 50 \
  --batch 16 \
  --device 0 \
  --project /root/Storage/cleanroom-ocr-py-ai/Models/digit_yolo26n \
  --name digit_yolo26n_v2