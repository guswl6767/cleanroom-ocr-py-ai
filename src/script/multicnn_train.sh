python multicnn_train_v4.py \
    --csv /root/Storage/cleanroom-ocr-py-ai/Data/origin/ocr_auto_labels/train_csv/train.csv \
    --out_dir /root/Storage/cleanroom-ocr-py-ai/Models/ocr_train_v4 \
    --epochs 200 \
    --batch_size 32 \
    --img_w 192 \
    --img_h 384 

# python train_crnn_ctc.py \
#     --csv /root/Storage/cleanroom-ocr-py-ai/Data/origin/ocr_auto_labels/train_csv/train.csv \
#     --out_dir /root/Storage/cleanroom-ocr-py-ai/Models/crnn_runs/crop \
#     --epochs 200 \
#     --batch_size 32 \
#     --img_w 128 \
#     --img_h 256 \
#     --lr 5e-4 \
#     --device cuda