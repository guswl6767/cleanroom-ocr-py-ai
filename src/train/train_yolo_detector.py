"""
YOLOv8n 등번호 영역 detector 학습 코드

입력:
- /root/Storage/cleanroom-ocr-py-ai/Data/output/number_crops/bboxes/images
- /root/Storage/cleanroom-ocr-py-ai/Data/output/number_crops/bboxes/labels

출력:
- train/val split 데이터셋
- data.yaml
- YOLOv8n 학습 결과
"""

import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


# =========================================================
# 경로 설정
# =========================================================

SOURCE_IMG_DIR = Path("/root/Storage/cleanroom-ocr-py-ai/Data/output/number_crops/bboxes/images")
SOURCE_LBL_DIR = Path("/root/Storage/cleanroom-ocr-py-ai/Data/output/number_crops/bboxes/labels")

DATASET_ROOT = Path("/root/Storage/cleanroom-ocr-py-ai/Data/origin/train_yolo_detector")

TRAIN_IMG_DIR = DATASET_ROOT / "images" / "train"
VAL_IMG_DIR = DATASET_ROOT / "images" / "val"

TRAIN_LBL_DIR = DATASET_ROOT / "labels" / "train"
VAL_LBL_DIR = DATASET_ROOT / "labels" / "val"

DATA_YAML = DATASET_ROOT / "data.yaml"

MODEL_OUT_DIR = Path("/root/Storage/cleanroom-ocr-py-ai/Models/yolo_detector/back_num")


# =========================================================
# 설정
# =========================================================

VAL_RATIO = 0.2
SEED = 42

MODEL_NAME = "yolov8n.pt"

EPOCHS = 200
IMGSZ = 640
BATCH = 16
DEVICE = 0


# =========================================================
# 폴더 생성
# =========================================================

for d in [
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    TRAIN_LBL_DIR,
    VAL_LBL_DIR,
    MODEL_OUT_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# 기존 split 결과 초기화
# =========================================================

def clear_dir(dir_path: Path):
    for p in dir_path.glob("*"):
        if p.is_file():
            p.unlink()


for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
    clear_dir(d)


# =========================================================
# 이미지-라벨 쌍 수집
# =========================================================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

pairs = []

for img_path in SOURCE_IMG_DIR.iterdir():
    if img_path.suffix.lower() not in IMG_EXTS:
        continue

    label_path = SOURCE_LBL_DIR / f"{img_path.stem}.txt"

    if not label_path.exists():
        continue

    # label 파일이 비어 있으면 제외
    if label_path.stat().st_size == 0:
        continue

    pairs.append((img_path, label_path))


print(f"전체 이미지-라벨 쌍: {len(pairs)}")

if len(pairs) == 0:
    raise RuntimeError("학습 가능한 이미지-라벨 쌍이 없습니다.")


# =========================================================
# train / val split
# =========================================================

random.seed(SEED)
random.shuffle(pairs)

val_count = int(len(pairs) * VAL_RATIO)

val_pairs = pairs[:val_count]
train_pairs = pairs[val_count:]

print(f"train: {len(train_pairs)}")
print(f"val  : {len(val_pairs)}")


# =========================================================
# 파일 복사
# =========================================================

def copy_pairs(pairs, img_out_dir, lbl_out_dir):
    for img_path, label_path in pairs:
        shutil.copy2(img_path, img_out_dir / img_path.name)
        shutil.copy2(label_path, lbl_out_dir / label_path.name)


copy_pairs(train_pairs, TRAIN_IMG_DIR, TRAIN_LBL_DIR)
copy_pairs(val_pairs, VAL_IMG_DIR, VAL_LBL_DIR)


# =========================================================
# data.yaml 생성
# =========================================================

yaml_text = f"""path: {DATASET_ROOT}
train: images/train
val: images/val

names:
  0: back_number
"""

with open(DATA_YAML, "w", encoding="utf-8") as f:
    f.write(yaml_text)

print(f"data.yaml 저장 완료: {DATA_YAML}")


# =========================================================
# YOLOv8n 학습
# =========================================================

model = YOLO(MODEL_NAME)

results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    project=str(MODEL_OUT_DIR),
    name="yolov8n_back_number",
    exist_ok=True,
    workers=4,
    patience=30,
    pretrained=True,
    optimizer="auto",
    verbose=True,
    plots=True,
)


print("\n" + "=" * 60)
print("학습 완료")
print("=" * 60)
print(f"결과 폴더: {MODEL_OUT_DIR / 'yolov8n_back_number'}")
print(f"best.pt: {MODEL_OUT_DIR / 'yolov8n_back_number' / 'weights' / 'best.pt'}")
print(f"last.pt: {MODEL_OUT_DIR / 'yolov8n_back_number' / 'weights' / 'last.pt'}")
print("=" * 60)