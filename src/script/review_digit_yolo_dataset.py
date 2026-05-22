import argparse
import os
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = [str(i) for i in range(10)]


def can_show_windows():
    if os.name != "nt" and "DISPLAY" not in os.environ:
        return False
    try:
        cv2.namedWindow("_cv2_test_window", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_cv2_test_window")
        return True
    except cv2.error:
        return False



def collect_images(image_dir: Path):
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def read_yolo_label(label_path: Path):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        boxes.append((cls, xc, yc, w, h))
    return boxes


def yolo_to_xyxy(box, width, height):
    cls, xc, yc, w, h = box
    x1 = int((xc - w / 2.0) * width)
    y1 = int((yc - h / 2.0) * height)
    x2 = int((xc + w / 2.0) * width)
    y2 = int((yc + h / 2.0) * height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return cls, x1, y1, x2, y2


def draw_preview(image, yolo_boxes, label_path=None):
    preview = image.copy()
    h, w = preview.shape[:2]
    for box in yolo_boxes:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
        color = (0, 255, 0)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        text = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
        cv2.putText(preview, text, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if label_path is not None:
        info = f"{label_path.name}"
        cv2.putText(preview, info, (10, preview.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return preview


def main():
    parser = argparse.ArgumentParser(description="Review YOLO digit dataset annotations by generating overlay previews")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Generated YOLO dataset root containing images/ and labels/")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Preview output folder. Defaults to <dataset_dir>/review_previews")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val"],
                        help="Which split to process")
    parser.add_argument("--show", action="store_true",
                        help="Display preview images interactively")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Skip images without matching label files")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    assert images_dir.exists(), f"Dataset images folder not found: {images_dir}"
    assert labels_dir.exists(), f"Dataset labels folder not found: {labels_dir}"

    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir / "review_previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"] if args.split == "all" else [args.split]
    total = 0

    show_enabled = args.show and can_show_windows()
    if args.show and not show_enabled:
        print("Warning: OpenCV GUI support is unavailable; image preview display is disabled.")

    for split in splits:
        split_images = images_dir / split
        split_labels = labels_dir / split
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        if not split_images.exists():
            print(f"Warning: image split folder not found: {split_images}")
            continue
        if not split_labels.exists():
            print(f"Warning: label split folder not found: {split_labels}")
            continue

        image_paths = collect_images(split_images)
        for image_path in image_paths:
            label_path = split_labels / (image_path.stem + ".txt")
            if not label_path.exists():
                if args.skip_missing:
                    continue
                print(f"Missing label for image: {image_path}")
                preview = image = cv2.imread(str(image_path))
                if image is None:
                    continue
                preview = cv2.putText(preview, "MISSING LABEL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                save_path = split_out / image_path.name
                cv2.imwrite(str(save_path), preview)
                total += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            yolo_boxes = read_yolo_label(label_path)
            preview = draw_preview(image, yolo_boxes, label_path=label_path)
            save_path = split_out / image_path.name
            cv2.imwrite(str(save_path), preview)
            total += 1

            if show_enabled:
                try:
                    cv2.imshow("YOLO Review", preview)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error:
                    print("Warning: OpenCV GUI failed during display; continuing without interactive preview.")
                    show_enabled = False

        if show_enabled:
            cv2.destroyAllWindows()

    print(f"Saved {total} review previews to {out_dir}")


if __name__ == "__main__":
    main()
