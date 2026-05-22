import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, w, h, pad_x=0.0, pad_y=0.0):
    bw = x2 - x1
    bh = y2 - y1

    px = bw * pad_x
    py = bh * pad_y

    nx1 = x1 - px
    ny1 = y1 - py
    nx2 = x2 + px
    ny2 = y2 + py

    return clamp_box(nx1, ny1, nx2, ny2, w, h)


def expand_box_margin(x1, y1, x2, y2, w, h, margin_ratio=0.05):
    bw = x2 - x1
    bh = y2 - y1
    return expand_box(x1, y1, x2, y2, w, h, pad_x=margin_ratio, pad_y=margin_ratio)


def is_trim_too_aggressive(original_shape, trimmed_shape, min_ratio=0.6):
    orig_h, orig_w = original_shape[:2]
    trim_h, trim_w = trimmed_shape[:2]
    if orig_w == 0 or orig_h == 0:
        return False
    if trim_w < int(orig_w * min_ratio) or trim_h < int(orig_h * min_ratio):
        return True
    return False


def is_cpu_device(device):
    return isinstance(device, str) and device.lower() == "cpu"


def init_easyocr_reader(use_gpu=False, lang="en"):
    try:
        import easyocr
    except ImportError as e:
        raise ImportError("easyocr is required for --ocr_backend easyocr. Install with pip install easyocr") from e
    return easyocr.Reader([lang], gpu=use_gpu)


def init_paddleocr_reader(use_gpu=False, lang="en"):
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise ImportError("paddleocr is required for --ocr_backend paddleocr. Install with pip install paddleocr") from e
    return PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)


def run_easyocr(reader, img):
    return reader.readtext(
        img,
        allowlist="0123456789",
        detail=1,
        paragraph=False,
        decoder="beamsearch",
        batch_size=1,
        text_threshold=0.45,
        low_text=0.25,
        link_threshold=0.25,
        mag_ratio=1.5,
    )


def run_paddleocr(reader, img):
    results = reader.ocr(np.asarray(img), det=True, rec=True, cls=True)
    parsed = []
    for item in results:
        bbox, text_info = item
        if len(text_info) != 2:
            continue
        raw_text, conf = text_info
        parsed.append((bbox, raw_text, float(conf)))
    return parsed


def ocr_results_to_digit_boxes(ocr_results, img_shape):
    digit_boxes = []
    for bbox, raw_text, conf in ocr_results:
        digits = re.findall(r"\d", raw_text)
        if not digits:
            continue

        coords = np.array(bbox, dtype=float)
        if coords.shape[0] == 4:
            xs = coords[:, 0]
            ys = coords[:, 1]
            left, right = xs.min(), xs.max()
            top, bottom = ys.min(), ys.max()
        else:
            flat = coords.flatten()
            left, top, right, bottom = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])

        width = max(1.0, right - left)
        height = max(1.0, bottom - top)
        count = len(digits)

        if count == 1:
            expanded = expand_box_margin(left, top, right, bottom, img_shape[1], img_shape[0], margin_ratio=0.05)
            if expanded is None:
                continue
            x1, y1, x2, y2 = expanded
            digit_boxes.append((int(digits[0]), x1, y1, x2, y2))
            continue

        segment = width / count
        for index, digit in enumerate(digits):
            x1 = left + index * segment
            x2 = left + (index + 1) * segment
            y1 = top
            y2 = bottom
            expanded = expand_box_margin(x1, y1, x2, y2, img_shape[1], img_shape[0], margin_ratio=0.05)
            if expanded is None:
                continue
            x1, y1, x2, y2 = expanded
            if x2 > x1 and y2 > y1:
                digit_boxes.append((int(digit), x1, y1, x2, y2))

    return digit_boxes


def write_yolo_label(label_path, boxes, crop_w, crop_h):
    lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        xc = (x1 + x2) / 2.0 / crop_w
        yc = (y1 + y2) / 2.0 / crop_h
        nw = bw / crop_w
        nh = bh / crop_h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    if lines:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")
        return True
    return False


def save_dataset_yaml(out_dir, train_rel, val_rel):
    yaml_path = Path(out_dir) / "digit_yolo_data.yaml"
    yaml_text = f"path: {Path(out_dir).resolve()}\n"
    yaml_text += f"train: {train_rel}\n"
    yaml_text += f"val: {val_rel}\n"
    yaml_text += "nc: 10\n"
    yaml_text += "names: ['0','1','2','3','4','5','6','7','8','9']\n"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate YOLO training dataset from person->number->digit pipeline"
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="원본 이미지 폴더")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="생성할 YOLO 데이터셋 폴더")
    parser.add_argument("--pose_model", type=str, required=True,
                        help="pose detector weight or model 파일")
    parser.add_argument("--number_detector", type=str, required=True,
                        help="숫자 영역 detector weight")
    parser.add_argument("--ocr_backend", type=str, default="yolo",
                        choices=["yolo", "easyocr", "paddleocr"],
                        help="OCR backend to use for text detection")
    parser.add_argument("--ocr_model", type=str, default=None,
                        help="digit OCR detector weight (YOLO) or OCR model path. required only for ocr_backend=yolo")
    parser.add_argument("--ocr_lang", type=str, default="en",
                        help="OCR language for easyocr/paddleocr")
    parser.add_argument("--ocr_gpu", action="store_true",
                        help="Enable GPU for easyocr/paddleocr")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--person_conf", type=float, default=0.3)
    parser.add_argument("--number_conf", type=float, default=0.3)
    parser.add_argument("--digit_conf", type=float, default=0.25)
    parser.add_argument("--person_iou", type=float, default=0.45)
    parser.add_argument("--number_iou", type=float, default=0.45)
    parser.add_argument("--digit_iou", type=float, default=0.45)
    parser.add_argument("--pad_x", type=float, default=0.0,
                        help="숫자 영역 crop 양옆 여유 비율")
    parser.add_argument("--pad_y", type=float, default=0.0,
                        help="숫자 영역 crop 상하 여유 비율")
    parser.add_argument("--trim", action="store_true",
                        help="숫자 영역 crop 후 추가 trim 적용")
    parser.add_argument("--trim_pad_x", type=float, default=0.08)
    parser.add_argument("--trim_pad_y", type=float, default=0.12)
    parser.add_argument("--dark_thresh", type=int, default=200)
    parser.add_argument("--min_dark_area", type=int, default=20)
    parser.add_argument("--ultra_aggressive", action="store_true",
                        help="trim을 더 공격적으로 수행")
    parser.add_argument("--save_debug", action="store_true",
                        help="디버그용 overlay 이미지를 저장")
    parser.add_argument("--train_ratio", type=float, default=0.85,
                        help="train/val 나눌 비율")
    parser.add_argument("--max_images", type=int, default=0,
                        help="처리할 최대 이미지 개수. 0이면 모두 처리")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def trim_number_crop_by_dark_region(
    crop,
    pad_x=0.08,
    pad_y=0.12,
    dark_thresh=200,
    min_area=20,
    aggressive_trim=True,
    edge_margin_x=0.02,
    edge_margin_y=0.05,
    ultra_aggressive=False,
):
    if crop is None or crop.size == 0:
        return crop, (0, 0)

    h, w = crop.shape[:2]
    if h <= 5 or w <= 5:
        return crop, (0, 0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if ultra_aggressive:
        effective_dark_thresh = int(dark_thresh * 0.8)
        min_area_eff = max(5, int(min_area * 0.5))
    else:
        effective_dark_thresh = dark_thresh
        min_area_eff = min_area

    mask = (gray < effective_dark_thresh).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if ultra_aggressive:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel_large = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_eff:
            clean_mask[labels == i] = 255

    ys, xs = np.where(clean_mask > 0)
    if len(xs) < 5 or len(ys) < 5:
        return crop, (0, 0)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 5 or bh <= 5:
        return crop, (0, 0)

    px = int(bw * pad_x)
    py = int(bh * pad_y)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w - 1, x2 + px)
    y2 = min(h - 1, y2 + py)

    if aggressive_trim:
        trimmed_temp = crop[y1:y2, x1:x2]
        th, tw = trimmed_temp.shape[:2]
        gray_temp = cv2.cvtColor(trimmed_temp, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_temp, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_ys, edge_xs = np.where(edges > 0)
        if len(edge_xs) > 10 and len(edge_ys) > 10:
            ex1 = max(0, int(edge_xs.min()))
            ex2 = min(tw - 1, int(edge_xs.max()))
            ey1 = max(0, int(edge_ys.min()))
            ey2 = min(th - 1, int(edge_ys.max()))
            edge_bw = ex2 - ex1
            edge_bh = ey2 - ey1
            if ultra_aggressive:
                margin_x = max(1, int(edge_bw * edge_margin_x * 0.5))
                margin_y = max(1, int(edge_bh * edge_margin_y * 0.5))
            else:
                margin_x = max(1, int(edge_bw * edge_margin_x))
                margin_y = max(1, int(edge_bh * edge_margin_y))
            ex1 = max(0, ex1 - margin_x)
            ey1 = max(0, ey1 - margin_y)
            ex2 = min(tw - 1, ex2 + margin_x)
            ey2 = min(th - 1, ey2 + margin_y)
            if ex2 > ex1 and ey2 > ey1:
                x1 = x1 + ex1
                y1 = y1 + ey1
                x2 = x1 + (ex2 - ex1)
                y2 = y1 + (ey2 - ey1)

    trimmed = crop[y1:y2, x1:x2]
    if trimmed.size == 0:
        return crop, (0, 0)

    if is_trim_too_aggressive(crop.shape, trimmed.shape, min_ratio=0.6):
        return crop, (0, 0)

    return trimmed, (x1, y1)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    assert input_dir.exists(), f"input_dir not found: {input_dir}"

    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    debug_out = out_dir / "debug"

    images_out.joinpath("train").mkdir(parents=True, exist_ok=True)
    images_out.joinpath("val").mkdir(parents=True, exist_ok=True)
    labels_out.joinpath("train").mkdir(parents=True, exist_ok=True)
    labels_out.joinpath("val").mkdir(parents=True, exist_ok=True)
    if args.save_debug:
        debug_out.mkdir(parents=True, exist_ok=True)

    pose_model = YOLO(args.pose_model)
    number_model = YOLO(args.number_detector)

    if args.ocr_backend == "yolo":
        assert args.ocr_model, "--ocr_model is required when --ocr_backend=yolo"
        digit_model = YOLO(args.ocr_model)
        ocr_reader = None
    elif args.ocr_backend == "easyocr":
        ocr_reader = init_easyocr_reader(use_gpu=args.ocr_gpu or not is_cpu_device(args.device), lang=args.ocr_lang)
        digit_model = None
    elif args.ocr_backend == "paddleocr":
        ocr_reader = init_paddleocr_reader(use_gpu=args.ocr_gpu or not is_cpu_device(args.device), lang=args.ocr_lang)
        digit_model = None
    else:
        raise ValueError(f"Unsupported OCR backend: {args.ocr_backend}")

    image_paths = sorted([p for p in input_dir.glob("**/*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    rng = np.random.default_rng(args.seed)
    total_saved = 0
    total_skipped = 0

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        pose_results = pose_model.predict(
            image,
            conf=args.person_conf,
            iou=args.person_iou,
            device=args.device,
            verbose=False,
        )

        sample_saved = False
        person_idx = 0
        for result in pose_results:
            if result.boxes is None:
                continue
            person_boxes = result.boxes.xyxy.cpu().numpy()
            person_confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(person_boxes))

            for pbox in person_boxes:
                px1, py1, px2, py2 = pbox
                pbox_clamped = clamp_box(px1, py1, px2, py2, width, height)
                if pbox_clamped is None:
                    continue
                px1, py1, px2, py2 = pbox_clamped
                person_crop = image[py1:py2, px1:px2]
                if person_crop.size == 0:
                    continue

                number_results = number_model.predict(
                    person_crop,
                    conf=args.number_conf,
                    iou=args.number_iou,
                    device=args.device,
                    verbose=False,
                )

                if len(number_results) == 0 or number_results[0].boxes is None:
                    person_idx += 1
                    continue

                nboxes = number_results[0].boxes.xyxy.cpu().numpy()
                nconfs = number_results[0].boxes.conf.cpu().numpy() if number_results[0].boxes.conf is not None else np.ones(len(nboxes))
                if len(nboxes) == 0:
                    person_idx += 1
                    continue

                best_idx = int(np.argmax(nconfs))
                bx1, by1, bx2, by2 = nboxes[best_idx]
                crop_box = expand_box(
                    bx1,
                    by1,
                    bx2,
                    by2,
                    person_crop.shape[1],
                    person_crop.shape[0],
                    pad_x=args.pad_x,
                    pad_y=args.pad_y,
                )
                if crop_box is None:
                    person_idx += 1
                    continue

                bx1, by1, bx2, by2 = crop_box
                number_crop = person_crop[by1:by2, bx1:bx2]
                if number_crop.size == 0:
                    person_idx += 1
                    continue

                if args.trim:
                    number_crop, trim_offset = trim_number_crop_by_dark_region(
                        number_crop,
                        pad_x=args.trim_pad_x,
                        pad_y=args.trim_pad_y,
                        dark_thresh=args.dark_thresh,
                        min_area=args.min_dark_area,
                        aggressive_trim=True,
                        ultra_aggressive=args.ultra_aggressive,
                    )
                else:
                    trim_offset = (0, 0)

                if args.ocr_backend == "yolo":
                    digits_results = digit_model.predict(
                        number_crop,
                        conf=args.digit_conf,
                        iou=args.digit_iou,
                        device=args.device,
                        verbose=False,
                    )

                    if len(digits_results) == 0 or digits_results[0].boxes is None:
                        person_idx += 1
                        continue

                    digit_boxes = digits_results[0].boxes.xyxy.cpu().numpy()
                    digit_classes = digits_results[0].boxes.cls.cpu().numpy().astype(int)
                    digit_confs = digits_results[0].boxes.conf.cpu().numpy()

                    if len(digit_boxes) == 0:
                        person_idx += 1
                        continue

                    norm_boxes = []
                    for cls_id, box in zip(digit_classes, digit_boxes):
                        dx1, dy1, dx2, dy2 = box
                        expanded = expand_box_margin(dx1, dy1, dx2, dy2, number_crop.shape[1], number_crop.shape[0], margin_ratio=0.05)
                        if expanded is None:
                            continue
                        dx1, dy1, dx2, dy2 = expanded
                        if dx2 <= dx1 or dy2 <= dy1:
                            continue
                        norm_boxes.append((int(cls_id), dx1, dy1, dx2, dy2))
                else:
                    if args.ocr_backend == "easyocr":
                        ocr_results = run_easyocr(ocr_reader, number_crop)
                    else:
                        ocr_results = run_paddleocr(ocr_reader, number_crop)

                    norm_boxes = ocr_results_to_digit_boxes(ocr_results, number_crop.shape)

                    if len(norm_boxes) == 0:
                        person_idx += 1
                        continue

                if not norm_boxes:
                    person_idx += 1
                    continue

                split = "train" if rng.random() < args.train_ratio else "val"
                out_image_name = f"{image_path.stem}_p{person_idx}.jpg"
                image_out_path = images_out / split / out_image_name
                label_out_path = labels_out / split / f"{image_path.stem}_p{person_idx}.txt"

                cv2.imwrite(str(image_out_path), number_crop)
                saved = write_yolo_label(label_out_path, norm_boxes, number_crop.shape[1], number_crop.shape[0])
                if not saved:
                    image_out_path.unlink(missing_ok=True)
                    person_idx += 1
                    continue

                total_saved += 1
                sample_saved = True

                if args.save_debug:
                    debug_image = number_crop.copy()
                    for cls_id, x1, y1, x2, y2 in norm_boxes:
                        cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(debug_image, str(cls_id), (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    debug_out_path = debug_out / f"{image_path.stem}_p{person_idx}.jpg"
                    cv2.imwrite(str(debug_out_path), debug_image)

                person_idx += 1

        if not sample_saved:
            total_skipped += 1

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(image_paths)}] saved={total_saved} skipped={total_skipped}")

    yaml_path = save_dataset_yaml(out_dir, "images/train", "images/val")
    print("\n=== DONE ===")
    print(f"saved samples: {total_saved}")
    print(f"skipped images: {total_skipped}")
    print(f"dataset yaml: {yaml_path}")
    print("train images:", len(list((images_out / "train").glob("*.jpg"))) )
    print("val images:", len(list((images_out / "val").glob("*.jpg"))) )


if __name__ == "__main__":
    main()
