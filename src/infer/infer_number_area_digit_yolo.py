import os
import csv
import cv2
import argparse
import numpy as np

from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque, Counter

from ultralytics import YOLO


# =========================================================
# util
# =========================================================

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
    """
    밝은 방호복 배경에서 어두운 숫자 영역만 찾아 crop을 타이트하게 줄임.
    aggressive_trim=True일 때 추가로 엣지 기반 여백 제거.
    ultra_aggressive=True일 때 더 강한 threshold와 morphology 적용.
    실패하면 원본 crop 반환.

    return:
        trimmed_crop, (offset_x, offset_y)
    """

    if crop is None or crop.size == 0:
        return crop, (0, 0)

    h, w = crop.shape[:2]

    if h <= 5 or w <= 5:
        return crop, (0, 0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ultra_aggressive 모드: 더 낮은 threshold 사용
    if ultra_aggressive:
        effective_dark_thresh = int(dark_thresh * 0.8)  # 80% of original
        min_area_eff = max(5, int(min_area * 0.5))      # 50% of original
    else:
        effective_dark_thresh = dark_thresh
        min_area_eff = min_area

    # 어두운 숫자 영역 추출
    mask = (gray < effective_dark_thresh).astype(np.uint8) * 255

    # 강화된 노이즈 제거 (더 강한 morphological operations)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # ultra_aggressive 모드: 추가 morphological operations
    if ultra_aggressive:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 추가: 큰 커널로 구조 강화
    kernel_large = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)

    # 작은 점 제거
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

    # aggressive_trim: 엣지 기반으로 추가로 여백 제거
    if aggressive_trim:
        trimmed_temp = crop[y1:y2, x1:x2]
        th, tw = trimmed_temp.shape[:2]

        gray_temp = cv2.cvtColor(trimmed_temp, cv2.COLOR_BGR2GRAY)
        
        # 엣지 감지
        edges = cv2.Canny(gray_temp, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        edge_ys, edge_xs = np.where(edges > 0)

        if len(edge_xs) > 10 and len(edge_ys) > 10:
            ex1 = max(0, int(edge_xs.min()))
            ex2 = min(tw - 1, int(edge_xs.max()))
            ey1 = max(0, int(edge_ys.min()))
            ey2 = min(th - 1, int(edge_ys.max()))

            # 엣지 주변 추가 여백 제거
            edge_bw = ex2 - ex1
            edge_bh = ey2 - ey1

            # ultra_aggressive 모드: 더 적극적인 margin 제거
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

    return trimmed, (x1, y1)


def recognize_digits_from_crop(
    digit_model,
    number_crop,
    device="0",
    conf=0.15,
    iou=0.45,
    imgsz=640,
    max_det=8,
    min_digit_conf=0.0,
    require_4digits=True,
):
    """
    번호 영역 crop 안에서 숫자 하나하나를 YOLO detector로 탐지한 뒤
    x좌표 기준으로 정렬해서 번호 문자열 생성
    """

    if number_crop is None or number_crop.size == 0:
        return "", []

    results = digit_model.predict(
        number_crop,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
        device=device,
    )

    if len(results) == 0 or results[0].boxes is None:
        return "", []

    digits = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        if score < min_digit_conf:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        xc = (x1 + x2) / 2.0

        digits.append({
            "digit": str(cls_id),
            "class_id": cls_id,
            "conf": score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "xc": xc,
        })

    if len(digits) == 0:
        return "", []

    # x좌표 기준 정렬
    digits = sorted(digits, key=lambda d: d["xc"])

    # 4개보다 많이 잡히면 confidence 높은 4개만 선택 후 다시 x좌표 정렬
    if len(digits) > 4:
        digits = sorted(digits, key=lambda d: d["conf"], reverse=True)[:4]
        digits = sorted(digits, key=lambda d: d["xc"])

    # 4자리만 인정할 경우
    if require_4digits and len(digits) != 4:
        return "", digits

    pred_number = "".join(d["digit"] for d in digits)

    return pred_number, digits


def find_number_candidates_by_connected_components(
    person_crop,
    dark_thresh=200,
    min_area=1200,
    min_aspect=2.0,
    max_aspect=10.0,
    pad_x=0.1,
    pad_y=0.1,
):
    """
    person crop에서 어두운 등번호 후보 영역을 추출.
    OpenCV threshold + connected component / contour 기반 필터링을 사용.
    """

    if person_crop is None or person_crop.size == 0:
        return []

    h, w = person_crop.shape[:2]
    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # connected component 기반 후보 추출
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

    candidates = []
    min_area = max(min_area, int(w * h * 0.0005))
    min_width = max(20, int(w * 0.15))
    min_height = max(12, int(h * 0.08))

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < min_area:
            continue

        if bw < min_width or bh < min_height:
            continue

        aspect = bw / max(1.0, bh)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        crop_box = expand_box(x, y, x + bw, y + bh, w, h, pad_x=pad_x, pad_y=pad_y)
        if crop_box is None:
            continue

        x1, y1, x2, y2 = crop_box
        candidates.append((x1, y1, x2, y2, area))

    if len(candidates) == 0:
        # contour 기반 fallback: 작은 영역들을 합쳐볼 수 있도록 contours도 확인
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < min_area:
                continue

            aspect = bw / max(1.0, bh)
            if aspect < min_aspect or aspect > max_aspect:
                continue

            crop_box = expand_box(x, y, x + bw, y + bh, w, h, pad_x=pad_x, pad_y=pad_y)
            if crop_box is None:
                continue

            x1, y1, x2, y2 = crop_box
            candidates.append((x1, y1, x2, y2, area))

    candidates = sorted(candidates, key=lambda item: item[4], reverse=True)
    return candidates


def recognize_number_from_candidates(
    digit_model,
    person_crop,
    candidates,
    device="0",
    conf=0.15,
    iou=0.45,
    imgsz=640,
    max_det=8,
    min_digit_conf=0.6,
    require_4digits=True,
):
    """
    후보 영역들을 순회하며 digit YOLO로 4자리 번호를 확정.
    """

    best_result = None

    for x1, y1, x2, y2, _ in candidates:
        number_crop = person_crop[y1:y2, x1:x2]
        if number_crop.size == 0:
            continue

        raw_number, digits = recognize_digits_from_crop(
            digit_model=digit_model,
            number_crop=number_crop,
            device=device,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            min_digit_conf=min_digit_conf,
            require_4digits=require_4digits,
        )

        if not raw_number:
            continue

        avg_conf = sum(d["conf"] for d in digits) / len(digits)

        if best_result is None or avg_conf > best_result["score"]:
            best_result = {
                "number": raw_number,
                "digits": digits,
                "bbox": (x1, y1, x2, y2),
                "score": avg_conf,
            }

    if best_result is None:
        return "", [], None

    return best_result["number"], best_result["digits"], best_result["bbox"]


def get_stable_number(history_deque, new_number, min_hits=5, lock_ratio=0.55):
    """
    temporal voting
    빈 문자열은 history에 넣지 않음
    """

    if new_number:
        history_deque.append(new_number)

    hist = list(history_deque)

    if len(hist) == 0:
        return ""

    if len(hist) < min_hits:
        return new_number if new_number else hist[-1]

    counter = Counter(hist)
    top_number, top_count = counter.most_common(1)[0]
    ratio = top_count / len(hist)

    if ratio >= lock_ratio:
        return top_number

    return new_number if new_number else top_number


def draw_digit_boxes(vis_frame, digits, offset_x, offset_y):
    """
    digit bbox를 원본 frame 좌표에 맞춰서 그림
    """

    for d in digits:
        dx1 = int(offset_x + d["x1"])
        dy1 = int(offset_y + d["y1"])
        dx2 = int(offset_x + d["x2"])
        dy2 = int(offset_y + d["y2"])

        label = f"{d['digit']} {d['conf']:.2f}"

        cv2.rectangle(
            vis_frame,
            (dx1, dy1),
            (dx2, dy2),
            (0, 220, 0),
            2
        )

        cv2.putText(
            vis_frame,
            label,
            (dx1, max(20, dy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA
        )


def make_output_paths(video_path, out_dir, run_name=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{video_stem}_digit_yolo_{timestamp}"

    out_video_path = out_dir / f"{run_name}.mp4"
    out_csv_path = out_dir / f"{run_name}.csv"

    return out_video_path, out_csv_path, run_name


# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, required=True)

    parser.add_argument(
        "--pose_model",
        type=str,
        default="yolo26m-pose.pt"
    )

    parser.add_argument(
        "--number_detector",
        type=str,
        required=True,
        help="기존 4자리 숫자 영역 detector weight"
    )

    parser.add_argument(
        "--digit_detector",
        type=str,
        required=True,
        help="학습한 digit YOLO detector best.pt"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="결과 영상/CSV 저장 폴더"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="결과 파일명 prefix. 없으면 입력 비디오명+시간으로 자동 생성"
    )

    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--person_conf", type=float, default=0.5)
    parser.add_argument("--number_conf", type=float, default=0.5)
    parser.add_argument("--digit_conf", type=float, default=0.15)
    parser.add_argument("--digit_min_conf", type=float, default=0.6,
                        help="각 digit 결과를 이 confidence 이상으로 필터링")

    parser.add_argument("--person_iou", type=float, default=0.45)
    parser.add_argument("--number_iou", type=float, default=0.45)
    parser.add_argument("--digit_iou", type=float, default=0.45)

    parser.add_argument("--frame_step", type=int, default=5)

    # number area detector crop padding
    # 현재 crop에 여백이 많다고 했으므로 기본값 0으로 둠
    parser.add_argument("--pad_x", type=float, default=0.0)
    parser.add_argument("--pad_y", type=float, default=0.0)

    # trim options
    parser.add_argument("--use_trim", action="store_true")
    parser.add_argument("--trim_pad_x", type=float, default=0.08)
    parser.add_argument("--trim_pad_y", type=float, default=0.12)
    parser.add_argument("--dark_thresh", type=int, default=200)
    parser.add_argument("--min_dark_area", type=int, default=20)
    
    # aggressive trim (edge-based refinement)
    parser.add_argument("--aggressive_trim", action="store_true", default=True,
                        help="엣지 기반 aggressive trim 활성화 (기본값: True)")
    parser.add_argument("--edge_margin_x", type=float, default=0.02,
                        help="엣지 주변 x 방향 여백 비율")
    parser.add_argument("--edge_margin_y", type=float, default=0.05,
                        help="엣지 주변 y 방향 여백 비율")
    parser.add_argument("--ultra_aggressive", action="store_true",
                        help="더 강력한 aggressive trim (낮은 threshold + 적극적 morphology)")

    # digit detector
    parser.add_argument("--digit_imgsz", type=int, default=640)
    parser.add_argument("--require_4digits", action="store_true", default=True,
                        help="4자리 숫자 결과만 인정합니다 (기본값: True)")
    parser.add_argument("--allow_less_digits", dest="require_4digits", action="store_false",
                        help="4자리 미만의 숫자도 결과로 허용합니다")

    # temporal voting
    parser.add_argument("--number_history", type=int, default=20)
    parser.add_argument("--number_min_hits", type=int, default=5)
    parser.add_argument("--number_lock_ratio", type=float, default=0.55)

    # debug crop 저장
    parser.add_argument("--save_crops", action="store_true")
    parser.add_argument("--crop_dir", type=str, default=None)
    parser.add_argument("--save_raw_trim_pair", action="store_true")

    args = parser.parse_args()

    # =====================================================
    # output paths
    # =====================================================

    out_video_path, out_csv_path, run_name = make_output_paths(
        video_path=args.video,
        out_dir=args.out_dir,
        run_name=args.run_name
    )

    if args.crop_dir is None:
        crop_dir = Path(args.out_dir) / f"{run_name}_number_crops"
    else:
        crop_dir = Path(args.crop_dir)

    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OUTPUT VIDEO] {out_video_path}")
    print(f"[OUTPUT CSV]   {out_csv_path}")

    if args.save_crops:
        print(f"[CROP DIR]     {crop_dir}")

    # =====================================================
    # model load
    # =====================================================

    print("loading models...")

    pose_model = YOLO(args.pose_model)
    number_detector = YOLO(args.number_detector)
    digit_detector = YOLO(args.digit_detector)

    print("models loaded")

    # =====================================================
    # video
    # =====================================================

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"video open failed: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(out_video_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"video writer open failed: {out_video_path}")

    # =====================================================
    # csv
    # =====================================================

    csv_file = open(
        str(out_csv_path),
        "w",
        newline="",
        encoding="utf-8-sig"
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "person_idx",
        "number_area_conf",
        "raw_number",
        "stable_number",
        "digit_count",
        "number_area_bbox_abs",
        "trim_offset",
        "crop_type",
        "digits_detail"
    ])

    # =====================================================
    # temporal history
    # 현재는 frame 내 person_idx 기준.
    # 추후 DeepSORT/ByteTrack track_id를 붙이면 더 안정적임.
    # =====================================================

    number_histories = defaultdict(
        lambda: deque(maxlen=args.number_history)
    )

    # =====================================================
    # inference
    # =====================================================

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        vis_frame = frame.copy()

        # frame_step 아닌 프레임은 원본 그대로 저장
        if frame_idx % args.frame_step != 0:
            writer.write(vis_frame)
            frame_idx += 1
            continue

        # =================================================
        # person detect
        # =================================================

        pose_results = pose_model.predict(
            frame,
            conf=args.person_conf,
            iou=args.person_iou,
            verbose=False,
            device=args.device
        )

        person_idx = 0

        for result in pose_results:
            if result.boxes is None:
                continue

            person_boxes = result.boxes.xyxy.cpu().numpy()

            for pbox in person_boxes:
                px1, py1, px2, py2 = pbox
                pbox_clamped = clamp_box(px1, py1, px2, py2, width, height)

                if pbox_clamped is None:
                    continue

                px1, py1, px2, py2 = pbox_clamped

                person_crop = frame[py1:py2, px1:px2]

                if person_crop.size == 0:
                    continue

                # person bbox
                cv2.rectangle(
                    vis_frame,
                    (px1, py1),
                    (px2, py2),
                    (255, 180, 0),
                    2
                )

                # =================================================
                # number area detector
                # =================================================

                number_results = number_detector.predict(
                    person_crop,
                    conf=args.number_conf,
                    iou=args.number_iou,
                    verbose=False,
                    device=args.device
                )

                raw_number = ""
                stable_number = ""
                selected_area_conf = 0.0
                selected_area_abs = ""
                selected_digits = []
                trim_offset = (0, 0)
                crop_type = "none"

                if len(number_results) > 0 and number_results[0].boxes is not None:
                    nboxes = number_results[0].boxes.xyxy.cpu().numpy()
                    nconfs = number_results[0].boxes.conf.cpu().numpy()

                    if len(nboxes) > 0:
                        # confidence 가장 높은 번호 영역 하나 선택
                        best_idx = int(np.argmax(nconfs))
                        selected_area_conf = float(nconfs[best_idx])

                        bx1, by1, bx2, by2 = nboxes[best_idx]

                        # 번호 영역 crop
                        crop_box = expand_box(
                            bx1,
                            by1,
                            bx2,
                            by2,
                            person_crop.shape[1],
                            person_crop.shape[0],
                            pad_x=args.pad_x,
                            pad_y=args.pad_y
                        )

                        if crop_box is not None:
                            bx1, by1, bx2, by2 = crop_box

                            number_crop_raw = person_crop[by1:by2, bx1:bx2]

                            if number_crop_raw.size != 0:

                                # =================================
                                # trim 적용
                                # =================================

                                if args.use_trim:
                                    number_crop, trim_offset = trim_number_crop_by_dark_region(
                                        number_crop_raw,
                                        pad_x=args.trim_pad_x,
                                        pad_y=args.trim_pad_y,
                                        dark_thresh=args.dark_thresh,
                                        min_area=args.min_dark_area,
                                        aggressive_trim=args.aggressive_trim,
                                        edge_margin_x=args.edge_margin_x,
                                        edge_margin_y=args.edge_margin_y,
                                        ultra_aggressive=args.ultra_aggressive,
                                    )
                                    crop_type = "trim"
                                else:
                                    number_crop = number_crop_raw
                                    trim_offset = (0, 0)
                                    crop_type = "raw"

                                # =================================
                                # digit detector
                                # =================================

                                raw_number, selected_digits = recognize_digits_from_crop(
                                    digit_model=digit_detector,
                                    number_crop=number_crop,
                                    device=args.device,
                                    conf=args.digit_conf,
                                    iou=args.digit_iou,
                                    imgsz=args.digit_imgsz,
                                    max_det=8,
                                    min_digit_conf=args.digit_min_conf,
                                    require_4digits=args.require_4digits,
                                )

                                # voting key
                                history_key = person_idx

                                stable_number = get_stable_number(
                                    number_histories[history_key],
                                    raw_number,
                                    min_hits=args.number_min_hits,
                                    lock_ratio=args.number_lock_ratio
                                )

                                abs_x1 = px1 + bx1
                                abs_y1 = py1 + by1
                                abs_x2 = px1 + bx2
                                abs_y2 = py1 + by2

                                selected_area_abs = [abs_x1, abs_y1, abs_x2, abs_y2]

                                trim_x, trim_y = trim_offset

                                digit_offset_x = abs_x1 + trim_x
                                digit_offset_y = abs_y1 + trim_y

                                # 번호 영역 bbox: raw detector crop 기준
                                cv2.rectangle(
                                    vis_frame,
                                    (abs_x1, abs_y1),
                                    (abs_x2, abs_y2),
                                    (0, 255, 255),
                                    3
                                )

                                # trim된 영역 bbox 추가 표시
                                if args.use_trim:
                                    th, tw = number_crop.shape[:2]
                                    tx1 = digit_offset_x
                                    ty1 = digit_offset_y
                                    tx2 = digit_offset_x + tw
                                    ty2 = digit_offset_y + th

                                    cv2.rectangle(
                                        vis_frame,
                                        (tx1, ty1),
                                        (tx2, ty2),
                                        (255, 0, 255),
                                        2
                                    )

                                label_text = stable_number if stable_number else raw_number
                                if not label_text:
                                    label_text = "none"

                                cv2.putText(
                                    vis_frame,
                                    label_text,
                                    (abs_x1, max(30, abs_y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    (0, 255, 255),
                                    3,
                                    cv2.LINE_AA
                                )

                                # digit bbox: trim offset 반영
                                draw_digit_boxes(
                                    vis_frame,
                                    selected_digits,
                                    offset_x=digit_offset_x,
                                    offset_y=digit_offset_y
                                )

                                # =================================
                                # crop 저장
                                # =================================

                                if args.save_crops:
                                    base_name = (
                                        f"frame_{frame_idx:06d}"
                                        f"_person_{person_idx:03d}"
                                        f"_raw_{raw_number if raw_number else 'none'}"
                                        f"_stable_{stable_number if stable_number else 'none'}"
                                    )

                                    if args.save_raw_trim_pair and args.use_trim:
                                        cv2.imwrite(
                                            str(crop_dir / f"{base_name}_raw.jpg"),
                                            number_crop_raw
                                        )
                                        cv2.imwrite(
                                            str(crop_dir / f"{base_name}_trim.jpg"),
                                            number_crop
                                        )
                                    else:
                                        cv2.imwrite(
                                            str(crop_dir / f"{base_name}_{crop_type}.jpg"),
                                            number_crop
                                        )

                csv_writer.writerow([
                    frame_idx,
                    person_idx,
                    f"{selected_area_conf:.4f}",
                    raw_number,
                    stable_number,
                    len(selected_digits),
                    selected_area_abs,
                    trim_offset,
                    crop_type,
                    [
                        {
                            "digit": d["digit"],
                            "conf": round(d["conf"], 4),
                            "bbox_crop": [
                                round(d["x1"], 1),
                                round(d["y1"], 1),
                                round(d["x2"], 1),
                                round(d["y2"], 1)
                            ]
                        }
                        for d in selected_digits
                    ]
                ])

                person_idx += 1

        writer.write(vis_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()

    print("DONE")
    print(f"video saved: {out_video_path}")
    print(f"csv saved: {out_csv_path}")

    if args.save_crops:
        print(f"crops saved: {crop_dir}")


if __name__ == "__main__":
    main()