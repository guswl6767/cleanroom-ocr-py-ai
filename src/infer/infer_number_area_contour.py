# import os
# import re
# import csv
# import cv2
# import argparse
# import numpy as np

# from pathlib import Path
# from datetime import datetime
# from collections import defaultdict, deque, Counter

# from ultralytics import YOLO


# # =========================================================
# # util
# # =========================================================

# def clamp_box(x1, y1, x2, y2, w, h):
#     x1 = max(0, min(int(x1), w - 1))
#     y1 = max(0, min(int(y1), h - 1))
#     x2 = max(0, min(int(x2), w - 1))
#     y2 = max(0, min(int(y2), h - 1))

#     if x2 <= x1 or y2 <= y1:
#         return None

#     return x1, y1, x2, y2


# def expand_box(x1, y1, x2, y2, w, h, pad_x=0.0, pad_y=0.0):
#     bw = x2 - x1
#     bh = y2 - y1

#     px = bw * pad_x
#     py = bh * pad_y

#     nx1 = x1 - px
#     ny1 = y1 - py
#     nx2 = x2 + px
#     ny2 = y2 + py

#     return clamp_box(nx1, ny1, nx2, ny2, w, h)


# def box_iou(box_a, box_b):
#     ax1, ay1, ax2, ay2 = box_a
#     bx1, by1, bx2, by2 = box_b

#     inter_x1 = max(ax1, bx1)
#     inter_y1 = max(ay1, by1)
#     inter_x2 = min(ax2, bx2)
#     inter_y2 = min(ay2, by2)

#     inter_w = max(0, inter_x2 - inter_x1)
#     inter_h = max(0, inter_y2 - inter_y1)
#     inter_area = inter_w * inter_h

#     area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
#     area_b = max(1, (bx2 - bx1) * (by2 - by1))

#     return inter_area / max(1, area_a + area_b - inter_area)


# def is_valid_4digit_number(text):
#     if text is None:
#         return False
#     return re.fullmatch(r"\d{4}", str(text)) is not None


# # =========================================================
# # bbox tracker
# # =========================================================

# class SimpleBBoxTracker:
#     """
#     번호 영역 bbox 기준 간단 tracker.
#     DeepSORT/ByteTrack 전까지 임시로 track_id를 유지하기 위한 용도.
#     """

#     def __init__(self, iou_th=0.3, max_age=15):
#         self.iou_th = iou_th
#         self.max_age = max_age
#         self.next_id = 0
#         self.tracks = {}

#     def update(self, bbox, frame_idx):
#         best_id = None
#         best_iou = 0.0

#         for track_id, track in self.tracks.items():
#             iou = box_iou(track["bbox"], bbox)

#             if iou > best_iou:
#                 best_iou = iou
#                 best_id = track_id

#         if best_id is not None and best_iou >= self.iou_th:
#             self.tracks[best_id]["bbox"] = bbox
#             self.tracks[best_id]["last_seen"] = frame_idx
#             return best_id

#         track_id = self.next_id
#         self.next_id += 1

#         self.tracks[track_id] = {
#             "bbox": bbox,
#             "last_seen": frame_idx
#         }

#         return track_id

#     def cleanup(self, frame_idx):
#         remove_ids = []

#         for track_id, track in self.tracks.items():
#             if frame_idx - track["last_seen"] > self.max_age:
#                 remove_ids.append(track_id)

#         for track_id in remove_ids:
#             del self.tracks[track_id]


# # =========================================================
# # crop preprocessing
# # =========================================================

# def trim_number_crop_by_dark_region(
#     crop,
#     pad_x=0.08,
#     pad_y=0.12,
#     dark_thresh=200,
#     min_area=20
# ):
#     """
#     밝은 방호복 배경에서 어두운 숫자 영역만 찾아 crop을 타이트하게 줄임.
#     실패하면 원본 crop 반환.

#     return:
#         trimmed_crop, (offset_x, offset_y)
#     """

#     if crop is None or crop.size == 0:
#         return crop, (0, 0)

#     h, w = crop.shape[:2]

#     if h <= 5 or w <= 5:
#         return crop, (0, 0)

#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)

#     mask = (gray < dark_thresh).astype(np.uint8) * 255

#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

#     clean_mask = np.zeros_like(mask)

#     for i in range(1, num_labels):
#         area = stats[i, cv2.CC_STAT_AREA]

#         if area >= min_area:
#             clean_mask[labels == i] = 255

#     ys, xs = np.where(clean_mask > 0)

#     if len(xs) < 5 or len(ys) < 5:
#         return crop, (0, 0)

#     x1, x2 = int(xs.min()), int(xs.max())
#     y1, y2 = int(ys.min()), int(ys.max())

#     bw = x2 - x1
#     bh = y2 - y1

#     if bw <= 5 or bh <= 5:
#         return crop, (0, 0)

#     px = int(bw * pad_x)
#     py = int(bh * pad_y)

#     x1 = max(0, x1 - px)
#     y1 = max(0, y1 - py)
#     x2 = min(w - 1, x2 + px)
#     y2 = min(h - 1, y2 + py)

#     trimmed = crop[y1:y2, x1:x2]

#     if trimmed.size == 0:
#         return crop, (0, 0)

#     return trimmed, (x1, y1)


# # =========================================================
# # digit detector
# # =========================================================

# def recognize_digits_from_crop(
#     digit_model,
#     number_crop,
#     device="0",
#     conf=0.15,
#     iou=0.45,
#     imgsz=640,
#     max_det=8
# ):
#     """
#     번호 영역 crop 안에서 숫자 하나하나를 YOLO detector로 탐지한 뒤
#     x좌표 기준으로 정렬해서 번호 문자열 생성.

#     정확히 4개 숫자가 잡힌 경우에만 raw_number 반환.
#     아니면 raw_number는 "" 반환.
#     """

#     if number_crop is None or number_crop.size == 0:
#         return "", []

#     results = digit_model.predict(
#         number_crop,
#         imgsz=imgsz,
#         conf=conf,
#         iou=iou,
#         max_det=max_det,
#         verbose=False,
#         device=device,
#     )

#     if len(results) == 0 or results[0].boxes is None:
#         return "", []

#     digits = []

#     for box in results[0].boxes:
#         cls_id = int(box.cls[0].item())
#         score = float(box.conf[0].item())
#         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

#         xc = (x1 + x2) / 2.0

#         digits.append({
#             "digit": str(cls_id),
#             "class_id": cls_id,
#             "conf": score,
#             "x1": x1,
#             "y1": y1,
#             "x2": x2,
#             "y2": y2,
#             "xc": xc,
#         })

#     if len(digits) == 0:
#         return "", []

#     # x좌표 기준 정렬
#     digits = sorted(digits, key=lambda d: d["xc"])

#     # 4개보다 많이 잡히면 confidence 높은 4개만 선택 후 다시 x좌표 정렬
#     if len(digits) > 4:
#         digits = sorted(digits, key=lambda d: d["conf"], reverse=True)[:4]
#         digits = sorted(digits, key=lambda d: d["xc"])

#     # 정확히 4자리일 때만 번호 생성
#     if len(digits) != 4:
#         return "", digits

#     pred_number = "".join(d["digit"] for d in digits)

#     if not is_valid_4digit_number(pred_number):
#         return "", digits

#     return pred_number, digits


# # =========================================================
# # confirmation logic
# # =========================================================

# def update_confirmed_number(
#     history_deque,
#     confirmed_state,
#     new_number,
#     min_confirm_frames=5,
#     lock_ratio=0.6
# ):
#     """
#     정확히 4자리 숫자만 history에 넣고,
#     같은 번호가 min_confirm_frames 이상 나오고 비율 조건을 만족하면 확정.
#     """

#     if confirmed_state["confirmed"]:
#         return {
#             "candidate": confirmed_state["number"],
#             "confirmed": confirmed_state["number"],
#             "is_confirmed": True,
#             "hit_count": confirmed_state["hit_count"],
#             "hit_ratio": confirmed_state["hit_ratio"],
#         }

#     if is_valid_4digit_number(new_number):
#         history_deque.append(new_number)

#     hist = list(history_deque)

#     if len(hist) == 0:
#         return {
#             "candidate": "",
#             "confirmed": "",
#             "is_confirmed": False,
#             "hit_count": 0,
#             "hit_ratio": 0.0,
#         }

#     counter = Counter(hist)
#     top_number, top_count = counter.most_common(1)[0]
#     ratio = top_count / len(hist)

#     if top_count >= min_confirm_frames and ratio >= lock_ratio:
#         confirmed_state["confirmed"] = True
#         confirmed_state["number"] = top_number
#         confirmed_state["hit_count"] = top_count
#         confirmed_state["hit_ratio"] = ratio

#         return {
#             "candidate": top_number,
#             "confirmed": top_number,
#             "is_confirmed": True,
#             "hit_count": top_count,
#             "hit_ratio": ratio,
#         }

#     return {
#         "candidate": top_number,
#         "confirmed": "",
#         "is_confirmed": False,
#         "hit_count": top_count,
#         "hit_ratio": ratio,
#     }


# # =========================================================
# # visualization
# # =========================================================

# def draw_digit_boxes(vis_frame, digits, offset_x, offset_y):
#     for d in digits:
#         dx1 = int(offset_x + d["x1"])
#         dy1 = int(offset_y + d["y1"])
#         dx2 = int(offset_x + d["x2"])
#         dy2 = int(offset_y + d["y2"])

#         label = f"{d['digit']} {d['conf']:.2f}"

#         cv2.rectangle(
#             vis_frame,
#             (dx1, dy1),
#             (dx2, dy2),
#             (0, 220, 0),
#             2
#         )

#         cv2.putText(
#             vis_frame,
#             label,
#             (dx1, max(20, dy1 - 5)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.55,
#             (0, 220, 0),
#             2,
#             cv2.LINE_AA
#         )


# def draw_status_text(
#     vis_frame,
#     x,
#     y,
#     raw_number,
#     confirm_info,
#     confirm_frames,
#     draw_pending=True
# ):
#     is_confirmed = confirm_info["is_confirmed"]
#     confirmed_number = confirm_info["confirmed"]
#     candidate = confirm_info["candidate"]
#     hit_count = confirm_info["hit_count"]
#     hit_ratio = confirm_info["hit_ratio"]

#     if is_confirmed:
#         text = f"CONFIRMED: {confirmed_number}"
#         color = (0, 255, 0)
#         thickness = 3
#     else:
#         if draw_pending and candidate:
#             text = f"PENDING: {candidate} {hit_count}/{confirm_frames} ({hit_ratio:.2f})"
#         elif raw_number:
#             text = f"RAW: {raw_number}"
#         else:
#             text = "pending"

#         color = (0, 200, 255)
#         thickness = 2

#     cv2.putText(
#         vis_frame,
#         text,
#         (x, max(30, y - 10)),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.9,
#         color,
#         thickness,
#         cv2.LINE_AA
#     )


# def make_output_paths(video_path, out_dir, run_name=None):
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     video_stem = Path(video_path).stem

#     if run_name is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         run_name = f"{video_stem}_digit_confirm_{timestamp}"

#     out_video_path = out_dir / f"{run_name}.mp4"
#     out_csv_path = out_dir / f"{run_name}_confirmed.csv"
#     debug_csv_path = out_dir / f"{run_name}_debug.csv"

#     return out_video_path, out_csv_path, debug_csv_path, run_name


# # =========================================================
# # main
# # =========================================================

# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--video", type=str, required=True)

#     parser.add_argument(
#         "--pose_model",
#         type=str,
#         default="yolo26m-pose.pt"
#     )

#     parser.add_argument(
#         "--number_detector",
#         type=str,
#         required=True,
#         help="기존 4자리 숫자 영역 detector weight"
#     )

#     parser.add_argument(
#         "--digit_detector",
#         type=str,
#         required=True,
#         help="학습한 digit YOLO detector best.pt"
#     )

#     parser.add_argument(
#         "--out_dir",
#         type=str,
#         required=True
#     )

#     parser.add_argument(
#         "--run_name",
#         type=str,
#         default=None
#     )

#     parser.add_argument("--device", type=str, default="0")

#     parser.add_argument("--person_conf", type=float, default=0.5)
#     parser.add_argument("--number_conf", type=float, default=0.5)
#     parser.add_argument("--digit_conf", type=float, default=0.15)

#     parser.add_argument("--person_iou", type=float, default=0.45)
#     parser.add_argument("--number_iou", type=float, default=0.45)
#     parser.add_argument("--digit_iou", type=float, default=0.45)

#     parser.add_argument("--frame_step", type=int, default=5)

#     # number area crop padding
#     parser.add_argument("--pad_x", type=float, default=0.0)
#     parser.add_argument("--pad_y", type=float, default=0.0)

#     # trim
#     parser.add_argument("--use_trim", action="store_true")
#     parser.add_argument("--trim_pad_x", type=float, default=0.08)
#     parser.add_argument("--trim_pad_y", type=float, default=0.12)
#     parser.add_argument("--dark_thresh", type=int, default=200)
#     parser.add_argument("--min_dark_area", type=int, default=20)

#     # digit detector
#     parser.add_argument("--digit_imgsz", type=int, default=640)

#     # confirmation
#     parser.add_argument("--number_history", type=int, default=20)
#     parser.add_argument("--confirm_frames", type=int, default=5)
#     parser.add_argument("--confirm_ratio", type=float, default=0.6)
#     parser.add_argument("--draw_pending", action="store_true")

#     # simple bbox tracker
#     parser.add_argument("--track_iou", type=float, default=0.3)
#     parser.add_argument("--track_max_age", type=int, default=15)

#     # output/debug
#     parser.add_argument("--save_crops", action="store_true")
#     parser.add_argument("--crop_dir", type=str, default=None)
#     parser.add_argument("--save_raw_trim_pair", action="store_true")
#     parser.add_argument("--save_debug_csv", action="store_true")

#     args = parser.parse_args()

#     # =====================================================
#     # output paths
#     # =====================================================

#     out_video_path, confirmed_csv_path, debug_csv_path, run_name = make_output_paths(
#         video_path=args.video,
#         out_dir=args.out_dir,
#         run_name=args.run_name
#     )

#     if args.crop_dir is None:
#         crop_dir = Path(args.out_dir) / f"{run_name}_number_crops"
#     else:
#         crop_dir = Path(args.crop_dir)

#     if args.save_crops:
#         crop_dir.mkdir(parents=True, exist_ok=True)

#     print(f"[OUTPUT VIDEO]      {out_video_path}")
#     print(f"[CONFIRMED CSV]     {confirmed_csv_path}")

#     if args.save_debug_csv:
#         print(f"[DEBUG CSV]         {debug_csv_path}")

#     if args.save_crops:
#         print(f"[CROP DIR]          {crop_dir}")

#     # =====================================================
#     # model load
#     # =====================================================

#     print("loading models...")

#     pose_model = YOLO(args.pose_model)
#     number_detector = YOLO(args.number_detector)
#     digit_detector = YOLO(args.digit_detector)

#     print("models loaded")

#     # =====================================================
#     # video
#     # =====================================================

#     cap = cv2.VideoCapture(args.video)

#     if not cap.isOpened():
#         raise RuntimeError(f"video open failed: {args.video}")

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     if fps <= 0:
#         fps = 30

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#     writer = cv2.VideoWriter(
#         str(out_video_path),
#         fourcc,
#         fps,
#         (width, height)
#     )

#     if not writer.isOpened():
#         raise RuntimeError(f"video writer open failed: {out_video_path}")

#     # =====================================================
#     # csv
#     # =====================================================

#     confirmed_csv_file = open(
#         str(confirmed_csv_path),
#         "w",
#         newline="",
#         encoding="utf-8-sig"
#     )

#     confirmed_writer = csv.writer(confirmed_csv_file)

#     confirmed_writer.writerow([
#         "confirmed_frame",
#         "track_id",
#         "confirmed_number",
#         "hit_count",
#         "hit_ratio",
#         "number_area_conf",
#         "number_area_bbox_abs"
#     ])

#     debug_csv_file = None
#     debug_writer = None

#     if args.save_debug_csv:
#         debug_csv_file = open(
#             str(debug_csv_path),
#             "w",
#             newline="",
#             encoding="utf-8-sig"
#         )

#         debug_writer = csv.writer(debug_csv_file)

#         debug_writer.writerow([
#             "frame",
#             "track_id",
#             "person_idx",
#             "number_area_conf",
#             "raw_number",
#             "candidate_number",
#             "confirmed_number",
#             "is_confirmed",
#             "hit_count",
#             "hit_ratio",
#             "digit_count",
#             "number_area_bbox_abs",
#             "trim_offset",
#             "digits_detail"
#         ])

#     # =====================================================
#     # states
#     # =====================================================

#     tracker = SimpleBBoxTracker(
#         iou_th=args.track_iou,
#         max_age=args.track_max_age
#     )

#     number_histories = defaultdict(
#         lambda: deque(maxlen=args.number_history)
#     )

#     confirmed_states = defaultdict(
#         lambda: {
#             "confirmed": False,
#             "number": "",
#             "hit_count": 0,
#             "hit_ratio": 0.0,
#             "saved": False
#         }
#     )

#     # =====================================================
#     # inference
#     # =====================================================

#     frame_idx = 0

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         vis_frame = frame.copy()

#         if frame_idx % args.frame_step != 0:
#             writer.write(vis_frame)
#             frame_idx += 1
#             continue

#         tracker.cleanup(frame_idx)

#         pose_results = pose_model.predict(
#             frame,
#             conf=args.person_conf,
#             iou=args.person_iou,
#             verbose=False,
#             device=args.device
#         )

#         person_idx = 0

#         for result in pose_results:
#             if result.boxes is None:
#                 continue

#             person_boxes = result.boxes.xyxy.cpu().numpy()

#             for pbox in person_boxes:
#                 px1, py1, px2, py2 = pbox
#                 pbox_clamped = clamp_box(px1, py1, px2, py2, width, height)

#                 if pbox_clamped is None:
#                     continue

#                 px1, py1, px2, py2 = pbox_clamped

#                 person_crop = frame[py1:py2, px1:px2]

#                 if person_crop.size == 0:
#                     continue

#                 cv2.rectangle(
#                     vis_frame,
#                     (px1, py1),
#                     (px2, py2),
#                     (255, 180, 0),
#                     2
#                 )

#                 number_results = number_detector.predict(
#                     person_crop,
#                     conf=args.number_conf,
#                     iou=args.number_iou,
#                     verbose=False,
#                     device=args.device
#                 )

#                 if len(number_results) == 0 or number_results[0].boxes is None:
#                     person_idx += 1
#                     continue

#                 nboxes = number_results[0].boxes.xyxy.cpu().numpy()
#                 nconfs = number_results[0].boxes.conf.cpu().numpy()

#                 if len(nboxes) == 0:
#                     person_idx += 1
#                     continue

#                 best_idx = int(np.argmax(nconfs))
#                 number_area_conf = float(nconfs[best_idx])

#                 bx1, by1, bx2, by2 = nboxes[best_idx]

#                 crop_box = expand_box(
#                     bx1,
#                     by1,
#                     bx2,
#                     by2,
#                     person_crop.shape[1],
#                     person_crop.shape[0],
#                     pad_x=args.pad_x,
#                     pad_y=args.pad_y
#                 )

#                 if crop_box is None:
#                     person_idx += 1
#                     continue

#                 bx1, by1, bx2, by2 = crop_box

#                 number_crop_raw = person_crop[by1:by2, bx1:bx2]

#                 if number_crop_raw.size == 0:
#                     person_idx += 1
#                     continue

#                 abs_x1 = px1 + bx1
#                 abs_y1 = py1 + by1
#                 abs_x2 = px1 + bx2
#                 abs_y2 = py1 + by2

#                 number_area_bbox_abs = [abs_x1, abs_y1, abs_x2, abs_y2]

#                 # track id
#                 track_id = tracker.update(number_area_bbox_abs, frame_idx)

#                 # trim
#                 if args.use_trim:
#                     number_crop, trim_offset = trim_number_crop_by_dark_region(
#                         number_crop_raw,
#                         pad_x=args.trim_pad_x,
#                         pad_y=args.trim_pad_y,
#                         dark_thresh=args.dark_thresh,
#                         min_area=args.min_dark_area
#                     )
#                 else:
#                     number_crop = number_crop_raw
#                     trim_offset = (0, 0)

#                 trim_x, trim_y = trim_offset

#                 # digit detect
#                 raw_number, selected_digits = recognize_digits_from_crop(
#                     digit_model=digit_detector,
#                     number_crop=number_crop,
#                     device=args.device,
#                     conf=args.digit_conf,
#                     iou=args.digit_iou,
#                     imgsz=args.digit_imgsz,
#                     max_det=8
#                 )

#                 confirm_info = update_confirmed_number(
#                     history_deque=number_histories[track_id],
#                     confirmed_state=confirmed_states[track_id],
#                     new_number=raw_number,
#                     min_confirm_frames=args.confirm_frames,
#                     lock_ratio=args.confirm_ratio
#                 )

#                 # draw number area bbox
#                 cv2.rectangle(
#                     vis_frame,
#                     (abs_x1, abs_y1),
#                     (abs_x2, abs_y2),
#                     (0, 255, 255),
#                     3
#                 )

#                 # draw trim bbox
#                 digit_offset_x = abs_x1 + trim_x
#                 digit_offset_y = abs_y1 + trim_y

#                 if args.use_trim:
#                     th, tw = number_crop.shape[:2]
#                     cv2.rectangle(
#                         vis_frame,
#                         (digit_offset_x, digit_offset_y),
#                         (digit_offset_x + tw, digit_offset_y + th),
#                         (255, 0, 255),
#                         2
#                     )

#                 draw_digit_boxes(
#                     vis_frame,
#                     selected_digits,
#                     offset_x=digit_offset_x,
#                     offset_y=digit_offset_y
#                 )

#                 draw_status_text(
#                     vis_frame=vis_frame,
#                     x=abs_x1,
#                     y=abs_y1,
#                     raw_number=raw_number,
#                     confirm_info=confirm_info,
#                     confirm_frames=args.confirm_frames,
#                     draw_pending=args.draw_pending
#                 )

#                 # save confirmed csv only once
#                 if confirm_info["is_confirmed"] and not confirmed_states[track_id]["saved"]:
#                     confirmed_writer.writerow([
#                         frame_idx,
#                         track_id,
#                         confirm_info["confirmed"],
#                         confirm_info["hit_count"],
#                         f"{confirm_info['hit_ratio']:.4f}",
#                         f"{number_area_conf:.4f}",
#                         number_area_bbox_abs
#                     ])

#                     confirmed_states[track_id]["saved"] = True

#                 # debug csv
#                 if args.save_debug_csv:
#                     debug_writer.writerow([
#                         frame_idx,
#                         track_id,
#                         person_idx,
#                         f"{number_area_conf:.4f}",
#                         raw_number,
#                         confirm_info["candidate"],
#                         confirm_info["confirmed"],
#                         confirm_info["is_confirmed"],
#                         confirm_info["hit_count"],
#                         f"{confirm_info['hit_ratio']:.4f}",
#                         len(selected_digits),
#                         number_area_bbox_abs,
#                         trim_offset,
#                         [
#                             {
#                                 "digit": d["digit"],
#                                 "conf": round(d["conf"], 4),
#                                 "bbox_crop": [
#                                     round(d["x1"], 1),
#                                     round(d["y1"], 1),
#                                     round(d["x2"], 1),
#                                     round(d["y2"], 1)
#                                 ]
#                             }
#                             for d in selected_digits
#                         ]
#                     ])

#                 # save crops
#                 if args.save_crops:
#                     base_name = (
#                         f"frame_{frame_idx:06d}"
#                         f"_track_{track_id:03d}"
#                         f"_raw_{raw_number if raw_number else 'none'}"
#                         f"_cand_{confirm_info['candidate'] if confirm_info['candidate'] else 'none'}"
#                         f"_conf_{confirm_info['confirmed'] if confirm_info['confirmed'] else 'none'}"
#                     )

#                     if args.save_raw_trim_pair and args.use_trim:
#                         cv2.imwrite(
#                             str(crop_dir / f"{base_name}_raw.jpg"),
#                             number_crop_raw
#                         )
#                         cv2.imwrite(
#                             str(crop_dir / f"{base_name}_trim.jpg"),
#                             number_crop
#                         )
#                     else:
#                         cv2.imwrite(
#                             str(crop_dir / f"{base_name}.jpg"),
#                             number_crop
#                         )

#                 person_idx += 1

#         writer.write(vis_frame)
#         frame_idx += 1

#     cap.release()
#     writer.release()
#     confirmed_csv_file.close()

#     if debug_csv_file is not None:
#         debug_csv_file.close()

#     print("DONE")
#     print(f"video saved: {out_video_path}")
#     print(f"confirmed csv saved: {confirmed_csv_path}")

#     if args.save_debug_csv:
#         print(f"debug csv saved: {debug_csv_path}")

#     if args.save_crops:
#         print(f"crops saved: {crop_dir}")


# if __name__ == "__main__":
#     main()

import os
import re
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


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter_area / max(1, area_a + area_b - inter_area)


def is_valid_4digit_number(text):
    if text is None:
        return False
    return re.fullmatch(r"\d{4}", str(text)) is not None


# =========================================================
# bbox tracker
# =========================================================

class SimpleBBoxTracker:
    """
    번호 영역 bbox 기준 간단 tracker.
    DeepSORT/ByteTrack 전까지 임시로 track_id를 유지하기 위한 용도.
    """

    def __init__(self, iou_th=0.3, max_age=15):
        self.iou_th = iou_th
        self.max_age = max_age
        self.next_id = 0
        self.tracks = {}

    def update(self, bbox, frame_idx):
        best_id = None
        best_iou = 0.0

        for track_id, track in self.tracks.items():
            iou = box_iou(track["bbox"], bbox)

            if iou > best_iou:
                best_iou = iou
                best_id = track_id

        if best_id is not None and best_iou >= self.iou_th:
            self.tracks[best_id]["bbox"] = bbox
            self.tracks[best_id]["last_seen"] = frame_idx
            return best_id

        track_id = self.next_id
        self.next_id += 1

        self.tracks[track_id] = {
            "bbox": bbox,
            "last_seen": frame_idx
        }

        return track_id

    def cleanup(self, frame_idx):
        remove_ids = []

        for track_id, track in self.tracks.items():
            if frame_idx - track["last_seen"] > self.max_age:
                remove_ids.append(track_id)

        for track_id in remove_ids:
            del self.tracks[track_id]


# =========================================================
# crop preprocessing
# =========================================================

def trim_number_crop_by_dark_region(
    crop,
    pad_x=0.08,
    pad_y=0.12,
    dark_thresh=200,
    min_area=20
):
    """
    밝은 방호복 배경에서 어두운 숫자 후보 영역을 기준으로 1차 trim.
    단, 숫자가 아닌 어두운 노이즈도 포함될 수 있으므로 최종 crop은 아님.

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

    mask = (gray < dark_thresh).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
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
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    trimmed = crop[y1:y2, x1:x2]

    if trimmed.size == 0:
        return crop, (0, 0)

    return trimmed, (x1, y1)


def get_tight_digit_crop_from_boxes(
    number_crop,
    digits,
    pad_x=0.05,
    pad_y=0.12,
    require_count=4
):
    """
    digit detector가 찾은 숫자 bbox들을 기준으로
    실제 숫자 영역만 최종 tight crop.

    require_count=4이면 숫자 bbox가 정확히 4개일 때만 tight crop 적용.
    숫자가 1~3개만 잡혔을 때 잘라버리면 일부 숫자만 저장될 수 있으므로 기본값은 4.

    return:
        tight_crop,
        tight_box_in_number_crop: (x1, y1, x2, y2),
        used_tight_crop: bool
    """

    if number_crop is None or number_crop.size == 0:
        return number_crop, None, False

    h, w = number_crop.shape[:2]

    if digits is None or len(digits) == 0:
        return number_crop, (0, 0, w, h), False

    if require_count is not None and len(digits) != require_count:
        return number_crop, (0, 0, w, h), False

    x1 = min(float(d["x1"]) for d in digits)
    y1 = min(float(d["y1"]) for d in digits)
    x2 = max(float(d["x2"]) for d in digits)
    y2 = max(float(d["y2"]) for d in digits)

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 3 or bh <= 3:
        return number_crop, (0, 0, w, h), False

    px = bw * pad_x
    py = bh * pad_y

    nx1 = max(0, int(np.floor(x1 - px)))
    ny1 = max(0, int(np.floor(y1 - py)))
    nx2 = min(w, int(np.ceil(x2 + px)))
    ny2 = min(h, int(np.ceil(y2 + py)))

    if nx2 <= nx1 or ny2 <= ny1:
        return number_crop, (0, 0, w, h), False

    tight_crop = number_crop[ny1:ny2, nx1:nx2]

    if tight_crop.size == 0:
        return number_crop, (0, 0, w, h), False

    return tight_crop, (nx1, ny1, nx2, ny2), True


# =========================================================
# digit detector
# =========================================================

def recognize_digits_from_crop(
    digit_model,
    number_crop,
    device="0",
    conf=0.15,
    iou=0.45,
    imgsz=640,
    max_det=8
):
    """
    번호 영역 crop 안에서 숫자 하나하나를 YOLO detector로 탐지한 뒤
    x좌표 기준으로 정렬해서 번호 문자열 생성.

    정확히 4개 숫자가 잡힌 경우에만 raw_number 반환.
    아니면 raw_number는 "" 반환.
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

    digits = sorted(digits, key=lambda d: d["xc"])

    # 4개보다 많이 잡히면 confidence 높은 4개만 선택 후 다시 x좌표 정렬
    if len(digits) > 4:
        digits = sorted(digits, key=lambda d: d["conf"], reverse=True)[:4]
        digits = sorted(digits, key=lambda d: d["xc"])

    if len(digits) != 4:
        return "", digits

    pred_number = "".join(d["digit"] for d in digits)

    if not is_valid_4digit_number(pred_number):
        return "", digits

    return pred_number, digits


# =========================================================
# confirmation logic
# =========================================================

def update_confirmed_number(
    history_deque,
    confirmed_state,
    new_number,
    min_confirm_frames=5,
    lock_ratio=0.6
):
    """
    정확히 4자리 숫자만 history에 넣고,
    같은 번호가 min_confirm_frames 이상 나오고 비율 조건을 만족하면 확정.
    """

    if confirmed_state["confirmed"]:
        return {
            "candidate": confirmed_state["number"],
            "confirmed": confirmed_state["number"],
            "is_confirmed": True,
            "hit_count": confirmed_state["hit_count"],
            "hit_ratio": confirmed_state["hit_ratio"],
        }

    if is_valid_4digit_number(new_number):
        history_deque.append(new_number)

    hist = list(history_deque)

    if len(hist) == 0:
        return {
            "candidate": "",
            "confirmed": "",
            "is_confirmed": False,
            "hit_count": 0,
            "hit_ratio": 0.0,
        }

    counter = Counter(hist)
    top_number, top_count = counter.most_common(1)[0]
    ratio = top_count / len(hist)

    if top_count >= min_confirm_frames and ratio >= lock_ratio:
        confirmed_state["confirmed"] = True
        confirmed_state["number"] = top_number
        confirmed_state["hit_count"] = top_count
        confirmed_state["hit_ratio"] = ratio

        return {
            "candidate": top_number,
            "confirmed": top_number,
            "is_confirmed": True,
            "hit_count": top_count,
            "hit_ratio": ratio,
        }

    return {
        "candidate": top_number,
        "confirmed": "",
        "is_confirmed": False,
        "hit_count": top_count,
        "hit_ratio": ratio,
    }


# =========================================================
# visualization
# =========================================================

def draw_digit_boxes(vis_frame, digits, offset_x, offset_y):
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


def draw_tight_digit_box(vis_frame, tight_box, offset_x, offset_y):
    if tight_box is None:
        return

    tx1, ty1, tx2, ty2 = tight_box

    x1 = int(offset_x + tx1)
    y1 = int(offset_y + ty1)
    x2 = int(offset_x + tx2)
    y2 = int(offset_y + ty2)

    cv2.rectangle(
        vis_frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    cv2.putText(
        vis_frame,
        "TIGHT_DIGIT_CROP",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


def draw_status_text(
    vis_frame,
    x,
    y,
    raw_number,
    confirm_info,
    confirm_frames,
    draw_pending=True
):
    is_confirmed = confirm_info["is_confirmed"]
    confirmed_number = confirm_info["confirmed"]
    candidate = confirm_info["candidate"]
    hit_count = confirm_info["hit_count"]
    hit_ratio = confirm_info["hit_ratio"]

    if is_confirmed:
        text = f"CONFIRMED: {confirmed_number}"
        color = (0, 255, 0)
        thickness = 3
    else:
        if draw_pending and candidate:
            text = f"PENDING: {candidate} {hit_count}/{confirm_frames} ({hit_ratio:.2f})"
        elif raw_number:
            text = f"RAW: {raw_number}"
        else:
            text = "pending"

        color = (0, 200, 255)
        thickness = 2

    cv2.putText(
        vis_frame,
        text,
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        thickness,
        cv2.LINE_AA
    )


def make_output_paths(video_path, out_dir, run_name=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{video_stem}_digit_confirm_{timestamp}"

    out_video_path = out_dir / f"{run_name}.mp4"
    out_csv_path = out_dir / f"{run_name}_confirmed.csv"
    debug_csv_path = out_dir / f"{run_name}_debug.csv"

    return out_video_path, out_csv_path, debug_csv_path, run_name


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
        required=True
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None
    )

    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--person_conf", type=float, default=0.5)
    parser.add_argument("--number_conf", type=float, default=0.5)
    parser.add_argument("--digit_conf", type=float, default=0.15)

    parser.add_argument("--person_iou", type=float, default=0.45)
    parser.add_argument("--number_iou", type=float, default=0.45)
    parser.add_argument("--digit_iou", type=float, default=0.45)

    parser.add_argument("--frame_step", type=int, default=5)

    # number area crop padding
    parser.add_argument("--pad_x", type=float, default=0.0)
    parser.add_argument("--pad_y", type=float, default=0.0)

    # dark trim
    parser.add_argument("--use_trim", action="store_true")
    parser.add_argument("--trim_pad_x", type=float, default=0.08)
    parser.add_argument("--trim_pad_y", type=float, default=0.12)
    parser.add_argument("--dark_thresh", type=int, default=200)
    parser.add_argument("--min_dark_area", type=int, default=20)

    # final tight digit crop
    parser.add_argument(
        "--disable_tight_digit_crop",
        action="store_true",
        help="digit bbox union 기반 최종 tight crop 비활성화"
    )
    parser.add_argument("--tight_pad_x", type=float, default=0.05)
    parser.add_argument("--tight_pad_y", type=float, default=0.12)

    # digit detector
    parser.add_argument("--digit_imgsz", type=int, default=640)

    # confirmation
    parser.add_argument("--number_history", type=int, default=20)
    parser.add_argument("--confirm_frames", type=int, default=5)
    parser.add_argument("--confirm_ratio", type=float, default=0.6)
    parser.add_argument("--draw_pending", action="store_true")

    # simple bbox tracker
    parser.add_argument("--track_iou", type=float, default=0.3)
    parser.add_argument("--track_max_age", type=int, default=15)

    # output/debug
    parser.add_argument("--save_crops", action="store_true")
    parser.add_argument("--crop_dir", type=str, default=None)
    parser.add_argument("--save_raw_trim_pair", action="store_true")
    parser.add_argument("--save_debug_csv", action="store_true")

    args = parser.parse_args()

    use_tight_digit_crop = not args.disable_tight_digit_crop

    # =====================================================
    # output paths
    # =====================================================

    out_video_path, confirmed_csv_path, debug_csv_path, run_name = make_output_paths(
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

    print(f"[OUTPUT VIDEO]      {out_video_path}")
    print(f"[CONFIRMED CSV]     {confirmed_csv_path}")

    if args.save_debug_csv:
        print(f"[DEBUG CSV]         {debug_csv_path}")

    if args.save_crops:
        print(f"[CROP DIR]          {crop_dir}")

    print(f"[TIGHT DIGIT CROP]  {use_tight_digit_crop}")

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

    confirmed_csv_file = open(
        str(confirmed_csv_path),
        "w",
        newline="",
        encoding="utf-8-sig"
    )

    confirmed_writer = csv.writer(confirmed_csv_file)

    confirmed_writer.writerow([
        "confirmed_frame",
        "track_id",
        "confirmed_number",
        "hit_count",
        "hit_ratio",
        "number_area_conf",
        "number_area_bbox_abs"
    ])

    debug_csv_file = None
    debug_writer = None

    if args.save_debug_csv:
        debug_csv_file = open(
            str(debug_csv_path),
            "w",
            newline="",
            encoding="utf-8-sig"
        )

        debug_writer = csv.writer(debug_csv_file)

        debug_writer.writerow([
            "frame",
            "track_id",
            "person_idx",
            "number_area_conf",
            "raw_number",
            "candidate_number",
            "confirmed_number",
            "is_confirmed",
            "hit_count",
            "hit_ratio",
            "digit_count",
            "number_area_bbox_abs",
            "trim_offset",
            "tight_crop_used",
            "tight_digit_box",
            "digits_detail"
        ])

    # =====================================================
    # states
    # =====================================================

    tracker = SimpleBBoxTracker(
        iou_th=args.track_iou,
        max_age=args.track_max_age
    )

    number_histories = defaultdict(
        lambda: deque(maxlen=args.number_history)
    )

    confirmed_states = defaultdict(
        lambda: {
            "confirmed": False,
            "number": "",
            "hit_count": 0,
            "hit_ratio": 0.0,
            "saved": False
        }
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

        if frame_idx % args.frame_step != 0:
            writer.write(vis_frame)
            frame_idx += 1
            continue

        tracker.cleanup(frame_idx)

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

                cv2.rectangle(
                    vis_frame,
                    (px1, py1),
                    (px2, py2),
                    (255, 180, 0),
                    2
                )

                number_results = number_detector.predict(
                    person_crop,
                    conf=args.number_conf,
                    iou=args.number_iou,
                    verbose=False,
                    device=args.device
                )

                if len(number_results) == 0 or number_results[0].boxes is None:
                    person_idx += 1
                    continue

                nboxes = number_results[0].boxes.xyxy.cpu().numpy()
                nconfs = number_results[0].boxes.conf.cpu().numpy()

                if len(nboxes) == 0:
                    person_idx += 1
                    continue

                best_idx = int(np.argmax(nconfs))
                number_area_conf = float(nconfs[best_idx])

                bx1, by1, bx2, by2 = nboxes[best_idx]

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

                if crop_box is None:
                    person_idx += 1
                    continue

                bx1, by1, bx2, by2 = crop_box

                number_crop_raw = person_crop[by1:by2, bx1:bx2]

                if number_crop_raw.size == 0:
                    person_idx += 1
                    continue

                abs_x1 = px1 + bx1
                abs_y1 = py1 + by1
                abs_x2 = px1 + bx2
                abs_y2 = py1 + by2

                number_area_bbox_abs = [abs_x1, abs_y1, abs_x2, abs_y2]

                # track id
                track_id = tracker.update(number_area_bbox_abs, frame_idx)

                # 1차 trim
                if args.use_trim:
                    number_crop, trim_offset = trim_number_crop_by_dark_region(
                        number_crop_raw,
                        pad_x=args.trim_pad_x,
                        pad_y=args.trim_pad_y,
                        dark_thresh=args.dark_thresh,
                        min_area=args.min_dark_area
                    )
                else:
                    number_crop = number_crop_raw
                    trim_offset = (0, 0)

                trim_x, trim_y = trim_offset

                # digit detect
                raw_number, selected_digits = recognize_digits_from_crop(
                    digit_model=digit_detector,
                    number_crop=number_crop,
                    device=args.device,
                    conf=args.digit_conf,
                    iou=args.digit_iou,
                    imgsz=args.digit_imgsz,
                    max_det=8
                )

                # 최종 tight digit crop
                tight_number_crop = number_crop
                tight_digit_box = None
                tight_crop_used = False

                if use_tight_digit_crop:
                    tight_number_crop, tight_digit_box, tight_crop_used = get_tight_digit_crop_from_boxes(
                        number_crop=number_crop,
                        digits=selected_digits,
                        pad_x=args.tight_pad_x,
                        pad_y=args.tight_pad_y,
                        require_count=4
                    )

                final_crop_to_save = tight_number_crop if tight_crop_used else number_crop

                confirm_info = update_confirmed_number(
                    history_deque=number_histories[track_id],
                    confirmed_state=confirmed_states[track_id],
                    new_number=raw_number,
                    min_confirm_frames=args.confirm_frames,
                    lock_ratio=args.confirm_ratio
                )

                # draw number area bbox
                cv2.rectangle(
                    vis_frame,
                    (abs_x1, abs_y1),
                    (abs_x2, abs_y2),
                    (0, 255, 255),
                    3
                )

                # trim bbox absolute offset
                digit_offset_x = abs_x1 + trim_x
                digit_offset_y = abs_y1 + trim_y

                if args.use_trim:
                    th, tw = number_crop.shape[:2]
                    cv2.rectangle(
                        vis_frame,
                        (digit_offset_x, digit_offset_y),
                        (digit_offset_x + tw, digit_offset_y + th),
                        (255, 0, 255),
                        2
                    )

                # individual digit boxes
                draw_digit_boxes(
                    vis_frame,
                    selected_digits,
                    offset_x=digit_offset_x,
                    offset_y=digit_offset_y
                )

                # final tight crop bbox
                if tight_crop_used:
                    draw_tight_digit_box(
                        vis_frame,
                        tight_digit_box,
                        offset_x=digit_offset_x,
                        offset_y=digit_offset_y
                    )

                draw_status_text(
                    vis_frame=vis_frame,
                    x=abs_x1,
                    y=abs_y1,
                    raw_number=raw_number,
                    confirm_info=confirm_info,
                    confirm_frames=args.confirm_frames,
                    draw_pending=args.draw_pending
                )

                # save confirmed csv only once
                if confirm_info["is_confirmed"] and not confirmed_states[track_id]["saved"]:
                    confirmed_writer.writerow([
                        frame_idx,
                        track_id,
                        confirm_info["confirmed"],
                        confirm_info["hit_count"],
                        f"{confirm_info['hit_ratio']:.4f}",
                        f"{number_area_conf:.4f}",
                        number_area_bbox_abs
                    ])

                    confirmed_states[track_id]["saved"] = True

                # debug csv
                if args.save_debug_csv:
                    debug_writer.writerow([
                        frame_idx,
                        track_id,
                        person_idx,
                        f"{number_area_conf:.4f}",
                        raw_number,
                        confirm_info["candidate"],
                        confirm_info["confirmed"],
                        confirm_info["is_confirmed"],
                        confirm_info["hit_count"],
                        f"{confirm_info['hit_ratio']:.4f}",
                        len(selected_digits),
                        number_area_bbox_abs,
                        trim_offset,
                        tight_crop_used,
                        tight_digit_box,
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

                # save crops
                if args.save_crops:
                    base_name = (
                        f"frame_{frame_idx:06d}"
                        f"_track_{track_id:03d}"
                        f"_raw_{raw_number if raw_number else 'none'}"
                        f"_cand_{confirm_info['candidate'] if confirm_info['candidate'] else 'none'}"
                        f"_conf_{confirm_info['confirmed'] if confirm_info['confirmed'] else 'none'}"
                        f"_tight_{int(tight_crop_used)}"
                    )

                    if args.save_raw_trim_pair:
                        cv2.imwrite(
                            str(crop_dir / f"{base_name}_raw.jpg"),
                            number_crop_raw
                        )

                        cv2.imwrite(
                            str(crop_dir / f"{base_name}_trim.jpg"),
                            number_crop
                        )

                        cv2.imwrite(
                            str(crop_dir / f"{base_name}_final.jpg"),
                            final_crop_to_save
                        )

                        if tight_crop_used:
                            cv2.imwrite(
                                str(crop_dir / f"{base_name}_tight.jpg"),
                                tight_number_crop
                            )
                    else:
                        cv2.imwrite(
                            str(crop_dir / f"{base_name}.jpg"),
                            final_crop_to_save
                        )

                person_idx += 1

        writer.write(vis_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    confirmed_csv_file.close()

    if debug_csv_file is not None:
        debug_csv_file.close()

    print("DONE")
    print(f"video saved: {out_video_path}")
    print(f"confirmed csv saved: {confirmed_csv_path}")

    if args.save_debug_csv:
        print(f"debug csv saved: {debug_csv_path}")

    if args.save_crops:
        print(f"crops saved: {crop_dir}")


if __name__ == "__main__":
    main()