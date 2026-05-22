"""
infer_pose_track_digit_cnn.py
=============================

YOLO Pose Tracking 기반 사람별 4자리 번호 인식 영상 추론 코드

동작 구조:
1. 영상 프레임 입력
2. YOLO Pose track()으로 사람 검출 + track_id 생성
3. track_id별 사람 영역 crop
4. DigitCNN으로 4자리 번호 예측
5. track_id별 예측 번호 누적
6. 일정 조건 만족 시 번호 LOCK
7. 결과 영상에 ID / 번호 / 현재 예측 / confidence 표시
8. CSV 저장

실행 예:
python infer_pose_track_digit_cnn.py \
  --video /root/Storage/test_video.mp4 \
  --ckpt ./digit_cnn_runs/best_digit_cnn.pt \
  --pose_model yolov8n-pose.pt \
  --out ./pose_digit_tracking_result.mp4 \
  --csv_out ./pose_digit_tracking_result.csv \
  --roi_mode person \
  --device cuda

상체 ROI 기준:
python infer_pose_track_digit_cnn.py \
  --video /root/Storage/test_video.mp4 \
  --ckpt ./digit_cnn_runs/best_digit_cnn.pt \
  --pose_model yolov8n-pose.pt \
  --out ./pose_digit_tracking_upper_result.mp4 \
  --csv_out ./pose_digit_tracking_upper_result.csv \
  --roi_mode upper_body \
  --device cuda
"""

import os
import csv
import argparse
from collections import defaultdict, deque, Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


# =========================================================
# Digit CNN Model
# train_digit_cnn.py와 동일 구조
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]

        if pool:
            layers.append(nn.MaxPool2d(2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DigitCNN(nn.Module):
    """
    4자리 숫자 인식 CNN
    - 가로 공간을 4칸으로 유지
    - 숫자 분류기는 4개 위치가 공유 (조합 암기 방지)
    """
    def __init__(self, num_positions=4):
        super().__init__()
        self.num_positions = num_positions

        self.backbone = nn.Sequential(
            ConvBlock(3, 32, pool=True),     # 64x160 -> 32x80
            ConvBlock(32, 64, pool=True),    # 32x80  -> 16x40
            ConvBlock(64, 128, pool=True),   # 16x40  -> 8x20
            ConvBlock(128, 256, pool=True),  # 8x20   -> 4x10
            # 세로는 1로, 가로는 4(자리수)로
            nn.AdaptiveAvgPool2d((1, num_positions)),
        )

        # 위치별로 통과시킬 작은 FC. 핵심은 4개 위치가 같은 weight를 씀
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.shared_head = nn.Linear(256, 10)

    def forward(self, x):
        feat = self.backbone(x)              # (B, 256, 1, 4)
        feat = feat.squeeze(2)               # (B, 256, 4)
        feat = feat.permute(0, 2, 1)         # (B, 4, 256)

        # 4개 위치를 batch처럼 펴서 같은 FC/Head를 적용
        B, P, C = feat.shape
        feat = feat.reshape(B * P, C)
        feat = self.shared_fc(feat)
        logits = self.shared_head(feat)      # (B*4, 10)
        logits = logits.reshape(B, P, 10)    # (B, 4, 10)

        # 기존 train 코드와 호환되도록 list로 반환
        return [logits[:, i, :] for i in range(P)]




# =========================================================
# Track ID별 번호 메모리
# =========================================================

class TrackNumberMemory:
    """
    track_id별 번호 인식 결과를 누적하고,
    일정 조건을 만족하면 해당 track_id의 번호를 확정한다.
    """

    def __init__(
        self,
        history_size=20,
        conf_th=0.65,
        min_hits=5,
        lock_ratio=0.55,
        unlock_when_changed=False
    ):
        self.history_size = history_size
        self.conf_th = conf_th
        self.min_hits = min_hits
        self.lock_ratio = lock_ratio
        self.unlock_when_changed = unlock_when_changed

        self.history = defaultdict(lambda: deque(maxlen=self.history_size))
        self.locked_number = {}
        self.last_pred = defaultdict(lambda: "----")
        self.last_conf = defaultdict(float)
        self.total_reads = defaultdict(int)
        self.valid_reads = defaultdict(int)

    def update(self, track_id, pred_label, conf):
        """
        track_id별 예측 번호 저장.
        confidence가 낮으면 history에는 넣지 않고 현재 표시값만 유지한다.
        """

        self.total_reads[track_id] += 1

        if pred_label is None:
            return self.get_display_number(track_id)

        pred_label = str(pred_label)

        if len(pred_label) != 4 or not pred_label.isdigit():
            return self.get_display_number(track_id)

        self.last_pred[track_id] = pred_label
        self.last_conf[track_id] = float(conf)

        if conf < self.conf_th:
            return self.get_display_number(track_id)

        self.history[track_id].append(pred_label)
        self.valid_reads[track_id] += 1

        numbers = list(self.history[track_id])

        if len(numbers) < self.min_hits:
            return self.get_display_number(track_id)

        counter = Counter(numbers)
        best_num, best_count = counter.most_common(1)[0]
        ratio = best_count / max(1, len(numbers))

        if best_count >= self.min_hits and ratio >= self.lock_ratio:
            if track_id not in self.locked_number:
                self.locked_number[track_id] = best_num
            else:
                if self.unlock_when_changed:
                    old_num = self.locked_number[track_id]
                    if old_num != best_num:
                        self.locked_number[track_id] = best_num

        return self.get_display_number(track_id)

    def get_display_number(self, track_id):
        """
        화면에 표시할 번호 반환.
        확정 번호가 있으면 확정 번호 우선.
        없으면 history 최빈값.
        history도 없으면 ----.
        """

        if track_id in self.locked_number:
            return self.locked_number[track_id], True

        numbers = list(self.history[track_id])

        if len(numbers) == 0:
            return "----", False

        counter = Counter(numbers)
        best_num, _ = counter.most_common(1)[0]

        return best_num, False

    def get_last_pred(self, track_id):
        return self.last_pred[track_id]

    def get_last_conf(self, track_id):
        return self.last_conf[track_id]

    def get_stats(self, track_id):
        return {
            "total_reads": self.total_reads[track_id],
            "valid_reads": self.valid_reads[track_id],
            "history_len": len(self.history[track_id])
        }


# =========================================================
# Common Utils
# =========================================================

def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def clip_box(x1, y1, x2, y2, frame_w, frame_h):
    x1 = max(0, min(int(round(x1)), frame_w - 1))
    y1 = max(0, min(int(round(y1)), frame_h - 1))
    x2 = max(0, min(int(round(x2)), frame_w))
    y2 = max(0, min(int(round(y2)), frame_h))

    if x2 <= x1:
        x2 = min(frame_w, x1 + 1)

    if y2 <= y1:
        y2 = min(frame_h, y1 + 1)

    return x1, y1, x2, y2


def expand_box(box, frame_w, frame_h, pad_ratio=0.08):
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    return clip_box(
        x1 - pad_x,
        y1 - pad_y,
        x2 + pad_x,
        y2 + pad_y,
        frame_w,
        frame_h
    )


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def resize_keep_ratio(img, target_w, target_h):
    """
    학습 코드와 동일하게 비율 유지 padding resize.
    """

    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        return np.ones((target_h, target_w, 3), dtype=np.uint8) * 235

    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 235

    xoff = (target_w - new_w) // 2
    yoff = (target_h - new_h) // 2

    canvas[yoff:yoff + new_h, xoff:xoff + new_w] = resized

    return canvas


def preprocess_crop(crop, img_w, img_h, device):
    crop = resize_keep_ratio(crop, img_w, img_h)

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype(np.float32) / 255.0
    crop = np.transpose(crop, (2, 0, 1))

    x = torch.from_numpy(crop).float().unsqueeze(0).to(device)

    return x


@torch.no_grad()
def predict_digit(model, crop, img_w, img_h, device):
    x = preprocess_crop(crop, img_w, img_h, device)

    outputs = model(x)

    digits = []
    digit_confs = []

    for out in outputs:
        prob = F.softmax(out, dim=1)
        conf, pred = torch.max(prob, dim=1)

        digits.append(str(int(pred.item())))
        digit_confs.append(float(conf.item()))

    pred_label = "".join(digits)
    avg_conf = float(np.mean(digit_confs))

    return pred_label, avg_conf, digit_confs


def load_digit_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    model = DigitCNN().to(device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        img_w = int(ckpt.get("img_w", 160))
        img_h = int(ckpt.get("img_h", 64))
        use_bbox = ckpt.get("use_bbox", None)
        bbox_pad = ckpt.get("bbox_pad", None)
        epoch = ckpt.get("epoch", None)
        exact_acc = ckpt.get("exact_acc", None)
        digit_acc = ckpt.get("digit_acc", None)
    else:
        model.load_state_dict(ckpt)
        img_w = 160
        img_h = 64
        use_bbox = None
        bbox_pad = None
        epoch = None
        exact_acc = None
        digit_acc = None

    model.eval()

    ckpt_info = {
        "img_w": img_w,
        "img_h": img_h,
        "use_bbox": use_bbox,
        "bbox_pad": bbox_pad,
        "epoch": epoch,
        "exact_acc": exact_acc,
        "digit_acc": digit_acc
    }

    return model, img_w, img_h, ckpt_info


# =========================================================
# Pose 기반 ROI 생성
# =========================================================

def get_upper_body_box_from_pose(
    kpts,
    person_box,
    frame_w,
    frame_h,
    conf_th=0.25,
    pad_ratio=0.35
):
    """
    COCO Pose index 기준:
    5  = left shoulder
    6  = right shoulder
    11 = left hip
    12 = right hip

    상체 번호가 등/가슴/팔 상부 쪽에 있을 때 사용.
    keypoint가 부족하면 person_box로 fallback.
    """

    if kpts is None:
        return expand_box(person_box, frame_w, frame_h, pad_ratio=0.08)

    target_idxs = [5, 6, 11, 12]
    pts = []

    for idx in target_idxs:
        if idx >= len(kpts):
            continue

        x, y, c = kpts[idx]

        if float(c) >= conf_th:
            pts.append((float(x), float(y)))

    if len(pts) < 2:
        return expand_box(person_box, frame_w, frame_h, pad_ratio=0.08)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    raw_x1 = min(xs)
    raw_y1 = min(ys)
    raw_x2 = max(xs)
    raw_y2 = max(ys)

    person_x1, person_y1, person_x2, person_y2 = person_box
    person_w = person_x2 - person_x1
    person_h = person_y2 - person_y1

    cx = (raw_x1 + raw_x2) / 2
    cy = (raw_y1 + raw_y2) / 2

    bw = max(raw_x2 - raw_x1, person_w * 0.38)
    bh = max(raw_y2 - raw_y1, person_h * 0.28)

    # 어깨-골반 중심 기준으로 상체 영역 확보
    x1 = cx - bw / 2
    x2 = cx + bw / 2
    y1 = cy - bh * 0.75
    y2 = cy + bh * 0.95

    # padding
    bw = x2 - x1
    bh = y2 - y1

    x1 -= bw * pad_ratio
    x2 += bw * pad_ratio
    y1 -= bh * pad_ratio
    y2 += bh * pad_ratio

    return clip_box(x1, y1, x2, y2, frame_w, frame_h)


def get_roi_box(
    roi_mode,
    person_box,
    kpts,
    frame_w,
    frame_h,
    person_pad=0.08,
    upper_body_pad=0.35,
    keypoint_conf_th=0.25
):
    if roi_mode == "upper_body":
        return get_upper_body_box_from_pose(
            kpts=kpts,
            person_box=person_box,
            frame_w=frame_w,
            frame_h=frame_h,
            conf_th=keypoint_conf_th,
            pad_ratio=upper_body_pad
        )

    return expand_box(
        person_box,
        frame_w,
        frame_h,
        pad_ratio=person_pad
    )


# =========================================================
# Visualization
# =========================================================

def draw_label_box(frame, x, y, lines, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """
    여러 줄 텍스트를 잘 보이게 반투명 박스 느낌으로 표시.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    line_h = 25
    pad = 6

    max_w = 0

    for line in lines:
        size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, size[0])

    box_w = max_w + pad * 2
    box_h = line_h * len(lines) + pad * 2

    x1 = max(0, x)
    y1 = max(0, y - box_h)

    x2 = min(frame.shape[1], x1 + box_w)
    y2 = min(frame.shape[0], y1 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text_y = y1 + pad + 18

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x1 + pad, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
        text_y += line_h


def draw_result(
    frame,
    person_box,
    roi_box,
    track_id,
    display_number,
    is_locked,
    current_pred,
    conf,
    digit_confs,
    valid_reads,
    total_reads
):
    px1, py1, px2, py2 = person_box
    x1, y1, x2, y2 = roi_box

    # 사람 bbox
    cv2.rectangle(
        frame,
        (px1, py1),
        (px2, py2),
        (255, 200, 0),
        2
    )

    # CNN 입력 ROI bbox
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    status = "LOCK" if is_locked else "READ"

    digit_conf_text = ",".join([f"{c:.2f}" for c in digit_confs])

    lines = [
        f"ID:{track_id}  NUM:{display_number}  {status}",
        f"cur:{current_pred}  conf:{conf:.2f}",
        f"read:{valid_reads}/{total_reads}"
    ]

    # digit별 conf까지 너무 복잡하면 아래 줄은 주석 처리 가능
    # lines.append(f"dconf:{digit_conf_text}")

    label_x = x1
    label_y = max(35, y1 - 8)

    draw_label_box(
        frame,
        label_x,
        label_y,
        lines,
        bg_color=(0, 0, 0),
        text_color=(0, 255, 0)
    )


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    # input / output
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--pose_model", type=str, default="yolov8n-pose.pt")

    parser.add_argument("--out", type=str, default="pose_digit_tracking_result.mp4")
    parser.add_argument("--csv_out", type=str, default="pose_digit_tracking_result.csv")

    # device
    parser.add_argument("--device", type=str, default="cuda")

    # YOLO Pose params
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pose_conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker", type=str, default="botsort.yaml")

    # ROI params
    parser.add_argument(
        "--roi_mode",
        type=str,
        default="person",
        choices=["person", "upper_body"],
        help="person: 사람 전체 crop / upper_body: 어깨-골반 기반 상체 crop"
    )
    parser.add_argument("--person_pad", type=float, default=0.08)
    parser.add_argument("--upper_body_pad", type=float, default=0.35)
    parser.add_argument("--keypoint_conf_th", type=float, default=0.25)
    parser.add_argument("--min_box_area", type=int, default=1000)

    # Number memory params
    parser.add_argument("--number_conf_th", type=float, default=0.65)
    parser.add_argument("--number_history", type=int, default=20)
    parser.add_argument("--number_min_hits", type=int, default=5)
    parser.add_argument("--number_lock_ratio", type=float, default=0.55)
    parser.add_argument("--unlock_when_changed", action="store_true")

    # runtime
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save_crop_dir", type=str, default="")
    parser.add_argument("--save_crop_every", type=int, default=15)

    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame_stride는 1 이상이어야 합니다.")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        yolo_device = 0
    else:
        device = torch.device("cpu")
        yolo_device = "cpu"

    ensure_parent_dir(args.out)
    ensure_parent_dir(args.csv_out)

    if args.save_crop_dir:
        os.makedirs(args.save_crop_dir, exist_ok=True)

    print("=" * 80)
    print("YOLO Pose Tracking + Digit CNN")
    print("=" * 80)
    print(f"video              : {args.video}")
    print(f"pose_model         : {args.pose_model}")
    print(f"digit ckpt         : {args.ckpt}")
    print(f"device             : {device}")
    print(f"roi_mode           : {args.roi_mode}")
    print(f"number_conf_th     : {args.number_conf_th}")
    print(f"number_history     : {args.number_history}")
    print(f"number_min_hits    : {args.number_min_hits}")
    print(f"number_lock_ratio  : {args.number_lock_ratio}")
    print("=" * 80)

    # Load models
    pose_model = YOLO(args.pose_model)

    digit_model, img_w, img_h, ckpt_info = load_digit_model(
        ckpt_path=args.ckpt,
        device=device
    )

    print("[Digit CNN checkpoint info]")
    print(f"img_w      : {img_w}")
    print(f"img_h      : {img_h}")
    print(f"use_bbox   : {ckpt_info['use_bbox']}")
    print(f"bbox_pad   : {ckpt_info['bbox_pad']}")
    print(f"epoch      : {ckpt_info['epoch']}")
    print(f"exact_acc  : {ckpt_info['exact_acc']}")
    print(f"digit_acc  : {ckpt_info['digit_acc']}")
    print("=" * 80)

    number_memory = TrackNumberMemory(
        history_size=args.number_history,
        conf_th=args.number_conf_th,
        min_hits=args.number_min_hits,
        lock_ratio=args.number_lock_ratio,
        unlock_when_changed=args.unlock_when_changed
    )

    # Video open
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("[Video info]")
    print(f"fps          : {fps}")
    print(f"frame_w      : {frame_w}")
    print(f"frame_h      : {frame_h}")
    print(f"total_frames : {total_frames}")
    print("=" * 80)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        raise RuntimeError(f"결과 영상 저장 객체 생성 실패: {args.out}")

    csv_file = open(args.csv_out, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame_idx",
        "track_id",
        "current_pred",
        "display_number",
        "is_locked",
        "confidence",
        "digit_conf_1",
        "digit_conf_2",
        "digit_conf_3",
        "digit_conf_4",
        "valid_reads",
        "total_reads",
        "person_x1",
        "person_y1",
        "person_x2",
        "person_y2",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "roi_mode",
        "crop_path"
    ])

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # frame_stride가 있어도 tracking은 매 프레임 돌리는 게 ID 유지에 유리함
        results = pose_model.track(
            source=frame,
            persist=True,
            verbose=False,
            conf=args.pose_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=yolo_device,
            tracker=args.tracker
        )

        if len(results) == 0:
            writer.write(frame)
            continue

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            writer.write(frame)
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        has_track_id = result.boxes.id is not None

        if has_track_id:
            track_ids = result.boxes.id.cpu().numpy().astype(int).tolist()
        else:
            # 원칙적으로 track_id가 있어야 함.
            # 초기 몇 프레임에서 id가 없을 수 있어 임시 id 부여.
            track_ids = [-(i + 1) for i in range(len(boxes))]

        if result.keypoints is not None and result.keypoints.data is not None:
            keypoints = result.keypoints.data.cpu().numpy()
        else:
            keypoints = None

        for det_idx, box in enumerate(boxes):
            track_id = int(track_ids[det_idx])

            px1, py1, px2, py2 = box
            person_box = clip_box(px1, py1, px2, py2, frame_w, frame_h)

            if box_area(person_box) < args.min_box_area:
                continue

            kpts = keypoints[det_idx] if keypoints is not None else None

            roi_box = get_roi_box(
                roi_mode=args.roi_mode,
                person_box=person_box,
                kpts=kpts,
                frame_w=frame_w,
                frame_h=frame_h,
                person_pad=args.person_pad,
                upper_body_pad=args.upper_body_pad,
                keypoint_conf_th=args.keypoint_conf_th
            )

            x1, y1, x2, y2 = roi_box
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # frame_stride에 해당하지 않는 프레임은 기존 display 번호만 표시
            if frame_idx % args.frame_stride == 0:
                current_pred, conf, digit_confs = predict_digit(
                    model=digit_model,
                    crop=crop,
                    img_w=img_w,
                    img_h=img_h,
                    device=device
                )

                display_number, is_locked = number_memory.update(
                    track_id=track_id,
                    pred_label=current_pred,
                    conf=conf
                )
            else:
                display_number, is_locked = number_memory.get_display_number(track_id)
                current_pred = number_memory.get_last_pred(track_id)
                conf = number_memory.get_last_conf(track_id)
                digit_confs = [0.0, 0.0, 0.0, 0.0]

            stats = number_memory.get_stats(track_id)

            crop_path = ""

            if args.save_crop_dir and frame_idx % args.save_crop_every == 0:
                crop_name = f"frame_{frame_idx:06d}_id_{track_id}_pred_{current_pred}.jpg"
                crop_path = os.path.join(args.save_crop_dir, crop_name)
                cv2.imwrite(crop_path, crop)

            csv_writer.writerow([
                frame_idx,
                track_id,
                current_pred,
                display_number,
                int(is_locked),
                round(float(conf), 4),
                round(float(digit_confs[0]), 4),
                round(float(digit_confs[1]), 4),
                round(float(digit_confs[2]), 4),
                round(float(digit_confs[3]), 4),
                stats["valid_reads"],
                stats["total_reads"],
                person_box[0],
                person_box[1],
                person_box[2],
                person_box[3],
                roi_box[0],
                roi_box[1],
                roi_box[2],
                roi_box[3],
                args.roi_mode,
                crop_path
            ])

            draw_result(
                frame=frame,
                person_box=person_box,
                roi_box=roi_box,
                track_id=track_id,
                display_number=display_number,
                is_locked=is_locked,
                current_pred=current_pred,
                conf=conf,
                digit_confs=digit_confs,
                valid_reads=stats["valid_reads"],
                total_reads=stats["total_reads"]
            )

        writer.write(frame)

        if args.display:
            cv2.imshow("YOLO Pose Tracking + Digit CNN", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        if frame_idx % 100 == 0:
            print(f"processed: {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    csv_file.close()

    if args.display:
        cv2.destroyAllWindows()

    print("=" * 80)
    print("완료")
    print(f"결과 영상: {args.out}")
    print(f"결과 CSV : {args.csv_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()