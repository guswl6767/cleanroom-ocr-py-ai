"""
infer_full_pipeline.py
======================

3단계 풀 파이프라인:
1. YOLO Pose: 사람 검출 + tracking (track_id)
2. YOLO 등번호 detector: 사람 안에서 등번호 영역 검출
3. CRNN/DigitCNN: 등번호 인식

번호가 없는 사람은 자동으로 인식 안 됨 (detector가 못 찾으니까).

실행:
python infer_full_pipeline.py \
    --video /path/to/video.mp4 \
    --pose_model yolov8n-pose.pt \
    --number_detector /root/.../runs/detect/train/weights/best.pt \
    --recognizer_ckpt /root/.../crnn_runs/best_crnn.pt \
    --device cuda
"""

import os
import csv
import argparse
from collections import defaultdict, deque, Counter
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


# CTC constants
NUM_CLASSES = 11
BLANK_IDX = 10
FIXED_LENGTH = 4


# ========================================================
# Models (학습 코드와 100% 동일)
# ========================================================

class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, kernel_size=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=hidden,
            num_layers=2, bidirectional=True,
            dropout=0.2, batch_first=False,
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2)
        feat = feat.permute(2, 0, 1)
        rnn_out, _ = self.rnn(feat)
        return self.fc(rnn_out)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=True),
            nn.AdaptiveAvgPool2d((1, 4)),
        )
        self.head1 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 10))
        self.head2 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 10))
        self.head3 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 10))
        self.head4 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 10))

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.squeeze(2)
        feat = feat.permute(0, 2, 1)
        return [
            self.head1(feat[:, 0, :]),
            self.head2(feat[:, 1, :]),
            self.head3(feat[:, 2, :]),
            self.head4(feat[:, 3, :]),
        ]


# ========================================================
# Recognizer 추론
# ========================================================

@torch.no_grad()
def predict_crnn(model, x, device):
    logits = model(x.to(device))
    probs = logits.softmax(dim=2)
    max_probs, preds = probs.max(dim=2)
    preds = preds.permute(1, 0).cpu().numpy()
    max_probs = max_probs.permute(1, 0).cpu().numpy()

    results = []
    for seq, conf_seq in zip(preds, max_probs):
        out_chars, out_confs = [], []
        prev = -1
        for p, c in zip(seq, conf_seq):
            if p != prev and p != BLANK_IDX:
                out_chars.append(str(int(p)))
                out_confs.append(float(c))
            prev = p
        pred_str = "".join(out_chars)

        if len(out_chars) == FIXED_LENGTH:
            avg_conf = float(np.mean(out_confs))
            digit_confs = out_confs
        elif len(out_chars) > FIXED_LENGTH:
            pred_str = pred_str[:FIXED_LENGTH]
            digit_confs = out_confs[:FIXED_LENGTH]
            avg_conf = float(np.mean(digit_confs)) * 0.7
        else:
            digit_confs = out_confs + [0.0] * (FIXED_LENGTH - len(out_confs))
            avg_conf = float(np.mean(out_confs)) * 0.5 if out_confs else 0.0
        results.append((pred_str, avg_conf, digit_confs))
    return results


@torch.no_grad()
def predict_digit_cnn(model, x, device):
    outputs = model(x.to(device))
    probs = [F.softmax(o, dim=1) for o in outputs]
    confs = torch.stack([p.max(dim=1).values for p in probs], dim=1)
    preds = torch.stack([p.argmax(dim=1) for p in probs], dim=1)

    results = []
    for i in range(preds.shape[0]):
        digits = preds[i].cpu().numpy().tolist()
        digit_confs = confs[i].cpu().numpy().tolist()
        pred_str = "".join(str(int(d)) for d in digits)
        avg_conf = float(np.mean(digit_confs))
        results.append((pred_str, avg_conf, digit_confs))
    return results


def load_recognizer(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("체크포인트 형식 이상")

    arch = ckpt.get("arch", "unknown")
    img_w = int(ckpt.get("img_w", 128))
    img_h = int(ckpt.get("img_h", 256))

    if arch == "CRNN_CTC":
        model = CRNN().to(device)
        predict_fn = predict_crnn
    elif "DigitCNN" in arch:
        model = DigitCNN().to(device)
        predict_fn = predict_digit_cnn
    else:
        if "crnn" in ckpt_path.lower():
            model = CRNN().to(device)
            predict_fn = predict_crnn
            arch = "CRNN_CTC (추측)"
        else:
            model = DigitCNN().to(device)
            predict_fn = predict_digit_cnn
            arch = "DigitCNN (추측)"

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, predict_fn, img_w, img_h, {
        "arch": arch, "epoch": ckpt.get("epoch"),
        "exact_acc": ckpt.get("exact_acc"),
    }


# ========================================================
# Track ID 메모리
# ========================================================

class TrackNumberMemory:
    def __init__(self, history_size=20, conf_th=0.70, min_hits=5,
                 lock_ratio=0.55, unlock_when_changed=False):
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

        nums = list(self.history[track_id])
        if len(nums) < self.min_hits:
            return self.get_display_number(track_id)

        counter = Counter(nums)
        best_num, best_cnt = counter.most_common(1)[0]
        ratio = best_cnt / max(1, len(nums))

        if best_cnt >= self.min_hits and ratio >= self.lock_ratio:
            if track_id not in self.locked_number:
                self.locked_number[track_id] = best_num
            elif self.unlock_when_changed and self.locked_number[track_id] != best_num:
                self.locked_number[track_id] = best_num

        return self.get_display_number(track_id)

    def get_display_number(self, track_id):
        if track_id in self.locked_number:
            return self.locked_number[track_id], True
        nums = list(self.history[track_id])
        if not nums:
            return "----", False
        best_num, _ = Counter(nums).most_common(1)[0]
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


# ========================================================
# Utils
# ========================================================

def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def expand_box(box, frame_w, frame_h, pad_ratio=0.10):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return clip_box(
        x1 - bw * pad_ratio, y1 - bh * pad_ratio,
        x2 + bw * pad_ratio, y2 + bh * pad_ratio,
        frame_w, frame_h
    )


def resize_keep_ratio(img, target_w, target_h):
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


def preprocess(img, img_w, img_h):
    img = resize_keep_ratio(img, img_w, img_h)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ========================================================
# Visualization
# ========================================================

def draw_label_box(frame, x, y, lines, bg_color=(0, 0, 0), text_color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_h = 23
    pad = 5
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
    text_y = y1 + pad + 17
    for line in lines:
        cv2.putText(frame, line, (x1 + pad, text_y), font, font_scale,
                    text_color, thickness, cv2.LINE_AA)
        text_y += line_h


# ========================================================
# Main
# ========================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--pose_model", type=str, default="yolov8n-pose.pt")
    p.add_argument("--number_detector", type=str, required=True,
                   help="등번호 검출 yolov8 ckpt (best.pt)")
    p.add_argument("--recognizer_ckpt", type=str, required=True,
                   help="CRNN/DigitCNN 인식 ckpt")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--csv_out", type=str, default="")
    p.add_argument("--out_dir", type=str, default="./inference_results")
    p.add_argument("--device", type=str, default="cuda")

    # YOLO Pose
    p.add_argument("--pose_conf", type=float, default=0.35)
    p.add_argument("--pose_iou", type=float, default=0.5)
    p.add_argument("--pose_imgsz", type=int, default=640)
    p.add_argument("--tracker", type=str, default="botsort.yaml")

    # 등번호 detector
    p.add_argument("--number_conf", type=float, default=0.30,
                   help="등번호 검출 conf 임계값")
    p.add_argument("--number_iou", type=float, default=0.45)
    p.add_argument("--number_imgsz", type=int, default=320,
                   help="사람 crop 입력 시 작아도 됨 (320이면 충분)")
    p.add_argument("--number_pad", type=float, default=0.10,
                   help="검출된 등번호 영역에 추가 padding")

    p.add_argument("--min_box_area", type=int, default=2000)

    # 번호 메모리
    p.add_argument("--number_conf_th", type=float, default=0.70)
    p.add_argument("--number_history", type=int, default=20)
    p.add_argument("--number_min_hits", type=int, default=5)
    p.add_argument("--number_lock_ratio", type=float, default=0.55)

    # Runtime
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--display", action="store_true")
    p.add_argument("--save_crop_dir", type=str, default="")
    p.add_argument("--save_crop_every", type=int, default=30)
    args = p.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        yolo_device = 0
    else:
        device = torch.device("cpu")
        yolo_device = "cpu"

    # 자동 출력 이름
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    rec_stem = os.path.splitext(os.path.basename(args.recognizer_ckpt))[0]
    det_stem = os.path.splitext(os.path.basename(args.number_detector))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_name = f"{video_stem}__det-{det_stem}__rec-{rec_stem}__{timestamp}"
    if not args.out:
        args.out = os.path.join(args.out_dir, f"{auto_name}.mp4")
    if not args.csv_out:
        args.csv_out = os.path.join(args.out_dir, f"{auto_name}.csv")

    ensure_parent_dir(args.out)
    ensure_parent_dir(args.csv_out)
    if args.save_crop_dir:
        os.makedirs(args.save_crop_dir, exist_ok=True)

    print("=" * 80)
    print("3단계 풀 파이프라인 추론")
    print("=" * 80)
    print(f"video           : {args.video}")
    print(f"pose model      : {args.pose_model}")
    print(f"number detector : {args.number_detector}")
    print(f"recognizer      : {args.recognizer_ckpt}")
    print(f"device          : {device}")
    print(f"out video       : {args.out}")
    print(f"out csv         : {args.csv_out}")
    print("=" * 80)

    # 모델 로드
    pose_model = YOLO(args.pose_model)
    number_detector = YOLO(args.number_detector)
    recognizer, predict_fn, img_w, img_h, rec_info = load_recognizer(
        args.recognizer_ckpt, device
    )

    print("\n[인식 모델 정보]")
    for k, v in rec_info.items():
        print(f"  {k}: {v}")
    print()

    number_memory = TrackNumberMemory(
        history_size=args.number_history,
        conf_th=args.number_conf_th,
        min_hits=args.number_min_hits,
        lock_ratio=args.number_lock_ratio,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] fps={fps}, {frame_w}x{frame_h}, total={total_frames}\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"결과 영상 저장 실패: {args.out}")

    csv_file = open(args.csv_out, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_idx", "track_id", "current_pred", "display_number", "is_locked",
        "rec_confidence", "det_confidence",
        "digit_conf_1", "digit_conf_2", "digit_conf_3", "digit_conf_4",
        "valid_reads", "total_reads",
        "person_x1", "person_y1", "person_x2", "person_y2",
        "number_x1", "number_y1", "number_x2", "number_y2",
        "crop_path"
    ])

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ========================================
        # 1단계: YOLO Pose 사람 검출 + tracking
        # ========================================
        pose_results = pose_model.track(
            source=frame, persist=True, verbose=False,
            conf=args.pose_conf, iou=args.pose_iou, imgsz=args.pose_imgsz,
            device=yolo_device, tracker=args.tracker
        )

        if not pose_results or pose_results[0].boxes is None or len(pose_results[0].boxes) == 0:
            writer.write(frame)
            if args.display:
                cv2.imshow("Result", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        pr = pose_results[0]
        person_boxes = pr.boxes.xyxy.cpu().numpy()
        has_id = pr.boxes.id is not None
        track_ids = (pr.boxes.id.cpu().numpy().astype(int).tolist()
                     if has_id else [-(i+1) for i in range(len(person_boxes))])

        # ========================================
        # 2단계 + 3단계: 각 사람마다 등번호 검출 + 인식
        # ========================================
        # 효율을 위해 사람 crop들을 모아서 batch 검출
        person_crops = []
        person_info = []  # [(track_id, person_box_in_frame, crop_offset)]

        for i, box in enumerate(person_boxes):
            track_id = int(track_ids[i])
            px1, py1, px2, py2 = box
            person_box = clip_box(px1, py1, px2, py2, frame_w, frame_h)
            if (person_box[2]-person_box[0]) * (person_box[3]-person_box[1]) < args.min_box_area:
                continue
            crop = frame[person_box[1]:person_box[3], person_box[0]:person_box[2]]
            if crop.size == 0:
                continue
            person_crops.append(crop)
            person_info.append((track_id, person_box))

        if not person_crops or frame_idx % args.frame_stride != 0:
            # 기존 메모리 기반 표시만
            for track_id, person_box in person_info:
                display_number, is_locked = number_memory.get_display_number(track_id)
                stats = number_memory.get_stats(track_id)
                px1, py1, px2, py2 = person_box
                color = (0, 255, 0) if is_locked else (255, 200, 0)
                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                if display_number != "----":
                    lines = [
                        f"ID:{track_id}  NUM:{display_number}  {'LOCK' if is_locked else 'READ'}",
                        f"read:{stats['valid_reads']}/{stats['total_reads']}"
                    ]
                    draw_label_box(frame, px1, max(35, py1-8), lines)
            writer.write(frame)
            if args.display:
                cv2.imshow("Result", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        # 등번호 detector — 각 사람 crop마다
        det_results = number_detector.predict(
            source=person_crops, verbose=False,
            conf=args.number_conf, iou=args.number_iou,
            imgsz=args.number_imgsz, device=yolo_device
        )

        # 검출된 등번호 영역들을 모아서 recognizer batch 실행
        number_crops = []
        number_meta = []  # (track_id, person_box, number_box_in_frame, det_conf)

        for i, det_result in enumerate(det_results):
            track_id, person_box = person_info[i]
            person_crop = person_crops[i]
            crop_h, crop_w = person_crop.shape[:2]
            ox, oy = person_box[0], person_box[1]

            if det_result.boxes is None or len(det_result.boxes) == 0:
                continue

            det_boxes = det_result.boxes.xyxy.cpu().numpy()
            det_confs = det_result.boxes.conf.cpu().numpy()

            # conf 가장 높은 detection만 사용 (한 사람당 등번호 1개)
            best_idx = np.argmax(det_confs)
            x1, y1, x2, y2 = det_boxes[best_idx]
            det_conf = float(det_confs[best_idx])

            # padding 추가
            x1, y1, x2, y2 = expand_box(
                (x1, y1, x2, y2), crop_w, crop_h, pad_ratio=args.number_pad
            )
            num_crop = person_crop[y1:y2, x1:x2]
            if num_crop.size == 0:
                continue

            # frame 좌표계로 변환
            number_box_frame = (ox + x1, oy + y1, ox + x2, oy + y2)

            number_crops.append(num_crop)
            number_meta.append((track_id, person_box, number_box_frame, det_conf))

        # ========================================
        # 3단계: Recognizer batch
        # ========================================
        if number_crops:
            batch_imgs = [preprocess(c, img_w, img_h) for c in number_crops]
            x = torch.from_numpy(np.stack(batch_imgs, axis=0)).float()
            rec_results = predict_fn(recognizer, x, device)
        else:
            rec_results = []

        # 인식 결과를 track_id별 메모리에 업데이트
        recognized_track_ids = set()
        for (track_id, person_box, number_box, det_conf), (pred_str, rec_conf, digit_confs) in zip(
            number_meta, rec_results
        ):
            recognized_track_ids.add(track_id)
            display_number, is_locked = number_memory.update(track_id, pred_str, rec_conf)
            stats = number_memory.get_stats(track_id)

            crop_path = ""
            if args.save_crop_dir and frame_idx % args.save_crop_every == 0:
                # number crop 저장
                idx_in_meta = number_meta.index((track_id, person_box, number_box, det_conf))
                num_crop = number_crops[idx_in_meta]
                crop_name = f"frame_{frame_idx:06d}_id_{track_id}_pred_{pred_str}.jpg"
                crop_path = os.path.join(args.save_crop_dir, crop_name)
                cv2.imwrite(crop_path, num_crop)

            dc = list(digit_confs) + [0.0] * 4
            dc = dc[:4]

            csv_writer.writerow([
                frame_idx, track_id, pred_str, display_number, int(is_locked),
                round(float(rec_conf), 4), round(float(det_conf), 4),
                round(float(dc[0]), 4), round(float(dc[1]), 4),
                round(float(dc[2]), 4), round(float(dc[3]), 4),
                stats["valid_reads"], stats["total_reads"],
                person_box[0], person_box[1], person_box[2], person_box[3],
                number_box[0], number_box[1], number_box[2], number_box[3],
                crop_path
            ])

            # 그리기 — 사람 박스 + 등번호 박스 + 라벨
            px1, py1, px2, py2 = person_box
            nx1, ny1, nx2, ny2 = number_box
            color = (0, 255, 0) if is_locked else (255, 200, 0)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 100, 255), 2)
            lines = [
                f"ID:{track_id}  NUM:{display_number}  {'LOCK' if is_locked else 'READ'}",
                f"cur:{pred_str}  rec:{rec_conf:.2f} det:{det_conf:.2f}",
                f"read:{stats['valid_reads']}/{stats['total_reads']}"
            ]
            draw_label_box(frame, px1, max(35, py1-8), lines)

        # 등번호 검출 안 된 사람들 — 박스만 그리기
        for track_id, person_box in person_info:
            if track_id in recognized_track_ids:
                continue
            px1, py1, px2, py2 = person_box
            display_number, is_locked = number_memory.get_display_number(track_id)
            stats = number_memory.get_stats(track_id)

            color = (0, 255, 0) if is_locked else (128, 128, 128)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            if display_number != "----":
                lines = [
                    f"ID:{track_id}  NUM:{display_number}  {'LOCK' if is_locked else 'READ'}",
                    f"(no detect this frame)"
                ]
                draw_label_box(frame, px1, max(35, py1-8), lines)
            else:
                cv2.putText(frame, f"ID:{track_id}", (px1+5, py1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(frame)
        if args.display:
            cv2.imshow("Result", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 100 == 0:
            print(f"processed: {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    csv_file.close()
    if args.display:
        cv2.destroyAllWindows()

    print(f"\n완료. video={args.out}, csv={args.csv_out}")


if __name__ == "__main__":
    main()