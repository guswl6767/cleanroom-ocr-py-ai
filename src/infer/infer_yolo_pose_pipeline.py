"""
infer_video_pose_pipeline.py
=============================

동영상 추론 풀 파이프라인:
1. YOLO Pose로 사람 검출 + 트래킹 (track_id 부여)
2. 어깨/허리 keypoint로 등 상단 ROI 계산 (합성과 동일 방식)
3. ROI crop을 학습된 모델(CRNN 또는 DigitCNN)에 입력
4. track_id별 번호 누적 + LOCK 메모리
5. 결과 영상 / CSV 저장

지원 모델: CRNN+CTC, DigitCNN v5 (자동 분기)

실행 예:
python infer_video_pose_pipeline.py \
    --video /path/to/video.mp4 \
    --ckpt ./crnn_runs/best_crnn.pt \
    --pose_model yolov8n-pose.pt \
    --out ./result.mp4 \
    --csv_out ./result.csv \
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


# ========================================================
# COCO Pose keypoint index
# ========================================================
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

# CTC constants
NUM_CLASSES = 11
BLANK_IDX = 10
FIXED_LENGTH = 4


# ========================================================
# CRNN 모델 (학습 코드와 100% 동일)
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


# ========================================================
# DigitCNN v5 (학습 코드와 100% 동일)
# ========================================================

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
# 추론 함수
# ========================================================

@torch.no_grad()
def predict_crnn(model, x, device):
    logits = model(x.to(device))    # (T, B, C)
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


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("체크포인트 형식 이상")

    arch = ckpt.get("arch", "unknown")
    img_w = int(ckpt.get("img_w", 128))
    img_h = int(ckpt.get("img_h", 256))

    if arch == "CRNN_CTC":
        model = CRNN().to(device)
        predict_fn = predict_crnn
    elif arch in ("DigitCNN_v5_per_position", "DigitCNN", "DigitCNN_v2_person_crop"):
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

    info = {
        "arch": arch, "img_w": img_w, "img_h": img_h,
        "epoch": ckpt.get("epoch"),
        "exact_acc": ckpt.get("exact_acc"),
        "digit_acc": ckpt.get("digit_acc"),
    }
    return model, predict_fn, img_w, img_h, info


# ========================================================
# Pose 기반 ROI (합성 코드와 동일 로직)
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


def compute_upper_back_roi(kpts_xy, kpts_conf, person_box, frame_w, frame_h,
                            kp_conf_thr=0.20, roi_pad_x=0.0, roi_pad_y=0.0):
    """
    합성 코드의 sample_upper_back_roi()와 동일한 방식.
    어깨와 허리 keypoint로 등 상단 ROI 계산.

    kpts_xy: (17, 2) — keypoint xy
    kpts_conf: (17,) — keypoint conf
    person_box: (x1, y1, x2, y2)
    반환: (x1, y1, x2, y2) ROI 또는 None (잡기 어려우면)
    """
    box_x1, box_y1, box_x2, box_y2 = person_box
    box_w = box_x2 - box_x1
    box_h = box_y2 - box_y1

    # 어깨 우선 확인
    left_sh_ok = float(kpts_conf[LEFT_SHOULDER]) >= kp_conf_thr
    right_sh_ok = float(kpts_conf[RIGHT_SHOULDER]) >= kp_conf_thr

    if not (left_sh_ok and right_sh_ok):
        # fallback: person_box 상단 사용
        cx = (box_x1 + box_x2) / 2
        shoulder_y = box_y1 + box_h * 0.25
        shoulder_width = box_w * 0.45
        torso_h = box_h * 0.42
    else:
        lsh = kpts_xy[LEFT_SHOULDER]
        rsh = kpts_xy[RIGHT_SHOULDER]
        cx = (float(lsh[0]) + float(rsh[0])) / 2
        shoulder_y = (float(lsh[1]) + float(rsh[1])) / 2
        shoulder_width = float(np.linalg.norm(lsh - rsh))
        if shoulder_width < 10:
            shoulder_width = box_w * 0.45

        # 허리 keypoint로 torso_h 계산
        left_hip_ok = float(kpts_conf[LEFT_HIP]) >= kp_conf_thr
        right_hip_ok = float(kpts_conf[RIGHT_HIP]) >= kp_conf_thr
        hip_ys = []
        if left_hip_ok:
            hip_ys.append(float(kpts_xy[LEFT_HIP][1]))
        if right_hip_ok:
            hip_ys.append(float(kpts_xy[RIGHT_HIP][1]))
        if hip_ys:
            hip_y = sum(hip_ys) / len(hip_ys)
            torso_h = hip_y - shoulder_y
        else:
            torso_h = box_h * 0.45
        if torso_h < frame_h * 0.05:
            torso_h = box_h * 0.45

    # 합성 코드와 동일한 ROI 위치 (어깨에서 24~42% 아래, 어깨 너비의 85~120%)
    # 추론에선 평균값 사용 (변동 X)
    roi_cx = cx
    roi_cy = shoulder_y + 0.33 * torso_h     # 합성: 0.24~0.42 중간
    roi_w = shoulder_width * 1.05            # 합성: 0.85~1.20 중간
    roi_h = torso_h * 0.30                   # 합성: 0.24~0.36 중간

    # 패딩 추가 (추론 시 약간 더 여유 있게)
    roi_w *= (1 + roi_pad_x)
    roi_h *= (1 + roi_pad_y)

    roi_w = max(70, roi_w)
    roi_h = max(38, roi_h)

    x1 = roi_cx - roi_w / 2
    y1 = roi_cy - roi_h / 2
    x2 = roi_cx + roi_w / 2
    y2 = roi_cy + roi_h / 2

    return clip_box(x1, y1, x2, y2, frame_w, frame_h)


# ========================================================
# Track ID별 번호 메모리
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
# 공통 전처리
# ========================================================

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


def draw_result(frame, person_box, roi_box, track_id, display_number,
                is_locked, current_pred, conf, valid_reads, total_reads):
    px1, py1, px2, py2 = person_box
    rx1, ry1, rx2, ry2 = roi_box

    person_color = (0, 255, 0) if is_locked else (255, 200, 0)
    cv2.rectangle(frame, (px1, py1), (px2, py2), person_color, 2)
    # ROI는 빨간색
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 100, 255), 2)

    status = "LOCK" if is_locked else "READ"
    lines = [
        f"ID:{track_id}  NUM:{display_number}  {status}",
        f"cur:{current_pred}  conf:{conf:.2f}",
        f"read:{valid_reads}/{total_reads}",
    ]
    draw_label_box(frame, px1, max(35, py1 - 8), lines)


# ========================================================
# Main
# ========================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--pose_model", type=str, default="yolov8n-pose.pt")
    p.add_argument("--out", type=str, default="",
                   help="결과 영상 경로. 비우면 자동 생성 (video명 + ckpt명 + 타임스탬프)")
    p.add_argument("--csv_out", type=str, default="",
                   help="결과 CSV 경로. 비우면 자동 생성")
    p.add_argument("--out_dir", type=str, default="./inference_results",
                   help="자동 생성 시 저장될 폴더")
    p.add_argument("--device", type=str, default="cuda")

    # YOLO Pose params
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--pose_conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--tracker", type=str, default="botsort.yaml")
    p.add_argument("--kp_conf_thr", type=float, default=0.20,
                   help="keypoint 신뢰도 임계값")

    # ROI params (합성과 동일 기준)
    p.add_argument("--roi_pad_x", type=float, default=0.10,
                   help="ROI 가로 패딩 비율 (추론 여유)")
    p.add_argument("--roi_pad_y", type=float, default=0.10)
    p.add_argument("--min_box_area", type=int, default=2000)

    # Number memory
    p.add_argument("--number_conf_th", type=float, default=0.70)
    p.add_argument("--number_history", type=int, default=20)
    p.add_argument("--number_min_hits", type=int, default=5)
    p.add_argument("--number_lock_ratio", type=float, default=0.55)
    p.add_argument("--unlock_when_changed", action="store_true")

    # Runtime
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--display", action="store_true")
    p.add_argument("--save_crop_dir", type=str, default="")
    p.add_argument("--save_crop_every", type=int, default=30)
    args = p.parse_args()

    # ===== 자동 출력 경로 생성 =====
    from datetime import datetime
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_name = f"{video_stem}__{ckpt_stem}__{timestamp}"

    if not args.out:
        args.out = os.path.join(args.out_dir, f"{auto_name}.mp4")
    if not args.csv_out:
        args.csv_out = os.path.join(args.out_dir, f"{auto_name}.csv")
    # ================================

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
    print("YOLO Pose Tracking + Pose ROI + 인식 모델 (CRNN 또는 DigitCNN)")
    print("=" * 80)
    print(f"video       : {args.video}")
    print(f"pose_model  : {args.pose_model}")
    print(f"ckpt        : {args.ckpt}")
    print(f"device      : {device}")
    print(f"out video   : {args.out}")
    print(f"out csv     : {args.csv_out}")
    print("=" * 80)

    pose_model = YOLO(args.pose_model)
    digit_model, predict_fn, img_w, img_h, info = load_model(args.ckpt, device)

    print("\n[인식 모델 정보]")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    number_memory = TrackNumberMemory(
        history_size=args.number_history,
        conf_th=args.number_conf_th,
        min_hits=args.number_min_hits,
        lock_ratio=args.number_lock_ratio,
        unlock_when_changed=args.unlock_when_changed
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
        "confidence", "digit_conf_1", "digit_conf_2", "digit_conf_3", "digit_conf_4",
        "valid_reads", "total_reads",
        "person_x1", "person_y1", "person_x2", "person_y2",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2",
        "crop_path"
    ])

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO Pose tracking (매 프레임)
        results = pose_model.track(
            source=frame, persist=True, verbose=False,
            conf=args.pose_conf, iou=args.iou, imgsz=args.imgsz,
            device=yolo_device, tracker=args.tracker
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            writer.write(frame)
            if args.display:
                cv2.imshow("Result", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        has_id = result.boxes.id is not None
        track_ids = (result.boxes.id.cpu().numpy().astype(int).tolist()
                     if has_id else [-(i+1) for i in range(len(boxes))])

        # keypoint
        if result.keypoints is not None and result.keypoints.xy is not None:
            kpts_xy = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                kpts_conf = result.keypoints.conf.cpu().numpy()
            else:
                kpts_conf = result.keypoints.data[:, :, 2].cpu().numpy()
        else:
            kpts_xy = None
            kpts_conf = None

        # 1단계: 각 사람마다 ROI 계산하고 crop 수집
        valid_dets = []
        crops_to_predict = []

        for det_idx, box in enumerate(boxes):
            track_id = int(track_ids[det_idx])
            px1, py1, px2, py2 = box
            person_box = clip_box(px1, py1, px2, py2, frame_w, frame_h)

            if (person_box[2] - person_box[0]) * (person_box[3] - person_box[1]) < args.min_box_area:
                continue

            if kpts_xy is None or det_idx >= len(kpts_xy):
                continue

            roi_box = compute_upper_back_roi(
                kpts_xy=kpts_xy[det_idx],
                kpts_conf=kpts_conf[det_idx],
                person_box=person_box,
                frame_w=frame_w,
                frame_h=frame_h,
                kp_conf_thr=args.kp_conf_thr,
                roi_pad_x=args.roi_pad_x,
                roi_pad_y=args.roi_pad_y,
            )

            rx1, ry1, rx2, ry2 = roi_box
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            valid_dets.append({
                "track_id": track_id,
                "person_box": person_box,
                "roi_box": roi_box,
                "crop": crop
            })
            if frame_idx % args.frame_stride == 0:
                crops_to_predict.append(crop)

        # 2단계: 배치 추론
        if frame_idx % args.frame_stride == 0 and crops_to_predict:
            batch_imgs = [preprocess(c, img_w, img_h) for c in crops_to_predict]
            x = torch.from_numpy(np.stack(batch_imgs, axis=0)).float()
            preds = predict_fn(digit_model, x, device)
        else:
            preds = []

        # 3단계: 결과 처리 + 그리기
        pred_iter = iter(preds)
        for det in valid_dets:
            track_id = det["track_id"]
            person_box = det["person_box"]
            roi_box = det["roi_box"]
            crop = det["crop"]

            if frame_idx % args.frame_stride == 0:
                current_pred, conf, digit_confs = next(pred_iter)
                display_number, is_locked = number_memory.update(
                    track_id, current_pred, conf
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

            dc = list(digit_confs) + [0.0] * 4
            dc = dc[:4]

            csv_writer.writerow([
                frame_idx, track_id, current_pred, display_number, int(is_locked),
                round(float(conf), 4),
                round(float(dc[0]), 4),
                round(float(dc[1]), 4),
                round(float(dc[2]), 4),
                round(float(dc[3]), 4),
                stats["valid_reads"], stats["total_reads"],
                person_box[0], person_box[1], person_box[2], person_box[3],
                roi_box[0], roi_box[1], roi_box[2], roi_box[3],
                crop_path
            ])

            draw_result(
                frame, person_box, roi_box, track_id,
                display_number, is_locked,
                current_pred, conf, stats["valid_reads"], stats["total_reads"]
            )

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