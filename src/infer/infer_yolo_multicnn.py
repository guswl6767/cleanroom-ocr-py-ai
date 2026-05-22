import os
import csv
import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn

from collections import defaultdict, deque, Counter

from ultralytics import YOLO
from torchvision import transforms


# =========================================================
# 모델 정의
# =========================================================

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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

        self.head1 = self.make_head()
        self.head2 = self.make_head()
        self.head3 = self.make_head()
        self.head4 = self.make_head()

    def make_head(self):
        return nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):

        feat = self.backbone(x)

        feat = feat.squeeze(2)
        feat = feat.permute(0, 2, 1)

        out1 = self.head1(feat[:, 0, :])
        out2 = self.head2(feat[:, 1, :])
        out3 = self.head3(feat[:, 2, :])
        out4 = self.head4(feat[:, 3, :])

        return [out1, out2, out3, out4]


# =========================================================
# util
# =========================================================

def resize_keep_ratio(img, target_w, target_h):

    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.ones(
        (target_h, target_w, 3),
        dtype=np.uint8
    ) * 235

    xoff = (target_w - new_w) // 2
    yoff = (target_h - new_h) // 2

    canvas[
        yoff:yoff+new_h,
        xoff:xoff+new_w
    ] = resized

    return canvas


# =========================================================
# argparse
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument("--video", type=str, required=True)

parser.add_argument("--ckpt", type=str, required=True)

parser.add_argument(
    "--pose_model",
    type=str,
    default="yolo26m-pose.pt"
)

parser.add_argument(
    "--number_detector",
    type=str,
    default="/root/Storage/cleanroom-ocr-py-ai/Models/yolo_detector/back_num/yolov8n_back_number/weights/best.pt"
)

parser.add_argument("--out", type=str, required=True)

parser.add_argument("--csv_out", type=str, required=True)

parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--person_conf", type=float, default=0.5)

parser.add_argument("--number_conf_th", type=float, default=0.65)

parser.add_argument("--frame_step", type=int, default=3)

# =========================================================
# temporal voting args
# =========================================================

parser.add_argument("--number_history", type=int, default=20)

parser.add_argument("--number_min_hits", type=int, default=5)

parser.add_argument("--number_lock_ratio", type=float, default=0.55)

args = parser.parse_args()

# =========================================================
# device
# =========================================================

DEVICE = args.device if torch.cuda.is_available() else "cpu"

# =========================================================
# 모델 로드
# =========================================================

print("loading models...")

pose_model = YOLO(args.pose_model)

number_detector = YOLO(args.number_detector)

digit_model = DigitCNN()

ckpt = torch.load(
    args.ckpt,
    map_location=DEVICE
)

digit_model.load_state_dict(ckpt["model"])

digit_model.to(DEVICE)
digit_model.eval()

print("models loaded")

# =========================================================
# transform
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
])

# =========================================================
# video
# =========================================================

cap = cv2.VideoCapture(args.video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    args.out,
    fourcc,
    fps,
    (width, height)
)

# =========================================================
# csv
# =========================================================

csv_file = open(
    args.csv_out,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "person_idx",
    "pred_number",
])

# =========================================================
# temporal history
# =========================================================

number_histories = defaultdict(
    lambda: deque(maxlen=args.number_history)
)

# =========================================================
# inference
# =========================================================

frame_idx = 0
global_person_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % args.frame_step != 0:
        frame_idx += 1
        continue

    vis_frame = frame.copy()

    # =====================================================
    # person detect
    # =====================================================

    pose_results = pose_model.predict(
        frame,
        conf=args.person_conf,
        verbose=False,
        device=DEVICE
    )

    for result in pose_results:

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # =================================================
            # number detector
            # =================================================

            det_results = number_detector.predict(
                person_crop,
                conf=args.number_conf_th,
                verbose=False,
                device=DEVICE
            )

            pred_text = "none"
            stable_text = "none"

            for det in det_results:

                if det.boxes is None:
                    continue

                det_boxes = det.boxes.xyxy.cpu().numpy()
                det_confs = det.boxes.conf.cpu().numpy()

                if len(det_boxes) == 0:
                    continue

                best_idx = np.argmax(det_confs)

                bx1, by1, bx2, by2 = map(
                    int,
                    det_boxes[best_idx]
                )

                bx1 = max(0, bx1)
                by1 = max(0, by1)

                bx2 = min(person_crop.shape[1]-1, bx2)
                by2 = min(person_crop.shape[0]-1, by2)

                if bx2 <= bx1 or by2 <= by1:
                    continue

                number_crop = person_crop[
                    by1:by2,
                    bx1:bx2
                ]

                if number_crop.size == 0:
                    continue

                # =============================================
                # digit cnn
                # =============================================

                inp = resize_keep_ratio(
                    number_crop,
                    128,
                    256
                )

                inp = cv2.cvtColor(
                    inp,
                    cv2.COLOR_BGR2RGB
                )

                inp = transform(inp)

                inp = inp.unsqueeze(0).to(DEVICE)

                with torch.no_grad():

                    outputs = digit_model(inp)

                    preds = []

                    for out in outputs:
                        p = out.argmax(dim=1).item()
                        preds.append(str(p))

                    pred_text = "".join(preds)

                # =============================================
                # temporal voting
                # =============================================

                number_histories[global_person_idx].append(
                    pred_text
                )

                hist = list(
                    number_histories[global_person_idx]
                )

                counter = Counter(hist)

                stable_text = pred_text

                if len(hist) >= args.number_min_hits:

                    top_number, top_count = (
                        counter.most_common(1)[0]
                    )

                    ratio = top_count / len(hist)

                    if ratio >= args.number_lock_ratio:
                        stable_text = top_number

                # =============================================
                # visualization
                # =============================================

                abs_x1 = x1 + bx1
                abs_y1 = y1 + by1

                abs_x2 = x1 + bx2
                abs_y2 = y1 + by2

                cv2.rectangle(
                    vis_frame,
                    (abs_x1, abs_y1),
                    (abs_x2, abs_y2),
                    (0,255,0),
                    3
                )

                cv2.putText(
                    vis_frame,
                    stable_text,
                    (abs_x1, max(30, abs_y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,255,0),
                    3
                )

            csv_writer.writerow([
                frame_idx,
                global_person_idx,
                stable_text
            ])

            global_person_idx += 1

    writer.write(vis_frame)

    frame_idx += 1

cap.release()
writer.release()
csv_file.close()

print("DONE")