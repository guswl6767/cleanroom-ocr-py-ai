"""
train_digit_cnn.py
==================

4자리 등번호 인식 CNN 학습 코드

CSV 헤더 예:
filename,path,pred_label,confidence,status,roi,raw_text,bbox,all_candidates,source_path

학습 방식:
- path 이미지 로드
- bbox가 있으면 숫자 영역 crop
- pred_label 4자리 숫자를 정답으로 사용
- CNN backbone + 4개 digit head
- best model 저장

실행 예:
python train_digit_cnn.py \
  --csv /root/Storage/cleanroom-ocr-py-ai/Data/origin/ocr_auto_labels/merged.csv \
  --out_dir ./digit_cnn_runs \
  --epochs 50 \
  --batch_size 64 \
  --device cuda
"""

import os
import re
import ast
import json
import random
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Seed
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# CSV / Label Utils
# =========================================================

def read_csv_safely(csv_path):
    try:
        return pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, dtype=str, encoding="cp949")


def clean_label(x):
    """
    pred_label을 4자리 문자열로 정리.
    0273 같은 앞자리 0 유지가 중요함.
    """
    if x is None:
        return None

    s = str(x).strip()

    if s == "" or s.lower() in ["nan", "none", "unknown"]:
        return None

    # 5064.0 같은 값 처리
    if re.fullmatch(r"\d+\.0", s):
        s = s.split(".")[0]

    # 숫자 4자리만 허용
    if not re.fullmatch(r"\d{4}", s):
        return None

    return s


def parse_bbox(bbox_str):
    """
    bbox 컬럼 파싱.
    입력 예:
    [[2038, 2159], [3842, 1948], [3922, 3637], [2118, 3847]]

    반환:
    x1, y1, x2, y2 또는 None
    """
    if bbox_str is None:
        return None

    s = str(bbox_str).strip()

    if s == "" or s.lower() in ["nan", "none"] or s == "[]":
        return None

    try:
        pts = json.loads(s)
    except Exception:
        try:
            pts = ast.literal_eval(s)
        except Exception:
            return None

    if not isinstance(pts, list) or len(pts) < 4:
        return None

    xs = []
    ys = []

    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            xs.append(float(p[0]))
            ys.append(float(p[1]))

    if len(xs) < 2 or len(ys) < 2:
        return None

    x1 = int(np.floor(min(xs)))
    y1 = int(np.floor(min(ys)))
    x2 = int(np.ceil(max(xs)))
    y2 = int(np.ceil(max(ys)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def crop_by_bbox(img, bbox, pad_ratio=0.15):
    """
    bbox가 있으면 숫자 영역 crop.
    bbox가 없으면 원본 이미지 그대로 사용.
    """
    if bbox is None:
        return img

    h, w = img.shape[:2]

    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1, y1, x2, y2 = clip_box(
        x1 - pad_x,
        y1 - pad_y,
        x2 + pad_x,
        y2 + pad_y,
        w,
        h
    )

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return img

    return crop


def resize_keep_ratio(img, target_w, target_h):
    """
    숫자 비율 유지하면서 padding resize.
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


# =========================================================
# Sample Build
# =========================================================

def build_samples(
    csv_paths,
    allowed_status=("auto_ok", "synthetic_ok"),
    min_conf=0.0
):
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    dfs = []

    for csv_path in csv_paths:
        df = read_csv_safely(csv_path)
        df["__csv_path__"] = str(csv_path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    required_cols = ["path", "pred_label", "status"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV에 '{col}' 컬럼이 없습니다.")

    if "bbox" not in df.columns:
        df["bbox"] = "[]"

    if "confidence" not in df.columns:
        df["confidence"] = "0.0"

    samples = []

    for _, row in df.iterrows():
        status = str(row.get("status", "")).strip()

        if allowed_status and status not in allowed_status:
            continue

        label = clean_label(row.get("pred_label", None))

        if label is None:
            continue

        img_path = str(row.get("path", "")).strip()

        if img_path == "" or not os.path.exists(img_path):
            continue

        try:
            conf = float(str(row.get("confidence", "0")).strip())
        except Exception:
            conf = 0.0

        # synthetic_ok는 confidence 1.0인 경우가 많음
        # auto_ok는 min_conf로 필터링 가능
        if conf < min_conf:
            continue

        bbox = parse_bbox(row.get("bbox", "[]"))

        samples.append({
            "path": img_path,
            "label": label,
            "bbox": bbox,
            "status": status,
            "confidence": conf
        })

    if len(samples) == 0:
        raise ValueError("사용 가능한 학습 샘플이 없습니다. CSV, status, pred_label, path를 확인하세요.")

    return samples


def split_samples(samples, val_ratio=0.2, seed=42):
    random.Random(seed).shuffle(samples)

    val_size = int(len(samples) * val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    return train_samples, val_samples


# =========================================================
# Dataset
# =========================================================

class Digit4Dataset(Dataset):
    def __init__(
        self,
        samples,
        img_w=160,
        img_h=64,
        train=True,
        use_bbox=True,
        bbox_pad=0.15
    ):
        self.samples = samples
        self.img_w = img_w
        self.img_h = img_h
        self.train = train
        self.use_bbox = use_bbox
        self.bbox_pad = bbox_pad

    def __len__(self):
        return len(self.samples)

    def augment(self, img):
        """
        숫자 crop에 적용할 가벼운 증강.
        """
        h, w = img.shape[:2]

        # affine: 작은 회전 + 이동 + 스케일
        if random.random() < 0.8:
            angle = random.uniform(-7, 7)
            tx = random.uniform(-0.05, 0.05) * w
            ty = random.uniform(-0.05, 0.05) * h
            scale = random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)

        # 살짝의 perspective (등번호가 옷 굴곡 때문에 휘니까)
        if random.random() < 0.3:
            dx = w * 0.04
            dy = h * 0.04
            src = np.float32([[0,0],[w,0],[w,h],[0,h]])
            dst = np.float32([
                [random.uniform(-dx, dx), random.uniform(-dy, dy)],
                [w + random.uniform(-dx, dx), random.uniform(-dy, dy)],
                [w + random.uniform(-dx, dx), h + random.uniform(-dy, dy)],
                [random.uniform(-dx, dx), h + random.uniform(-dy, dy)],
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)
        # 밝기 / 대비
        if random.random() < 0.7:
            alpha = random.uniform(0.75, 1.25)
            beta = random.randint(-25, 25)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # blur
        if random.random() < 0.25:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # noise
        if random.random() < 0.35:
            noise = np.random.normal(0, random.uniform(3, 10), img.shape)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # JPEG 압축 느낌
        if random.random() < 0.4:
            quality = random.randint(45, 90)
            ok, enc = cv2.imencode(
                ".jpg",
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    img = dec

        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = cv2.imread(sample["path"])

        if img is None:
            img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 235
        else:
            if self.use_bbox:
                img = crop_by_bbox(
                    img,
                    sample["bbox"],
                    pad_ratio=self.bbox_pad
                )

            img = resize_keep_ratio(img, self.img_w, self.img_h)

            if self.train:
                img = self.augment(img)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        x = torch.from_numpy(img).float()

        label = sample["label"]

        y = torch.tensor([
            int(label[0]),
            int(label[1]),
            int(label[2]),
            int(label[3])
        ], dtype=torch.long)

        return x, y


# =========================================================
# Model
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
# Train / Eval
# =========================================================

def compute_loss(outputs, y):
    loss = 0.0

    for i in range(4):
        loss += F.cross_entropy(outputs[i], y[:, i])

    return loss / 4.0


@torch.no_grad()
def evaluate(model, loader, device, desc="Valid"):
    model.eval()

    total_loss = 0.0
    total = 0

    exact_correct = 0
    digit_correct = 0
    digit_total = 0

    pbar = tqdm(loader, desc=desc, ncols=120, leave=False)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        loss = compute_loss(outputs, y)

        preds = torch.stack(
            [out.argmax(dim=1) for out in outputs],
            dim=1
        )

        exact_correct += (preds == y).all(dim=1).sum().item()
        digit_correct += (preds == y).sum().item()
        digit_total += y.numel()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

        avg_loss = total_loss / max(1, total)
        exact_acc = exact_correct / max(1, total)
        digit_acc = digit_correct / max(1, digit_total)

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "exact": f"{exact_acc:.4f}",
            "digit": f"{digit_acc:.4f}"
        })

    avg_loss = total_loss / max(1, total)
    exact_acc = exact_correct / max(1, total)
    digit_acc = digit_correct / max(1, digit_total)

    return avg_loss, exact_acc, digit_acc


def train(args):
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print(f"🔥 device: {device}")
    print("=" * 70)

    csv_paths = [p.strip() for p in args.csv.split(",") if p.strip()]

    allowed_status = tuple(
        s.strip() for s in args.status.split(",") if s.strip()
    )

    samples = build_samples(
        csv_paths=csv_paths,
        allowed_status=allowed_status,
        min_conf=args.min_conf
    )
    bbox_none = sum(1 for s in samples if s["bbox"] is None)
    bbox_ok   = len(samples) - bbox_none
    print(f"bbox 있음: {bbox_ok}, bbox 없음: {bbox_none}")

    # 그리고 실제 데이터셋이 모델에 어떤 이미지를 넣는지 시각화
    import cv2
    ds = Digit4Dataset(samples[:20], img_w=160, img_h=64, train=False, use_bbox=True)
    for i, (x, y) in enumerate(ds):
        img = (x.permute(1,2,0).numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        label = "".join(map(str, y.tolist()))
        cv2.imwrite(f"/tmp/check_{i}_{label}.jpg", img)
        train_samples, val_samples = split_samples(
            samples,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

    print("=" * 70)
    print(f"전체 샘플 수      : {len(samples)}")
    print(f"학습 샘플 수      : {len(train_samples)}")
    print(f"검증 샘플 수      : {len(val_samples)}")
    print(f"사용 status       : {allowed_status}")
    print(f"min_conf          : {args.min_conf}")
    print(f"use_bbox          : {not args.no_bbox}")
    print(f"image size        : {args.img_w} x {args.img_h}")
    print("=" * 70)

    train_dataset = Digit4Dataset(
        samples=train_samples,
        img_w=args.img_w,
        img_h=args.img_h,
        train=True,
        use_bbox=not args.no_bbox,
        bbox_pad=args.bbox_pad
    )

    val_dataset = Digit4Dataset(
        samples=val_samples,
        img_w=args.img_w,
        img_h=args.img_h,
        train=False,
        use_bbox=not args.no_bbox,
        bbox_pad=args.bbox_pad
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    model = DigitCNN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_exact_acc = 0.0

    best_path = os.path.join(args.out_dir, "best_digit_cnn.pt")
    last_path = os.path.join(args.out_dir, "last_digit_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            ncols=120,
            leave=True
        )
        for step, (x, y) in enumerate(pbar, 1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = compute_loss(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

            current_lr = optimizer.param_groups[0]["lr"]

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{total_loss / max(1, total):.4f}",
                "lr": f"{current_lr:.6f}"
            })

        scheduler.step()

        train_loss = total_loss / max(1, total)

        val_loss, exact_acc, digit_acc = evaluate(
            model,
            val_loader,
            device,
            desc=f"Valid {epoch}/{args.epochs}"
        )

        print(
            f"\n[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"exact_acc={exact_acc:.4f} "
            f"digit_acc={digit_acc:.4f}\n"
        )

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "exact_acc": exact_acc,
            "digit_acc": digit_acc,
            "img_w": args.img_w,
            "img_h": args.img_h,
            "status": args.status,
            "use_bbox": not args.no_bbox,
            "bbox_pad": args.bbox_pad
        }

        torch.save(ckpt, last_path)

        if exact_acc > best_exact_acc:
            best_exact_acc = exact_acc
            torch.save(ckpt, best_path)

            print(
                f"✅ best saved: {best_path} "
                f"exact_acc={best_exact_acc:.4f}"
            )

    print("=" * 70)
    print("🏁 학습 완료")
    print(f"best exact acc : {best_exact_acc:.4f}")
    print(f"best model     : {best_path}")
    print(f"last model     : {last_path}")
    print("=" * 70)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV 경로. 여러 개면 쉼표로 구분"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./digit_cnn_runs"
    )

    parser.add_argument("--img_w", type=int, default=160)
    parser.add_argument("--img_h", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # 기본은 자동 라벨 성공 + 합성 데이터만 사용
    parser.add_argument(
        "--status",
        type=str,
        default="auto_ok,synthetic_ok"
    )

    # auto_ok 중 confidence 낮은 데이터 제외하고 싶으면 0.7 추천
    parser.add_argument("--min_conf", type=float, default=0.0)

    # bbox crop 사용 안 하고 전체 이미지로 학습할 때 사용
    parser.add_argument("--no_bbox", action="store_true")

    parser.add_argument("--bbox_pad", type=float, default=0.15)

    parser.add_argument("--print_interval", type=int, default=50)

    args = parser.parse_args()

    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError("--val_ratio는 0~1 사이여야 합니다.")

    train(args)


if __name__ == "__main__":
    main()