"""
train_digit_cnn.py
==================

사람 crop 전체를 입력으로 받는 4자리 등번호 인식 CNN 학습 코드.

핵심 변경점 (이전 버전 대비):
1. bbox 사용 X — 사람 crop 전체를 입력
2. 입력 크기 192x384 (세로로 긴 사람 crop 비율)
3. backbone에 ConvBlock 한 층 추가 — receptive field 확보
4. augmentation 강화 — 위치/스케일 변동 크게
5. label-level split — 같은 4자리 조합이 train/val에 겹치지 않게
6. cutout 추가 — 외우기 방지

실행 예:
python train_digit_cnn.py \
  --csv /root/Storage/.../merged.csv \
  --out_dir ./digit_cnn_runs \
  --epochs 80 \
  --batch_size 32 \
  --img_w 192 \
  --img_h 384 \
  --device cuda
"""

import os
import re
import json
import random
import argparse
from collections import defaultdict, Counter

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
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "unknown"]:
        return None
    if re.fullmatch(r"\d+\.0", s):
        s = s.split(".")[0]
    if not re.fullmatch(r"\d{4}", s):
        return None
    return s


def resize_keep_ratio(img, target_w, target_h):
    """비율 유지하면서 padding resize. 추론 코드와 100% 동일하게 유지할 것."""
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

def build_samples(csv_paths, allowed_status=("auto_ok", "synthetic_ok"), min_conf=0.0):
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

        if conf < min_conf:
            continue

        samples.append({
            "path": img_path,
            "label": label,
            "status": status,
            "confidence": conf
        })

    if len(samples) == 0:
        raise ValueError("사용 가능한 학습 샘플이 없습니다.")

    return samples


def split_samples_by_label(samples, val_ratio=0.2, seed=42):
    """
    라벨(4자리 조합) 단위로 split.
    같은 4자리 조합이 train/val에 겹치지 않게 보장.
    val acc가 진짜 일반화 성능을 반영하게 됨.
    """
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)

    labels = sorted(by_label.keys())
    random.Random(seed).shuffle(labels)

    val_count = max(1, int(len(labels) * val_ratio))
    val_labels = set(labels[:val_count])

    train_samples = [s for s in samples if s["label"] not in val_labels]
    val_samples = [s for s in samples if s["label"] in val_labels]

    print(f"[split] 전체 고유 라벨: {len(labels)}")
    print(f"[split] train 라벨: {len(labels) - val_count}, val 라벨: {val_count}")
    print(f"[split] train 샘플: {len(train_samples)}, val 샘플: {len(val_samples)}")
    print(f"[split] train/val 라벨 겹침: 0 (label-level split)")

    return train_samples, val_samples


# =========================================================
# Dataset
# =========================================================

class Digit4Dataset(Dataset):
    def __init__(self, samples, img_w=192, img_h=384, train=True):
        self.samples = samples
        self.img_w = img_w
        self.img_h = img_h
        self.train = train

    def __len__(self):
        return len(self.samples)

    def augment(self, img):
        h, w = img.shape[:2]

        # 1. Affine — 평행이동을 크게, 회전과 스케일도 흔들기
        # 핵심: 사람과 등번호가 화면의 다양한 위치에 오게 만듦
        if random.random() < 0.9:
            angle = random.uniform(-8, 8)
            tx = random.uniform(-0.15, 0.15) * w
            ty = random.uniform(-0.10, 0.10) * h
            scale = random.uniform(0.75, 1.15)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 2. 약한 perspective — 옷 굴곡 시뮬레이션
        if random.random() < 0.25:
            dx, dy = w * 0.05, h * 0.05
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [random.uniform(-dx, dx), random.uniform(-dy, dy)],
                [w + random.uniform(-dx, dx), random.uniform(-dy, dy)],
                [w + random.uniform(-dx, dx), h + random.uniform(-dy, dy)],
                [random.uniform(-dx, dx), h + random.uniform(-dy, dy)],
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 3. 밝기 / 대비
        if random.random() < 0.7:
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-30, 30)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 4. 색조 변화 (BR shift — 조명 색온도 시뮬)
        if random.random() < 0.3:
            shift = random.randint(-12, 12)
            img_int = img.astype(np.int16)
            img_int[:, :, 0] = np.clip(img_int[:, :, 0] + shift, 0, 255)
            img_int[:, :, 2] = np.clip(img_int[:, :, 2] - shift, 0, 255)
            img = img_int.astype(np.uint8)

        # 5. blur (motion / gaussian)
        if random.random() < 0.3:
            if random.random() < 0.5:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)
            else:
                k = random.choice([5, 7])
                kernel = np.zeros((k, k), dtype=np.float32)
                kernel[k // 2, :] = np.ones(k, dtype=np.float32) / k
                img = cv2.filter2D(img, -1, kernel)

        # 6. noise
        if random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(3, 12), img.shape)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 7. JPEG 압축 아티팩트
        if random.random() < 0.5:
            quality = random.randint(35, 90)
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    img = dec

        # 8. 다운샘플링 후 업샘플링 (저해상도 CCTV 느낌)
        if random.random() < 0.4:
            scale = random.uniform(0.45, 0.85)
            sw = max(32, int(w * scale))
            sh = max(64, int(h * scale))
            small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 9. Cutout — 외우기 방지에 효과적
        # 이미지 일부를 가려도 글자를 읽을 수 있어야 한다는 압력
        if random.random() < 0.4:
            n_cuts = random.randint(1, 3)
            for _ in range(n_cuts):
                ch = random.randint(h // 12, h // 5)
                cw = random.randint(w // 12, w // 4)
                cy = random.randint(0, h - ch)
                cx = random.randint(0, w - cw)
                color = random.randint(0, 255)
                img[cy:cy + ch, cx:cx + cw] = color

        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["path"])

        if img is None:
            img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 235
        else:
            # bbox 사용 안 함 — 사람 crop 통째로
            img = resize_keep_ratio(img, self.img_w, self.img_h)
            if self.train:
                img = self.augment(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        x = torch.from_numpy(img).float()

        label = sample["label"]
        y = torch.tensor([int(c) for c in label], dtype=torch.long)
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
    사람 crop 입력용 4자리 등번호 CNN
    - 입력: 192x384 (가로x세로) 사람 crop
    - 5번 풀링으로 receptive field를 충분히 확보
    - (1,4) pooling + shared head로 위치별 분류 + 조합 암기 방지
    """
    def __init__(self, num_positions=4):
        super().__init__()
        self.num_positions = num_positions

        # 입력: (B, 3, 384, 192)
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, pool=True),     # 384x192 -> 192x96
            ConvBlock(32, 64, pool=True),    # 192x96  -> 96x48
            ConvBlock(64, 128, pool=True),   # 96x48   -> 48x24
            ConvBlock(128, 256, pool=True),  # 48x24   -> 24x12
            ConvBlock(256, 256, pool=True),  # 24x12   -> 12x6
            # 세로는 1로, 가로는 4(자리수)로 압축
            nn.AdaptiveAvgPool2d((1, num_positions)),
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.shared_head = nn.Linear(256, 10)

    def forward(self, x):
        feat = self.backbone(x)              # (B, 256, 1, 4)
        feat = feat.squeeze(2)               # (B, 256, 4)
        feat = feat.permute(0, 2, 1)         # (B, 4, 256)

        B, P, C = feat.shape
        feat = feat.reshape(B * P, C)
        feat = self.shared_fc(feat)
        logits = self.shared_head(feat)      # (B*4, 10)
        logits = logits.reshape(B, P, 10)    # (B, 4, 10)

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

    pred_distribution = Counter()

    pbar = tqdm(loader, desc=desc, ncols=120, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        loss = compute_loss(outputs, y)

        preds = torch.stack([out.argmax(dim=1) for out in outputs], dim=1)

        for p in preds.cpu().numpy():
            pred_distribution["".join(map(str, p))] += 1

        exact_correct += (preds == y).all(dim=1).sum().item()
        digit_correct += (preds == y).sum().item()
        digit_total += y.numel()
        total_loss += loss.item() * y.size(0)
        total += y.size(0)

        pbar.set_postfix({
            "loss": f"{total_loss / max(1, total):.4f}",
            "exact": f"{exact_correct / max(1, total):.4f}",
            "digit": f"{digit_correct / max(1, digit_total):.4f}"
        })

    avg_loss = total_loss / max(1, total)
    exact_acc = exact_correct / max(1, total)
    digit_acc = digit_correct / max(1, digit_total)

    return avg_loss, exact_acc, digit_acc, pred_distribution


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"🔥 device: {device}")
    print("=" * 70)

    csv_paths = [p.strip() for p in args.csv.split(",") if p.strip()]
    allowed_status = tuple(s.strip() for s in args.status.split(",") if s.strip())
    
    samples = build_samples(
        csv_paths=csv_paths,
        allowed_status=allowed_status,
        min_conf=args.min_conf
    )

    train_samples, val_samples = split_samples_by_label(
        samples, val_ratio=args.val_ratio, seed=args.seed
    )

    print("=" * 70)
    print(f"전체 샘플 수      : {len(samples)}")
    print(f"학습 샘플 수      : {len(train_samples)}")
    print(f"검증 샘플 수      : {len(val_samples)}")
    print(f"사용 status       : {allowed_status}")
    print(f"image size        : {args.img_w} x {args.img_h}  (W x H)")
    print(f"split strategy    : label-level (조합 외우기 검증 가능)")
    print("=" * 70)

    train_dataset = Digit4Dataset(train_samples, args.img_w, args.img_h, train=True)
    val_dataset = Digit4Dataset(val_samples, args.img_w, args.img_h, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    model = DigitCNN().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수  : {n_params:,}")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_exact_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_digit_cnn.pt")
    last_path = os.path.join(args.out_dir, "last_digit_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120, leave=True)

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

        val_loss, exact_acc, digit_acc, pred_dist = evaluate(
            model, val_loader, device, desc=f"Valid {epoch}/{args.epochs}"
        )

        # 예측 다양성 — 모델이 몇 개의 서로 다른 답을 뱉는지
        n_val_samples = len(val_dataset)
        n_unique_preds = len(pred_dist)
        top1_share = pred_dist.most_common(1)[0][1] / max(1, n_val_samples) if pred_dist else 0

        print(
            f"\n[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"exact={exact_acc:.4f} "
            f"digit={digit_acc:.4f} "
            f"unique_preds={n_unique_preds}/{n_val_samples} "
            f"top1_share={top1_share:.2%}"
        )

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "exact_acc": exact_acc,
            "digit_acc": digit_acc,
            "img_w": args.img_w,
            "img_h": args.img_h,
            "use_bbox": False,
            "arch": "DigitCNN_v2_person_crop"
        }
        torch.save(ckpt, last_path)

        if exact_acc > best_exact_acc:
            best_exact_acc = exact_acc
            torch.save(ckpt, best_path)
            print(f"✅ best saved: {best_path} exact_acc={best_exact_acc:.4f}")

    print("=" * 70)
    print("🏁 학습 완료")
    print(f"best exact acc : {best_exact_acc:.4f}")
    print(f"best model     : {best_path}")
    print("=" * 70)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="CSV 경로. 여러 개면 쉼표로")
    parser.add_argument("--out_dir", type=str, default="./digit_cnn_runs")
    parser.add_argument("--img_w", type=int, default=192)
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--status", type=str, default="auto_ok,synthetic_ok")
    parser.add_argument("--min_conf", type=float, default=0.0)
    args = parser.parse_args()

    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError("--val_ratio는 0~1 사이여야 합니다.")

    train(args)


if __name__ == "__main__":
    main()