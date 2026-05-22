"""
train_digit_cnn_v5.py
=====================

v4 결과: 100장 overfit에서 exact 0.6~0.65 정체. 학습은 되나 4자리 완전 매칭은 못 함.
원인: flatten 후 fc를 거치면서 4개 head가 모두 같은 글로벌 피처를 받음.
      → 모델이 "어느 자리 어느 위치"를 자력으로 매핑해야 함.

v5 해결: 가로 8칸을 4자리에 명시적으로 매핑.
- AdaptiveAvgPool((1, 4)) → 가로를 정확히 4칸으로
- 각 head는 자기 자리 슬라이스만 받음 (256차원)
- 자리별 head는 독립 weight (shared 아님 — 학습 안정성 위해)
- 추가로 잔차로 글로벌 정보도 합쳐주기 (위치가 흔들려도 강건하게)

실행:
python train_digit_cnn_v5.py \
    --csv /path/train.csv,/path/syn/labels.csv \
    --out_dir ./digit_cnn_runs_v5 \
    --epochs 60 \
    --batch_size 64 \
    --device cuda
"""

import os
import re
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def build_samples(csv_paths, allowed_status=("auto_ok", "synthetic_ok"), min_conf=0.0):
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    dfs = []
    for csv_path in csv_paths:
        df = read_csv_safely(csv_path)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    for col in ["path", "pred_label", "status"]:
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
            "path": img_path, "label": label,
            "status": status, "confidence": conf
        })
    if len(samples) == 0:
        raise ValueError("학습 샘플 없음")
    return samples


def split_samples_by_label(samples, val_ratio=0.2, seed=42):
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)
    labels = sorted(by_label.keys())
    random.Random(seed).shuffle(labels)
    val_count = max(1, int(len(labels) * val_ratio))
    val_labels = set(labels[:val_count])
    train_samples = [s for s in samples if s["label"] not in val_labels]
    val_samples = [s for s in samples if s["label"] in val_labels]
    print(f"[split] train 라벨 {len(labels)-val_count} / val 라벨 {val_count}")
    print(f"[split] train {len(train_samples)} / val {len(val_samples)}")
    return train_samples, val_samples


class Digit4Dataset(Dataset):
    def __init__(self, samples, img_w=128, img_h=256, train=True):
        self.samples = samples
        self.img_w = img_w
        self.img_h = img_h
        self.train = train

    def __len__(self):
        return len(self.samples)

    def augment(self, img):
        h, w = img.shape[:2]
        if random.random() < 0.5:
            angle = random.uniform(-4, 4)
            tx = random.uniform(-0.05, 0.05) * w
            ty = random.uniform(-0.04, 0.04) * h
            scale = random.uniform(0.92, 1.08)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.randint(-15, 15)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if random.random() < 0.2:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(2, 6), img.shape)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            q = random.randint(60, 90)
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
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
            img = resize_keep_ratio(img, self.img_w, self.img_h)
            if self.train:
                img = self.augment(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        x = torch.from_numpy(img).float()
        y = torch.tensor([int(c) for c in sample["label"]], dtype=torch.long)
        return x, y


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
    """
    v5: 가로축을 4자리에 명시적으로 매핑.

    구조:
      입력 (B, 3, 256, 128) [H=256, W=128]
      backbone → (B, 256, H', W')
      AdaptiveAvgPool((1, 4)) → (B, 256, 1, 4)
      squeeze + permute → (B, 4, 256)

      각 자리별로 독립 head가 자기 자리 256차원만 받음:
        head_i(feat[:, i, :]) → (B, 10)

    핵심: head4가 가장 오른쪽 자리의 피처(왼쪽 무관)만 받음 → 위치 강제.
    """
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            ConvBlock(3, 32, pool=True),     # 256x128 -> 128x64
            ConvBlock(32, 64, pool=True),    # 128x64  -> 64x32
            ConvBlock(64, 128, pool=True),   # 64x32   -> 32x16
            ConvBlock(128, 256, pool=True),  # 32x16   -> 16x8
            # 세로는 1로, 가로는 정확히 4칸으로
            nn.AdaptiveAvgPool2d((1, 4)),
        )

        # 자리별 독립 헤드. 각 헤드는 256차원만 받음.
        # 자기 자리 피처에 집중하게 됨.
        self.head1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
        self.head2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
        self.head3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
        self.head4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        feat = self.backbone(x)            # (B, 256, 1, 4)
        feat = feat.squeeze(2)             # (B, 256, 4)
        feat = feat.permute(0, 2, 1)       # (B, 4, 256)

        # 자리별로 각각의 head에 분리해서 통과
        out1 = self.head1(feat[:, 0, :])
        out2 = self.head2(feat[:, 1, :])
        out3 = self.head3(feat[:, 2, :])
        out4 = self.head4(feat[:, 3, :])

        return [out1, out2, out3, out4]


def compute_loss(outputs, y):
    loss = 0.0
    for i in range(4):
        loss += F.cross_entropy(outputs[i], y[:, i])
    return loss / 4.0


@torch.no_grad()
def evaluate(model, loader, device, desc="Valid"):
    model.eval()
    total_loss, total = 0.0, 0
    exact_correct, digit_correct, digit_total = 0, 0, 0
    pred_distribution = Counter()

    pbar = tqdm(loader, desc=desc, ncols=120, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
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

    return (total_loss / max(1, total),
            exact_correct / max(1, total),
            digit_correct / max(1, digit_total),
            pred_distribution)


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"🔥 device: {device}")

    csv_paths = [p.strip() for p in args.csv.split(",") if p.strip()]
    allowed_status = tuple(s.strip() for s in args.status.split(",") if s.strip())

    samples = build_samples(csv_paths, allowed_status=allowed_status, min_conf=args.min_conf)
    train_samples, val_samples = split_samples_by_label(samples, args.val_ratio, args.seed)

    print(f"전체 {len(samples)} / 학습 {len(train_samples)} / 검증 {len(val_samples)}")
    print(f"image size: {args.img_w} x {args.img_h}")

    train_ds = Digit4Dataset(train_samples, args.img_w, args.img_h, train=True)
    val_ds = Digit4Dataset(val_samples, args.img_w, args.img_h, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = DigitCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_exact = 0.0
    best_path = os.path.join(args.out_dir, "best_digit_cnn.pt")
    last_path = os.path.join(args.out_dir, "last_digit_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        tr_exact, tr_dc, tr_dt = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = compute_loss(outputs, y)
            loss.backward()
            optimizer.step()

            preds = torch.stack([out.argmax(dim=1) for out in outputs], dim=1)
            tr_exact += (preds == y).all(dim=1).sum().item()
            tr_dc += (preds == y).sum().item()
            tr_dt += y.numel()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ex": f"{tr_exact/max(1,total):.3f}",
                "dg": f"{tr_dc/max(1,tr_dt):.3f}",
            })

        scheduler.step()
        train_loss = total_loss / max(1, total)
        train_exact_acc = tr_exact / max(1, total)
        train_digit_acc = tr_dc / max(1, tr_dt)

        val_loss, val_exact, val_digit, pred_dist = evaluate(
            model, val_loader, device, desc=f"Valid {epoch}/{args.epochs}"
        )

        n_unique = len(pred_dist)
        top1_share = pred_dist.most_common(1)[0][1] / max(1, len(val_ds)) if pred_dist else 0

        print(f"\n[Epoch {epoch}/{args.epochs}] "
              f"train: loss={train_loss:.4f} ex={train_exact_acc:.4f} dg={train_digit_acc:.4f} | "
              f"val: loss={val_loss:.4f} ex={val_exact:.4f} dg={val_digit:.4f} | "
              f"uniq={n_unique} top1={top1_share:.2%}")

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "exact_acc": val_exact,
            "digit_acc": val_digit,
            "img_w": args.img_w,
            "img_h": args.img_h,
            "arch": "DigitCNN_v5_per_position",
        }
        torch.save(ckpt, last_path)
        if val_exact > best_exact:
            best_exact = val_exact
            torch.save(ckpt, best_path)
            print(f"✅ best: val_exact={best_exact:.4f}")

    print(f"\n🏁 완료. best val exact: {best_exact:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./digit_cnn_runs_v5")
    p.add_argument("--img_w", type=int, default=128)
    p.add_argument("--img_h", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--status", type=str, default="auto_ok,synthetic_ok")
    p.add_argument("--min_conf", type=float, default=0.0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()