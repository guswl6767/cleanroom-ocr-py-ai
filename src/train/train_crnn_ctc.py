"""
train_crnn_ctc.py
=================

CRNN + CTC 기반 4자리 등번호 OCR.
- 위치 변동(사람 자세, 기울기 등)에 강건
- 자릿수가 항상 4자리 고정인 경우 사용

핵심:
- CNN backbone → (B, C, 1, W') 시퀀스 피처
- BiLSTM → 시퀀스 디코딩
- CTCLoss → 정렬 자동 학습

실행:
python train_crnn_ctc.py \
    --csv /path/train.csv,/path/syn/labels.csv \
    --out_dir ./crnn_runs \
    --epochs 60 \
    --batch_size 32 \
    --img_w 128 \
    --img_h 256 \
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


# CTC 클래스: 0~9 + blank
NUM_CLASSES = 11
BLANK_IDX = 10
FIXED_LENGTH = 4   # 4자리 등번호 고정


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
    if not samples:
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


# =========================================================
# Dataset
# =========================================================

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
            angle = random.uniform(-5, 5)
            tx = random.uniform(-0.08, 0.08) * w
            ty = random.uniform(-0.06, 0.06) * h
            scale = random.uniform(0.90, 1.10)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
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
        s = self.samples[idx]
        img = cv2.imread(s["path"])
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
        y = torch.tensor([int(c) for c in s["label"]], dtype=torch.long)
        return x, y


# =========================================================
# CRNN Model
# =========================================================

class CRNN(nn.Module):
    """
    CNN backbone + BiLSTM + CTC head.

    입력: (B, 3, H=256, W=128)
    목표: (B, C, 1, W') 형태로 만들어서 가로축을 시퀀스로 사용.

    pooling 전략:
    - 세로축: 점진적으로 1까지 줄임
    - 가로축: 적당히 유지 (W'=8 정도)

    각 timestep마다 0~9 + blank 분류 → CTC로 디코딩
    """
    def __init__(self, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()

        # 입력 H x W 표기: (C, H, W)
        # 256x128 → 점진적으로 줄임
        self.cnn = nn.Sequential(
            # 256x128 -> 128x64
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 128x64 -> 64x32
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 64x32 -> 32x16 (둘 다 절반)
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 32x16 -> 16x16 (세로만 절반)
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # 16x16 -> 8x16
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # 8x16 -> 4x16
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # 4x16 -> 2x16
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # 2x16 -> 1x15 (세로만 1로 압축)
            nn.Conv2d(512, 512, kernel_size=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                   # (B, 512, 1, W')
        b, c, h, w = feat.shape
        assert h == 1, f"세로가 1이어야 하는데 {h}임. 입력 크기 조정 필요"

        feat = feat.squeeze(2)               # (B, 512, W')
        feat = feat.permute(2, 0, 1)         # (W', B, 512)  ← LSTM 입력 포맷

        rnn_out, _ = self.rnn(feat)          # (W', B, 2*hidden)
        logits = self.fc(rnn_out)            # (W', B, num_classes)
        return logits


# =========================================================
# CTC Loss / Decode
# =========================================================

def compute_ctc_loss(logits, y, ctc_loss_fn):
    """
    logits: (T, B, C)
    y: (B, 4)
    """
    log_probs = logits.log_softmax(dim=2)
    T, B, _ = log_probs.shape
    input_lengths = torch.full((B,), T, dtype=torch.long, device=y.device)
    target_lengths = torch.full((B,), FIXED_LENGTH, dtype=torch.long, device=y.device)
    targets = y.reshape(-1)
    return ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)


@torch.no_grad()
def greedy_decode(logits):
    """
    logits: (T, B, C)
    반환: list[str], 각 샘플의 디코딩된 숫자 문자열 (가변 길이 가능)
    """
    preds = logits.argmax(dim=2)          # (T, B)
    preds = preds.permute(1, 0)           # (B, T)

    results = []
    for seq in preds.cpu().numpy():
        out = []
        prev = -1
        for p in seq:
            if p != prev and p != BLANK_IDX:
                out.append(str(int(p)))
            prev = p
        results.append("".join(out))
    return results


# =========================================================
# Train / Eval
# =========================================================

@torch.no_grad()
def evaluate(model, loader, device, ctc_loss_fn, desc="Valid"):
    model.eval()
    total_loss, total = 0.0, 0
    exact, digit_c, digit_t = 0, 0, 0
    pred_dist = Counter()
    sample_preds = []

    pbar = tqdm(loader, desc=desc, ncols=120, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = compute_ctc_loss(logits, y, ctc_loss_fn)

        pred_strs = greedy_decode(logits)
        true_strs = ["".join(map(str, t.tolist())) for t in y]

        for ps, ts in zip(pred_strs, true_strs):
            if ps == ts:
                exact += 1
            # digit_acc: 길이가 4일 때만 자릿수별 비교
            if len(ps) == FIXED_LENGTH:
                for pc, tc in zip(ps, ts):
                    if pc == tc:
                        digit_c += 1
            digit_t += FIXED_LENGTH
            pred_dist[ps] += 1

        # 샘플 몇 개 저장
        if len(sample_preds) < 5:
            for ps, ts in zip(pred_strs[:3], true_strs[:3]):
                if len(sample_preds) < 5:
                    sample_preds.append((ts, ps))

        total += y.size(0)
        total_loss += loss.item() * y.size(0)

    return (total_loss / max(1, total),
            exact / max(1, total),
            digit_c / max(1, digit_t),
            pred_dist,
            sample_preds)


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"🔥 device: {device}")

    csv_paths = [c.strip() for c in args.csv.split(",") if c.strip()]
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

    model = CRNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {n_params:,}")

    # 출력 시퀀스 길이 확인 (디버그)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.img_h, args.img_w).to(device)
        out = model(dummy)
        print(f"CTC 시퀀스 길이: {out.shape[0]} (FIXED_LENGTH={FIXED_LENGTH}, 더 커야 정상)")

    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_exact = 0.0
    best_path = os.path.join(args.out_dir, "best_crnn.pt")
    last_path = os.path.join(args.out_dir, "last_crnn.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        tr_exact, tr_dc, tr_dt = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = compute_ctc_loss(logits, y, ctc_loss_fn)
            loss.backward()
            # CTC는 종종 그라디언트가 폭주하므로 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            pred_strs = greedy_decode(logits)
            true_strs = ["".join(map(str, t.tolist())) for t in y]
            for ps, ts in zip(pred_strs, true_strs):
                if ps == ts:
                    tr_exact += 1
                if len(ps) == FIXED_LENGTH:
                    for pc, tc in zip(ps, ts):
                        if pc == tc:
                            tr_dc += 1
                tr_dt += FIXED_LENGTH

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

        val_loss, val_exact, val_digit, pred_dist, sample_preds = evaluate(
            model, val_loader, device, ctc_loss_fn, desc=f"Valid {epoch}/{args.epochs}"
        )

        n_unique = len(pred_dist)
        top1_share = pred_dist.most_common(1)[0][1] / max(1, len(val_ds)) if pred_dist else 0

        print(f"\n[Epoch {epoch}/{args.epochs}] "
              f"train: loss={train_loss:.4f} ex={train_exact_acc:.4f} dg={train_digit_acc:.4f} | "
              f"val: loss={val_loss:.4f} ex={val_exact:.4f} dg={val_digit:.4f} | "
              f"uniq={n_unique} top1={top1_share:.2%}")
        print(f"  val 샘플: {sample_preds[:3]}")

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "exact_acc": val_exact,
            "digit_acc": val_digit,
            "img_w": args.img_w,
            "img_h": args.img_h,
            "arch": "CRNN_CTC",
            "num_classes": NUM_CLASSES,
            "blank_idx": BLANK_IDX,
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
    p.add_argument("--out_dir", type=str, default="./crnn_runs")
    p.add_argument("--img_w", type=int, default=128)
    p.add_argument("--img_h", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)   # CTC는 lr 작게
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