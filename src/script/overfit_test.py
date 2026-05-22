"""
overfit_test.py
===============

학습/모델 코드가 정상 작동하는지 검증.

원리:
- 100장만 떼서 augmentation 없이 50 epoch 학습
- 모델이 정상이면 train accuracy가 90% 이상으로 올라가야 함 (외우기 가능해야)
- 30 epoch 지나도 train acc가 안 오르면 모델/학습 코드 자체에 문제 있음

실행:
python overfit_test.py \
    --csv /root/.../train.csv,/root/.../synthetic_v2/labels.csv \
    --device cuda
"""

import os
import random
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 학습 코드에서 그대로 import
from multicnn_train_v4 import (
    DigitCNN, build_samples, resize_keep_ratio, set_seed
)


class SimpleDataset(Dataset):
    """augmentation 완전 OFF, 매번 같은 이미지 반환"""
    def __init__(self, samples, img_w, img_h):
        self.samples = samples
        self.img_w = img_w
        self.img_h = img_h

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["path"])
        if img is None:
            img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 235

        img = resize_keep_ratio(img, self.img_w, self.img_h)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        x = torch.from_numpy(img).float()
        y = torch.tensor([int(c) for c in sample["label"]], dtype=torch.long)
        return x, y


def compute_loss(outputs, y):
    loss = 0.0
    for i in range(4):
        loss += F.cross_entropy(outputs[i], y[:, i])
    return loss / 4.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--img_w", type=int, default=192)
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    csv_paths = [p.strip() for p in args.csv.split(",") if p.strip()]
    all_samples = build_samples(csv_paths)
    print(f"전체 샘플: {len(all_samples)}")

    # 100장만 떼기
    random.shuffle(all_samples)
    samples = all_samples[:args.n_samples]
    print(f"실험 샘플: {len(samples)}")

    dataset = SimpleDataset(samples, args.img_w, args.img_h)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = DigitCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    # weight_decay 0 — overfit 실험이므로 정규화 끔

    print("\n=== Overfit 시작 ===")
    print("정상이면 train_exact 가 점점 1.0에 가까워져야 함")
    print("30 epoch에도 0.0 근처면 → 모델/학습 코드에 버그 있음\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        exact = 0
        digit_c = 0
        digit_t = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = compute_loss(outputs, y)
            loss.backward()

            # 그라디언트 norm 확인 (첫 epoch만)
            if epoch == 1 and total == 0:
                grad_norms = []
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        grad_norms.append((name, p.grad.norm().item()))
                print("[grad norms — first batch]")
                for name, norm in grad_norms[:5]:
                    print(f"  {name}: {norm:.6f}")
                print(f"  ... (총 {len(grad_norms)} 파라미터)")
                # 모든 grad가 0이면 backward가 안 됨
                if all(norm == 0 for _, norm in grad_norms):
                    print("❌ 모든 그라디언트가 0! backward가 작동 안 함")
                    return

            optimizer.step()

            preds = torch.stack([out.argmax(dim=1) for out in outputs], dim=1)
            exact += (preds == y).all(dim=1).sum().item()
            digit_c += (preds == y).sum().item()
            digit_t += y.numel()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)

        train_loss = total_loss / max(1, total)
        train_exact = exact / max(1, total)
        train_digit = digit_c / max(1, digit_t)

        marker = ""
        if epoch == 1:
            marker = " (start)"
        elif train_exact > 0.5:
            marker = " ✅ 학습 중!"
        elif epoch == args.epochs and train_exact < 0.1:
            marker = " ❌ 학습 안 됨!"

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | "
              f"exact={train_exact:.4f} | digit={train_digit:.4f}{marker}")

    print("\n=== 결과 ===")
    if train_exact > 0.8:
        print("✅ 모델/학습 코드 정상. 외우기 성공.")
        print("   → 본 학습에서 학습이 안 됐던 건 데이터 양/품질 문제일 가능성")
    elif train_exact > 0.3:
        print("⚠️ 부분적으로 학습. 모델 capacity 부족하거나 LR 조정 필요.")
    else:
        print("❌ 학습 자체가 안 됨. 모델 구조 또는 데이터 파이프라인에 버그.")


if __name__ == "__main__":
    main()