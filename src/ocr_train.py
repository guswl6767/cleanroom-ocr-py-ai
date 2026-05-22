"""
4자리 작업복 숫자 인식 모델 (SVHN-style multi-head CNN)
- 입력: 상체 crop 이미지 (64x128 권장)
- 출력: 4자리 숫자 (예: "0427")

사용법:
    1) 합성 데이터로 사전학습:  python digit4_model.py --mode synth_train
    2) 실제 데이터 fine-tune:    python digit4_model.py --mode finetune --data ./real_data
    3) 추론:                     python digit4_model.py --mode infer --image test.jpg
    4) YOLO 파이프라인 연동:     아래 infer_on_crop() 함수 사용

설치:
    pip install torch torchvision pillow numpy
"""

import argparse
import os
import random
import string
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


# ===== 설정 =====
IMG_H, IMG_W = 64, 128       # 상체 crop 표준 크기
NUM_DIGITS = 4               # 자릿수 고정
NUM_CLASSES = 11             # 0~9 + blank(10)
BLANK = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ===== 모델 =====
class Digit4Net(nn.Module):
    """경량 CNN backbone + 4개의 분류 head."""

    def __init__(self, num_classes=NUM_CLASSES, num_digits=NUM_DIGITS):
        super().__init__()
        self.num_digits = num_digits

        # Backbone: 작고 빠르게. 필요시 MobileNetV3로 교체 가능.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x64

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x32

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x16

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),  # 256 x 4 x 8
        )

        self.flatten_dim = 256 * 4 * 8
        self.shared = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 자리마다 독립적인 분류 head
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_digits)
        ])

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.shared(x)
        # 각 head의 logits를 stack: (B, num_digits, num_classes)
        logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits


# ===== 합성 데이터 생성기 =====
class SyntheticDigitDataset(Dataset):
    """폰트로 4자리 숫자를 렌더링 + augmentation으로 작업복 느낌."""

    def __init__(self, length=20000, fonts=None, train=True):
        self.length = length
        self.train = train
        # 시스템 폰트 자동 탐색. 없으면 PIL 기본 폰트 사용.
        self.fonts = fonts or self._find_fonts()

        if train:
            self.tf = transforms.Compose([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                transforms.RandomAffine(degrees=8, translate=(0.05, 0.05),
                                        scale=(0.85, 1.1), shear=5),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
                transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])

    @staticmethod
    def _find_fonts():
        candidates = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            'C:/Windows/Fonts/arialbd.ttf',
        ]
        found = [f for f in candidates if os.path.exists(f)]
        return found or [None]  # None이면 PIL 기본 폰트

    def __len__(self):
        return self.length

    def _render(self, digits):
        """4자리 숫자를 작업복 패치 느낌으로 렌더링."""
        # 천 색상 흉내 (어두운 톤 위주)
        bg_color = tuple(random.randint(30, 200) for _ in range(3))
        fg_color = tuple(255 - c + random.randint(-30, 30) for c in bg_color)
        fg_color = tuple(max(0, min(255, c)) for c in fg_color)

        img = Image.new('RGB', (IMG_W, IMG_H), bg_color)
        draw = ImageDraw.Draw(img)

        font_path = random.choice(self.fonts)
        font_size = random.randint(36, 52)
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path \
                else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        text = ''.join(str(d) if d != BLANK else '' for d in digits)
        # 가운데 정렬
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (IMG_W - tw) // 2 + random.randint(-8, 8)
        y = (IMG_H - th) // 2 + random.randint(-5, 5) - bbox[1]
        draw.text((x, y), text, fill=fg_color, font=font)

        # 약간의 노이즈
        arr = np.array(img).astype(np.int16)
        noise = np.random.randint(-15, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __getitem__(self, idx):
        # 가변 자릿수도 학습 (1~4자리, 나머지는 blank)
        n = random.choices([2, 3, 4], weights=[1, 2, 7])[0]
        digits = [random.randint(0, 9) for _ in range(n)]
        # blank로 패딩 (왼쪽 정렬: 실제 번호는 보통 그렇게 표기)
        labels = digits + [BLANK] * (NUM_DIGITS - n)

        img = self._render(labels)
        x = self.tf(img)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y


# ===== 학습 =====
def train(model, loader, val_loader, epochs=50, lr=1e-3, save_path='digit4.pt'):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)  # y: (B, 4)
            logits = model(x)                  # (B, 4, 11)

            # 4개 head 각각 CE loss 합산
            loss = sum(F.cross_entropy(logits[:, i], y[:, i])
                       for i in range(NUM_DIGITS))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        sched.step()
        val_acc = evaluate(model, val_loader)
        print(f"[Epoch {epoch+1}/{epochs}] loss={total_loss/len(loader):.4f} "
              f"val_acc(seq)={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved {save_path}")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)               # (B, 4, 11)
        pred = logits.argmax(-1)        # (B, 4)
        # 시퀀스 전체가 맞아야 정답 (엄격한 기준)
        correct += (pred == y).all(dim=1).sum().item()
        total += x.size(0)
    return correct / total


# ===== 추론 =====
def decode(pred_indices):
    """[0, 4, 2, 7] -> '0427', [4, 2, 10, 10] -> '42'"""
    s = ''
    for d in pred_indices:
        if d == BLANK:
            continue
        s += str(int(d))
    return s


@torch.no_grad()
def infer_on_crop(model, pil_image):
    """YOLO crop 한 장 → 숫자 문자열. 파이프라인에서 호출."""
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    x = tf(pil_image).unsqueeze(0).to(DEVICE)
    logits = model(x)                    # (1, 4, 11)
    probs = F.softmax(logits, dim=-1)
    confs, pred = probs.max(-1)          # (1, 4), (1, 4)
    text = decode(pred[0].cpu().numpy())
    avg_conf = confs[0].mean().item()
    return text, avg_conf


def load_model(weights_path='digit4.pt'):
    model = Digit4Net()
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    return model


# ===== CLI =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['synth_train', 'infer'],
                    default='synth_train')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--weights', default='digit4.pt')
    ap.add_argument('--image', default=None)
    args = ap.parse_args()

    if args.mode == 'synth_train':
        print("[*] 합성 데이터셋 생성 중...")
        train_ds = SyntheticDigitDataset(length=20000, train=True)
        val_ds = SyntheticDigitDataset(length=2000, train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch,
                                shuffle=False, num_workers=2)
        model = Digit4Net()
        train(model, train_loader, val_loader,
              epochs=args.epochs, save_path=args.weights)

    elif args.mode == 'infer':
        assert args.image, "--image 경로 필요"
        model = load_model(args.weights)
        img = Image.open(args.image).convert('RGB')
        text, conf = infer_on_crop(model, img)
        print(f"인식 결과: {text} (avg conf: {conf:.3f})")


if __name__ == '__main__':
    main()