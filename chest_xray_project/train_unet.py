import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
IMAGE_FOLDER = r"E:\normal_dataset\Normal images"
MASK_FOLDER  = r"data\masks"
EXCEL_PATH   = r"E:\normal_dataset\auto_annotations.xlsx"
MODEL_PATH   = r"models\unet_xray.pth"
IMG_SIZE     = 128
BATCH_SIZE   = 4
EPOCHS       = 15
LR           = 0.001
VAL_SPLIT    = 0.1

os.makedirs("models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class XrayDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, mask_path = self.records[idx]

        # Load image
        img  = cv2.imread(img_path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img  = img.astype(np.float32) / 255.0
        img  = torch.from_numpy(img).permute(2, 0, 1)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask.astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

# ─────────────────────────────────────────────
# BUILD DATASET RECORDS
# ─────────────────────────────────────────────
df      = pd.read_excel(EXCEL_PATH, engine='openpyxl')
records = []

for _, row in df.iterrows():
    image_num = str(row['image']).strip()

    # Find image
    img_path = None
    for ext in ['.jpeg', '.jpg', '.png', '(1).jpeg']:
        p = os.path.join(IMAGE_FOLDER, f"{image_num}{ext}")
        if os.path.exists(p):
            img_path = p
            break

    mask_path = os.path.join(MASK_FOLDER, f"{image_num}_mask.png")

    if img_path and os.path.exists(mask_path):
        records.append((img_path, mask_path))

print(f"✅ Found {len(records)} image-mask pairs")

# ─────────────────────────────────────────────
# U-NET ARCHITECTURE
# ─────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Encoder ──
        self.enc1 = DoubleConv(3,   64)
        self.enc2 = DoubleConv(64,  128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ──
        self.bottleneck = DoubleConv(512, 1024)

        # ── Decoder ──
        self.up4    = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4   = DoubleConv(1024, 512)

        self.up3    = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3   = DoubleConv(512,  256)

        self.up2    = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2   = DoubleConv(256,  128)

        self.up1    = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec1   = DoubleConv(128,  64)

        # ── Output ──
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred   = pred.view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return 1 - (2. * inter + smooth) / \
               (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    bce  = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

# ─────────────────────────────────────────────
# SPLIT DATASET
# ─────────────────────────────────────────────
dataset   = XrayDataset(records)
val_size  = int(len(dataset) * VAL_SPLIT)
train_size= len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print(f"Train: {train_size} | Val: {val_size}")

# ─────────────────────────────────────────────
# INITIALIZE MODEL
# ─────────────────────────────────────────────
unet      = UNet().to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)

print(f"\n✅ U-Net initialized!")
print(f"--- Starting Training ---\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):

    # ── Train ──
    unet.train()
    train_loss = 0.0

    for imgs, masks in tqdm(train_loader,
                            desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = unet(imgs)
        loss  = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── Validate ──
    unet.eval()
    val_loss = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds    = unet(imgs)
            loss     = combined_loss(preds, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(unet.state_dict(), MODEL_PATH)
        print(f"   💾 Best U-Net saved! (Val Loss: {val_loss:.4f})")

print(f"\n✅ U-Net Training Complete!")
print(f"✅ Best Val Loss : {best_val_loss:.4f}")
print(f"✅ Model saved   : {MODEL_PATH}")