import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
IMAGE_FOLDER  = r"E:\normal_dataset\Normal images"
EXCEL_PATH    = r"E:\normal_dataset\auto_annotations.xlsx"
MODEL_PATH    = r"models\stn_xray.pth"
IMG_SIZE      = 224
BATCH_SIZE    = 16
EPOCHS        = 30
LR            = 0.0001
VAL_SPLIT     = 0.1

os.makedirs("models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class XraySTNDataset(Dataset):
    def __init__(self, records):
        self.records   = records
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, angle, label = self.records[idx]

        # Load image
        img  = cv2.imread(img_path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img  = self.transform(img)

        # For rotated images — create target (corrected version)
        if label == "Rotated" and abs(angle) > 0.5:
            # Load original and rotate back to get target
            orig    = cv2.imread(img_path)
            orig    = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            h, w    = orig.shape
            center  = (w//2, h//2)
            M       = cv2.getRotationMatrix2D(center, -angle, 1.0)
            target  = cv2.warpAffine(
                orig, M, (w, h),
                flags      = cv2.INTER_LINEAR,
                borderMode = cv2.BORDER_REPLICATE
            )
            target = self.transform(target)
        else:
            target = img.clone()

        return img, target, torch.tensor(
            angle, dtype=torch.float32
        )

# ─────────────────────────────────────────────
# BUILD RECORDS
# ─────────────────────────────────────────────
df      = pd.read_excel(EXCEL_PATH, engine='openpyxl')
records = []

for _, row in df.iterrows():
    image_num = str(row['image']).strip()
    img_path  = None
    for ext in ['.jpeg', '.jpg', '.png', '(1).jpeg']:
        p = os.path.join(IMAGE_FOLDER, f"{image_num}{ext}")
        if os.path.exists(p):
            img_path = p
            break
    if img_path:
        angle = float(row['asymmetry_cm']) * 3.0
        label = str(row['label'])
        if label == 'Rotated':
            angle = angle
        else:
            angle = 0.0
        records.append((img_path, angle, label))

print(f"✅ Found {len(records)} records")

# ─────────────────────────────────────────────
# STN MODEL
# ─────────────────────────────────────────────
class STN(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Localization Network ──
        # Learns WHERE and HOW to transform
        self.locnet = nn.Sequential(
            nn.Conv2d(1,  32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(4),
        )

        # ── Regression to transformation params ──
        self.fc_loc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 6)  # 6 params for affine transform
        )

        # Initialize with identity transform
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor(
                [1, 0, 0, 0, 1, 0],
                dtype=torch.float
            )
        )

        # ── Feature extraction for reconstruction ──
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1,  3, padding=1),
        )

    def stn(self, x):
        # Get transformation parameters
        feat  = self.locnet(x)
        feat  = feat.view(feat.size(0), -1)
        theta = self.fc_loc(feat)
        theta = theta.view(-1, 2, 3)

        # Create sampling grid
        grid = F.affine_grid(
            theta, x.size(), align_corners=False
        )

        # Sample from input using grid
        x_transformed = F.grid_sample(
            x, grid,
            align_corners=False,
            mode='bilinear',
            padding_mode='border'
        )
        return x_transformed, theta

    def forward(self, x):
        # Apply spatial transformation
        x_transformed, theta = self.stn(x)

        # Enhance transformed image
        output = self.encoder(x_transformed)
        output = x_transformed + output * 0.1

        return output, theta

# ─────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────
def stn_loss(output, target, theta):
    # Reconstruction loss
    recon_loss = F.mse_loss(output, target)

    # Regularization — keep transformation reasonable
    identity = torch.tensor(
        [[[1, 0, 0], [0, 1, 0]]],
        dtype=torch.float32
    ).to(theta.device)
    reg_loss = F.mse_loss(
        theta, identity.expand_as(theta)
    )

    return recon_loss + 0.01 * reg_loss

# ─────────────────────────────────────────────
# SPLIT DATASET
# ─────────────────────────────────────────────
dataset    = XraySTNDataset(records)
val_size   = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2
)
val_loader   = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2
)

print(f"Train: {train_size} | Val: {val_size}")

# ─────────────────────────────────────────────
# INITIALIZE
# ─────────────────────────────────────────────
stn_model = STN().to(device)
optimizer = torch.optim.Adam(stn_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

print(f"\n✅ STN initialized!")
print(f"--- Starting Training ---\n")

best_val_loss = float('inf')

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
for epoch in range(EPOCHS):

    # ── Train ──
    stn_model.train()
    train_loss = 0.0

    for imgs, targets, angles in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"
    ):
        imgs    = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output, theta = stn_model(imgs)
        loss          = stn_loss(output, targets, theta)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            stn_model.parameters(), 1.0
        )

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── Validate ──
    stn_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for imgs, targets, angles in val_loader:
            imgs    = imgs.to(device)
            targets = targets.to(device)
            output, theta = stn_model(imgs)
            loss          = stn_loss(output, targets, theta)
            val_loss     += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss:   {val_loss:.4f}")

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(stn_model.state_dict(), MODEL_PATH)
        print(f"   💾 Best STN saved! "
              f"(Val Loss: {val_loss:.4f})")

print(f"\n{'='*45}")
print(f"✅ STN Training Complete!")
print(f"{'='*45}")
print(f"✅ Best Val Loss : {best_val_loss:.4f}")
print(f"✅ Model saved   : {MODEL_PATH}")