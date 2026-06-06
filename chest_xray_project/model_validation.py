import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device("cpu")
print("Using device:", device)

# ─────────────────────────────────────────────
# 1. STN MODEL (MATCHES YOUR TRAINING EXACTLY)
# ─────────────────────────────────────────────
class STN(nn.Module):
    def __init__(self):
        super().__init__()

        self.locnet = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
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

        self.fc_loc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 6)
        )

        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(
            torch.tensor([1,0,0,0,1,0], dtype=torch.float)
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def stn(self, x):
        feat = self.locnet(x)
        feat = feat.view(feat.size(0), -1)
        theta = self.fc_loc(feat).view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_trans = F.grid_sample(
            x, grid,
            align_corners=False,
            mode='bilinear',
            padding_mode='border'
        )
        return x_trans, theta

    def forward(self, x):
        x_trans, theta = self.stn(x)
        out = self.encoder(x_trans)
        return x_trans + out * 0.1, theta


# ─────────────────────────────────────────────
# 2. U-NET (MATCHES YOUR TRAINING EXACTLY)
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

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return torch.sigmoid(self.out(d1))


# ─────────────────────────────────────────────
# 3. LOAD MODELS
# ─────────────────────────────────────────────
STN_PATH  = "models/stn_xray.pth"
UNET_PATH = "models/unet_xray (2).pth"

stn = STN().to(device)
unet = UNet().to(device)

# STN LOAD
try:
    stn.load_state_dict(torch.load(STN_PATH, map_location=device))
    stn.eval()
    print("✅ STN loaded successfully")
except Exception as e:
    print("❌ STN load failed:", e)

# UNET LOAD
try:
    unet.load_state_dict(torch.load(UNET_PATH, map_location=device))
    unet.eval()
    print("✅ U-Net loaded successfully")
except Exception as e:
    print("❌ U-Net load failed:", e)


# ─────────────────────────────────────────────
# 4. TEST IMAGE
# ─────────────────────────────────────────────
IMAGE_PATH = r"D:\4SO23CS139\major_project_demo\chest_xray_project\data\rotated\5.jpeg"

img = Image.open(IMAGE_PATH).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

x = transform(img).unsqueeze(0).to(device)


# ─────────────────────────────────────────────
# 5. STN TEST
# ─────────────────────────────────────────────
with torch.no_grad():
    stn_out, theta = stn(x[:,0:1,:,:])  # grayscale for STN

print("\n===== STN OUTPUT =====")
print("Shape:", stn_out.shape)


# ─────────────────────────────────────────────
# 6. U-NET TEST
# ─────────────────────────────────────────────
with torch.no_grad():
    out = unet(x)

print("\n===== U-NET OUTPUT =====")
print("Shape:", out.shape)
print("Min:", out.min().item())
print("Max:", out.max().item())


# ─────────────────────────────────────────────
# 7. SAVE OUTPUT
# ─────────────────────────────────────────────
out_img = out.squeeze().cpu().numpy()
out_img = (out_img * 255).astype(np.uint8)

cv2.imwrite("model_output.png", out_img)

print("\n✅ Saved: model_output.png")


# ─────────────────────────────────────────────
# 8. VALIDATION CHECK
# ─────────────────────────────────────────────
if out.shape[2:] == (224, 224):
    print("\n🎉 MODELS WORKING CORRECTLY")
else:
    print("\n⚠️ Output size mismatch")