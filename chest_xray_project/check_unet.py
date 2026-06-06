import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────────
# 1. DEVICE
# ─────────────────────────────────────────────
device = torch.device("cpu")
print("Using device:", device)

# ─────────────────────────────────────────────
# 2. YOUR U-NET ARCHITECTURE (MUST MATCH TRAINING)
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

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))


# ─────────────────────────────────────────────
# 3. LOAD MODEL SAFELY
# ─────────────────────────────────────────────
MODEL_PATH = "models/unet_xray (2).pth"

model = UNet().to(device)

try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully (state_dict)")
except Exception as e:
    print("❌ Model loading failed!")
    print("Error:", e)
    exit()


# ─────────────────────────────────────────────
# 4. IMAGE TEST
# ─────────────────────────────────────────────
IMAGE_PATH = r"D:\4SO23CS139\major_project_demo\chest_xray_project\data\rotated\5.jpeg"

img = Image.open(IMAGE_PATH).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

x = transform(img).unsqueeze(0).to(device)


# ─────────────────────────────────────────────
# 5. INFERENCE TEST
# ─────────────────────────────────────────────
with torch.no_grad():
    out = model(x)

print("\n===== OUTPUT CHECK =====")
print("Type  :", type(out))
print("Shape :", out.shape)
print("Min   :", out.min().item())
print("Max   :", out.max().item())


# ─────────────────────────────────────────────
# 6. SAVE OUTPUT IMAGE
# ─────────────────────────────────────────────
out_img = out.squeeze().cpu().numpy()
out_img = (out_img * 255).astype(np.uint8)

cv2.imwrite("unet_output.png", out_img)
print("\n✅ Saved output image: unet_output.png")


# ─────────────────────────────────────────────
# 7. VALIDATION RESULT
# ─────────────────────────────────────────────
if out.shape == (1, 1, 256, 256):
    print("\n🎉 MODEL IS VALID AND READY TO USE")
else:
    print("\n⚠️ WARNING: Unexpected output shape - model mismatch possible")