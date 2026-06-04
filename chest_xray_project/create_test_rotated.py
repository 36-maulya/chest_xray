import cv2
import numpy as np
import os

# ─────────────────────────────────────────────
# Take a normal image and artificially rotate it
# to create a clearly rotated test image
# ─────────────────────────────────────────────

IMAGE_PATH  = r"E:\normal_dataset\Normal images\8.jpeg"
OUTPUT_PATH = r"test_rotated.jpeg"

# Read image
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]
center = (w // 2, h // 2)

# Rotate by 15 degrees — clearly visible rotation
angle = 15
M     = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(
    img, M, (w, h),
    flags      = cv2.INTER_LINEAR,
    borderMode = cv2.BORDER_REPLICATE
)

# Save
cv2.imwrite(OUTPUT_PATH, rotated)
print(f"✅ Rotated test image saved: {OUTPUT_PATH}")
print(f"   Rotation applied: {angle} degrees")
print(f"   Image size: {w}x{h}")
