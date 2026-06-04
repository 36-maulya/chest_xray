import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
EXCEL_PATH   = r"E:\normal_dataset\auto_annotations.xlsx"
IMAGE_FOLDER = r"E:\normal_dataset\Normal images"
MASK_FOLDER  = r"data\masks"
IMG_SIZE     = 256  # U-Net input size

os.makedirs(MASK_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD EXCEL
# ─────────────────────────────────────────────
df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
print(f"✅ Loaded {len(df)} records from Excel")

# ─────────────────────────────────────────────
# GENERATE MASK FOR ONE IMAGE
# ─────────────────────────────────────────────
def generate_mask(row, img_w, img_h):
    """
    Generate segmentation mask with 3 classes:
    0 = background
    1 = spine region (white)
    2 = left clavicle (gray)
    3 = right clavicle (light gray)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Scale factors (original image → 256x256)
    sx = img_w  / float(row['image_width'])
    sy = img_h  / float(row['image_height'])

    # ── Draw Spine ──
    spine_x     = int(row['spine_x']        * sx)
    spine_top   = int(row['spine_top_y']    * sy)
    spine_bot   = int(row['spine_bottom_y'] * sy)
    spine_width = 15  # pixels wide

    cv2.rectangle(
        mask,
        (spine_x - spine_width, spine_top),
        (spine_x + spine_width, spine_bot),
        color=255,  # white = spine
        thickness=-1
    )

    # ── Draw Left Clavicle ──
    lx = int(row['clavicle_left_x']  * sx)
    ly = int(row['clavicle_left_y']  * sy)
    cv2.circle(mask, (lx, ly), radius=20, color=180, thickness=-1)

    # ── Draw Right Clavicle ──
    rx = int(row['clavicle_right_x'] * sx)
    ry = int(row['clavicle_right_y'] * sy)
    cv2.circle(mask, (rx, ry), radius=20, color=120, thickness=-1)

    return mask

# ─────────────────────────────────────────────
# PROCESS ALL IMAGES
# ─────────────────────────────────────────────
success  = 0
failed   = 0

print("\nGenerating masks...")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Masks"):
    image_num = str(row['image']).strip()

    # Find image
    found    = False
    possible = [
        f"{image_num}.jpeg",
        f"{image_num}.jpg",
        f"{image_num}.png",
        f"{image_num}(1).jpeg",
    ]

    img_path = None
    for fname in possible:
        path = os.path.join(IMAGE_FOLDER, fname)
        if os.path.exists(path):
            img_path = path
            found    = True
            break

    if not found:
        failed += 1
        continue

    # Read image to get dimensions
    img = cv2.imread(img_path)
    if img is None:
        failed += 1
        continue

    # Generate mask at 256x256
    mask = generate_mask(row, IMG_SIZE, IMG_SIZE)

    # Save mask
    mask_name = f"{image_num}_mask.png"
    mask_path = os.path.join(MASK_FOLDER, mask_name)
    cv2.imwrite(mask_path, mask)
    success += 1

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"✅ Mask Generation Complete!")
print(f"{'='*45}")
print(f"✅ Masks generated : {success}")
print(f"❌ Failed          : {failed}")
print(f"✅ Masks saved to  : {MASK_FOLDER}")