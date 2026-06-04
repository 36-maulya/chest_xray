import pandas as pd
import shutil
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
EXCEL_PATH   = r"E:\normal_dataset\auto_annotations.xlsx"
IMAGE_FOLDER = r"E:\normal_dataset\Normal images"
NORMAL_DIR   = r"data\normal"
ROTATED_DIR  = r"data\rotated"

# Threshold: if difference > this → Rotated
THRESHOLD = 0.5  # in cm

# ─────────────────────────────────────────────
# Create folders
# ─────────────────────────────────────────────
os.makedirs(NORMAL_DIR,  exist_ok=True)
os.makedirs(ROTATED_DIR, exist_ok=True)

# Clear existing images in folders first
for folder in [NORMAL_DIR, ROTATED_DIR]:
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))
print("✅ Cleared old data folders")

# ─────────────────────────────────────────────
# Read Excel
# ─────────────────────────────────────────────
df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
print(f"Total images in Excel: {len(df)}")

normal_count  = 0
rotated_count = 0
not_found     = []

for _, row in df.iterrows():
    image_num  = str(row['image']).strip()
    right_dist = float(row['right_cm'])
    left_dist  = float(row['left_cm'])

    # Calculate asymmetry
    diff  = abs(right_dist - left_dist)
    label = "rotated" if diff > THRESHOLD else "normal"

    # Try different filename formats
    found    = False
    possible = [
        f"{image_num}.jpeg",
        f"{image_num}.jpg",
        f"{image_num}.png",
        f"{image_num}(2).jpeg",
        f"{image_num}(2).jpg",
    ]

    for filename in possible:
        src_path = os.path.join(IMAGE_FOLDER, filename)
        if os.path.exists(src_path):
            # Copy to correct folder
            if label == "normal":
                shutil.copy(src_path, os.path.join(NORMAL_DIR, filename))
                normal_count += 1
            else:
                shutil.copy(src_path, os.path.join(ROTATED_DIR, filename))
                rotated_count += 1
            found = True
            break

    if not found:
        not_found.append(f"{image_num}.jpeg")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"✅ Dataset Preparation Complete!")
print(f"{'='*45}")
print(f"✅ Normal  images : {normal_count}")
print(f"✅ Rotated images : {rotated_count}")
print(f"✅ Total processed: {normal_count + rotated_count}")

if not_found:
    print(f"\n⚠️  Not found ({len(not_found)} images):")
    for f in not_found[:10]:
        print(f"   - {f}")
    if len(not_found) > 10:
        print(f"   ... and {len(not_found)-10} more")
else:
    print("\n✅ All images found and sorted successfully!")