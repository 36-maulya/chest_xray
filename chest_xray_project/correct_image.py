import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
MODEL_PATH  = r"models\efficientnet_xray.pth"
EXCEL_PATH  = r"C:\Users\mauly\OneDrive\Desktop\major_project_annotation.xlsx"
OUTPUT_DIR  = r"outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
IMG_SIZE   = 224
THRESHOLD  = 0.5   # asymmetry threshold in cm
CLASS_NAMES = ['normal', 'rotated']

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)
print("✅ Model loaded successfully!")

# ─────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# LOAD EXCEL — get geometric measurements
# ─────────────────────────────────────────────
df = pd.read_excel(EXCEL_PATH, engine='openpyxl')

def get_measurements(image_num):
    row = df[df['image'] == image_num]
    if row.empty:
        return None, None
    right = float(row['right (cm)'].values[0])
    left  = float(row['left (cm)'].values[0])
    return right, left

# ─────────────────────────────────────────────
# GEOMETRIC ANALYSIS
# ─────────────────────────────────────────────
def analyze_rotation(right_cm, left_cm):
    diff = right_cm - left_cm
    abs_diff = abs(diff)

    if abs_diff <= THRESHOLD:
        status    = "NORMAL"
        direction = "None"
        severity  = "None"
        angle     = 0.0
    else:
        status = "ROTATED"
        # Direction
        if diff > 0:
            direction = "Rotated LEFT  (right clavicle farther from spine)"
        else:
            direction = "Rotated RIGHT (left clavicle farther from spine)"

        # Severity
        if abs_diff < 1.0:
            severity = "Mild"
        elif abs_diff < 2.0:
            severity = "Moderate"
        else:
            severity = "Severe"

        # Estimate correction angle (scaled from asymmetry)
        angle = round(abs_diff * 3.5, 1)
        if diff < 0:
            angle = -angle  # negative = rotate right

    return status, direction, severity, angle

# ─────────────────────────────────────────────
# CNN PREDICTION
# ─────────────────────────────────────────────
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output     = model(tensor)
        probs      = torch.softmax(output, dim=1)
        confidence = probs.max().item() * 100
        pred_class = CLASS_NAMES[probs.argmax().item()]
    return pred_class, confidence

# ─────────────────────────────────────────────
# AFFINE CORRECTION
# ─────────────────────────────────────────────
def correct_rotation(image_path, angle, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return None

    h, w  = img.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Apply affine transformation
    corrected = cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite(output_path, corrected)
    return corrected

# ─────────────────────────────────────────────
# VISUALIZE RESULT
# ─────────────────────────────────────────────
def show_result(original_path, corrected_img, image_num,
                pred_class, confidence, status,
                direction, severity, angle,
                right_cm, left_cm):

    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f"ORIGINAL — Image {image_num}\n"
                      f"CNN: {pred_class.upper()} ({confidence:.1f}%)\n"
                      f"Right: {right_cm} cm | Left: {left_cm} cm",
                      fontsize=10)
    axes[0].axis('off')

    if corrected_img is not None:
        corrected_rgb = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(corrected_rgb, cmap='gray')
        axes[1].set_title(f"CORRECTED — Rotation Applied\n"
                          f"Status: {status} | {severity}\n"
                          f"{direction}\n"
                          f"Correction Angle: {angle}°",
                          fontsize=10)
    else:
        axes[1].imshow(original, cmap='gray')
        axes[1].set_title("NO CORRECTION NEEDED\nImage is Normal", fontsize=10)

    axes[1].axis('off')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"result_{image_num}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Result saved: {save_path}")

# ─────────────────────────────────────────────
# MAIN — PROCESS ONE IMAGE
# ─────────────────────────────────────────────
def process_image(image_path, image_num):
    print(f"\n{'='*50}")
    print(f"Processing Image {image_num}: {image_path}")
    print(f"{'='*50}")

    # Step 1 — CNN Detection
    pred_class, confidence = predict_image(image_path)
    print(f"CNN Prediction  : {pred_class.upper()} ({confidence:.1f}% confidence)")

    # Step 2 — Geometric Analysis
    right_cm, left_cm = get_measurements(image_num)
    if right_cm is None:
        print(f"⚠️  No measurements found for image {image_num} in Excel")
        return

    print(f"Right clavicle  : {right_cm} cm")
    print(f"Left clavicle   : {left_cm} cm")
    print(f"Asymmetry       : {abs(right_cm - left_cm):.2f} cm")

    status, direction, severity, angle = analyze_rotation(right_cm, left_cm)
    print(f"Geometric Status: {status}")
    print(f"Direction       : {direction}")
    print(f"Severity        : {severity}")
    print(f"Correction Angle: {angle}°")

    # Step 3 — Affine Correction
    corrected_img = None
    if status == "ROTATED":
        out_path      = os.path.join(OUTPUT_DIR, f"corrected_{image_num}.jpeg")
        corrected_img = correct_rotation(image_path, angle, out_path)
        print(f"✅ Corrected image saved: {out_path}")
    else:
        print("✅ Image is Normal — no correction needed")

    # Step 4 — Show Result
    show_result(image_path, corrected_img, image_num,
                pred_class, confidence, status,
                direction, severity, angle,
                right_cm, left_cm)

# ─────────────────────────────────────────────
# RUN — change image number here to test any image
# ─────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_NUM  = 2  # ← change this to test any image (1 to 99)
    IMAGE_PATH = fr"D:\4SO23CS139\dataset_major_demo\{IMAGE_NUM}.jpeg"
    process_image(IMAGE_PATH, IMAGE_NUM)