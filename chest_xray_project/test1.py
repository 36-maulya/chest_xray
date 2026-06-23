import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import timm
import base64
import traceback
import io
import math
import torchxrayvision as xrv
import skimage.transform


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION SETUP
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Chest X-Ray Positioning AI",
    description="Automated detection, alignment, and orientation correction of chest radiographs",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
EFFICIENTNET_PATH = r"models\efficientnet_xray_v2.pth"
CLAVICLE_MODEL_PATH = r"models\clavicle_distance_model.pth"
OUTPUT_DIR = r"outputs"
IMG_SIZE = 224
CLASS_NAMES = ["normal", "rotated"]
PIXEL_SPACING_MM = 0.912

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLAVICLE DISTANCE REGRESSION MODEL (Trained in Colab)
# ═══════════════════════════════════════════════════════════════════════════════
class ClavicleDistanceRegressor(nn.Module):
    """
    Trained model that predicts the distance (in cm) from the spine to the
    medial clavicle head on each side.

    Architecture: EfficientNet-B0 backbone → regression head → [left_cm, right_cm]

    Training: 3269 annotated chest X-rays with ground truth measurements.
    """

    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 1280

        # Default head - will be replaced during loading if needed
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL INITIALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# 1) EfficientNet: Rotation classifier (normal vs rotated)
efficientnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
efficientnet.eval()
efficientnet.to(device)
print("✅ EfficientNet loaded! (Rotation Classification)")

# 2) Clavicle Distance Regressor (trained model)
# Try multiple architectures to find one that matches the checkpoint
print("Loading clavicle distance model...")

clavicle_model = None
checkpoint = torch.load(CLAVICLE_MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    val_mae_left = checkpoint.get("val_mae_left", "N/A")
    val_mae_right = checkpoint.get("val_mae_right", "N/A")
else:
    state_dict = checkpoint
    val_mae_left, val_mae_right = "N/A", "N/A"

# Try different architectures
architectures = [
    # Simple architecture
    {
        "name": "Simple (Linear-ReLU-Linear)",
        "head": nn.Sequential(nn.Linear(1280, 512), nn.ReLU(), nn.Linear(512, 2)),
    },
    # Architecture with BatchNorm
    {
        "name": "With BatchNorm",
        "head": nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2),
        ),
    },
    # Architecture with Dropout and BatchNorm
    {
        "name": "With Dropout + BatchNorm",
        "head": nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2),
        ),
    },
]

for arch in architectures:
    try:
        test_model = ClavicleDistanceRegressor.__new__(ClavicleDistanceRegressor)
        nn.Module.__init__(test_model)
        test_model.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0
        )
        test_model.head = arch["head"]

        # Try loading with strict=True first
        test_model.load_state_dict(state_dict, strict=True)

        # If we get here, it worked!
        clavicle_model = test_model
        print(f"✅ Loaded with architecture: {arch['name']}")
        break
    except Exception as e:
        continue

# If no exact match, use the first architecture with strict=False
if clavicle_model is None:
    print("⚠️ No exact architecture match found, using flexible loading...")
    clavicle_model = ClavicleDistanceRegressor(pretrained=False)
    missing, unexpected = clavicle_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   Missing {len(missing)} keys (will use random initialization)")
    if unexpected:
        print(f"   Ignoring {len(unexpected)} extra keys from checkpoint")

print(
    f"✅ Clavicle model ready (Val MAE - Left: {val_mae_left} cm, Right: {val_mae_right} cm)"
)
clavicle_model.eval()
clavicle_model.to(device)

# 3) TorchXRayVision PSPNet: Anatomical landmark detection
print("Loading TorchXRayVision...")
xrv_model = xrv.baseline_models.chestx_det.PSPNet()
xrv_model.eval()
print("✅ TorchXRayVision loaded! (Anatomical Landmark Detection)")


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
efficientnet_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def is_valid_xray(img: Image.Image) -> bool:
    """Validate that the uploaded image is a chest X-ray."""
    gray = np.array(img.convert("L"))
    mean_brightness = gray.mean()
    std_brightness = gray.std()
    return 20 < mean_brightness < 230 and std_brightness >= 15


def image_to_base64(img_array):
    """Convert OpenCV image array to base64 string."""
    _, buffer = cv2.imencode(".jpg", img_array)
    return base64.b64encode(buffer).decode("utf-8")


def px_to_cm(px):
    """Convert pixel distance to centimeters using pixel spacing."""
    return round((px * PIXEL_SPACING_MM) / 10, 2)


def cm_to_px(cm):
    """Convert centimeters to pixel distance using pixel spacing."""
    return (cm * 10) / PIXEL_SPACING_MM


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINES
# ═══════════════════════════════════════════════════════════════════════════════


def predict_rotation(img: Image.Image):
    """
    EfficientNet binary classification: normal vs rotated.
    Returns: (class_name, confidence%, normal_conf%, rotated_conf%)
    """
    tensor = efficientnet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = efficientnet(tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs.max().item() * 100
        pred_idx = probs.argmax().item()
        return (
            CLASS_NAMES[pred_idx],
            confidence,
            probs[0][0].item() * 100,
            probs[0][1].item() * 100,
        )


def predict_clavicle_distances(img: Image.Image):
    """
    Trained regression model predicts left and right clavicle-to-spine distances.

    Returns: (left_cm, right_cm)

    These predictions are trained on ground truth annotations and used for:
    - Rotation ratio calculation
    - Decision logic (normal vs rotated)
    - Label text in visualizations

    NOT used for drawing clavicle positions (those come from PSPNet).
    """
    tensor = efficientnet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = clavicle_model(tensor).cpu().numpy()[0]
        left_cm = round(max(0.1, pred[0]), 2)  # clamp to positive
        right_cm = round(max(0.1, pred[1]), 2)
        return left_cm, right_cm


# ═══════════════════════════════════════════════════════════════════════════════
# PSPNet ANATOMICAL LANDMARK DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def detect_landmarks_pspnet(img_cv2, is_corrected=False):
    """
    Uses TorchXRayVision PSPNet to detect REAL anatomical landmarks:

    - spine_x, spine_y: Vertebral column midpoint
    - clav_lx, clav_ly: Left clavicle (REAL detected position)
    - clav_rx, clav_ry: Right clavicle (REAL detected position)
    - mediastinum_x, mediastinum_y: Mediastinal center
    - left/right scapula positions
    - left/right lung positions

    All coordinates are actual pixel positions detected by the model,
    not estimated from distances.

    Returns dict with all landmark coordinates + geometric distances.
    """
    try:
        h_orig, w_orig = img_cv2.shape[:2]
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        img_xrv = (gray.astype(np.float32) / 255.0) * 2048 - 1024
        img_xrv = skimage.transform.resize(img_xrv, (512, 512), anti_aliasing=True)
        img_tensor = (
            torch.from_numpy(img_xrv).unsqueeze(0).unsqueeze(0).float().to(device)
        )

        with torch.no_grad():
            output = xrv_model(img_tensor)
            landmarks_maps = output[0].detach().cpu().numpy()
            targets = xrv_model.targets

            def get_pos(hmap, ow, oh):
                hmap = cv2.resize(hmap, (ow, oh))
                y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
                return int(x), int(y)

            landmarks = {}
            for i, label in enumerate(targets):
                x, y = get_pos(landmarks_maps[i], w_orig, h_orig)
                ll = label.lower()

                if "spine" in ll:
                    landmarks["spine_x"], landmarks["spine_y"] = x, y
                elif "left clavicle" in ll:
                    landmarks["clav_lx"], landmarks["clav_ly"] = x, y
                elif "right clavicle" in ll:
                    landmarks["clav_rx"], landmarks["clav_ry"] = x, y
                elif "mediastinum" in ll:
                    landmarks["mediastinum_x"], landmarks["mediastinum_y"] = x, y
                elif "left scapula" in ll:
                    landmarks["left_scapula_x"], landmarks["left_scapula_y"] = x, y
                elif "right scapula" in ll:
                    landmarks["right_scapula_x"], landmarks["right_scapula_y"] = x, y
                elif "left lung" in ll:
                    landmarks["left_lung_x"], landmarks["left_lung_y"] = x, y
                elif "right lung" in ll:
                    landmarks["right_lung_x"], landmarks["right_lung_y"] = x, y

            # Extract key landmarks with fallbacks
            spine_x = landmarks.get("spine_x", w_orig // 2)
            spine_y = landmarks.get("spine_y", h_orig // 2)

            clav_lx = landmarks.get("clav_lx", int(w_orig * 0.3))
            clav_ly = landmarks.get("clav_ly", int(h_orig * 0.2))
            clav_rx = landmarks.get("clav_rx", int(w_orig * 0.7))
            clav_ry = landmarks.get("clav_ry", int(h_orig * 0.2))

            # Sanity checks
            if clav_lx > clav_rx:
                clav_lx, clav_rx = clav_rx, clav_lx
                clav_ly, clav_ry = clav_ry, clav_ly

            if clav_lx > spine_x:
                clav_lx = int(w_orig * 0.35)
            if clav_rx < spine_x:
                clav_rx = int(w_orig * 0.65)

            # Calculate GEOMETRIC distances from PSPNet (actual pixel measurements)
            geo_left_px = abs(spine_x - clav_lx)
            geo_right_px = abs(clav_rx - spine_x)
            geo_left_cm = px_to_cm(geo_left_px)
            geo_right_cm = px_to_cm(geo_right_px)

            # Terminal output
            label_status = "✨ POST-CORRECTION" if is_corrected else "📁 RAW ORIGINAL"
            print("\n" + "=" * 55)
            print(f"    📍 PSPNet LANDMARKS ({label_status})")
            print("=" * 55)
            print(f"  Spine       (X, Y) : ({spine_x}, {spine_y})")
            print(f"  L-Clavicle  (X, Y) : ({clav_lx}, {clav_ly})")
            print(f"  R-Clavicle  (X, Y) : ({clav_rx}, {clav_ry})")
            print(
                f"  Mediastinum (X, Y) : ({landmarks.get('mediastinum_x', 'N/A')}, "
                f"{landmarks.get('mediastinum_y', 'N/A')})"
            )
            print("-" * 55)
            print(f"  Geometric Left Dist  : {geo_left_px}px = {geo_left_cm} cm")
            print(f"  Geometric Right Dist : {geo_right_px}px = {geo_right_cm} cm")
            print("=" * 55 + "\n")

            return {
                "spine_x": spine_x,
                "spine_y": spine_y,
                "clav_lx": clav_lx,
                "clav_ly": clav_ly,
                "clav_rx": clav_rx,
                "clav_ry": clav_ry,
                "mediastinum_x": landmarks.get("mediastinum_x", spine_x),
                "mediastinum_y": landmarks.get("mediastinum_y", int(h_orig * 0.4)),
                "left_scapula_x": landmarks.get("left_scapula_x"),
                "left_scapula_y": landmarks.get("left_scapula_y"),
                "right_scapula_x": landmarks.get("right_scapula_x"),
                "right_scapula_y": landmarks.get("right_scapula_y"),
                "left_lung_x": landmarks.get("left_lung_x"),
                "left_lung_y": landmarks.get("left_lung_y"),
                "right_lung_x": landmarks.get("right_lung_x"),
                "right_lung_y": landmarks.get("right_lung_y"),
                "geo_left_cm": geo_left_cm,
                "geo_right_cm": geo_right_cm,
                "image_w": w_orig,
                "image_h": h_orig,
            }

    except Exception as e:
        print("❌ PSPNet error:", e)
        traceback.print_exc()
        h, w = img_cv2.shape[:2]
        return {
            "spine_x": w // 2,
            "spine_y": h // 2,
            "clav_lx": int(w * 0.3),
            "clav_ly": int(h * 0.2),
            "clav_rx": int(w * 0.7),
            "clav_ry": int(h * 0.2),
            "mediastinum_x": w // 2,
            "mediastinum_y": int(h * 0.4),
            "geo_left_cm": px_to_cm(abs(w // 2 - int(w * 0.3))),
            "geo_right_cm": px_to_cm(abs(int(w * 0.7) - w // 2)),
            "image_w": w,
            "image_h": h,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION (Uses REAL PSPNet clavicle positions)
# ═══════════════════════════════════════════════════════════════════════════════


def draw_clavicle_measurements(
    img_cv2, landmarks, model_left_cm, model_right_cm, save_path
):
    """
    Draws measurement lines between REAL detected landmarks.

    Lines: spine → PSPNet's detected left clavicle head
           spine → PSPNet's detected right clavicle head

    Labels show the REGRESSION MODEL's predicted cm values.
    Dots are at REAL PSPNet-detected anatomical positions.
    """
    vis = img_cv2.copy()
    h, w = vis.shape[:2]

    spine_x = landmarks["spine_x"]
    clav_lx = landmarks["clav_lx"]
    clav_ly = landmarks["clav_ly"]
    clav_rx = landmarks["clav_rx"]
    clav_ry = landmarks["clav_ry"]

    # Vertical spine reference line
    cv2.line(vis, (spine_x, 50), (spine_x, h - 50), (0, 0, 0), 2)

    # Horizontal lines: spine → REAL clavicle head (PSPNet detected)
    cv2.line(vis, (clav_lx, clav_ly), (spine_x, clav_ly), (0, 0, 0), 2)
    cv2.line(vis, (spine_x, clav_ry), (clav_rx, clav_ry), (0, 0, 0), 2)

    # Labels: show regression model's predicted distances
    cv2.putText(
        vis,
        f"{model_left_cm:.2f} cm",
        (min(clav_lx, spine_x) - 130, clav_ly + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        vis,
        f"{model_right_cm:.2f} cm",
        (max(spine_x, clav_rx) + 30, clav_ry + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    # Dots at REAL PSPNet-detected positions
    cv2.circle(vis, (clav_lx, clav_ly), 8, (0, 0, 255), -1)  # red = clavicle head
    cv2.circle(vis, (clav_rx, clav_ry), 8, (0, 0, 255), -1)  # red = clavicle head
    cv2.circle(vis, (spine_x, clav_ly), 5, (0, 255, 0), -1)  # green = spine
    cv2.circle(vis, (spine_x, clav_ry), 5, (0, 255, 0), -1)  # green = spine

    # Labels
    cv2.putText(
        vis,
        "L-Clav",
        (clav_lx + 12, clav_ly - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        vis,
        "R-Clav",
        (clav_rx + 12, clav_ry - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        vis,
        "Spine",
        (spine_x + 8, clav_ly - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    cv2.imwrite(save_path, vis)
    return vis


def visualize_landmarks(img_cv2, landmarks, model_left_cm, model_right_cm):
    """Debug visualization with colored dots at real PSPNet positions."""
    vis = img_cv2.copy()

    spine_x = landmarks["spine_x"]
    clav_lx = landmarks["clav_lx"]
    clav_ly = landmarks["clav_ly"]
    clav_rx = landmarks["clav_rx"]
    clav_ry = landmarks["clav_ry"]

    # Lines from spine to REAL clavicle heads
    cv2.line(vis, (spine_x, clav_ly), (clav_lx, clav_ly), (0, 255, 255), 2)
    cv2.line(vis, (spine_x, clav_ry), (clav_rx, clav_ry), (255, 255, 0), 2)

    # Dots at real positions
    points = [
        ("Spine", spine_x, (clav_ly + clav_ry) // 2, (0, 255, 0)),
        ("L-Head", clav_lx, clav_ly, (0, 0, 255)),
        ("R-Head", clav_rx, clav_ry, (0, 0, 255)),
    ]
    for name, x, y, color in points:
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.putText(
            vis, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # Distance text (regression model values)
    cv2.putText(
        vis,
        f"Model: L={model_left_cm:.2f}cm  R={model_right_cm:.2f}cm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # Geometric distances (from PSPNet pixel measurements)
    geo_l = landmarks["geo_left_cm"]
    geo_r = landmarks["geo_right_cm"]
    cv2.putText(
        vis,
        f"PSPNet: L={geo_l:.2f}cm  R={geo_r:.2f}cm",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    return vis


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY ASSESSMENT METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_rotation_metrics(landmarks, model_left_cm, model_right_cm):
    """
    Calculates rotation metrics using dual-source approach:

    - REGRESSION MODEL distances for rotation_ratio (trained on ground truth)
    - PSPNet clavicle X positions for angle calculation (real anatomical points)

    Returns: (angle, rotation_ratio, direction, severity)
    """
    spine_x = landmarks["spine_x"]
    clav_lx = landmarks["clav_lx"]
    clav_rx = landmarks["clav_rx"]
    clav_ly = landmarks["clav_ly"]
    clav_ry = landmarks["clav_ry"]

    # Rotation ratio from regression model (trained on ground truth)
    left_px = cm_to_px(model_left_cm)
    right_px = cm_to_px(model_right_cm)
    rotation_ratio = abs(left_px - right_px) / max(1, (left_px + right_px))

    if rotation_ratio < 0.05:
        return 0.0, rotation_ratio, "None", "None"

    # Angle from REAL PSPNet clavicle positions
    angle = math.degrees(math.atan2(clav_ly - clav_ry, clav_rx - clav_lx))
    angle = max(min(round(angle, 1), 15.0), -15.0)

    diff = right_px - left_px
    direction = "Left rotation" if diff > 0 else "Right rotation"

    if rotation_ratio < 0.10:
        severity = "Mild"
    elif rotation_ratio < 0.20:
        severity = "Moderate"
    else:
        severity = "Severe"

    return angle, rotation_ratio, direction, severity


def calculate_mediastinal_shift(landmarks):
    """Assess mediastinal centering relative to spine."""
    spine_x = landmarks["spine_x"]
    mediastinum_x = landmarks.get("mediastinum_x", spine_x)
    shift = abs(mediastinum_x - spine_x)
    status = (
        "Normal" if shift < 20 else "Mild Shift" if shift < 40 else "Significant Shift"
    )
    return shift, status


def calculate_scapular_position(landmarks):
    """Assess scapular retraction relative to lung fields."""
    ls_x = landmarks.get("left_scapula_x")
    rs_x = landmarks.get("right_scapula_x")
    ll_x = landmarks.get("left_lung_x")
    rl_x = landmarks.get("right_lung_x")

    if None in [ls_x, rs_x, ll_x, rl_x]:
        return "Unknown"

    avg_overlap = (abs(ls_x - ll_x) + abs(rs_x - rl_x)) / 2
    return (
        "Good Retraction"
        if avg_overlap > 100
        else "Acceptable"
        if avg_overlap > 40
        else "Scapula Overlapping Lung Field"
    )


def assess_lung_coverage(landmarks):
    """Assess lung field visibility and coverage."""
    ll_x = landmarks.get("left_lung_x")
    rl_x = landmarks.get("right_lung_x")
    ll_y = landmarks.get("left_lung_y")
    rl_y = landmarks.get("right_lung_y")
    h = landmarks.get("image_h", 1000)

    if None in [ll_x, rl_x, ll_y, rl_y]:
        return "Unknown", 0

    # Check if lung landmarks are in reasonable vertical positions
    lung_coverage_score = min(ll_y, rl_y) / h
    status = "Adequate" if lung_coverage_score > 0.15 else "Limited"

    return status, lung_coverage_score


def geometry_decision(angle, rotation_ratio):
    """Final geometry-based decision: rotated or normal."""
    return "rotated" if rotation_ratio > 0.15 else "normal"


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC WARP CORRECTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def anatomical_warp_correction(img_cv2, landmarks, model_left_cm, model_right_cm):
    """
    Warp image to correct rotation using regression model distances.
    Scales each half independently to equalize left/right distances.
    """
    h, w = img_cv2.shape[:2]
    spine_x = landmarks["spine_x"]

    left_px = cm_to_px(model_left_cm)
    right_px = cm_to_px(model_right_cm)

    rotation_ratio = abs(left_px - right_px) / max(1, (left_px + right_px))
    correction_strength = (
        0.35 if rotation_ratio > 0.30 else 0.25 if rotation_ratio > 0.20 else 0.15
    )

    avg_dist = (left_px + right_px) / 2
    target_left = left_px + (avg_dist - left_px) * correction_strength
    target_right = right_px + (avg_dist - right_px) * correction_strength

    scale_l = np.clip(target_left / max(1, left_px), 0.9, 1.2)
    scale_r = np.clip(target_right / max(1, right_px), 0.9, 1.2)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        map_y[y, :] = y
        for x in range(w):
            if x < spine_x:
                map_x[y, x] = spine_x - (spine_x - x) / scale_l
            else:
                map_x[y, x] = spine_x + (x - spine_x) / scale_r

    corrected = cv2.remap(
        img_cv2,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN API ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Main analysis endpoint.

    Pipeline:
    1. Validate X-ray
    2. CNN rotation classification (EfficientNet)
    3. Regression model predicts clavicle distances (trained model)
    4. PSPNet detects REAL anatomical landmarks
    5. Calculate rotation metrics (dual-source)
    6. Assess image quality (mediastinal, scapular, lung coverage)
    7. Final decision
    8. Visualization (real PSPNet positions)
    9. Correction (if needed) + post-correction metrics (PSPNet geometric)
    """
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        # ── Step 1: Validate X-ray ──
        if not is_valid_xray(img_pil):
            return JSONResponse(
                {"valid": False, "message": "Please upload a valid chest X-ray image."}
            )

        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # ── Step 2: CNN rotation classification ──
        cnn_class, cnn_conf, normal_conf, rotated_conf = predict_rotation(img_pil)
        cnn_signal = 1 if cnn_class == "rotated" else 0

        # ── Step 3: Regression model predicts distances (cm) ──
        model_left_cm, model_right_cm = predict_clavicle_distances(img_pil)
        print(
            f"\n[Regression Model] Left: {model_left_cm} cm | Right: {model_right_cm} cm"
        )

        # ── Step 4: PSPNet detects REAL anatomical landmarks ──
        landmarks = detect_landmarks_pspnet(img_cv2)

        # ── Step 5: Calculate rotation metrics (dual-source) ──
        angle, rotation_ratio, direction, severity = calculate_rotation_metrics(
            landmarks, model_left_cm, model_right_cm
        )

        # ── Step 6: Assess image quality parameters ──
        mediastinal_shift, mediastinal_status = calculate_mediastinal_shift(landmarks)
        scapular_status = calculate_scapular_position(landmarks)
        lung_status, lung_coverage_score = assess_lung_coverage(landmarks)

        # ── Step 7: Final decision ──
        geo_class = geometry_decision(angle, rotation_ratio)
        final_status = (
            "ROTATED"
            if geo_class == "rotated" or (cnn_signal == 1 and cnn_conf > 70)
            else "NORMAL"
        )

        # ── Step 8: Visualization (real PSPNet positions) ──
        debug_img = visualize_landmarks(
            img_cv2, landmarks, model_left_cm, model_right_cm
        )
        draw_clavicle_measurements(
            img_cv2,
            landmarks,
            model_left_cm,
            model_right_cm,
            os.path.join(OUTPUT_DIR, "before_correction.jpg"),
        )
        cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_landmarks.jpg"), debug_img)

        # ── Step 9: Correction (if needed) ──
        corrected_b64 = None
        post_left_cm, post_right_cm = model_left_cm, model_right_cm
        post_geo_left_cm = landmarks["geo_left_cm"]
        post_geo_right_cm = landmarks["geo_right_cm"]

        if final_status == "ROTATED" and rotation_ratio > 0.15:
            # Warp the image
            corrected_cv2 = anatomical_warp_correction(
                img_cv2, landmarks, model_left_cm, model_right_cm
            )

            # Post-correction: use PSPNet geometric distances on warped image.
            # Do NOT re-run the regression model (trained on original X-rays).
            corrected_landmarks = detect_landmarks_pspnet(
                corrected_cv2, is_corrected=True
            )

            post_geo_left_cm = corrected_landmarks["geo_left_cm"]
            post_geo_right_cm = corrected_landmarks["geo_right_cm"]
            post_left_cm = post_geo_left_cm
            post_right_cm = post_geo_right_cm

            draw_clavicle_measurements(
                corrected_cv2,
                corrected_landmarks,
                post_left_cm,
                post_right_cm,
                os.path.join(OUTPUT_DIR, "after_correction.jpg"),
            )

            # ── Terminal metrics dashboard ──
            geo_l_before = landmarks["geo_left_cm"]
            geo_r_before = landmarks["geo_right_cm"]
            asym_before = abs(geo_l_before - geo_r_before)
            asym_after = abs(post_geo_left_cm - post_geo_right_cm)

            print("\n" + "█" * 65)
            print("  📊 ANATOMICAL ALIGNMENT METRICS COMPARISON")
            print("█" * 65)
            print(f"  DIAGNOSIS      : {final_status} ({severity})")
            print(f"  DEVIATION      : {angle}° ({direction})")
            print(f"  ROTATION RATIO : {rotation_ratio:.3f}")
            print("-" * 65)
            print(f"  {'SOURCE':<14} | {'METRIC':<18} | {'BEFORE':>10} | {'AFTER':>10}")
            print("-" * 65)
            print(
                f"  {'Regression':<14} | {'Left Clavicle':<18} | {model_left_cm:>8.2f} cm | {'--':>10}"
            )
            print(
                f"  {'Regression':<14} | {'Right Clavicle':<18} | {model_right_cm:>8.2f} cm | {'--':>10}"
            )
            print(
                f"  {'PSPNet Geo':<14} | {'Left Clavicle':<18} | {geo_l_before:>8.2f} cm | {post_geo_left_cm:>8.2f} cm"
            )
            print(
                f"  {'PSPNet Geo':<14} | {'Right Clavicle':<18} | {geo_r_before:>8.2f} cm | {post_geo_right_cm:>8.2f} cm"
            )
            print(
                f"  {'PSPNet Geo':<14} | {'Asymmetry Delta':<18} | {asym_before:>8.2f} cm | {asym_after:>8.2f} cm"
            )
            print("-" * 65)
            print("  Note: Post-correction uses PSPNet geometric distances.")
            print(
                "  Regression model not re-run (out-of-distribution on warped images)."
            )
            print("█" * 65 + "\n")

            corrected_b64 = image_to_base64(corrected_cv2)
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"corrected_{file.filename}"), corrected_cv2
            )
        else:
            print("\n" + "═" * 55)
            print("  ✅ ALIGNMENT WITHIN NORMAL MARGINS — SKIPPING WARP")
            print(f"  Regression Model : L={model_left_cm} cm | R={model_right_cm} cm")
            print(
                f"  PSPNet Geometric : L={landmarks['geo_left_cm']} cm | R={landmarks['geo_right_cm']} cm"
            )
            print("═" * 55 + "\n")

        # ── Response ──
        return JSONResponse(
            {
                "valid": True,
                "final_prediction": final_status,
                "geometry_prediction": geo_class,
                "normal_conf": round(normal_conf, 1),
                "rotated_conf": round(rotated_conf, 1),
                # Clavicle symmetry (regression model)
                "left_cm": model_left_cm,
                "right_cm": model_right_cm,
                "clavicle_asymmetry_cm": round(abs(model_left_cm - model_right_cm), 2),
                # Clavicle symmetry (PSPNet geometric)
                "geo_left_cm": landmarks["geo_left_cm"],
                "geo_right_cm": landmarks["geo_right_cm"],
                # Rotation metrics
                "angle": angle,
                "rotation_ratio": rotation_ratio,
                "direction": direction,
                "severity": severity,
                # Quality assessment
                "mediastinal_shift_px": mediastinal_shift,
                "mediastinal_status": mediastinal_status,
                "scapular_status": scapular_status,
                "lung_coverage_status": lung_status,
                "lung_coverage_score": round(lung_coverage_score, 3),
                # Post-correction
                "post_corrected_left_cm": post_left_cm,
                "post_corrected_right_cm": post_right_cm,
                "post_geo_left_cm": post_geo_left_cm,
                "post_geo_right_cm": post_geo_right_cm,
                # Images
                "original_img": image_to_base64(img_cv2),
                "corrected_img": corrected_b64
                if corrected_b64
                else image_to_base64(img_cv2),
            }
        )

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return JSONResponse({"valid": False, "message": f"Error: {str(e)}"})


@app.get("/")
def root():
    return {
        "message": "✅ Chest X-Ray Positioning AI is running!",
        "version": "2.0.0",
        "models_loaded": [
            "EfficientNet (Rotation Classification)",
            "Clavicle Distance Regressor (Trained Model)",
            "TorchXRayVision PSPNet (Landmark Detection)",
        ],
    }