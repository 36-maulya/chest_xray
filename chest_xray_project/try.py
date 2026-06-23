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


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
EFFICIENTNET_PATH = r"models\efficientnet_xray_v2.pth"
CLAVICLE_MODEL_PATH = r"models\clavicle_distance_model.pth"
OUTPUT_DIR = r"outputs"
IMG_SIZE = 224
CLASS_NAMES = ["normal", "rotated"]
PIXEL_SPACING_MM = 0.912

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# CLAVICLE DISTANCE REGRESSOR (trained in Colab)
# ─────────────────────────────────────────────
class ClavicleDistanceRegressor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────
# MODEL INITIALIZATIONS
# ─────────────────────────────────────────────

# 1) EfficientNet rotation classifier
efficientnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
efficientnet.eval()
efficientnet.to(device)
print("Loaded EfficientNet (Rotation Classification)")

# 2) Clavicle distance regressor
clavicle_model = ClavicleDistanceRegressor(pretrained=False)
checkpoint = torch.load(CLAVICLE_MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    clavicle_model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Clavicle model loaded! "
        f"(Val MAE - Left: {checkpoint.get('val_mae_left', 'N/A')}, "
        f"Right: {checkpoint.get('val_mae_right', 'N/A')})"
    )
else:
    clavicle_model.load_state_dict(checkpoint)
    print("Clavicle model loaded! (raw state_dict)")
clavicle_model.eval()
clavicle_model.to(device)

# 3) TorchXRayVision PSPNet (spine + clavicle landmark detection)
print("Loading TorchXRayVision...")
xrv_model = xrv.baseline_models.chestx_det.PSPNet()
xrv_model.eval()
print("Loaded TorchXRayVision PSPNet (Spine + Clavicle Detection)")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
efficientnet_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def is_valid_xray(img: Image.Image) -> bool:
    gray = np.array(img.convert("L"))
    return 20 < gray.mean() < 230 and gray.std() >= 15


def image_to_base64(img_array):
    _, buffer = cv2.imencode(".jpg", img_array)
    return base64.b64encode(buffer).decode("utf-8")


def px_to_cm(px):
    return round((px * PIXEL_SPACING_MM) / 10, 2)


# ─────────────────────────────────────────────
# PREDICTION: Rotation classifier
# ─────────────────────────────────────────────
def predict_rotation(img: Image.Image):
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


# ─────────────────────────────────────────────
# PREDICTION: Clavicle distances (regression model)
# ─────────────────────────────────────────────
def predict_clavicle_distances(img: Image.Image):
    """
    Predicts left_cm and right_cm from the trained regression model.
    These are used for the rotation ratio / decision logic.
    NOT used for visualization positions (those come from PSPNet).
    """
    tensor = efficientnet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = clavicle_model(tensor).cpu().numpy()[0]
        left_cm = round(max(0.1, pred[0]), 2)
        right_cm = round(max(0.1, pred[1]), 2)
        return left_cm, right_cm


# ─────────────────────────────────────────────
# PSPNet LANDMARK DETECTION
# ─────────────────────────────────────────────
def detect_landmarks_pspnet(img_cv2, is_corrected=False):
    """
    Uses TorchXRayVision PSPNet to detect REAL anatomical landmarks:
      - spine_x, spine_y
      - clav_lx, clav_ly  (left clavicle - REAL detected position)
      - clav_rx, clav_ry  (right clavicle - REAL detected position)
      - mediastinum_x, mediastinum_y

    Returns dict with all landmark coordinates.
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

            landmarks = {}
            for i, label in enumerate(targets):
                hmap = cv2.resize(landmarks_maps[i], (w_orig, h_orig))
                y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
                ll = label.lower()

                if "spine" in ll:
                    landmarks["spine_x"], landmarks["spine_y"] = int(x), int(y)
                elif "left clavicle" in ll:
                    landmarks["clav_lx"], landmarks["clav_ly"] = int(x), int(y)
                elif "right clavicle" in ll:
                    landmarks["clav_rx"], landmarks["clav_ry"] = int(x), int(y)
                elif "mediastinum" in ll:
                    landmarks["mediastinum_x"], landmarks["mediastinum_y"] = (
                        int(x),
                        int(y),
                    )
                elif "left scapula" in ll:
                    landmarks["left_scapula_x"], landmarks["left_scapula_y"] = (
                        int(x),
                        int(y),
                    )
                elif "right scapula" in ll:
                    landmarks["right_scapula_x"], landmarks["right_scapula_y"] = (
                        int(x),
                        int(y),
                    )
                elif "left lung" in ll:
                    landmarks["left_lung_x"], landmarks["left_lung_y"] = int(x), int(y)
                elif "right lung" in ll:
                    landmarks["right_lung_x"], landmarks["right_lung_y"] = (
                        int(x),
                        int(y),
                    )

            # Ensure spine exists
            spine_x = landmarks.get("spine_x", w_orig // 2)
            spine_y = landmarks.get("spine_y", h_orig // 2)

            # Ensure clavicle positions exist with fallbacks
            clav_lx = landmarks.get("clav_lx", int(w_orig * 0.3))
            clav_ly = landmarks.get("clav_ly", int(h_orig * 0.2))
            clav_rx = landmarks.get("clav_rx", int(w_orig * 0.7))
            clav_ry = landmarks.get("clav_ry", int(h_orig * 0.2))

            # Sanity: left clavicle should be left of right clavicle
            if clav_lx > clav_rx:
                clav_lx, clav_rx = clav_rx, clav_lx
                clav_ly, clav_ry = clav_ry, clav_ly

            # Sanity: clavicles should not cross spine
            if clav_lx > spine_x:
                clav_lx = int(w_orig * 0.35)
            if clav_rx < spine_x:
                clav_rx = int(w_orig * 0.65)

            # Geometric distances from PSPNet (actual pixel measurements)
            geo_left_px = abs(spine_x - clav_lx)
            geo_right_px = abs(clav_rx - spine_x)
            geo_left_cm = px_to_cm(geo_left_px)
            geo_right_cm = px_to_cm(geo_right_px)

            label = "POST-CORRECTION" if is_corrected else "RAW ORIGINAL"
            print(f"\n{'=' * 50}")
            print(f"  PSPNet LANDMARKS ({label})")
            print(f"{'=' * 50}")
            print(f"  Spine       (X,Y) : ({spine_x}, {spine_y})")
            print(f"  L-Clavicle  (X,Y) : ({clav_lx}, {clav_ly})")
            print(f"  R-Clavicle  (X,Y) : ({clav_rx}, {clav_ry})")
            print(
                f"  Mediastinum (X,Y) : ({landmarks.get('mediastinum_x', 'N/A')}, "
                f"{landmarks.get('mediastinum_y', 'N/A')})"
            )
            print(f"  ---")
            print(f"  Geo Left Dist  : {geo_left_px}px = {geo_left_cm} cm")
            print(f"  Geo Right Dist : {geo_right_px}px = {geo_right_cm} cm")
            print(f"{'=' * 50}\n")

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
        print("PSPNet error:", e)
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


# ─────────────────────────────────────────────
# VISUALIZATION: Uses REAL PSPNet clavicle positions
# ─────────────────────────────────────────────
def draw_clavicle_measurements(
    img_cv2, landmarks, model_left_cm, model_right_cm, save_path
):
    """
    Draws measurement lines between REAL detected landmarks.

    Lines go from:
      spine -> PSPNet's detected left clavicle head
      spine -> PSPNet's detected right clavicle head

    Labels show the REGRESSION MODEL's predicted cm values.
    The dots on the image are at REAL PSPNet-detected positions.
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

    # Horizontal lines: spine -> REAL clavicle head (PSPNet detected)
    cv2.line(vis, (clav_lx, clav_ly), (spine_x, clav_ly), (0, 0, 0), 2)
    cv2.line(vis, (spine_x, clav_ry), (clav_rx, clav_ry), (0, 0, 0), 2)

    # Labels: show regression model's predicted distances
    cv2.putText(
        vis,
        f"{model_left_cm:.2f} cm",
        (min(clav_lx, spine_x) - 120, clav_ly + 10),
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


# ─────────────────────────────────────────────
# METRICS: Uses PSPNet geometric distances
# ─────────────────────────────────────────────
def calculate_correction_metrics(landmarks, model_left_cm, model_right_cm):
    """
    Calculates rotation metrics.

    Uses REGRESSION MODEL distances for the primary rotation_ratio
    (since those are trained on ground truth annotations).

    Uses PSPNet clavicle X positions for the angle calculation
    (since those are real detected anatomical points).
    """
    spine_x = landmarks["spine_x"]
    clav_lx = landmarks["clav_lx"]
    clav_rx = landmarks["clav_rx"]
    clav_ly = landmarks["clav_ly"]
    clav_ry = landmarks["clav_ry"]

    # Rotation ratio from regression model (trained on ground truth)
    left_px = (model_left_cm * 10) / PIXEL_SPACING_MM
    right_px = (model_right_cm * 10) / PIXEL_SPACING_MM
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
    spine_x = landmarks["spine_x"]
    mediastinum_x = landmarks.get("mediastinum_x", spine_x)
    shift = abs(mediastinum_x - spine_x)
    status = (
        "Normal" if shift < 20 else "Mild Shift" if shift < 40 else "Significant Shift"
    )
    return shift, status


def calculate_scapular_position(landmarks):
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


# ─────────────────────────────────────────────
# GEOMETRIC WARP CORRECTION
# ─────────────────────────────────────────────
def anatomical_warp_correction(img_cv2, landmarks, model_left_cm, model_right_cm):
    """
    Warp image to correct rotation.
    Uses regression model distances for determining warp scale factors.
    """
    h, w = img_cv2.shape[:2]
    spine_x = landmarks["spine_x"]

    left_px = (model_left_cm * 10) / PIXEL_SPACING_MM
    right_px = (model_right_cm * 10) / PIXEL_SPACING_MM

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


# ─────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        if not is_valid_xray(img_pil):
            return JSONResponse(
                {"valid": False, "message": "Please upload a valid chest X-ray image."}
            )

        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # ── Step 1: CNN rotation classification ──
        cnn_class, cnn_conf, normal_conf, rotated_conf = predict_rotation(img_pil)
        cnn_signal = 1 if cnn_class == "rotated" else 0

        # ── Step 2: Regression model predicts distances (cm) ──
        # These are trained on ground truth annotations.
        # Used for: rotation_ratio, decision logic, label text.
        model_left_cm, model_right_cm = predict_clavicle_distances(img_pil)
        print(
            f"\n[Regression Model] Left: {model_left_cm} cm | Right: {model_right_cm} cm"
        )

        # ── Step 3: PSPNet detects REAL anatomical landmarks ──
        # Returns actual pixel positions of spine, clavicle heads, etc.
        # Used for: visualization, angle calculation, geometric distances.
        landmarks = detect_landmarks_pspnet(img_cv2)

        # ── Step 4: Calculate rotation metrics ──
        angle, rotation_ratio, direction, severity = calculate_correction_metrics(
            landmarks, model_left_cm, model_right_cm
        )
        mediastinal_shift, mediastinal_status = calculate_mediastinal_shift(landmarks)
        scapular_status = calculate_scapular_position(landmarks)

        # ── Step 5: Final decision ──
        geo_class = "rotated" if rotation_ratio > 0.15 else "normal"
        final_status = (
            "ROTATED"
            if geo_class == "rotated" or (cnn_signal == 1 and cnn_conf > 70)
            else "NORMAL"
        )

        # ── Step 6: Visualization ──
        # All dots and lines are at REAL PSPNet-detected positions.
        # Labels show regression model's cm predictions.
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

        # ── Step 7: Correction (if needed) ──
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
            # Do NOT re-run the regression model here because it was trained
            # on original (un-warped) X-rays. Running it on warped images
            # would be an out-of-distribution prediction.
            corrected_landmarks = detect_landmarks_pspnet(
                corrected_cv2, is_corrected=True
            )

            post_geo_left_cm = corrected_landmarks["geo_left_cm"]
            post_geo_right_cm = corrected_landmarks["geo_right_cm"]

            # For the response, also use geometric distances as post-correction values
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
            print("\n" + "=" * 65)
            print("  ANATOMICAL ALIGNMENT METRICS COMPARISON")
            print("=" * 65)
            print(f"  DIAGNOSIS      : {final_status} ({severity})")
            print(f"  DEVIATION      : {angle} deg ({direction})")
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
                f"  {'PSPNet Geo':<14} | {'Left Clavicle':<18} | {landmarks['geo_left_cm']:>8.2f} cm | {post_geo_left_cm:>8.2f} cm"
            )
            print(
                f"  {'PSPNet Geo':<14} | {'Right Clavicle':<18} | {landmarks['geo_right_cm']:>8.2f} cm | {post_geo_right_cm:>8.2f} cm"
            )
            print(
                f"  {'PSPNet Geo':<14} | {'Asymmetry Delta':<18} | {abs(landmarks['geo_left_cm'] - landmarks['geo_right_cm']):>8.2f} cm | {abs(post_geo_left_cm - post_geo_right_cm):>8.2f} cm"
            )
            print("-" * 65)
            print("  Note: Post-correction uses PSPNet geometric distances.")
            print(
                "  Regression model not re-run (out-of-distribution on warped images)."
            )
            print("=" * 65 + "\n")

            corrected_b64 = image_to_base64(corrected_cv2)
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"corrected_{file.filename}"), corrected_cv2
            )
        else:
            print("\n" + "=" * 50)
            print("  ALIGNMENT WITHIN NORMAL MARGINS - SKIPPING WARP")
            print(f"  Regression Model : L={model_left_cm} cm | R={model_right_cm} cm")
            print(
                f"  PSPNet Geometric : L={landmarks['geo_left_cm']} cm | R={landmarks['geo_right_cm']} cm"
            )
            print("=" * 50 + "\n")

        return JSONResponse(
            {
                "valid": True,
                "final_prediction": final_status,
                "geometry_prediction": geo_class,
                "normal_conf": round(normal_conf, 1),
                "rotated_conf": round(rotated_conf, 1),
                "mediastinal_shift": mediastinal_shift,
                "mediastinal_status": mediastinal_status,
                "scapular_status": scapular_status,
                "status": final_status,
                "direction": direction,
                "severity": severity,
                "angle": angle,
                "rotation_ratio": rotation_ratio,
                "left_cm": model_left_cm,
                "right_cm": model_right_cm,
                "geo_left_cm": landmarks["geo_left_cm"],
                "geo_right_cm": landmarks["geo_right_cm"],
                "post_corrected_left_cm": post_left_cm,
                "post_corrected_right_cm": post_right_cm,
                "post_geo_left_cm": post_geo_left_cm,
                "post_geo_right_cm": post_geo_right_cm,
                "original_img": image_to_base64(img_cv2),
                "corrected_img": corrected_b64
                if corrected_b64
                else image_to_base64(img_cv2),
            }
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return JSONResponse({"valid": False, "message": f"Error: {str(e)}"})


@app.get("/")
def root():
    return {"message": "Chest X-Ray Analysis API is running!"}
