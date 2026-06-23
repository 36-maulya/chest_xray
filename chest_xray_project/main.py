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
# CONFIGURATION & TOGGLES
# ─────────────────────────────────────────────
EFFICIENTNET_PATH = r"models\efficientnet_xray_v2.pth"
CLAVICLE_MODEL_PATH = r"models\clavicle_distance_model.pth"

OUTPUT_DIR = r"outputs"
IMG_SIZE = 224
CLASS_NAMES = ['normal', 'rotated']
PIXEL_SPACING_MM = 0.912

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# CLAVICLE DISTANCE REGRESSION MODEL
# ─────────────────────────────────────────────
class ClavicleDistanceRegressor(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0
        )

        self.head = nn.Sequential(
            nn.Dropout(0.3),       # 0
            nn.Linear(1280, 512),  # 1
            nn.ReLU(),             # 2
            nn.BatchNorm1d(512),   # 3
            nn.Dropout(0.2),       # 4
            nn.Linear(512, 128),   # 5
            nn.ReLU(),             # 6
            nn.Identity(),         # 7
            nn.Linear(128, 2)      # 8
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
# ─────────────────────────────────────────────
# MODEL INITIALIZATIONS
# ─────────────────────────────────────────────
efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
efficientnet.eval()
efficientnet.to(device)
print("✅ EfficientNet loaded! (Rotation Detection)")
# ─────────────────────────────────────────────
# LOAD CLAVICLE DISTANCE MODEL
# ─────────────────────────────────────────────
print("Loading clavicle distance model...")

clavicle_model = ClavicleDistanceRegressor(pretrained=False)

checkpoint = torch.load(CLAVICLE_MODEL_PATH, map_location=device)

print("\n===== CHECKPOINT TYPE =====")
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\n===== CHECKPOINT KEYS =====")
    print(checkpoint.keys())

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

print("\n===== HEAD KEYS =====")
for k in state_dict.keys():
    if k.startswith("head"):
        print(k)

missing, unexpected = clavicle_model.load_state_dict(
    state_dict,
    strict=False
)

print("\n===== LOAD RESULT =====")
print("Missing:", missing)
print("Unexpected:", unexpected)

clavicle_model.eval()
clavicle_model.to(device)

print("✅ Clavicle Distance Model loaded!")

print("Loading TorchXRayVision...")
xrv_model = xrv.baseline_models.chestx_det.PSPNet()
xrv_model.eval()
print("✅ TorchXRayVision loaded! (Landmark Detection)")

# ─────────────────────────────────────────────
# PREPROCESSING & HELPER UTILITIES
# ─────────────────────────────────────────────
efficientnet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def is_valid_xray(img: Image.Image) -> bool:
    gray = np.array(img.convert("L"))
    mean_brightness = gray.mean()
    std_brightness = gray.std()
    return 20 < mean_brightness < 230 and std_brightness >= 15

def image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def get_medial_clavicle(lx, ly, spine_x, side="left"):
    if side == "left":
        medial_x = lx + (spine_x - lx) * 0.85
    else:
        medial_x = lx - (lx - spine_x) * 0.85
    return int(medial_x), int(ly)

# ─────────────────────────────────────────────
# ANALYSIS & PREDICTION ENGINES
# ─────────────────────────────────────────────
def predict_image(img: Image.Image):
    tensor = efficientnet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = efficientnet(tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs.max().item() * 100
        pred_idx = probs.argmax().item()
        return CLASS_NAMES[pred_idx], confidence, probs[0][0].item() * 100, probs[0][1].item() * 100
def predict_clavicle_distances(img: Image.Image):
    """
    Returns:
        left_cm, right_cm
    """

    tensor = efficientnet_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = clavicle_model(tensor).cpu().numpy()[0]

    left_cm = round(max(0.1, float(pred[0])), 2)
    right_cm = round(max(0.1, float(pred[1])), 2)

    return left_cm, right_cm

def detect_landmarks_xrv(img_cv2, is_corrected=False):
    try:
        h_orig, w_orig = img_cv2.shape[:2]
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        img_xrv = (gray.astype(np.float32) / 255.0) * 2048 - 1024
        img_xrv = skimage.transform.resize(img_xrv, (512, 512), anti_aliasing=True)
        img_tensor = torch.from_numpy(img_xrv).unsqueeze(0).unsqueeze(0).float().to(device)
        
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
                ll = label.lower()
                x, y = get_pos(landmarks_maps[i], w_orig, h_orig)
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

            spine_x = landmarks.get("spine_x", w_orig // 2)
            clav_lx = landmarks.get("clav_lx", int(w_orig * 0.3))
            clav_rx = landmarks.get("clav_rx", int(w_orig * 0.7))
            
            if clav_lx > clav_rx:
                clav_lx, clav_rx = clav_rx, clav_lx
                landmarks["clav_ly"], landmarks["clav_ry"] = landmarks.get("clav_ry", int(h_orig * 0.2)), landmarks.get("clav_ly", int(h_orig * 0.2))
            
            if clav_lx > spine_x: clav_lx = int(w_orig * 0.35)
            if clav_rx < spine_x: clav_rx = int(w_orig * 0.65)
            
            mediastinum_y = landmarks.get("mediastinum_y", int(h_orig * 0.4))
            
            # ────────────────────────────────────────────────────────
            # 🖥️ DYNAMIC TERMINAL HEADING
            # ────────────────────────────────────────────────────────
            label_status = "✨ POST-CORRECTION" if is_corrected else "📁 RAW ORIGINAL"
            print("\n" + "="*40)
            print(f"    📍 DETECTED LANDMARKS ({label_status})      ")
            print("="*40)
            print(f"Spine Midpoint (X, Y) : ({spine_x}, {landmarks.get('spine_y')})")
            print(f"Left Clavicle (X, Y)  : ({clav_lx}, {landmarks.get('clav_ly')})")
            print(f"Right Clavicle (X, Y) : ({clav_rx}, {landmarks.get('clav_ry')})")
            print(f"Mediastinum (X, Y)    : ({landmarks.get('mediastinum_x')}, {mediastinum_y})")
            print("="*40 + "\n")
            # ────────────────────────────────────────────────────────
            
            return {
                "spine_x": spine_x,
                "spine_y": landmarks.get("spine_y", h_orig // 2),
                "mediastinum_x": landmarks.get("mediastinum_x", spine_x),
                "mediastinum_y": mediastinum_y,
                "search_x1": max(0, spine_x - 80),
                "search_x2": min(w_orig, spine_x + 80),
                "search_y1": max(0, mediastinum_y - 60),
                "search_y2": min(h_orig, mediastinum_y + 60),
                "left_scapula_x": landmarks.get("left_scapula_x"),
                "left_scapula_y": landmarks.get("left_scapula_y"),
                "right_scapula_x": landmarks.get("right_scapula_x"),
                "right_scapula_y": landmarks.get("right_scapula_y"),
                "left_lung_y": landmarks.get("left_lung_y"),
                "right_lung_y": landmarks.get("right_lung_y"),
                "left_lung_x": landmarks.get("left_lung_x"),
                "right_lung_x": landmarks.get("right_lung_x"),
                "spine_top_y": int(h_orig * 0.1),
                "spine_bottom_y": int(h_orig * 0.85),
                "clavicle_left_x": clav_lx,
                "clavicle_left_y": landmarks.get("clav_ly", int(h_orig * 0.2)),
                "clavicle_right_x": clav_rx,
                "clavicle_right_y": landmarks.get("clav_ry", int(h_orig * 0.2)),
                "image_w": w_orig,
                "image_h": h_orig,
            }
    except Exception as e:
        print("XRV error:", e)
        return None
def draw_clavicle_measurements(img, landmarks, save_path):
    vis = img.copy()

    spine_x = landmarks["spine_x"]

    lx = landmarks["clavicle_left_x"]
    ly = landmarks["clavicle_left_y"]

    rx = landmarks["clavicle_right_x"]
    ry = landmarks["clavicle_right_y"]

    # medial points
    medial_lx, _ = get_medial_clavicle(lx, ly, spine_x, "left")
    medial_rx, _ = get_medial_clavicle(rx, ry, spine_x, "right")

    left_px = abs(spine_x - medial_lx)
    right_px = abs(medial_rx - spine_x)

    left_cm = round((left_px * PIXEL_SPACING_MM) / 10, 2)
    right_cm = round((right_px * PIXEL_SPACING_MM) / 10, 2)

    top_y = min(ly, ry) - 20

    # vertical spine line
    cv2.line(
        vis,
        (spine_x, 50),
        (spine_x, vis.shape[0]-50),
        (0,0,0),
        2
    )

    # horizontal left
    cv2.line(
        vis,
        (medial_lx, ly),
        (spine_x, ly),
        (0,0,0),
        2
    )

    # horizontal right
    cv2.line(
        vis,
        (spine_x, ry),
        (medial_rx, ry),
        (0,0,0),
        2
    )

    cv2.putText(
        vis,
        f"{left_cm:.2f} cm",
        (medial_lx - 120, ly + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,0),
        2
    )

    cv2.putText(
        vis,
        f"{right_cm:.2f} cm",
        (spine_x + 30, ry + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,0),
        2
    )

    cv2.imwrite(save_path, vis)

    return vis    

def visualize_landmarks(img_cv2, landmarks):
    vis = img_cv2.copy()
    spine_x = landmarks["spine_x"]
    lx, ly = landmarks["clavicle_left_x"], landmarks["clavicle_left_y"]
    rx, ry = landmarks["clavicle_right_x"], landmarks["clavicle_right_y"]
    
    medial_left_x = int(lx + (spine_x - lx) * 0.90)
    medial_right_x = int(rx - (rx - spine_x) * 0.90)
    avg_clav_y = int((ly + ry) / 2)

    cv2.line(vis, (spine_x, ly), (lx, ly), (0, 255, 255), 2)
    cv2.line(vis, (spine_x, ry), (rx, ry), (255, 255, 0), 2)

    points = [
        ("Spine", spine_x, img_cv2.shape[0] // 2),
        ("L-Clav", lx, ly),
        ("R-Clav", rx, ry),
        ("L-Med", medial_left_x, avg_clav_y),
        ("R-Med", medial_right_x, avg_clav_y)
    ]
    
    for name, x, y in points:
        cv2.circle(vis, (x, y), 8, (0, 255, 0) if "Med" not in name else (0, 255, 255), -1)
        cv2.putText(vis, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return vis

def calculate_correction_angle(lm):
    spine_x = lm["spine_x"]

    lx, ly = lm["clavicle_left_x"], lm["clavicle_left_y"]
    rx, ry = lm["clavicle_right_x"], lm["clavicle_right_y"]

    medial_lx = int(lx + (spine_x - lx) * 0.85)
    medial_rx = int(rx - (rx - spine_x) * 0.85)

    left_dist = abs(spine_x - medial_lx)
    right_dist = abs(medial_rx - spine_x)

    left_cm = round((left_dist * PIXEL_SPACING_MM) / 10, 2)
    right_cm = round((right_dist * PIXEL_SPACING_MM) / 10, 2)

    rotation_ratio = abs(left_dist - right_dist) / max(1, (left_dist + right_dist))
    diff = right_dist - left_dist

    if rotation_ratio < 0.05:
        return 0.0, left_cm, right_cm, rotation_ratio, "None", "None"

    angle = math.degrees(math.atan2(ly - ry, rx - lx))
    angle = max(min(round(angle, 1), 15.0), -15.0)

    direction = "Left rotation" if diff > 0 else "Right rotation"

    if rotation_ratio < 0.10:
        severity = "Mild"
    elif rotation_ratio < 0.20:
        severity = "Moderate"
    else:
        severity = "Severe"

    return angle, left_cm, right_cm, rotation_ratio, direction, severity

def calculate_mediastinal_shift(lm):
    spine_x = lm["spine_x"]
    mediastinum_x = lm.get("mediastinum_x")
    if mediastinum_x is None: return 0, "Unknown"
    
    shift = abs(mediastinum_x - spine_x)
    status = "Normal" if shift < 20 else "Mild Shift" if shift < 40 else "Significant Shift"
    return shift, status

def calculate_scapular_position(lm):
    ls_x, rs_x = lm.get("left_scapula_x"), lm.get("right_scapula_x")
    ll_x, rl_x = lm.get("left_lung_x"), lm.get("right_lung_x")
    if None in [ls_x, rs_x, ll_x, rl_x]: return "Unknown"
    
    avg_overlap = (abs(ls_x - ll_x) + abs(rs_x - rl_x)) / 2
    return "Good Retraction" if avg_overlap > 100 else "Acceptable" if avg_overlap > 40 else "Scapula Overlapping Lung Field"

def geometry_decision(angle, rotation_ratio):
    return "rotated" if rotation_ratio > 0.15 else "normal"

# ─────────────────────────────────────────────
# GEOMETRIC WARPING ALIGNMENT PIPELINE
# ─────────────────────────────────────────────
def anatomical_warp_correction(img_cv2, landmarks):
    h, w = img_cv2.shape[:2]
    spine_x = landmarks["spine_x"]
    lx, rx = landmarks["clavicle_left_x"], landmarks["clavicle_right_x"]

    left_dist = abs(spine_x - lx)
    right_dist = abs(rx - spine_x)
    rotation_ratio = abs(left_dist - right_dist) / max(1, (left_dist + right_dist))

    correction_strength = 0.35 if rotation_ratio > 0.30 else 0.25 if rotation_ratio > 0.20 else 0.15
    
    target_left = left_dist + (((left_dist + right_dist)/2) - left_dist) * correction_strength
    target_right = right_dist + (((left_dist + right_dist)/2) - right_dist) * correction_strength    
    
    scale_l = np.clip(target_left / max(1, left_dist), 0.9, 1.2)
    scale_r = np.clip(target_right / max(1, right_dist), 0.9, 1.2)

    map_x, map_y = np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        map_y[y, :] = y
        for x in range(w):
            if x < spine_x:
                map_x[y, x] = spine_x - (spine_x - x) / scale_l
            else:
                map_x[y, x] = spine_x + (x - spine_x) / scale_r
    
    corrected = cv2.remap(img_cv2, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_corrected.png"), corrected)
    return corrected

# ─────────────────────────────────────────────
# MAIN ROUTE ENDPOINT
# ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if not is_valid_xray(img_pil):
            return JSONResponse({"valid": False, "message": "Please upload a valid chest X-ray image."})
        
        cnn_class, cnn_conf, normal_conf, rotated_conf = predict_image(img_pil)
        model_left_cm, model_right_cm = predict_clavicle_distances(img_pil)
        left_cm = model_left_cm
        right_cm = model_right_cm

        delta_cm = abs(left_cm - right_cm)

        rotation_ratio = delta_cm / max(left_cm, right_cm)
        print(
            f"\n[Regression Model] "
            f"Left: {model_left_cm} cm | "
            f"Right: {model_right_cm} cm"
        )
        cnn_signal = 1 if cnn_class == "rotated" else 0
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        landmarks = detect_landmarks_xrv(img_cv2)
        
        h, w = img_cv2.shape[:2]
        if landmarks is None:
            landmarks = {
                "spine_x": w // 2, "spine_top_y": int(h * 0.1), "spine_bottom_y": int(h * 0.85),
                "clavicle_left_x": int(w * 0.28), "clavicle_left_y": int(h * 0.22),
                "clavicle_right_x": int(w * 0.72), "clavicle_right_y": int(h * 0.22),
                "image_w": w, "image_h": h,
            }
        
        debug_img = visualize_landmarks(img_cv2, landmarks)
        draw_clavicle_measurements(
            img_cv2,
            landmarks,
            os.path.join(OUTPUT_DIR, "before_correction.jpg")
        )
        cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_landmarks.jpg"), debug_img)
        
        angle, left_cm, right_cm, rotation_ratio, direction, severity = calculate_correction_angle(landmarks)
        mediastinal_shift, mediastinal_status = calculate_mediastinal_shift(landmarks)
        scapular_status = calculate_scapular_position(landmarks)
        
        geo_class = geometry_decision(angle, rotation_ratio)
        final_status = "ROTATED" if geo_class == "rotated" or (cnn_signal == 1 and cnn_conf > 70) else "NORMAL"
        
        corrected_b64 = None
        post_left_cm, post_right_cm = left_cm, right_cm 
        
        if final_status == "ROTATED" and rotation_ratio > 0.15:
            # Run the geometric transformation 
            corrected_cv2 = anatomical_warp_correction(img_cv2, landmarks)
            corrected_landmarks = detect_landmarks_xrv(corrected_cv2, is_corrected=True) or landmarks
            corrected_landmarks["spine_x"] = landmarks["spine_x"]
            draw_clavicle_measurements(
            corrected_cv2,
            corrected_landmarks,
            os.path.join(OUTPUT_DIR, "after_correction.jpg")
        )

            # Calculate Post-Correction Metrics
            spine_x_post = corrected_landmarks["spine_x"]
            lx_post, _ = get_medial_clavicle(corrected_landmarks["clavicle_left_x"], corrected_landmarks["clavicle_left_y"], spine_x_post, "left")
            rx_post, _ = get_medial_clavicle(corrected_landmarks["clavicle_right_x"], corrected_landmarks["clavicle_right_y"], spine_x_post, "right")
            
            post_left_cm = round((abs(spine_x_post - lx_post) * PIXEL_SPACING_MM) / 10, 2)
            post_right_cm = round((abs(rx_post - spine_x_post) * PIXEL_SPACING_MM) / 10, 2)
            
            # ────────────────────────────────────────────────────────
            # 🖥️ TERMINAL METRICS DASHBOARD (BEFORE VS AFTER)
            # ────────────────────────────────────────────────────────
            print("\n" + "█"*50)
            print(" 📊 ANATOMICAL ALIGNMENT METRICS COMPARISON")
            print("█"*50)
            print(f"  DIAGNOSIS    : {final_status} ({severity})")
            print(f"  DEVIATION    : {angle}° ({direction})")
            print(f"  ROTATION RATIO: {rotation_ratio:.3f}")
            print("-" * 50)
            print(f"  METRIC          │  BEFORE CORRECTION  │  AFTER CORRECTION")
            print("-" * 50)
            print(f"  Left Clavicle   │  {left_cm:<16} cm │  {post_left_cm:<15} cm")
            print(f"  Right Clavicle  │  {right_cm:<16} cm │  {post_right_cm:<15} cm")
            print(f"  Asymmetry Delta │  {round(abs(left_cm - right_cm), 2):<16} cm │  {round(abs(post_left_cm - post_right_cm), 2):<15} cm")
            print("█"*50 + "\n")
            # ────────────────────────────────────────────────────────

            corrected_b64 = image_to_base64(corrected_cv2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"corrected_{file.filename}"), corrected_cv2)
        else:
            print("\n" + "═"*50)
            print(" ✅ ALIGNMENT WITHIN NORMAL MARGINS — SKIPPING WARP")
            print(f"  Left Clavicle: {left_cm} cm  |  Right Clavicle: {right_cm} cm")
            print("═"*50 + "\n")
            
        return JSONResponse({
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
            "left_cm": left_cm,
            "right_cm": right_cm,
            "post_corrected_left_cm": post_left_cm,
            "post_corrected_right_cm": post_right_cm,
            "original_img": image_to_base64(img_cv2),
            "corrected_img": corrected_b64 if corrected_b64 else image_to_base64(img_cv2),
        })
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return JSONResponse({"valid": False, "message": f"Error: {str(e)}"})

@app.get("/")
def root():
    return {"message": "✅ Chest X-Ray Analysis API is running!"}