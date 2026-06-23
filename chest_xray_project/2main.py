import os
import cv2
import torch
from skimage.transform import PiecewiseAffineTransform
from skimage.transform import warp
import torch.nn as nn
import torch.nn.functional as F
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
# CONFIGURATION & TOGGLES (CRITICAL FOR DEMO)
# ─────────────────────────────────────────────
EFFICIENTNET_PATH = r"models\efficientnet_xray_v2.pth"
CYCLEGAN_PATH = r"models\cyclegan_xray.pth"
OUTPUT_DIR = r"outputs"
IMG_SIZE = 224
CYCLEGAN_SIZE = 256
CLASS_NAMES = ['normal', 'rotated']

# SET TO TRUE ONLY IF YOUR CYCLEGAN IS FULLY RETRAINED TO HIGH RESOLUTION.
# Keeping this False forces the system to use high-fidelity geometric alignment,
# which keeps your demo images beautifully sharp instead of blurry/squished.
USE_EXPERIMENTAL_CYCLEGAN = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# MODEL 1 — EFFICIENTNET
# ─────────────────────────────────────────────
efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
efficientnet.eval()
efficientnet.to(device)
print("✅ EfficientNet loaded! (Rotation Detection)")

# ─────────────────────────────────────────────
# MODEL 2 — TORCHXRAYVISION
# ─────────────────────────────────────────────
print("Loading TorchXRayVision...")
xrv_model = xrv.baseline_models.chestx_det.PSPNet()
xrv_model.eval()
print("✅ TorchXRayVision loaded! (Landmark Detection)")

# ─────────────────────────────────────────────
# MODEL 3 — CYCLEGAN GENERATOR ARCHITECTURE
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, n_res=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(256) for _ in range(n_res)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, 7),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(self.res_blocks(self.encoder(x)))

# Safe Initialization of CycleGAN Weights
cyclegan = Generator().to(device)
if os.path.exists(CYCLEGAN_PATH):
    cyclegan.load_state_dict(torch.load(CYCLEGAN_PATH, map_location=device))
    cyclegan.eval()
    print("✅ CycleGAN weights loaded successfully!")
else:
    cyclegan = None
    print("⚠️ CycleGAN weights missing — falling back to geometric matrix adjustments")

# ─────────────────────────────────────────────
# PREPROCESSING & IMAGE LIFTS
# ─────────────────────────────────────────────
efficientnet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cyclegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CYCLEGAN_SIZE, CYCLEGAN_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def is_valid_xray(img: Image.Image) -> bool:
    gray = np.array(img.convert("L"))
    mean_brightness = gray.mean()
    std_brightness = gray.std()
    return 20 < mean_brightness < 230 and std_brightness >= 15

def predict_image(img: Image.Image):
    tensor = efficientnet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = efficientnet(tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs.max().item() * 100
        pred_idx = probs.argmax().item()
        pred_class = CLASS_NAMES[pred_idx]
        normal_conf = probs[0][0].item() * 100
        rotated_conf = probs[0][1].item() * 100
        return pred_class, confidence, normal_conf, rotated_conf

def detect_landmarks_xrv(img_cv2):
    try:
        h_orig, w_orig = img_cv2.shape[:2]
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        img_xrv = gray.astype(np.float32)
        img_xrv = img_xrv / 255.0 * 2048 - 1024
        img_xrv = skimage.transform.resize(img_xrv, (512, 512), anti_aliasing=True)
        img_tensor = torch.from_numpy(img_xrv).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            output = xrv_model(img_tensor)
            landmarks_maps = output[0].detach().cpu().numpy()
            targets = xrv_model.targets
            
            print("\n===== PSPNet Targets =====")
            for i, label in enumerate(targets):
                print(i, label)
            
            def get_pos(hmap, ow, oh):
                hmap = cv2.resize(hmap, (ow, oh))
                y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
                return int(x), int(y)
            
            print("\n===== ALL LANDMARKS =====")
            for i, label in enumerate(targets):
                x, y = get_pos(landmarks_maps[i], w_orig, h_orig)
                print(f"{label}: ({x},{y})")
            
            spine_x = None
            clav_lx = clav_ly = None
            clav_rx = clav_ry = None
            mediastinum_x = mediastinum_y = None
            left_scapula_x = left_scapula_y = None
            right_scapula_x = right_scapula_y = None
            left_lung_x = left_lung_y = None
            right_lung_x = right_lung_y = None
            
            for i, label in enumerate(targets):
                ll = label.lower()
                x, y = get_pos(landmarks_maps[i], w_orig, h_orig)
                if "spine" in ll:
                    spine_x = x
                    spine_y = y
                elif "left clavicle" in ll:
                    clav_lx, clav_ly = x, y
                elif "right clavicle" in ll:
                    clav_rx, clav_ry = x, y
                elif "mediastinum" in ll:
                    mediastinum_x, mediastinum_y = x, y
                elif "left scapula" in ll:
                    left_scapula_x, left_scapula_y = x, y
                elif "right scapula" in ll:
                    right_scapula_x, right_scapula_y = x, y
                elif "left lung" in ll:
                    left_lung_x, left_lung_y = x, y
                elif "right lung" in ll:
                    right_lung_x, right_lung_y = x, y
            
            # Fix swapped clavicles
            if clav_lx is not None and clav_rx is not None:
                if clav_lx > clav_rx:
                    clav_lx, clav_rx = clav_rx, clav_lx
                    clav_ly, clav_ry = clav_ry, clav_ly

            # Dynamic fallback placement based on frame scale
            spine_x = spine_x or w_orig // 2
            clav_lx = clav_lx or int(w_orig * 0.3)
            clav_rx = clav_rx or int(w_orig * 0.7)
            clav_ly = clav_ly or int(h_orig * 0.2)
            clav_ry = clav_ry or int(h_orig * 0.2)
            
            print("\n===== DETECTED LANDMARKS =====")
            print("Spine X:", spine_x)
            print("Left Clavicle :", clav_lx, clav_ly)
            print("Right Clavicle:", clav_rx, clav_ry)
            
            print("\n===== WARP LANDMARKS =====")
            print("Left Scapula :", left_scapula_x, left_scapula_y)
            print("Right Scapula:", right_scapula_x, right_scapula_y)
            print("Left Lung :", left_lung_x, left_lung_y)
            print("Right Lung :", right_lung_x, right_lung_y)
            print("Spine:", spine_x, spine_y)
            print("Mediastinum:", mediastinum_x, mediastinum_y)
            print("Left Scapula:", left_scapula_x, left_scapula_y)
            print("Right Scapula:", right_scapula_x, right_scapula_y)
            print("Left Lung:", left_lung_x, left_lung_y)
            print("Right Lung:", right_lung_x, right_lung_y)
            return {
                "spine_x": spine_x,
                "spine_y": spine_y,
                "mediastinum_x": mediastinum_x,
                "mediastinum_y": mediastinum_y,
                "left_scapula_x": left_scapula_x,
                "left_scapula_y": left_scapula_y,
                "right_scapula_x": right_scapula_x,
                "right_scapula_y": right_scapula_y,
                "left_lung_y": left_lung_y,
                "right_lung_y": right_lung_y,
                "left_lung_x": left_lung_x,
                "right_lung_x": right_lung_x,
                "spine_top_y": int(h_orig * 0.1),
                "spine_bottom_y": int(h_orig * 0.85),
                "clavicle_left_x": clav_lx,
                "clavicle_left_y": clav_ly,
                "clavicle_right_x": clav_rx,
                "clavicle_right_y": clav_ry,
                "image_w": w_orig,
                "image_h": h_orig,
            }
    except Exception as e:
        print("XRV error:", e)
        return None

def visualize_landmarks(img_cv2, landmarks):
    vis = img_cv2.copy()
    spine_x = landmarks["spine_x"]

    left_medial_x = int(
        landmarks["clavicle_left_x"] +
        (spine_x - landmarks["clavicle_left_x"]) * 0.75
    )

    right_medial_x = int(
        landmarks["clavicle_right_x"] -
        (landmarks["clavicle_right_x"] - spine_x) * 0.75
    )

    left_medial_y = landmarks["clavicle_left_y"]
    right_medial_y = landmarks["clavicle_right_y"]
    points = [
    ("Spine", landmarks["spine_x"], img_cv2.shape[0] // 2),
    #("L-Clav", landmarks["clavicle_left_x"], landmarks["clavicle_left_y"]),
    #("R-Clav", landmarks["clavicle_right_x"], landmarks["clavicle_right_y"]),
    #("L-Med", left_medial_x, left_medial_y),
    #("R-Med", right_medial_x, right_medial_y)
    ]
    # Medical clavicle points
    spine_x = landmarks["spine_x"]

    left_medial_x = int(
        landmarks["clavicle_left_x"] +
        (spine_x - landmarks["clavicle_left_x"]) * 0.75
    )

    right_medial_x = int(
        landmarks["clavicle_right_x"] -
        (landmarks["clavicle_right_x"] - spine_x) * 0.75
    )

    avg_clav_y = int(
        (landmarks["clavicle_left_y"] +
         landmarks["clavicle_right_y"]) / 2
    )

    points.append(("L-Med", left_medial_x, avg_clav_y))
    points.append(("R-Med", right_medial_x, avg_clav_y))

    cv2.line(
        vis,
        (landmarks["spine_x"], landmarks["clavicle_left_y"]),
        (landmarks["clavicle_left_x"], landmarks["clavicle_left_y"]),
        (0,255,255),
        2
    )

    cv2.line(
        vis,
        (landmarks["spine_x"], landmarks["clavicle_right_y"]),
        (landmarks["clavicle_right_x"], landmarks["clavicle_right_y"]),
        (255,255,0),
        2
    )
    
    for name, x, y in points:
        cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.putText(
            vis, name, (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    spine_x = landmarks["spine_x"]

    medial_left_x = int(
        landmarks["clavicle_left_x"] +
        (spine_x - landmarks["clavicle_left_x"]) * 0.90
    )

    medial_right_x = int(
        landmarks["clavicle_right_x"] -
        (landmarks["clavicle_right_x"] - spine_x) * 0.90
    )

    cv2.circle(
        vis,
        (medial_left_x, landmarks["clavicle_left_y"]),
        8,
        (0, 255, 255),
        -1
    )
    cv2.putText(
    vis,
    "L-Med",
    (medial_left_x + 10, landmarks["clavicle_left_y"] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0,255,255),
    2
)

    cv2.circle(
        vis,
        (medial_right_x, landmarks["clavicle_right_y"]),
        8,
        (0, 255, 255),
        -1
    )  
    cv2.putText(
    vis,
    "R-Med",
    (medial_right_x + 10, landmarks["clavicle_right_y"] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0,255,255),
    2
)  
    
    cv2.line(
    vis,
    (left_medial_x, left_medial_y),
    (spine_x, left_medial_y),
    (0, 255, 255),
    2
    )

    cv2.line(
    vis,
    (spine_x, right_medial_y),
    (right_medial_x, right_medial_y),
    (255, 255, 0),
    2
    )
    print("Left clavicle:", landmarks["clavicle_left_x"])
    print("Right clavicle:", landmarks["clavicle_right_x"])
    print("Spine:", landmarks["spine_x"])    
    
    return vis

# ─────────────────────────────────────────────
# CALCULATE CORRECTION ANGLE (MATH FIXED)
# ─────────────────────────────────────────────
def calculate_correction_angle(lm):
    spine_x = lm["spine_x"]
    print("Spine:", lm["spine_x"], lm["spine_y"])
    
    lx, ly = lm["clavicle_left_x"], lm["clavicle_left_y"]
    rx, ry = lm["clavicle_right_x"], lm["clavicle_right_y"]
    # Estimated medial clavicle points
    medial_lx = int(lx + (spine_x - lx) * 0.85)
    medial_rx = int(rx - (rx - spine_x) * 0.85)

    print("Estimated Left Medial:", medial_lx)
    print("Estimated Right Medial:", medial_rx)
    left_dist = abs(spine_x - medial_lx)
    right_dist = abs(medial_rx - spine_x)
    print("Medical Left Distance:", left_dist)
    print("Medical Right Distance:", right_dist)
    PIXEL_SPACING_MM = 0.912

    left_cm = round((left_dist * PIXEL_SPACING_MM) / 10, 2)
    right_cm = round((right_dist * PIXEL_SPACING_MM) / 10, 2)
    print("Left Pixel Distance:", left_dist)
    print("Right Pixel Distance:", right_dist)
    print("Left CM:", left_cm)
    print("Right CM:", right_cm)
    print("Spine X:", spine_x)
    print("Left Clavicle X:", lx)
    print("Right Clavicle X:", rx)
    rotation_ratio = abs(left_dist - right_dist) / (left_dist + right_dist)

    
    
    diff = right_dist - left_dist
    if rotation_ratio < 0.05:
        return 0.0, left_cm, right_cm, rotation_ratio, "None", "None"
    
    # FIX: Account for downward-increasing Y pixel coordinate grid in CV2
    angle = math.degrees(math.atan2(ly - ry, rx - lx))
    angle = max(min(round(angle, 1), 15.0), -15.0)
    
    direction = "Left rotation" if diff > 0 else "Right rotation"
    if rotation_ratio < 0.05:
        severity = "None"
    elif rotation_ratio < 0.10:
        severity = "Mild"
    elif rotation_ratio < 0.20:
        severity = "Moderate"
    else:
        severity = "Severe"
    
    return angle, left_cm, right_cm, rotation_ratio, direction, severity

def calculate_mediastinal_shift(lm):
    spine_x = lm["spine_x"]
    mediastinum_x = lm.get("mediastinum_x")
    
    if mediastinum_x is None:
        return 0, "Unknown"
    
    shift = abs(mediastinum_x - spine_x)
    if shift < 20:
        status = "Normal"
    elif shift < 40:
        status = "Mild Shift"
    else:
        status = "Significant Shift"
    
    return shift, status

def calculate_scapular_position(lm):
    ls_x = lm.get("left_scapula_x")
    rs_x = lm.get("right_scapula_x")
    ll_x = lm.get("left_lung_x")
    rl_x = lm.get("right_lung_x")
    
    if None in [ls_x, rs_x, ll_x, rl_x]:
        return "Unknown"
    
    left_overlap = abs(ls_x - ll_x)
    right_overlap = abs(rs_x - rl_x)
    avg_overlap = (left_overlap + right_overlap) / 2
    
    if avg_overlap > 100:
        return "Good Retraction"
    elif avg_overlap > 40:
        return "Acceptable"
    else:
        return "Scapula Overlapping Lung Field"

# ─────────────────────────────────────────────
# GEOMETRY-BASED DECISION ENGINE (NEW)
# ─────────────────────────────────────────────
def geometry_decision(angle, rotation_ratio):
    """
    Strong rule-based anatomical decision system
    This becomes the PRIMARY decision maker
    """
    if rotation_ratio > 0.15:
        return "rotated"
    else:
        return "normal"

# ─────────────────────────────────────────────
# HIGH-FIDELITY RADIAL TRANSFORM MATRIX
# ─────────────────────────────────────────────
def rotate_spine_centered(img_cv2, angle, landmarks):
    h, w = img_cv2.shape[:2]
    
    # Establish rotation center strictly on the patient's spinal path midline
    center = (landmarks["spine_x"], h // 2)
    
    # Negative angle counteracts the positioning deviation cleanly
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # LANCZOS4 preserves structural crispness perfectly, avoiding blur
    return cv2.warpAffine(
        img_cv2, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )

def anatomical_warp_correction(img_cv2, landmarks):
    h, w = img_cv2.shape[:2]
    
    src = np.array([
        [0, 0],
        [w-1, 0],
        [0, h-1],
        [w-1, h-1],
        [landmarks["clavicle_left_x"], landmarks["clavicle_left_y"]],
        [landmarks["clavicle_right_x"], landmarks["clavicle_right_y"]],
        [landmarks["spine_x"], h//2],
        [landmarks["left_scapula_x"], landmarks["left_scapula_y"]],
        [landmarks["right_scapula_x"], landmarks["right_scapula_y"]],
        [landmarks["left_lung_x"], landmarks["left_lung_y"]],
        [landmarks["right_lung_x"], landmarks["right_lung_y"]],
        [landmarks["mediastinum_x"], landmarks["mediastinum_y"]]
    ], dtype=np.float32)
    
    dst = src.copy()
    avg_clav_y = (landmarks["clavicle_left_y"] + landmarks["clavicle_right_y"]) / 2
    spine_x = landmarks["spine_x"]

    # Estimate medial clavicle ends
    medial_left_x = int(
        landmarks["clavicle_left_x"] +
        (spine_x - landmarks["clavicle_left_x"]) * 0.90
    )

    medial_right_x = int(
        landmarks["clavicle_right_x"] -
        (landmarks["clavicle_right_x"] - spine_x) * 0.90
    )

    left_dist = spine_x - medial_left_x
    right_dist = medial_right_x - spine_x

    print("Medial Left :", medial_left_x)
    print("Medial Right:", medial_right_x)

    print("Left Dist :", left_dist)
    print("Right Dist:", right_dist)

    rotation_ratio = abs(left_dist - right_dist) / (left_dist + right_dist)

    if rotation_ratio > 0.30:
        CORRECTION_FACTOR = 0.60

    elif rotation_ratio > 0.20:
        CORRECTION_FACTOR = 0.40

    else:
        CORRECTION_FACTOR = 0.25

    print("Correction Factor:", CORRECTION_FACTOR)

    diff = right_dist - left_dist

    

    new_left_dist = left_dist + (right_dist - left_dist) * CORRECTION_FACTOR
    new_right_dist = right_dist - (right_dist - left_dist) * CORRECTION_FACTOR

    new_left_x = spine_x - new_left_dist
    new_right_x = spine_x + new_right_dist
    print("Target Dist:", diff)

    print("New Left :", new_left_x)
    print("New Right:", new_right_x)

    
    # Amount to move each medial clavicle
    left_shift = (new_left_x - medial_left_x) * 0.75
    right_shift = (new_right_x - medial_right_x) * 0.75
    MAX_SHIFT = 30

    left_shift = np.clip(left_shift, -MAX_SHIFT, MAX_SHIFT)
    right_shift = np.clip(right_shift, -MAX_SHIFT, MAX_SHIFT)
    print("Left Shift:", left_shift)
    print("Right Shift:", right_shift)
    print("SRC[4] =", src[4])
    print("SRC[5] =", src[5])
    # Small corrections only
    # Move clavicles only by required medial correction amount
    dst[4, 0] = src[4, 0] + left_shift
    dst[5, 0] = src[5, 0] + right_shift
    dst[4, 1] = avg_clav_y
    dst[5, 1] = avg_clav_y
    dst[6, 0] = src[6, 0]
    correction = (left_dist - right_dist) / 2
    print("Correction:", correction)
    mediastinum_shift = correction

    dst[11, 0] = src[11, 0] 
    dst[11, 1] = src[11, 1]
    #dst[9,0] = src[9,0] - correction
    #dst[10,0] = src[10,0] + correction
    tform = PiecewiseAffineTransform()
    print("Left shift:", left_shift)
    print("Right shift:", right_shift)
    print("CLAVICLE SRC LEFT :", src[4,0])
    print("CLAVICLE SRC RIGHT:", src[5,0])

    print("CLAVICLE DST LEFT :", dst[4,0])
    print("CLAVICLE DST RIGHT:", dst[5,0])
    tform.estimate(src, dst)
    
    warped = warp(
        img_cv2, tform.inverse,
        output_shape=(h, w),
        preserve_range=True
    )
    
    return warped.astype(np.uint8)
print("NEW WARP VERSION RUNNING")
# ─────────────────────────────────────────────
# CYCLEGAN INFERENCE INTERACTION
# ─────────────────────────────────────────────
def cyclegan_inference(img_cv2):
    if cyclegan is None:
        return img_cv2
    
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    tensor = cyclegan_transform(gray).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = cyclegan(tensor)
        out = out.squeeze().cpu().numpy()
        out = (out + 1) / 2
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
    
    # Scale up reconstruction space to match native resolution bounds
    out = cv2.resize(out, (img_cv2.shape[1], img_cv2.shape[0]))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

# ─────────────────────────────────────────────
# CORE IMAGE ALIGNMENT CONTROLLER
# ─────────────────────────────────────────────
def correct_with_cyclegan(img_cv2, angle, landmarks):
    try:
        # Step 1: Perform precise, high-resolution geometric rotation first
        corrected_geometric = rotate_spine_centered(img_cv2, angle, landmarks)
        
        # Step 2: Route through the experimental CycleGAN if enabled and loaded
        if USE_EXPERIMENTAL_CYCLEGAN and cyclegan is not None:
            corrected = cyclegan_inference(corrected_geometric)
            print("🔬 Processed via Geometric Transformation + CycleGAN Module")
        else:
            corrected = corrected_geometric
            print("📐 Processed via High-Fidelity Geometric Matrix Alignment")
        
        return corrected
    except Exception as e:
        print("\n===== RECONSTRUCTION FAULT =====")
        traceback.print_exc()
        return img_cv2

def image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# ─────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if not is_valid_xray(img_pil):
            return JSONResponse({
                "valid": False,
                "message": "Please upload a valid chest X-ray image."
            })
        
        
        cnn_class, cnn_conf, normal_conf, rotated_conf = predict_image(img_pil)
        cnn_signal = 1 if cnn_class == "rotated" else 0
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        landmarks = detect_landmarks_xrv(img_cv2)
        
        if landmarks is None:
            h, w = img_cv2.shape[:2]
            landmarks = {
                "spine_x": w // 2,
                "spine_top_y": int(h * 0.1),
                "spine_bottom_y": int(h * 0.85),
                "clavicle_left_x": int(w * 0.28),
                "clavicle_left_y": int(h * 0.22),
                "clavicle_right_x": int(w * 0.72),
                "clavicle_right_y": int(h * 0.22),
                "image_w": w,
                "image_h": h,
            }
        
        debug_img = visualize_landmarks(img_cv2, landmarks)
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "debug_landmarks.jpg"),
            debug_img
        )
        angle, left_cm, right_cm, rotation_ratio, direction, severity = calculate_correction_angle(landmarks)
        mediastinal_shift, mediastinal_status = calculate_mediastinal_shift(landmarks)
        scapular_status = calculate_scapular_position(landmarks)
        # ─────────────────────────────────────────────
        # STEP 3: HYBRID DECISION ENGINE
        # ─────────────────────────────────────────────

        
        

        geo_class = geometry_decision(angle, rotation_ratio)
        print("Geo Class:", geo_class)
        print("Angle:", angle)
        
        print("\n🧠 FINAL DIAGNOSIS ENGINE")
        
        print(f"👉 Rotation Angle: {angle:.2f}°")
        print(f"👉 Rotation Ratio: {rotation_ratio:.3f}")
        print(f"👉 Direction: {direction}")
        print(f"👉 Severity: {severity}")
        # FINAL DECISION LOGIC (GEOMETRY FIRST)
        if geo_class == "rotated":
            final_status = "ROTATED"
        elif cnn_signal == 1 and cnn_conf > 70:
            final_status = "ROTATED"
        else:
            final_status = "NORMAL"

        print(f"✅ FINAL STATUS: {final_status}")
        
        
       
        print("\n===== BEFORE CORRECTION LANDMARKS =====")
        print("Spine:", landmarks["spine_x"])
        print("Left Clavicle:", landmarks["clavicle_left_x"], landmarks["clavicle_left_y"])
        print("Right Clavicle:", landmarks["clavicle_right_x"], landmarks["clavicle_right_y"])
        corrected_b64 = None
        if final_status == "ROTATED" and rotation_ratio > 0.15:
            corrected_cv2 = anatomical_warp_correction(img_cv2, landmarks)
            
            corrected_landmarks = detect_landmarks_xrv(corrected_cv2)
            if corrected_landmarks is not None:
                angle2, left_cm2, right_cm2, ratio2, direction2, severity2 = \
                    calculate_correction_angle(corrected_landmarks)

                print("\n========== AFTER CORRECTION ==========")
                print("Left CM :", left_cm2)
                print("Right CM:", right_cm2)
                print("Ratio   :", ratio2)
                print("Angle   :", angle2)
            corrected_b64 = image_to_base64(corrected_cv2)
            out_path = os.path.join(OUTPUT_DIR, f"corrected_{file.filename}")
            cv2.imwrite(out_path, corrected_cv2)
        else:
            print("✅ Alignment within normal margins — skipping correction pipeline")
        
        original_b64 = image_to_base64(img_cv2)
        final_prediction = final_status
        return JSONResponse({
            "valid": True,
            "final_prediction": final_prediction,
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
            "right_cm": right_cm,
            "left_cm": left_cm,
            "original_img": original_b64,
            "corrected_img": corrected_b64 if corrected_b64 else original_b64,
        })
    
    except Exception as e:
        print(f"❌ Error encountered: {str(e)}")
        return JSONResponse({
            "valid": False,
            "message": f"Error: {str(e)}"
        })

@app.get("/")
def root():
    return {"message": "✅ Chest X-Ray Analysis API is running!"}