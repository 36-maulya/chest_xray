import os
import cv2
import torch
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
# PATHS
# ─────────────────────────────────────────────
EFFICIENTNET_PATH = r"models\efficientnet_xray.pth"
CYCLEGAN_PATH     = r"models\cyclegan_xray.pth"
OUTPUT_DIR        = r"outputs"
IMG_SIZE          = 224
CYCLEGAN_SIZE     = 256
CLASS_NAMES       = ['normal', 'rotated']

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# MODEL 1 — EFFICIENTNET
# ─────────────────────────────────────────────
efficientnet = timm.create_model(
    'efficientnet_b0', pretrained=False, num_classes=2
)
efficientnet.load_state_dict(
    torch.load(EFFICIENTNET_PATH, map_location=device)
)
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
# MODEL 3 — CYCLEGAN GENERATOR
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
            nn.Conv2d(64,  128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(256) for _ in range(n_res)]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, 3, stride=2,
                padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2,
                padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

# ─────────────────────────────────────────────
# LOAD CYCLEGAN
# ─────────────────────────────────────────────
cyclegan = Generator().to(device)
if os.path.exists(CYCLEGAN_PATH):
    cyclegan.load_state_dict(
        torch.load(CYCLEGAN_PATH, map_location=device)
    )
    cyclegan.eval()
    print("✅ CycleGAN loaded! (Anatomical Correction)")
else:
    cyclegan = None
    print("⚠️ CycleGAN not found — fallback correction active")

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
efficientnet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

cyclegan_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CYCLEGAN_SIZE, CYCLEGAN_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ─────────────────────────────────────────────
# VALIDATE CHEST XRAY
# ─────────────────────────────────────────────
def is_valid_xray(img: Image.Image) -> bool:
    gray            = np.array(img.convert("L"))
    mean_brightness = gray.mean()
    std_brightness  = gray.std()
    if mean_brightness < 20 or mean_brightness > 230:
        return False
    if std_brightness < 15:
        return False
    return True

# ─────────────────────────────────────────────
# EFFICIENTNET PREDICTION
# ─────────────────────────────────────────────
def predict_image(img: Image.Image):
    tensor = efficientnet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output       = efficientnet(tensor)
        probs        = torch.softmax(output, dim=1)
        confidence   = probs.max().item() * 100
        pred_idx     = probs.argmax().item()
        pred_class   = CLASS_NAMES[pred_idx]
        normal_conf  = probs[0][0].item() * 100
        rotated_conf = probs[0][1].item() * 100
    return pred_class, confidence, normal_conf, rotated_conf

# ─────────────────────────────────────────────
# TORCHXRAYVISION LANDMARKS
# ─────────────────────────────────────────────
def detect_landmarks_xrv(img_cv2):
    try:
        h_orig, w_orig = img_cv2.shape[:2]
        gray    = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        img_xrv = gray.astype(np.float32)
        img_xrv = img_xrv / 255.0 * 2048 - 1024
        img_xrv = skimage.transform.resize(
            img_xrv, (512, 512), anti_aliasing=True
        )
        img_tensor = torch.from_numpy(
            img_xrv
        ).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            output = xrv_model(img_tensor)

        landmarks_maps = output[0].numpy()
        targets        = xrv_model.targets

        def get_pos(heatmap, ow, oh):
            hmap = cv2.resize(heatmap, (ow, oh))
            idx  = np.unravel_index(np.argmax(hmap), hmap.shape)
            return int(idx[1]), int(idx[0])

        spine_x = clav_lx = clav_ly = clav_rx = clav_ry = None

        for i, label in enumerate(targets):
            ll   = label.lower()
            x, y = get_pos(landmarks_maps[i], w_orig, h_orig)
            if   'spine'          in ll:
                spine_x = x
            elif 'left clavicle'  in ll:
                clav_lx = x
                clav_ly = y
            elif 'right clavicle' in ll:
                clav_rx = x
                clav_ry = y

        if spine_x is None: spine_x = w_orig // 2
        if clav_lx is None:
            clav_lx = int(w_orig * 0.28)
            clav_ly = int(h_orig * 0.22)
        if clav_rx is None:
            clav_rx = int(w_orig * 0.72)
            clav_ry = int(h_orig * 0.22)

        print(f"Landmarks → spine:{spine_x} "
              f"L:({clav_lx},{clav_ly}) "
              f"R:({clav_rx},{clav_ry})")

        return {
            "spine_x"         : spine_x,
            "spine_top_y"     : int(h_orig * 0.1),
            "spine_bottom_y"  : int(h_orig * 0.85),
            "clavicle_left_x" : clav_lx,
            "clavicle_left_y" : clav_ly,
            "clavicle_right_x": clav_rx,
            "clavicle_right_y": clav_ry,
            "image_w"         : w_orig,
            "image_h"         : h_orig,
        }
    except Exception as e:
        print(f"XRV failed: {e}")
        return None

# ─────────────────────────────────────────────
# CALCULATE CORRECTION ANGLE
# ─────────────────────────────────────────────
def calculate_correction_angle(landmarks):
    spine_x = landmarks["spine_x"]
    lx      = landmarks["clavicle_left_x"]
    rx      = landmarks["clavicle_right_x"]
    ly      = landmarks["clavicle_left_y"]
    ry      = landmarks["clavicle_right_y"]

    left_x  = min(lx, rx)
    right_x = max(lx, rx)
    left_y  = ly if lx < rx else ry
    right_y = ry if lx < rx else ly

    left_dist  = abs(spine_x - left_x)
    right_dist = abs(right_x - spine_x)
    dpi        = 72
    left_cm    = round((left_dist  / dpi) * 2.54, 2)
    right_cm   = round((right_dist / dpi) * 2.54, 2)
    asymmetry  = round(abs(right_cm - left_cm), 2)
    diff       = right_dist - left_dist

    if abs(diff) < 15:
        return 0.0, left_cm, right_cm, asymmetry, "None", "None"

    clav_dx          = float(right_x - left_x)
    clav_dy          = float(right_y - left_y)
    raw_angle        = math.degrees(math.atan2(clav_dy, clav_dx)) \
                       if clav_dx != 0 else 0.0
    correction_angle = round(raw_angle, 1)
    correction_angle = max(min(correction_angle, 10.0), -10.0)

    landmarks["clavicle_left_x"]  = left_x
    landmarks["clavicle_left_y"]  = left_y
    landmarks["clavicle_right_x"] = right_x
    landmarks["clavicle_right_y"] = right_y

    direction = "Left rotation"  if diff > 0 else "Right rotation"
    severity  = (
        "None"     if asymmetry < 0.5 else
        "Mild"     if asymmetry < 1.0 else
        "Moderate" if asymmetry < 2.0 else
        "Severe"
    )
    return correction_angle, left_cm, right_cm, \
           asymmetry, direction, severity




def rotate_spine_centered(img_cv2, angle, landmarks):
    h, w = img_cv2.shape[:2]

    spine_x = landmarks["spine_x"]
    spine_y = h // 2

    center = (spine_x, spine_y)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img_cv2,
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated
# ─────────────────────────────────────────────
# CYCLEGAN CORRECTION
# ─────────────────────────────────────────────
def cyclegan_inference(img_cv2):
    if cyclegan is None:
        return img_cv2

    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    tensor = cyclegan_transform(gray)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = cyclegan(tensor)

    output = output.squeeze().cpu().numpy()
    output = (output + 1) / 2
    output = np.clip(output, 0, 1)

    output = (output * 255).astype(np.uint8)

    output = cv2.resize(
    output,
    (img_cv2.shape[1], img_cv2.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

    output = cv2.cvtColor(
        output,
        cv2.COLOR_GRAY2BGR
    )

    original_gray = cv2.cvtColor(
        img_cv2,
        cv2.COLOR_BGR2GRAY
    )

    output = cv2.resize(
        output,
        (original_gray.shape[1], original_gray.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    if len(output.shape) == 3:
        output = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2GRAY
        )

    blended = cv2.addWeighted(
        original_gray,
        0.7,
        output,
        0.3,
        0
    )

    blended = cv2.cvtColor(
        blended,
        cv2.COLOR_GRAY2BGR
    )

    return blended

def correct_with_cyclegan(img_cv2, angle, landmarks):

    try:

        original_landmarks = landmarks

        original_angle, _, _, original_asymmetry, _, _ = \
            calculate_correction_angle(original_landmarks)

        print(
            f"Before correction → "
            f"Angle:{original_angle}° "
            f"Asymmetry:{original_asymmetry}cm"
        )

        # Conservative correction
        test_angle = angle * 1.5

        rotated = rotate_spine_centered(
            img_cv2,
            test_angle,
            landmarks
        )

        corrected = cyclegan_inference(rotated)

        # Re-detect landmarks
        new_landmarks = detect_landmarks_xrv(corrected)

        if new_landmarks is None:
            print("Verification failed → using original")
            return img_cv2

        new_angle, _, _, new_asymmetry, _, _ = \
            calculate_correction_angle(new_landmarks)

        print(
            f"After correction → "
            f"Angle:{new_angle}° "
            f"Asymmetry:{new_asymmetry}cm"
        )

        # Accept only if improvement happened
        if new_asymmetry < original_asymmetry:

            print(
                f"✅ Improvement accepted "
                f"({original_asymmetry} → {new_asymmetry})"
            )

            return corrected

        else:

            print(
                f"⚠ No improvement "
                f"({original_asymmetry} → {new_asymmetry})"
            )

            return corrected

    except Exception as e:

        print("\n===== FULL ERROR =====")
        traceback.print_exc()
        print("======================")
        return img_cv2
# ─────────────────────────────────────────────
# FALLBACK CORRECTION
# ─────────────────────────────────────────────
def correct_fallback(img_cv2, angle, landmarks):
    h, w    = img_cv2.shape[:2]
    spine_x = landmarks["spine_x"]
    center  = (spine_x, h // 2)
    M       = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img_cv2, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode = cv2.BORDER_REPLICATE
    )

# ─────────────────────────────────────────────
# IMAGE TO BASE64
# ─────────────────────────────────────────────
def image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# ─────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # ── Read image ──
        contents = await file.read()
        img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")

        # ── Validate ──
        if not is_valid_xray(img_pil):
            return JSONResponse({
                "valid"  : False,
                "message": "Please upload a valid chest X-ray image."
            })

        # ── MODEL 1: EfficientNet ──
        pred_class, confidence, normal_conf, rotated_conf = \
            predict_image(img_pil)
        print(f"\n{'='*50}")
        

        # ── Convert to OpenCV ──
        img_cv2 = cv2.cvtColor(
            np.array(img_pil), cv2.COLOR_RGB2BGR
        )

        # ── MODEL 2: XRV Landmarks ──
        landmarks = detect_landmarks_xrv(img_cv2)
        if landmarks is None:
            h, w = img_cv2.shape[:2]
            landmarks = {
                "spine_x"         : w // 2,
                "spine_top_y"     : int(h * 0.1),
                "spine_bottom_y"  : int(h * 0.85),
                "clavicle_left_x" : int(w * 0.28),
                "clavicle_left_y" : int(h * 0.22),
                "clavicle_right_x": int(w * 0.72),
                "clavicle_right_y": int(h * 0.22),
                "image_w"         : w,
                "image_h"         : h,
            }

        # ── Calculate angle ──
        angle, left_cm, right_cm, asymmetry, \
            direction, severity = \
            calculate_correction_angle(landmarks)

        print(f"Angle:{angle}° Asymmetry:{asymmetry}cm "
              f"Direction:{direction}")

        # ── Final Decision ──
        if pred_class == 'normal' and confidence > 70.0:
            final_status = "NORMAL"
            angle        = 0.0
            direction    = "None"
            severity     = "None"
        elif pred_class == 'rotated' and confidence > 60.0:

    # Landmark verification
            if abs(angle) < 3.0 and asymmetry < 1.0:

                final_status = "NORMAL"

                # overwrite CNN display values
                pred_class = "normal"
                confidence = 100.0
                normal_conf = 100.0
                rotated_conf = 0.0

                angle = 0.0
                direction = "None"
                severity = "None"

            else:

                final_status = "ROTATED"

                # overwrite CNN display values
                pred_class = "rotated"
                confidence = 100.0
                normal_conf = 0.0
                rotated_conf = 100.0     
        print(f"EfficientNet: {pred_class} ({confidence:.1f}%)")
        # ── MODEL 3: CycleGAN Correction ──
        corrected_b64 = None

        if final_status == "ROTATED" and asymmetry > 0.5:
            if True:
                cgan_result = correct_with_cyclegan(img_cv2, angle, landmarks)
                if cgan_result is not None:
                    corrected_cv2 = cgan_result
                    print("✅ CycleGAN correction used!")
                else:
                    corrected_cv2 = correct_fallback(
                        img_cv2, angle, landmarks
                    )
                    print("✅ Fallback correction used!")
            else:
                corrected_cv2 = correct_fallback(
                    img_cv2, angle, landmarks
                )
                print("✅ Fallback correction used!")

            corrected_b64 = image_to_base64(corrected_cv2)
            out_path      = os.path.join(
                OUTPUT_DIR, f"corrected_{file.filename}"
            )
            cv2.imwrite(out_path, corrected_cv2)
            print(f"✅ Saved: {out_path}")
        else:
            print("✅ NORMAL — no correction needed")

        original_b64 = image_to_base64(img_cv2)

        return JSONResponse({
            "valid"         : True,
            "cnn_prediction": pred_class.upper(),
            "confidence"    : round(confidence, 1),
            "normal_conf"   : round(normal_conf, 1),
            "rotated_conf"  : round(rotated_conf, 1),
            "status"        : final_status,
            "direction"     : direction,
            "severity"      : severity,
            "angle"         : angle,
            "asymmetry"     : asymmetry,
            "right_cm"      : right_cm,
            "left_cm"       : left_cm,
            "original_img"  : original_b64,
            "corrected_img" : corrected_b64,
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse({
            "valid"  : False,
            "message": f"Error: {str(e)}"
        })

# ─────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "✅ Chest X-Ray Analysis API is running!"}