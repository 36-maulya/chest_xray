import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import timm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchxrayvision as xrv
import skimage.transform

# ───────────────────────────────────────────────────────────────
# 1. SETUP & PATHS
# ───────────────────────────────────────────────────────────────
EFFICIENTNET_PATH = r"models\efficientnet_xray_v2.pth"
DATASET_DIR = r"D:\4SO23CS139\major_project_demo\chest_xray_project\data\All"  # Change this to your folder path containing the 3,000 unlabelled images
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using execution device: {device}")

# Transform pipeline matching your training domain space
efficientnet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ───────────────────────────────────────────────────────────────
# 2. LOAD MODELS & CORES
# ───────────────────────────────────────────────────────────────
print("Loading EfficientNet backbone...")
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
model.eval()
model.to(device)

print("Loading TorchXRayVision (PSPNet) for anatomical verification...")
xrv_model = xrv.baseline_models.chestx_det.PSPNet().to(device)
xrv_model.eval()

# ───────────────────────────────────────────────────────────────
# 3. ANATOMICAL RULES ENGINE (Bypasses manual labeling)
# ───────────────────────────────────────────────────────────────
def get_anatomy_label(img_path):
    """Evaluates geometric asymmetry to automatically assign evaluation categories."""
    try:
        img_cv2 = cv2.imread(img_path)
        if img_cv2 is None:
            return "Uncertain"
            
        h_orig, w_orig = img_cv2.shape[:2]
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        
        # Scale intensity to match XRV expectations
        img_xrv = gray.astype(np.float32) / 255.0 * 2048 - 1024
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
            
            spine_x = None
            clav_lx = None
            clav_rx = None
            
            for i, label in enumerate(targets):
                ll = label.lower()
                x, y = get_pos(landmarks_maps[i], w_orig, h_orig)
                if "spine" in ll: 
                    spine_x = x
                elif "left clavicle" in ll: 
                    clav_lx = x
                elif "right clavicle" in ll: 
                    clav_rx = x
            
            # Fallbacks if landmarks aren't clearly captured
            spine_x = spine_x or w_orig // 2
            clav_lx = clav_lx or int(w_orig * 0.3)
            clav_rx = clav_rx or int(w_orig * 0.7)
            
            left_dist = abs(spine_x - clav_lx)
            right_dist = abs(clav_rx - spine_x)
            
            dpi = 72
            left_cm = (left_dist / dpi) * 2.54
            right_cm = (right_dist / dpi) * 2.54
            asymmetry = abs(left_cm - right_cm)
            
            # Label threshold matches main.py limits
            return "Rotated" if asymmetry >= 0.8 else "Normal"
    except Exception:
        return "Uncertain"

# ───────────────────────────────────────────────────────────────
# 4. ITERATE UNLABELLED CLINICAL INVENTORY
# ───────────────────────────────────────────────────────────────
features_list = []
labels_list = []

if not os.path.exists(DATASET_DIR):
    print(f"⚠️ Error: The directory '{DATASET_DIR}' was not found. Please create it and add your images.")
    exit()

valid_extensions = ('.jpg', '.jpeg', '.png', '.PNG', '.JPG')
all_images = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(valid_extensions)]
print(f"Discovered {len(all_images)} unlabelled images inside file repository.")

# Process up to 1000 images for clean spatial visualization modeling
target_limit = min(300, len(all_images))
if target_limit == 0:
    print("⚠️ No images found to process. Please check your folder.")
    exit()

print(f"Processing {target_limit} samples for structural representation profiling...")

for idx, img_path in enumerate(all_images[:target_limit]):
    try:
        # Step A: Run through the anatomical engine to determine group context
        pseudo_label = get_anatomy_label(img_path)
        if pseudo_label == "Uncertain":
            continue
            
        # Step B: Extract raw latent feature activations
        img_pil = Image.open(img_path).convert("RGB")
        tensor = efficientnet_transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Call internal forward_features directly to avoid submodule subclass errors
            raw_features = model.forward_features(tensor) 
            pooled_features = model.global_pool(raw_features) 
            feat = torch.flatten(pooled_features, 1).cpu().numpy().flatten()
            
        features_list.append(feat)
        labels_list.append(pseudo_label)
        
        if (idx + 1) % 100 == 0 or (idx + 1) == target_limit:
            print(f"Progress Status: Cached feature arrays for {idx + 1} images.")
    except Exception as e:
        continue

features_arr = np.array(features_list)
labels_arr = np.array(labels_list)

if len(features_arr) == 0:
    print("❌ Critical Error: Could not extract features from any image files.")
    exit()

# ───────────────────────────────────────────────────────────────
# 5. EXECUTE MANIFOLD PROJECTION (t-SNE)
# ───────────────────────────────────────────────────────────────
print(f"Running t-SNE dimensionality reduction on shape array {features_arr.shape}...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(features_arr)

# ───────────────────────────────────────────────────────────────
# 6. SAVE HIGH-RESOLUTION CHART FOR PUBLICATION
# ───────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))

# Classic high-contrast academic color mappings
color_map = {"Normal": "#1f77b4", "Rotated": "#d62728"}

for class_name in ["Normal", "Rotated"]:
    indices = np.where(labels_arr == class_name)[0]
    if len(indices) > 0:
        plt.scatter(
            tsne_results[indices, 0], 
            tsne_results[indices, 1],
            label=f"Discovered {class_name}",
            c=color_map[class_name],
            alpha=0.75,
            s=45,
            edgecolors='w',
            linewidths=0.5
        )

plt.title("t-SNE Spatial Clustering of Discovered Latent Feature Vectors\n(Trained on Unlabelled Clinical Hospital Data Matrix)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("t-SNE Axis 1", fontsize=11, fontweight='semibold')
plt.ylabel("t-SNE Axis 2", fontsize=11, fontweight='semibold')
plt.legend(title="Anatomical Groupings", title_fontsize='11', loc='upper right', frameon=True)
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
output_fig_path = "unsupervised_anatomy_clusters.png"
plt.savefig(output_fig_path, dpi=300)
print(f"\n🎉 Success! High-resolution t-SNE plot written to disk: {output_fig_path}")