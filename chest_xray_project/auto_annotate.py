import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
IMAGE_FOLDER = r"E:\normal_dataset\Normal images"
OUTPUT_EXCEL = r"E:\normal_dataset\auto_annotations.xlsx"

# ─────────────────────────────────────────────
# DETECT SPINE CENTER COLUMN
# ─────────────────────────────────────────────
def detect_spine_column(gray):
    h, w     = gray.shape
    cx       = w // 2
    margin   = w // 5
    strip    = gray[:, cx - margin : cx + margin]
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(strip)
    _, thresh = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    col_sums    = np.sum(thresh, axis=0)
    spine_local = np.argmax(col_sums)
    spine_x     = (cx - margin) + spine_local
    return int(spine_x)

# ─────────────────────────────────────────────
# DETECT CLAVICLES
# ─────────────────────────────────────────────
def detect_clavicles(gray, spine_x):
    h, w         = gray.shape
    upper_region = gray[:int(h * 0.35), :]
    clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced     = clahe.apply(upper_region)
    _, thresh    = cv2.threshold(enhanced, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None

    valid = [c for c in contours if cv2.contourArea(c) > 300]
    if not valid:
        return None, None, None, None

    def get_cx(c):
        M = cv2.moments(c)
        if M['m00'] == 0:
            return 0
        return M['m10'] / M['m00']

    left_contours  = [c for c in valid if get_cx(c) < spine_x]
    right_contours = [c for c in valid if get_cx(c) > spine_x]

    def get_centroid(contours_list):
        if not contours_list:
            return None, None
        largest = max(contours_list, key=cv2.contourArea)
        M       = cv2.moments(largest)
        if M['m00'] == 0:
            return None, None
        return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

    lx, ly = get_centroid(left_contours)
    rx, ry = get_centroid(right_contours)
    return lx, ly, rx, ry

# ─────────────────────────────────────────────
# PIXEL TO CM
# ─────────────────────────────────────────────
def pixel_to_cm(pixels, dpi=72):
    return round((pixels / dpi) * 2.54, 2)

# ─────────────────────────────────────────────
# PROCESS SINGLE IMAGE
# ─────────────────────────────────────────────
def process_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape
        spine_x = detect_spine_column(gray)

        lx, ly, rx, ry = detect_clavicles(gray, spine_x)

        if lx is None or rx is None:
            # Fallback — use image proportions
            lx = int(w * 0.3)
            ly = int(h * 0.2)
            rx = int(w * 0.7)
            ry = int(h * 0.2)

        left_dist_px  = abs(spine_x - lx)
        right_dist_px = abs(rx - spine_x)
        left_cm       = pixel_to_cm(left_dist_px)
        right_cm      = pixel_to_cm(right_dist_px)
        asymmetry     = round(abs(right_cm - left_cm), 2)
        label         = "Rotated" if asymmetry > 0.5 else "Normal"

        return {
            "spine_x"         : spine_x,
            "spine_top_y"     : int(h * 0.1),
            "spine_bottom_y"  : int(h * 0.85),
            "clavicle_left_x" : lx,
            "clavicle_left_y" : ly,
            "clavicle_right_x": rx,
            "clavicle_right_y": ry,
            "left_dist_px"    : left_dist_px,
            "right_dist_px"   : right_dist_px,
            "left_cm"         : left_cm,
            "right_cm"        : right_cm,
            "asymmetry_cm"    : asymmetry,
            "label"           : label,
            "image_width"     : w,
            "image_height"    : h,
        }
    except Exception:
        return None

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Get ALL image files regardless of filename format
    all_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.jpeg', '.jpg', '.png'))
    ]

    # Sort safely — handles both numeric and non-numeric names
    def safe_sort_key(filename):
        name = os.path.splitext(filename)[0]
        # Remove any (1), (2) suffixes
        name = name.split('(')[0].strip()
        try:
            return (0, int(name))
        except ValueError:
            return (1, name)

    all_files.sort(key=safe_sort_key)

    print(f"Found {len(all_files)} images in folder")
    print(f"Starting auto annotation...\n")

    results     = []
    success     = 0
    failed      = 0
    failed_list = []

    for filename in tqdm(all_files, desc="Annotating"):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        image_num  = os.path.splitext(filename)[0]
        result     = process_image(image_path)

        if result is not None:
            result["image"]    = image_num
            result["filename"] = filename
            results.append(result)
            success += 1
        else:
            failed += 1
            failed_list.append(filename)

    # ── Save Excel ──
    df   = pd.DataFrame(results)
    cols = [
        "image", "filename",
        "right_cm", "left_cm", "asymmetry_cm", "label",
        "spine_x", "spine_top_y", "spine_bottom_y",
        "clavicle_left_x",  "clavicle_left_y",
        "clavicle_right_x", "clavicle_right_y",
        "left_dist_px", "right_dist_px",
        "image_width",  "image_height"
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_excel(OUTPUT_EXCEL, index=False)

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"✅ Auto Annotation Complete!")
    print(f"{'='*50}")
    print(f"Total images   : {len(all_files)}")
    print(f"Successfully   : {success}")
    print(f"Failed         : {failed}")
    if len(df) > 0:
        print(f"Normal images  : {len(df[df['label'] == 'Normal'])}")
        print(f"Rotated images : {len(df[df['label'] == 'Rotated'])}")
    print(f"\n✅ Excel saved: {OUTPUT_EXCEL}")

    if failed_list:
        print(f"\n⚠️  Failed images ({len(failed_list)}):")
        for f in failed_list[:5]:
            print(f"   - {f}")
        if len(failed_list) > 5:
            print(f"   ... and {len(failed_list)-5} more")

if __name__ == "__main__":
    main()