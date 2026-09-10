"""
Automatic License Plate Recognition (ALPR) System
===================================================
Uses OpenCV for plate localization and EasyOCR for character recognition.
Upload your own car images via a file dialog to detect and read number plates.
"""

import cv2
import imutils
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import sys
from utils import fix_plate_text


sys.stdout.reconfigure(encoding='utf-8')


def select_image():
    """Open a file dialog to let the user pick an image."""
    root = tk.Tk()
    root.withdraw()  
    root.attributes('-topmost', True)  

    file_path = filedialog.askopenfilename(
        title="Select a Car Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path


def find_plate_contour(edged, img_width, img_height):
    """
    Try to find a license plate contour in the edge image.
    Returns the contour location or None.
    """
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    img_area = img_width * img_height

    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < img_area * 0.005 or area > img_area * 0.5:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        
        if 4 <= len(approx) <= 6:
            
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            if 1.2 <= aspect_ratio <= 7.0:
                return approx

    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 1.0 <= aspect_ratio <= 8.0:
                return approx

    return None


def detect_license_plate(image_path):
    """
    Main pipeline:
    1. Preprocess with grayscale + bilateral filter + Canny edges
    2. Locate rectangular contour (the plate)
    3. Mask and crop the plate region
    4. Run EasyOCR to read characters/digits
    5. Display results
    """
    print(f"\n{'='*60}")
    print(f"  Processing: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    
    img = imutils.resize(img, width=620)
    img_height, img_width = img.shape[:2]
    print("[1/5] Image loaded and resized.")

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  
    print("[2/5] Preprocessing done (Grayscale → Bilateral Filter).")

    
    location = None
    edged = None

    
    canny_params = [(30, 200), (50, 150), (20, 100), (100, 250)]
    for low, high in canny_params:
        edged_try = cv2.Canny(bfilter, low, high)
        location = find_plate_contour(edged_try, img_width, img_height)
        if location is not None:
            edged = edged_try
            print(f"[3/5] License plate contour located! (Canny {low},{high})")
            break

    
    if location is None:
        thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        location = find_plate_contour(thresh, img_width, img_height)
        if location is not None:
            edged = thresh
            print("[3/5] License plate contour located! (Adaptive threshold)")

    
    if location is None:
        edged_morph = cv2.Canny(bfilter, 30, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged_morph = cv2.dilate(edged_morph, kernel, iterations=1)
        edged_morph = cv2.erode(edged_morph, kernel, iterations=1)
        location = find_plate_contour(edged_morph, img_width, img_height)
        if location is not None:
            edged = edged_morph
            print("[3/5] License plate contour located! (Morphological)")

    if edged is None:
        edged = cv2.Canny(bfilter, 30, 200)

    if location is None:
        print("[WARNING] Could not find a rectangular plate contour.")
        print("          Falling back to full-image OCR scan...")

        
        print("[5/5] Running EasyOCR on full image...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        result = reader.readtext(gray)

        if not result:
            print("\n[RESULT] No text detected in the image at all.")
            print("         Try a clearer image where the plate is visible.")
        else:
            print(f"\n{'─'*60}")
            print(f"  DETECTED TEXT (FULL IMAGE SCAN)")
            print(f"{'─'*60}")
            for (bbox, text, confidence) in result:
                print(f"  Text: {text:<20}  Confidence: {confidence:.1%}")
            all_text = " | ".join([f"{res[1]} ({res[2]:.0%})" for res in result])
            print(f"{'─'*60}")
            print(f"  All detections: {all_text}")
            print(f"{'─'*60}")

        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('License Plate Detection — Full Image Scan', fontsize=16, fontweight='bold')

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        annotated = img.copy()
        if result:
            for (bbox, text, confidence) in result:
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                cv2.putText(annotated, f"{text} ({confidence:.0%})",
                           (pts[0][0], pts[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        axes[0].set_title('Original Image')
        axes[0].imshow(img_rgb)
        axes[0].axis('off')

        axes[1].set_title('OCR Detections')
        axes[1].imshow(annotated_rgb)
        axes[1].axis('off')

        plt.tight_layout()

        output_path = os.path.splitext(image_path)[0] + '_result.jpg'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] Result visualization → {output_path}")
        plt.show()
        return

    
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    print("[4/5] Plate region cropped.")

    
    print("[5/5] Running EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    result = reader.readtext(cropped_image)

    if not result:
        print("\n[RESULT] EasyOCR could not extract any text from the plate.")
        print("         The image might be too blurry or the plate too small.")
    else:
        print(f"\n{'─'*60}")
        print(f"  DETECTED LICENSE PLATE TEXT")
        print(f"{'─'*60}")
        for (bbox, text, confidence) in result:
            print(f"  Text: {text:<20}  Confidence: {confidence:.1%}")
        plate_text_raw = " ".join([res[1] for res in result])
        plate_text = fix_plate_text(plate_text_raw)
        print(f"{'─'*60}")
        print(f"  Raw OCR:  {plate_text_raw}")
        print(f"  Corrected: {plate_text}")
        print(f"{'─'*60}")

    
    final_img = img.copy()
    cv2.polylines(final_img, [location], isClosed=True, color=(0, 255, 0), thickness=3)

    
    plate_text_display = plate_text if result else "N/A"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_org = (location[0][0][0], location[0][0][1] - 15)
    cv2.putText(final_img, plate_text_display, text_org, font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    cropped_rgb = cropped_image  

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('License Plate Detection Results', fontsize=16, fontweight='bold')

    axes[0, 0].set_title('Original Image')
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].axis('off')

    axes[0, 1].set_title('Edge Detection (Canny)')
    axes[0, 1].imshow(edged, cmap='gray')
    axes[0, 1].axis('off')

    axes[1, 0].set_title('Cropped License Plate')
    axes[1, 0].imshow(cropped_rgb, cmap='gray')
    axes[1, 0].axis('off')

    axes[1, 1].set_title(f'Result: {plate_text_display}')
    axes[1, 1].imshow(final_rgb)
    axes[1, 1].axis('off')

    plt.tight_layout()

    
    output_path = os.path.splitext(image_path)[0] + '_result.jpg'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Result visualization → {output_path}")

    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("  AUTOMATIC LICENSE PLATE RECOGNITION SYSTEM")
    print("=" * 60)
    print("\nA file dialog will open — please select a car image.\n")

    image_path = select_image()

    if not image_path:
        print("[INFO] No image selected. Exiting.")
    else:
        detect_license_plate(image_path)