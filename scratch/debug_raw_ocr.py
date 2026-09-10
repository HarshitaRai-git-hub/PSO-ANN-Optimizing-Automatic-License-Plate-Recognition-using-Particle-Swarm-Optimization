"""Debug: Show ALL raw EasyOCR detections before any filtering."""
import cv2
import imutils
import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False, verbose=False)

img = cv2.imread('car_image.jpg')
img = imutils.resize(img, width=1000)
h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

# Enhanced image
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_ch, a_ch, b_ch = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
l_enh = clahe.apply(l_ch)
lab_enh = cv2.merge([l_enh, a_ch, b_ch])
img_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

# 2x scale
scale = 2.0
plate_enh = cv2.resize(img_enh, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

results = reader.readtext(
    plate_enh,
    contrast_ths=0.2,
    adjust_contrast=0.8,
    paragraph=False,
    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
)

print(f"\nTotal raw detections: {len(results)}\n")
for i, (bbox, text, conf) in enumerate(results):
    # Scale bbox back
    pts = [[int(p[0]/scale), int(p[1]/scale)] for p in bbox]
    x1, y1 = pts[0]
    x2, y2 = pts[2]
    bw = x2 - x1
    bh = y2 - y1
    ar = bw / max(bh, 1)
    area_ratio = (bw * bh) / (w * h)
    print(f"  [{i}] text={text!r:25s} conf={conf:.3f}  box=({x1},{y1})-({x2},{y2})  size={bw}x{bh}  AR={ar:.2f}  area%={area_ratio*100:.3f}")
