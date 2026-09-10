"""
ALPR Web Application
====================
Flask backend that serves a web UI for uploading car images
and displays license plate detection results.

Uses EasyOCR text-region detection to find plate-like text,
then validates with alphanumeric pattern matching.
"""

import cv2
import imutils
import easyocr
import numpy as np
import os
import sys
import re
import json
import base64
import time
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from train_pso import SimpleANN, load_weights

sys.stdout.reconfigure(encoding='utf-8')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app = Flask(__name__)
app.json_encoder = NumpyEncoder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}


_reader = None


def get_reader():
    global _reader
    if _reader is None:
        print("[INIT] Loading EasyOCR reader (first time may download models)...")
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[INIT] EasyOCR reader ready.")
    return _reader


_pso_nn = None
def get_pso_model():
    global _pso_nn
    if _pso_nn is None:
        weights_file = "pso_ann_weights.npy"
        if os.path.exists(weights_file):
            print(f"[INIT] Loading PSO-ANN weights from {weights_file}...")
            layer_sizes = [784, 32, 10]
            _pso_nn = SimpleANN(layer_sizes)
            _pso_nn.set_weights(load_weights(weights_file))
            print("[INIT] PSO-ANN model ready.")
        else:
            print("[WARNING] PSO-ANN weights file not found. Falling back to EasyOCR only.")
    return _pso_nn


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image(img):
    """Encode a cv2 image (BGR or grayscale) to base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def looks_like_plate(text):
    """
    Score how likely a text string is a license plate number.
    License plates typically have a mix of uppercase letters and digits,
    e.g. 'MH 12 AB 1234', 'DL 8C N 1234', 'KA01MG1234', 'ABC-1234'.
    """
    clean = re.sub(r'[^A-Za-z0-9]', '', text)

    if len(clean) < 3:
        return 0.0

    score = 0.0

    
    has_digit = bool(re.search(r'\d', clean))
    has_letter = bool(re.search(r'[A-Za-z]', clean))

    if has_digit and has_letter:
        score += 40  
    elif has_digit and len(clean) >= 4:
        score += 15  
    else:
        return 0.0  

    
    if 4 <= len(clean) <= 12:
        score += 20
    elif 3 <= len(clean) <= 15:
        score += 10

    
    if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$', clean, re.IGNORECASE):
        score += 30  

    
    if re.match(r'^\d{2}BH\d{4}[A-Z]{1,2}$', clean, re.IGNORECASE):
        score += 35  

    
    if re.match(r'^[A-Z]{1,4}\d{1,5}$', clean, re.IGNORECASE):
        score += 20
    if re.match(r'^[A-Z]{2,3}[\s\-]?\d{3,4}$', clean, re.IGNORECASE):
        score += 20

    
    if len(clean) > 15:
        score -= 20

    
    if len(clean) < 4:
        
        if clean.isalpha():
            score -= 30
        
        elif len(clean) <= 3:
            score -= 15

    
    lowercase_ratio = sum(1 for c in text if c.islower()) / max(len(text), 1)
    if lowercase_ratio > 0.7:
        score -= 15

    return max(score, 0.0)


from utils import fix_plate_text


BRAND_NAMES = {
    'KIA', 'SUZUKI', 'HONDA', 'TOYOTA', 'HYUNDAI', 'FORD', 'BMW', 'MERCEDES', 'AUDI',
    'VW', 'VOLKSWAGEN', 'NISSAN', 'MAZDA', 'LEXUS', 'FIAT', 'JEEP', 'TESLA',
}


EMBLEM_TEXT = {
    'IND', 'IN', 'INDIA', 'GOVT', 'GOV', 'TRANSPORT', 'BHARAT', '1ND', '1MD', 'IWD',
}

def pso_recognize_digits(cropped_plate):
    """
    Experimental: Segment characters and use PSO-ANN to recognize digits.
    """
    nn = get_pso_model()
    if nn is None:
        return None

    
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    
    char_locs = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        
        aspect_ratio = w / float(h)
        if 0.15 <= aspect_ratio <= 0.95 and 25 <= h <= 120 and w >= 5:
            char_locs.append((x, y, w, h))
    
    
    char_locs = sorted(char_locs, key=lambda x: x[0])
    unique_locs = []
    for loc in char_locs:
        is_dup = False
        for u in unique_locs:
            
            overlap = min(loc[0]+loc[2], u[0]+u[2]) - max(loc[0], u[0])
            if overlap > min(loc[2], u[2]) * 0.7:
                is_dup = True; break
        if not is_dup: unique_locs.append(loc)
    char_locs = unique_locs

    pso_digits = []
    for (x, y, w, h) in char_locs:
        
        char = thresh[y:y+h, x:x+w]
        
        
        pad = max(w, h)
        square_char = np.zeros((pad, pad), dtype="uint8")
        dx = (pad - w) // 2
        dy = (pad - h) // 2
        square_char[dy:dy+h, dx:dx+w] = char
        
        
        char_resized = cv2.resize(square_char, (28, 28), interpolation=cv2.INTER_AREA)
        char_flat = char_resized.flatten() / 255.0
        
        
        output = nn.forward(char_flat)
        digit = np.argmax(output)
        pso_digits.append(str(digit))
        
    return "".join(pso_digits)


def process_image(image_path):
    """
    Run the ALPR pipeline:
    1. Preprocess image
    2. Run EasyOCR to detect ALL text regions
    3. Score each region for plate-likeness
    4. Pick the best candidate as the license plate
    5. Return results with visualizations
    """
    start_time = time.time()

    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': 'Could not read the uploaded image.'}

    img = imutils.resize(img, width=1000)
    img_height, img_width = img.shape[:2]
    img_area = img_width * img_height

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    reader = get_reader()

    
    
    ocr_scale = 2.0
    plate_enhanced = cv2.resize(img_enhanced, (0,0), fx=ocr_scale, fy=ocr_scale, interpolation=cv2.INTER_LANCZOS4)
    
    
    results = reader.readtext(
        plate_enhanced,
        contrast_ths=0.2, 
        adjust_contrast=0.8,
        paragraph=False,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',  
    )

    if not results:
        
        elapsed = round(time.time() - start_time, 2)
        return {
            'success': True,
            'plate_found': False,
            'strategy': 'No text detected in image',
            'detections': [],
            'combined_text': 'No text detected',
            'original_image': encode_image(img),
            'edge_image': encode_image(edged),
            'result_image': encode_image(img),
            'cropped_image': None,
            'processing_time': elapsed
        }

    
    
    scaled_results = []
    for (bbox, text, confidence) in results:
        
        bbox_scaled = [[int(pt[0] / ocr_scale), int(pt[1] / ocr_scale)] for pt in bbox]
        scaled_results.append((bbox_scaled, text, confidence))

    
    scored_results = []
    for (bbox, text, confidence) in scaled_results:
        plate_score = looks_like_plate(text)

        
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        box_area = w * h
        aspect_ratio = w / max(h, 1)

        
        if 1.5 <= aspect_ratio <= 6.0:
            plate_score += 15
        elif 1.0 <= aspect_ratio <= 8.0:
            plate_score += 5

        
        area_ratio = box_area / img_area
        if 0.005 <= area_ratio <= 0.15:
            plate_score += 10
        elif 0.002 <= area_ratio <= 0.3:
            plate_score += 5

        
        plate_score += confidence * 10

        scored_results.append({
            'bbox': bbox,
            'text': text,
            'confidence': confidence,
            'plate_score': plate_score,
            'bounds': (x, y, w, h)
        })

    
    scored_results.sort(key=lambda r: r['plate_score'], reverse=True)

    
    best = scored_results[0]
    plate_found = best['plate_score'] >= 30  

    
    plate_texts = []
    plate_regions = []

    if plate_found:
        
        
        best_clean = re.sub(r'[^A-Z0-9]', '', best['text'].upper())
        best_has_mix = bool(re.search(r'\d', best_clean)) and bool(re.search(r'[A-Z]', best_clean))
        best_is_complete = (7 <= len(best_clean) <= 10) and best_has_mix and best['plate_score'] >= 45

        if best_is_complete:
            
            
            plate_texts = [best]
            plate_regions = [best['bbox']]
        else:
            
            bx, by, bw, bh = best['bounds']
            plate_center_y = by + bh / 2

            for r in scored_results:
                rx, ry, rw, rh = r['bounds']
                r_center_y = ry + rh / 2
                r_text_clean = re.sub(r'[^A-Z]', '', r['text'].upper())

                
                if r_text_clean in BRAND_NAMES or r_text_clean in EMBLEM_TEXT:
                    continue

                
                
                vertical_dist = abs(r_center_y - plate_center_y)
                horizontal_dist = min(abs(rx - (bx + bw)), abs(bx - (rx + rw)))
                
                
                
                
                
                height_ratio = r['bounds'][3] / max(bh, 1)
                
                if vertical_dist < bh * 0.5 and (r['plate_score'] >= 20 or horizontal_dist < bw * 0.15):
                    if 0.5 <= height_ratio <= 1.5:
                        plate_texts.append(r)
                        plate_regions.append(r['bbox'])

            
            if not plate_texts:
                plate_texts = [best]
                plate_regions = [best['bbox']]

            
            plate_texts.sort(key=lambda r: r['bounds'][0])

            
            unique_plate_texts = []
            for r in plate_texts:
                is_redundant = False
                r_clean = re.sub(r'[^A-Z0-9]', '', r['text'].upper())
                r_x, r_y, r_w, r_h = r['bounds']
                for existing in unique_plate_texts:
                    ex_clean = re.sub(r'[^A-Z0-9]', '', existing['text'].upper())
                    ex_x, ex_y, ex_w, ex_h = existing['bounds']

                    
                    is_substr = (r_clean in ex_clean or ex_clean in r_clean) and len(r_clean) > 0

                    
                    
                    if not is_substr and len(r_clean) > 3 and len(ex_clean) > 3:
                        shorter = min(len(r_clean), len(ex_clean))
                        matches = sum(1 for a, b in zip(r_clean, ex_clean) if a == b)
                        is_substr = (matches / shorter) >= 0.6

                    
                    if not is_substr:
                        xi1 = max(r_x, ex_x)
                        yi1 = max(r_y, ex_y)
                        xi2 = min(r_x + r_w, ex_x + ex_w)
                        yi2 = min(r_y + r_h, ex_y + ex_h)
                        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                        union = r_w * r_h + ex_w * ex_h - inter
                        iou = inter / max(union, 1)
                        is_substr = iou > 0.3

                    if is_substr:
                        
                        if r['plate_score'] > existing['plate_score']:
                            unique_plate_texts.remove(existing)
                            unique_plate_texts.append(r)
                        is_redundant = True
                        break
                if not is_redundant:
                    unique_plate_texts.append(r)
            plate_texts = unique_plate_texts

    combined_plate_text = " ".join([r['text'] for r in plate_texts]) if plate_found else ""

    
    if plate_found and combined_plate_text:
        corrected_text = fix_plate_text(combined_plate_text)
    else:
        corrected_text = combined_plate_text

    
    all_detections = []
    for r in scored_results:
        is_plate = bool(r in plate_texts) if plate_found else False
        
        if plate_found and not is_plate:
            continue
        all_detections.append({
            'text': str(r['text']),
            'confidence': float(round(r['confidence'] * 100, 1)),
            'plate_score': float(round(r['plate_score'], 1)),
            'is_plate': is_plate
        })

    
    
    annotated = img.copy()

    for r in scored_results:
        pts = np.array(r['bbox'], dtype=np.int32)
        is_plate_region = plate_found and r in plate_texts

        if is_plate_region:
            
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)
            cv2.putText(annotated, corrected_text,
                       (pts[0][0], pts[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    
    strategy = "EasyOCR Text Detection"
    if plate_found:
        strategy += f" (plate score: {best['plate_score']:.0f})"

    
    cropped_b64 = None
    if plate_found and plate_regions:
        
        all_pts = np.vstack([np.array(r, dtype=np.int32) for r in plate_regions])
        px, py, pw, ph = cv2.boundingRect(all_pts)

        
        pad = 10
        py1 = max(0, py - pad)
        py2 = min(img_height, py + ph + pad)
        px1 = max(0, px - pad)
        px2 = min(img_width, px + pw + pad)

        cropped = img[py1:py2, px1:px2]
        if cropped.size > 0:
            cropped_b64 = encode_image(cropped)

    
    pso_validation = ""
    if plate_found and cropped_b64:
        pso_result = pso_recognize_digits(cropped)
        if pso_result:
            pso_validation = f" (PSO-ANN digits: {pso_result})"
            print(f"[DEBUG] PSO-ANN recognized digits: {pso_result}")
            
            
            clean_corrected = re.sub(r'[^A-Z0-9]', '', corrected_text)
            
            
            if len(pso_result) >= 2 and len(clean_corrected) >= 4:
                if clean_corrected[2:4] == "00" and pso_result[0:2] != "00":
                    
                    corrected_text = corrected_text[:2] + pso_result[0] + pso_result[1] + corrected_text[4:]
            
            
            
            
            pass

    elapsed = round(time.time() - start_time, 2)

    return {
        'success': True,
        'plate_found': plate_found,
        'strategy': strategy + pso_validation,
        'detections': all_detections,
        'combined_text': corrected_text if plate_found else "No license plate detected",
        'raw_ocr_text': combined_plate_text if plate_found else "",
        'original_image': encode_image(img),
        'edge_image': encode_image(edged),
        'cropped_image': cropped_b64,
        'result_image': encode_image(annotated),
        'processing_time': elapsed
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Use JPG, PNG, BMP, or WebP.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = process_image(filepath)
        return Response(
            json.dumps(result, cls=NumpyEncoder),
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  ALPR Web Application")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
