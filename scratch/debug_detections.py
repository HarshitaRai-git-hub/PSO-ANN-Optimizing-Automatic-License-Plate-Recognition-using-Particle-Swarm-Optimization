import requests
import json

files = {'image': open('car_image.jpg', 'rb')}
r = requests.post('http://localhost:5000/upload', files=files)
d = r.json()

print('Plate found:', d.get('plate_found'))
print('Combined text:', d.get('combined_text'))
print('Raw OCR:', d.get('raw_ocr_text'))
print()
print('All detections:')
for det in d.get('detections', []):
    text = det['text']
    conf = det['confidence']
    score = det['plate_score']
    is_plate = det['is_plate']
    print(f'  text={text!r:20s}  conf={conf:5.1f}%  score={score:5.1f}  plate={is_plate}')
