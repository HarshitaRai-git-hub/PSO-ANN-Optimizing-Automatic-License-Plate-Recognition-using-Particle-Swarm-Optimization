import cv2
import numpy as np


img = np.ones((600, 800, 3), dtype=np.uint8) * 100


plate_top_left = (250, 400)
plate_bottom_right = (550, 480)
cv2.rectangle(img, plate_top_left, plate_bottom_right, (0, 255, 255), -1)


cv2.rectangle(img, plate_top_left, plate_bottom_right, (0, 0, 0), 4)


font = cv2.FONT_HERSHEY_SIMPLEX
text = "DL 8C N 1234"
text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
text_x = plate_top_left[0] + (plate_bottom_right[0] - plate_top_left[0] - text_size[0]) // 2
text_y = plate_top_left[1] + (plate_bottom_right[1] - plate_top_left[1] + text_size[1]) // 2
cv2.putText(img, text, (text_x, text_y), font, 1.5, (0, 0, 0), 3)


cv2.circle(img, (200, 300), 50, (200, 200, 200), -1) 
cv2.circle(img, (600, 300), 50, (200, 200, 200), -1) 
cv2.rectangle(img, (300, 320), (500, 380), (50, 50, 50), -1) 


cv2.imwrite('test_car.jpg', img)
print("Saved test_car.jpg")
