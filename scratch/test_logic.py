import os
import sys
import cv2
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

try:
    from app import process_image, get_reader
    print("Imports successful")
    
    # Generate test image if not exists
    if not os.path.exists('test_car.jpg'):
        import generate_test_image
        print("Generated test image")
    
    # Test process_image
    print("Testing process_image...")
    result = process_image('test_car.jpg')
    
    if result.get('success'):
        print(f"Success! Plate found: {result.get('plate_found')}")
        print(f"Text detected: {result.get('combined_text')}")
    else:
        print(f"Error: {result.get('error')}")
        
except Exception as e:
    import traceback
    traceback.print_exc()
