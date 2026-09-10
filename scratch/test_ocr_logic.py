import sys
import os
import re

# Add parent dir to path to import app
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import fix_plate_text

test_cases = [
    # (Input,                    Expected,           Description)
    ("MH20DV2366",              "MH 20 DV 2366",    "Clean Indian plate"),
    ("OH HZ0DV 2366 1GC",      "MH 20 DV 2366",    "Screenshot case: OH->MH, split H, trailing 1GC"),
    ("KL01BT4567",             "KL 01 BT 4567",    "Kerala plate, 0 in district"),
    ("IND MH12AB1234",         "MH 12 AB 1234",    "IND prefix strip"),
    ("MH12AB12341GC",          "MH 12 AB 1234",    "Trailing badge noise 1GC"),
    ("DL 8C N 1234",           "DL 8C N 1234",     "Delhi single-digit district"),
    ("OH 12 AB 1234",          "MH 12 AB 1234",    "OH -> MH state correction"),
    ("KA01MG1234",             "KA 01 MG 1234",    "Karnataka plate"),
    ("MH20DV2366",             "MH 20 DV 2366",    "Direct clean plate"),
]

print("Running OCR logic tests...")
print("=" * 60)

passed = 0
failed = 0
for inp, expected, desc in test_cases:
    actual = fix_plate_text(inp)
    status = "PASS" if actual == expected else "FAIL"
    if actual == expected:
        passed += 1
        print(f"  [PASS] {desc}")
    else:
        failed += 1
        print(f"  [FAIL] {desc}")
        print(f"         Input:    '{inp}'")
        print(f"         Expected: '{expected}'")
        print(f"         Actual:   '{actual}'")
    print()

print("=" * 60)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
