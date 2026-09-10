import re

# Dictionaries from app.py
LETTER_TO_DIGIT = {
    'I': '1', 'l': '1', 'L': '1',
    'O': '0', 'o': '0',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8', 'b': '8',
    'G': '6', 'g': '6',
    'T': '7',
    'A': '4',
}

DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
    '6': 'G',
    '4': 'A',
}

def fix_plate_text(text):
    """
    Auto-correct common OCR misreads using National plate format rules (Indian and UK).
    """
    # Remove any non-alphanumeric characters and convert to uppercase
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())

    if len(clean) < 4:
        return text

    # 1. Identify format (UK vs Indian)
    # UK format: LLDD LLL (7 chars)
    is_uk = len(clean) == 7 and (clean[:2].isalpha() or any(c in DIGIT_TO_LETTER for c in clean[:2]))

    # 2. Strip "IND" prefix for Indian plates
    if not is_uk:
        if clean.startswith("IND"):
            clean = clean[3:]
        elif clean.startswith("IN"):
            clean = clean[2:]
        # Strip leading residue after IND (like a single '1' or 'D')
        if len(clean) > 8 and clean[0] in ('1', 'D', 'I') and clean[1:3].isalpha():
            clean = clean[1:]

    corrected = list(clean)

    if is_uk:
        # UK Format: LLDD LLL
        # Positions 0-1: Letters
        for i in range(2):
            if corrected[i] in DIGIT_TO_LETTER:
                corrected[i] = DIGIT_TO_LETTER[corrected[i]]
        # Positions 2-3: Digits
        for i in range(2, 4):
            if corrected[i] == 'Z':
                corrected[i] = '7' # UK specific: Z is usually 7 in digits
            elif corrected[i] in LETTER_TO_DIGIT:
                corrected[i] = LETTER_TO_DIGIT[corrected[i]]
        # Positions 4-6: Letters
        for i in range(4, 7):
            if corrected[i] in DIGIT_TO_LETTER:
                corrected[i] = DIGIT_TO_LETTER[corrected[i]]
        
        res = "".join(corrected)
        return res[:4] + " " + res[4:]

    else:
        # Indian Format: SS DD SS DDDD
        # Positions 0-1: State Code (Letters)
        for i in range(min(2, len(corrected))):
            if corrected[i] in DIGIT_TO_LETTER:
                corrected[i] = DIGIT_TO_LETTER[corrected[i]]
                
        # Positions 2-?: District Code (Digits)
        district_end = 2
        for i in range(2, min(4, len(corrected))):
            if corrected[i] in LETTER_TO_DIGIT:
                corrected[i] = LETTER_TO_DIGIT[corrected[i]]
                district_end = i + 1
            elif corrected[i].isdigit():
                district_end = i + 1
            else:
                break
                
        # Positions district_end to ?: Series (Letters)
        digit_start = len(corrected)
        for i in range(len(corrected) - 1, district_end - 1, -1):
            if corrected[i].isdigit() or corrected[i] in LETTER_TO_DIGIT:
                digit_start = i
            else:
                break
                
        # Characters between district_end and digit_start: Letters
        for i in range(district_end, digit_start):
            if corrected[i] in DIGIT_TO_LETTER:
                corrected[i] = DIGIT_TO_LETTER[corrected[i]]
                
        # Final group: Digits
        for i in range(digit_start, len(corrected)):
            if corrected[i] in LETTER_TO_DIGIT:
                corrected[i] = LETTER_TO_DIGIT[corrected[i]]

        result = "".join(corrected)
        if len(result) >= 7:
            try:
                formatted = result[:2] + ' ' + result[2:district_end] + ' ' + result[district_end:digit_start] + ' ' + result[digit_start:]
                return formatted.strip().replace('  ', ' ')
            except: return result
        return result

# Test cases
test_inputs = [
    "IN D1 HRZGD K 8337",  # India case
    "IND HRZGDK 8337",
    "MH1ZDE1433",
    "LXIZPYD",             # UK case: LX17 PYD (I->1, Z->7)
    "LX 12 PYD",           # UK case: LX17 PYD (2->7 or 2->2)
    "LXIZ PYD",
    "LX1ZPYD"
]

print("-" * 40)
print(f"{'Input':<25} | {'Corrected':<20}")
print("-" * 40)
for inp in test_inputs:
    out = fix_plate_text(inp)
    print(f"{inp:<25} | {out:<20}")
print("-" * 40)
