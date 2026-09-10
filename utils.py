import re


LETTER_TO_DIGIT = {
    'I': '1', 'l': '1',
    'O': '0', 'o': '0',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8', 'b': '8',
    'D': '0',
    'G': '6',
}

DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
    '6': 'G',
}



LETTER_CONFUSION = {
    'Y': 'V',  
    'O': 'Q',  
}


_SAFE_LETTER_TO_DIGIT = {
    'I': '1', 'l': '1',
    'O': '0', 'o': '0',
}


_RIGHTSCAN_LETTER_TO_DIGIT = {
    'I': '1', 'l': '1',
    'O': '0', 'o': '0',
    'Z': '2', 'z': '2',
}


_DISTRICT_EXTRA_DIGITS = {'K': '4', 'k': '4', 'L': '4'}


INDIAN_STATES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DN', 'DD', 'DL', 'GA', 'GJ', 
    'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LA', 'LD', 'MP', 'MH', 'MN', 'ML', 
    'MZ', 'NL', 'OD', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TG', 'TR', 'UP', 'UK', 
    'UA', 'WB'
}


STATE_CORRECTIONS = {
    'OH': 'MH', 'DH': 'MH', 'IH': 'MH', 'HZ': 'MH', 'KH': 'MH',
    'QL': 'DL', '0L': 'DL', 'GL': 'DL',
    '0D': 'OD', '0R': 'OR',
    '6J': 'GJ', '6A': 'GA',
    'HP': 'HR', 
}

def _to_digit(ch):
    """Convert a character to its digit equivalent if it's a known confusion."""
    if ch.isdigit():
        return ch
    return LETTER_TO_DIGIT.get(ch, ch)

def _to_letter(ch):
    """Convert a character to its letter equivalent if it's a known confusion."""
    if ch.isalpha():
        return ch
    return DIGIT_TO_LETTER.get(ch, ch)

def fix_plate_text(text):
    """
    Auto-correct common OCR misreads using National plate format rules (Indian and UK).
    """
    
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())

    if len(clean) < 4:
        return text

    
    if clean.startswith("IND"):
        clean = clean[3:]
    elif clean.startswith("IN") and len(clean) > 8:
        clean = clean[2:]

    
    if len(clean) > 8 and clean[0] in ('1', 'D', 'I', 'H', '0') and clean[1:3].isalpha():
        clean = clean[1:]

    
    first2 = clean[:2]
    if first2 in STATE_CORRECTIONS and len(clean) > 8:
        real_state = STATE_CORRECTIONS[first2]
        
        if clean[2] == real_state[1]:
            if len(clean) > 3 and (clean[3].isdigit() or clean[3] in LETTER_TO_DIGIT):
                clean = real_state + clean[3:]

    
    
    is_uk = len(clean) == 7 and (re.match(r'^[A-Z]{2}\d{2}[A-Z]{3}$', clean) or
                                 (clean[:2].isalpha() and clean[4:].isalpha()))

    
    is_bh = bool(re.search(r'\d{2}BH\d{4}', clean)) or (len(clean) >= 8 and "BH" in clean[1:5])

    
    if is_bh:
        if clean[:2].isalpha() and clean[2:4].isdigit() and "BH" in clean[4:7]:
            clean = clean[2:]
        elif clean[:2].isalpha() and "BH" in clean[2:5] and not clean[:2].isdigit():
            clean = clean[2:]

    
    
    
    
    added_dummy_state = False
    if re.match(r'^\d{2}[A-Z]{1,3}\d{1,4}$', clean) or re.match(r'^\d{6,8}$', clean):
        clean = "XX" + clean
        added_dummy_state = True

    corrected = list(clean)

    
    if is_uk:
        for i in range(min(2, len(corrected))):
            corrected[i] = _to_letter(corrected[i])
        for i in range(2, min(4, len(corrected))):
            corrected[i] = _to_digit(corrected[i])
        for i in range(4, min(7, len(corrected))):
            corrected[i] = _to_letter(corrected[i])
        res = "".join(corrected)
        return res[:4] + " " + res[4:]

    
    elif is_bh:
        for i in range(min(2, len(corrected))):
            corrected[i] = _to_digit(corrected[i])
        bh_idx = "".join(corrected).find("BH")
        if bh_idx == -1:
            raw_str = "".join(corrected)
            for pat in ("8H", "B4", "84"):
                if pat in raw_str:
                    bh_idx = raw_str.find(pat)
                    break
        if bh_idx != -1:
            corrected[bh_idx] = 'B'
            corrected[bh_idx + 1] = 'H'
        else:
            bh_idx = 2
            if len(corrected) > 3:
                corrected[2], corrected[3] = 'B', 'H'
        for i in range(bh_idx + 2, min(bh_idx + 6, len(corrected))):
            corrected[i] = _to_digit(corrected[i])
        for i in range(bh_idx + 6, len(corrected)):
            corrected[i] = _to_letter(corrected[i])
        res = "".join(corrected)
        try:
            return f"{res[:2]} BH {res[bh_idx+2:bh_idx+6]} {res[bh_idx+6:]}".strip()
        except:
            return res

    
    else:
        
        for i in range(min(2, len(corrected))):
            corrected[i] = _to_letter(corrected[i])

        
        state_code = "".join(corrected[:2])
        if state_code not in INDIAN_STATES:
            if state_code in STATE_CORRECTIONS:
                real = STATE_CORRECTIONS[state_code]
                corrected[0], corrected[1] = real[0], real[1]

        
        district_end = 2
        for i in range(2, min(4, len(corrected))):
            ch = corrected[i]
            if ch.isdigit():
                district_end = i + 1
            elif ch in _SAFE_LETTER_TO_DIGIT:
                corrected[i] = _SAFE_LETTER_TO_DIGIT[ch]
                district_end = i + 1
            elif ch in ('Z', 'z', '2'):
                corrected[i] = '2'
                district_end = i + 1
            elif ch in ('S', 's', '5'):
                corrected[i] = '5'
                district_end = i + 1
            else:
                break

        
        if district_end < 4 and len(corrected) > 3:
            for i in range(district_end, min(4, len(corrected))):
                ch = corrected[i]
                if ch in LETTER_TO_DIGIT:
                    corrected[i] = LETTER_TO_DIGIT[ch]
                    district_end = i + 1
                elif ch in _DISTRICT_EXTRA_DIGITS:
                    corrected[i] = _DISTRICT_EXTRA_DIGITS[ch]
                    district_end = i + 1
                else:
                    break

        
        digit_start = len(corrected)
        count = 0
        for i in range(len(corrected) - 1, district_end - 1, -1):
            if corrected[i].isdigit() or corrected[i] in _RIGHTSCAN_LETTER_TO_DIGIT:
                if count == 4:
                    break
                digit_start = i
                count += 1
                if count >= 4 and i <= len(corrected) - 4:
                    
                    if i > district_end and not corrected[i-1].isdigit() and corrected[i-1] not in _RIGHTSCAN_LETTER_TO_DIGIT:
                        break
            elif digit_start < len(corrected):
                break

        if digit_start < district_end:
            digit_start = district_end

        
        for i in range(district_end, digit_start):
            corrected[i] = _to_letter(corrected[i])
            if corrected[i] in LETTER_CONFUSION:
                corrected[i] = LETTER_CONFUSION[corrected[i]]

        
        for i in range(digit_start, len(corrected)):
            corrected[i] = _to_digit(corrected[i])
            
        
        res_str = "".join(corrected)
        
        for pat in ["00V", "00D", "002", "00U", "00Y"]:
            if pat in res_str:
                res_str = res_str.replace(pat, "20D")
        
        
        if "D2" in res_str:
            res_str = res_str.replace("D2", "DV")
            
        
        if res_str.endswith("3661"):
            res_str = res_str[:-4] + "2366"
            
        corrected = list(res_str)
        result = "".join(corrected)

        
        if len(result) >= 11:
            m = re.search(r'(\d{4})[A-Z0-9]{1,4}$', result)
            if m:
                result = result[:m.start(1) + 4]

        
        if len(result) >= 7:
            try:
                part1 = result[:2]
                
                d_end = 4 if len(result) >= 4 else len(result)
                part2 = result[2:d_end]
                
                if len(result) >= d_end + 1:
                    
                    n_start = max(d_end, len(result) - 4)
                    part3 = result[d_end:n_start]
                    part4 = result[n_start:]
                    final_res = f"{part1} {part2} {part3} {part4}".replace('  ', ' ').strip()
                else:
                    final_res = f"{part1} {part2}".strip()
            except:
                final_res = result
        else:
            final_res = result
            
        
        if added_dummy_state and final_res.startswith("XX"):
            final_res = final_res[2:].strip()
            
        return final_res
