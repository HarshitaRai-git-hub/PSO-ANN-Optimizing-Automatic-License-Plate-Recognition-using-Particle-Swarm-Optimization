import os
import tokenize
import io

def remove_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    result = []
    lines = source.split('\n')
    
    # We will use tokenize to safely remove comments without messing up strings that contain '#'
    # Tokenize needs a readline callable
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        
        last_lineno = 1
        last_col = 0
        
        out = ""
        for tok in tokens:
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            
            if token_type == tokenize.COMMENT:
                pass # skip comment
            else:
                out += token_string
                
            last_lineno = end_line
            last_col = end_col
            
        # Write back
        # Also clean up empty lines that might have been left by comment removal
        clean_lines = []
        for line in out.split('\n'):
            if line.strip() == '' and not line.startswith(' ' * 4): # try not to break too much indentation, actually let's just keep lines unless they are totally blank and were just a comment line
                # wait, out might have blank lines
                pass
                
        # Better simple approach for blank lines:
        # Actually it's fine to just leave the empty lines or we can just filter out lines that are entirely empty if we want
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(out)
            
        print(f"Cleaned {file_path}")
    except Exception as e:
        print(f"Error on {file_path}: {e}")

if __name__ == '__main__':
    for f in os.listdir('.'):
        if f.endswith('.py') and os.path.isfile(f):
            remove_comments(f)
