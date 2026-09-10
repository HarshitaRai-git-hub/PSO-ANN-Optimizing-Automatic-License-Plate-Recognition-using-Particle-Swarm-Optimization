import re

def remove_html_css_js_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove HTML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

    # Remove CSS/JS block comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Remove JS single line comments (rough regex: // followed by anything, but not preceded by http: or https:)
    # We can use a negative lookbehind for :
    content = re.sub(r'(?<!:)//.*', '', content)

    # Remove empty lines
    clean_lines = [line for line in content.split('\n') if line.strip() != '']
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(clean_lines))

remove_html_css_js_comments('templates/index.html')
print("Cleaned index.html")
