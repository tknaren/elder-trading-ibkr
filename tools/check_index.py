from pathlib import Path
p = Path('backend/templates/index.html')
s = p.read_text()
# Extract script content
start = s.find('<script>')
end = s.rfind('</script>')
script = s[start+8:end] if start != -1 and end != -1 else s
counts = {
    'backticks': script.count('`'),
    'single_quotes': script.count("'"),
    'double_quotes': script.count('"'),
    'open_brace': script.count('{'),
    'close_brace': script.count('}'),
    'open_paren': script.count('('),
    'close_paren': script.count(')'),
    'functions': script.count('function ')
}
print(counts)
if counts['backticks'] % 2 != 0:
    print('UNMATCHED backticks')
if counts['open_brace'] != counts['close_brace']:
    print('BRACE MISMATCH', counts['open_brace'], counts['close_brace'])
if '${`' in script:
    print('Found ${` pattern')
idx = script.find('function backtestView')
if idx != -1:
    start_line = script[:idx].count('\n')
    lines = script.splitlines()
    for i in range(max(0, start_line-5), min(len(lines), start_line+200)):
        print(f"{i+1}: {lines[i]}")
else:
    print('backtestView not found')
