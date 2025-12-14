#!/usr/bin/env python3
"""
Fix duplicate div tags in HTML
"""

template_file = "C:/Naren/Working/Source/GitHubRepo/Claude_Trade/elder-trading-ibkr/backend/templates/index.html"

print("Reading file...")
with open(template_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Fixing duplicate divs...")
output_lines = []
skip_next = False

for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue

    # Remove duplicate div at line 1048-1049
    if i == 1047 and lines[i].strip() == '<div>' and i+1 < len(lines) and lines[i+1].strip() == '<div>':
        output_lines.append(lines[i])  # Keep first div
        skip_next = True  # Skip second div
        continue

    # Remove duplicate div at line 1058-1059
    if i == 1057 and lines[i].strip() == '<div>' and i+1 < len(lines) and lines[i+1].strip() == '<div>':
        output_lines.append(lines[i])  # Keep first div
        skip_next = True  # Skip second div
        continue

    output_lines.append(line)

print("Writing file...")
with open(template_file, 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print("\n[OK] HTML structure fixed!")
