#!/usr/bin/env python3
"""
Quick fix for Unicode encoding issues in analysis/summary.py
"""

import os

def fix_summary_file():
    """Fix the Unicode issue in summary.py"""
    
    summary_file = 'analysis/summary.py'
    
    if not os.path.exists(summary_file):
        print(f"File {summary_file} not found!")
        return
    
    # Read the file
    with open(summary_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace problematic Unicode characters
    fixes = [
        ('α', 'alpha'),  # Replace Greek alpha with word 'alpha'
        ("'charmap'", "'utf-8'"),  # Force UTF-8 encoding
        ('with open(filepath, \'w\')', 'with open(filepath, \'w\', encoding=\'utf-8\')'),
        ('with open(filepath, \'w\') as f:', 'with open(filepath, \'w\', encoding=\'utf-8\') as f:')
    ]
    
    for old, new in fixes:
        content = content.replace(old, new)
    
    # Add encoding parameter to all file opens that don't have it
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if 'with open(' in line and 'encoding=' not in line and '.write(' not in line:
            if ', \'w\')' in line:
                line = line.replace(', \'w\')', ', \'w\', encoding=\'utf-8\')')
            elif ', \'w\' as f:' in line:
                line = line.replace(', \'w\' as f:', ', \'w\', encoding=\'utf-8\') as f:')
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Write back the fixed content
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed Unicode issues in {summary_file}")

if __name__ == "__main__":
    fix_summary_file()
    