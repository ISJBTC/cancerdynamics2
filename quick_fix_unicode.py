#!/usr/bin/env python3
"""
Quick fix for Unicode error in enhanced_superiority_proof.py
"""

import os

def fix_unicode_error():
    """Fix the Unicode encoding issue"""
    
    file_path = 'enhanced_superiority_proof.py'
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the file writing line
    old_line = "with open(report_file, 'w') as f:"
    new_line = "with open(report_file, 'w', encoding='utf-8') as f:"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Also remove problematic Unicode characters
        content = content.replace('✅', '[SUCCESS]')
        content = content.replace('❌', '[FAILED]')
        
        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Fixed Unicode error in {file_path}")
        print(f"✓ Added UTF-8 encoding to file operations")
        print(f"✓ Replaced Unicode characters with ASCII equivalents")
    else:
        print("Could not find the problematic line to fix")

if __name__ == "__main__":
    fix_unicode_error()
    print("\nNow run: python enhanced_superiority_proof.py")