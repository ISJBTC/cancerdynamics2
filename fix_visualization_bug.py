#!/usr/bin/env python3
"""
Quick fix for the visualization formatting error
"""
import re

def fix_comparison_plots():
    """Fix the string formatting error"""
    
    file_path = 'visualization/comparison_plots.py'
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and fix the problematic line
    # Look for: plt.annotate(f'{value:.3f}',
    # The issue is 'value' might be a string, not a number
    
    old_pattern = r"plt\.annotate\(f'\{value:.3f\}',"
    new_pattern = "plt.annotate(f'{float(value):.3f}' if isinstance(value, (int, float)) else f'{value}',"
    
    # Apply the fix
    content = re.sub(old_pattern, new_pattern, content)
    
    # Also fix any other similar patterns
    content = re.sub(
        r"plt\.annotate\(f'\{(.+?):.3f\}',",
        r"plt.annotate(f'{float(\1):.3f}' if isinstance(\1, (int, float)) else f'{\1}',",
        content
    )
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed visualization bug in comparison_plots.py")

if __name__ == "__main__":
    fix_comparison_plots()