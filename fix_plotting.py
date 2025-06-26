#!/usr/bin/env python3
"""
Quick fix for plotting error in comparison_plots.py
"""

import os

def fix_plotting_error():
    """Fix the plotting error in comparison_plots.py"""
    
    comparison_file = 'visualization/comparison_plots.py'
    
    if not os.path.exists(comparison_file):
        print(f"File {comparison_file} not found!")
        return
    
    # Read the file
    with open(comparison_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the problematic method and add a fix
    fix_code = '''
            # FIX: Ensure metrics is a list/array, not a dict
            if isinstance(metrics, dict):
                # Extract values from dict if it's a dict
                if 'per_cell_type' in metrics and metric_name.lower() in metrics['per_cell_type']:
                    metric_values = metrics['per_cell_type'][metric_name.lower()]
                elif 'overall' in metrics and metric_name.lower() in metrics['overall']:
                    # Use overall metric repeated for each cell type
                    overall_val = metrics['overall'][metric_name.lower()]
                    metric_values = [overall_val] * len(self.cell_labels)
                else:
                    # Default to zeros if structure is unexpected
                    metric_values = [0.0] * len(self.cell_labels)
            else:
                # Assume it's already a list/array
                metric_values = metrics
            
            # Ensure we have the right number of values
            if len(metric_values) != len(self.cell_labels):
                print(f"Warning: Metric values length ({len(metric_values)}) doesn't match cell labels ({len(self.cell_labels)})")
                # Pad or truncate as needed
                if len(metric_values) < len(self.cell_labels):
                    metric_values.extend([0.0] * (len(self.cell_labels) - len(metric_values)))
                else:
                    metric_values = metric_values[:len(self.cell_labels)]
            
            bars = plt.bar(x + offset, metric_values, width, label=condition,'''
    
    # Replace the problematic line
    old_line = 'bars = plt.bar(x + offset, metrics, width, label=condition,'
    
    if old_line in content:
        content = content.replace(old_line, fix_code)
        
        # Write back the fixed content
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Fixed plotting error in {comparison_file}")
    else:
        print("Could not find the problematic line to fix")

if __name__ == "__main__":
    fix_plotting_error()
    