"""
Label Fixer for HERITAGEDEFECT Dataset
======================================
Some labels in this dataset mix detection (bbox) and segmentation (polygon) rows.
This script normalizes ALL labels to segmentation format by converting any 
bbox-only rows (5 values) into rectangle polygons, making the dataset consistent.
"""

import os
from pathlib import Path

def fix_label_file(label_path):
    """Fix a single label file by converting bbox rows to polygon format."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    has_bbox = False
    has_seg = False
    fixed_lines = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        
        num_values = len(parts) - 1  # exclude class id
        
        if num_values == 4:
            # This is a bbox row: class_id cx cy w h
            # Convert to polygon (rectangle): class_id x1 y1 x2 y1 x2 y2 x1 y2
            has_bbox = True
            cls_id = parts[0]
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            # Clamp to [0, 1]
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            fixed_lines.append(f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}\n")
        elif num_values > 4:
            # This is already a segmentation row
            has_seg = True
            fixed_lines.append(line)
        else:
            # Invalid row, skip
            continue
    
    if has_bbox:
        with open(label_path, 'w') as f:
            f.writelines(fixed_lines)
        return True
    return False

def fix_all_labels(dataset_dir):
    """Fix all label files in the dataset."""
    dataset_dir = Path(dataset_dir)
    total_fixed = 0
    total_files = 0
    
    for split in ['train', 'valid', 'test']:
        label_dir = dataset_dir / split / 'labels'
        if not label_dir.exists():
            continue
        
        label_files = list(label_dir.glob('*.txt'))
        split_fixed = 0
        
        for label_file in label_files:
            total_files += 1
            if fix_label_file(label_file):
                split_fixed += 1
                total_fixed += 1
        
        print(f"  {split}: Fixed {split_fixed}/{len(label_files)} label files")
    
    print(f"\n  Total: Fixed {total_fixed}/{total_files} label files")
    return total_fixed

if __name__ == "__main__":
    dataset_dir = Path(__file__).parent
    print("🔧 Fixing mixed label formats (bbox → polygon conversion)...\n")
    fixed = fix_all_labels(dataset_dir)
    if fixed > 0:
        print(f"\n✅ Fixed {fixed} files. All labels are now in segmentation format.")
    else:
        print("\n✅ All labels were already in correct format.")
