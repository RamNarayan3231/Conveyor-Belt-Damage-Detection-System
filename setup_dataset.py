"""
Helper script to organize your dataset for training
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np

def organize_dataset(source_dir, output_dir='organized_dataset'):
    """
    Organize scattered image and label files into proper YOLO structure
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = []
    
    for ext in image_extensions:
        images.extend(source_path.rglob(f'*{ext}'))
    
    print(f"Found {len(images)} images")
    
    if len(images) == 0:
        print("\n❌ No images found! Please provide the correct source directory.")
        print(f"Looking in: {source_path.absolute()}")
        return False
    
    # Copy images and their corresponding labels
    copied_count = 0
    for img_path in images:
        # Copy image
        dest_img = output_path / 'images' / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # Look for label file (same name with .txt extension)
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            dest_label = output_path / 'labels' / label_path.name
            shutil.copy2(label_path, dest_label)
            copied_count += 1
        else:
            # Check for label in same directory with different case
            possible_labels = list(img_path.parent.glob(f"{img_path.stem}.*.txt"))
            if possible_labels:
                shutil.copy2(possible_labels[0], output_path / 'labels' / f"{img_path.stem}.txt")
                copied_count += 1
            else:
                print(f"Warning: No label found for {img_path.name}")
    
    print(f"\n✅ Organized {copied_count} images with labels")
    print(f"Output directory: {output_path.absolute()}")
    
    return True

def main():
    print("=" * 60)
    print("Dataset Organizer for Conveyor Belt Damage Detection")
    print("=" * 60)
    
    # Ask for source directory
    source_dir = input("\n📁 Enter the path to your dataset directory: ").strip()
    
    if not source_dir:
        print("No directory provided. Using current directory.")
        source_dir = "."
    
    if not Path(source_dir).exists():
        print(f"❌ Directory not found: {source_dir}")
        return
    
    # Organize dataset
    success = organize_dataset(source_dir, 'organized_dataset')
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Dataset organized successfully!")
        print("\nNow you can train the model using:")
        print(f"python backend/train.py --dataset_dir organized_dataset --epochs 50")
        print("=" * 60)

if __name__ == "__main__":
    main()