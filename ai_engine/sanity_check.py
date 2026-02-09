"""
sanity_check.py
---------------
Milestone 1.1, Task 3: Data Sanity Check
Opens one random image from each class to ensure they're not corrupted.
"""

import os
import random
from PIL import Image

BASE_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
CLASSES = ["Potato___Early_Blight", "Potato___Late_Blight", "Potato___Healthy"]


def main():
    print("🔍 Running Data Sanity Check...\n")
    
    all_good = True
    for class_name in CLASSES:
        folder = os.path.join(BASE_DIR, class_name)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(files) < 50:
            print(f"❌ {class_name}: Only {len(files)} images (need ≥50)")
            all_good = False
            continue
        
        # Try to open a random image
        sample = random.choice(files)
        try:
            img = Image.open(os.path.join(folder, sample))
            img.verify()  # Check for corruption
            img = Image.open(os.path.join(folder, sample))  # Reopen after verify
            width, height = img.size
            print(f"✅ {class_name}: {len(files)} images | Sample: {sample} ({width}x{height})")
        except Exception as e:
            print(f"❌ {class_name}: Corrupted image {sample} → {e}")
            all_good = False
    
    print("\n" + ("="*60))
    if all_good:
        print("🎉 SUCCESS! All images are valid. Ready for training.")
    else:
        print("⚠️  FAILED. Fix the issues above before training.")
    print("="*60)


if __name__ == "__main__":
    main()
