"""
download_dataset.py — PlantVillage Dataset Downloader
======================================================
Run this script ONCE before training to download the dataset.

WHAT IS PLANVILLAGE?
  PlantVillage is a publicly available dataset of 54,000+ leaf images
  across 38 disease classes from 14 crop types. It is the standard
  benchmark dataset for plant disease detection research.
  Source: Hughes & Salathé, 2015 (Penn State University)

USAGE:
  C:\\hs_venv\\Scripts\\python.exe download_dataset.py

The dataset will be saved to:  harvest-savior-ai/dataset/PlantVillage/
Each class has its own sub-folder named after the disease label.

REQUIREMENTS:
  - A Kaggle account (free) at https://www.kaggle.com
  - Your kaggle.json API key placed at C:\\Users\\<you>\\.kaggle\\kaggle.json
    (Download it from: https://www.kaggle.com/settings → API → Create New Token)
"""

import os
import sys
import zipfile
import subprocess

# ── Where the dataset will be stored ─────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
PLANVILLAGE_DIR = os.path.join(DATASET_DIR, 'PlantVillage')

# ── Kaggle dataset identifier ─────────────────────────────────────────────────
KAGGLE_DATASET = "emmarex/plantdisease"


def check_kaggle_credentials():
    """
    Checks that the Kaggle API key file exists before attempting to download.
    The file must be at: C:\\Users\\<username>\\.kaggle\\kaggle.json
    """
    kaggle_dir  = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json):
        print("=" * 60)
        print("  Kaggle API key not found!")
        print("=" * 60)
        print()
        print("  Steps to get your API key:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print("  4. A file called kaggle.json will download")
        print(f"  5. Move it to: {kaggle_json}")
        print()
        print("  Then run this script again.")
        sys.exit(1)

    print(f"✅ Kaggle credentials found at: {kaggle_json}")


def download_planvillage():
    """
    Uses the Kaggle CLI to download the PlantVillage dataset.
    Equivalent to running: kaggle datasets download -d emmarex/plantdisease
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_path = os.path.join(DATASET_DIR, "plantdisease.zip")

    if os.path.exists(PLANVILLAGE_DIR) and len(os.listdir(PLANVILLAGE_DIR)) > 0:
        print(f"✅ Dataset already downloaded at: {PLANVILLAGE_DIR}")
        _print_class_summary()
        return

    print("⬇  Downloading PlantVillage dataset from Kaggle (~1.5 GB)...")
    print("   This will take a few minutes depending on your connection.")
    print()

    result = subprocess.run([
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", KAGGLE_DATASET,
        "--path", DATASET_DIR,
    ], capture_output=False)

    if result.returncode != 0:
        print()
        print("ERROR: Kaggle download failed.")
        print("Make sure kaggle is installed:  pip install kaggle")
        sys.exit(1)

    # ── Extract the zip ───────────────────────────────────────────────────────
    print()
    print("📦 Extracting dataset...")

    # Find the downloaded zip
    zip_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.zip')]
    if not zip_files:
        print("ERROR: No zip file found after download.")
        sys.exit(1)

    zip_path = os.path.join(DATASET_DIR, zip_files[0])
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATASET_DIR)

    os.remove(zip_path)
    print("✅ Extraction complete.")

    _print_class_summary()


def _print_class_summary():
    """Prints the list of class folders found in the dataset directory."""

    # The zip might extract into 'PlantVillage' or 'plant_disease_recognition_dataset'
    # Find whichever folder contains class sub-folders
    candidates = [
        os.path.join(DATASET_DIR, "PlantVillage"),
        os.path.join(DATASET_DIR, "plant_disease_recognition_dataset", "PlantVillage"),
        DATASET_DIR,
    ]

    dataset_root = None
    for c in candidates:
        if os.path.isdir(c):
            subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
            if len(subdirs) >= 10:
                dataset_root = c
                break

    if dataset_root is None:
        print("⚠  Could not locate class folders. Check the dataset directory manually.")
        return

    classes = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])

    total_images = 0
    print()
    print(f"📂 Dataset root: {dataset_root}")
    print(f"   {len(classes)} classes found:")
    print()
    for cls in classes:
        cls_path = os.path.join(dataset_root, cls)
        n = len([f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_images += n
        print(f"   {cls:<55}  {n:>5} images")

    print()
    print(f"   TOTAL: {total_images:,} images across {len(classes)} classes")
    print()
    print(f"✅ Dataset ready.  Detected root: {dataset_root}")
    print()
    print("   UPDATE train_model.py → DATASET_ROOT if this path differs from:")
    print(f"   {os.path.join(os.path.dirname(__file__), 'dataset', 'PlantVillage')}")


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  Harvest Savior — PlantVillage Dataset Downloader")
    print("=" * 60)
    print()
    check_kaggle_credentials()
    download_planvillage()
