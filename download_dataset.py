import os
import zipfile
import shutil

"""
Script to download and extract the BSL MediaPipe dataset from Kaggle.
This dataset contains pre-extracted MediaPipe hand landmarks for BSL alphabet and numbers.
"""

print("=" * 60)
print("BSL MediaPipe Dataset Download & Setup")
print("=" * 60)

# Configuration
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "bsl_data")
KAGGLE_DIR = os.path.join(DATA_DIR, "kaggle_dataset")

# Dataset information
DATASET_NAME = "erentatepe/bsl-numbers-and-alphabet-hand-position-for-mediapipe"
DATASET_SIZE = "~47.7 GB"

print(f"\nProject Directory: {PROJECT_DIR}")
print(f"Data Directory: {DATA_DIR}")
print(f"\n⚠️  DATASET SIZE: {DATASET_SIZE}")
print("Ensure you have sufficient disk space before proceeding.\n")

# Check if kaggle is installed
try:
    import kaggle
    print("✓ Kaggle API found")
except ImportError:
    print("✗ Kaggle API not installed")
    print("\nPlease install it with:")
    print("  pip install kaggle")
    print("\nThen configure your API credentials:")
    print("  1. Go to https://www.kaggle.com/settings/account")
    print("  2. Click 'Create New API Token'")
    print("  3. Save kaggle.json to C:\\Users\\ahann\\.kaggle\\")
    exit(1)

# Check if API credentials exist
KAGGLE_CONFIG = os.path.expanduser("~/.kaggle/kaggle.json")
if not os.path.exists(KAGGLE_CONFIG):
    print("✗ Kaggle API credentials not found")
    print(f"\nExpected location: {KAGGLE_CONFIG}")
    print("\nTo set up:")
    print("  1. Go to https://www.kaggle.com/settings/account")
    print("  2. Click 'Create New API Token'")
    print("  3. Save kaggle.json to the location above")
    exit(1)

print(f"✓ Kaggle API credentials found at {KAGGLE_CONFIG}")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KAGGLE_DIR, exist_ok=True)

print(f"\n{'=' * 60}")
print("Step 1: Downloading Dataset from Kaggle")
print(f"{'=' * 60}")
print(f"Dataset: {DATASET_NAME}")
print(f"This may take a while depending on your internet connection...")

# Download dataset
print("\nDownloading...")
os.system(f'kaggle datasets download -d {DATASET_NAME} -p "{KAGGLE_DIR}"')

print(f"\n{'=' * 60}")
print("Step 2: Extracting Dataset")
print(f"{'=' * 60}")

# Find the downloaded zip file
zip_files = [f for f in os.listdir(KAGGLE_DIR) if f.endswith('.zip')]

if not zip_files:
    print("✗ No zip file found. Download may have failed.")
    print(f"Check {KAGGLE_DIR} for any error messages.")
    exit(1)

zip_path = os.path.join(KAGGLE_DIR, zip_files[0])
print(f"Found: {zip_files[0]}")
print("Extracting... (this may take several minutes)")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(KAGGLE_DIR)

print("✓ Extraction complete")

# Remove zip file to save space
os.remove(zip_path)
print(f"✓ Removed zip file to save space")

print(f"\n{'=' * 60}")
print("Step 3: Dataset Information")
print(f"{'=' * 60}")

# List extracted contents
print(f"\nExtracted files in {KAGGLE_DIR}:")
for root, dirs, files in os.walk(KAGGLE_DIR):
    level = root.replace(KAGGLE_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    sub_indent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{sub_indent}{file}')
    if len(files) > 5:
        print(f'{sub_indent}... and {len(files) - 5} more files')

print(f"\n{'=' * 60}")
print("✓ Download Complete!")
print(f"{'=' * 60}")

print("\nDataset location:", KAGGLE_DIR)
print("\nNext steps:")
print("  1. Run 'python convert_kaggle_dataset.py' to convert to project format")
print("  2. Run 'python preprocess_data.py' to prepare for training")
print("  3. Run 'python train.py' to train the model")

print("\n" + "=" * 60)
