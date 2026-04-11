# Kaggle BSL Datasets - Analysis & Recommendation

## Dataset 1: BSL Numbers & Alphabet (MediaPipe) ⭐ **RECOMMENDED**
**URL**: https://www.kaggle.com/datasets/erentatepe/bsl-numbers-and-alphabet-hand-position-for-mediapipe

### Key Information:
- **Size**: 34,000 images + CSV files with MediaPipe landmarks
- **Content**: 
  - BSL Numbers: 0-10 (11 classes)
  - BSL Alphabet: **23 letters** (excludes H, J, Y)
  - Total: **34 classes**
- **Format**: 
  - Images (for reference)
  - **2 CSV files with 21 MediaPipe hand landmark positions** ✓
  - Pre-trained SVM models included
- **Usability**: **HIGH** - Already has MediaPipe landmarks extracted!
- **Downloads**: 956
- **Size**: ~47.7 GB

### Advantages:
✅ **Perfect match** - Landmarks already extracted with MediaPipe  
✅ Large dataset size (34,000 samples)  
✅ Ready to use - can skip data collection phase  
✅ Includes both training data and pre-trained models

### Disadvantages:
❌ Missing 3 letters (H, J, Y) - these are dynamic gestures in BSL  
❌ Large download size (47.7 GB)

---

## Dataset 2: BSL Fingerspelling Dataset
**URL**: https://www.kaggle.com/datasets/alifsathar/bsl-fingerspelling-dataset

### Key Information:
- **Size**: Labeled images of BSL alphabet A-Z
- **Content**: Full alphabet (26 letters) ✓
- **Format**: Images only (no pre-extracted landmarks)
- **Conditions**: Varied lighting and orientations
- **Purpose**: Real-time webcam recognition
- **Downloads**: 64
- **Size**: ~192 MB

### Advantages:
✅ Complete alphabet (A-Z)  
✅ Smaller download size  
✅ Designed for real-time webcam use  
✅ Varied conditions (good for generalization)

### Disadvantages:
❌ No pre-extracted MediaPipe landmarks (you'd need to extract them)  
❌ Smaller dataset size  
❌ Newer/less tested

---

## Recommendation: Use Dataset 1

**Dataset 1** is the best choice for your project because:

1. **Time Savings**: Landmarks are already extracted - you can skip straight to model training
2. **Large Dataset**: 34,000 samples provide excellent training data
3. **Pre-trained Models**: Includes SVM models you can use as baseline/comparison
4. **Ready to Use**: CSV format matches what your `preprocess_data.py` expects

### Handling Missing Letters (H, J, Y):

**Option A** (Recommended): Start with 23-letter alphabet
- Build and test your system with available letters
- Add H, J, Y later by collecting your own data using `data_collection.py`

**Option B**: Combine datasets
- Use Dataset 1 for all available letters (A-G, I, K-X, Z)
- Use Dataset 2 for missing letters (H, J, Y)
- Extract MediaPipe landmarks from Dataset 2 images

**Option C**: Use only 21 one-handed letters
- Skip two-handed letters entirely (simpler system)

---

## Next Steps

### Step 1: Set Up Kaggle API

```bash
pip install kaggle
```

Create Kaggle API credentials:
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place it in `C:\Users\ahann\.kaggle\kaggle.json`

### Step 2: Download Dataset

```bash
kaggle datasets download -d erentatepe/bsl-numbers-and-alphabet-hand-position-for-mediapipe
```

### Step 3: Extract and Process

Run the provided script to extract and convert to your project format.

---

## File Structure After Download

```
hands/
├── bsl_data/
│   ├── kaggle_dataset/          # Downloaded dataset
│   │   ├── images/              # BSL images
│   │   ├── train.csv            # Training landmarks
│   │   ├── test.csv             # Test landmarks
│   │   └── models/              # Pre-trained SVM models
│   ├── bsl_landmarks.csv        # Converted to project format
│   └── processed/               # Preprocessed data
```
