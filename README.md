# BSL Sign Language Recognition System

A real-time British Sign Language (BSL) alphabet recognition system using MediaPipe hand tracking and machine learning.

## Project Goal

- **Phase 1** (Current): Recognize BSL alphabet (A-Z) from video input
- **Future Phases**: Expand to words, grammar, and full sentence translation

## Requirements

```bash
pip install -r requirements.txt
```

## Workflow

### Option A: Use Kaggle Dataset (RECOMMENDED) ⭐

Use pre-collected dataset with 34,000 samples - saves 30+ hours of manual collection!

#### Step 1: Set Up Kaggle API

1. Install dependencies: `pip install -r requirements.txt`
2. Get Kaggle API credentials:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New API Token"
   - Save `kaggle.json` to `C:\Users\ahann\.kaggle\`

#### Step 2: Download Dataset

```bash
python download_dataset.py
```

This downloads the BSL MediaPipe dataset (~47.7 GB) with pre-extracted hand landmarks.

**Dataset Info:**
- 34,000 images with MediaPipe landmarks
- BSL alphabet (23 letters - excludes H, J, Y)
- BSL numbers (0-10)
- See `DATASET_GUIDE.md` for full details

#### Step 3: Convert to Project Format

```bash
python convert_kaggle_dataset.py
```

Converts Kaggle CSV format to project format (`bsl_landmarks.csv`).

#### Step 4: Preprocess Data

```bash
python preprocess_data.py
```

Normalizes, augments, and splits data (70% train, 15% val, 15% test).

#### Step 5: Train Model

*(Coming next)*

```bash
python train.py
```

#### Step 6: Real-time Recognition

*(Coming next)*

```bash
python sign_language_detector.py
```

---

### Option B: Manual Data Collection

Collect your own training data (useful for missing letters H, J, Y or customization).

#### Step 1: Collect Data

```bash
python data_collection.py
```

**Instructions:**
- Position your hand to form a BSL letter
- Press **SPACE** to capture
- Press **N** to move to next letter
- Collect **500+ samples per letter**
- See `BSL_REFERENCE.md` for hand shapes

#### Step 2-6: Same as Option A (Steps 4-6)

---

## Project Structure

```
hands/
├── project.py                  # Original hand detection demo
├── download_dataset.py         # Download Kaggle dataset
├── convert_kaggle_dataset.py   # Convert dataset format
├── data_collection.py          # Manually collect training data
├── preprocess_data.py          # Preprocess and augment data
├── train.py                    # (Next) Train classification model
├── sign_language_detector.py   # (Next) Real-time BSL recognition
├── DATASET_GUIDE.md            # Dataset comparison and guide
├── BSL_REFERENCE.md            # BSL alphabet reference
├── bsl_data/                   # Data directory
│   ├── kaggle_dataset/         # Downloaded Kaggle data
│   ├── bsl_landmarks.csv       # Landmark data
│   └── processed/              # Preprocessed train/val/test data
└── models/                     # Trained models (coming soon)
```

## Current Status

- [x] Hand detection with MediaPipe
- [x] Kaggle dataset integration
- [x] Data collection tool (for manual collection)
- [x] Data preprocessing pipeline
- [ ] Model training
- [ ] Real-time recognition
- [ ] Letter-to-word formation

## Notes

- Using Kaggle dataset provides 34,000 pre-labeled samples
- Dataset excludes H, J, Y (dynamic gestures) - can add later
- Target model accuracy: >85% on test set
