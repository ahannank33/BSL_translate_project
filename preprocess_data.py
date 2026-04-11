import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Configuration
DATA_DIR = "bsl_data"
INPUT_CSV = os.path.join(DATA_DIR, "bsl_landmarks.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("=== BSL Data Preprocessing ===\n")

# Load data
print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

print(f"Total samples: {len(df)}")
print(f"\nSamples per letter:")
print(df['label'].value_counts().sort_index())

# Extract features and labels
feature_df = df.drop(['label', 'timestamp'], axis=1)

# Convert all feature columns to numeric (fixes string data issues)
feature_df = feature_df.apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN values (bad data)
valid_mask = ~feature_df.isna().any(axis=1)
if not valid_mask.all():
    bad_count = (~valid_mask).sum()
    print(f"\n⚠ Dropping {bad_count} rows with non-numeric data")
    feature_df = feature_df[valid_mask]
    df = df[valid_mask]

labels = df['label'].values
features = feature_df.values

print(f"\nFeature shape: {features.shape}")

# Normalize landmarks
def normalize_landmarks(landmarks):
    """
    Normalize landmarks to be relative to wrist (landmark 0)
    This makes the model invariant to hand position
    """
    landmarks = landmarks.reshape(-1, 3)  # 21 landmarks x 3 coords
    wrist = landmarks[0].copy()
    
    # Subtract wrist position (translation invariance)
    normalized = landmarks - wrist
    
    # Calculate hand size (distance from wrist to middle finger MCP)
    hand_size = np.linalg.norm(normalized[9])  # Middle finger MCP
    
    # Scale by hand size (scale invariance)
    if hand_size > 0:
        normalized = normalized / hand_size
    
    return normalized.flatten()

print("\nNormalizing landmarks...")
normalized_features = np.array([normalize_landmarks(f) for f in features])

# Data Augmentation
def augment_with_noise(features, labels, noise_factor=0.02, augment_count=2):
    """
    Augment data by adding slight random noise
    This helps model generalize better
    """
    augmented_features = [features]
    augmented_labels = [labels]
    
    for i in range(augment_count):
        noise = np.random.normal(0, noise_factor, features.shape)
        augmented_features.append(features + noise)
        augmented_labels.append(labels)
    
    return np.vstack(augmented_features), np.concatenate(augmented_labels)

print("Augmenting data...")
augmented_features, augmented_labels = augment_with_noise(normalized_features, labels)

print(f"After augmentation: {augmented_features.shape[0]} samples")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(augmented_labels)

print(f"\nLabel mapping:")
for i, letter in enumerate(label_encoder.classes_):
    print(f"{letter} -> {i}")

# Split data: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    augmented_features, encoded_labels, 
    test_size=0.15, 
    random_state=42, 
    stratify=encoded_labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=0.176,  # 0.176 * 0.85 ≈ 0.15 of total
    random_state=42, 
    stratify=y_temp
)

print(f"\nDataset split:")
print(f"Training: {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

# Save processed data
print(f"\nSaving processed data to {OUTPUT_DIR}...")

np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)

# Save label encoder
import pickle
with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n=== Preprocessing Complete ===")
print(f"Files saved:")
print(f"  - X_train.npy, y_train.npy")
print(f"  - X_val.npy, y_val.npy")
print(f"  - X_test.npy, y_test.npy")
print(f"  - label_encoder.pkl")

# Display statistics
print(f"\nStatistics:")
print(f"Feature dimensions: {X_train.shape[1]}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Class distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
for label_idx, count in zip(unique, counts):
    letter = label_encoder.inverse_transform([label_idx])[0]
    print(f"  {letter}: {count} samples")
