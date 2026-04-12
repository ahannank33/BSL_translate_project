import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import pickle

"""
Trains a Deep Neural Network on the BSL landmark CSV produced by
convert_kaggle_dataset.py or data_collection.py.

Saves two files:
  bsl_data/bsl_model.keras      ← the trained neural network
  bsl_data/bsl_model.h5      ← the trained neural network
  bsl_data/label_encoder.pkl    ← maps class indices back to letter names
"""

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "bsl_data"
CSV_FILE   = os.path.join(DATA_DIR, "bsl_landmarks.csv")
MODEL_FILE = os.path.join(DATA_DIR, "bsl_model.h5")
ENCODER_FILE = os.path.join(DATA_DIR, "label_encoder.pkl")

print("=" * 60)
print("BSL Neural Network Training")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────────────
if not os.path.exists(CSV_FILE):
    print(f"\n✗ No CSV found at {CSV_FILE}")
    print("  Run convert_kaggle_dataset.py or data_collection.py first.")
    exit(1)

df = pd.read_csv(CSV_FILE)
print(f"\nLoaded {len(df)} samples")

feature_cols = [col for col in df.columns if col not in ['label', 'timestamp']]
features = df[feature_cols].apply(pd.to_numeric, errors='coerce').values
labels   = df['label'].values

print(f"Features per sample: {len(feature_cols)}")
print(f"Letters: {sorted(set(labels))}")
print(f"Samples per letter:")
for letter, count in sorted(pd.Series(labels).value_counts().items()):
    print(f"  {letter}: {count}")

# ── Normalize landmarks ────────────────────────────────────────────────────────
# Must match exactly what test_detection.py does during live detection

def normalize_hand(hand_vec):
    """Translate to wrist origin + scale by middle-knuckle distance."""
    if np.all(hand_vec == 0):
        return hand_vec
    pts = hand_vec.reshape(3, 21).T   # (21, 3)
    pts = pts - pts[0]                # translate: wrist → origin
    scale = np.linalg.norm(pts[9])   # distance to middle knuckle
    if scale > 0:
        pts = pts / scale
    return pts.T.flatten()


def normalize_row(row):
    if len(row) == 126:
        return np.concatenate([normalize_hand(row[:63]),
                               normalize_hand(row[63:])])
    return normalize_hand(row)


print("\nNormalizing landmarks...")
X = np.array([normalize_row(r) for r in features])

# ── Encode labels ──────────────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(labels)
num_classes = len(le.classes_)
print(f"\nClasses ({num_classes}): {list(le.classes_)}")

# ── Train / validation split ───────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train)} samples")
print(f"Val   : {len(X_val)} samples")

# ── Build model ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Building Neural Network")
print("=" * 60)

input_dim = X.shape[1]   # 126 for two-hand, 63 for one-hand

model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(num_classes, activation='softmax'),
], name="BSL_DNN")

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── Train ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training")
print("=" * 60)

callbacks = [
    # Stop early if validation accuracy stops improving
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True, verbose=1
    ),
    # Reduce learning rate when stuck
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-5, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Results")
print("=" * 60)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation accuracy : {val_acc * 100:.1f}%")
print(f"Validation loss     : {val_loss:.4f}")

# Per-class accuracy
y_pred     = np.argmax(model.predict(X_val, verbose=0), axis=1)
class_names = le.classes_
print(f"\nPer-letter accuracy:")
for i, letter in enumerate(class_names):
    mask    = y_val == i
    if mask.sum() == 0:
        continue
    correct = (y_pred[mask] == i).sum()
    total   = mask.sum()
    bar     = "█" * int(correct / total * 20)
    print(f"  {letter}: {correct}/{total} ({correct/total*100:.0f}%)  {bar}")

# ── Save ───────────────────────────────────────────────────────────────────────
model.save(MODEL_FILE)
with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(le, f)

print(f"\n✓ Model saved   → {MODEL_FILE}")
print(f"✓ Encoder saved → {ENCODER_FILE}")
print("\nNext step: python test_detection.py")
