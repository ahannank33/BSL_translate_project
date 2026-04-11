"""
Trains a DNN classifier on BSL word signs using landmarks extracted
from the Slingo dataset (word_landmarks.csv).

Produces:
  bsl_data/word_model.keras          ← trained model
  bsl_data/word_label_encoder.pkl    ← label encoder (index → word)

Run after convert_slingo_dataset.py:
    python train_words.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, callbacks

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR     = "bsl_data"
CSV_PATH     = os.path.join(DATA_DIR, "word_landmarks.csv")
MODEL_PATH   = os.path.join(DATA_DIR, "word_model.keras")
ENCODER_PATH = os.path.join(DATA_DIR, "word_label_encoder.pkl")

EPOCHS       = 100
BATCH_SIZE   = 64
LEARNING_RATE = 1e-3

HAND1_COLS = [f'h1_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
HAND2_COLS = [f'h2_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
FEATURE_COLS = HAND1_COLS + HAND2_COLS      # 126 features


def load_data(csv_path):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Total rows  : {len(df)}")
    print(f"  Word classes: {sorted(df['label'].unique())}")
    print(f"  Class counts:\n{df['label'].value_counts().to_string()}\n")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['label'].values
    return X, y


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax'),
    ], name='word_dnn')
    return model


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Load & encode ──────────────────────────────────────────────────────────
    X, y_raw = load_data(CSV_PATH)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    print(f"Feature dim : {X.shape[1]}")
    print(f"Num classes : {num_classes}")
    print(f"Samples     : {len(X)}\n")

    # ── Train / val split ──────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # ── Class weights (handles imbalanced categories) ─────────────────────────
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # ── Build & compile ────────────────────────────────────────────────────────
    model = build_model(X.shape[1], num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # ── Callbacks ──────────────────────────────────────────────────────────────
    cbs = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7,
            min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training word sign model...")
    print("=" * 60 + "\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=cbs,
        verbose=1,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'='*60}")
    print(f"✓ Best val accuracy : {val_acc*100:.1f}%")
    print(f"  Val loss          : {val_loss:.4f}")
    print(f"  Model saved  → {MODEL_PATH}")

    # ── Save label encoder ─────────────────────────────────────────────────────
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"  Encoder saved → {ENCODER_PATH}")
    print(f"\nWord classes learned: {list(le.classes_)}")
    print("=" * 60)
    print("\nNext step: python test_detection.py  (word model loaded automatically)")


if __name__ == '__main__':
    main()
