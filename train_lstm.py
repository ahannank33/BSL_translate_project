"""
Trains a bidirectional LSTM classifier on BSL word sign sequences.

Input data: bsl_data/lstm_sequences/<word>/*.npy
Each .npy file is shape (30, 126) — 30 frames × 126 landmark features.

Produces:
  bsl_data/lstm_model.keras          ← trained LSTM model
  bsl_data/lstm_label_encoder.pkl    ← label encoder (index → word)

Run after data_collection_lstm.py:
    python train_lstm.py
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, callbacks

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR     = "bsl_data"
SEQ_DIR      = os.path.join(DATA_DIR, "lstm_sequences")
MODEL_PATH   = os.path.join(DATA_DIR, "lstm_model.h5")
ENCODER_PATH = os.path.join(DATA_DIR, "lstm_label_encoder.pkl")

SEQUENCE_LEN  = 30    # frames per sequence (must match data_collection_lstm.py)
FEATURE_DIM   = 126   # 2 hands × 21 landmarks × 3 (x,y,z)

EPOCHS        = 150
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3


# ── Load sequences ─────────────────────────────────────────────────────────────
def load_sequences(seq_dir):
    X, y = [], []
    words = sorted(os.listdir(seq_dir))
    print(f"\nLoading sequences from: {seq_dir}")
    for word in words:
        word_dir = os.path.join(seq_dir, word)
        if not os.path.isdir(word_dir):
            continue
        files = [f for f in os.listdir(word_dir) if f.endswith(".npy")]
        print(f"  {word:<12} — {len(files)} sequences")
        for fname in files:
            seq = np.load(os.path.join(word_dir, fname))
            if seq.shape == (SEQUENCE_LEN, FEATURE_DIM):
                X.append(seq)
                y.append(word)
            else:
                print(f"    ⚠  Skipping {fname} — unexpected shape {seq.shape}")
    return np.array(X, dtype=np.float32), np.array(y)



def normalize_hand(hand_vec):
    """translate to wrist origin, scale by palm size."""
    if np.all(hand_vec == 0):
        return hand_vec
    pts   = hand_vec.reshape(3, 21).T     # (21, 3)
    pts   = pts - pts[0]                  # wrist at origin
    scale = np.linalg.norm(pts[9])        # middle finger MCP as scale
    if scale > 0:
        pts = pts / scale
    return pts.T.flatten()


def normalize_sequence(seq):
    """ normalization of every frame in sequence. shape -> (T, 126)."""
    out = np.zeros_like(seq)
    for t, frame in enumerate(seq):
        h1 = normalize_hand(frame[:63])
        h2 = normalize_hand(frame[63:])
        out[t] = np.concatenate([h1, h2])
    return out


# ── Build LSTM model ───────────────────────────────────────────────────────────
def build_model(seq_len: int, feature_dim: int, num_classes: int) -> tf.keras.Model:
    
    """    biLSTM with attention-style global pooling."""
    
    inp = layers.Input(shape=(seq_len, feature_dim), name="landmark_sequence")

    # block 1 - first Bi-LSTM returns sequences
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1"
    )(inp)
    x = layers.LayerNormalization()(x)

    # block 2 - second Bi-LSTM, return sequences for global pooling
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_2"
    )(x)
    x = layers.LayerNormalization()(x)

    # global average pooling
    x = layers.GlobalAveragePooling1D(name="temporal_pool")(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(num_classes, activation='softmax', name="word_proba")(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="bsl_lstm")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load & normalize
    X_raw, y_raw = load_sequences(SEQ_DIR)
    if len(X_raw) == 0:
        print("\n✗ No sequences found. Run data_collection_lstm.py first.")
        return

    print(f"\nTotal sequences : {len(X_raw)}")
    print(f"Sequence shape  : {X_raw.shape}")

    X = np.array([normalize_sequence(seq) for seq in X_raw])

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}\n")

    # Train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Build & compile
    model = build_model(SEQUENCE_LEN, FEATURE_DIM, num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Callbacks
    cbs = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8,
            min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    # Train
    print("\n" + "=" * 60)
    print("Training LSTM sequence model...")
    print("=" * 60 + "\n")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=cbs,
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'='*60}")
    print(f"✓ Best val accuracy : {val_acc*100:.1f}%")
    print(f"  Val loss          : {val_loss:.4f}")
    print(f"  Model saved  → {MODEL_PATH}")

    # Save label encoder
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"  Encoder saved → {ENCODER_PATH}")
    print(f"\nWord classes learned: {list(le.classes_)}")
    print("=" * 60)
    print("\nNext step: python test_detection_lstm.py")


if __name__ == '__main__':
    main()
