"""
measure_performance.py
======================
Run this once to get the three numbers needed for the Chapter 5 NFR table:

  1. LSTM inference latency (average ms per forward pass)
  2. BiLSTM validation accuracy on held-out data
  3. Gemini API round-trip time (average seconds per sentence call)

Usage:
    python measure_performance.py

Results are printed clearly at the end.
"""

import time
import os
import pickle
import numpy as np

# Must be set BEFORE importing tensorflow to force legacy Keras 2 behaviour
# so that .h5 models saved by older TF/Keras versions can be loaded correctly.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Config (must match your actual paths) ─────────────────────────────────────
DATA_DIR     = "bsl_data"
SEQ_DIR      = os.path.join(DATA_DIR, "sequences")
MODEL_PATH   = os.path.join(DATA_DIR, "lstm_model.h5")
ENCODER_PATH = os.path.join(DATA_DIR, "lstm_label_encoder.pkl")
SEQUENCE_LEN = 30
FEATURE_DIM  = 126

print("=" * 60)
print("  BSL Performance Measurement Tool")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LSTM MODEL LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("[1/3] Loading LSTM model...")
if not os.path.exists(MODEL_PATH):
    print(f"  ✗ Model not found at {MODEL_PATH}")
    exit(1)

# Use the same load strategy as the main application (compatible with
# models saved by older TF/Keras versions)
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            'Orthogonal': tf.keras.initializers.Orthogonal,
        }
    )
except Exception:
    # Final fallback: load via legacy H5 format
    import keras
    model = keras.saving.load_model(MODEL_PATH, compile=False)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Warm-up pass (forces graph compilation so it doesn't skew timings)
_dummy = np.zeros((1, SEQUENCE_LEN, FEATURE_DIM), dtype=np.float32)
_ = model(_dummy, training=False)
print(f"  ✓ Model loaded — {len(le.classes_)} classes")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MEASURE LSTM INFERENCE LATENCY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Measuring LSTM inference latency (50 calls)...")

latencies = []
for _ in range(50):
    seq = np.random.rand(1, SEQUENCE_LEN, FEATURE_DIM).astype(np.float32)
    t0 = time.perf_counter()
    _ = model(seq, training=False)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)  # convert to ms

avg_latency = np.mean(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)

print(f"  ✓ Average : {avg_latency:.1f} ms")
print(f"  ✓ Min     : {min_latency:.1f} ms")
print(f"  ✓ Max     : {max_latency:.1f} ms")

# ══════════════════════════════════════════════════════════════════════════════
# 3. MEASURE VALIDATION ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/3a] Loading sequences for validation accuracy...")

def normalize_hand(hand_vec):
    if np.all(hand_vec == 0):
        return hand_vec
    pts   = hand_vec.reshape(3, 21).T
    pts   = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts = pts / scale
    return pts.T.flatten()

def normalize_frame(frame_vec):
    return np.concatenate([normalize_hand(frame_vec[:63]),
                           normalize_hand(frame_vec[63:])])

sequences = []
labels = []

if not os.path.isdir(SEQ_DIR):
    print(f"  ✗ Sequences directory not found at {SEQ_DIR}")
else:
    for label in sorted(os.listdir(SEQ_DIR)):
        label_dir = os.path.join(SEQ_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for npy_file in sorted(os.listdir(label_dir)):
            if not npy_file.endswith(".npy"):
                continue
            seq_path = os.path.join(label_dir, npy_file)
            try:
                seq = np.load(seq_path)  # shape: (30, 126)
                normed = np.array([normalize_frame(f) for f in seq])
                sequences.append(normed)
                labels.append(label)
            except Exception as e:
                print(f"  ⚠ Skipping {seq_path}: {e}")

    if sequences:
        X = np.array(sequences, dtype=np.float32)
        y = le.transform(labels)

        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        print(f"  Evaluating on {len(X_val)} validation samples...")
        preds   = np.argmax(model.predict(X_val, verbose=0), axis=1)
        val_acc = np.mean(preds == y_val) * 100

        print(f"  ✓ Validation accuracy: {val_acc:.1f}%")

        # Per-class breakdown
        print("\n  Per-class accuracy:")
        for i, cls in enumerate(le.classes_):
            mask = y_val == i
            if mask.sum() == 0:
                continue
            acc = np.mean(preds[mask] == i) * 100
            bar = "█" * int(acc / 5)
            print(f"    {cls:<15} {acc:5.1f}%  {bar}")
    else:
        print("  ✗ No .npy sequence files found. Run data_collection_lstm.py first.")
        val_acc = None

# ══════════════════════════════════════════════════════════════════════════════
# 4. MEASURE API ROUND-TRIP TIME
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/3b] Measuring API round-trip time (5 calls)...")

try:
    from sentence_builder import SentenceBuilder
    sb = SentenceBuilder()

    # Wait for ready if needed
    wait_start = time.time()
    while not sb.ready and (time.time() - wait_start) < 10:
        time.sleep(0.1)

    if not sb.ready:
        print("  ✗ Sentence builder not ready after 10s")
    else:
        test_inputs = [
            ["hello", "my", "name"],
            ["how", "you"],
            ["good", "morning"],
            ["what", "your", "name"],
            ["hello", "you"],
        ]

        api_times = []
        for words in test_inputs:
            t0 = time.perf_counter()
            result = sb.build(words)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            api_times.append(elapsed)
            print(f"  Input: {words}")
            print(f"  Output: \"{result}\"  ({elapsed:.2f}s)")
            print()
            time.sleep(2)  # avoid rate limiting

        avg_api = np.mean(api_times)
        print(f"  ✓ Average API round-trip: {avg_api:.2f}s")
        print(f"  ✓ Min: {np.min(api_times):.2f}s  Max: {np.max(api_times):.2f}s")

except Exception as e:
    print(f"  ✗ API test failed: {e}")
    avg_api = None

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  RESULTS SUMMARY  (copy these into your Chapter 5 NFR table)")
print("=" * 60)
print(f"  NFR-02  LSTM inference latency : {avg_latency:.1f} ms average")
if 'val_acc' in dir() and val_acc is not None:
    print(f"  NFR-03  Validation accuracy    : {val_acc:.1f}%")
else:
    print(f"  NFR-03  Validation accuracy    : could not compute (no sequence data)")
if 'avg_api' in dir() and avg_api is not None:
    print(f"  NFR-04  API round-trip time    : {avg_api:.2f}s average")
else:
    print(f"  NFR-04  API round-trip time    : could not compute (see error above)")
print("=" * 60)
