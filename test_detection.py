import cv2
import time
import numpy as np
import os
import pickle
import urllib.request
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

"""
Live BSL detection running TWO models simultaneously:
  - Alphabet model  (bsl_model.h5)          → letters A-Z
  - Word sign model (word_model.h5)         → common words (hello, baby, ...)

Run from the hands/ folder:
    python test_detection.py
"""

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "bsl_data"

# Set to False to disable alphabet detection (words only)
# Set to True  to run both alphabet + word models simultaneously
SHOW_LETTERS = False

# Alphabet model
MODEL_FILE   = os.path.join(DATA_DIR, "bsl_model.h5")
ENCODER_FILE = os.path.join(DATA_DIR, "label_encoder.pkl")

# Word model
WORD_MODEL_FILE   = os.path.join(DATA_DIR, "word_model.h5")
WORD_ENCODER_FILE = os.path.join(DATA_DIR, "word_label_encoder.pkl")

# MediaPipe
MP_MODEL_FILE = os.path.join(DATA_DIR, "hand_landmarker.task")
MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

ZERO_HAND              = np.zeros(63)
SMOOTH_FRAMES          = 10       # rolling average window
LETTER_CONF_THRESHOLD  = 0.60    # green when confident
WORD_CONF_THRESHOLD    = 0.55    # word shown when above this

# ── Hand skeleton connections ──────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,17)
]


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 220, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 80, 0), -1)


# ── Normalization (must match train.py exactly) ────────────────────────────────
def normalize_hand(hand_vec):
    if np.all(hand_vec == 0):
        return hand_vec
    pts   = hand_vec.reshape(3, 21).T
    pts   = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts = pts / scale
    return pts.T.flatten()


def normalize_landmarks(row):
    if len(row) == 126:
        return np.concatenate([normalize_hand(row[:63]),
                               normalize_hand(row[63:])])
    return normalize_hand(row)


# ── Download MediaPipe model ───────────────────────────────────────────────────
def download_mp_model():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(MP_MODEL_FILE):
        print("Downloading MediaPipe hand landmark model...")
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
        print(f"✓ Saved to {MP_MODEL_FILE}")


# ── Load models ────────────────────────────────────────────────────────────────
print("=" * 60)
print("BSL Live Detection — Alphabet + Word Signs")
print("=" * 60)

import tensorflow as tf

# -- Alphabet model
letter_model, letter_le = None, None
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    letter_model = tf.keras.models.load_model(MODEL_FILE)
    with open(ENCODER_FILE, 'rb') as f:
        letter_le = pickle.load(f)
    print(f"\n✓ Alphabet model  — {len(letter_le.classes_)} classes: {list(letter_le.classes_)}")
else:
    print("\n⚠  Alphabet model not found. Run: python train.py")

# -- Word model
word_model, word_le = None, None
if os.path.exists(WORD_MODEL_FILE) and os.path.exists(WORD_ENCODER_FILE):
    word_model = tf.keras.models.load_model(WORD_MODEL_FILE)
    with open(WORD_ENCODER_FILE, 'rb') as f:
        word_le = pickle.load(f)
    print(f"✓ Word model      — {len(word_le.classes_)} classes: {list(word_le.classes_)}")
else:
    print("⚠  Word model not found. Run: python train_words.py")

if letter_model is None and word_model is None:
    print("\n✗ No models found. Run train.py and train_words.py first.")
    exit(1)

print()


# ── Predict helpers ────────────────────────────────────────────────────────────
def predict_letter(feature_vec):
    """Returns (letter, confidence) or (None, 0) if model unavailable."""
    if letter_model is None:
        return None, 0.0
    norm  = normalize_landmarks(feature_vec).reshape(1, -1)
    proba = letter_model.predict(norm, verbose=0)[0]
    idx   = np.argmax(proba)
    return letter_le.classes_[idx], float(proba[idx])


def predict_word(feature_vec):
    """Returns (word, confidence) or (None, 0) if model unavailable."""
    if word_model is None:
        return None, 0.0
    norm  = normalize_landmarks(feature_vec).reshape(1, -1)
    proba = word_model.predict(norm, verbose=0)[0]
    idx   = np.argmax(proba)
    return word_le.classes_[idx], float(proba[idx])


# ── MediaPipe ──────────────────────────────────────────────────────────────────
download_mp_model()

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_FILE),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Webcam open — show a hand sign. Press ESC to quit.\n")

letter_buffer = deque(maxlen=SMOOTH_FRAMES)
word_buffer   = deque(maxlen=SMOOTH_FRAMES)


def get_left_right(detection_result):
    """Return (left_vec, right_vec) — always Left→h1, Right→h2."""
    left, right = None, None
    if detection_result.hand_landmarks:
        for hand_lms, hand_class in zip(detection_result.hand_landmarks,
                                        detection_result.handedness):
            side = hand_class[0].category_name
            xs   = [lm.x for lm in hand_lms]
            ys   = [lm.y for lm in hand_lms]
            zs   = [lm.z for lm in hand_lms]
            vec  = np.array(xs + ys + zs)
            if side == 'Left':
                left  = vec
            else:
                right = vec
    return (left  if left  is not None else ZERO_HAND.copy()), \
           (right if right is not None else ZERO_HAND.copy())


def smooth_prediction(buffer):
    """Returns (label, avg_confidence) from rolling buffer."""
    if not buffer:
        return None, 0.0
    counts = Counter(p[0] for p in buffer)
    label  = counts.most_common(1)[0][0]
    conf   = np.mean([p[1] for p in buffer if p[0] == label])
    return label, float(conf)


# ── Main loop ──────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)
    result   = landmarker.detect_for_video(mp_image, timestamp_ms)

    h1, h2    = get_left_right(result)
    num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

    # Draw hand skeleton
    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            draw_hand(frame, hand_lms)

    # ── Inference ──────────────────────────────────────────────────────────────
    if num_hands > 0:
        features = np.concatenate([h1, h2])   # always 126 features

        # Run both models
        raw_letter, raw_lconf = predict_letter(features) if SHOW_LETTERS else (None, 0.0)
        raw_word,   raw_wconf = predict_word(features)

        if raw_letter:
            letter_buffer.append((raw_letter, raw_lconf))
        if raw_word:
            word_buffer.append((raw_word, raw_wconf))

        smooth_letter, smooth_lconf = smooth_prediction(letter_buffer)
        smooth_word,   smooth_wconf = smooth_prediction(word_buffer)

        # ── Draw letter panel (left side) ──────────────────────────────────
        if SHOW_LETTERS and smooth_letter and letter_model is not None:
            lcolor = (0, 255, 0) if smooth_lconf >= LETTER_CONF_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, f"Letter: {smooth_letter}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, lcolor, 3)
            cv2.putText(frame, f"{smooth_lconf*100:.0f}%",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, lcolor, 2)

        # ── Draw word panel (right side) ───────────────────────────────────
        if smooth_word and word_model is not None:
            wcolor = (0, 255, 255) if smooth_wconf >= WORD_CONF_THRESHOLD else (100, 100, 255)
            fw = frame.shape[1]
            cv2.putText(frame, f"Word: {smooth_word}",
                        (fw - 340, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, wcolor, 2)
            cv2.putText(frame, f"{smooth_wconf*100:.0f}%",
                        (fw - 340, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.85, wcolor, 2)

        # ── Status bar ─────────────────────────────────────────────────────
        status = f"{num_hands} hand(s) detected"
        cv2.putText(frame, status, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    else:
        letter_buffer.clear()
        word_buffer.clear()
        cv2.putText(frame, "No hand detected",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ── Bottom hint bar ────────────────────────────────────────────────────────
    h_frame = frame.shape[0]
    if word_model is not None:
        words_text = "Words: " + " | ".join(sorted(word_le.classes_))
        cv2.putText(frame, words_text, (10, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 200), 1)

    cv2.imshow("BSL Detector — Letters + Words  (ESC to quit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("\n✓ Session ended.")
