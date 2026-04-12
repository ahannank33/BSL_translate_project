import cv2
import numpy as np
import os
import pickle
import urllib.request
import threading
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tensorflow as tf

"""
Live BSL detection using the LSTM sequence model.

Optimisations over v1:
  1. Inference runs in a background thread — camera read loop never blocks.
  2. Motion-change detection — if the hand moves significantly between frames,
     the smoothing buffer is flushed immediately so the new gesture registers
     without waiting for old predictions to decay.
  3. Recency-weighted smoothing — recent predictions count more than old ones.
  4. model(seq) called directly (no Python overhead of .predict()).

Run from the hands/ folder:
    python test_detection_lstm.py

Controls:
  ESC  — quit
"""

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR       = "bsl_data"
MODEL_PATH     = os.path.join(DATA_DIR, "lstm_model.h5")
ENCODER_PATH   = os.path.join(DATA_DIR, "lstm_label_encoder.pkl")
MP_MODEL_FILE  = os.path.join(DATA_DIR, "hand_landmarker.task")
MP_MODEL_URL   = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

SEQUENCE_LEN      = 30     # must match training
FEATURE_DIM       = 126
CONF_THRESHOLD    = 0.60   # min confidence to lock a prediction

# How many frames to keep showing the last word after the hand leaves frame.
# ~90 frames ≈ 3 seconds at 30 fps. Set to 0 to disable.
HOLD_FRAMES       = 90

# Number of recent raw predictions kept for smoothing.
PRED_BUFFER_SIZE  = 3

ZERO_FRAME = np.zeros(FEATURE_DIM, dtype=np.float32)

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
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 220, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 80, 0), -1)


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
    h1 = normalize_hand(frame_vec[:63])
    h2 = normalize_hand(frame_vec[63:])
    return np.concatenate([h1, h2])


def get_left_right(detection_result):
    left = right = None
    if detection_result.hand_landmarks:
        for hand_lms, hand_class in zip(detection_result.hand_landmarks,
                                        detection_result.handedness):
            side = hand_class[0].category_name
            xs   = [lm.x for lm in hand_lms]
            ys   = [lm.y for lm in hand_lms]
            zs   = [lm.z for lm in hand_lms]
            vec  = np.array(xs + ys + zs, dtype=np.float32)
            if side == 'Left':
                left  = vec
            else:
                right = vec
    zeros = np.zeros(63, dtype=np.float32)
    return (left  if left  is not None else zeros.copy()), \
           (right if right is not None else zeros.copy())


# ── Download MediaPipe model ───────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(MP_MODEL_FILE):
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
    print(f"✓ Saved to {MP_MODEL_FILE}")

# ── Load LSTM model ────────────────────────────────────────────────────────────
print("=" * 60)
print("  BSL LSTM Live Detection  (Optimised)")
print("=" * 60)

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("\n✗ LSTM model not found.")
    print("  1. Collect sequences: python data_collection_lstm.py")
    print("  2. Train model:       python train_lstm.py")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
# Warm up the model so the first real inference isn't slow
_dummy = np.zeros((1, SEQUENCE_LEN, FEATURE_DIM), dtype=np.float32)
_ = model(_dummy, training=False)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

print(f"\n✓ LSTM model loaded — {len(le.classes_)} classes")
print(f"  Words: {', '.join(le.classes_)}\n")

# ── MediaPipe ──────────────────────────────────────────────────────────────────
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_FILE),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Webcam open — perform a BSL sign. Press ESC to quit.\n")

# ── Shared state between main thread and inference thread ─────────────────────
frame_buffer    = deque(maxlen=SEQUENCE_LEN)
pred_buffer     = deque(maxlen=PRED_BUFFER_SIZE)
lock            = threading.Lock()

# Output from inference thread
infer_result    = {"word": None, "conf": 0.0}
infer_pending   = threading.Event()   # signals the inference thread to run

# ── Locked-word state ───────────────────────────────────────────────
# Once a gesture crosses CONF_THRESHOLD it is "locked" and stays on screen
# until a DIFFERENT gesture is detected or the hold timer expires.
locked_word      = None   # word currently locked on screen
locked_conf      = 0.0
hold_frames_left = 0      # countdown frames while hand is absent

# Pre-fill frame buffer
for _ in range(SEQUENCE_LEN):
    frame_buffer.append(ZERO_FRAME.copy())


# ── Inference thread ─────────────────────────────────────────────────────────
def inference_worker():
    """Runs LSTM inference in a background thread to keep the camera loop smooth."""
    while True:
        infer_pending.wait()          # block until main thread signals
        infer_pending.clear()

        with lock:
            seq = np.array(frame_buffer, dtype=np.float32)[np.newaxis]  # (1,30,126)

        proba = model(seq, training=False).numpy()[0]
        idx   = int(np.argmax(proba))
        infer_result["word"] = le.classes_[idx]
        infer_result["conf"] = float(proba[idx])


infer_thread = threading.Thread(target=inference_worker, daemon=True)
infer_thread.start()

# ── Smoothing helpers ─────────────────────────────────────────────────────────
def recency_weighted_prediction(buf):
    """
    Weight recent predictions more heavily than older ones.
    Returns (word, weighted_confidence).
    """
    if not buf:
        return None, 0.0

    items    = list(buf)
    n        = len(items)
    # Weights: oldest = 1, newest = n  (linear ramp)
    weights  = np.arange(1, n + 1, dtype=np.float32)

    word_scores = {}
    for w, (word, conf) in zip(weights, items):
        word_scores.setdefault(word, 0.0)
        word_scores[word] += w * conf

    best_word = max(word_scores, key=word_scores.__getitem__)
    # Normalised average confidence for the winning word
    win_weights = [weights[i] for i, (word, _) in enumerate(items) if word == best_word]
    win_confs   = [conf       for (word, conf)   in items           if word == best_word]
    avg_conf    = float(np.dot(win_weights, win_confs) / sum(win_weights))

    return best_word, avg_conf


# ── FPS tracking ─────────────────────────────────────────────────────────────
import time
fps_buf      = deque(maxlen=30)
last_t       = time.perf_counter()
infer_frame  = 0   # frame counter for skipping inference every Nth frame
INFER_EVERY  = 2   # run inference every 2nd frame (halves GPU load)

# ── Main loop ─────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    now = time.perf_counter()
    fps_buf.append(1.0 / max(now - last_t, 1e-6))
    last_t = now
    fps = np.mean(fps_buf)

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect(mp_image)

    h1, h2    = get_left_right(result)
    num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            draw_hand(frame, hand_lms)

    # ── Update landmark buffer ─────────────────────────────────────────────────
    raw_frame  = np.concatenate([h1, h2])
    norm_frame = normalize_frame(raw_frame)

    with lock:
        frame_buffer.append(norm_frame)

    # ── Trigger inference (non-blocking, every INFER_EVERY frames) ───────────
    infer_frame += 1
    if num_hands > 0 and infer_frame % INFER_EVERY == 0:
        infer_pending.set()   # wake inference thread — returns immediately

    # ── Consume last inference result ─────────────────────────────────────
    if infer_result["word"] is not None and num_hands > 0:
        pred_buffer.append((infer_result["word"], infer_result["conf"]))
        infer_result["word"] = None

    smooth_word, smooth_conf = recency_weighted_prediction(pred_buffer)

    # ── Lock-in logic ─────────────────────────────────────────────────────
    if num_hands > 0:
        hold_frames_left = 0   # hand is present — cancel any hold countdown

        if smooth_word and smooth_conf >= CONF_THRESHOLD:
            if smooth_word != locked_word:
                # New gesture detected — switch immediately
                pred_buffer.clear()   # flush old votes so next gesture starts fresh
            locked_word = smooth_word
            locked_conf = smooth_conf

    else:
        # Hand has left frame
        pred_buffer.clear()
        if locked_word is not None and hold_frames_left == 0:
            # Start hold countdown
            hold_frames_left = HOLD_FRAMES
        elif hold_frames_left > 0:
            hold_frames_left -= 1
            if hold_frames_left == 0:
                locked_word = None
                locked_conf = 0.0

    # ── Draw UI ────────────────────────────────────────────────────────────────
    fw, fh = frame.shape[1], frame.shape[0]

    # FPS counter (top right)
    cv2.putText(frame, f"FPS: {fps:.0f}", (fw - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 50), 2)

    # Model label
    cv2.putText(frame, "LSTM (Optimised)", (fw - 230, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 50), 1)

    # ── Prediction display ────────────────────────────────────────────────────
    if locked_word and num_hands > 0:
        # Gesture actively locked — hand present
        wcolor = (0, 255, 180)
        cv2.putText(frame, locked_word.upper(),
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, wcolor, 4)
        cv2.putText(frame, f"{locked_conf*100:.0f}% confident",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, wcolor, 2)

    elif locked_word and hold_frames_left > 0:
        # Hand gone — fade out locked word over hold period
        fade   = hold_frames_left / HOLD_FRAMES          # 1.0 → 0.0
        bright = int(140 + 115 * fade)
        wcolor = (0, bright, int(bright * 0.7))
        cv2.putText(frame, locked_word.upper(),
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, wcolor, 4)
        bar_total  = fw - 20
        bar_filled = int(fade * bar_total)
        cv2.rectangle(frame, (10, 125), (10 + bar_filled, 135), wcolor, -1)
        cv2.rectangle(frame, (10, 125), (10 + bar_total, 135), (80, 80, 80), 1)
        cv2.putText(frame, "Holding...",
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, wcolor, 1)

    elif num_hands == 0 and locked_word is None:
        cv2.putText(frame, "Show a sign...",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 100, 100), 2)

    # Hand count
    if num_hands > 0:
        h_color = (0, 255, 0) if num_hands == 2 else (0, 200, 255)
        cv2.putText(frame, f"{num_hands} hand(s)",
                    (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_color, 1)

    # Word list at bottom
    words_text = "Signs: " + " | ".join(sorted(le.classes_))
    cv2.putText(frame, words_text, (10, fh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 200, 200), 1)

    cv2.imshow("BSL LSTM Detector  (ESC to quit)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ── Cleanup ────────────────────────────────────────────────────────────────────
cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("\n✓ Session ended.")
