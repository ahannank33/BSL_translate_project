import cv2
import numpy as np
import os
import pickle
import threading
import urllib.request
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from sentence_builder import SentenceBuilder



# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "bsl_data"

WORD_MODEL_FILE   = os.path.join(DATA_DIR, "word_model.keras")
WORD_ENCODER_FILE = os.path.join(DATA_DIR, "word_label_encoder.pkl")
MODEL_FILE        = os.path.join(DATA_DIR, "bsl_model.keras")
ENCODER_FILE      = os.path.join(DATA_DIR, "label_encoder.pkl")
MP_MODEL_FILE     = os.path.join(DATA_DIR, "hand_landmarker.task")
MP_MODEL_URL      = ("https://storage.googleapis.com/mediapipe-models/"
                     "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# Set to True to show letter model alongside word model
SHOW_LETTERS = False

# ── Word accumulator tuning ────────────────────────────────────────────────────
STABILITY_FRAMES  = 20   # frames same word must be predicted to lock it
LOCK_CONFIDENCE   = 0.55  # minimum confidence to count toward stability
COOLDOWN_FRAMES   = 40   # frames of cooldown after locking a word
NO_HAND_TIMEOUT   = 75   # frames without hands → sentence complete (~2.5s @ 30fps)
MAX_BUFFER_WORDS  = 7    # maximum words before auto-completing sentence

ZERO_HAND          = np.zeros(63)
SMOOTH_FRAMES      = 8    # per-frame prediction smoothing window

# ── Hand skeleton ──────────────────────────────────────────────────────────────
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


# ── Landmark normalization (must match train.py exactly) ──────────────────────
def normalize_hand(v):
    if np.all(v == 0):
        return v
    pts = v.reshape(3, 21).T
    pts = pts - pts[0]
    s   = np.linalg.norm(pts[9])
    if s > 0:
        pts = pts / s
    return pts.T.flatten()


def normalize_landmarks(row):
    if len(row) == 126:
        return np.concatenate([normalize_hand(row[:63]), normalize_hand(row[63:])])
    return normalize_hand(row)


# ── Download MediaPipe model ───────────────────────────────────────────────────
def download_mp_model():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(MP_MODEL_FILE):
        print("Downloading MediaPipe hand landmark model...")
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
        print(f"✓ Saved to {MP_MODEL_FILE}")


# ── Load ML models ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  BSL Translator — Word Signs + Sentence Formation")
print("=" * 60)

import tensorflow as tf

word_model, word_le = None, None
if os.path.exists(WORD_MODEL_FILE) and os.path.exists(WORD_ENCODER_FILE):
    word_model = tf.keras.models.load_model(WORD_MODEL_FILE)
    with open(WORD_ENCODER_FILE, 'rb') as f:
        word_le = pickle.load(f)
    print(f"\n✓ Word model — {len(word_le.classes_)} words: {list(word_le.classes_)}")
else:
    print("\n✗ Word model not found. Run: python train_words.py")
    exit(1)

letter_model, letter_le = None, None
if SHOW_LETTERS and os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    letter_model = tf.keras.models.load_model(MODEL_FILE)
    with open(ENCODER_FILE, 'rb') as f:
        letter_le = pickle.load(f)
    print(f"✓ Alphabet model — {len(letter_le.classes_)} letters")

# ── Start sentence builder (loads GPT-2 in background) ────────────────────────
print("\n  Starting GPT-2 sentence builder in background...")
sentence_builder = SentenceBuilder()

# ── MediaPipe ──────────────────────────────────────────────────────────────────
download_mp_model()
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
print("\nWebcam open. Press ESC to quit, C to clear, Z to undo.\n")

# ── Per-frame prediction smoothing ────────────────────────────────────────────
word_smooth_buf = deque(maxlen=SMOOTH_FRAMES)


def get_left_right(det):
    left, right = None, None
    if det.hand_landmarks:
        for lms, cls in zip(det.hand_landmarks, det.handedness):
            side = cls[0].category_name
            v    = np.array([lm.x for lm in lms] +
                            [lm.y for lm in lms] +
                            [lm.z for lm in lms])
            if side == 'Left':
                left  = v
            else:
                right = v
    return (left  if left  is not None else ZERO_HAND.copy()), \
           (right if right is not None else ZERO_HAND.copy())


def predict_word_raw(feat):
    """Returns (word, confidence) from word model."""
    norm  = normalize_landmarks(feat).reshape(1, -1)
    proba = word_model.predict(norm, verbose=0)[0]
    idx   = np.argmax(proba)
    return word_le.classes_[idx], float(proba[idx])


def smooth_word():
    """Returns (word, avg_conf) from rolling buffer."""
    if not word_smooth_buf:
        return None, 0.0
    counts = Counter(p[0] for p in word_smooth_buf)
    word   = counts.most_common(1)[0][0]
    conf   = np.mean([p[1] for p in word_smooth_buf if p[0] == word])
    return word, float(conf)


# ── Word accumulator state ─────────────────────────────────────────────────────
word_buffer       = []     # locked words this sentence
pending_word      = None   # word currently accumulating stability
stability_counter = 0      # frames current pending_word has been predicted
cooldown_counter  = 0      # frames remaining in post-lock cooldown
no_hand_counter   = 0      # frames with no hand detected

# ── Sentence state ─────────────────────────────────────────────────────────────
last_sentence     = ""     # last completed + formed sentence
sentence_building = False  # True while GPT-2 thread is running
_sentence_result  = [None] # shared list for background thread output
lock_flash        = 0      # frames to show lock flash


def _run_sentence_builder(words):
    global last_sentence, sentence_building
    result = sentence_builder.build(list(words))
    _sentence_result[0] = result
    last_sentence        = result
    sentence_building    = False
    print(f"\n  ✓ Sentence: \"{result}\"")


def complete_sentence():
    """Called when sentence input is finished."""
    global sentence_building, word_buffer, last_sentence
    global pending_word, stability_counter, cooldown_counter

    if not word_buffer:
        return

    print(f"  Signing complete. Words: {word_buffer}")
    words_copy = list(word_buffer)

    if sentence_builder.ready:
        sentence_building = True
        t = threading.Thread(
            target=_run_sentence_builder, args=(words_copy,), daemon=True
        )
        t.start()
    else:
        last_sentence = ' '.join(words_copy)

    word_buffer.clear()
    pending_word      = None
    stability_counter = 0
    cooldown_counter  = 0


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.55, radius=10):
    """Semi-transparent filled rounded rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text_shadow(img, text, pos, scale, color, thickness=2):
    """Text with drop shadow for readability on any background."""
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ── Main loop ──────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W  = frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect(mp_image)

    h1, h2    = get_left_right(result)
    num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

    # Draw hand skeleton
    if result.hand_landmarks:
        for lms in result.hand_landmarks:
            draw_hand(frame, lms)

    # ── Lock flash overlay ─────────────────────────────────────────────────────
    if lock_flash > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 255, 120), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        lock_flash -= 1

    # ── Inference + accumulator ────────────────────────────────────────────────
    current_word, current_conf = None, 0.0

    if num_hands > 0:
        no_hand_counter = 0
        feat = np.concatenate([h1, h2])

        raw_word, raw_conf = predict_word_raw(feat)
        word_smooth_buf.append((raw_word, raw_conf))
        current_word, current_conf = smooth_word()

        if cooldown_counter > 0:
            cooldown_counter -= 1
        else:
            if current_conf >= LOCK_CONFIDENCE:
                if current_word == pending_word:
                    stability_counter += 1
                else:
                    pending_word      = current_word
                    stability_counter = 1

                # ── LOCK WORD ──────────────────────────────────────────────
                if stability_counter >= STABILITY_FRAMES:
                    word_buffer.append(pending_word)
                    print(f"  + Locked: {pending_word}  [{' → '.join(word_buffer)}]")
                    lock_flash        = 5
                    cooldown_counter  = COOLDOWN_FRAMES
                    stability_counter = 0

                    if len(word_buffer) >= MAX_BUFFER_WORDS:
                        complete_sentence()
            else:
                stability_counter = max(0, stability_counter - 1)
                if stability_counter == 0:
                    pending_word = None

    else:
        # No hands
        word_smooth_buf.clear()
        stability_counter = max(0, stability_counter - 1)
        if stability_counter == 0:
            pending_word = None

        no_hand_counter += 1
        if no_hand_counter >= NO_HAND_TIMEOUT and word_buffer:
            complete_sentence()

    # ══════════════════════════════════════════════════════════════════════════
    # DRAWING
    # ══════════════════════════════════════════════════════════════════════════

    # ── Top-right: current word + confidence ──────────────────────────────────
    if current_word and num_hands > 0:
        conf_color = (0, 255, 200) if current_conf >= LOCK_CONFIDENCE else (80, 80, 255)
        draw_rounded_rect(frame, W - 300, 5, W - 5, 100, (20, 20, 20), alpha=0.6)
        put_text_shadow(frame, current_word.upper(),
                        (W - 290, 45), 1.0, conf_color, 2)
        put_text_shadow(frame, f"{current_conf*100:.0f}% confidence",
                        (W - 290, 75), 0.55, conf_color, 1)

    # ── Stability bar ──────────────────────────────────────────────────────────
    if pending_word and num_hands > 0 and cooldown_counter == 0:
        bar_x1, bar_y1 = 10, 110
        bar_x2, bar_y2 = W - 10, 135
        progress = min(stability_counter / STABILITY_FRAMES, 1.0)
        filled_w = int(progress * (bar_x2 - bar_x1))

        draw_rounded_rect(frame, bar_x1 - 2, bar_y1 - 2,
                          bar_x2 + 2, bar_y2 + 2, (30, 30, 30), alpha=0.7)
        if filled_w > 0:
            cv2.rectangle(frame, (bar_x1, bar_y1),
                          (bar_x1 + filled_w, bar_y2), (0, 220, 120), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1),
                      (bar_x2, bar_y2), (150, 150, 150), 1)

        remaining_s = (STABILITY_FRAMES - stability_counter) / 30
        bar_label   = (f"Locking: {pending_word.upper()} "
                       f"({remaining_s:.1f}s)" if progress < 1.0 else "Locked!")
        put_text_shadow(frame, bar_label, (bar_x1 + 4, bar_y1 - 6),
                        0.5, (200, 255, 200), 1)

    elif cooldown_counter > 0 and num_hands > 0:
        put_text_shadow(frame, "Cooldown — move to next sign",
                        (10, 118), 0.5, (180, 180, 100), 1)

    # ── Word buffer panel ──────────────────────────────────────────────────────
    buf_y = H - 130
    draw_rounded_rect(frame, 5, buf_y, W - 5, buf_y + 45, (10, 20, 40), alpha=0.75)

    if word_buffer:
        buf_text = "  ➜  ".join(w.upper() for w in word_buffer)
        put_text_shadow(frame, buf_text, (15, buf_y + 30), 0.75, (0, 200, 255), 2)
    else:
        put_text_shadow(frame, "(sign a word to start...)",
                        (15, buf_y + 30), 0.65, (120, 120, 120), 1)

    # ── Sentence panel ─────────────────────────────────────────────────────────
    sent_y = H - 78
    draw_rounded_rect(frame, 5, sent_y, W - 5, sent_y + 48, (10, 10, 30), alpha=0.80)

    if sentence_building:
        dots = "." * ((cv2.getTickCount() // int(cv2.getTickFrequency() * 0.5)) % 4)
        put_text_shadow(frame, f"Building sentence{dots}",
                        (15, sent_y + 32), 0.75, (255, 200, 50), 2)
    elif last_sentence:
        put_text_shadow(frame, f'"{last_sentence}"',
                        (15, sent_y + 32), 0.85, (255, 255, 160), 2)
    else:
        put_text_shadow(frame, "(sentence will appear here)",
                        (15, sent_y + 32), 0.65, (90, 90, 90), 1)

    # ── GPT-2 status indicator ─────────────────────────────────────────────────
    if sentence_builder.loading:
        put_text_shadow(frame, "GPT-2 loading...", (10, 25),
                        0.5, (255, 200, 50), 1)
    elif sentence_builder.ready and not sentence_builder.loading:
        cv2.circle(frame, (14, 19), 6, (0, 200, 100), -1)   # green dot = ready

    # ── No hand warning ────────────────────────────────────────────────────────
    if num_hands == 0:
        put_text_shadow(frame, "No hand detected",
                        (10, 60), 1.0, (0, 60, 255), 2)
        if word_buffer:
            remaining = max(0, NO_HAND_TIMEOUT - no_hand_counter)
            put_text_shadow(frame, f"Completing in {remaining/30:.1f}s...",
                            (10, 95), 0.6, (180, 120, 0), 1)

    # ── Bottom hints ───────────────────────────────────────────────────────────
    put_text_shadow(frame, "ESC: quit   C: clear   Z: undo last word",
                    (10, H - 8), 0.42, (160, 160, 160), 1)

    cv2.imshow("BSL Translator", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:      # ESC — quit
        break

    elif key in (ord('c'), ord('C')):   # Clear
        word_buffer.clear()
        last_sentence     = ""
        pending_word      = None
        stability_counter = 0
        cooldown_counter  = 0
        no_hand_counter   = 0
        print("  Cleared.")

    elif key in (ord('z'), ord('Z')):   # Undo
        if word_buffer:
            removed = word_buffer.pop()
            print(f"  Undone: {removed}")

cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("\n✓ Session ended.")
