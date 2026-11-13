import cv2
import numpy as np
import os
import pickle
import urllib.request
import threading
import time
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tensorflow as tf
import soundfile as sf
import sounddevice as sd
from kokoro_onnx import Kokoro

from sentence_builder import SentenceBuilder



# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR       = "bsl_data"
MODEL_PATH     = os.path.join(DATA_DIR, "lstm_model.h5")
ENCODER_PATH   = os.path.join(DATA_DIR, "lstm_label_encoder.pkl")
MP_MODEL_FILE  = os.path.join(DATA_DIR, "hand_landmarker.task")
MP_MODEL_URL   = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
KOKORO_MODEL_FILE = os.path.join(DATA_DIR, "kokoro-v1.0.onnx")
KOKORO_MODEL_URL  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = os.path.join(DATA_DIR, "voices-v1.0.bin")
KOKORO_VOICES_URL  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

SEQUENCE_LEN      = 30
FEATURE_DIM       = 126
CONF_THRESHOLD    = 0.60
HOLD_FRAMES       = 90
PRED_BUFFER_SIZE  = 3
INFER_EVERY       = 2

# ── Word accumulator ───────────────────────────────────────────────────────────
# How many frames the LSTM must continuously predict the SAME word
# (above CONF_THRESHOLD) before it is committed to the sentence buffer.
COMMIT_FRAMES    = 25    # ~0.8s at 30 fps
COMMIT_COOLDOWN  = 50    # frames to ignore after committing (~1.7s)
SENTENCE_TIMEOUT = 90    # frames of no hand → sentence complete (~3s)
MAX_BUFFER_WORDS = 7     # auto-complete after this many words

ZERO_FRAME = np.zeros(FEATURE_DIM, dtype=np.float32)

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
    return np.concatenate([normalize_hand(frame_vec[:63]),
                           normalize_hand(frame_vec[63:])])


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


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_panel(img, x1, y1, x2, y2, color=(10, 20, 40), alpha=0.72):
    """Semi-transparent filled rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_shadow(img, text, pos, scale, color, thickness=2, max_width=None):
    """Text with drop shadow for readability."""
    if max_width:
        while scale > 0.35:
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
            if tw <= max_width:
                break
            scale -= 0.02

    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ── Download MediaPipe model ───────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(MP_MODEL_FILE):
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
    print(f"✓ Saved to {MP_MODEL_FILE}")

if not os.path.exists(KOKORO_MODEL_FILE):
    print(f"Downloading Kokoro TTS model (this might take a minute)...")
    urllib.request.urlretrieve(KOKORO_MODEL_URL, KOKORO_MODEL_FILE)
    print(f"✓ Saved to {KOKORO_MODEL_FILE}")

if not os.path.exists(KOKORO_VOICES_FILE):
    print(f"Downloading Kokoro TTS voices...")
    urllib.request.urlretrieve(KOKORO_VOICES_URL, KOKORO_VOICES_FILE)
    print(f"✓ Saved to {KOKORO_VOICES_FILE}")

# ── Load LSTM model ────────────────────────────────────────────────────────────
print("=" * 60)
print("Yani  BSL LSTM Translator  (Word Accumulator + Sentence Formation)")
print("=" * 60)

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("\n✗ LSTM model not found.")
    print("  1. Collect sequences: python data_collection_lstm.py")
    print("  2. Train model:       python train_lstm.py")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
_dummy = np.zeros((1, SEQUENCE_LEN, FEATURE_DIM), dtype=np.float32)
_ = model(_dummy, training=False)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

print(f"\n✓ LSTM model  — {len(le.classes_)} classes: {', '.join(le.classes_)}")

# ── Sentence builder (loads GPT-2 in background) ──────────────────────────────
print("\n  Starting GPT-2 sentence builder in background...")
sentence_builder = SentenceBuilder()

print("\n  Starting Kokoro TTS...")
try:
    kokoro = Kokoro(KOKORO_MODEL_FILE, KOKORO_VOICES_FILE)
    print("  ✓ Kokoro TTS ready")
except Exception as e:
    kokoro = None
    print(f"  ✗ Kokoro TTS failed to load: {e}")

# ── TTS Audio worker ───────────────────────────────────────────────────────────
def _play_audio(text):
    if kokoro is None or not text:
        return
    try:
        samples, sample_rate = kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
        sd.play(samples, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"\n  ✗ TTS Playback error: {e}")

# ── MediaPipe ──────────────────────────────────────────────────────────────────
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
print("\nWebcam open. ESC: quit   C: clear   Z: undo\n")

# ── Shared LSTM inference state ────────────────────────────────────────────────
frame_buffer    = deque(maxlen=SEQUENCE_LEN)
pred_buffer     = deque(maxlen=PRED_BUFFER_SIZE)
lock            = threading.Lock()
infer_result    = {"word": None, "conf": 0.0}
infer_pending   = threading.Event()

locked_word      = None
locked_conf      = 0.0
hold_frames_left = 0

for _ in range(SEQUENCE_LEN):
    frame_buffer.append(ZERO_FRAME.copy())


def inference_worker():
    while True:
        infer_pending.wait()
        infer_pending.clear()
        with lock:
            seq = np.array(frame_buffer, dtype=np.float32)[np.newaxis]
        proba = model(seq, training=False).numpy()[0]
        idx   = int(np.argmax(proba))
        infer_result["word"] = le.classes_[idx]
        infer_result["conf"] = float(proba[idx])


infer_thread = threading.Thread(target=inference_worker, daemon=True)
infer_thread.start()


def recency_weighted_prediction(buf):
    if not buf:
        return None, 0.0
    items   = list(buf)
    n       = len(items)
    weights = np.arange(1, n + 1, dtype=np.float32)
    word_scores = {}
    for w, (word, conf) in zip(weights, items):
        word_scores.setdefault(word, 0.0)
        word_scores[word] += w * conf
    best_word   = max(word_scores, key=word_scores.__getitem__)
    win_weights = [weights[i] for i, (word, _) in enumerate(items) if word == best_word]
    win_confs   = [conf       for (word, conf)   in items           if word == best_word]
    avg_conf    = float(np.dot(win_weights, win_confs) / sum(win_weights))
    return best_word, avg_conf


# ── Word accumulator state ─────────────────────────────────────────────────────
word_buffer      = []    # committed words this sentence
commit_word      = None  # word currently building toward commit
commit_frames    = 0     # frames current commit_word has been predicted
commit_cooldown  = 0     # post-commit cooldown frames
no_hand_timer    = 0     # frames since last hand detection

# ── Sentence state ─────────────────────────────────────────────────────────────
last_sentence    = ""
sentence_building = False
lock_flash        = 0


def _run_sentence_builder(words):
    global last_sentence, sentence_building
    result        = sentence_builder.build(list(words))
    last_sentence = result
    sentence_building = False
    print(f'\n  ✓ Sentence: "{result}"')


def complete_sentence():
    global sentence_building, word_buffer, last_sentence
    global commit_word, commit_frames, commit_cooldown, no_hand_timer
    if not word_buffer:
        return
    print(f"  Signing complete → words: {word_buffer}")
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
    commit_word   = None
    commit_frames = 0
    commit_cooldown = 0
    no_hand_timer   = 0


# ── FPS ────────────────────────────────────────────────────────────────────────
fps_buf     = deque(maxlen=30)
last_t      = time.perf_counter()
infer_frame = 0

# ── Main loop ──────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.perf_counter()
    fps_buf.append(1.0 / max(now - last_t, 1e-6))
    last_t = now
    fps    = np.mean(fps_buf)

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W  = frame.shape[:2]

    if sentence_building:
        # Pause hand detection while Gemini API is fetching the sentence
        h1 = np.zeros(63, dtype=np.float32)
        h2 = np.zeros(63, dtype=np.float32)
        num_hands = 0
    else:
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result       = landmarker.detect_for_video(mp_image, timestamp_ms)

        h1, h2    = get_left_right(result)
        num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                draw_hand(frame, hand_lms)

    # ── Lock flash overlay ─────────────────────────────────────────────────────
    if lock_flash > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 255, 120), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        lock_flash -= 1

    # ── LSTM frame buffer + inference ──────────────────────────────────────────
    raw_frame  = np.concatenate([h1, h2])
    norm_frame = normalize_frame(raw_frame)
    with lock:
        frame_buffer.append(norm_frame)

    infer_frame += 1
    if num_hands > 0 and infer_frame % INFER_EVERY == 0:
        infer_pending.set()

    if infer_result["word"] is not None and num_hands > 0:
        pred_buffer.append((infer_result["word"], infer_result["conf"]))
        infer_result["word"] = None

    smooth_word, smooth_conf = recency_weighted_prediction(pred_buffer)

    # ── LSTM lock-in (unchanged from original) ────────────────────────────────
    if num_hands > 0:
        hold_frames_left = 0
        if smooth_word and smooth_conf >= CONF_THRESHOLD:
            if smooth_word != locked_word:
                pred_buffer.clear()
            locked_word = smooth_word
            locked_conf = smooth_conf
    else:
        pred_buffer.clear()
        if locked_word is not None and hold_frames_left == 0:
            hold_frames_left = HOLD_FRAMES
        elif hold_frames_left > 0:
            hold_frames_left -= 1
            if hold_frames_left == 0:
                locked_word = None
                locked_conf = 0.0

    # ── Word accumulator (built on top of LSTM lock-in) ───────────────────────
    if num_hands > 0 and locked_word:
        no_hand_timer = 0

        if commit_cooldown > 0:
            commit_cooldown -= 1
        else:
            # Track stability of locked_word
            if locked_word == commit_word:
                commit_frames += 1
            else:
                commit_word   = locked_word
                commit_frames = 1

            # Commit when stable long enough
            if commit_frames >= COMMIT_FRAMES:
                # Prevent consecutive duplicates natively
                if not word_buffer or word_buffer[-1] != commit_word:
                    word_buffer.append(commit_word)
                    print(f"  + Committed: {commit_word}  [{' → '.join(word_buffer)}]")
                else:
                    print(f"  ~ Skipped duplicate: {commit_word}")
                
                lock_flash      = 5
                commit_cooldown = COMMIT_COOLDOWN
                commit_frames   = 0
                # Auto-complete removed; waiting for explicit ENTER press.

    elif num_hands == 0 and hold_frames_left == 0:
        # Hand fully gone (hold expired)
        no_hand_timer += 1
        commit_frames  = 0
        commit_word    = None
        # Auto-timeout removed; waiting for explicit ENTER press.

    # ══════════════════════════════════════════════════════════════════════════
    # DRAWING
    # ══════════════════════════════════════════════════════════════════════════

    # ── FPS + model label (top right) ─────────────────────────────────────────
    put_shadow(frame, f"FPS: {fps:.0f}", (W - 120, 30), 0.65, (180, 180, 50))
    put_shadow(frame, "LSTM + Sentence", (W - 210, 58), 0.5, (200, 200, 50))

    # ── GPT-2 ready indicator ─────────────────────────────────────────────────
    if sentence_builder.loading:
        put_shadow(frame, "GPT-2 loading...", (10, 22), 0.48, (255, 200, 50))
    else:
        cv2.circle(frame, (14, 18), 6, (0, 200, 100), -1)

    # ── Word display (LSTM prediction — existing style) ───────────────────────
    if locked_word and num_hands > 0:
        wcolor = (0, 255, 180)
        put_shadow(frame, locked_word.upper(), (10, 80), 2.2, wcolor, 4)
        put_shadow(frame, f"{locked_conf*100:.0f}% confident", (10, 115), 0.75, wcolor, 2)

    elif locked_word and hold_frames_left > 0:
        fade   = hold_frames_left / HOLD_FRAMES
        bright = int(140 + 115 * fade)
        wcolor = (0, bright, int(bright * 0.7))
        put_shadow(frame, locked_word.upper(), (10, 80), 2.2, wcolor, 4)
        bar_total  = W - 20
        bar_filled = int(fade * bar_total)
        cv2.rectangle(frame, (10, 125), (10 + bar_filled, 135), wcolor, -1)
        cv2.rectangle(frame, (10, 125), (10 + bar_total, 135), (80, 80, 80), 1)
        put_shadow(frame, "Holding...", (10, 152), 0.55, wcolor, 1)

    elif num_hands == 0 and locked_word is None:
        put_shadow(frame, "Show a sign...", (10, 80), 1.1, (100, 100, 100), 2)

    # ── Commit stability bar ───────────────────────────────────────────────────
    if commit_word and num_hands > 0 and commit_cooldown == 0:
        bar_x1, bar_y1 = 10, 162
        bar_x2, bar_y2 = W - 10, 182
        progress = min(commit_frames / COMMIT_FRAMES, 1.0)
        filled_w = int(progress * (bar_x2 - bar_x1))

        draw_panel(frame, bar_x1 - 2, bar_y1 - 14,
                   bar_x2 + 2, bar_y2 + 2, (20, 20, 20), alpha=0.65)
        if filled_w > 0:
            cv2.rectangle(frame, (bar_x1, bar_y1),
                          (bar_x1 + filled_w, bar_y2), (0, 220, 120), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1),
                      (bar_x2, bar_y2), (150, 150, 150), 1)
        remaining_s = (COMMIT_FRAMES - commit_frames) / 30
        lbl = (f"Committing: {commit_word.upper()} ({remaining_s:.1f}s)"
               if progress < 1.0 else f"Committed: {commit_word.upper()}!")
        put_shadow(frame, lbl, (bar_x1 + 4, bar_y1 - 2), 0.46, (200, 255, 200), 1)

    elif commit_cooldown > 0 and num_hands > 0:
        put_shadow(frame, "Move to next sign...", (10, 167), 0.48, (180, 180, 80), 1)

    # ── No-hand countdown ─────────────────────────────────────────────────────
    if num_hands == 0 and hold_frames_left == 0 and word_buffer:
        put_shadow(frame, "Press ENTER to build sentence",
                   (10, 195), 0.5, (100, 255, 100), 1)

    # ── Hand count ────────────────────────────────────────────────────────────
    if num_hands > 0:
        h_color = (0, 255, 0) if num_hands == 2 else (0, 200, 255)
        put_shadow(frame, f"{num_hands} hand(s)", (10, 210), 0.55, h_color, 1)

    # ── Word buffer panel ──────────────────────────────────────────────────────
    buf_y = H - 125
    draw_panel(frame, 5, buf_y, W - 5, buf_y + 42)
    if word_buffer:
        buf_text = "  ➜  ".join(w.upper() for w in word_buffer)
        put_shadow(frame, buf_text, (12, buf_y + 28), 0.72, (0, 200, 255), 2, max_width=W-30)
    else:
        put_shadow(frame, "(sign a word to begin...)",
                   (12, buf_y + 26), 0.6, (100, 100, 100), 1)

    # ── Sentence panel ─────────────────────────────────────────────────────────
    sent_y = H - 75
    draw_panel(frame, 5, sent_y, W - 5, sent_y + 45, color=(10, 10, 30), alpha=0.80)
    if sentence_building:
        dots = "." * ((int(time.time() * 2)) % 4)
        put_shadow(frame, f"Building sentence{dots}", (12, sent_y + 30),
                   0.72, (255, 200, 50), 2)
    elif last_sentence:
        put_shadow(frame, f'"{last_sentence}"', (12, sent_y + 30),
                   0.80, (255, 255, 160), 2, max_width=W-30)
    else:
        put_shadow(frame, "(sentence will appear here)",
                   (12, sent_y + 30), 0.6, (80, 80, 80), 1)

    # ── Bottom hints ───────────────────────────────────────────────────────────
    put_shadow(frame, "ENTER: build   ESC: quit   C: clear   Z: undo   S: speak",
               (10, H - 8), 0.38, (150, 150, 150), 1)

    cv2.imshow("Yani BSL LSTM Translator", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:   # ESC
        break

    elif key == 13: # ENTER
        complete_sentence()

    elif key in (ord('s'), ord('S')):
        if last_sentence and not sentence_building:
            print("  🔊 Speaking sentence...")
            threading.Thread(target=_play_audio, args=(last_sentence,), daemon=True).start()

    elif key in (ord('c'), ord('C')):
        word_buffer.clear()
        last_sentence    = ""
        commit_word      = None
        commit_frames    = 0
        commit_cooldown  = 0
        no_hand_timer    = 0
        print("  Cleared.")

    elif key in (ord('z'), ord('Z')):
        if word_buffer:
            removed = word_buffer.pop()
            print(f"  Undone: {removed}")

# ── Cleanup ────────────────────────────────────────────────────────────────────
cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("\n✓ Session ended.")
