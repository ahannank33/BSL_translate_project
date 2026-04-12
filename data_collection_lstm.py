import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
import time
import json

"""
LSTM sequence data collection tool.
Records 30-frame landmark sequences for BSL word signs.

Each recorded sequence = 30 frames × 126 features (2 hands × 21 landmarks × xyz).
Saved as .npy files under:
    bsl_data/lstm_sequences/<word>/<sequence_id>.npy

Controls:
  SPACE    — start/stop recording a sequence (hold the sign while recording)
  N        — next word
  P        — previous word
  ESC      — quit

Tip: Hold your sign pose steady and press SPACE to start the 30-frame capture.
     For motion signs (wave, rise, rotate) — perform the motion DURING recording.
"""

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "bsl_data"
SEQ_DIR       = os.path.join(DATA_DIR, "lstm_sequences")
SEQUENCE_LEN  = 30    # frames per sequence
TARGET        = 200   # target sequences per word

# Countdown delay before recording starts (useful for two-handed signs)
# Set USE_DELAY = False to disable
USE_DELAY     = True
DELAY_SECONDS = 1

WORDS_FILE = "bsl_words.json"

if not os.path.exists(WORDS_FILE):
    # Default words if the file doesn't exist yet
    default_words = {
        "what"     : "Index finger extended, rock side to side",
        "hello"    : "Open hand, wave side to side",
        "thanks"   : "Flat hand from chin, move forward",
        "good"     : "Thumbs up",
        "bad"      : "Thumbs down",
        "morning"  : "Flat hand rises upward (like sunrise)",
        "afternoon": "Arm at 45° angle pointing forward",
        "you"      : "Point index finger directly at camera",
        "my"       : "Flat hand touches chest",
        "your"     : "Point index finger away from you",
        "name"     : "Two index fingers tap together",
        "how"      : "Fists together, rotate outward",
        "meet"     : "Both index fingers come together",
        "home"     : "Flat hand on cheek",
        "baby"     : "Cradle arms, rock side to side",
        "mother"   : "M-hand touch chin",
        "father"   : "F-hand touch forehead",
        "brother"  : "Both B-hands side by side",
        "sister"   : "Both S-hands, touch cheek then move out",
        "son"      : "B-hand on forehead, then flat hand",
        "daughter" : "D-hand on chin, then flat hand",
        "luck"     : "L-hand twist at chin",
        "step"     : "Both flat hands, alternate stepping motion"
    }
    with open(WORDS_FILE, "w") as f:
        json.dump(default_words, f, indent=4)

with open(WORDS_FILE, "r") as f:
    BSL_WORDS = json.load(f)

WORD_LIST  = list(BSL_WORDS.keys())
ZERO_HAND  = [0.0] * 63

# ── MediaPipe setup ────────────────────────────────────────────────────────────
os.makedirs(SEQ_DIR, exist_ok=True)
for word in WORD_LIST:
    os.makedirs(os.path.join(SEQ_DIR, word), exist_ok=True)

MP_MODEL_FILE = os.path.join(DATA_DIR, "hand_landmarker.task")
MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

if not os.path.exists(MP_MODEL_FILE):
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
    print(f"✓ Saved to {MP_MODEL_FILE}")


def count_sequences(word):
    word_dir = os.path.join(SEQ_DIR, word)
    return len([f for f in os.listdir(word_dir) if f.endswith(".npy")])


def next_seq_id(word):
    return count_sequences(word)


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


def get_left_right(detection_result):
    left, right = None, None
    if detection_result.hand_landmarks:
        for hand_lms, hand_class in zip(detection_result.hand_landmarks,
                                        detection_result.handedness):
            side = hand_class[0].category_name
            xs   = [lm.x for lm in hand_lms]
            ys   = [lm.y for lm in hand_lms]
            zs   = [lm.z for lm in hand_lms]
            vec  = xs + ys + zs
            if side == 'Left':
                left  = vec
            else:
                right = vec
    return (left or list(ZERO_HAND)), (right or list(ZERO_HAND))


# ── MediaPipe landmarker ───────────────────────────────────────────────────────
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_FILE),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ── State ──────────────────────────────────────────────────────────────────────
cap          = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
word_idx     = 0
recording    = False
frame_buffer = []   # accumulates frames during recording
countdown_active = False
countdown_start_time = 0

sample_counts = {w: count_sequences(w) for w in WORD_LIST}

print("\n" + "=" * 60)
print("  BSL LSTM Sequence Data Collection")
print("=" * 60)
print("  SPACE    — start / stop recording a 30-frame sequence")
print("  N / P    — next / previous word")
print("  ESC      — quit")
print("=" * 60)
print("\nExisting sequence counts:")
for w, c in sample_counts.items():
    bar    = "█" * (c // 10) + "░" * max(0, (TARGET - c) // 10)
    status = " ✓" if c >= TARGET else ""
    print(f"  {w:<12} {c:>4}/{TARGET}  {bar}{status}")
print()

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

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            draw_hand(frame, hand_lms)

    current_word  = WORD_LIST[word_idx]
    current_count = sample_counts[current_word]
    hint          = BSL_WORDS[current_word]

    # ── Countdown logic ────────────────────────────────────────────────────────
    if countdown_active:
        elapsed = time.time() - countdown_start_time
        remaining = DELAY_SECONDS - elapsed
        if remaining <= 0:
            countdown_active = False
            if num_hands > 0:
                recording = True
                frame_buffer = []
                print(f"\n  ● Recording {current_word}...")
            else:
                print("  ✗ No hand detected after countdown — cancelled")
        else:
            cv2.putText(frame, f"Starting in {int(np.ceil(remaining))}...",
                        (frame.shape[1]//2 - 180, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)

    # ── Recording logic ────────────────────────────────────────────────────────
    if recording:
        if num_hands > 0:
            frame_buffer.append(h1 + h2)   # 126 values per frame
        else:
            # pad with zeros if hand momentarily lost
            frame_buffer.append(ZERO_HAND + ZERO_HAND)

        if len(frame_buffer) >= SEQUENCE_LEN:
            # Save the completed sequence
            seq_array = np.array(frame_buffer[:SEQUENCE_LEN], dtype=np.float32)
            seq_id    = next_seq_id(current_word)
            save_path = os.path.join(SEQ_DIR, current_word, f"{seq_id}.npy")
            np.save(save_path, seq_array)
            sample_counts[current_word] += 1
            print(f"  [{current_word}] sequence {seq_id} saved  "
                  f"({sample_counts[current_word]}/{TARGET})")
            recording    = False
            frame_buffer = []

        # Red recording border
        progress = len(frame_buffer) / SEQUENCE_LEN
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                      (0, 0, 255), 8)
        # Progress bar along top
        bar_w = int(progress * frame.shape[1])
        cv2.rectangle(frame, (0, 0), (bar_w, 8), (0, 80, 255), -1)
        cv2.putText(frame, f"● REC  {len(frame_buffer)}/{SEQUENCE_LEN}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # ── Word & progress UI ────────────────────────────────────────────────────
    done_color = (0, 255, 100) if current_count >= TARGET else (0, 200, 255)
    cv2.putText(frame, f"WORD: {current_word.upper()}",
                (10, 65 if not recording else 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, done_color, 3)

    prog_text = f"Sequences: {current_count} / {TARGET}"
    cv2.putText(frame, prog_text,
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    # Progress bar
    bar_w_full = frame.shape[1] - 20
    filled_w   = int(min(current_count / TARGET, 1.0) * bar_w_full)
    cv2.rectangle(frame, (10, 118), (10 + filled_w, 133), (0, 220, 0), -1)
    cv2.rectangle(frame, (10, 118), (10 + bar_w_full, 133), (180, 180, 180), 2)

    if current_count >= TARGET:
        cv2.putText(frame, "TARGET MET!", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    cv2.putText(frame, f"Hint: {hint}",
                (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    # Hand status
    if num_hands == 2:
        s_color, s_text = (0, 255, 0), "2 Hands  Ready!"
    elif num_hands == 1:
        s_color, s_text = (0, 200, 255), "1 Hand   Ready!"
    else:
        s_color, s_text = (0, 0, 255), "No Hand Detected"
    cv2.putText(frame, s_text, (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, s_color, 2)

    # Word index
    cv2.putText(frame, f"Word {word_idx + 1} of {len(WORD_LIST)}",
                (frame.shape[1] - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Bottom hints
    hint_text = "SPACE:Record  N:Next  P:Prev  ESC:Quit"
    cv2.putText(frame, hint_text,
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("BSL LSTM Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:   # ESC
        break

    elif key == ord(' '):
        if not recording and not countdown_active:
            if USE_DELAY and DELAY_SECONDS > 0:
                countdown_active = True
                countdown_start_time = time.time()
                print(f"\n  ⏱ Starting in {DELAY_SECONDS} seconds... Get ready!")
            else:
                if num_hands > 0:
                    recording    = True
                    frame_buffer = []
                    print(f"\n  ● Recording {current_word}...")
                else:
                    print("  ✗ No hand detected — show your hand first")
        else:
            # Cancel recording or countdown
            recording    = False
            countdown_active = False
            frame_buffer = []
            print("  ✗ Recording / Countdown cancelled")

    elif key in (ord('n'), ord('N')):
        word_idx = (word_idx + 1) % len(WORD_LIST)
        recording    = False
        frame_buffer = []
        w = WORD_LIST[word_idx]
        print(f"\n→ {w.upper()}  ({sample_counts[w]}/{TARGET})  — {BSL_WORDS[w]}")

    elif key in (ord('p'), ord('P')):
        word_idx = (word_idx - 1) % len(WORD_LIST)
        recording    = False
        frame_buffer = []
        w = WORD_LIST[word_idx]
        print(f"\n← {w.upper()}  ({sample_counts[w]}/{TARGET})  — {BSL_WORDS[w]}")

# ── Cleanup ────────────────────────────────────────────────────────────────────
cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("  Session Summary")
print("=" * 60)
for w in WORD_LIST:
    c      = sample_counts[w]
    status = "✓" if c >= TARGET else f"need {TARGET - c} more"
    print(f"  {w:<12}  {c:>4}/{TARGET}  {status}")
print(f"\n  Sequences saved in: {SEQ_DIR}/")
print("\n  Run 'python train_lstm.py' to train the LSTM model.")
print("=" * 60)
