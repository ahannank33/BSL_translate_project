import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os
import urllib.request



# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "bsl_data"
CSV_FILE   = os.path.join(DATA_DIR, "word_landmarks.csv")
TARGET     = 200   # target samples per word

# ── Word list — same 22 classes the model already knows ───────────────────────
# Add or remove words here to expand the vocabulary.
# Each entry: "word": "brief BSL hint"
BSL_WORDS = {
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
    "step"     : "Both flat hands, alternate stepping motion",
}

WORD_LIST = list(BSL_WORDS.keys())

# ── CSV columns (must match word_landmarks.csv — no timestamp) ─────────────────
HAND1_COLS = [f'h1_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
HAND2_COLS = [f'h2_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
FIELDNAMES = ['label'] + HAND1_COLS + HAND2_COLS
ZERO_HAND  = [0.0] * 63

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)

MP_MODEL_FILE = os.path.join(DATA_DIR, "hand_landmarker.task")
MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

if not os.path.exists(MP_MODEL_FILE):
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
    print(f"✓ Saved to {MP_MODEL_FILE}")

# Ensure CSV exists with header
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# Load existing counts
sample_counts = {w: 0 for w in WORD_LIST}
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'r') as f:
        for row in csv.DictReader(f):
            lbl = row.get('label', '').strip()
            if lbl in sample_counts:
                sample_counts[lbl] += 1

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


# ── MediaPipe ──────────────────────────────────────────────────────────────────
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_FILE),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

word_idx     = 0
flash_frames = 0    # green flash after capture

print("\n" + "=" * 60)
print("  BSL Word Sign Data Collection")
print("=" * 60)
print("  SPACE    — capture frame")
print("  N / P    — next / previous word")
print("  R        — retrain model now")
print("  ESC      — quit")
print("=" * 60)
print(f"\nCurrent word: {WORD_LIST[word_idx].upper()}")
print("Look up how to sign it, then hold the pose and press SPACE.\n")

# Print existing counts summary
print("Existing sample counts:")
for w, c in sample_counts.items():
    bar = "█" * (c // 10) + "░" * max(0, (TARGET - c) // 10)
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

    # ── Green flash overlay after capture ─────────────────────────────────────
    if flash_frames > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        flash_frames -= 1

    # ── Word & progress ────────────────────────────────────────────────────────
    done_color = (0, 255, 100) if current_count >= TARGET else (0, 200, 255)
    cv2.putText(frame, f"WORD: {current_word.upper()}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, done_color, 3)

    prog_text = f"Samples: {current_count} / {TARGET}"
    cv2.putText(frame, prog_text,
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Progress bar
    bar_w    = frame.shape[1] - 20
    filled_w = int(min(current_count / TARGET, 1.0) * bar_w)
    cv2.rectangle(frame, (10, 98), (10 + filled_w, 115), (0, 220, 0), -1)
    cv2.rectangle(frame, (10, 98), (10 + bar_w, 115), (180, 180, 180), 2)

    # Done badge
    if current_count >= TARGET:
        cv2.putText(frame, "TARGET MET!", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    # ── BSL hint ───────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Hint: {hint}",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)

    # ── Hand status ────────────────────────────────────────────────────────────
    if num_hands == 2:
        s_color, s_text = (0, 255, 0), "2 Hands  Ready!"
    elif num_hands == 1:
        s_color, s_text = (0, 200, 255), "1 Hand   Ready!"
    else:
        s_color, s_text = (0, 0, 255), "No Hand Detected"
    cv2.putText(frame, s_text, (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)

    # ── Word index breadcrumb ──────────────────────────────────────────────────
    cv2.putText(frame, f"Word {word_idx + 1} of {len(WORD_LIST)}",
                (frame.shape[1] - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # ── Bottom key hints ───────────────────────────────────────────────────────
    cv2.putText(frame, "SPACE:Capture  N:Next  P:Prev  R:Retrain  ESC:Quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("BSL Word Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # ── ESC — quit ─────────────────────────────────────────────────────────────
    if key == 27:
        break

    # ── SPACE — capture ────────────────────────────────────────────────────────
    elif key == ord(' '):
        if num_hands > 0:
            row = {'label': current_word}
            for col, val in zip(HAND1_COLS, h1):
                row[col] = round(val, 6)
            for col, val in zip(HAND2_COLS, h2):
                row[col] = round(val, 6)
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(row)
            sample_counts[current_word] += 1
            flash_frames = 4
            print(f"  [{current_word}] {sample_counts[current_word]} samples")
        else:
            print("  ✗ No hand in frame — move your hand into view")

    # ── N — next word ──────────────────────────────────────────────────────────
    elif key in (ord('n'), ord('N')):
        word_idx = (word_idx + 1) % len(WORD_LIST)
        print(f"\n→ {WORD_LIST[word_idx].upper()}  "
              f"({sample_counts[WORD_LIST[word_idx]]}/{TARGET} samples)  "
              f"— {BSL_WORDS[WORD_LIST[word_idx]]}")

    # ── P — previous word ──────────────────────────────────────────────────────
    elif key in (ord('p'), ord('P')):
        word_idx = (word_idx - 1) % len(WORD_LIST)
        print(f"\n← {WORD_LIST[word_idx].upper()}  "
              f"({sample_counts[WORD_LIST[word_idx]]}/{TARGET} samples)  "
              f"— {BSL_WORDS[WORD_LIST[word_idx]]}")

    # ── R — retrain ────────────────────────────────────────────────────────────
    elif key in (ord('r'), ord('R')):
        print("\nClosing webcam to retrain model...")
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("Running train_words.py...\n")
        os.system("python train_words.py")
        print("\nRetraining complete. Restart test_detection.py to use the new model.")
        exit(0)

# ── Session summary ────────────────────────────────────────────────────────────
cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("  Session Summary")
print("=" * 60)
total_new = sum(sample_counts.values())
for w in WORD_LIST:
    c      = sample_counts[w]
    status = "✓" if c >= TARGET else f"need {TARGET - c} more"
    print(f"  {w:<12}  {c:>4}/{TARGET}  {status}")
print(f"\n  CSV: {CSV_FILE}")
print("\n  Run 'python train_words.py' to retrain, then 'python test_detection.py'.")
print("=" * 60)
