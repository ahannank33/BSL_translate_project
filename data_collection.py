import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os
import urllib.request
from datetime import datetime

# Configuration
DATA_DIR = "bsl_data"
CSV_FILE = os.path.join(DATA_DIR, "bsl_landmarks.csv")

# CSV columns: label + 21 landmarks × 3 coords × 2 hands + timestamp
HAND1_COLS = [f'h1_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
HAND2_COLS = [f'h2_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
FIELDNAMES = ['label'] + HAND1_COLS + HAND2_COLS + ['timestamp']

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

MP_MODEL_FILE = os.path.join(DATA_DIR, "hand_landmarker.task")
MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(MP_MODEL_FILE):
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
    print(f"✓ Saved to {MP_MODEL_FILE}")

# ── Hand connections for drawing ──────────────────────────────────────────────
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

# ── MediaPipe Tasks hand landmarker ──────────────────────────────────────────
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_FILE),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

# BSL Alphabet letters (A-Z)
BSL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
current_letter_index = 0
current_letter = BSL_LETTERS[current_letter_index]

# Sample counter
sample_counts = {letter: 0 for letter in BSL_LETTERS}

# Load existing data if available
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            if label in sample_counts:
                sample_counts[label] += 1
    print(f"Loaded existing data. Current counts: {sample_counts}")

# Initialize CSV file if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

# Video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("\n=== BSL Alphabet Data Collection (Two-Hand Mode) ===")
print("Instructions:")
print("- Press SPACE to capture current hand gesture(s)")
print("- Press N for next letter, P for previous letter")
print("- Press R to reset count for current letter")
print("- Press ESC to quit")
print("- Target: 500+ samples per letter")
print(f"\nStarting with letter: {current_letter}")

ZERO_HAND = [0.0] * 63  # 21 landmarks × 3 coords, used when a hand is absent


def get_left_right(detection_result):
    """Return (left_data, right_data) — always Left→h1, Right→h2."""
    left, right = None, None
    if detection_result.hand_landmarks:
        for hand_lms, hand_class in zip(detection_result.hand_landmarks,
                                        detection_result.handedness):
            side = hand_class[0].category_name  # 'Left' or 'Right'
            xs   = [lm.x for lm in hand_lms]
            ys   = [lm.y for lm in hand_lms]
            zs   = [lm.z for lm in hand_lms]
            vec  = xs + ys + zs
            if side == 'Left':
                left  = vec
            else:
                right = vec
    return (left or list(ZERO_HAND)), (right or list(ZERO_HAND))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result    = landmarker.detect(mp_image)

    hand1_data, hand2_data = get_left_right(result)
    num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            draw_hand(frame, hand_lms)

    hand_detected = num_hands > 0

    # Display information overlay
    overlay_y = 30

    # Current letter and count
    cv2.putText(frame, f"Current Letter: {current_letter}", (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    overlay_y += 40

    cv2.putText(frame, f"Samples: {sample_counts[current_letter]}/500", (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    overlay_y += 35

    # Hand detection status
    if num_hands == 2:
        status_color = (0, 255, 0)
        status_text = "2 Hands Detected - Ready!"
    elif num_hands == 1:
        status_color = (0, 200, 255)
        status_text = "1 Hand Detected - Ready!"
    else:
        status_color = (0, 0, 255)
        status_text = "No Hand Detected"
    cv2.putText(frame, status_text, (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    overlay_y += 35

    # Instructions
    cv2.putText(frame, "SPACE: Capture | N: Next | P: Prev | R: Reset | ESC: Quit",
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Progress bar
    progress = min(sample_counts[current_letter] / 500.0, 1.0)
    bar_width = int(progress * (frame.shape[1] - 20))
    cv2.rectangle(frame, (10, frame.shape[0] - 60), (10 + bar_width, frame.shape[0] - 40), (0, 255, 0), -1)
    cv2.rectangle(frame, (10, frame.shape[0] - 60), (frame.shape[1] - 10, frame.shape[0] - 40), (255, 255, 255), 2)

    cv2.imshow("BSL Data Collection", frame)

    # Key handling
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE - capture
        if hand_detected:
            h1 = hand1_data
            h2 = hand2_data

            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                row_data = {'label': current_letter, 'timestamp': datetime.now().isoformat()}
                for col, val in zip(HAND1_COLS, h1):
                    row_data[col] = val
                for col, val in zip(HAND2_COLS, h2):
                    row_data[col] = val
                writer.writerow(row_data)

            sample_counts[current_letter] += 1
            print(f"Captured {current_letter}: {sample_counts[current_letter]} samples  "
                  f"({num_hands} hand(s) detected)")
        else:
            print("No hand detected! Position your hand in frame.")

    elif key == ord('n') or key == ord('N'):  # Next letter
        current_letter_index = (current_letter_index + 1) % len(BSL_LETTERS)
        current_letter = BSL_LETTERS[current_letter_index]
        print(f"\nSwitched to letter: {current_letter} (Samples: {sample_counts[current_letter]})")

    elif key == ord('p') or key == ord('P'):  # Previous letter
        current_letter_index = (current_letter_index - 1) % len(BSL_LETTERS)
        current_letter = BSL_LETTERS[current_letter_index]
        print(f"\nSwitched to letter: {current_letter} (Samples: {sample_counts[current_letter]})")

    elif key == ord('r') or key == ord('R'):  # Reset current letter count
        confirm = input(f"\nAre you sure you want to reset count for '{current_letter}'? (y/n): ")
        if confirm.lower() == 'y':
            sample_counts[current_letter] = 0
            print(f"Reset count for {current_letter}")

cap.release()
cv2.destroyAllWindows()

print("\n=== Data Collection Summary ===")
total_samples = sum(sample_counts.values())
print(f"Total samples collected: {total_samples}")
for letter in BSL_LETTERS:
    status = "✓" if sample_counts[letter] >= 500 else "○"
    print(f"{status} {letter}: {sample_counts[letter]}/500")

print(f"\nData saved to: {CSV_FILE}")
