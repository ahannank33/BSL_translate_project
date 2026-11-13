import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os
import glob
import multiprocessing
import time
import urllib.request



# ── Paths ──────────────────────────────────────────────────────────────────────
SLINGO_ROOT = r"C:\Users\ahann\Downloads\SlingoModels-main\SlingoModels-main"

# Scan both datasets, both train + test splits
DATASET_DIRS = [
    os.path.join(SLINGO_ROOT, "greetings_dataset", "train"),
    os.path.join(SLINGO_ROOT, "greetings_dataset", "test"),
    os.path.join(SLINGO_ROOT, "family_dataset",    "train"),
    os.path.join(SLINGO_ROOT, "family_dataset",    "test"),
]

DATA_DIR   = "bsl_data"
OUTPUT_CSV = os.path.join(DATA_DIR, "word_landmarks.csv")

HAND1_COLS = [f'h1_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
HAND2_COLS = [f'h2_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
FIELDNAMES = ['label'] + HAND1_COLS + HAND2_COLS
ZERO_HAND  = [0.0] * 63

os.makedirs(DATA_DIR, exist_ok=True)

# ── MediaPipe model ────────────────────────────────────────────────────────────
MP_MODEL_FILE = os.path.join(DATA_DIR, "hand_landmarker.task")
MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


# ── Worker functions (module-level required for Windows multiprocessing) ───────

def _worker_init():
    """Each worker process creates its own MediaPipe Tasks detector."""
    global _detector
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_FILE),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    _detector = vision.HandLandmarker.create_from_options(opts)


def _process_image(args):
    """
    Run MediaPipe on one image.
    Returns (label, h1, h2) on success or None if no hand detected.
    Hand ordering: Left → h1, Right → h2.
    """
    img_path, label = args

    img = cv2.imread(img_path)
    if img is None:
        return None

    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = _detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    left, right = None, None
    for hand_lms, hand_class in zip(result.hand_landmarks, result.handedness):
        side = hand_class[0].category_name        # 'Left' or 'Right'
        xs   = [lm.x for lm in hand_lms]
        ys   = [lm.y for lm in hand_lms]
        zs   = [lm.z for lm in hand_lms]
        vec  = xs + ys + zs
        if side == 'Left':
            left  = vec
        else:
            right = vec

    h1 = left  or list(ZERO_HAND)
    h2 = right or list(ZERO_HAND)
    return (label, h1, h2)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Download MediaPipe model if missing
    if not os.path.exists(MP_MODEL_FILE):
        print("Downloading MediaPipe hand landmark model...")
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
        print(f"✓ Model saved to {MP_MODEL_FILE}\n")

    # ── Collect all (image_path, label) pairs ─────────────────────────────────
    print("=" * 60)
    print("Scanning Slingo dataset folders...")
    print("=" * 60)

    image_pairs = []
    label_counts = {}

    for folder in DATASET_DIRS:
        if not os.path.isdir(folder):
            print(f"  ⚠  Folder not found, skipping: {folder}")
            continue

        jpgs = glob.glob(os.path.join(folder, "*.jpg"))
        for jpg_path in jpgs:
            # Extract label from filename prefix before first dot
            basename = os.path.basename(jpg_path)                 # e.g. afternoon.abc.jpg
            label    = basename.split('.')[0].lower().strip()     # e.g. "afternoon"
            if label:
                image_pairs.append((jpg_path, label))
                label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\n  Found {len(image_pairs)} images across {len(label_counts)} word classes:")
    for word, count in sorted(label_counts.items()):
        print(f"    {word:<20} {count} images")

    if not image_pairs:
        print("\n✗ No images found. Check SLINGO_ROOT path.")
        exit(1)

    # ── Parallel MediaPipe extraction ──────────────────────────────────────────
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"\n{'='*60}")
    print(f"Extracting landmarks ({num_workers} parallel workers)...")
    print(f"{'='*60}\n")

    t_start = time.time()
    saved   = 0
    skipped = 0

    file_exists = os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_worker_init,
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_process_image, image_pairs, chunksize=16),
                start=1
            ):
                if i % 200 == 0 or i == len(image_pairs):
                    elapsed = time.time() - t_start
                    print(f"  [{i:>5}/{len(image_pairs)}]  "
                          f"saved={saved}  skipped={skipped}  "
                          f"elapsed={elapsed:.1f}s")

                if result is None:
                    skipped += 1
                    continue

                label, h1, h2 = result
                row = {'label': label}
                for col, val in zip(HAND1_COLS, h1):
                    row[col] = round(val, 6)
                for col, val in zip(HAND2_COLS, h2):
                    row[col] = round(val, 6)
                writer.writerow(row)
                saved += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"✓ Done!  Saved {saved} rows, skipped {skipped} (no hand detected)")
    print(f"  Output : {OUTPUT_CSV}")
    print(f"  Time   : {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\nNext step: python train_words.py")
