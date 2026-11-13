import kagglehub
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



# ── Output config ──────────────────────────────────────────────────────────────
DATA_DIR    = "bsl_data"
OUTPUT_CSV  = os.path.join(DATA_DIR, "bsl_landmarks.csv")

HAND1_COLS  = [f'h1_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
HAND2_COLS  = [f'h2_{c}{i}' for c in ('x', 'y', 'z') for i in range(21)]
FIELDNAMES  = ['label'] + HAND1_COLS + HAND2_COLS + ['timestamp']
ZERO_HAND   = [0.0] * 63

os.makedirs(DATA_DIR, exist_ok=True)

# ── Worker functions (must be at module level for multiprocessing on Windows) ──

MP_MODEL_FILE = os.path.join("bsl_data", "hand_landmarker.task")
MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


def _worker_init():
    """Each worker process creates its own MediaPipe Tasks detector."""
    global _detector
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_FILE),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    _detector = vision.HandLandmarker.create_from_options(opts)


def _extract_landmarks(hand_lm):
    """Return [x0..x20, y0..y20, z0..z20] for one hand."""
    xs = [lm.x for lm in hand_lm.landmark]
    ys = [lm.y for lm in hand_lm.landmark]
    zs = [lm.z for lm in hand_lm.landmark]
    return xs + ys + zs
def _process_image(args):
    """
    Run MediaPipe on a single image.
    Returns (label, h1, h2) on success, or None if no hand detected.
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
        side = hand_class[0].category_name  # 'Left' or 'Right'
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


# ── Entry point guard required for multiprocessing on Windows ─────────────────
if __name__ == '__main__':

    # ── Download MediaPipe hand model if needed ────────────────────────────────
    os.makedirs("bsl_data", exist_ok=True)
    if not os.path.exists(MP_MODEL_FILE):
        print("Downloading MediaPipe hand landmark model...")
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
        print(f"✓ Model saved to {MP_MODEL_FILE}\n")

    # ── Step 1: Download dataset ───────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Downloading BSL Static Alphabet Dataset")
    print("=" * 60)

    path = kagglehub.dataset_download("philipilono/bsl-static-alphabet")
    print(f"\nDataset downloaded to: {path}")

    # ── Step 2: Find image files ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Scanning for images")
    print("=" * 60)

    # Dataset structure: <path>/Dataset/Dataset/train/<letter>/
    dataset_root = os.path.join(path, "Dataset", "Dataset")
    if not os.path.isdir(dataset_root):
        dataset_root = os.path.join(path, "Dataset")
    if not os.path.isdir(dataset_root):
        dataset_root = path

    image_pairs = []
    for split in ("train", "test"):
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            print(f"  '{split}' folder not found, skipping.")
            continue

        letter_dirs = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
            and len(d) == 1 and d.isalpha()
        ])

        split_count = 0
        for letter in letter_dirs:
            letter_dir = os.path.join(split_dir, letter)
            images = (glob.glob(os.path.join(letter_dir, "*.jpg"))  +
                      glob.glob(os.path.join(letter_dir, "*.jpeg")) +
                      glob.glob(os.path.join(letter_dir, "*.png")))
            for img_path in images:
                image_pairs.append((img_path, letter.upper()))
            split_count += len(images)

        print(f"  '{split}': {split_count} images across {len(letter_dirs)} "
              f"letters ({', '.join(letter_dirs)})")

    total = len(image_pairs)
    print(f"\n  Total images to process: {total}")

    if not image_pairs:
        print("✗ Could not find letter sub-folders.")
        print(f"  Please inspect: {path}")
        exit(1)

    # ── Step 3: Parallel MediaPipe extraction ──────────────────────────────────
    num_workers = max(1, multiprocessing.cpu_count() - 1)  # leave 1 core free
    print("\n" + "=" * 60)
    print(f"Step 3: Extracting landmarks ({num_workers} parallel workers)")
    print("=" * 60)
    print(f"  Your CPU has {multiprocessing.cpu_count()} cores — using {num_workers}\n")

    t_start   = time.time()
    saved     = 0
    skipped   = 0
    per_label = {}

    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='') as f:   # 'a' = append, never overwrites
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()   # only write header on a brand new file

        # Pool: each worker gets its own MediaPipe detector via _worker_init
        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_worker_init
        ) as pool:

            # imap_unordered returns results as they complete (fastest)
            for idx, result in enumerate(
                pool.imap_unordered(_process_image, image_pairs, chunksize=16)
            ):
                if idx % 500 == 0:
                    elapsed = time.time() - t_start
                    rate    = (idx + 1) / elapsed if elapsed > 0 else 0
                    eta     = (total - idx) / rate if rate > 0 else 0
                    print(f"  {idx + 1}/{total}  "
                          f"({rate:.0f} img/s  ETA {eta/60:.1f} min)", flush=True)

                if result is None:
                    skipped += 1
                    continue

                label, h1, h2 = result
                row = {'label': label, 'timestamp': ''}
                for col, val in zip(HAND1_COLS, h1):
                    row[col] = val
                for col, val in zip(HAND2_COLS, h2):
                    row[col] = val
                writer.writerow(row)

                saved += 1
                per_label[label] = per_label.get(label, 0) + 1

    total_time = time.time() - t_start

    # ── Step 4: Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\n  Total time       : {total_time/60:.1f} minutes")
    print(f"  Images processed : {total}")
    print(f"  Landmarks saved  : {saved}")
    print(f"  Skipped (no hand): {skipped}")
    print(f"\n  Samples per letter:")
    for letter in sorted(per_label):
        bar = "█" * (per_label[letter] // 100)
        print(f"    {letter}: {per_label[letter]:>5}  {bar}")

    print(f"\n✓ Saved to: {OUTPUT_CSV}")
    print("\nNext steps:")
    print("  python test_detection.py  ← accuracy check + live webcam")
