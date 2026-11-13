#!/usr/bin/env python3
"""
BSL Translator — CustomTkinter UI  (v3 — matches user mockup)
=============================================================
Layout (3-column grid):

       col 0          col 1+2 (colspan)
 row 0: [webcam]    [detected sign + commit bar      ]
 row 1: [word buf + sentence + 🔊 (colspan 0+1)] [btns]
"""

import cv2
import numpy as np
import os, pickle, urllib.request, threading, time
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tensorflow as tf

try:
    import sounddevice as sd
    from kokoro_onnx import Kokoro
    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False

import customtkinter as ctk
from PIL import Image

from sentence_builder import SentenceBuilder

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR       = "bsl_data"
MODEL_PATH     = os.path.join(DATA_DIR, "lstm_model.h5")
ENCODER_PATH   = os.path.join(DATA_DIR, "lstm_label_encoder.pkl")
MP_MODEL_FILE  = os.path.join(DATA_DIR, "hand_landmarker.task")
MP_MODEL_URL   = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
KOKORO_MODEL_FILE  = os.path.join(DATA_DIR, "kokoro-v1.0.onnx")
KOKORO_VOICES_FILE = os.path.join(DATA_DIR, "voices-v1.0.bin")
KOKORO_MODEL_URL   = ("https://github.com/thewh1teagle/kokoro-onnx/"
                      "releases/download/model-files-v1.0/kokoro-v1.0.onnx")
KOKORO_VOICES_URL  = ("https://github.com/thewh1teagle/kokoro-onnx/"
                      "releases/download/model-files-v1.0/voices-v1.0.bin")

SEQUENCE_LEN      = 30
FEATURE_DIM       = 126
CONF_THRESHOLD    = 0.60
HOLD_FRAMES       = 90
PRED_BUFFER_SIZE  = 3
INFER_EVERY       = 2
COMMIT_FRAMES     = 25
COMMIT_COOLDOWN   = 50

ZERO_FRAME = np.zeros(FEATURE_DIM, dtype=np.float32)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,17),
]

# ══════════════════════════════════════════════════════════════════════════════
# Helpers (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
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
            xs, ys, zs = ([lm.x for lm in hand_lms],
                          [lm.y for lm in hand_lms],
                          [lm.z for lm in hand_lms])
            vec = np.array(xs + ys + zs, dtype=np.float32)
            if side == 'Left':
                left  = vec
            else:
                right = vec
    zeros = np.zeros(63, dtype=np.float32)
    return (left  if left  is not None else zeros.copy()), \
           (right if right is not None else zeros.copy())

def recency_weighted(buf):
    if not buf:
        return None, 0.0
    items   = list(buf)
    n       = len(items)
    weights = np.arange(1, n + 1, dtype=np.float32)
    scores  = {}
    for w, (word, conf) in zip(weights, items):
        scores.setdefault(word, 0.0)
        scores[word] += w * conf
    best = max(scores, key=scores.__getitem__)
    ww = [weights[i] for i, (wd, _) in enumerate(items) if wd == best]
    wc = [conf       for (wd, conf) in items            if wd == best]
    return best, float(np.dot(ww, wc) / sum(ww))

# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(MP_MODEL_FILE):
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_FILE)
    print(f"✓ Saved to {MP_MODEL_FILE}")

if _KOKORO_AVAILABLE:
    for url, path, label in [
        (KOKORO_MODEL_URL, KOKORO_MODEL_FILE, "Kokoro engine"),
        (KOKORO_VOICES_URL, KOKORO_VOICES_FILE, "Kokoro voices"),
    ]:
        if not os.path.exists(path):
            print(f"Downloading {label}...")
            urllib.request.urlretrieve(url, path)
            print(f"✓ {label} ready.")

print("=" * 60)
print("  Yani BSL Translator  UI")
print("=" * 60)

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("\n✗ LSTM model not found.")
    exit(1)

lstm_model = tf.keras.models.load_model(MODEL_PATH)
_ = lstm_model(np.zeros((1, SEQUENCE_LEN, FEATURE_DIM), dtype=np.float32), training=False)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)
print(f"\n✓ LSTM — {len(le.classes_)} classes: {', '.join(le.classes_)}")

print("\n  Starting sentence builder...")
sentence_builder = SentenceBuilder()

kokoro = None
if _KOKORO_AVAILABLE and os.path.exists(KOKORO_MODEL_FILE) and os.path.exists(KOKORO_VOICES_FILE):
    try:
        kokoro = Kokoro(KOKORO_MODEL_FILE, KOKORO_VOICES_FILE)
        print("  ✓ Kokoro TTS ready")
    except Exception as e:
        print(f"  ✗ Kokoro failed: {e}")

mp_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_FILE),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(mp_options)

def _play_audio(text):
    if kokoro is None or not text:
        return
    try:
        samples, sr = kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
        sd.play(samples, sr); sd.wait()
    except Exception as e:
        print(f"  ✗ TTS error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG      = "#111111"
PANEL   = "#1e1e1e"
BORDER  = "#2d2d2d"
GREEN   = "#7FFF00"
BLUE    = "#0099CC"
RED     = "#8B0000"
BGREEN  = "#228B22"
WHITE   = "#FFFFFF"
MUTED   = "#555555"
BUFCOL  = "#FFFFFF"
SENTCOL = "#CCCCCC"


class BSLApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Yani BSL Translator")
        self.geometry("1100x580")
        self.minsize(800, 480)
        self.configure(fg_color=BG)

        # Window icon — use logo.ico directly if present, otherwise generate from logo.png
        def _set_icon():
            try:
                _dir      = os.path.dirname(os.path.abspath(__file__))
                _ico_path = os.path.join(_dir, "logo.ico")

                if not os.path.exists(_ico_path):
                    # Generate ICO from logo.png (strips white background)
                    _png_path = os.path.join(_dir, "logo.png")
                    _src = Image.open(_png_path).convert("RGBA")
                    new_data = [
                        (255, 255, 255, 0) if px[0] > 230 and px[1] > 230 and px[2] > 230
                        else px
                        for px in _src.getdata()
                    ]
                    _src.putdata(new_data)
                    _sizes  = [16, 24, 32, 48, 64, 128, 256]
                    _frames = [_src.resize((s, s), Image.LANCZOS) for s in _sizes]
                    _frames[0].save(_ico_path, format="ICO",
                                    append_images=_frames[1:],
                                    sizes=[(s, s) for s in _sizes])

                self.iconbitmap(_ico_path)
            except Exception as _e:
                print(f"  ℹ  Window icon not loaded: {_e}")
        self.after(200, _set_icon)

        # shared state
        self._lock             = threading.Lock()
        self._frame_pil        = None
        self._fps_val          = 0.0
        self._locked_word      = None
        self._locked_conf      = 0.0
        self._commit_progress  = 0.0
        self._commit_word_cur  = None
        self._hold_progress    = 0.0   # 1.0 = just lifted, counts down to 0
        self._hold_after_commit = False  # True only if hold follows a commit
        self._sentence_building = False
        self._cancel_build     = False
        self._pending_glosses  = []
        self.word_buffer       = []
        self.last_sentence     = ""
        self._running          = True

        self._build_layout()

        threading.Thread(target=self._camera_loop, daemon=True).start()
        self.after(33, self._tick)

        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.bind("<Return>",  lambda _: self._complete_sentence())
        self.bind("<Escape>",  lambda _: self._quit())
        self.bind("<z>",       lambda _: self._undo())
        self.bind("<Z>",       lambda _: self._undo())
        self.bind("<c>",       lambda _: self._clear())
        self.bind("<C>",       lambda _: self._clear())
        self.bind("<s>",       lambda _: self._speak())
        self.bind("<S>",       lambda _: self._speak())

    # ── Layout ──────────────────────────────────────────────────────────────────
    def _build_layout(self):
        """
        3-column grid:

               col 0        col 1       col 2
        row 0: [webcam]    [detection panel (colspan 1+2) ]
        row 1: [content area (colspan 0+1)          ][btns]
        """
        P = 6   # outer padding

        self.grid_columnconfigure(0, weight=4)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Top-left: webcam ──────────────────────────────────────────────────
        vid_frame = ctk.CTkFrame(self, fg_color="#000000", corner_radius=6,
                                  border_width=2, border_color=BORDER)
        vid_frame.grid(row=0, column=0,
                       padx=(P, P//2), pady=(P, P//2), sticky="nsew")
        vid_frame.grid_propagate(False)   # children do NOT resize this frame

        self._video_label = ctk.CTkLabel(vid_frame, text="",
                                          fg_color="transparent")
        self._video_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        # FPS badge — top-right corner of webcam
        self._fps_label = ctk.CTkLabel(vid_frame, text="fps",
                                        font=ctk.CTkFont("Segoe UI", 12, "bold"),
                                        text_color=GREEN,
                                        fg_color="transparent")
        self._fps_label.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)

        self._vid_frame = vid_frame

        # ── Top-right: detection panel (col 1+2) ─────────────────────────────
        det = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=6,
                            border_width=2, border_color=BORDER)
        det.grid(row=0, column=1, columnspan=2,
                  padx=(P//2, P), pady=(P, P//2), sticky="nsew")
        det.grid_columnconfigure(0, weight=1)
        det.grid_rowconfigure(0, weight=1)
        det.grid_rowconfigure(1, weight=0)
        det.grid_rowconfigure(2, weight=0)

        self._det_word_label = ctk.CTkLabel(
            det, text="DETECTED SIGN",
            font=ctk.CTkFont("Segoe UI", 44, "bold"),
            text_color=MUTED)
        self._det_word_label.grid(row=0, column=0, sticky="s", pady=(0, 12))

        self._commit_bar = ctk.CTkProgressBar(
            det, height=14, corner_radius=7,
            progress_color=GREEN, fg_color="#3a3a3a")
        self._commit_bar.set(0.0)
        self._commit_bar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))

        self._commit_hint = ctk.CTkLabel(
            det, text="",
            font=ctk.CTkFont("Segoe UI", 11),
            text_color=MUTED)
        self._commit_hint.grid(row=2, column=0, pady=(0, 6))

        # ── Bottom: content area (col 0+1) ────────────────────────────────────
        content = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=6,
                                border_width=2, border_color=BORDER)
        content.grid(row=1, column=0, columnspan=2,
                      padx=(P, P//2), pady=(P//2, P), sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)   # speaker button col
        content.grid_rowconfigure(0, weight=1)       # word buffer row
        content.grid_rowconfigure(1, weight=0)       # thin divider
        content.grid_rowconfigure(2, weight=1)       # sentence row

        self._buffer_label = ctk.CTkLabel(
            content, text="WORD BUFFER",
            font=ctk.CTkFont("Segoe UI", 22, "bold"),
            text_color=MUTED, anchor="w",
            wraplength=750, justify="left")
        self._buffer_label.grid(row=0, column=0, columnspan=2,
                                 sticky="sw", padx=18, pady=(12, 4))

        # thin divider
        ctk.CTkFrame(content, height=1, fg_color=BORDER).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=14)

        self._sentence_label = ctk.CTkLabel(
            content, text="sentence",
            font=ctk.CTkFont("Segoe UI", 22),
            text_color=MUTED, anchor="w",
            wraplength=670, justify="left")
        self._sentence_label.grid(row=2, column=0,
                                   sticky="nw", padx=18, pady=(8, 12))

        # Large speaker button — bottom-right of content area
        self._speak_btn = ctk.CTkButton(
            content, text="🔊",
            font=ctk.CTkFont("Segoe UI", 34),
            width=70, height=70,
            fg_color="transparent",
            hover_color="#2a2a2a",
            corner_radius=8,
            command=self._speak)
        self._speak_btn.grid(row=2, column=1,
                              sticky="se", padx=12, pady=12)

        # ── Bottom-right: action buttons (col 2) ─────────────────────────────
        btn_panel = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=6,
                                  border_width=2, border_color=BORDER)
        btn_panel.grid(row=1, column=2,
                        padx=(P//2, P), pady=(P//2, P), sticky="nsew")
        btn_panel.grid_columnconfigure(0, weight=1)
        btn_panel.grid_rowconfigure(0, weight=1)
        btn_panel.grid_rowconfigure(1, weight=1)
        btn_panel.grid_rowconfigure(2, weight=1)

        _BTN = dict(font=ctk.CTkFont("Segoe UI", 18, "bold"),
                    corner_radius=4, text_color=WHITE)

        ctk.CTkButton(btn_panel, text="undo",
                      fg_color=BLUE, hover_color="#00AADD",
                      command=self._undo, **_BTN
                      ).grid(row=0, column=0, sticky="nsew",
                             padx=10, pady=(10, 4))

        ctk.CTkButton(btn_panel, text="clear",
                      fg_color=RED, hover_color="#AA0000",
                      command=self._clear, **_BTN
                      ).grid(row=1, column=0, sticky="nsew",
                             padx=10, pady=4)

        ctk.CTkButton(btn_panel, text="build",
                      fg_color=BGREEN, hover_color="#2DA52D",
                      command=self._complete_sentence, **_BTN
                      ).grid(row=2, column=0, sticky="nsew",
                             padx=10, pady=(4, 10))

    # ── Camera / inference thread ───────────────────────────────────────────────
    def _camera_loop(self):
        frame_buffer    = deque(maxlen=SEQUENCE_LEN)
        pred_buffer     = deque(maxlen=PRED_BUFFER_SIZE)
        infer_result    = {"word": None, "conf": 0.0}
        infer_lock      = threading.Lock()
        infer_pending   = threading.Event()

        locked_word      = None
        locked_conf      = 0.0
        hold_frames_left = 0
        commit_word      = None
        commit_frames_c  = 0
        commit_cooldown  = 0
        just_committed   = False   # True only after a word is added to the buffer

        for _ in range(SEQUENCE_LEN):
            frame_buffer.append(ZERO_FRAME.copy())

        fps_buf = deque(maxlen=30)
        last_t  = time.perf_counter()
        cnt     = 0
        flash   = 0

        def _infer():
            while self._running:
                infer_pending.wait(); infer_pending.clear()
                with infer_lock:
                    seq = np.array(frame_buffer, dtype=np.float32)[np.newaxis]
                proba = lstm_model(seq, training=False).numpy()[0]
                idx   = int(np.argmax(proba))
                infer_result["word"] = le.classes_[idx]
                infer_result["conf"] = float(proba[idx])

        threading.Thread(target=_infer, daemon=True).start()

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05); continue

            now = time.perf_counter()
            fps_buf.append(1.0 / max(now - last_t, 1e-6))
            last_t = now
            fps    = float(np.mean(fps_buf))

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self._lock:
                sb = self._sentence_building

            if not sb:
                mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result    = landmarker.detect_for_video(mp_img, int(time.time()*1000))
                h1, h2    = get_left_right(result)
                num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
                if result.hand_landmarks:
                    for lms in result.hand_landmarks:
                        draw_hand(frame, lms)
            else:
                h1 = h2 = np.zeros(63, dtype=np.float32)
                num_hands = 0

            if flash > 0:
                ov = frame.copy()
                cv2.rectangle(ov, (0,0), (frame.shape[1],frame.shape[0]),(0,255,120),-1)
                cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
                flash -= 1

            raw  = np.concatenate([h1, h2])
            norm = normalize_frame(raw)
            with infer_lock:
                frame_buffer.append(norm)

            cnt += 1
            if num_hands > 0 and cnt % INFER_EVERY == 0:
                infer_pending.set()

            if infer_result["word"] is not None and num_hands > 0:
                pred_buffer.append((infer_result["word"], infer_result["conf"]))
                infer_result["word"] = None

            smooth_word, smooth_conf = recency_weighted(pred_buffer)

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
                        locked_word = None; locked_conf = 0.0
                        just_committed = False   # hold expired, clear flag

            with self._lock:
                _wb = list(self.word_buffer)

            if num_hands > 0 and locked_word:
                if commit_cooldown > 0:
                    commit_cooldown -= 1
                else:
                    if locked_word == commit_word:
                        commit_frames_c += 1
                    else:
                        commit_word = locked_word; commit_frames_c = 1
                    if commit_frames_c >= COMMIT_FRAMES:
                        if not _wb or _wb[-1] != commit_word:
                            _wb.append(commit_word)
                            print(f"  + Committed: {commit_word}  [{' → '.join(_wb)}]")
                            flash = 5
                            just_committed = True   # flag: hold bar now valid
                        else:
                            print(f"  ~ Skipped duplicate: {commit_word}")
                        commit_cooldown = COMMIT_COOLDOWN; commit_frames_c = 0
            elif num_hands == 0 and hold_frames_left == 0:
                commit_frames_c = 0; commit_word = None

            with self._lock:
                self.word_buffer      = _wb
                self._locked_word     = locked_word
                self._locked_conf     = locked_conf
                self._commit_word_cur = commit_word if commit_cooldown == 0 else None
                self._commit_progress = (
                    min(commit_frames_c / COMMIT_FRAMES, 1.0)
                    if commit_cooldown == 0 else 0.0)
                self._hold_progress      = (
                    hold_frames_left / HOLD_FRAMES if hold_frames_left > 0 else 0.0)
                self._hold_after_commit  = just_committed
                self._frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self._fps_val   = fps

        cap.release(); landmarker.close()

    # ── UI refresh ──────────────────────────────────────────────────────────────
    def _tick(self):
        if not self._running:
            return

        with self._lock:
            pil_img          = self._frame_pil
            locked_word      = self._locked_word
            locked_conf      = self._locked_conf
            commit_word      = self._commit_word_cur
            commit_progress  = self._commit_progress
            hold_progress    = self._hold_progress
            hold_after_commit = self._hold_after_commit
            word_buffer      = list(self.word_buffer)
            last_sentence    = self.last_sentence
            sb               = self._sentence_building
            fps              = self._fps_val

        # ── Webcam ──────────────────────────────────────────────────────────────
        if pil_img is not None:
            vw = self._vid_frame.winfo_width()
            vh = self._vid_frame.winfo_height()
            if vw > 4 and vh > 4:
                # fit 640×480 inside container preserving 4:3
                aspect = 640 / 480
                fw = vw; fh = int(fw / aspect)
                if fh > vh:
                    fh = vh; fw = int(fh * aspect)
                img = pil_img.resize((fw, fh), Image.BILINEAR)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                       size=(fw, fh))
                self._video_label.configure(image=ctk_img, text="")
                self._video_label._image = ctk_img

        # ── FPS ─────────────────────────────────────────────────────────────────
        self._fps_label.configure(text=f"{fps:.0f} fps")

        # ── Detected sign ────────────────────────────────────────────────────────
        if locked_word:
            self._det_word_label.configure(text=locked_word.upper(),
                                            text_color=WHITE)
        else:
            self._det_word_label.configure(text="DETECTED SIGN",
                                            text_color=MUTED)

        # ── Commit / Hold bar ──────────────────────────────────────────────────
        if commit_progress > 0 or commit_word:
            # Hand is present and word is being committed (GREEN bar)
            self._commit_bar.configure(progress_color=GREEN)
            self._commit_bar.set(commit_progress)
            if commit_word and commit_progress < 1.0:
                secs = (1.0 - commit_progress) * (COMMIT_FRAMES / 30)
                self._commit_hint.configure(
                    text=f"COMMITTING: {commit_word.upper()}  ({secs:.1f}s)",
                    text_color=WHITE)
            else:
                self._commit_hint.configure(text="")
        elif hold_progress > 0 and hold_after_commit:
            # Hand lifted AFTER a commit — orange bar drains to zero
            self._commit_bar.configure(progress_color="#FF8C00")
            self._commit_bar.set(hold_progress)
            self._commit_hint.configure(
                text=f"HOLDING: {locked_word.upper() if locked_word else ''}...",
                text_color="#FF8C00")
        else:
            # Idle
            self._commit_bar.configure(progress_color=GREEN)
            self._commit_bar.set(0.0)
            self._commit_hint.configure(text="")

        # ── Word buffer ─────────────────────────────────────────────────────────
        if word_buffer:
            self._buffer_label.configure(
                text="  +  ".join(w.upper() for w in word_buffer),
                text_color=BUFCOL)
        else:
            self._buffer_label.configure(
                text="WORD BUFFER", text_color=MUTED)

        # ── Sentence ────────────────────────────────────────────────────────────
        if sb:
            dots = "." * ((int(time.time() * 2)) % 4)
            self._sentence_label.configure(
                text=f"BUILDING{dots}", text_color="#f59e0b")
        elif last_sentence:
            self._sentence_label.configure(
                text=last_sentence.upper(), text_color=SENTCOL)
        else:
            self._sentence_label.configure(text="SENTENCE", text_color=MUTED)

        self.after(33, self._tick)

    # ── Actions ─────────────────────────────────────────────────────────────────
    def _run_sentence_builder(self, words):
        result = sentence_builder.build(list(words))
        with self._lock:
            cancelled = self._cancel_build
        if cancelled:
            # A second Enter was pressed — the fallback was already applied
            return
        with self._lock:
            self.last_sentence      = result
            self._sentence_building = False
        print(f'\n  ✓ Sentence: "{result}"')

    def _complete_sentence(self):
        with self._lock:
            already_building = self._sentence_building
            wb               = list(self.word_buffer)
            glosses          = list(self._pending_glosses)

        # ── Second Enter while Gemini is still thinking: bail out immediately ──
        if already_building:
            fallback = " ".join(glosses) if glosses else ""
            if fallback:
                print(f"  ⚡ Build cancelled — using raw glosses: {glosses}")
                with self._lock:
                    self._cancel_build      = True
                    self._sentence_building = False
                    self.last_sentence      = fallback
            return

        if not wb:
            return
        print(f"  Signing complete → words: {wb}")
        with self._lock:
            self.word_buffer        = []
            self._sentence_building = True
            self._cancel_build      = False
            self._pending_glosses   = list(wb)
        if sentence_builder.ready:
            threading.Thread(target=self._run_sentence_builder,
                             args=(wb,), daemon=True).start()
        else:
            with self._lock:
                self.last_sentence      = " ".join(wb)
                self._sentence_building = False
                self._pending_glosses   = []

    def _speak(self):
        with self._lock:
            s = self.last_sentence; sb = self._sentence_building
        if s and not sb:
            print("  🔊 Speaking...")
            threading.Thread(target=_play_audio, args=(s,), daemon=True).start()

    def _undo(self):
        with self._lock:
            if self.word_buffer:
                print(f"  Undone: {self.word_buffer.pop()}")

    def _clear(self):
        with self._lock:
            self.word_buffer   = []
            self.last_sentence = ""
        print("  Cleared.")

    def _quit(self):
        self._running = False
        self.after(300, self.destroy)
        print("\n✓ Session ended.")


if __name__ == "__main__":
    app = BSLApp()
    app.mainloop()
