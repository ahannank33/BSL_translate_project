"""
Two-stage BSL sentence formation:

  Stage 1 — GPT-2 reordering:
    Score all word orderings by perplexity, return the most probable arrangement.
    BSL:  [what, name, your]  →  "what your name"

  Stage 2 — T5 Grammar Error Correction (GEC):
    Feed the reordered phrase through a T5 model fine-tuned for grammar correction.
    It adds missing function words (is, are, the, a, …) and capitalises.
    "what your name"  →  "What is your name?"

Both models load in a shared background thread so startup is non-blocking.
Falls back gracefully if transformers / torch are not installed.

Install dependencies:
    pip install transformers torch sentencepiece
"""

import threading
from itertools import permutations

# ── Dependency check ───────────────────────────────────────────────────────────
USE_MODELS = False
try:
    import torch
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer,
        T5ForConditionalGeneration, AutoTokenizer,
    )
    USE_MODELS = True
except ImportError:
    print("  ⚠  transformers/torch not installed — sentence builder in fallback mode.")
    print("     Run: pip install transformers torch sentencepiece")

# T5 model fine-tuned for grammar error correction (GEC).
# Handles: missing auxiliary verbs, missing articles, capitalisation, etc.
# "gec: what your name"  →  "What is your name?"
T5_MODEL_NAME = "prithivida/grammar_error_correcter_v1"


class SentenceBuilder:
    """
    Async two-stage sentence builder.

    Usage:
        sb = SentenceBuilder()           # starts loading models in background
        ...                              # do other setup while models load
        if sb.ready:
            print(sb.build(['what', 'name', 'your']))
            # → "What is your name?"
    """

    def __init__(self):
        # GPT-2 state
        self._gpt2        = None
        self._gpt2_tok    = None
        self._gpt2_ready  = False

        # T5 GEC state
        self._t5          = None
        self._t5_tok      = None
        self._t5_ready    = False

        self._loading     = False

        if USE_MODELS:
            self._loading = True
            t = threading.Thread(target=self._load_all, daemon=True)
            t.start()
        else:
            # Fallback mode — both stages treated as "ready" (no-ops)
            self._gpt2_ready = True
            self._t5_ready   = True

    # ── Background loading ─────────────────────────────────────────────────────
    def _load_all(self):
        """Load GPT-2 then T5 sequentially in the background thread."""
        # Stage 1: GPT-2
        try:
            print("  [1/2] Loading GPT-2 for word reordering...")
            self._gpt2_tok = GPT2Tokenizer.from_pretrained('gpt2')
            self._gpt2     = GPT2LMHeadModel.from_pretrained('gpt2')
            self._gpt2.eval()
            self._gpt2_ready = True
            print("  ✓ GPT-2 ready")
        except Exception as e:
            print(f"  ✗ GPT-2 load failed: {e}")
            self._gpt2_ready = True   # mark ready so we don't block forever

        # Stage 2: T5 GEC
        try:
            print(f"  [2/2] Loading T5 grammar correction ({T5_MODEL_NAME})...")
            self._t5_tok = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
            self._t5     = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)
            self._t5.eval()
            self._t5_ready = True
            print("  ✓ T5 grammar correction ready")
        except Exception as e:
            print(f"  ✗ T5 load failed: {e}")
            self._t5_ready = True   # fall back to GPT-2 reordering only

        self._loading = False
        print("  ✓ Sentence builder fully ready\n")

    # ── Status ─────────────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        """True once both models have loaded (or fallen back)."""
        return self._gpt2_ready and self._t5_ready

    @property
    def loading(self) -> bool:
        return self._loading

    # ── Stage 1: GPT-2 perplexity scoring ─────────────────────────────────────
    def _score(self, text: str) -> float:
        """Log-probability for `text` under GPT-2. Higher = more natural."""
        if not self._gpt2:
            return 0.0
        tokens = self._gpt2_tok.encode(text, return_tensors='pt')
        with torch.no_grad():
            loss = self._gpt2(tokens, labels=tokens).loss
        return -loss.item()

    def _best_permutation(self, words: list) -> str:
        """Try all N! orderings; return the highest-scoring one (N ≤ 4)."""
        best, best_score = None, float('-inf')
        for perm in permutations(words):
            text  = ' '.join(perm)
            score = self._score(text)
            if score > best_score:
                best_score = score
                best       = text
        return best

    def _greedy_order(self, words: list) -> str:
        """Greedy left-to-right ordering — O(N²) GPT-2 calls (N ≤ 6)."""
        remaining = list(words)
        chosen    = []
        while remaining:
            best_word, best_score = None, float('-inf')
            for w in remaining:
                score = self._score(' '.join(chosen + [w]))
                if score > best_score:
                    best_score = score
                    best_word  = w
            chosen.append(best_word)
            remaining.remove(best_word)
        return ' '.join(chosen)

    # ── Stage 2: T5 grammar error correction ──────────────────────────────────
    def _correct_grammar(self, text: str) -> str:
        """
        Run T5 GEC to add missing function words and fix capitalisation.
        "what your name"  →  "What is your name?"
        "hello my name"   →  "Hello, my name."
        Falls back to the input if T5 is unavailable.
        """
        if not self._t5:
            return text

        input_text = f"gec: {text}"
        tokens     = self._t5_tok.encode(
            input_text, return_tensors='pt', max_length=128, truncation=True
        )
        with torch.no_grad():
            output = self._t5.generate(
                tokens,
                max_length=128,
                num_beams=5,
                early_stopping=True,
            )
        return self._t5_tok.decode(output[0], skip_special_tokens=True)

    # ── Public API ─────────────────────────────────────────────────────────────
    def build(self, words: list) -> str:
        """
        Convert a list of BSL-ordered signed words into a grammatical English sentence.

        Steps:
          1. Reorder with GPT-2 (perplexity scoring)
          2. Correct grammar with T5 GEC (adds missing words)

        Falls back to plain concatenation if neither model is available.
        """
        if not words:
            return ""
        if len(words) == 1:
            # Single word — still run GEC to capitalise / complete
            word = words[0]
            return self._correct_grammar(word) if self._t5 else word

        # ── Stage 1: find best word ordering ──────────────────────────────────
        if self._gpt2:
            n = len(words)
            if n <= 4:
                ordered = self._best_permutation(words)
            elif n <= 6:
                ordered = self._greedy_order(words)
            else:
                ordered = ' '.join(words)   # too many to score — keep sign order
        else:
            ordered = ' '.join(words)

        print(f"  [Stage 1] GPT-2 ordered: \"{ordered}\"")

        # ── Stage 2: grammar correction ────────────────────────────────────────
        corrected = self._correct_grammar(ordered)
        print(f"  [Stage 2] T5 corrected:  \"{corrected}\"")

        return corrected
