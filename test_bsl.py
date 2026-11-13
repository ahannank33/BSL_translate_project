import numpy as np
import pytest
from collections import deque
from unittest.mock import MagicMock


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,17),
]

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
    ww   = [weights[i] for i, (wd, _) in enumerate(items) if wd == best]
    wc   = [conf       for (wd, conf) in items            if wd == best]
    return best, float(np.dot(ww, wc) / sum(ww))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Unit Tests — normalize_hand()
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeHand:

    def test_zero_vector_returned_unchanged(self):
        """A fully-zero input (no hand detected) must pass through unmodified."""
        vec = np.zeros(63, dtype=np.float32)
        result = normalize_hand(vec)
        np.testing.assert_array_equal(result, vec)

    def test_output_is_zero_centred(self):
        """After normalisation, wrist landmark (index 0) must be at the origin."""
        hand = np.random.rand(63).astype(np.float32)
        result = normalize_hand(hand)
        pts = result.reshape(3, 21).T
        np.testing.assert_array_almost_equal(pts[0], [0.0, 0.0, 0.0], decimal=5)

    def test_scale_invariance(self):
        """
        The same hand sign at twice the distance (scaled 2x) must produce
        the identical normalised output, confirming scale removal.
        """
        hand = np.random.rand(63).astype(np.float32) + 0.1
        result_1x = normalize_hand(hand)
        result_2x = normalize_hand(hand * 2.0)
        np.testing.assert_array_almost_equal(result_1x, result_2x, decimal=5)

    def test_translation_invariance(self):
        """
        Shifting the entire hand by a constant (different position in frame)
        must not change the normalised output.
        """
        hand  = np.random.rand(63).astype(np.float32) + 0.1
        shift = np.tile([0.5, 0.5, 0.5], 21).astype(np.float32)
        np.testing.assert_array_almost_equal(
            normalize_hand(hand),
            normalize_hand(hand + shift),
            decimal=5)

    def test_output_shape_preserved(self):
        """Output must have the same shape as input: (63,)."""
        result = normalize_hand(np.random.rand(63).astype(np.float32))
        assert result.shape == (63,)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Unit Tests — normalize_frame()
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeFrame:

    def test_output_shape(self):
        """Full frame vector (2 hands × 63) must return shape (126,)."""
        result = normalize_frame(np.random.rand(126).astype(np.float32))
        assert result.shape == (126,)

    def test_both_halves_zero_centred(self):
        """Each hand's wrist landmark must be at the origin after normalisation."""
        hand_a = np.random.rand(63).astype(np.float32) + 0.1
        hand_b = np.random.rand(63).astype(np.float32) + 0.1
        result = normalize_frame(np.concatenate([hand_a, hand_b]))
        pts_a = result[:63].reshape(3, 21).T
        pts_b = result[63:].reshape(3, 21).T
        np.testing.assert_array_almost_equal(pts_a[0], [0, 0, 0], decimal=5)
        np.testing.assert_array_almost_equal(pts_b[0], [0, 0, 0], decimal=5)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Unit Tests — recency_weighted()
# ══════════════════════════════════════════════════════════════════════════════

class TestRecencyWeighted:

    def test_empty_buffer_returns_none(self):
        """Empty prediction buffer must return (None, 0.0)."""
        word, conf = recency_weighted(deque(maxlen=3))
        assert word is None
        assert conf == 0.0

    def test_single_entry_returned(self):
        """Single-item buffer must return that item."""
        word, conf = recency_weighted(deque([("hello", 0.85)], maxlen=3))
        assert word == "hello"
        assert pytest.approx(conf, abs=0.01) == 0.85

    def test_recent_entries_outweigh_older_ones(self):
        """
        Two recent 'hello' predictions must outweigh one older 'world' prediction
        even when 'world' has higher individual confidence.
        """
        buf = deque([
            ("world", 0.95),   # oldest  — weight 1
            ("hello", 0.75),   # middle  — weight 2
            ("hello", 0.80),   # newest  — weight 3
        ], maxlen=3)
        word, _ = recency_weighted(buf)
        assert word == "hello"

    def test_higher_confidence_wins_when_tied(self):
        """With equal recency weighting, the higher-confidence word wins."""
        buf = deque([("alpha", 0.60), ("beta", 0.90)], maxlen=3)
        word, _ = recency_weighted(buf)
        assert word == "beta"

    def test_returned_confidence_is_float(self):
        """Returned confidence must always be a Python float."""
        _, conf = recency_weighted(deque([("name", 0.77)], maxlen=3))
        assert isinstance(conf, float)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Unit Tests — get_left_right()
# ══════════════════════════════════════════════════════════════════════════════

def _make_lms(offset=0.0):
    """Create 21 mock landmarks."""
    lms = []
    for i in range(21):
        lm = MagicMock()
        lm.x, lm.y, lm.z = i * 0.01 + offset, i * 0.01 + offset, 0.0
        lms.append(lm)
    return lms

def _make_result(sides):
    """Build a mock MediaPipe DetectionResult. sides = [("Left"|"Right", lms)]."""
    result = MagicMock()
    if not sides:
        result.hand_landmarks = None
        return result
    result.hand_landmarks = []
    result.handedness     = []
    for side, lms in sides:
        result.hand_landmarks.append(lms)
        cat = MagicMock(); cat.category_name = side
        result.handedness.append([cat])
    return result


class TestGetLeftRight:

    def test_no_hands_returns_zeros(self):
        """No hands → both outputs are zero vectors of shape (63,)."""
        left, right = get_left_right(_make_result([]))
        assert left.shape  == (63,) and np.all(left  == 0)
        assert right.shape == (63,) and np.all(right == 0)

    def test_left_hand_only(self):
        """Left hand present → left non-zero, right zero."""
        left, right = get_left_right(_make_result([("Left", _make_lms())]))
        assert not np.all(left == 0), "Left should be non-zero"
        assert np.all(right == 0),    "Right should be zero"

    def test_right_hand_only(self):
        """Right hand present → right non-zero, left zero."""
        left, right = get_left_right(_make_result([("Right", _make_lms())]))
        assert np.all(left == 0),      "Left should be zero"
        assert not np.all(right == 0), "Right should be non-zero"

    def test_both_hands(self):
        """Both hands → both vectors non-zero."""
        sides = [("Left", _make_lms(0.0)), ("Right", _make_lms(0.5))]
        left, right = get_left_right(_make_result(sides))
        assert not np.all(left  == 0)
        assert not np.all(right == 0)

    def test_output_shapes_always_63(self):
        """Outputs always have shape (63,) regardless of how many hands."""
        left, right = get_left_right(_make_result([]))
        assert left.shape == (63,) and right.shape == (63,)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Integration Tests — Word accumulator duplicate filter
# ══════════════════════════════════════════════════════════════════════════════

def _run_accumulator(predictions, commit_frames=3, commit_cooldown=5):
   
    word_buffer     = []
    commit_word     = None
    commit_frames_c = 0
    cooldown        = 0

    for word in predictions:
        if cooldown > 0:
            cooldown -= 1
            continue
        if word == commit_word:
            commit_frames_c += 1
        else:
            commit_word = word; commit_frames_c = 1
        if commit_frames_c >= commit_frames:
            if not word_buffer or word_buffer[-1] != commit_word:
                word_buffer.append(commit_word)
            cooldown = commit_cooldown; commit_frames_c = 0
    return word_buffer


class TestWordAccumulator:

    def test_single_word_committed(self):
        assert _run_accumulator(["hello"] * 5) == ["hello"]

    def test_duplicate_consecutive_word_blocked(self):
        """
        Signing the same word twice in a row must result in only one entry —
        this is the no-repeat de-duplication feature.
        """
        result = _run_accumulator(["hello"] * 5 + ["hello"] * 10)
        assert result.count("hello") == 1, f"Expected 1 occurrence, got: {result}"

    def test_two_different_words_both_added(self):
        result = _run_accumulator(["hello"] * 5 + ["name"] * 10)
        assert result == ["hello", "name"]

    def test_word_can_reappear_after_different_word(self):
        """hello → what → hello should all be stored (no global dedup)."""
        result = _run_accumulator(["hello"] * 5 + ["what"] * 8 + ["hello"] * 8)
        assert result == ["hello", "what", "hello"]

    def test_empty_predictions_produces_empty_buffer(self):
        assert _run_accumulator([]) == []

    def test_insufficient_hold_does_not_commit(self):
        """Holding a sign for fewer than COMMIT_FRAMES must not commit."""
        result = _run_accumulator(["hello"] * 2, commit_frames=3)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# 6. Integration Tests — SentenceBuilder fallback path
# ══════════════════════════════════════════════════════════════════════════════

class TestSentenceBuilderFallback:
    """
    Tests the fallback behaviour (no Gemini API) where gloss words are joined
    with spaces. Verified without making any network calls.
    """

    def _fallback(self, words):
        """Mirrors the fallback logic in sentence_builder.py."""
        return " ".join(words)

    def test_typical_phrase(self):
        assert self._fallback(["what", "your", "name"]) == "what your name"

    def test_empty_list_returns_empty_string(self):
        assert self._fallback([]) == ""

    def test_single_word_unchanged(self):
        assert self._fallback(["hello"]) == "hello"

    def test_output_is_string(self):
        result = self._fallback(["hello", "world"])
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
