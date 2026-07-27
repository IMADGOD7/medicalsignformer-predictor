from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 100
FEATURE_DIM = 1629


class SlidingWindowError(RuntimeError):
    """Raised for invalid usage (e.g. wrong feature vector shape)."""


class SlidingWindow:

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        feature_dim: int = FEATURE_DIM,
        pad_during_warmup: bool = False,
    ) -> None:
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.pad_during_warmup = pad_during_warmup

        self._buffer: deque[np.ndarray] = deque(maxlen=sequence_length)

    def add_frame(self, features: np.ndarray) -> None:
        if features.shape != (self.feature_dim,):
            raise SlidingWindowError(
                f"Expected a ({self.feature_dim},) feature vector, got shape {features.shape}."
            )
        self._buffer.append(features)

    def is_ready(self) -> bool:
        return len(self._buffer) == self.sequence_length

    def get_window(self) -> Optional[np.ndarray]:
        if self.is_ready():
            return np.stack(list(self._buffer), axis=0)

        if not self.pad_during_warmup:
            return None
        num_real_frames = len(self._buffer)
        num_pad_frames = self.sequence_length - num_real_frames
        padding = np.zeros((num_pad_frames, self.feature_dim), dtype=np.float32)
        real_frames = np.stack(list(self._buffer), axis=0).astype(np.float32)
        return np.concatenate([padding, real_frames], axis=0)

    def reset(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


if __name__ == "__main__":
    # --- Standalone verification script ---
    print("--- Test 1: not ready before sequence_length frames added ---")
    window = SlidingWindow()
    for i in range(50):
        window.add_frame(np.full((1629,), fill_value=float(i), dtype=np.float32))
    print(f"len(window)={len(window)}, is_ready()={window.is_ready()}")
    assert not window.is_ready()
    assert window.get_window() is None
    print("PASSED - not ready with only 50/100 frames, get_window() returns None.")

    print("\n--- Test 2: ready exactly at sequence_length frames ---")
    for i in range(50, 100):
        window.add_frame(np.full((1629,), fill_value=float(i), dtype=np.float32))
    print(f"len(window)={len(window)}, is_ready()={window.is_ready()}")
    assert window.is_ready()
    batch = window.get_window()
    print(f"get_window() shape: {batch.shape}, dtype: {batch.dtype}")
    assert batch.shape == (100, 1629)
    print("PASSED")

    print("\n--- Test 3: correct temporal order (oldest first, newest last) ---")
    # Every value in frame i was filled with float(i), so batch[t, 0] should equal t exactly.
    expected_order = np.arange(100, dtype=np.float32)
    actual_order = batch[:, 0]
    print(f"First 5 values: {actual_order[:5]}, Last 5 values: {actual_order[-5:]}")
    assert np.array_equal(actual_order, expected_order), "Frames are not in the correct temporal order!"
    print("PASSED - frame 0 is oldest (index 0), frame 99 is newest (index 99).")

    print("\n--- Test 4: sliding behavior - oldest frame evicted once buffer exceeds sequence_length ---")
    window.add_frame(np.full((1629,), fill_value=100.0, dtype=np.float32))  # 101st frame added
    batch_after_slide = window.get_window()
    print(f"len(window) after 101st frame: {len(window)} (should still be 100, oldest evicted)")
    assert len(window) == 100
    print(f"First value now: {batch_after_slide[0, 0]} (should be 1.0, since frame 0 was evicted)")
    print(f"Last value now: {batch_after_slide[-1, 0]} (should be 100.0, the newest frame)")
    assert batch_after_slide[0, 0] == 1.0
    assert batch_after_slide[-1, 0] == 100.0
    print("PASSED - sliding window correctly evicts the oldest frame and keeps the latest 100.")

    print("\n--- Test 5: wrong feature shape raises SlidingWindowError ---")
    window2 = SlidingWindow()
    try:
        window2.add_frame(np.zeros((100,)))  # wrong shape
        print("UNEXPECTED: did not raise.")
    except SlidingWindowError as e:
        print(f"Correctly raised SlidingWindowError: {e}")

    print("\n--- Test 6: pad_during_warmup=True produces an early (padded) window ---")
    window3 = SlidingWindow(pad_during_warmup=True)
    for i in range(30):
        window3.add_frame(np.full((1629,), fill_value=float(i + 1), dtype=np.float32))  # avoid 0.0 to distinguish from padding
    assert not window3.is_ready()  # still true - is_ready() is independent of pad_during_warmup
    padded_batch = window3.get_window()
    print(f"get_window() with pad_during_warmup=True and only 30 real frames: shape={padded_batch.shape}")
    assert padded_batch is not None and padded_batch.shape == (100, 1629)
    num_zero_frames = np.sum(np.all(padded_batch == 0.0, axis=1))
    print(f"Number of all-zero (padded) frames: {num_zero_frames} (expected 70)")
    assert num_zero_frames == 70
    print(f"Leading frames are padding, trailing frames are real: "
          f"first value={padded_batch[0, 0]}, last value={padded_batch[-1, 0]} (expected 30.0)")
    assert padded_batch[-1, 0] == 30.0
    print("PASSED - padding correctly placed at the start (leading), real frames trailing/most-recent.")

    print("\n--- Test 7: reset() clears the buffer ---")
    window.reset()
    print(f"len(window) after reset(): {len(window)}")
    assert len(window) == 0
    assert not window.is_ready()
    print("PASSED")

    print("\nAll checks PASSED.")