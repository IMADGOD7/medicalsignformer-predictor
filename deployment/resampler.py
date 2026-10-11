from __future__ import annotations

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SEQUENCE_LENGTH = 100
FEATURE_DIM = 1629


class ResamplerError(ValueError):
    """Raised when resampling receives invalid inputs."""


def resample_gesture(
    gesture: np.ndarray,
    target_length: int = TARGET_SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Resamples a gesture matrix of shape (N, feature_dim) to exactly (target_length, feature_dim)
    using 1D linear interpolation along the temporal axis.

    Args:
        gesture: Input array of shape (N, feature_dim).
        target_length: Desired number of output frames (default: 100).

    Returns:
        Resampled numpy array of shape (target_length, feature_dim) with dtype float32.
    """
    if gesture.ndim != 2:
        raise ResamplerError(
            f"Expected 2D gesture matrix (N, feature_dim), got shape {gesture.shape}"
        )

    num_frames, feature_dim = gesture.shape

    if num_frames == 0:
        raise ResamplerError("Cannot resample an empty gesture (0 frames).")

    if target_length <= 0:
        raise ResamplerError(f"Target length must be > 0, got {target_length}.")

    if num_frames == target_length:
        return gesture.astype(np.float32, copy=True)

    if num_frames == 1:
        return np.repeat(gesture, target_length, axis=0).astype(np.float32)

    # 1D linear interpolation along time axis for each feature column
    x_orig = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    x_target = np.linspace(0.0, 1.0, target_length, dtype=np.float32)

    resampled = np.zeros((target_length, feature_dim), dtype=np.float32)

    try:
        from scipy.interpolate import interp1d
        f = interp1d(x_orig, gesture, axis=0, kind="linear", assume_sorted=True)
        return f(x_target).astype(np.float32)
    except ImportError:
        # Fallback to pure numpy interpolation
        for col in range(feature_dim):
            resampled[:, col] = np.interp(x_target, x_orig, gesture[:, col])
        return resampled


class TemporalResampler:
    """
    Stateful or configurable wrapper for temporal resampling operations.
    """

    def __init__(
        self,
        target_length: int = TARGET_SEQUENCE_LENGTH,
        feature_dim: int = FEATURE_DIM,
    ) -> None:
        self.target_length = target_length
        self.feature_dim = feature_dim

    def process(self, gesture: np.ndarray) -> np.ndarray:
        if gesture.ndim == 2 and gesture.shape[1] != self.feature_dim:
            raise ResamplerError(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {gesture.shape[1]}"
            )
        return resample_gesture(gesture, target_length=self.target_length)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- Testing resampler.py ---")

    # Test 1: 38 frames -> 100 frames
    g38 = np.random.randn(38, 1629).astype(np.float32)
    out100 = resample_gesture(g38, 100)
    assert out100.shape == (100, 1629)
    print("Test 1 (38 -> 100 frames): PASSED")

    # Test 2: 145 frames -> 100 frames
    g145 = np.random.randn(145, 1629).astype(np.float32)
    out100_2 = resample_gesture(g145, 100)
    assert out100_2.shape == (100, 1629)
    print("Test 2 (145 -> 100 frames): PASSED")

    # Test 3: 100 frames -> 100 frames pass-through
    g100 = np.random.randn(100, 1629).astype(np.float32)
    out100_3 = resample_gesture(g100, 100)
    assert np.array_equal(g100, out100_3)
    print("Test 3 (100 -> 100 frames pass-through): PASSED")

    # Test 4: 1 frame -> 100 frames
    g1 = np.ones((1, 1629), dtype=np.float32) * 5.0
    out1 = resample_gesture(g1, 100)
    assert out1.shape == (100, 1629)
    assert np.allclose(out1, 5.0)
    print("Test 4 (1 -> 100 frames tile): PASSED")

    print("All resampler tests PASSED.")
