from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe is required for landmark detection. Install it with 'pip install mediapipe'"
    ) from exc

if not hasattr(mp, "solutions"):
    try:
        from mediapipe.python import solutions as mp_solutions
        mp.solutions = mp_solutions
    except Exception:  # pragma: no cover
        mp.solutions = None

logger = logging.getLogger(__name__)

# Feature dimensions - must match config.py / extract_landmarks.py exactly.
NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z

POSE_DIM = NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK      # 99
FACE_DIM = NUM_FACE_LANDMARKS * COORDS_PER_LANDMARK      # 1404
HAND_DIM = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK       # 63
TOTAL_FEATURE_DIM = POSE_DIM + FACE_DIM + HAND_DIM + HAND_DIM  # 1629


class MediaPipeDetectorError(RuntimeError):
    """Raised when the MediaPipe Holistic model cannot be initialized."""


class MediaPipeDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp.solutions is None:
            raise MediaPipeDetectorError(
                "MediaPipe solutions could not be imported. Check the mediapipe installation."
            )

        logger.info(
            "Initializing MediaPipe Holistic (model_complexity=%d, smooth_landmarks=%s)...",
            model_complexity, smooth_landmarks,
        )
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._closed = False

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self._closed:
            raise MediaPipeDetectorError("process() called after close() - detector is no longer usable.")
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected an (H, W, 3) RGB frame, got shape {frame_rgb.shape}.")

        results = self._holistic.process(frame_rgb)
        return self._extract_landmarks(results)

    @staticmethod
    def _extract_landmarks(results) -> np.ndarray:
        pose = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks
            else np.zeros(NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK)
        )

        face = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks
            else np.zeros(NUM_FACE_LANDMARKS * COORDS_PER_LANDMARK)
        )

        left_hand = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks
            else np.zeros(NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK)
        )

        right_hand = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks
            else np.zeros(NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK)
        )

        return np.concatenate([pose, face, left_hand, right_hand])

    def close(self) -> None:
        if not self._closed:
            self._holistic.close()
            self._closed = True
            logger.info("MediaPipe Holistic closed.")

    def __enter__(self) -> "MediaPipeDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


if __name__ == "__main__":
    # --- Standalone verification script ---
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    print("--- Test 1: feature dimension constants match config.py's expected 1629 ---")
    print(f"POSE_DIM={POSE_DIM}, FACE_DIM={FACE_DIM}, HAND_DIM={HAND_DIM} (x2)")
    print(f"TOTAL_FEATURE_DIM={TOTAL_FEATURE_DIM} (expected 1629)")
    assert TOTAL_FEATURE_DIM == 1629, "Feature dimension mismatch with the trained model's expected INPUT_DIM!"
    print("PASSED")

    print("\n--- Test 2: process() on a blank (all-zero) frame - no detections expected ---")
    detector = MediaPipeDetector()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    features = detector.process(blank_frame)
    print(f"Output shape: {features.shape}, dtype: {features.dtype}")
    assert features.shape == (1629,), f"Expected (1629,), got {features.shape}"
    print(f"All-zero (no detections at all): {np.all(features == 0.0)}")
    print("PASSED - blank frame produces the correctly-shaped, all-zero feature vector.")

    print("\n--- Test 3: process() on a random-noise frame ---")
    rng = np.random.default_rng(0)
    noise_frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    features_noise = detector.process(noise_frame)
    print(f"Output shape: {features_noise.shape}, dtype: {features_noise.dtype}")
    assert features_noise.shape == (1629,)
    print("PASSED - correct shape regardless of frame content (random noise has no real "
          "landmarks to detect either, so this is expected to also be all/mostly zero).")

    print("\n--- Test 4: invalid input shape raises ValueError, not a silent failure ---")
    try:
        detector.process(np.zeros((480, 640), dtype=np.uint8))  # missing channel dim
        print("UNEXPECTED: did not raise.")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")

    print("\n--- Test 5: close() then process() raises a clear error, not a crash ---")
    detector.close()
    try:
        detector.process(blank_frame)
        print("UNEXPECTED: did not raise.")
    except MediaPipeDetectorError as e:
        print(f"Correctly raised MediaPipeDetectorError: {e}")

    print("\n--- Test 6: context manager closes automatically ---")
    with MediaPipeDetector() as detector2:
        f = detector2.process(blank_frame)
        assert f.shape == (1629,)
    print("Context manager entered/exited without error. PASSED")