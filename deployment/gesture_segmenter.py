from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

from motion_utils import MotionDetector, is_empty_frame

logger = logging.getLogger(__name__)

FEATURE_DIM = 1629


class GestureState(Enum):
    """States for the gesture segmenter finite state machine."""
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()
    COOLDOWN = auto()


class GestureSegmenterError(ValueError):
    """Raised for invalid features or configuration."""


@dataclass
class GestureConfig:
    """Configuration parameters for motion detection and gesture segmentation."""
    motion_threshold: float = 0.003
    motion_smoothing_alpha: float = 0.25
    min_gesture_frames: int = 10
    max_gesture_frames: int = 120
    end_timeout_frames: int = 6
    cooldown_frames: int = 10
    target_sequence_length: int = 100


class GestureSegmenter:
    """
    Finite State Machine (FSM) that detects gesture start/end boundaries
    from continuous frame landmark streams and yields isolated, complete gestures.
    """

    def __init__(self, config: Optional[GestureConfig] = None) -> None:
        self.config = config or GestureConfig()
        self.detector = MotionDetector(
            threshold=self.config.motion_threshold,
            alpha=self.config.motion_smoothing_alpha,
        )

        self._state: GestureState = GestureState.IDLE
        self._buffer: list[np.ndarray] = []
        self._low_motion_counter: int = 0
        self._cooldown_counter: int = 0
        self._last_motion_score: float = 0.0

    @property
    def state(self) -> GestureState:
        return self._state

    @property
    def buffer_length(self) -> int:
        return len(self._buffer)

    @property
    def last_motion_score(self) -> float:
        return self._last_motion_score

    def reset(self) -> None:
        """Resets FSM state, buffer, motion detector, and counters."""
        self._state = GestureState.IDLE
        self._buffer.clear()
        self._low_motion_counter = 0
        self._cooldown_counter = 0
        self._last_motion_score = 0.0
        self.detector.reset()
        logger.info("GestureSegmenter reset to IDLE.")

    def process_frame(
        self, features: np.ndarray
    ) -> Tuple[GestureState, Optional[np.ndarray]]:
        """
        Process a single landmark feature vector (1629,).

        Returns:
            (current_state, gesture_array_or_None)
            where gesture_array is returned only when a gesture completes (PROCESSING state).
        """
        if features.shape != (FEATURE_DIM,):
            raise GestureSegmenterError(
                f"Expected feature vector of shape ({FEATURE_DIM},), got {features.shape}"
            )

        motion_score, is_moving = self.detector.update(features)
        self._last_motion_score = motion_score

        # --- Handle COOLDOWN state transition ---
        if self._state == GestureState.COOLDOWN:
            self._cooldown_counter -= 1
            if self._cooldown_counter <= 0:
                self._state = GestureState.IDLE
                logger.info("Cooldown completed. FSM returned to IDLE.")
            return self._state, None

        # --- Handle PROCESSING → COOLDOWN transition (no frame evaluation) ---
        if self._state == GestureState.PROCESSING:
            if self.config.cooldown_frames > 0:
                self._state = GestureState.COOLDOWN
                self._cooldown_counter = self.config.cooldown_frames
            else:
                self._state = GestureState.IDLE
            return self._state, None

        # --- FSM Logic ---
        if self._state == GestureState.IDLE:
            if is_moving and not is_empty_frame(features):
                self._state = GestureState.RECORDING
                self._buffer = [features.copy()]
                self._low_motion_counter = 0
                logger.info("Motion detected! FSM transitioned IDLE -> RECORDING.")
                return self._state, None
            return self._state, None

        if self._state == GestureState.RECORDING:
            self._buffer.append(features.copy())

            if is_moving:
                self._low_motion_counter = 0
            elif is_empty_frame(features):
                # Tracking loss — don't advance the timeout counter.
                # The MotionDetector already counts empty frames and returns
                # is_moving=False only after sustained loss (>=15 frames).
                # We just pin the counter so the gesture survives brief outages.
                logger.debug(
                    "Empty frame during RECORDING (tracking loss, pinning timeout counter at %d).",
                    self._low_motion_counter,
                )
            else:
                # Valid landmarks but below motion threshold — genuine stillness
                self._low_motion_counter += 1

            reached_timeout = self._low_motion_counter >= self.config.end_timeout_frames
            reached_max = len(self._buffer) >= self.config.max_gesture_frames

            if reached_timeout or reached_max:
                gesture_length = len(self._buffer)

                if gesture_length < self.config.min_gesture_frames:
                    logger.info(
                        "Gesture ended but discarded (too short: %d < min %d frames).",
                        gesture_length,
                        self.config.min_gesture_frames,
                    )
                    self._buffer.clear()
                    if self.config.cooldown_frames > 0:
                        self._state = GestureState.COOLDOWN
                        self._cooldown_counter = self.config.cooldown_frames
                    else:
                        self._state = GestureState.IDLE
                    return self._state, None

                # Valid gesture completed!
                logger.info(
                    "Gesture completed successfully (%d frames recorded, reason: %s).",
                    gesture_length,
                    "timeout" if reached_timeout else "max_frames",
                )
                gesture_array = np.stack(self._buffer, axis=0).astype(np.float32)
                self._buffer.clear()
                self._state = GestureState.PROCESSING

                if self.config.cooldown_frames > 0:
                    self._cooldown_counter = self.config.cooldown_frames

                return GestureState.PROCESSING, gesture_array

            return self._state, None

        return self._state, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- Testing gesture_segmenter.py ---")

    cfg = GestureConfig(
        motion_threshold=0.01,
        min_gesture_frames=10,
        max_gesture_frames=50,
        end_timeout_frames=5,
        cooldown_frames=3,
    )
    segmenter = GestureSegmenter(cfg)

    # 1. Idle frames (no motion)
    zero_frame = np.zeros((1629,), dtype=np.float32)
    for _ in range(10):
        st, g = segmenter.process_frame(zero_frame)
        assert st == GestureState.IDLE
        assert g is None

    # 2. Start gesture with motion — need 2 valid frames: first warms up
    # the MotionDetector (no previous features → score=0), second produces
    # a real motion score that crosses the threshold.
    moving_frame_1 = np.ones((1629,), dtype=np.float32) * 0.1
    st, g = segmenter.process_frame(moving_frame_1)
    assert st == GestureState.IDLE, "First valid frame after idle warms up detector (no prev features)."

    moving_frame_2 = np.ones((1629,), dtype=np.float32) * 0.2
    st, g = segmenter.process_frame(moving_frame_2)
    assert st == GestureState.RECORDING

    # Add frames with increasing motion
    for i in range(15):
        mf = np.ones((1629,), dtype=np.float32) * (0.2 + (i + 1) * 0.05)
        st, g = segmenter.process_frame(mf)
        assert st == GestureState.RECORDING

    # 3. Stop motion (trigger timeout)
    still_frame = np.ones((1629,), dtype=np.float32) * 0.85
    captured_gesture = None
    for _ in range(12):
        st, g = segmenter.process_frame(still_frame)
        if g is not None:
            captured_gesture = g

    assert captured_gesture is not None
    print(f"Captured gesture shape: {captured_gesture.shape}")
    assert captured_gesture.ndim == 2
    assert captured_gesture.shape[1] == 1629
    print("GestureSegmenter FSM test PASSED.")
