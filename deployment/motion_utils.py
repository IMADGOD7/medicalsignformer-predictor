from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def is_empty_frame(features: np.ndarray) -> bool:
    return np.count_nonzero(features) == 0


def compute_motion_score(
    previous: np.ndarray,
    current: np.ndarray,
) -> float:
    if previous.shape != current.shape:
        raise ValueError(
            f"Shape mismatch: {previous.shape} vs {current.shape}"
        )
    if is_empty_frame(previous) or is_empty_frame(current):
        return 0.0
    return float(np.mean(np.abs(current - previous)))


@dataclass
class MotionFilter:
    alpha: float = 0.25
    _value: Optional[float] = None

    def reset(self) -> None:
        self._value = None

    def update(self, score: float) -> float:
        if self._value is None:
            self._value = score
        else:
            self._value = self.alpha * score + (1.0 - self.alpha) * self._value
        return self._value


def is_motion(score: float, threshold: float) -> bool:
    return score >= threshold


class MotionDetector:
    """Motion detection that distinguishes genuine stillness from tracking loss.

    When MediaPipe loses the person mid-gesture (returns all-zero features),
    the old code would immediately drop the motion score to 0 and trigger
    gesture end.  The new code counts consecutive empty frames and only ends
    a gesture after sustained tracking loss, giving MediaPipe time to recover.
    """

    def __init__(
        self,
        threshold: float = 0.005,
        alpha: float = 0.25,
        max_tracking_loss_frames: int = 15,
    ):
        self.threshold = threshold
        self.filter = MotionFilter(alpha=alpha)
        self.max_tracking_loss_frames = max_tracking_loss_frames
        self._previous_features: Optional[np.ndarray] = None
        self._tracking_loss_counter: int = 0

    def update(self, features: np.ndarray) -> tuple[float, bool]:
        if is_empty_frame(features):
            self._tracking_loss_counter += 1

            # If tracking was already lost, keep the score pinned at 0
            if self._tracking_loss_counter > 1:
                self._previous_features = None
                smoothed_score = self.filter.update(0.0)
                # Only report "not moving" after sustained loss
                is_lost = self._tracking_loss_counter >= self.max_tracking_loss_frames
                return smoothed_score, False

            # First empty frame — treat as a brief gap, not motion end
            # Keep the previous score (don't reset filter) so the FSM
            # doesn't see a sudden drop to 0.
            smoothed_score = self.filter._value if self.filter._value is not None else 0.0
            return smoothed_score, True  # Pretend motion continues during brief loss

        # Valid frame recovered
        if self._tracking_loss_counter > 0:
            self._tracking_loss_counter = 0

        if self._previous_features is None:
            self._previous_features = features.copy()
            smoothed_score = self.filter.update(0.0)
            return smoothed_score, False

        raw_score = compute_motion_score(self._previous_features, features)
        self._previous_features = features.copy()

        smoothed_score = self.filter.update(raw_score)
        moving = is_motion(smoothed_score, self.threshold)
        return smoothed_score, moving

    def reset(self) -> None:
        self._previous_features = None
        self.filter.reset()
        self._tracking_loss_counter = 0
