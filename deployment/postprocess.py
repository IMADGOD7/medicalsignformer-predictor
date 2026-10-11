from __future__ import annotations

import logging
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

try:
    from inference import InferenceResult
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference import InferenceResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmoothedResult:
    display_label: Optional[str]
    agreement_ratio: float
    is_stable: bool
    accepted: bool
    raw_prediction: int
    raw_label: str
    raw_confidence: float


class TemporalSmoother:

    def __init__(
        self,
        window_size: int = 10,
        min_agreement_ratio: float = 0.6,
        confidence_threshold: float = 0.5,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        if not (0.0 < min_agreement_ratio <= 1.0):
            raise ValueError(f"min_agreement_ratio must be in (0.0, 1.0], got {min_agreement_ratio}.")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}.")

        self.window_size = window_size
        self.min_agreement_ratio = min_agreement_ratio
        self.confidence_threshold = confidence_threshold
        self._history: Deque[Tuple[int, str]] = deque(maxlen=window_size)

    def update(self, result: InferenceResult) -> SmoothedResult:
        # Use calibrated confidence (top1 - top2 margin) for thresholding.
        # Raw softmax probability is unreliable for multiclass with FocalLoss.
        eff_confidence = getattr(result, "calibrated_confidence", result.confidence)
        accepted = eff_confidence >= self.confidence_threshold

        if accepted:
            self._history.append((result.prediction, result.predicted_label))
        else:
            logger.debug(
                "Rejected window: calibrated_conf=%.4f < threshold=%.2f (label='%s')",
                eff_confidence, self.confidence_threshold, result.predicted_label,
            )

        if not self._history:
            return SmoothedResult(
                display_label=None,
                agreement_ratio=0.0,
                is_stable=False,
                accepted=accepted,
                raw_prediction=result.prediction,
                raw_label=result.predicted_label,
                raw_confidence=result.confidence,
            )

        label_votes = Counter(label for _, label in self._history)
        top_label, top_count = label_votes.most_common(1)[0]
        agreement_ratio = top_count / len(self._history)
        is_stable = agreement_ratio >= self.min_agreement_ratio

        return SmoothedResult(
            display_label=top_label if is_stable else None,
            agreement_ratio=agreement_ratio,
            is_stable=is_stable,
            accepted=accepted,
            raw_prediction=result.prediction,
            raw_label=result.predicted_label,
            raw_confidence=result.confidence,
        )

    def reset(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 70)
    print("postprocess.py - standalone verification")
    print("=" * 70)

    def _fake_result(prediction: int, label: str, confidence: float, calibrated: Optional[float] = None) -> InferenceResult:
        import numpy as np
        num_classes = 48
        probs = np.zeros(num_classes, dtype=np.float32)
        probs[prediction] = confidence
        remaining = (1.0 - confidence) / max(num_classes - 1, 1)
        probs += remaining
        probs[prediction] = confidence
        if calibrated is None:
            calibrated = confidence  # default: clear winner
        return InferenceResult(
            prediction=prediction,
            predicted_label=label,
            confidence=confidence,
            calibrated_confidence=calibrated,
            probabilities=probs,
            latency_ms=1.0,
        )

    for bad_kwargs, bad_field in [
        ({"window_size": 0}, "window_size"),
        ({"min_agreement_ratio": 0.0}, "min_agreement_ratio"),
        ({"min_agreement_ratio": 1.5}, "min_agreement_ratio"),
        ({"confidence_threshold": -0.1}, "confidence_threshold"),
        ({"confidence_threshold": 1.1}, "confidence_threshold"),
    ]:
        try:
            TemporalSmoother(**bad_kwargs)
            raise AssertionError(f"Expected ValueError for bad {bad_field}={bad_kwargs}")
        except ValueError:
            pass
    print("[PASS] Constructor validation rejects out-of-range parameters.")

    smoother = TemporalSmoother(window_size=5, min_agreement_ratio=0.6, confidence_threshold=0.5)
    r = smoother.update(_fake_result(3, "Fever", confidence=0.3))
    assert r.accepted is False
    assert r.is_stable is False
    assert r.display_label is None
    assert len(smoother) == 0
    print("[PASS] Below-threshold prediction is rejected and does not populate history.")

    smoother.reset()
    r = smoother.update(_fake_result(3, "Fever", confidence=0.9))
    assert r.accepted is True
    assert len(smoother) == 1
    assert r.agreement_ratio == 1.0
    assert r.is_stable is True
    assert r.display_label == "Fever"
    print("[PASS] A single accepted vote is trivially its own majority (1/1).")

    # Test calibrated confidence: high raw confidence but low margin -> rejected
    smoother.reset()
    # Two close classes: top1=0.52, top2=0.48 -> calibrated = 0.04 < 0.5 threshold
    r_close = smoother.update(_fake_result(3, "Fever", confidence=0.52, calibrated=0.04))
    assert r_close.accepted is False
    assert r_close.display_label is None
    assert len(smoother) == 0
    print("[PASS] Close prediction (calibrated_conf=0.04 < 0.5) correctly rejected.")

    smoother.reset()
    sequence = [
        ("Fever", 0.9), ("Fever", 0.85), ("Cold", 0.7),
        ("Fever", 0.95), ("Fever", 0.8),
    ]
    last = None
    for label, conf in sequence:
        last = smoother.update(_fake_result(hash(label) % 48, label, conf))
    assert last.agreement_ratio == 0.8, f"Expected 0.8, got {last.agreement_ratio}"
    assert last.is_stable is True
    assert last.display_label == "Fever"
    print(f"[PASS] Majority vote correctly resolves to 'Fever' at agreement_ratio={last.agreement_ratio}.")

    smoother.reset()
    sequence = [("A", 0.9), ("B", 0.9), ("C", 0.9), ("D", 0.9), ("E", 0.9)]
    last = None
    for i, (label, conf) in enumerate(sequence):
        last = smoother.update(_fake_result(i, label, conf))
    assert last.agreement_ratio == 0.2
    assert last.is_stable is False
    assert last.display_label is None
    print("[PASS] Low agreement (0.2 < 0.6) correctly yields is_stable=False, display_label=None.")

    smoother = TemporalSmoother(window_size=3, min_agreement_ratio=0.6, confidence_threshold=0.5)
    smoother.update(_fake_result(1, "Cold", 0.9))
    smoother.update(_fake_result(1, "Cold", 0.9))
    smoother.update(_fake_result(1, "Cold", 0.9))
    assert len(smoother) == 3
    r = smoother.update(_fake_result(2, "Flu", 0.9))
    assert len(smoother) == 3
    assert r.agreement_ratio > 0.6
    assert r.display_label == "Cold"
    print("[PASS] Sliding window correctly evicts oldest votes once window_size is exceeded.")

    smoother.reset()
    assert len(smoother) == 0
    r = smoother.update(_fake_result(0, "Allergies", 0.3))
    assert len(smoother) == 0
    print("[PASS] reset() clears history correctly.")

    smoother = TemporalSmoother(window_size=1, min_agreement_ratio=0.6, confidence_threshold=0.5)
    r1 = smoother.update(_fake_result(5, "Cough", 0.9))
    assert r1.display_label == "Cough"
    r2 = smoother.update(_fake_result(6, "Sneeze", 0.9))
    assert r2.display_label == "Sneeze"
    print("[PASS] window_size=1 correctly behaves as a pure threshold pass-through.")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
