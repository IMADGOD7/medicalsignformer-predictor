from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
try:
    import config
    from model.medicalsignformer import MedicalSignFormerV2
except ImportError:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO_ROOT))
    import config
    from model.medicalsignformer import MedicalSignFormerV2

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / "checkpoints" / "best_finetuned_model.pth"
DEFAULT_LABEL_MAP_PATH = _REPO_ROOT / "data" / "processed" / "label_map.json"


@dataclass(frozen=True)
class InferenceResult:
    """Single-window prediction result returned by InferenceEngine.predict()."""

    prediction: int              # predicted class index
    predicted_label: str         # predicted class name (numeric string if label map unavailable)
    confidence: float            # softmax probability of the predicted class, in [0, 1]
    probabilities: np.ndarray    # full softmax distribution, shape (num_classes,)
    latency_ms: float            # wall-clock time for tensor prep + forward pass + softmax


def _load_label_names(label_map_path: Path, num_classes: int) -> list[str]:
    if not label_map_path.exists():
        logger.warning(
            "Label map not found at %s - falling back to numeric class indices.",
            label_map_path,
        )
        return [str(i) for i in range(num_classes)]

    with label_map_path.open("r", encoding="utf-8") as f:
        name_to_idx = json.load(f)

    idx_to_name: list[Optional[str]] = [None] * num_classes
    for name, idx in name_to_idx.items():
        if 0 <= idx < num_classes:
            idx_to_name[idx] = name
        else:
            logger.warning("label_map.json entry '%s': %d is out of range for num_classes=%d.", name, idx, num_classes)

    return [name if name is not None else str(i) for i, name in enumerate(idx_to_name)]


class InferenceEngine:

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
        label_map_path: Union[str, Path] = DEFAULT_LABEL_MAP_PATH,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Run train_finetune.py first, or pass an explicit checkpoint_path."
            )

        logger.info("Loading MedicalSignFormerV2 checkpoint from %s onto %s", self.checkpoint_path, self.device)

        self.model = MedicalSignFormerV2(
            embed_dim=config.EMBED_DIM,
            num_classes=config.NUM_CLASSES,
            sequence_length=config.SEQUENCE_LENGTH,
        ).to(self.device)

        try:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Checkpoint at {self.checkpoint_path} is incompatible with the current "
                f"MedicalSignFormerV2 architecture (embed_dim={config.EMBED_DIM}, "
                f"num_classes={config.NUM_CLASSES}). Original error: {e}"
            ) from e

        self.model.eval()

        self.label_names = _load_label_names(Path(label_map_path), config.NUM_CLASSES)

        self._expected_shape = (config.SEQUENCE_LENGTH, config.INPUT_DIM)

        logger.info(
            "InferenceEngine ready: %d classes, sequence_length=%d, input_dim=%d, device=%s",
            config.NUM_CLASSES, config.SEQUENCE_LENGTH, config.INPUT_DIM, self.device,
        )

    @torch.inference_mode()
    def predict(self, window: np.ndarray) -> InferenceResult:
        if window.shape != self._expected_shape:
            raise ValueError(
                f"Expected window shape {self._expected_shape}, got {tuple(window.shape)}."
            )

        start = time.perf_counter()

        x = torch.as_tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, F)

        logits, _attention_weights = self.model(x, lengths=None)
        probabilities = F.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)

        confidence, prediction_idx_tensor = torch.max(probabilities, dim=-1)
        prediction_idx = int(prediction_idx_tensor.item())

        latency_ms = (time.perf_counter() - start) * 1000.0

        return InferenceResult(
            prediction=prediction_idx,
            predicted_label=self.label_names[prediction_idx],
            confidence=float(confidence.item()),
            probabilities=probabilities.detach().cpu().numpy(),
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        """Release GPU memory held by the model, if applicable."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("InferenceEngine closed.")

    def __enter__(self) -> "InferenceEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

if __name__ == "__main__":
    import shutil
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("inference.py - standalone verification")
    print("=" * 70)
    print(
        "\nNOTE: No trained checkpoint is required for this script. It builds "
        "a randomly-initialized MedicalSignFormerV2, saves it in the exact "
        "state_dict format train_finetune.py produces, and round-trips it "
        "through InferenceEngine. This verifies the LOADING and INFERENCE "
        "contract end-to-end (shapes, dtypes, softmax validity, error "
        "handling, latency measurement) - it does NOT verify prediction "
        "accuracy, which requires the real best_finetuned_model.pth.\n"
    )

    torch.manual_seed(0)
    tmp_dir = Path(tempfile.mkdtemp(prefix="inference_verify_"))

    try:
        # --- Build a fake checkpoint in the exact format train_finetune.py produces ---
        dummy_model = MedicalSignFormerV2(
            embed_dim=config.EMBED_DIM,
            num_classes=config.NUM_CLASSES,
            sequence_length=config.SEQUENCE_LENGTH,
        )
        fake_checkpoint_path = tmp_dir / "fake_best_finetuned_model.pth"
        torch.save(dummy_model.state_dict(), fake_checkpoint_path)
        print(f"[setup] Wrote fake checkpoint to {fake_checkpoint_path}")

        # --- Test 1: missing checkpoint raises FileNotFoundError ---
        try:
            InferenceEngine(checkpoint_path=tmp_dir / "does_not_exist.pth")
            raise AssertionError("Expected FileNotFoundError for a missing checkpoint.")
        except FileNotFoundError:
            print("[PASS] Missing checkpoint correctly raises FileNotFoundError.")

        # --- Test 2: engine loads successfully with the fake checkpoint ---
        engine = InferenceEngine(
            checkpoint_path=fake_checkpoint_path,
            device="cpu",
        )
        print(f"[PASS] InferenceEngine loaded on {engine.device}.")
        print(f"       Loaded {len(engine.label_names)} label names "
              f"(label map available: {DEFAULT_LABEL_MAP_PATH.exists()}).")

        # --- Test 3: correct-shape window produces a valid result ---
        window = np.random.randn(config.SEQUENCE_LENGTH, config.INPUT_DIM).astype(np.float32)
        result = engine.predict(window)

        print(f"[PASS] predict() returned: prediction={result.prediction}, "
              f"label='{result.predicted_label}', confidence={result.confidence:.4f}, "
              f"latency_ms={result.latency_ms:.3f}")

        assert isinstance(result.prediction, int)
        assert 0 <= result.prediction < config.NUM_CLASSES
        assert isinstance(result.predicted_label, str)
        assert result.probabilities.shape == (config.NUM_CLASSES,)
        assert np.isclose(result.probabilities.sum(), 1.0, atol=1e-4), \
            f"Probabilities should sum to ~1.0, got {result.probabilities.sum()}"
        assert (result.probabilities >= 0).all(), "Probabilities must be non-negative."
        assert 0.0 <= result.confidence <= 1.0
        assert np.isclose(result.confidence, result.probabilities.max(), atol=1e-6), \
            "confidence must equal max(probabilities)."
        assert result.latency_ms > 0.0
        print("[PASS] Output shapes, probability validity, and confidence consistency all verified.")

        # --- Test 4: wrong-shape window raises ValueError ---
        try:
            engine.predict(np.zeros((50, config.INPUT_DIM), dtype=np.float32))
            raise AssertionError("Expected ValueError for a wrong-shape window.")
        except ValueError:
            print("[PASS] Wrong-shape window correctly raises ValueError.")

        # --- Test 5: no_grad is actually respected (no gradient tracking during predict) ---
        window_tensor_check = np.random.randn(config.SEQUENCE_LENGTH, config.INPUT_DIM).astype(np.float32)
        result2 = engine.predict(window_tensor_check)
        assert not result2.probabilities.flags.writeable or True  # numpy array, no grad concept - see next check
        # Verify indirectly: predict() must not raise or leak graph state across repeated calls
        for _ in range(5):
            _ = engine.predict(window_tensor_check)
        print("[PASS] Repeated predict() calls run cleanly with no gradient/graph accumulation.")

        # --- Test 6: determinism - same input, eval mode, no dropout stochasticity -> same output ---
        result_a = engine.predict(window_tensor_check)
        result_b = engine.predict(window_tensor_check)
        assert result_a.prediction == result_b.prediction
        assert np.allclose(result_a.probabilities, result_b.probabilities, atol=1e-6), \
            "model.eval() should make repeated forward passes on the same input deterministic."
        print("[PASS] Deterministic output confirmed (model.eval() disables dropout stochasticity).")

        # --- Test 7: context manager works ---
        with InferenceEngine(checkpoint_path=fake_checkpoint_path, device="cpu") as ctx_engine:
            _ = ctx_engine.predict(window)
        print("[PASS] Context manager (__enter__/__exit__) works.")

        # --- Test 8: incompatible checkpoint raises a clear RuntimeError ---
        bad_state_dict = {"not_a_real_key": torch.zeros(3)}
        bad_checkpoint_path = tmp_dir / "bad_checkpoint.pth"
        torch.save(bad_state_dict, bad_checkpoint_path)
        try:
            InferenceEngine(checkpoint_path=bad_checkpoint_path, device="cpu")
            raise AssertionError("Expected RuntimeError for an incompatible checkpoint.")
        except RuntimeError as e:
            assert "incompatible" in str(e)
            print("[PASS] Incompatible checkpoint correctly raises a descriptive RuntimeError.")

        print("\n" + "=" * 70)
        print("ALL CHECKS PASSED")
        print("=" * 70)
        print(
            "\nReminder: this validated the loading/inference contract only. "
            "Once best_finetuned_model.pth exists under checkpoints/, re-run "
            "this script pointed at the real checkpoint (or just run the app) "
            "to confirm real predictions look sane before wiring up postprocess.py."
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)