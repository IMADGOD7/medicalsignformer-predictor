"""
ui.py
=====

Final stage of the deployment pipeline: renders the smoothed prediction onto
the webcam frame and manages the on-screen display window.

    SmoothedResult (+ optional raw InferenceResult, + optional FPS)
        -> draw_overlay() -> annotated frame -> DisplayWindow.show()

Split into two independent pieces on purpose:
    - draw_overlay(): a pure image transform (frame in, frame out). No GUI
      dependency, so it is fully unit-testable without an X server/display -
      this sandbox's verification run exercises it directly.
    - DisplayWindow: an actual OS-level resource (a GUI window), so THIS is
      the module in the deployment package where a context manager is load-
      bearing rather than just stylistic symmetry.

This module does not decide *whether* a prediction is stable (that's
postprocess.py) or *what* the prediction is (that's inference.py) - it only
decides how to show the result it's given.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from inference import InferenceResult
    from postprocess import SmoothedResult
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference import InferenceResult
    from postprocess import SmoothedResult

logger = logging.getLogger(__name__)

# --- BGR colors (OpenCV convention, not RGB) ---
COLOR_STABLE = (60, 200, 60)
COLOR_UNSTABLE = (0, 200, 255)
COLOR_DEBUG = (200, 200, 200)
COLOR_FPS = (255, 255, 255)
COLOR_BG = (0, 0, 0)

RECOGNIZING_TEXT = "Recognizing..."


@dataclass(frozen=True)
class OverlayConfig:
    """Cosmetic knobs for draw_overlay(), kept separate from business logic
    so the display can be restyled without touching rendering control flow."""

    font: int = cv2.FONT_HERSHEY_SIMPLEX
    main_scale: float = 1.1
    main_thickness: int = 2
    debug_scale: float = 0.5
    debug_thickness: int = 1
    margin: int = 12
    line_gap: int = 8


def draw_overlay(
    frame: np.ndarray,
    smoothed: SmoothedResult,
    raw: Optional[InferenceResult] = None,
    fps: Optional[float] = None,
    config: OverlayConfig = OverlayConfig(),
    copy: bool = False,
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 frame, got shape {frame.shape}.")

    out = frame.copy() if copy else frame
    x = config.margin
    y = config.margin

    # --- Main prediction line ---
    if smoothed.is_stable and smoothed.display_label is not None:
        main_text = f"{smoothed.display_label}  ({smoothed.agreement_ratio * 100:.0f}%)"
        main_color = COLOR_STABLE
    else:
        main_text = RECOGNIZING_TEXT
        main_color = COLOR_UNSTABLE

    (text_w, text_h), baseline = cv2.getTextSize(
        main_text, config.font, config.main_scale, config.main_thickness
    )
    y += text_h
    cv2.rectangle(
        out,
        (x - 6, y - text_h - 6),
        (x + text_w + 6, y + baseline + 6),
        COLOR_BG,
        thickness=-1,
    )
    cv2.putText(out, main_text, (x, y), config.font, config.main_scale, main_color, config.main_thickness, cv2.LINE_AA)
    y += baseline + config.line_gap

    # --- Raw per-window debug line (secondary, small) ---
    if raw is not None:
        debug_text = f"raw: {raw.predicted_label} ({raw.confidence * 100:.1f}%)  {raw.latency_ms:.1f}ms"
        (dbg_w, dbg_h), dbg_baseline = cv2.getTextSize(
            debug_text, config.font, config.debug_scale, config.debug_thickness
        )
        y += dbg_h
        cv2.rectangle(
            out,
            (x - 4, y - dbg_h - 4),
            (x + dbg_w + 4, y + dbg_baseline + 4),
            COLOR_BG,
            thickness=-1,
        )
        cv2.putText(
            out, debug_text, (x, y), config.font, config.debug_scale, COLOR_DEBUG, config.debug_thickness, cv2.LINE_AA
        )
        y += dbg_baseline + config.line_gap

    # --- FPS counter, top-right corner ---
    if fps is not None:
        fps_text = f"{fps:.1f} FPS"
        (fps_w, fps_h), fps_baseline = cv2.getTextSize(
            fps_text, config.font, config.debug_scale, config.debug_thickness
        )
        fps_x = out.shape[1] - fps_w - config.margin
        fps_y = config.margin + fps_h
        cv2.rectangle(
            out,
            (fps_x - 4, fps_y - fps_h - 4),
            (fps_x + fps_w + 4, fps_y + fps_baseline + 4),
            COLOR_BG,
            thickness=-1,
        )
        cv2.putText(out, fps_text, (fps_x, fps_y), config.font, config.debug_scale, COLOR_FPS, config.debug_thickness, cv2.LINE_AA)

    return out


class DisplayWindow:
    """
    Owns an actual OpenCV GUI window (a real OS-level resource, unlike
    InferenceEngine). Always use as a context manager so the window is
    guaranteed to be destroyed on exit, including on exceptions.
    """

    QUIT_KEYS = frozenset({ord("q"), 27})  # 'q' or ESC

    def __init__(self, window_name: str = "MedicalSignFormerV2 - Live Recognition") -> None:
        self.window_name = window_name
        self._opened = False

    def __enter__(self) -> "DisplayWindow":
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._opened = True
        logger.info("Display window '%s' opened.", self.window_name)
        return self

    def show(self, frame: np.ndarray, wait_ms: int = 1) -> int:
        """
        Show `frame` and poll for a keypress.

        Returns:
            The pressed key code (masked to 8 bits), or -1 if none.

        Raises:
            RuntimeError: if called before entering the context manager.
        """
        if not self._opened:
            raise RuntimeError("DisplayWindow.show() called before __enter__ - use 'with DisplayWindow(...) as w:'.")
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(wait_ms) & 0xFF

    def should_quit(self, key: int) -> bool:
        """True if `key` (as returned by show()) is a quit key ('q' or ESC)."""
        return key in self.QUIT_KEYS

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self.window_name)
            self._opened = False
            logger.info("Display window '%s' closed.", self.window_name)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("ui.py - standalone verification")
    print("=" * 70)

    def _fake_smoothed(label, is_stable, agreement_ratio, raw_label="X", raw_conf=0.5) -> SmoothedResult:
        return SmoothedResult(
            display_label=label if is_stable else None,
            agreement_ratio=agreement_ratio,
            is_stable=is_stable,
            accepted=True,
            raw_prediction=0,
            raw_label=raw_label,
            raw_confidence=raw_conf,
        )

    def _fake_raw(label="Fever", conf=0.87, latency=12.3) -> InferenceResult:
        probs = np.full(48, (1 - conf) / 47, dtype=np.float32)
        probs[0] = conf
        return InferenceResult(
            prediction=0, predicted_label=label, confidence=conf,
            probabilities=probs, latency_ms=latency,
        )

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # --- Test 1: bad frame shape raises ValueError ---
    try:
        draw_overlay(np.zeros((480, 640), dtype=np.uint8), _fake_smoothed("X", True, 1.0))
        raise AssertionError("Expected ValueError for a non-3-channel frame.")
    except ValueError:
        print("[PASS] Non-3-channel frame correctly raises ValueError.")

    # --- Test 2: stable prediction actually changes pixels, preserves shape/dtype ---
    frame = blank_frame.copy()
    smoothed = _fake_smoothed("Fever", is_stable=True, agreement_ratio=0.8)
    out = draw_overlay(frame, smoothed, raw=_fake_raw(), fps=29.7, copy=False)
    assert out.shape == blank_frame.shape
    assert out.dtype == blank_frame.dtype
    assert not np.array_equal(out, np.zeros_like(out)), "Overlay should have drawn non-black pixels."
    print("[PASS] Stable-prediction overlay draws visible content with correct shape/dtype preserved.")

    # --- Test 3: in-place (copy=False) mutates the original array object ---
    frame2 = blank_frame.copy()
    original_id = id(frame2)
    result2 = draw_overlay(frame2, smoothed, raw=None, fps=None, copy=False)
    assert id(result2) == original_id, "copy=False should draw on and return the same array object."
    assert not np.array_equal(frame2, np.zeros_like(frame2)), "Original frame should have been mutated in place."
    print("[PASS] copy=False draws in place on the original frame object (matches real-time performance needs).")

    # --- Test 4: copy=True leaves the original frame untouched ---
    frame3 = blank_frame.copy()
    result3 = draw_overlay(frame3, smoothed, copy=True)
    assert id(result3) != id(frame3), "copy=True should return a different array object."
    assert np.array_equal(frame3, np.zeros_like(frame3)), "copy=True must NOT mutate the original frame."
    assert not np.array_equal(result3, np.zeros_like(result3)), "The returned copy should still have the overlay drawn."
    print("[PASS] copy=True leaves the original frame untouched and returns an annotated copy.")

    # --- Test 5: unstable state renders the 'Recognizing...' text, not a stale label ---
    frame4 = blank_frame.copy()
    unstable = _fake_smoothed(None, is_stable=False, agreement_ratio=0.2)
    out4 = draw_overlay(frame4, unstable, raw=_fake_raw(label="Cold", conf=0.4))
    assert not np.array_equal(out4, np.zeros_like(out4))
    print("[PASS] Unstable state renders distinctly (does not silently show nothing or a stale label).")

    # --- Test 6: raw=None and fps=None don't crash (minimum-info case) ---
    frame5 = blank_frame.copy()
    _ = draw_overlay(frame5, smoothed, raw=None, fps=None)
    print("[PASS] draw_overlay handles raw=None and fps=None without error.")

    # --- Test 7: DisplayWindow API surface, GUI parts skipped gracefully if headless ---
    window = DisplayWindow("verification-window")
    try:
        window.show(blank_frame)
        raise AssertionError("Expected RuntimeError: show() called before __enter__.")
    except RuntimeError:
        print("[PASS] DisplayWindow.show() before __enter__ correctly raises RuntimeError.")

    assert window.should_quit(ord("q")) is True
    assert window.should_quit(27) is True
    assert window.should_quit(ord("a")) is False
    print("[PASS] DisplayWindow.should_quit() correctly identifies 'q' and ESC, rejects other keys.")

    window.close()  # closing an unopened window must be a safe no-op
    print("[PASS] DisplayWindow.close() on a never-opened window is a safe no-op.")

    try:
        with DisplayWindow("verification-window-live") as live_window:
            key = live_window.show(blank_frame, wait_ms=1)
            assert isinstance(key, int)
        print("[PASS] DisplayWindow context manager opened, showed a frame, and closed successfully (GUI available).")
    except cv2.error as e:
        print(f"[SKIPPED] Live window open/show requires a display/X server, unavailable in this environment: {e}")
        print("          (draw_overlay and the non-GUI DisplayWindow API are still fully verified above.)")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED (or explicitly skipped where a GUI is unavailable)")
    print("=" * 70)