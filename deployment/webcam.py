from __future__ import annotations

import logging
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamError(RuntimeError):
    """Raised when the webcam cannot be opened or a fatal read failure
    occurs. Callers (e.g. app.py) should catch this specifically to show
    a clean user-facing error rather than an unhandled traceback."""


class Webcam:
    def __init__(
        self,
        camera_index: int = 0,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        max_consecutive_read_failures: int = 10,
    ) -> None:
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_consecutive_read_failures = max_consecutive_read_failures

        self._capture: Optional[cv2.VideoCapture] = None
        self._is_open = False

    def open(self) -> None:
        if self._is_open:
            logger.debug("Webcam already open (camera_index=%d); open() is a no-op.", self.camera_index)
            return

        logger.info("Opening webcam (camera_index=%d)...", self.camera_index)
        capture = cv2.VideoCapture(self.camera_index)

        if not capture.isOpened():
            capture.release()
            raise WebcamError(
                f"Could not open webcam at camera_index={self.camera_index}. "
                f"Check that a camera is connected, not in use by another "
                f"application, and that camera_index is correct (0 is usually "
                f"the default camera; try 1, 2, ... for additional cameras)."
            )

        if self.frame_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        if self.frame_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        self._capture = capture
        self._is_open = True
        logger.info("Webcam opened successfully (camera_index=%d).", self.camera_index)

    def frames(self) -> Iterator[np.ndarray]:
        if not self._is_open or self._capture is None:
            raise WebcamError(
                "Webcam.frames() called before open() succeeded. "
                "Call open() first, or use Webcam as a context manager."
            )

        consecutive_failures = 0

        while True:
            success, frame_bgr = self._capture.read()

            if not success or frame_bgr is None:
                consecutive_failures += 1
                logger.warning(
                    "Failed to read frame from webcam (%d/%d consecutive failures).",
                    consecutive_failures, self.max_consecutive_read_failures,
                )
                if consecutive_failures >= self.max_consecutive_read_failures:
                    raise WebcamError(
                        f"Webcam read failed {consecutive_failures} times in a row - "
                        f"the camera appears to have disconnected. Stopping capture."
                    )
                continue

            consecutive_failures = 0  # reset on any successful read
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_rgb

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            logger.info("Webcam released (camera_index=%d).", self.camera_index)
        self._capture = None
        self._is_open = False

    def __enter__(self) -> "Webcam":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._is_open


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    print("--- Test 1: opening a very unlikely-to-exist camera index ---")
    webcam = Webcam(camera_index=99)
    try:
        webcam.open()
        print("UNEXPECTED: camera_index=99 opened successfully in this environment.")
    except WebcamError as e:
        print(f"Correctly raised WebcamError: {e}")

    print("\n--- Test 2: close() is safe even though open() failed ---")
    webcam.close()  # should not raise
    webcam.close()  # calling twice should also not raise
    print("close() called twice after a failed open() - no exception. PASSED")

    print("\n--- Test 3: frames() before a successful open() raises WebcamError ---")
    webcam2 = Webcam(camera_index=99)
    try:
        next(webcam2.frames())
        print("UNEXPECTED: frames() did not raise.")
    except WebcamError as e:
        print(f"Correctly raised WebcamError: {e}")

    print("\n--- Test 4: context manager surfaces the same failure correctly ---")
    try:
        with Webcam(camera_index=99):
            print("UNEXPECTED: context manager entered successfully.")
    except WebcamError as e:
        print(f"Correctly raised WebcamError via context manager: {e}")

    print("\n--- Test 5: attempting camera_index=0 (may or may not exist here) ---")
    webcam3 = Webcam(camera_index=0)
    try:
        webcam3.open()
        print("Camera 0 opened - this environment DOES have a usable camera device.")
        frame = next(webcam3.frames())
        print(f"Captured one frame: shape={frame.shape}, dtype={frame.dtype}")
        assert frame.ndim == 3 and frame.shape[2] == 3, "Expected an (H, W, 3) RGB frame"
        print("Frame shape verification: PASSED")
    except WebcamError as e:
        print(f"No usable camera at index 0 in this environment (expected in a sandbox): {e}")
    finally:
        webcam3.close()

    print("\nAll verifiable-without-hardware checks complete.")
    print("Run this file on a machine with a real webcam to verify actual frame capture.")