from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Optional

import cv2
import numpy as np

from gesture_segmenter import GestureConfig, GestureSegmenter, GestureState
from inference import InferenceEngine, InferenceResult
from mediapipe_detector import MediaPipeDetector, MediaPipeDetectorError
from postprocess import SmoothedResult, TemporalSmoother
from resampler import resample_gesture
from ui import DisplayWindow, draw_overlay
from webcam import Webcam, WebcamError

logger = logging.getLogger(__name__)

EMPTY_FRAME_RESET_THRESHOLD = 30


@dataclass
class DeploymentConfig:
    """All tunable parameters in one place."""

    gesture: GestureConfig = field(default_factory=lambda: GestureConfig(
        motion_threshold=0.005,
        motion_smoothing_alpha=0.15,
        min_gesture_frames=10,
        max_gesture_frames=120,
        end_timeout_frames=10,
        cooldown_frames=10,
        target_sequence_length=100,
    ))
    smoother_window: int = 10
    smoother_agreement: float = 0.6
    confidence_threshold: float = 0.5
    mediapipe_max_loss_frames: int = 15
    frame_queue_size: int = 4


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def log_startup_summary(engine: InferenceEngine, config: DeploymentConfig) -> None:
    logger.info("=" * 55)
    logger.info("  MedicalSignFormer Gesture Deployment")
    logger.info("=" * 55)
    logger.info("  Device:              %s", engine.device)
    logger.info("  Sequence length:     %d", config.gesture.target_sequence_length)
    logger.info("  Input dim:           %d", engine._expected_shape[1])
    logger.info("  Classes:             %d", engine.label_names.__len__())
    logger.info("  Smoothing window:    %d", config.smoother_window)
    logger.info("  Smoothing agreement: %.2f", config.smoother_agreement)
    logger.info("  Confidence threshold:%.2f", config.confidence_threshold)
    logger.info("  Motion threshold:    %.4f", config.gesture.motion_threshold)
    logger.info("  Min gesture frames:  %d", config.gesture.min_gesture_frames)
    logger.info("  Max gesture frames:  %d", config.gesture.max_gesture_frames)
    logger.info("=" * 55)


class MediaPipeWorker:
    """Runs MediaPipe in a background thread so the main loop stays free
    for model inference and UI rendering."""

    def __init__(self, max_loss_frames: int = 15, queue_size: int = 4):
        self._detector: Optional[MediaPipeDetector] = None
        self._max_loss_frames = max_loss_frames
        self._input_queue: Queue = Queue(maxsize=queue_size)
        self._output_queue: Queue = Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_count = 0
        self._empty_streak = 0

    def start(self) -> None:
        self._detector = MediaPipeDetector()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame_rgb = self._input_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                features = self._detector.process(frame_rgb)
            except (MediaPipeDetectorError, ValueError, cv2.error):
                features = np.zeros(1629, dtype=np.float32)
            self._frame_count += 1
            if np.count_nonzero(features) == 0:
                self._empty_streak += 1
            else:
                self._empty_streak = 0
            try:
                self._output_queue.put_nowait(features)
            except Exception:
                pass  # drop frame if consumer is behind

    def submit(self, frame_rgb: np.ndarray) -> None:
        try:
            self._input_queue.put_nowait(frame_rgb)
        except Exception:
            pass  # drop frame if worker is behind

    def get_features(self) -> Optional[np.ndarray]:
        try:
            return self._output_queue.get_nowait()
        except Empty:
            return None

    def tracking_lost(self) -> bool:
        return self._empty_streak >= self._max_loss_frames

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._detector is not None:
            self._detector.close()


def main() -> None:
    configure_logging()
    deploy_config = DeploymentConfig()

    gesture_config = deploy_config.gesture
    smoother = TemporalSmoother(
        window_size=deploy_config.smoother_window,
        min_agreement_ratio=deploy_config.smoother_agreement,
        confidence_threshold=deploy_config.confidence_threshold,
    )
    segmenter = GestureSegmenter(gesture_config)

    try:
        with (
            Webcam() as webcam,
            InferenceEngine() as engine,
            DisplayWindow() as window,
        ):
            mp_worker = MediaPipeWorker(
                max_loss_frames=deploy_config.mediapipe_max_loss_frames,
                queue_size=deploy_config.frame_queue_size,
            )
            mp_worker.start()
            log_startup_summary(engine, deploy_config)

            fps_ema: Optional[float] = None
            alpha = 0.1
            empty_frame_counter = 0
            latest_raw: Optional[InferenceResult] = None
            latest_smoothed: Optional[SmoothedResult] = None
            gesture_active = False  # True while a gesture is being recorded/processed

            for frame_bgr in webcam.frames():
                loop_start = time.perf_counter()

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_worker.submit(frame_rgb)
                features = mp_worker.get_features()

                if features is None:
                    empty_frame_counter += 1
                    if empty_frame_counter >= EMPTY_FRAME_RESET_THRESHOLD:
                        smoother.reset()
                        segmenter.reset()
                        latest_raw = None
                        latest_smoothed = None
                        gesture_active = False
                    continue
                empty_frame_counter = 0

                if mp_worker.tracking_lost():
                    smoother.reset()
                    segmenter.reset()
                    latest_raw = None
                    latest_smoothed = None
                    gesture_active = False

                # Process frame through FSM
                state, completed_gesture = segmenter.process_frame(features)

                # Run inference on completed gesture
                if completed_gesture is not None:
                    raw_frames = completed_gesture.shape[0]
                    logger.info("Gesture captured (%d frames). Resampling to %d...",
                                raw_frames, gesture_config.target_sequence_length)

                    resampled = resample_gesture(
                        completed_gesture,
                        target_length=gesture_config.target_sequence_length,
                    )
                    try:
                        latest_raw = engine.predict(resampled)
                        latest_smoothed = smoother.update(latest_raw)
                        gesture_active = False  # gesture done, now showing result

                        logger.info(
                            "Prediction -> %s | %.1f%% | %.1fms",
                            latest_raw.predicted_label,
                            latest_raw.confidence * 100.0,
                            latest_raw.latency_ms,
                        )
                    except RuntimeError as e:
                        logger.warning("Inference failed: %s", e)
                        latest_raw = None
                        latest_smoothed = None
                        gesture_active = False

                # Mark gesture as active if we're recording
                if state == GestureState.RECORDING:
                    gesture_active = True
                    # Clear stale predictions from a previous gesture
                    if latest_raw is not None and latest_smoothed is not None:
                        latest_raw = None
                        latest_smoothed = None

                # Build display data — hide stale predictions during active recording
                display_smoothed = latest_smoothed if not gesture_active else None
                display_raw = latest_raw if not gesture_active else None

                draw_overlay(
                    frame_rgb,
                    smoothed=display_smoothed,
                    raw=display_raw,
                    fps=fps_ema,
                    gesture_state=state.name,
                    recorded_frames=segmenter.buffer_length,
                    motion_score=segmenter.last_motion_score,
                    motion_threshold=gesture_config.motion_threshold,
                    copy=False,
                )

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                key = window.show(frame_bgr)
                if window.should_quit(key):
                    logger.info("Quit requested.")
                    break

                delta = time.perf_counter() - loop_start
                if delta > 0:
                    instant_fps = 1.0 / delta
                    fps_ema = (
                        instant_fps
                        if fps_ema is None
                        else (alpha * instant_fps + (1.0 - alpha) * fps_ema)
                    )

    except WebcamError as e:
        logger.error("Webcam error: %s", e)
    except MediaPipeDetectorError as e:
        logger.error("MediaPipe error: %s", e)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error("Runtime error: %s", e)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception:
        logger.exception("Unexpected error.")
    finally:
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
