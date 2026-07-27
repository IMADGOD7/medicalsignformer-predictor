from __future__ import annotations

import logging
import sys
import time
from typing import Optional

import cv2
import numpy as np

from inference import InferenceEngine
from mediapipe_detector import MediaPipeDetector, MediaPipeDetectorError
from postprocess import TemporalSmoother
from sliding_window import SlidingWindow, SlidingWindowError
from ui import DisplayWindow, OverlayConfig, draw_overlay
from webcam import Webcam, WebcamError

logger = logging.getLogger(__name__)

# Number of consecutive empty frames before resetting temporal smoother
EMPTY_FRAME_RESET_THRESHOLD = 10


def configure_logging() -> None:
    """Configures system-wide logging format and log level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def log_startup_summary(engine: InferenceEngine, window_length: int) -> None:
    """Logs system startup specifications and model metadata."""
    device = getattr(engine, "device", "CUDA")
    input_dim = getattr(engine, "input_dim", 1629)
    num_classes = getattr(engine, "num_classes", 48)
    checkpoint = getattr(engine, "model_path", "best_finetuned_model.pth")

    logger.info("==================================================")
    logger.info("           MedicalSignFormerV2 Startup            ")
    logger.info("==================================================")
    logger.info(" Device:           %s", device)
    logger.info(" Sequence Length:  %d", window_length)
    logger.info(" Input Dimension:  %d", input_dim)
    logger.info(" Classes:          %d", num_classes)
    logger.info(" Checkpoint:       %s", checkpoint)
    logger.info("==================================================")


def _draw_warmup_overlay(
    frame: np.ndarray,
    current_frames: int,
    total_frames: int,
    fps: Optional[float] = None,
    proc_time_ms: Optional[float] = None,
    config: OverlayConfig = OverlayConfig(),
) -> np.ndarray:
    """Draws warm-up state overlay showing frame count and percentage progress."""
    x = config.margin
    y = config.margin

    pct = int((current_frames / total_frames) * 100) if total_frames > 0 else 0
    line1 = "Collecting Frames..."
    line2 = f"{current_frames} / {total_frames} ({pct}%)"

    # Line 1: Header
    (w1, h1), b1 = cv2.getTextSize(
        line1, config.font, config.main_scale, config.main_thickness
    )
    y += h1
    cv2.rectangle(
        frame,
        (x - 6, y - h1 - 6),
        (x + w1 + 6, y + b1 + 6),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        frame,
        line1,
        (x, y),
        config.font,
        config.main_scale,
        (0, 200, 255),
        config.main_thickness,
        cv2.LINE_AA,
    )
    y += b1 + config.line_gap

    # Line 2: Progress string "37 / 100 (37%)"
    (w2, h2), b2 = cv2.getTextSize(
        line2, config.font, config.main_scale, config.main_thickness
    )
    y += h2
    cv2.rectangle(
        frame,
        (x - 6, y - h2 - 6),
        (x + w2 + 6, y + b2 + 6),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        frame,
        line2,
        (x, y),
        config.font,
        config.main_scale,
        (0, 200, 255),
        config.main_thickness,
        cv2.LINE_AA,
    )

    # Top-right Debug Metrics: FPS and processing latency
    if fps is not None:
        metrics_text = f"{fps:.1f} FPS"
        if proc_time_ms is not None:
            metrics_text += f" | {proc_time_ms:.1f} ms"

        (fps_w, fps_h), fps_b = cv2.getTextSize(
            metrics_text, config.font, config.debug_scale, config.debug_thickness
        )
        fps_x = frame.shape[1] - fps_w - config.margin
        fps_y = config.margin + fps_h
        cv2.rectangle(
            frame,
            (fps_x - 4, fps_y - fps_h - 4),
            (fps_x + fps_w + 4, fps_y + fps_b + 4),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            frame,
            metrics_text,
            (fps_x, fps_y),
            config.font,
            config.debug_scale,
            (255, 255, 255),
            config.debug_thickness,
            cv2.LINE_AA,
        )

    return frame


def main() -> None:
    """Main application loop executing live webcam recognition."""
    configure_logging()
    logger.info("Starting MedicalSignFormerV2 deployment application...")

    try:
        with (
            Webcam() as webcam,
            MediaPipeDetector() as detector,
            InferenceEngine() as engine,
            DisplayWindow() as window,
        ):
            sliding_window = SlidingWindow()
            smoother = TemporalSmoother()

            # Point 3: Log startup configuration summary
            log_startup_summary(engine, sliding_window.sequence_length)

            fps_ema: Optional[float] = None
            proc_time_ms: Optional[float] = None
            alpha = 0.1  # EMA smoothing factor for FPS tracking
            empty_frame_counter = 0

            for frame_rgb in webcam.frames():
                # Point 2: True end-to-end loop timing start
                loop_start = time.perf_counter()

                # Point 1: Fault-tolerant landmark extraction
                try:
                    features = detector.process(frame_rgb)
                except (MediaPipeDetectorError, ValueError, cv2.error) as e:
                    logger.warning("Landmark extraction failed on frame: %s", e)
                    continue

                # Point 7: Auto-reset smoother when hands/landmarks are lost continuously
                if np.all(features == 0):
                    empty_frame_counter += 1
                    if empty_frame_counter == EMPTY_FRAME_RESET_THRESHOLD:
                        smoother.reset()
                else:
                    empty_frame_counter = 0

                # Append feature vector to buffer
                sliding_window.add_frame(features)

                # Point 4: Warm-up state handling with updated UI feedback
                if not sliding_window.is_ready():
                    _draw_warmup_overlay(
                        frame_rgb,
                        len(sliding_window),
                        sliding_window.sequence_length,
                        fps=fps_ema,
                        proc_time_ms=proc_time_ms,
                    )
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    key = window.show(frame_bgr)
                    if window.should_quit(key):
                        logger.info("Quit key pressed during warm-up phase.")
                        break

                    loop_end = time.perf_counter()
                    delta = loop_end - loop_start
                    if delta > 0:
                        instant_fps = 1.0 / delta
                        fps_ema = (
                            instant_fps
                            if fps_ema is None
                            else (alpha * instant_fps + (1.0 - alpha) * fps_ema)
                        )
                        proc_time_ms = delta * 1000.0
                    continue

                # Point 5: Direct extraction (guaranteed non-None by is_ready check)
                window_data = sliding_window.get_window()

                try:
                    raw_result = engine.predict(window_data)
                except RuntimeError as e:
                    logger.warning("Inference failed: %s", e)
                    continue

                smoothed = smoother.update(raw_result)

                # Point 8: Draw overlay with live FPS and frame latency
                draw_overlay(
                    frame_rgb,
                    smoothed=smoothed,
                    raw=raw_result,
                    fps=fps_ema,
                    copy=False,
                )

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                key = window.show(frame_bgr)
                if window.should_quit(key):
                    logger.info("Quit command ('q' or ESC) received from user.")
                    break

                # Points 2 & 8: Compute end-to-end loop latency and FPS
                loop_end = time.perf_counter()
                delta = loop_end - loop_start
                if delta > 0:
                    instant_fps = 1.0 / delta
                    fps_ema = (
                        instant_fps
                        if fps_ema is None
                        else (alpha * instant_fps + (1.0 - alpha) * fps_ema)
                    )
                    proc_time_ms = delta * 1000.0

    except WebcamError as e:
        logger.error("Webcam hardware error: %s", e)

    except MediaPipeDetectorError as e:
        logger.error("MediaPipe detector initialization error: %s", e)

    except SlidingWindowError as e:
        logger.error("Sliding window buffer error: %s", e)

    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error("Application runtime error: %s", e)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught. Initiating graceful shutdown...")

    except Exception:
        logger.exception("Unexpected system exception.")

    finally:
        logger.info("Application shutdown completed cleanly. All resources released.")


if __name__ == "__main__":
    main()