"""Reusable MediaPipe Holistic landmark extraction for MedicalSignFormer.

This module preserves the extraction logic from the Colab reference notebook
while removing Colab-specific behavior and making paths configurable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required for landmark extraction. Install it with 'pip install mediapipe'"
    ) from exc

if not hasattr(mp, "solutions"):
    try:
        from mediapipe.python import solutions as mp_solutions
        mp.solutions = mp_solutions
    except Exception:  # pragma: no cover
        mp.solutions = None


def initialize_mediapipe(
    static_image_mode: bool = False,
    model_complexity: int = 1,
    smooth_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Initialize and return a MediaPipe Holistic model."""
    if mp.solutions is None:
        raise ImportError("MediaPipe solutions could not be imported. Check the mediapipe installation.")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return holistic


def extract_landmarks(results) -> np.ndarray:
    """Convert MediaPipe Holistic results into a flat landmark feature vector."""
    pose = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 3)
    )

    face = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )

    left_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    right_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([pose, face, left_hand, right_hand])


def extract_video(video_path: str | os.PathLike, model) -> np.ndarray:
    """Extract a landmark sequence from a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.process(rgb)
        landmarks = extract_landmarks(results)
        sequence.append(landmarks)

    cap.release()

    return np.array(sequence)


def save_landmarks(sequence: np.ndarray, output_path: str | os.PathLike) -> None:
    """Save a landmark sequence as a .npy file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sequence)


def process_dataset(
    input_root: str | os.PathLike,
    output_root: str | os.PathLike,
    model=None,
    recursive: bool = True,
) -> list[Path]:
    """Process all videos under input_root and save .npy landmark files."""
    input_root = Path(input_root)
    output_root = Path(output_root)

    if model is None:
        model = initialize_mediapipe()

    saved_files: list[Path] = []

    video_files = sorted(
        p for p in input_root.rglob("*.mp4") if p.is_file()
    ) if recursive else sorted(
        p for p in input_root.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
    )

    for video_path in video_files:
        rel_path = video_path.relative_to(input_root)
        output_path = output_root / rel_path
        output_path = output_path.with_suffix(".npy")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sequence = extract_video(video_path, model)
        if sequence.size == 0:
            continue

        save_landmarks(sequence, output_path)
        saved_files.append(output_path)

    return saved_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks from videos")
    parser.add_argument("--input", type=str, required=True, help="Root directory containing raw videos")
    parser.add_argument("--output", type=str, required=True, help="Directory to write .npy landmark files")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan the input tree")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model = initialize_mediapipe()
    saved_files = process_dataset(args.input, args.output, model=model, recursive=args.recursive)
    print(f"Extracted {len(saved_files)} landmark files to {args.output}")


if __name__ == "__main__":
    main()
