from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config

POSE_END = 99
FACE_END = 1503
LEFT_HAND_END = 1566
RIGHT_HAND_END = 1629


def normalize_skeleton(features: np.ndarray) -> np.ndarray:
    """Wrist/shoulder-centered normalization with FIXED reference frame.

    Uses the first valid frame's shoulder midpoint + distance as reference
    for ALL frames. Preserves absolute position while removing camera-distance
    variation. Previous per-frame normalization destroyed velocity signals.

    Args:
        features: (1629,) or (N, 1629) landmark vector.
            POSE(0:99) | FACE(99:1503) | LEFT_HAND(1503:1566) | RIGHT_HAND(1566:1629)
    Returns:
        Normalized array, same shape as input.
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    pose = features[:, :POSE_END].reshape(-1, 33, 3)
    left_shoulder = pose[:, 11]
    right_shoulder = pose[:, 12]
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0

    valid = (
        (left_shoulder[:, 0] > 0) & (right_shoulder[:, 0] > 0) &
        (left_shoulder[:, 1] > 0) & (right_shoulder[:, 1] > 0)
    )

    if not valid.any():
        return features.reshape(-1, 1629).squeeze(0) if squeeze else features

    ref_mid = shoulder_mid[valid][0]
    ref_dist = np.linalg.norm(left_shoulder[valid][0] - right_shoulder[valid][0])
    ref_dist = max(ref_dist, 1e-6)

    out = features.copy()
    out[:, :POSE_END] = (out[:, :POSE_END] - ref_mid.repeat(33)) / ref_dist
    out[:, POSE_END:FACE_END] = (out[:, POSE_END:FACE_END] - ref_mid.repeat(468)) / ref_dist
    out[:, LEFT_HAND_END:RIGHT_HAND_END] = (out[:, LEFT_HAND_END:RIGHT_HAND_END] - ref_mid.repeat(21)) / ref_dist

    return out.reshape(-1, 1629).squeeze(0) if squeeze else out


def normalize_skeleton_tensor(tensor: torch.Tensor) -> torch.Tensor:
    device = tensor.device
    dtype = tensor.dtype
    norm_arr = normalize_skeleton(tensor.detach().cpu().numpy())
    return torch.tensor(norm_arr, dtype=dtype, device=device)


if __name__ == "__main__":
    np.random.seed(0)
    d1 = np.random.randn().astype(np.float32)
    d2 = np.random.randn(100, 1629).astype(np.float32)
    assert normalize_skeleton(d1).shape == (1629,)
    assert normalize_skeleton(d2).shape == (100, 1629)
    print("skeleton_normalization: VERIFIED")
