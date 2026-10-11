from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config import SEQUENCE_LENGTH
except ImportError:
    SEQUENCE_LENGTH = 100

from preprocessing.skeleton_normalization import normalize_skeleton


class MedicalSignDataset(Dataset):
    def __init__(self, csv_file: str | Path, root_dir: str | Path = None,
                 sequence_length: int = SEQUENCE_LENGTH, is_train: bool = False, normalize: bool = True):
        self.csv_path = Path(csv_file)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        self.root_dir = Path(root_dir) if root_dir else self.csv_path.parent.parent.parent
        self.data_info = pd.read_csv(self.csv_path)
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.data_info)

    def _augment(self, features: np.ndarray) -> np.ndarray:
        augmented = features.copy()
        scale = np.random.uniform(0.90, 1.10)
        shift = np.random.uniform(-0.02, 0.02)
        augmented = augmented * scale + shift
        if np.random.rand() < 0.5:
            augmented = augmented * np.where(np.arange(augmented.shape[1]) % 3 == 0, -1.0, 1.0)
        if np.random.rand() < 0.5:
            augmented = augmented + np.random.randn(*augmented.shape).astype(np.float32) * 0.005
        return augmented

    def _pad_or_trim(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array, got {features.ndim}D")
        if features.shape[0] == self.sequence_length:
            return features
        if features.shape[0] > self.sequence_length:
            return features[:self.sequence_length, :]
        pad = np.zeros((self.sequence_length - features.shape[0], features.shape[1]), dtype=features.dtype)
        return np.vstack([features, pad])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.data_info.iloc[idx]
        npy_path = self.root_dir / row["filepath"]

        features = np.load(npy_path)
        original_length = min(features.shape[0], self.sequence_length)
        features = np.nan_to_num(features, nan=0.0)

        if self.normalize:
            features = normalize_skeleton(features)

        if self.is_train:
            features = self._augment(features)

        features = self._pad_or_trim(features)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(row["label"], dtype=torch.long),
            torch.tensor(original_length, dtype=torch.long),
        )
