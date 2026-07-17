"""
PyTorch Dataset implementation for MedicalSignFormer.

Loads MediaPipe landmark sequences from .npy files and provides 
tensors for the DataLoader.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from ..config import SEQUENCE_LENGTH
except ImportError:
    from config import SEQUENCE_LENGTH

class MedicalSignDataset(Dataset):
    """
    Custom Dataset for loading medical sign language landmark sequences.
    """
    def __init__(self, csv_file: str | Path, root_dir: str | Path = None, sequence_length: int = SEQUENCE_LENGTH):
        """
        Args:
            csv_file (str or Path): Path to the split CSV file (train, val, test).
            root_dir (str or Path, optional): Project root for resolving relative paths.
            sequence_length (int): Fixed number of frames per sample.
        """
        self.csv_path = Path(csv_file)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
            
        self.root_dir = Path(root_dir) if root_dir else self.csv_path.parent.parent.parent
        self.data_info = pd.read_csv(self.csv_path)
        self.sequence_length = sequence_length
        
    def __len__(self) -> int:
        return len(self.data_info)
        
    def _pad_or_trim(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError(f"Expected feature array with 2 dimensions, got {features.ndim}.")

        num_frames = features.shape[0]
        if num_frames == self.sequence_length:
            return features
        if num_frames > self.sequence_length:
            return features[: self.sequence_length, :]

        pad_len = self.sequence_length - num_frames
        padding = np.zeros((pad_len, features.shape[1]), dtype=features.dtype)
        return np.vstack([features, padding])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data_info.iloc[idx]
        
        # Load the numpy array containing landmark features
        npy_path = self.root_dir / row['filepath']
        try:
            features = np.load(npy_path)
            original_length = min(features.shape[0], self.sequence_length)
        except Exception as e:
            raise RuntimeError(f"Error loading {npy_path}: {e}")
            
        # Sanitize NaNs from MediaPipe failure frames
        features = np.nan_to_num(features, nan=0.0)
        
        # Enforce fixed sequence length for Stage 3 compatibility
        features = self._pad_or_trim(features)
        
        # Convert features to a float32 tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Convert label to a long tensor (standard for PyTorch classification)
        label_tensor = torch.tensor(row['label'], dtype=torch.long)

        length_tensor = torch.tensor(original_length, dtype=torch.long)
        
        return features_tensor, label_tensor, length_tensor
