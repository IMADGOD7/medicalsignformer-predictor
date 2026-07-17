"""
DataLoader configuration for MedicalSignFormer.

Handles batching, shuffling, and sequence padding via a custom collate_fn
since the .npy sequences can have varying number of frames.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import config

from .sign_dataset import MedicalSignDataset


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels, lengths = zip(*batch)
    
    # Record original lengths (useful for masking in transformer later)
    lengths = torch.stack(lengths)
    
    # Pad sequences (batch_first=True makes shape: [batch, seq_len, feature_dim])
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    labels_tensor = torch.stack(labels)
    
    return padded_features, labels_tensor, lengths


def get_dataloaders(data_dir,batch_size=config.BATCH_SIZE,num_workers=config.NUM_WORKERS,):

    data_dir = Path(data_dir)
    
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        csv_path = data_dir / f"{split}.csv"
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping {split} DataLoader.")
            continue
            
        dataset = MedicalSignDataset(csv_file=csv_path)
        
        # Only shuffle the training set
        shuffle = True if split == 'train' else False
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
    return dataloaders


if __name__ == "__main__":
    # Standalone verification script for Phase 2
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    
    print("Initializing DataLoaders...")
    loaders = get_dataloaders(processed_dir, batch_size=4)
    
    if 'train' in loaders:
        print(f"Total batches in train loader: {len(loaders['train'])}")
        
        # Fetch one batch
        features, labels, lengths = next(iter(loaders['train']))
        
        print("\n" + "="*40)
        print("DATALOADER VERIFICATION SUMMARY")
        print("="*40)
        print(f"Batch Size requested : 4")
        print(f"Padded Features Shape: {features.shape}  -> (batch_size, max_seq_len, features)")
        print(f"Labels Shape:          {labels.shape}  -> (batch_size)")
        print(f"Original Lengths:      {lengths.tolist()}  -> (batch_size)")
        print(f"Labels Content:        {labels.tolist()}")
        print("\nVerification Successful! Data Pipeline is ready.")
