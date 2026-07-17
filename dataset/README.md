# Dataset package

This directory contains dataset indexing and PyTorch dataset utilities.

Files:
- `build_dataset_index.py`: build `dataset_index.csv` and `label_map.json`
- `sign_dataset.py`: `MedicalSignDataset` returning fixed-length tensors
- `dataloader.py`: `get_dataloaders()` and batch collate logic
