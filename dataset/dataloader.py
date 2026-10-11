from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import config

from .sign_dataset import MedicalSignDataset


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels, lengths = zip(*batch)
    lengths = torch.stack(lengths)
    features_tensor = torch.stack(features)
    labels_tensor = torch.stack(labels)
    return features_tensor, labels_tensor, lengths


def get_dataloaders(data_dir, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    data_dir = Path(data_dir)
    dataloaders = {}
    for split in ["train", "val", "test"]:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        is_train = split == "train"
        # pin_memory=True speeds up CPU→GPU transfer; safe with num_workers=0 on Windows
        pin_mem = torch.cuda.is_available()
        dataloaders[split] = DataLoader(
            MedicalSignDataset(csv_file=csv_path, is_train=is_train),
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_mem,
            persistent_workers=False,
        )
    return dataloaders


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    loaders = get_dataloaders(project_root / "data" / "processed", batch_size=4)

    if "train" in loaders:
        features, labels, lengths = next(iter(loaders["train"]))
        print(f"features: {tuple(features.shape)}  labels: {tuple(labels.shape)}  lengths: {tuple(lengths.shape)}")
        print("DataLoader: VERIFIED")
