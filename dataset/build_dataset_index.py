"""
Dataset index builder for MedicalSignFormer.

This module scans the landmarks directory and generates a master CSV file 
describing every sample. It automatically generates integer labels based on
alphabetical sorting of the disease folders.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import json
import sys

# --------------------------------------------------------------------------
# Signer name normalization
# --------------------------------------------------------------------------
# The raw signer folder names on disk contain case-only duplicates (e.g.
# "AKASH" and "Akash" are the same person recorded under two differently-
# cased folder names), which previously caused dataset_index.csv to report
# 29 unique signers instead of the real 18. This normalization is applied
# HERE, at index-build time, rather than as a separate patch script - a
# separate patch (see merge_signer_folders.py's docstring for the prior
# failed attempt at this via normalize_signer_names.py) gets silently
# reverted every time this script re-scans the folder names from scratch.
# Baking it in here means every regeneration (including via
# run_stage5_pipeline.py) permanently produces the correct 18 signers.
#
# NOTE: this normalizes the signer NAME recorded in the CSV only - it does
# NOT rename anything on disk. If you also want the actual landmarks/ (and
# ISL_MED Dataset/, if it has the same duplicates) folder names fixed on
# disk, that's a separate, deliberate step - see merge_signer_folders.py.
MANUAL_SPELLING_FIXES: dict[str, str] = {
    "harkapratim": "harkhapratim gogoi",
    "harkha pratim gogoi": "harkhapratim gogoi",
    "harkha p0ratim gogoi": "harkhapratim gogoi",
    "harkhapratim gogoi": "harkhapratim gogoi",

    "anandita": "anindita",
    "anindita": "anindita",

    "arinjit": "arinjit kataki",
    "arinjit kataki": "arinjit kataki",
}


def normalize_signer(raw: str) -> str:
    """Normalize a raw signer folder name to a canonical identity string.

    Applies case/whitespace normalization first, then the manual
    spelling-fix map for known typo variants that differ by more than just
    case. Keep this in sync with merge_signer_folders.py's normalize_signer()
    if that script is ever used to also fix folder names on disk - they must
    agree, or the CSV's signer identities and the actual folder names will
    drift apart again.
    """
    case_normalized = raw.strip().lower()
    return MANUAL_SPELLING_FIXES.get(case_normalized, case_normalized)

def get_disease_folders(dataset_path: Path) -> List[Path]:
    """
    Retrieve and sort all valid disease directories in the dataset.
    
    Args:
        dataset_path (Path): Path to the root dataset directory containing landmarks.
        
    Returns:
        List[Path]: A list of paths to valid disease directories, sorted alphabetically.
    """
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{dataset_path}' does not exist.")
        sys.exit(1)
        
    # Get all subdirectories and sort alphabetically for consistent label mapping
    disease_folders = [p for p in dataset_path.iterdir() if p.is_dir()]
    disease_folders.sort(key=lambda p: p.name)
    return disease_folders

def generate_label_mapping(disease_folders: List[Path]) -> Dict[str, int]:
    """
    Generate an integer label mapping from sorted disease names.
    
    Args:
        disease_folders (List[Path]): A sorted list of disease directory paths.
        
    Returns:
        Dict[str, int]: Mapping from disease name to integer label (0-indexed).
    """
    return {folder.name: idx for idx, folder in enumerate(disease_folders)}

def scan_dataset(dataset_path: Path, label_mapping: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Scan the dataset structure and gather information for all samples.
    
    Args:
        dataset_path (Path): Root dataset path containing disease folders.
        label_mapping (Dict[str, int]): Mapping of diseases to their integer labels.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dict represents a single sample.
    """
    samples = []
    
    for disease_folder in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
        disease_name = disease_folder.name
        label = label_mapping[disease_name]
        
        for signer_folder in sorted(p for p in disease_folder.iterdir() if p.is_dir()):
            signer_name = normalize_signer(signer_folder.name)
            
            for file_path in sorted(signer_folder.glob("*.npy")):
                samples.append({
                    "filepath": file_path.relative_to(dataset_path.parent).as_posix(),
                    "disease": disease_name,
                    "signer": signer_name,
                    "label": label
                })
                
    return samples

def build_dataset_index(
    dataset_dir: str = "../landmarks", 
    output_csv: str = "../dataset_index.csv"
) -> None:
    """
    Main function to scan the dataset and build the CSV index.
    
    Args:
        dataset_dir (str): Relative or absolute path to the landmarks directory.
        output_csv (str): Path to save the generated CSV file.
    """
    dataset_path = Path(dataset_dir).resolve()
    
    disease_folders = get_disease_folders(dataset_path)
    if not disease_folders:
        print(f"Error: No disease folders found in '{dataset_path}'.")
        sys.exit(1)
        
    label_mapping = generate_label_mapping(disease_folders)
    samples = scan_dataset(dataset_path, label_mapping)
    
    if not samples:
        print("Warning: No .npy files were found in the dataset structure.")
        sys.exit(1)
        
    df = pd.DataFrame(samples)
    
    output_path = Path(output_csv).resolve()
    try:
        df.to_csv(output_path, index=False)
        
        # Save label_map.json
        label_map_path = output_path.parent / "data" / "processed" / "label_map.json"
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_map_path, "w") as f:
            json.dump(label_mapping, f, indent=4)
            
    except Exception as e:
        print(f"Error saving files: {e}")
        sys.exit(1)
        
    # Output required summary
    print("\n" + "="*50)
    print("DATASET INDEX SUMMARY")
    print("="*50)
    print(f"Total Samples     : {len(df)}")
    print(f"Number of Classes : {len(label_mapping)}")
    print(f"Unique Signers    : {df['signer'].nunique()} (after normalization)")
    
    print("\nLabel Mapping:")
    for disease, label in label_mapping.items():
        print(f"  {disease} -> {label}")
        
    print("\nPreview of the first five rows:")
    print(df.head().to_string())

if __name__ == "__main__":
    # Point dataset_dir to the actual workspace landmarks structure
    # and output the CSV to the root of the workspace.
    build_dataset_index(
        dataset_dir=Path(__file__).parent.parent / "landmarks",
        output_csv=Path(__file__).parent.parent / "dataset_index.csv"
    )