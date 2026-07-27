from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import json
import sys

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

    case_normalized = raw.strip().lower()
    return MANUAL_SPELLING_FIXES.get(case_normalized, case_normalized)

def get_disease_folders(dataset_path: Path) -> List[Path]:
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{dataset_path}' does not exist.")
        sys.exit(1)
        
    # Get all subdirectories and sort alphabetically for consistent label mapping
    disease_folders = [p for p in dataset_path.iterdir() if p.is_dir()]
    disease_folders.sort(key=lambda p: p.name)
    return disease_folders

def generate_label_mapping(disease_folders: List[Path]) -> Dict[str, int]:
    return {folder.name: idx for idx, folder in enumerate(disease_folders)}

def scan_dataset(dataset_path: Path, label_mapping: Dict[str, int]) -> List[Dict[str, Any]]:
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