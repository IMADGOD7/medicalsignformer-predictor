# MedicalSignFormer

A Transformer-state-space framework for **medical sign recognition** from video-derived **MediaPipe Holistic landmarks**. The model combines graph attention, motion-feature fusion, a Mamba temporal encoder, temporal attention pooling, and a classification head. Training uses contrastive pretraining followed by supervised fine-tuning.

---

# Overview

MedicalSignFormer is a two-stage deep learning framework designed for medical sign language recognition.

The framework first learns landmark-sequence representations through **self-supervised contrastive pretraining** on augmented views, followed by **supervised fine-tuning** for 48-class sign classification.

The complete pipeline consists of:

```
Raw Videos
      │
      ▼
MediaPipe Holistic
      │
      ▼
Landmark Extraction
      │
      ▼
Graph Attention Network
      │
      ▼
Motion Feature Fusion
      │
      ▼
Graph-aware Encoder
      │
      ▼
Mamba Temporal Encoder
      │
      ▼
Temporal Attention Pooling
      │
      ▼
Classification Head
      │
      ▼
Disease Prediction
      │
      ▼
Monte Carlo Dropout
```

---

# Features

- Adaptive Graph Attention Network
- Motion Feature Fusion
- Contrastive self-supervised pretraining
- Mamba Temporal Encoder
- Temporal Attention Pooling
- Monte Carlo Dropout Uncertainty Estimation
- Self-Supervised Pretraining
- End-to-End Fine-Tuning
- Attention-based Sequence Modeling
- Class-balanced Training
- Early Stopping
- Learning Rate Scheduling
- Automatic Checkpointing
- Comprehensive Evaluation Pipeline

---

# Project Structure

```
medicalsignformer-predictor/
│
├── data/
│   └── processed/
│
├── landmarks/
│
├── preprocessing/
│
├── dataset/
│
├── model/
│
├── checkpoints/
│
├── evaluation/
│
├── config.py
├── train_pretrain.py
├── train_finetune.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── deployment/
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/IMADGOD7/medicalsignformer-predictor.git

cd medicalsignformer-predictor
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

The model also requires **PyTorch**, which is not included in `requirements.txt`.
Install the PyTorch build appropriate for your operating system and CPU/GPU from
the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
The optional computational profiling script `computational_comparison.py` also
requires `thop`.

---

# Dataset

The project uses MediaPipe Holistic landmarks extracted from medical sign language videos.

Each processed sample consists of

```
Sequence Length × 1629
```

where 1,629 features correspond to 543 MediaPipe landmarks across 3 spatial coordinates (x, y, z):

```
1629 features = 543 landmarks × 3 coordinates (x, y, z)
  = (468 Face Landmarks
   + 33 Pose Landmarks
   + 21 Left Hand Landmarks
   + 21 Right Hand Landmarks) × 3
```

All sequences are normalized to the same temporal length before training.

---

# Training Pipeline

The framework follows a two-stage training strategy.

---

## Stage 1 — Contrastive Self-Supervised Pretraining

The encoder is trained without using disease labels. Two augmented views of each
landmark sequence are created using time warping, small noise, and temporal
dropout; a contrastive loss brings representations of the paired views together.

Run:

```bash
python train_pretrain.py
```

Output:

```
checkpoints/

└── pretrained_encoder.pth
```

---

## Stage 2 — Supervised Fine-Tuning

Loads the pretrained encoder and trains the complete network for disease classification.

Run:

```bash
python train_finetune.py
```

Output:

```
checkpoints/

└── best_model.pth
```

---

# Evaluation

Evaluate the trained model on the test dataset.

```bash
python evaluate.py
```

Generated outputs:

```
evaluation/

classification_report.csv

confusion_matrix.png

mc_dropout_predictions.csv
```

---

# Model Architecture

```
MediaPipe Holistic Landmarks
              │
              ▼
Adaptive Graph Attention Network
              │
              ▼
Motion Feature Fusion
              │
              ▼
Graph-aware Encoder
              │
              ▼
Mamba Temporal Encoder
              │
              ▼
Temporal Attention Pooling
              │
              ▼
Classification Head
              │
              ▼
Disease Prediction
              │
              ▼
Monte Carlo Dropout
```

---

# Local Webcam Demo

The deployment entry point runs a local OpenCV webcam window; it is not a hosted
web application. From the project root, connect a webcam and make the trained
checkpoint available at `checkpoints/best_model.pth`. Class names are loaded
from `data/processed/label_map.json`.

Windows:

```powershell
.venv\Scripts\python.exe deployment\app.py
```

Linux/macOS:

```bash
python deployment/app.py
```

Press `q` or `Esc` in the display window to exit. The checkpoint is ignored by
Git, so distribute it separately and only where you have permission to share it.

---

# Evaluation Notes

- **Main Test Performance:** The baseline evaluation report records **95.56% accuracy** and **0.951 macro-F1** on a 1,396-sample test split (evaluated locally under `evaluation/classification_report.csv`).
- **Active Dataset Partitions:** The current active processed dataset (`data/processed/`) contains 6,979 total landmark samples across 48 sign vocabulary classes, partitioned into:
  - **Train:** 5,583 samples (80%)
  - **Validation:** 698 samples (10%)
  - **Test:** 698 samples (10%)
- **Signer Overlap:** All 18 signers appear across training, validation, and test splits. Therefore, this split evaluates gesture recognition across known signers and does not establish performance on unseen signers.
- **Cross-Dataset Generalization:** External evaluation on the mapped MSL/Kaggle dataset (`evaluation_kaggle/`) yields ~9–11% accuracy on 76 samples due to domain shifts and semantic label differences.
- **Artifact Tracking:** Model weights (`.pth`), extracted landmark arrays (`.npy`), and evaluation reports (`evaluation/*.csv`) are ignored by Git via `.gitignore` to keep the repository lightweight. Checkpoints must be generated or provided locally.
- **Medical Disclaimer:** This project is a research prototype for sign vocabulary classification, not a clinical medical diagnostic system.

---

# Configuration

All hyperparameters are centralized inside

```
config.py
```

including

- model architecture
- optimizer
- scheduler
- learning rates
- dropout
- label smoothing
- batch size
- training epochs
- Monte Carlo Dropout
- early stopping

---

# Outputs

## Stage 1

```
checkpoints/

pretrained_encoder.pth
```

---

## Stage 2

```
checkpoints/

best_model.pth
```

*(Note: scripts standardize on `best_model.pth`; standalone profiling or analysis scripts may also look for `best_finetuned_model.pth`)*

---

## Evaluation

```
evaluation/

classification_report.csv

confusion_matrix.png

mc_dropout_predictions.csv
```

---

# Requirements

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

# Citation

If you use this project in your research, please cite the repository.

---

# License

This project is released under the MIT License.

---

# Author

**team**

Computer Science & Engineering

MedicalSignFormer Research Project
