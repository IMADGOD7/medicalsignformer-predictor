# MedicalSignFormer

A Transformer-State Space based framework for **Medical Sign Language Recognition** using **MediaPipe Holistic landmarks**, **Adaptive Graph Attention Networks**, **Motion Feature Fusion**, **Graph-aware Masked Temporal Pretraining**, **Mamba Temporal Encoder**, **Temporal Attention Pooling**, and **Monte Carlo Dropout Uncertainty Estimation**.

---

# Overview

MedicalSignFormer is a two-stage deep learning framework designed for medical sign language recognition.

The framework first learns general spatio-temporal landmark representations through **self-supervised masked pretraining**, followed by **supervised end-to-end fine-tuning** for disease classification.

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
- Graph-aware Masked Temporal Pretraining
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
MedicalSignFormerV2/
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
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/IMADGOD7/medicalsignformer-predictor.git

cd MedicalSignFormerV2
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

---

# Dataset

The project uses MediaPipe Holistic landmarks extracted from medical sign language videos.

Each processed sample consists of

```
Sequence Length × 1629
```

where

```
1629 =
468 Face Landmarks
+
33 Pose Landmarks
+
21 Left Hand Landmarks
+
21 Right Hand Landmarks
```

All sequences are normalized to the same temporal length before training.

---

# Training Pipeline

The framework follows a two-stage training strategy.

---

## Stage 1 — Self-Supervised Pretraining

The encoder is trained without using disease labels.

Masked temporal segments are reconstructed to learn meaningful spatial-temporal representations.

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

└── best_finetuned_model.pth
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

best_finetuned_model.pth
```

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
