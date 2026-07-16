"""
Configuration for MedicalSignFormer v2 (new research architecture).

Extends the original project's config.py with hyperparameters for the new
pipeline: Adaptive Graph Attention Network, Motion Feature Fusion,
Graph-aware Masked Temporal Pretraining, Mamba Temporal Encoder, Temporal
Attention Pooling, and Monte Carlo Dropout. All new hyperparameters live
here - no hardcoded values in any module.
"""

import json
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
LABEL_MAP_PATH = ROOT / "data" / "processed" / "label_map.json"
DEFAULT_NUM_CLASSES = 48

try:
    with LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
        _label_map = json.load(f)
    NUM_CLASSES = len(_label_map)
    LABEL_MAP_AVAILABLE = True
except (FileNotFoundError, json.JSONDecodeError):
    NUM_CLASSES = DEFAULT_NUM_CLASSES
    LABEL_MAP_AVAILABLE = False
    warnings.warn(
        f"Could not load label map from {LABEL_MAP_PATH}. "
        f"Falling back to NUM_CLASSES={DEFAULT_NUM_CLASSES}. "
        "Run preprocessing to generate the label map.",
        RuntimeWarning,
        stacklevel=2,
    )

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Split ratios must sum to 1.0"

# --- Original / shared settings ---
EMBED_DIM = 256          # D: common feature dimension used throughout the new pipeline
SEQUENCE_LENGTH = 100    # T
DROPOUT_RATE = 0.3
INPUT_DIM = 1629
BATCH_SIZE = 32
NUM_WORKERS = 2

# --- True on-disk feature offsets (verified against extract_landmarks.py) ---
# concatenation order: pose, face, left_hand, right_hand
POSE_DIM = 99
FACE_DIM = 1404
LEFT_HAND_DIM = 63
RIGHT_HAND_DIM = 63
LANDMARK_COORD_DIM = 3       # x, y, z per landmark
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21      # per hand

# --- Module 2: Adaptive Graph Attention Network ---
GRAPH_HIDDEN_DIM = EMBED_DIM     # per-node feature dim inside GAT layers
GAT_NUM_HEADS = 4
GAT_NUM_LAYERS = 2
GAT_DROPOUT = 0.2
GAT_LEAKY_RELU_SLOPE = 0.2       # standard GAT attention nonlinearity slope

# --- Face Encoder (MLP + self-attention, no graph) ---
FACE_ENCODER_HIDDEN_DIM = EMBED_DIM
FACE_ENCODER_NUM_HEADS = 4
FACE_ENCODER_DROPOUT = 0.2

# --- Module 3: Motion Feature Fusion ---
MOTION_FEATURE_DIM = 3           # delta x, delta y, delta z per landmark, aggregated per modality
GATED_FUSION_DROPOUT = 0.1

# --- Module 4: Graph-aware Masked Temporal Pretraining ---
PRETRAIN_MASK_RATIO = 0.15
PRETRAIN_MIN_SPAN_LEN = 3
PRETRAIN_MAX_SPAN_LEN = 8
PRETRAIN_MAX_EPOCHS = 30
PRETRAIN_LEARNING_RATE = 1e-4
PRETRAIN_WEIGHT_DECAY = 1e-4

# --- Module 5: Mamba Temporal Encoder ---
MAMBA_NUM_LAYERS = 4
MAMBA_STATE_DIM = 16             # SSM state expansion dimension (d_state)
MAMBA_CONV_KERNEL = 4            # local conv kernel size (d_conv)
MAMBA_EXPAND_FACTOR = 2          # inner expansion factor (typical Mamba default)
MAMBA_DROPOUT = 0.2

# --- Module 6: Temporal Attention Pooling ---
TEMPORAL_POOLING_HIDDEN_DIM = EMBED_DIM // 2

# --- Module 7: Classification Head ---
CLASSIFIER_HIDDEN_DIM = EMBED_DIM // 2
CLASSIFIER_DROPOUT = 0.3

# --- Module 8: Monte Carlo Dropout ---
MC_DROPOUT_NUM_SAMPLES = 20      # stochastic forward passes at inference

# --- Fine-tuning (Stage 2) ---
FINETUNE_LEARNING_RATE = 1e-4
FINETUNE_WEIGHT_DECAY = 1e-3
FINETUNE_MAX_EPOCHS = 50
FINETUNE_EARLY_STOPPING_PATIENCE = 10