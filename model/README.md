# Model package

This directory contains the stage-3 PyTorch implementation of MedicalSignFormer.

Files:
- `medicalsignformer.py`: full model assembly and entrypoint
- `embedding.py`: modality embedding layers
- `positional_encoding.py`: sinusoidal positional encoding
- `transformer_encoder.py`: cross-attention encoder blocks
- `medical_semantic_attention.py`: semantic attention head
- `classifier.py`: classification head
