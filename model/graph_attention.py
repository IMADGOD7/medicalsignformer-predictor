
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

try:
    import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


# --------------------------------------------------------------------------
# Anatomical adjacency definitions (MediaPipe standard connectivity)
# --------------------------------------------------------------------------

# Standard MediaPipe Hands connectivity (21 landmarks, indices 0-20).
# Matches mediapipe.solutions.hands.HAND_CONNECTIONS.
HAND_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # wrist -> pinky base (palm)
]


POSE_EDGES: list[tuple[int, int]] = [
    (11, 12),                 # shoulder to shoulder
    (11, 13), (13, 15),       # left arm: shoulder -> elbow -> wrist
    (12, 14), (14, 16),       # right arm: shoulder -> elbow -> wrist
    (11, 23), (12, 24),       # shoulder -> hip (torso orientation)
    (23, 24),                 # hip to hip
    (15, 17), (15, 19), (15, 21),  # left wrist -> hand reference points
    (16, 18), (16, 20), (16, 22),  # right wrist -> hand reference points
    (0, 11), (0, 12),         # nose -> shoulders (head/torso orientation)
]


def build_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:

    adj = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj.fill_diagonal_(1.0)  # self-loops, standard in GAT/GCN
    return adj


# --------------------------------------------------------------------------
# Adaptive Graph Attention layer
# --------------------------------------------------------------------------

class AdaptiveGraphAttentionLayer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        anatomical_adjacency: torch.Tensor,
        num_heads: int = config.GAT_NUM_HEADS,
        dropout: float = config.GAT_DROPOUT,
        leaky_relu_slope: float = config.GAT_LEAKY_RELU_SLOPE,
    ):
        super().__init__()
        assert out_dim % num_heads == 0, (
            f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        self.register_buffer("anatomical_prior", anatomical_adjacency)
        self.learnable_edge_weight = nn.Parameter(anatomical_adjacency.clone())

        self.linear = nn.Linear(in_dim, out_dim)

        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:

        n_frames, num_nodes, _ = node_features.shape

        h = self.linear(node_features)  # (N, num_nodes, out_dim)
        h = h.view(n_frames, num_nodes, self.num_heads, self.head_dim)

        # Per-head attention scores: e_ij = LeakyReLU(a_src . h_i + a_dst . h_j)
        src_scores = torch.einsum("nihd,hd->nih", h, self.attn_src)  # (N, nodes, heads)
        dst_scores = torch.einsum("njhd,hd->njh", h, self.attn_dst)  # (N, nodes, heads)
        # Broadcast to (N, nodes_i, nodes_j, heads)
        e = src_scores.unsqueeze(2) + dst_scores.unsqueeze(1)
        e = self.leaky_relu(e)

        edge_bias = self.learnable_edge_weight.unsqueeze(0).unsqueeze(-1)  # (1, nodes, nodes, 1)
        e = e + edge_bias


        mask = self.anatomical_prior.unsqueeze(0).unsqueeze(-1) == 0  # (1, nodes, nodes, 1)
        e = e.masked_fill(mask, float("-inf"))

        attention = F.softmax(e, dim=2)  # normalize over source nodes j, per target i

        # Weighted aggregation: out_i = sum_j attention_ij * h_j, per head.
        out = torch.einsum("nijh,njhd->nihd", attention, h)
        out = out.reshape(n_frames, num_nodes, self.out_dim)

        out = self.norm(out)
        return out


class GraphBranch(nn.Module):


    def __init__(
        self,
        num_nodes: int,
        anatomical_edges: list[tuple[int, int]],
        node_input_dim: int = config.LANDMARK_COORD_DIM,
        hidden_dim: int = config.GRAPH_HIDDEN_DIM,
        num_layers: int = config.GAT_NUM_LAYERS,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        adjacency = build_adjacency(num_nodes, anatomical_edges)

        layers = []
        in_dim = node_input_dim
        for _ in range(num_layers):
            layers.append(
                AdaptiveGraphAttentionLayer(
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                    anatomical_adjacency=adjacency,
                )
            )
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape
        # Reshape to per-frame node features: (B*T, num_nodes, 3)
        nodes = x.reshape(batch_size * seq_len, self.num_nodes, config.LANDMARK_COORD_DIM)

        for layer in self.layers:
            nodes = layer(nodes)

        pooled = nodes.mean(dim=1)  # (B*T, hidden_dim)
        return pooled.view(batch_size, seq_len, -1)


# --------------------------------------------------------------------------
# Face Encoder (MLP + self-attention, NOT a graph)
# --------------------------------------------------------------------------

class FaceEncoder(nn.Module):


    def __init__(
        self,
        input_dim: int = config.FACE_DIM,
        hidden_dim: int = config.FACE_ENCODER_HIDDEN_DIM,
        num_heads: int = config.FACE_ENCODER_NUM_HEADS,
        dropout: float = config.FACE_ENCODER_DROPOUT,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.mlp(x)
        attended, _ = self.self_attention(h, h, h, key_padding_mask=key_padding_mask)
        return self.norm(h + attended)


# --------------------------------------------------------------------------
# Top-level module
# --------------------------------------------------------------------------

class AdaptiveGraphAttentionNetwork(nn.Module):


    def __init__(self, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        self._pose_slice = slice(0, config.POSE_DIM)
        self._face_slice = slice(config.POSE_DIM, config.POSE_DIM + config.FACE_DIM)
        self._left_hand_slice = slice(
            config.POSE_DIM + config.FACE_DIM,
            config.POSE_DIM + config.FACE_DIM + config.LEFT_HAND_DIM,
        )
        self._right_hand_slice = slice(
            config.POSE_DIM + config.FACE_DIM + config.LEFT_HAND_DIM,
            config.POSE_DIM + config.FACE_DIM + config.LEFT_HAND_DIM + config.RIGHT_HAND_DIM,
        )

        self.left_hand_branch = GraphBranch(
            num_nodes=config.NUM_HAND_LANDMARKS, anatomical_edges=HAND_EDGES
        )
        self.right_hand_branch = GraphBranch(
            num_nodes=config.NUM_HAND_LANDMARKS, anatomical_edges=HAND_EDGES
        )
        self.pose_branch = GraphBranch(
            num_nodes=config.NUM_POSE_LANDMARKS, anatomical_edges=POSE_EDGES
        )
        self.face_encoder = FaceEncoder()

        # Learnable fusion of left+right hand branch outputs into one
        # hand_features tensor - mirrors the existing project's hand_fusion
        # pattern (concat -> Linear -> LayerNorm -> GELU).
        self.hand_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if x.ndim != 3 or x.size(-1) != config.INPUT_DIM:
            raise ValueError(
                f"Expected input shape (B, T, {config.INPUT_DIM}), got {tuple(x.shape)}."
            )

        pose_raw = x[:, :, self._pose_slice]
        face_raw = x[:, :, self._face_slice]
        left_hand_raw = x[:, :, self._left_hand_slice]
        right_hand_raw = x[:, :, self._right_hand_slice]

        left_hand_features = self.left_hand_branch(left_hand_raw)
        right_hand_features = self.right_hand_branch(right_hand_raw)
        hand_features = self.hand_fusion(
            torch.cat([left_hand_features, right_hand_features], dim=-1)
        )

        pose_features = self.pose_branch(pose_raw)
        face_features = self.face_encoder(face_raw, key_padding_mask=key_padding_mask)

        return hand_features, pose_features, face_features


if __name__ == "__main__":
    # --- Verification: shapes + gradient flow, per the one-module-at-a-time rule ---
    torch.manual_seed(0)
    batch_size = 2
    seq_len = config.SEQUENCE_LENGTH

    dummy_input = torch.randn(batch_size, seq_len, config.INPUT_DIM, requires_grad=False)
    lengths = torch.tensor([seq_len, seq_len - 20])
    position_ids = torch.arange(seq_len).unsqueeze(0)
    padding_mask = position_ids >= lengths.unsqueeze(1)

    model = AdaptiveGraphAttentionNetwork(embed_dim=config.EMBED_DIM)
    hand_features, pose_features, face_features = model(dummy_input, key_padding_mask=padding_mask)

    print("hand_features:", hand_features.shape)
    print("pose_features:", pose_features.shape)
    print("face_features:", face_features.shape)

    expected_shape = (batch_size, seq_len, config.EMBED_DIM)
    assert hand_features.shape == expected_shape, "hand_features shape mismatch"
    assert pose_features.shape == expected_shape, "pose_features shape mismatch"
    assert face_features.shape == expected_shape, "face_features shape mismatch"
    print("\nShape verification: PASSED")

    # Gradient flow check: sum outputs into a scalar loss, backward, confirm
    # every parameter that should receive a gradient does.
    loss = hand_features.sum() + pose_features.sum() + face_features.sum()
    loss.backward()

    missing_grad = [
        name for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    if missing_grad:
        print(f"\nWARNING: {len(missing_grad)} parameter(s) received NO gradient:")
        for name in missing_grad:
            print(f"  - {name}")
    else:
        print("Gradient flow verification: PASSED (all parameters received gradients)")

    # Confirm the anatomical adjacency buffers are NOT being trained (fixed
    # reference prior), while the learnable edge weights ARE.
    sample_layer = model.left_hand_branch.layers[0]
    print(
        f"\nanatomical_prior.requires_grad (should be False): "
        f"{sample_layer.anatomical_prior.requires_grad}"
    )
    print(
        f"learnable_edge_weight.requires_grad (should be True): "
        f"{sample_layer.learnable_edge_weight.requires_grad}"
    )