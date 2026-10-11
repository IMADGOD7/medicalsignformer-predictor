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

try:
    from .motion_fusion import MotionFeatureFusion
except ImportError:
    from motion_fusion import MotionFeatureFusion


HAND_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_EDGES: list[tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22), (0, 11), (0, 12),
]


def build_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj.fill_diagonal_(1.0)
    return adj


class AdaptiveGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, anatomical_adjacency: torch.Tensor,
                 num_heads: int = config.GAT_NUM_HEADS, dropout: float = config.GAT_DROPOUT,
                 leaky_relu_slope: float = config.GAT_LEAKY_RELU_SLOPE):
        super().__init__()
        assert out_dim % num_heads == 0, f"out_dim ({out_dim}) not divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        self.register_buffer("anatomical_prior", anatomical_adjacency)
        self.learnable_edge_weight = nn.Parameter(anatomical_adjacency.clone())

        self.linear = nn.Linear(in_dim, out_dim)
        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        n_frames, num_nodes, _ = node_features.shape
        h = self.linear(node_features)
        h = h.view(n_frames, num_nodes, self.num_heads, self.head_dim)

        src_scores = torch.einsum("nihd,hd->nih", h, self.attn_src)
        dst_scores = torch.einsum("njhd,hd->njh", h, self.attn_dst)
        e = self.leaky_relu(src_scores.unsqueeze(2) + dst_scores.unsqueeze(1))
        e = e + self.learnable_edge_weight.unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(self.anatomical_prior.unsqueeze(0).unsqueeze(-1) == 0, float("-inf"))

        attention = F.softmax(e, dim=2)
        attention = self.dropout(attention)
        out = torch.einsum("nijh,njhd->nihd", attention, h)
        out = out.reshape(n_frames, num_nodes, self.out_dim)
        return self.norm(out)


class GraphBranch(nn.Module):
    def __init__(self, num_nodes: int, anatomical_edges: list[tuple[int, int]],
                 node_input_dim: int = config.LANDMARK_COORD_DIM,
                 hidden_dim: int = config.GRAPH_HIDDEN_DIM,
                 num_layers: int = config.GAT_NUM_LAYERS,
                 dropout: float = config.GAT_DROPOUT):
        super().__init__()
        self.num_nodes = num_nodes
        adjacency = build_adjacency(num_nodes, anatomical_edges)

        layers = []
        in_dim = node_input_dim
        for _ in range(num_layers):
            layers.append(AdaptiveGraphAttentionLayer(
                in_dim=in_dim, out_dim=hidden_dim, anatomical_adjacency=adjacency,
            ))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        nodes = x.reshape(batch_size * seq_len, self.num_nodes, config.LANDMARK_COORD_DIM)
        for layer in self.layers:
            nodes = layer(nodes)
        pooled = nodes.mean(dim=1)
        return pooled.view(batch_size, seq_len, -1)


class FaceEncoder(nn.Module):
    def __init__(self, input_dim: int = config.FACE_DIM,
                 hidden_dim: int = config.FACE_ENCODER_HIDDEN_DIM,
                 num_heads: int = config.FACE_ENCODER_NUM_HEADS,
                 dropout: float = config.FACE_ENCODER_DROPOUT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.mlp(x)
        attended, _ = self.self_attention(h, h, h, key_padding_mask=key_padding_mask)
        return self.norm(h + attended)


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

        self.left_hand_branch = GraphBranch(num_nodes=config.NUM_HAND_LANDMARKS, anatomical_edges=HAND_EDGES)
        self.right_hand_branch = GraphBranch(num_nodes=config.NUM_HAND_LANDMARKS, anatomical_edges=HAND_EDGES)
        self.pose_branch = GraphBranch(num_nodes=config.NUM_POSE_LANDMARKS, anatomical_edges=POSE_EDGES)
        self.face_encoder = FaceEncoder()
        self.hand_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.size(-1) != config.INPUT_DIM:
            raise ValueError(f"Expected (B, T, {config.INPUT_DIM}), got {tuple(x.shape)}.")

        pose_raw = x[:, :, self._pose_slice]
        face_raw = x[:, :, self._face_slice]
        left_hand_raw = x[:, :, self._left_hand_slice]
        right_hand_raw = x[:, :, self._right_hand_slice]

        left_hand_features = self.left_hand_branch(left_hand_raw)
        right_hand_features = self.right_hand_branch(right_hand_raw)
        hand_features = self.hand_fusion(torch.cat([left_hand_features, right_hand_features], dim=-1))

        pose_features = self.pose_branch(pose_raw)
        face_features = self.face_encoder(face_raw, key_padding_mask=key_padding_mask)

        return hand_features, pose_features, face_features


class GatedModalityCombiner(nn.Module):
    def __init__(self, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        nn.init.xavier_uniform_(self.gate_proj[0].weight)
        nn.init.zeros_(self.gate_proj[0].bias)
        nn.init.xavier_uniform_(self.gate_proj[3].weight)
        nn.init.zeros_(self.gate_proj[3].bias)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, hand: torch.Tensor, pose: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([hand, pose, face], dim=-1)
        gates = torch.softmax(self.gate_proj(combined), dim=-1)
        gated = gates[..., 0:1] * hand + gates[..., 1:2] * pose + gates[..., 2:3] * face
        return self.norm(gated)


class GraphAwareEncoder(nn.Module):
    def __init__(self, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        self.graph_attention_network = AdaptiveGraphAttentionNetwork(embed_dim=embed_dim)
        self.motion_fusion = MotionFeatureFusion(embed_dim=embed_dim)
        self.modality_combiner = GatedModalityCombiner(embed_dim=embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hand_f, pose_f, face_f = self.graph_attention_network(x, key_padding_mask=key_padding_mask)
        hand_f, pose_f, face_f = self.motion_fusion(x, hand_f, pose_f, face_f)
        return self.modality_combiner(hand_f, pose_f, face_f)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, config.SEQUENCE_LENGTH
    D = config.EMBED_DIM

    x = torch.randn(B, T, config.INPUT_DIM)
    pad = torch.arange(T).unsqueeze(0) >= torch.tensor([T, T - 20]).unsqueeze(1)

    enc = GraphAwareEncoder(embed_dim=D)
    out = enc(x, key_padding_mask=pad)
    assert out.shape == (B, T, D)
    out.sum().backward()
    assert all(g.grad is not None for g in enc.parameters() if g.requires_grad)
    print("GraphAwareEncoder: VERIFIED")
