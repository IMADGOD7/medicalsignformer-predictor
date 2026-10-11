from __future__ import annotations

import math
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
    from mamba_ssm import Mamba as _OfficialMamba
    USING_OFFICIAL_MAMBA = True
except ImportError:
    _OfficialMamba = None
    USING_OFFICIAL_MAMBA = False


class _SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = config.MAMBA_STATE_DIM,
                 d_conv: int = config.MAMBA_CONV_KERNEL, expand: int = config.MAMBA_EXPAND_FACTOR):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x_branch, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_branch.transpose(1, 2))[:, :, :seq_len]).transpose(1, 2)

        delta_raw, B_param, C_param = torch.split(self.x_proj(x_conv), [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log)

        A_bar = torch.exp(torch.einsum("btd,dn->btdn", delta, A))
        B_bar_x = torch.einsum("btd,btn,btd->btdn", delta, B_param, x_conv)

        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar_x[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, C_param[:, t]))
        y = torch.stack(ys, dim=1)
        y = (y + x_conv * self.D) * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int = config.EMBED_DIM, d_state: int = config.MAMBA_STATE_DIM,
                 d_conv: int = config.MAMBA_CONV_KERNEL, expand: int = config.MAMBA_EXPAND_FACTOR,
                 dropout: float = config.MAMBA_DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = _OfficialMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) if USING_OFFICIAL_MAMBA else _SelectiveSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.ssm(self.norm(x))) + x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MambaTemporalEncoder(nn.Module):
    def __init__(self, d_model: int = config.EMBED_DIM, num_layers: int = config.MAMBA_NUM_LAYERS,
                 d_state: int = config.MAMBA_STATE_DIM, d_conv: int = config.MAMBA_CONV_KERNEL,
                 expand: int = config.MAMBA_EXPAND_FACTOR, dropout: float = config.MAMBA_DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"Expected (B, T, {self.d_model}), got {tuple(x.shape)}.")
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


if __name__ == "__main__":
    print(f"USING_OFFICIAL_MAMBA: {USING_OFFICIAL_MAMBA}")
    torch.manual_seed(0)
    model = MambaTemporalEncoder(d_model=config.EMBED_DIM)

    for T in (config.SEQUENCE_LENGTH, 37):
        x = torch.randn(2, T, config.EMBED_DIM, requires_grad=True)
        out = model(x)
        assert out.shape == (2, T, config.EMBED_DIM)
        out.sum().backward()
        assert x.grad is not None

    if not USING_OFFICIAL_MAMBA:
        with torch.no_grad():
            model.eval()
            base = torch.randn(1, 10, config.EMBED_DIM)
            perturbed = base.clone()
            perturbed[:, 7:, :] += torch.randn(1, 3, config.EMBED_DIM) * 50.0
            early = (model(base)[:, :7] - model(perturbed)[:, :7]).abs().max()
            late = (model(base)[:, 7:] - model(perturbed)[:, 7:]).abs().max()
        assert early < 1e-4, "Future frames leaked into past output"
        assert late > 1e-2

    print("MambaTemporalEncoder: VERIFIED")
