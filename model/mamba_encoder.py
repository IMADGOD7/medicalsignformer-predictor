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
    from mamba_ssm import Mamba as _OfficialMamba
    USING_OFFICIAL_MAMBA = True
except ImportError:
    _OfficialMamba = None
    USING_OFFICIAL_MAMBA = False


class _SelectiveSSM(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = config.MAMBA_STATE_DIM,
        d_conv: int = config.MAMBA_CONV_KERNEL,
        expand: int = config.MAMBA_EXPAND_FACTOR,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model

        # Input projection: splits into the "main" branch (x) and the gate (z).
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Short causal depthwise conv over time, applied to the main branch,
        # standard in Mamba to give the SSM a small amount of local context
        # before the recurrence.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # Input-dependent (selective) SSM parameters: delta (step size), B, C
        # are all computed from the current input, not fixed - this is what
        # makes it "selective" rather than a plain fixed-parameter SSM/LTI.
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        # delta gets its own small projection + softplus, matching Mamba's
        # parameterization (delta must be positive - it's a discretization
        # step size).
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A is parameterized in log-space and kept strictly negative (real,
        # stable) via -exp(A_log), standard Mamba initialization (S4D-real
        # style): A_log initialized so A starts at -1, -2, ..., -d_state per
        # channel.
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))  # skip connection scalar per channel

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        batch_size, seq_len, _ = x.shape

        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Causal depthwise conv over time: (B, d_inner, T) -> conv -> trim
        # the extra right-padding introduced by `padding=d_conv-1` so the
        # output stays causal (no peeking at future frames) and the same
        # length as the input.
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :seq_len]
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, T, d_inner)

        # Selective parameters, all input-dependent.
        x_dbl = self.x_proj(x_conv)  # (B, T, 2*d_state + d_inner)
        delta_raw, B_param, C_param = torch.split(
            x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, T, d_inner), > 0

        A = -torch.exp(self.A_log)  # (d_inner, d_state), strictly negative

        # Discretize: A_bar = exp(delta * A), per timestep, per channel.
        # (B, T, d_inner, d_state)
        delta_A = torch.einsum("btd,dn->btdn", delta, A)
        A_bar = torch.exp(delta_A)
        # B_bar * x, per timestep: (B, T, d_inner, d_state)
        B_bar_x = torch.einsum("btd,btn,btd->btdn", delta, B_param, x_conv)

        # Sequential scan over time (the recurrence itself). This is the
        # part a fused CUDA kernel parallelizes; here it's an explicit
        # Python-level loop over T, correct but not parallel across time.
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar_x[:, t]  # (B, d_inner, d_state)
            y_t = torch.einsum("bdn,bn->bd", h, C_param[:, t])  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)

        y = y + x_conv * self.D  # skip connection
        y = y * F.silu(z)  # gating, standard Mamba block

        return self.out_proj(y)  # (B, T, d_model)


class MambaBlock(nn.Module):

    def __init__(
        self,
        d_model: int = config.EMBED_DIM,
        d_state: int = config.MAMBA_STATE_DIM,
        d_conv: int = config.MAMBA_CONV_KERNEL,
        expand: int = config.MAMBA_EXPAND_FACTOR,
        dropout: float = config.MAMBA_DROPOUT,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        if USING_OFFICIAL_MAMBA:
            self.ssm = _OfficialMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.ssm = _SelectiveSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return residual + x

class MambaTemporalEncoder(nn.Module):


    def __init__(
        self,
        d_model: int = config.EMBED_DIM,
        num_layers: int = config.MAMBA_NUM_LAYERS,
        d_state: int = config.MAMBA_STATE_DIM,
        d_conv: int = config.MAMBA_CONV_KERNEL,
        expand: int = config.MAMBA_EXPAND_FACTOR,
        dropout: float = config.MAMBA_DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, d_model) - T may be any length, not fixed to
                config.SEQUENCE_LENGTH (verified below with two different
                sequence lengths).
        Returns:
            (B, T, d_model)
        """
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected input shape (B, T, {self.d_model}), got {tuple(x.shape)}."
            )

        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


if __name__ == "__main__":
    # --- Verification: shapes (incl. variable T) + gradient flow ---
    print(f"USING_OFFICIAL_MAMBA: {USING_OFFICIAL_MAMBA}")
    if not USING_OFFICIAL_MAMBA:
        print(
            "mamba_ssm not importable in this environment (likely no nvcc/CUDA) "
            "- verifying the pure-PyTorch _SelectiveSSM fallback instead.\n"
        )

    torch.manual_seed(0)
    batch_size = 2
    embed_dim = config.EMBED_DIM

    model = MambaTemporalEncoder(d_model=embed_dim)

    # Test 1: the project's standard fixed length.
    seq_len_a = config.SEQUENCE_LENGTH
    x_a = torch.randn(batch_size, seq_len_a, embed_dim, requires_grad=True)
    out_a = model(x_a)
    print(f"Input (B,{seq_len_a},D) -> Output {tuple(out_a.shape)}")
    assert out_a.shape == (batch_size, seq_len_a, embed_dim)

    # Test 2: a DIFFERENT sequence length, to confirm variable-length
    # support (no hardcoded SEQUENCE_LENGTH anywhere in this module).
    seq_len_b = 37
    x_b = torch.randn(batch_size, seq_len_b, embed_dim, requires_grad=True)
    out_b = model(x_b)
    print(f"Input (B,{seq_len_b},D) -> Output {tuple(out_b.shape)}")
    assert out_b.shape == (batch_size, seq_len_b, embed_dim)
    print("\nShape verification (incl. variable sequence length): PASSED")

    # Gradient flow check (using the T=100 pass).
    loss = out_a.sum()
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

    upstream_ok = x_a.grad is not None and torch.any(x_a.grad != 0)
    print(f"Upstream gradient reach-through (input x): {'PASSED' if upstream_ok else 'FAILED'}")

    if not USING_OFFICIAL_MAMBA:
        with torch.no_grad():
            model.eval()
            x_base = torch.randn(1, 10, embed_dim)
            out_base = model(x_base)

            x_perturbed = x_base.clone()
            perturbation = torch.randn(1, 3, embed_dim) * 50.0  # non-uniform across channels
            x_perturbed[:, 7:, :] += perturbation

            out_perturbed = model(x_perturbed)

            early_diff = (out_base[:, :7] - out_perturbed[:, :7]).abs().max().item()
            late_diff = (out_base[:, 7:] - out_perturbed[:, 7:]).abs().max().item()

        print(f"\nCausality check: max diff at frames <7 (should be ~0): {early_diff:.8f}")
        print(f"Causality check: max diff at frames >=7 (should be large): {late_diff:.4f}")
        assert early_diff < 1e-4, "Future perturbation leaked into earlier output - not causal!"
        assert late_diff > 1e-2, "Perturbed frames show no change at all - test itself may be broken"
        print("Causality verification: PASSED (fallback SSM is causal)")