"""W8A8 INT8 for the prepared BF16 DiT: the block GEMMs run as int8 x int8 -> int32 (``torch._int_mm``).

The same scheme as the released FP8r packages, with INT8 instead of E4M3: each weight row (output channel)
has its own scale, and each activation row (token) is quantized on the fly with its own scale. Only the
three GEMMs per block that dominate the sampler are converted: the packed q/k/v/MLP input projection,
``attn_out`` and ``mlp_out``. Embeddings, modulations, final layers and attention stay BF16.

On GPUs whose INT8 matrix rate is well above their FP8 rate (Intel Arc Pro B70: 268 against 78 dense
TOP/s at these shapes, 136 for BF16), this is the fast low-precision path. Call
``quantize_prepared_dit(policy.dit)`` after ``prepare_inference`` and before the first prediction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

_ROW_ALIGNMENT = 32  # torch._int_mm needs more than 16 rows; keep them a multiple of 32


def _quantize_rows(value: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-row INT8: ``value ~= q * scale[:, None]``."""
    rows = value.float()
    scale = (rows.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-12)
    return torch.round(rows / scale).clamp(-127, 127).to(torch.int8), scale


class Int8Linear(nn.Module):
    """``y = x @ W.T`` with an INT8 weight (per output channel) and dynamically quantized INT8 activations."""

    def __init__(self, weight: Tensor):
        super().__init__()
        assert weight.ndim == 2, "expected an (out, in) weight"
        weight_q, weight_scale = _quantize_rows(weight)
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight_t", weight_q.t().contiguous())  # (in, out) for torch._int_mm
        self.register_buffer("weight_scale", weight_scale.reshape(1, -1))

    def forward(self, x: Tensor) -> Tensor:
        leading = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        rows = flat.shape[0]
        padded = -(-max(rows, _ROW_ALIGNMENT + 1) // _ROW_ALIGNMENT) * _ROW_ALIGNMENT
        if padded != rows:
            flat = nn.functional.pad(flat, (0, 0, 0, padded - rows))
        activation_q, activation_scale = _quantize_rows(flat)
        accumulated = torch._int_mm(activation_q, self.weight_t)
        out = (accumulated.float() * activation_scale * self.weight_scale).to(x.dtype)
        return out[:rows].reshape(*leading, self.out_features)


@torch.no_grad()
def quantize_prepared_dit(dit: nn.Module) -> int:
    """Convert every block's packed input projection, ``attn_out`` and ``mlp_out``; returns how many GEMMs."""
    converted = 0
    for block in dit.modules():
        weight = getattr(block, "in_proj_weight", None)
        if not isinstance(weight, Tensor):
            continue
        block.in_proj_int8 = Int8Linear(weight.data)
        block.in_proj_weight = None
        for name in ("attn_out", "mlp_out"):
            setattr(block, name, Int8Linear(getattr(block, name).weight.data))
        converted += 3
    return converted
