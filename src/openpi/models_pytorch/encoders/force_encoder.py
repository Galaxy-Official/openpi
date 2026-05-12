"""Force-window encoder for pi0.5 multi-modal pretraining."""

from __future__ import annotations

import torch
from torch import nn

from openpi.models_pytorch.encoders._utils import choose_num_heads
from openpi.models_pytorch.encoders._utils import resample_token_mask
from openpi.models_pytorch.encoders._utils import resample_tokens


class ForceEncoder(nn.Module):
    """Encode `(B, T, C)` force windows into a small fixed token set."""

    def __init__(
        self,
        embed_dim: int,
        *,
        input_dim: int = 2,
        hidden_dim: int | None = None,
        num_tokens: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")

        hidden_dim = hidden_dim or embed_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.input_dim = input_dim

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=3, padding=1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=choose_num_heads(embed_dim, num_heads),
            dim_feedforward=max(embed_dim * 4, 64),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        force: torch.Tensor | dict[str, torch.Tensor],
        mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        force, time_mask = self._prepare_force(force, mask)
        if force.ndim != 3:
            raise ValueError(f"Expected force tensor with shape `(B, T, C)`, got {tuple(force.shape)}")
        if force.shape[-1] != self.input_dim:
            raise ValueError(f"Expected force input_dim={self.input_dim}, got {force.shape[-1]}")

        force = force.to(torch.float32)
        force = force * time_mask.unsqueeze(-1).to(force.dtype)

        x = self.conv(force.transpose(1, 2)).transpose(1, 2)
        key_padding_mask = _safe_key_padding_mask(time_mask)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = x * time_mask.unsqueeze(-1).to(x.dtype)
        x = resample_tokens(self.out_norm(x), self.num_tokens)
        pad_mask = resample_token_mask(time_mask, self.num_tokens)
        x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x, pad_mask

    def _prepare_force(
        self,
        force: torch.Tensor | dict[str, torch.Tensor],
        mask: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(force, dict):
            if force.ndim == 2:
                force = force.unsqueeze(-1)
            return force, self._time_mask(mask, force.shape[0], force.shape[1], force.device)

        if self.input_dim != 2:
            raise ValueError("dict force input requires input_dim=2")
        left_value = force.get("left")
        right_value = force.get("right")
        if left_value is None and right_value is None:
            raise ValueError("force dict must contain at least one of `left` or `right`")

        left = _side_force_tensor(left_value, right_value, "left")
        right = _side_force_tensor(right_value, left, "right").to(device=left.device)
        if left.shape != right.shape:
            raise ValueError(f"left/right force shapes must match, got {tuple(left.shape)} and {tuple(right.shape)}")

        batch_size, window = left.shape
        left_mask = self._side_time_mask(mask, "left", batch_size, window, left.device, present=left_value is not None)
        right_mask = self._side_time_mask(
            mask,
            "right",
            batch_size,
            window,
            left.device,
            present=right_value is not None,
        )
        force_tensor = torch.stack([left, right], dim=-1).to(torch.float32)
        side_mask = torch.stack([left_mask, right_mask], dim=-1)
        force_tensor = force_tensor * side_mask.to(force_tensor.dtype)
        return force_tensor, left_mask | right_mask

    @staticmethod
    def _time_mask(
        mask: bool | torch.Tensor | None,
        batch_size: int,
        window: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch_size, window, dtype=torch.bool, device=device)
        if isinstance(mask, bool):
            return torch.full((batch_size, window), mask, dtype=torch.bool, device=device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if mask.ndim == 0:
            return torch.full((batch_size, window), bool(mask.item()), dtype=torch.bool, device=device)
        if mask.ndim == 2:
            if mask.shape != (batch_size, window):
                raise ValueError(f"Expected force mask shape {(batch_size, window)}, got {tuple(mask.shape)}")
            return mask
        if mask.shape != (batch_size,):
            raise ValueError(f"Expected force mask shape {(batch_size,)}, got {tuple(mask.shape)}")
        return mask[:, None].expand(-1, window)

    @classmethod
    def _side_time_mask(
        cls,
        mask: torch.Tensor | dict[str, torch.Tensor] | None,
        side: str,
        batch_size: int,
        window: int,
        device: torch.device,
        *,
        present: bool,
    ) -> torch.Tensor:
        if not present:
            return torch.zeros(batch_size, window, dtype=torch.bool, device=device)
        if isinstance(mask, dict):
            return cls._time_mask(mask.get(side, True), batch_size, window, device)
        return cls._time_mask(mask, batch_size, window, device)


def _side_force_tensor(
    value: torch.Tensor | None,
    reference: torch.Tensor | None,
    side: str,
) -> torch.Tensor:
    if value is None:
        if reference is None:
            raise ValueError(f"force dict is missing `{side}`")
        return torch.zeros_like(torch.as_tensor(reference, dtype=torch.float32))
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected `{side}` force shape `(B, T)`, got {tuple(tensor.shape)}")
    return tensor


def _safe_key_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    key_padding_mask = ~valid_mask.to(dtype=torch.bool)
    all_invalid = key_padding_mask.all(dim=1)
    if all_invalid.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_invalid] = False
    return key_padding_mask
