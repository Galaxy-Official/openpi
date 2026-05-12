from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


def ensure_bchw_images(images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Normalize image or frame-stack tensors to `(B*T, C, H, W)`.

    Accepts `(B, C, H, W)`, `(B, H, W, C)`, `(B, T, C, H, W)`, and `(B, T, H, W, C)`.
    Returns the flattened image tensor plus the original batch and stack lengths.
    """
    if images.ndim == 4:
        batch_size, stack = images.shape[0], 1
        if images.shape[1] in (1, 3, 4):
            out = images
        elif images.shape[-1] in (1, 3, 4):
            out = images.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Cannot infer image channel axis from shape {tuple(images.shape)}")
    elif images.ndim == 5:
        batch_size, stack = images.shape[:2]
        if images.shape[2] in (1, 3, 4):
            out = images.reshape(batch_size * stack, *images.shape[2:])
        elif images.shape[-1] in (1, 3, 4):
            out = images.permute(0, 1, 4, 2, 3).reshape(batch_size * stack, images.shape[-1], *images.shape[2:4])
        else:
            raise ValueError(f"Cannot infer image channel axis from shape {tuple(images.shape)}")
    else:
        raise ValueError(f"Expected 4-D or 5-D image tensor, got {tuple(images.shape)}")

    if out.dtype == torch.uint8:
        out = out.to(torch.float32) / 255.0
    else:
        out = out.to(torch.float32)
        if out.numel() and torch.nan_to_num(out.detach()).max() > 1.5:
            out = out / 255.0
    return out.contiguous(), batch_size, stack


def resample_tokens(tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Resample a token sequence to a fixed token count with adaptive average pooling."""
    if tokens.shape[1] == num_tokens:
        return tokens
    return F.adaptive_avg_pool1d(tokens.transpose(1, 2), num_tokens).transpose(1, 2)


def resample_token_mask(mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Resample a boolean token mask to a fixed token count."""
    if mask.shape[1] == num_tokens:
        return mask
    pooled = F.adaptive_max_pool1d(mask.to(torch.float32).unsqueeze(1), num_tokens)
    return pooled.squeeze(1).to(torch.bool)


def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool tokens while returning the per-sample validity mask."""
    if mask is None:
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    else:
        valid = mask.to(dtype=torch.bool, device=tokens.device)
    weights = valid.unsqueeze(-1).to(tokens.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    pooled = (tokens * weights).sum(dim=1) / denom
    sample_valid = valid.any(dim=1)
    pooled = torch.where(sample_valid[:, None], pooled, torch.zeros_like(pooled))
    return pooled, sample_valid


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def choose_num_heads(embed_dim: int, requested: int) -> int:
    """Pick a valid attention head count not larger than `requested`."""
    for heads in range(min(requested, embed_dim), 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


def square_grid_size(num_tokens: int) -> int | None:
    root = int(math.sqrt(num_tokens))
    return root if root * root == num_tokens else None
