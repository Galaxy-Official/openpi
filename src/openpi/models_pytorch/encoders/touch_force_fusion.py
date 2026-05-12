"""Fusion modules for tactile and force token streams."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.encoders._utils import choose_num_heads
from openpi.models_pytorch.encoders._utils import masked_mean_pool
from openpi.models_pytorch.encoders._utils import resample_token_mask
from openpi.models_pytorch.encoders._utils import resample_tokens


class _CrossBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        heads = choose_num_heads(embed_dim, num_heads)
        self.force_to_tactile = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.tactile_to_force = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.force_norm = nn.LayerNorm(embed_dim)
        self.tactile_norm = nn.LayerNorm(embed_dim)
        self.force_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.tactile_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self,
        tactile: torch.Tensor,
        force: torch.Tensor,
        tactile_mask: torch.Tensor | None,
        force_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tactile = _zero_invalid_tokens(tactile, tactile_mask)
        force = _zero_invalid_tokens(force, force_mask)
        tactile_key_padding = _safe_key_padding_mask(tactile_mask)
        force_key_padding = _safe_key_padding_mask(force_mask)
        tactile_valid = _sample_valid(tactile_mask, tactile.shape[0], tactile.device)
        force_valid = _sample_valid(force_mask, force.shape[0], force.device)

        force_delta, _ = self.force_to_tactile(
            self.force_norm(force),
            self.tactile_norm(tactile),
            self.tactile_norm(tactile),
            key_padding_mask=tactile_key_padding,
            need_weights=False,
        )
        force_delta = force_delta * tactile_valid[:, None, None].to(force_delta.dtype)
        force = force + force_delta
        force = force + self.force_ffn(force)
        force = _zero_invalid_tokens(force, force_mask)

        tactile_delta, _ = self.tactile_to_force(
            self.tactile_norm(tactile),
            self.force_norm(force),
            self.force_norm(force),
            key_padding_mask=force_key_padding,
            need_weights=False,
        )
        tactile_delta = tactile_delta * force_valid[:, None, None].to(tactile_delta.dtype)
        tactile = tactile + tactile_delta
        tactile = tactile + self.tactile_ffn(tactile)

        tactile = _zero_invalid_tokens(tactile, tactile_mask)
        force = _zero_invalid_tokens(force, force_mask)
        return tactile, force


class TouchForceFusion(nn.Module):
    """Cross-attention fusion of tactile and force tokens."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_fused_tokens: int = 32,
        proj_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_fused_tokens <= 0:
            raise ValueError("num_fused_tokens must be positive")
        self.num_fused_tokens = num_fused_tokens
        self.layers = nn.ModuleList([_CrossBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.fused_norm = nn.LayerNorm(embed_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(
        self,
        tactile_tokens: torch.Tensor,
        force_tokens: torch.Tensor,
        tactile_mask: torch.Tensor | None = None,
        force_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tactile_mask = _default_mask(tactile_tokens, tactile_mask)
        force_mask = _default_mask(force_tokens, force_mask)

        tactile = tactile_tokens
        force = force_tokens
        for layer in self.layers:
            tactile, force = layer(tactile, force, tactile_mask, force_mask)

        fused_tokens = torch.cat([tactile, force], dim=1)
        fused_mask = torch.cat([tactile_mask, force_mask], dim=1)
        fused_tokens = resample_tokens(self.fused_norm(fused_tokens), self.num_fused_tokens)
        fused_mask = resample_token_mask(fused_mask, self.num_fused_tokens)
        fused_tokens = fused_tokens * fused_mask.unsqueeze(-1).to(fused_tokens.dtype)

        pooled, valid = masked_mean_pool(fused_tokens, fused_mask)
        z_touchforce = F.normalize(self.projector(pooled), dim=-1)
        z_touchforce = torch.where(valid[:, None], z_touchforce, torch.zeros_like(z_touchforce))
        return fused_tokens, fused_mask, z_touchforce


class SimpleFusion(nn.Module):
    """MLP-free concat/pool baseline for ablations."""

    def __init__(self, embed_dim: int, *, num_fused_tokens: int = 32, proj_dim: int = 256):
        super().__init__()
        self.num_fused_tokens = num_fused_tokens
        self.norm = nn.LayerNorm(embed_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(
        self,
        tactile_tokens: torch.Tensor,
        force_tokens: torch.Tensor,
        tactile_mask: torch.Tensor | None = None,
        force_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tactile_mask = _default_mask(tactile_tokens, tactile_mask)
        force_mask = _default_mask(force_tokens, force_mask)
        fused_tokens = torch.cat([tactile_tokens, force_tokens], dim=1)
        fused_mask = torch.cat([tactile_mask, force_mask], dim=1)
        fused_tokens = resample_tokens(self.norm(fused_tokens), self.num_fused_tokens)
        fused_mask = resample_token_mask(fused_mask, self.num_fused_tokens)
        pooled, valid = masked_mean_pool(fused_tokens, fused_mask)
        z_touchforce = F.normalize(self.projector(pooled), dim=-1)
        z_touchforce = torch.where(valid[:, None], z_touchforce, torch.zeros_like(z_touchforce))
        return fused_tokens * fused_mask.unsqueeze(-1).to(fused_tokens.dtype), fused_mask, z_touchforce


def _default_mask(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    return mask.to(dtype=torch.bool, device=tokens.device)


def _safe_key_padding_mask(valid_mask: torch.Tensor | None) -> torch.Tensor | None:
    if valid_mask is None:
        return None
    key_padding_mask = ~valid_mask.to(dtype=torch.bool)
    all_invalid = key_padding_mask.all(dim=1)
    if all_invalid.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_invalid] = False
    return key_padding_mask


def _zero_invalid_tokens(tokens: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return tokens
    return tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)


def _sample_valid(mask: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return mask.to(dtype=torch.bool, device=device).any(dim=1)
