"""Tactile video/image encoder for pi0.5 multi-modal pretraining."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from openpi.models_pytorch.encoders._utils import ensure_bchw_images
from openpi.models_pytorch.encoders._utils import resample_tokens


class _PatchTactileStem(nn.Module):
    """Small fallback patch encoder used when no SigLIP/PaliGemma image embedder is provided."""

    def __init__(self, embed_dim: int, *, input_channels: int = 3, patch_size: int = 14):
        super().__init__()
        self.patch = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch(images)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(torch.nn.functional.gelu(x))


class TactileEncoder(nn.Module):
    """Encode left/right tactile streams into fixed-width prefix tokens.

    `vision_embedder` can be `PaliGemmaWithExpertModel.embed_image` or another SigLIP-compatible callable returning
    `(B, N, D)` image tokens. If omitted, a lightweight patch stem is used so this module can be tested independently.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_tokens_per_side: int = 16,
        input_channels: int = 3,
        patch_size: int = 14,
        vision_embedder: Callable[[torch.Tensor], torch.Tensor] | nn.Module | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_tokens_per_side <= 0:
            raise ValueError("num_tokens_per_side must be positive")

        self.embed_dim = embed_dim
        self.num_tokens_per_side = num_tokens_per_side
        self.vision_embedder = vision_embedder
        self.stem = (
            None
            if vision_embedder is not None
            else _PatchTactileStem(embed_dim, input_channels=input_channels, patch_size=patch_size)
        )
        self.input_proj = nn.LazyLinear(embed_dim) if vision_embedder is not None else nn.Identity()
        self.out_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    @property
    def num_tokens(self) -> int:
        return self.num_tokens_per_side * 2

    def forward(
        self,
        tactile_left: torch.Tensor,
        tactile_right: torch.Tensor,
        mask: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_tokens = self._encode_side(tactile_left)
        right_tokens = self._encode_side(tactile_right)

        tokens = torch.cat([left_tokens, right_tokens], dim=1)
        left_mask = self._side_mask(mask, "left", left_tokens.shape[0], left_tokens.device)
        right_mask = self._side_mask(mask, "right", right_tokens.shape[0], right_tokens.device)
        pad_mask = torch.cat(
            [
                left_mask[:, None].expand(-1, left_tokens.shape[1]),
                right_mask[:, None].expand(-1, right_tokens.shape[1]),
            ],
            dim=1,
        )
        tokens = tokens * pad_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, pad_mask

    def _encode_side(self, images: torch.Tensor) -> torch.Tensor:
        flat_images, batch_size, stack = ensure_bchw_images(images)
        if self.vision_embedder is not None:
            tokens = self.vision_embedder(flat_images)
        else:
            if self.stem is None:
                raise RuntimeError("Patch stem is not initialized")
            tokens = self.stem(flat_images)
        tokens = self.input_proj(tokens)
        tokens = resample_tokens(tokens, self.num_tokens_per_side)
        tokens = tokens.reshape(batch_size, stack, self.num_tokens_per_side, self.embed_dim).mean(dim=1)
        return self.dropout(self.out_norm(tokens))

    @staticmethod
    def _side_mask(
        mask: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None,
        side: str,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if isinstance(mask, dict):
            value = mask.get(side, True)
        elif isinstance(mask, (tuple, list)):
            value = mask[0 if side == "left" else 1]
        else:
            value = mask
        if isinstance(value, bool):
            return torch.full((batch_size,), value, dtype=torch.bool, device=device)
        return torch.as_tensor(value, dtype=torch.bool, device=device).reshape(batch_size)
