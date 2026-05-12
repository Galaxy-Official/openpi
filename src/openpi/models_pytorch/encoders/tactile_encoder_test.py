from __future__ import annotations

import torch
from torch import nn

from openpi.models_pytorch.encoders.tactile_encoder import TactileEncoder


class _TinyVisionEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 12, kernel_size=16, stride=16)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)


def test_tactile_encoder_fallback_stem_shape_and_backward():
    encoder = TactileEncoder(embed_dim=32, num_tokens_per_side=8, patch_size=16)
    left = torch.rand(2, 3, 224, 224)
    right = torch.rand(2, 224, 224, 3)

    tokens, mask = encoder(left, right, mask={"left": torch.tensor([True, False]), "right": torch.ones(2, dtype=bool)})

    assert tokens.shape == (2, 16, 32)
    assert mask.shape == (2, 16)
    assert mask[0].all()
    assert not mask[1, :8].any()
    assert mask[1, 8:].all()
    tokens.sum().backward()
    assert encoder.stem is not None
    assert encoder.stem.patch.weight.grad is not None


def test_tactile_encoder_accepts_frame_stack_and_external_embedder():
    encoder = TactileEncoder(embed_dim=24, num_tokens_per_side=4, vision_embedder=_TinyVisionEmbedder())
    left = torch.rand(2, 3, 3, 64, 64)
    right = torch.rand(2, 3, 64, 64, 3)

    tokens, mask = encoder(left, right)

    assert tokens.shape == (2, 8, 24)
    assert mask.shape == (2, 8)
    assert mask.all()
    tokens.square().mean().backward()
    assert encoder.input_proj.weight.grad is not None
