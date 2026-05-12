from __future__ import annotations

import torch

from openpi.models_pytorch.encoders.touch_force_fusion import SimpleFusion
from openpi.models_pytorch.encoders.touch_force_fusion import TouchForceFusion


def test_touch_force_fusion_shape_norm_and_backward():
    fusion = TouchForceFusion(embed_dim=32, num_fused_tokens=12, proj_dim=16, num_layers=1, num_heads=4)
    tactile = torch.randn(3, 10, 32, requires_grad=True)
    force = torch.randn(3, 4, 32, requires_grad=True)
    tactile_mask = torch.ones(3, 10, dtype=torch.bool)
    force_mask = torch.tensor([[True, True, True, True], [False, False, False, False], [True, True, False, False]])

    fused, fused_mask, z = fusion(tactile, force, tactile_mask=tactile_mask, force_mask=force_mask)

    assert fused.shape == (3, 12, 32)
    assert fused_mask.shape == (3, 12)
    assert z.shape == (3, 16)
    assert torch.allclose(z.norm(dim=-1), torch.ones(3), atol=1e-5)
    (fused.square().mean() + z.square().mean()).backward()
    assert tactile.grad is not None
    assert force.grad is not None


def test_simple_fusion_handles_all_invalid_sample():
    fusion = SimpleFusion(embed_dim=16, num_fused_tokens=6, proj_dim=8)
    tactile = torch.randn(2, 4, 16)
    force = torch.randn(2, 2, 16)
    tactile_mask = torch.tensor([[True, True, True, True], [False, False, False, False]])
    force_mask = torch.tensor([[True, True], [False, False]])

    fused, fused_mask, z = fusion(tactile, force, tactile_mask=tactile_mask, force_mask=force_mask)

    assert fused.shape == (2, 6, 16)
    assert fused_mask[0].any()
    assert not fused_mask[1].any()
    assert torch.all(z[1] == 0)
