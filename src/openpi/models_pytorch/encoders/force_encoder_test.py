from __future__ import annotations

import torch

from openpi.models_pytorch.encoders.force_encoder import ForceEncoder


def test_force_encoder_shape_mask_and_backward():
    encoder = ForceEncoder(embed_dim=32, input_dim=2, num_tokens=4, num_layers=1)
    force = torch.randn(3, 16, 2)
    mask = torch.tensor([True, False, True])

    tokens, pad_mask = encoder(force, mask=mask)

    assert tokens.shape == (3, 4, 32)
    assert pad_mask.shape == (3, 4)
    assert pad_mask[0].all()
    assert not pad_mask[1].any()
    assert torch.all(tokens[1] == 0)
    tokens.sum().backward()
    assert encoder.conv[0].weight.grad is not None


def test_force_encoder_accepts_time_mask():
    encoder = ForceEncoder(embed_dim=16, input_dim=2, num_tokens=2, num_layers=1)
    force = torch.randn(2, 8, 2)
    time_mask = torch.tensor([[False] * 8, [False, True, False, False, False, False, False, False]])

    _, pad_mask = encoder(force, mask=time_mask)

    assert not pad_mask[0].any()
    assert pad_mask[1].any()


def test_force_encoder_accepts_left_right_dict():
    encoder = ForceEncoder(embed_dim=16, input_dim=2, num_tokens=2, num_layers=1)
    force = {
        "left": torch.randn(2, 8),
        "right": torch.randn(2, 8),
    }
    mask = {
        "left": torch.tensor([True, False]),
        "right": torch.tensor([False, False]),
    }

    tokens, pad_mask = encoder(force, mask=mask)

    assert tokens.shape == (2, 2, 16)
    assert pad_mask[0].all()
    assert not pad_mask[1].any()
    assert torch.all(tokens[1] == 0)
