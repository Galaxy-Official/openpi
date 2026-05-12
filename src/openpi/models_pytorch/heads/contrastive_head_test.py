from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.heads.contrastive_head import ContrastiveHead
from openpi.models_pytorch.heads.contrastive_head import VisionProjector
from openpi.models_pytorch.heads.contrastive_head import build_task_aware_neg_mask
from openpi.models_pytorch.heads.contrastive_head import info_nce


def test_vision_projector_prefers_valid_wrist_tokens():
    projector = VisionProjector(embed_dim=12, proj_dim=8)
    phone = torch.randn(3, 5, 12)
    wrist = torch.randn(3, 5, 12)
    masks = {
        "left_wrist_0_rgb": torch.tensor([False, True, True]),
        "base_0_rgb": torch.tensor([True, True, True]),
    }

    z, valid = projector({"base_0_rgb": phone, "left_wrist_0_rgb": wrist}, masks)

    assert z.shape == (3, 8)
    assert valid.all()
    assert torch.allclose(z.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_task_aware_neg_mask_blocks_same_task_neighbor_frames():
    mask = build_task_aware_neg_mask(
        torch.tensor([0, 0, 0, 1]),
        torch.tensor([10, 12, 30, 11]),
        window=8,
    )

    assert mask[0, 1]
    assert mask[1, 0]
    assert not mask[0, 2]
    assert not mask[0, 3]
    assert not mask.diag().any()


def test_info_nce_optimization_decreases_loss():
    torch.manual_seed(0)
    z_a = F.normalize(torch.randn(8, 16), dim=-1)
    z_b = torch.nn.Parameter(F.normalize(z_a + 0.5 * torch.randn(8, 16), dim=-1))
    opt = torch.optim.SGD([z_b], lr=0.5)

    initial = None
    final = None
    for step in range(40):
        opt.zero_grad()
        loss, _ = info_nce(z_a, F.normalize(z_b, dim=-1), temperature=0.2)
        if step == 0:
            initial = float(loss.detach())
        loss.backward()
        opt.step()
        final = float(loss.detach())

    assert initial is not None
    assert final is not None
    assert final < initial


def test_contrastive_head_returns_metrics():
    head = ContrastiveHead(temperature=0.1, neg_window=4)
    z = F.normalize(torch.randn(4, 8), dim=-1)
    out = head(z, z, task_index=torch.tensor([0, 0, 1, 1]), frame_index=torch.tensor([0, 2, 0, 20]))

    assert set(out) == {"loss", "acc@1", "pos_sim", "neg_sim"}
    assert out["loss"].ndim == 0
    assert out["acc@1"] > 0
