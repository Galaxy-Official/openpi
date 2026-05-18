"""Tests for PI05Multimodal.

Split into two layers:
- Logic-only tests that exercise phase splitting, warmup ramp, mask reductions,
  and config validation without building the heavy pi0.5 backbone.
- A `manual` integration test that builds the dummy paligemma variant and runs
  a full forward + backward; gated behind `pytest -m manual` so the regular
  suite remains fast.
"""

from __future__ import annotations

import pytest
import torch

from openpi.models.pi0_config_mm import Pi0ConfigMM
from openpi.models_pytorch import pi05_multimodal_pytorch as pi05_mm


def _make_config(**overrides) -> Pi0ConfigMM:
    base = {
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "action_dim": 10,
        "action_horizon": 4,
        "max_token_len": 8,
        "pytorch_compile_mode": None,
        "encoder_warmup_steps": 2_000,
        "contrast_warmup_start": 5_000,
        "contrast_warmup_end": 20_000,
    }
    base.update(overrides)
    return Pi0ConfigMM(**base)


# ---------------------------------------------------------------- pure logic


def test_phase_classification_respects_step_thresholds():
    cfg = _make_config(encoder_warmup_steps=100, contrast_warmup_start=500, contrast_warmup_end=1500)
    cases = {
        0: "encoder_warmup",
        99: "encoder_warmup",
        100: "action_only",
        499: "action_only",
        500: "joint",
        1499: "joint",
        100_000: "joint",
    }
    for step, expected in cases.items():
        assert pi05_mm.PI05Multimodal.phase_for(_FakeModel(cfg), step) == expected


def test_phase_classification_with_warmup_disabled():
    cfg = _make_config(encoder_warmup_steps=0, contrast_warmup_start=100, contrast_warmup_end=200)
    # step 0 should now fall straight into action_only, not encoder_warmup.
    assert pi05_mm.PI05Multimodal.phase_for(_FakeModel(cfg), 0) == "action_only"
    assert pi05_mm.PI05Multimodal.phase_for(_FakeModel(cfg), 99) == "action_only"
    assert pi05_mm.PI05Multimodal.phase_for(_FakeModel(cfg), 100) == "joint"


def test_contrast_warmup_factor_is_monotonic_and_clamped():
    cfg = _make_config(contrast_warmup_start=1_000, contrast_warmup_end=2_000)
    factor = pi05_mm.PI05Multimodal.contrast_warmup_factor.__get__(_FakeModel(cfg))
    assert factor(0) == 0.0
    assert factor(1_000) == 0.0
    assert 0.0 < factor(1_500) < 1.0
    assert factor(1_500) == pytest.approx(0.5, rel=0, abs=1e-6)
    assert factor(2_000) == 1.0
    assert factor(10_000) == 1.0


def test_contrast_warmup_factor_handles_zero_span():
    # Degenerate end == start should not divide-by-zero; we just clamp to 1.0 after the boundary.
    cfg = _make_config(contrast_warmup_start=100, contrast_warmup_end=100)
    factor = pi05_mm.PI05Multimodal.contrast_warmup_factor.__get__(_FakeModel(cfg))
    assert factor(99) == 0.0
    assert factor(100) == 0.0
    assert factor(101) == 1.0


def test_combine_valid_intersection_and_size_check():
    vv = torch.tensor([True, True, False, True])
    tv = torch.tensor([True, False, True, True])
    combined = pi05_mm._combine_valid(vv, tv, batch_size=4, device=torch.device("cpu"))  # noqa: SLF001
    assert combined.tolist() == [True, False, False, True]

    # Defaults to all-True when None is passed.
    assert pi05_mm._combine_valid(None, None, batch_size=3, device=torch.device("cpu")).tolist() == [True] * 3  # noqa: SLF001

    # Shape mismatch is rejected.
    with pytest.raises(ValueError, match="batch_size=4"):
        pi05_mm._combine_valid(torch.tensor([True, True]), tv, batch_size=4, device=torch.device("cpu"))  # noqa: SLF001


def test_sample_valid_merges_dict_masks():
    masks = {"left": torch.tensor([True, False, False]), "right": torch.tensor([False, True, False])}
    out = pi05_mm._sample_valid(masks)  # noqa: SLF001
    assert out.tolist() == [True, True, False]

    assert pi05_mm._sample_valid(None).item() is True  # noqa: SLF001
    assert pi05_mm._sample_valid({}).item() is True  # noqa: SLF001


def test_to_siglip_image_range_converts_zero_one_and_preserves_minus_one_one():
    zero_one = torch.tensor([0.0, 0.5, 1.0])
    converted = pi05_mm._to_siglip_image_range(zero_one)  # noqa: SLF001
    assert torch.allclose(converted, torch.tensor([-1.0, 0.0, 1.0]))

    minus_one_one = torch.tensor([-1.0, 0.0, 1.0])
    preserved = pi05_mm._to_siglip_image_range(minus_one_one)  # noqa: SLF001
    assert torch.allclose(preserved, minus_one_one)


# ---------------------------------------------------------- config validation


def test_use_fusion_requires_tactile_and_force():
    cfg = _make_config(use_tactile=False, use_force=True, use_fusion=True)
    with pytest.raises(ValueError, match="use_fusion"):
        # Only the validation in __init__ needs to fire; the backbone build is
        # gated behind a property reachable before super().__init__ returns,
        # so trigger the check via the dedicated helper instead of building
        # the full model (which requires the transformers_replace patch).
        pi05_mm.PI05Multimodal.validate_toggles(cfg)


def test_use_contrast_requires_fusion():
    cfg = _make_config(use_fusion=False, use_contrast=True)
    with pytest.raises(ValueError, match="use_contrast"):
        pi05_mm.PI05Multimodal.validate_toggles(cfg)


def test_use_vision_latent_contrast_is_rejected_until_implemented():
    cfg = _make_config(use_vision_latent_contrast=True)
    with pytest.raises(NotImplementedError, match="use_vision_latent_contrast"):
        pi05_mm.PI05Multimodal.validate_toggles(cfg)


# ------------------------------------------------------ optional integration


@pytest.mark.manual
def test_dummy_model_compute_loss_and_freeze_cycle():
    """Build a dummy-sized PI05Multimodal and exercise loss aggregation + freeze.

    Avoids actually calling `forward(...)` because the upstream `PI0Pytorch`
    dummy variant has a known config mismatch (hard-coded
    `vision_config.projection_dim = 2048` vs `text_config.hidden_size = 64`)
    that makes any real forward fail. The end-to-end forward smoke is what
    `scripts/debug/dbg_pi05_mm_forward.py` is for; it should be run on the
    cloud server against a real `gemma_2b` / `gemma_300m` backbone.

    This test instead:
    - confirms the new modules are constructed when the toggles are on,
    - feeds synthetic `outputs` into `compute_total_loss` and asserts the
      three phases produce the expected loss composition + `phase` label,
    - confirms `freeze_backbone_for_warmup` actually flips trainable
      parameter counts and that `unfreeze_all` restores them.
    """
    cfg = _make_config(
        encoder_warmup_steps=2,
        contrast_warmup_start=4,
        contrast_warmup_end=6,
        n_fused_tokens=4,
        n_force_tokens=2,
        n_tactile_tokens_per_side=2,
    )
    model = pi05_mm.PI05Multimodal(cfg)

    for name in cfg.encoder_warmup_modules:
        assert getattr(model, name, None) is not None, f"expected `{name}` to be constructed"

    batch_size = 4
    proj_dim = cfg.contrast_proj_dim
    action_horizon = cfg.action_horizon
    action_dim = cfg.action_dim
    outputs = {
        "action_loss": torch.ones(batch_size, action_horizon, action_dim, requires_grad=True),
        "z_vision": _normalize(torch.randn(batch_size, proj_dim, requires_grad=True)),
        "z_touchforce": _normalize(torch.randn(batch_size, proj_dim, requires_grad=True)),
        "vision_valid": torch.ones(batch_size, dtype=torch.bool),
        "touchforce_valid": torch.ones(batch_size, dtype=torch.bool),
        "task_index": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        "frame_index": torch.tensor([10, 50, 5, 70], dtype=torch.int64),
    }

    loss_warmup = model.compute_total_loss(outputs, global_step=0)
    loss_action = model.compute_total_loss(outputs, global_step=3)
    loss_joint = model.compute_total_loss(outputs, global_step=cfg.contrast_warmup_end)

    assert loss_warmup["phase"] == "encoder_warmup"
    assert loss_warmup["contrast_loss"] is not None
    # During warm-up the action loss is recorded only and detached from the graph.
    assert not loss_warmup["action_loss"].requires_grad
    assert loss_warmup["total"].requires_grad

    assert loss_action["phase"] == "action_only"
    assert loss_action["contrast_loss"] is not None  # logged for visibility
    assert torch.allclose(loss_action["total"], outputs["action_loss"].mean())

    assert loss_joint["phase"] == "joint"
    # Joint total should be action_loss + 1.0 * weight * contrast_loss.
    expected_joint = outputs["action_loss"].mean() + cfg.contrast_weight * loss_joint["contrast_loss"]
    assert torch.allclose(loss_joint["total"], expected_joint)

    # Freeze / unfreeze cycle.
    total_param_count = sum(p.numel() for p in model.parameters())
    model.freeze_backbone_for_warmup()
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert frozen_trainable < total_param_count, "freeze did not reduce trainable param count"
    # The white-listed new modules must remain trainable during warm-up.
    for name in cfg.encoder_warmup_modules:
        module = getattr(model, name, None)
        if module is None:
            continue
        assert all(p.requires_grad for p in module.parameters()), (
            f"`{name}` should keep requires_grad=True during warm-up"
        )

    model.unfreeze_all()
    fully_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert fully_trainable == total_param_count, "unfreeze_all did not restore full trainability"

    # Smoke: joint backward should not error after unfreezing.
    loss_joint["total"].backward(retain_graph=True)


# ----------------------------------------------------------------- helpers


class _FakeModel:
    """Just enough to satisfy `PI05Multimodal.phase_for` / `contrast_warmup_factor`."""

    def __init__(self, cfg: Pi0ConfigMM):
        self.mm_config = cfg


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, dim=-1)
