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


def testcontrast_warmup_factor_is_monotonic_and_clamped():
    cfg = _make_config(contrast_warmup_start=1_000, contrast_warmup_end=2_000)
    factor = pi05_mm.PI05Multimodal.contrast_warmup_factor.__get__(_FakeModel(cfg))
    assert factor(0) == 0.0
    assert factor(1_000) == 0.0
    assert 0.0 < factor(1_500) < 1.0
    assert factor(1_500) == pytest.approx(0.5, rel=0, abs=1e-6)
    assert factor(2_000) == 1.0
    assert factor(10_000) == 1.0


def testcontrast_warmup_factor_handles_zero_span():
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


# ------------------------------------------------------ optional integration


@pytest.mark.manual
def test_dummy_forward_and_loss_phases():
    """Build a dummy-sized PI05Multimodal and exercise all three loss phases.

    Marked `manual` so the routine pytest run on machines without the
    `transformers_replace` patch does not pay the full backbone build cost.
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
    obs = _fake_observation(cfg)
    actions = torch.randn(2, cfg.action_horizon, cfg.action_dim)

    outputs = model(obs, actions)
    assert "action_loss" in outputs
    assert outputs["action_loss"].shape == (2, cfg.action_horizon, cfg.action_dim)

    losses_by_phase = {step: model.compute_total_loss(outputs, step) for step in (0, 3, 5)}
    assert losses_by_phase[0]["phase"] == "encoder_warmup"
    assert losses_by_phase[3]["phase"] == "action_only"
    assert losses_by_phase[5]["phase"] == "joint"

    # Backward only the joint-phase total -- backbone params should be unfrozen by default.
    losses_by_phase[5]["total"].backward()


# ----------------------------------------------------------------- helpers


class _FakeModel:
    """Just enough to satisfy `PI05Multimodal.phase_for` / `contrast_warmup_factor`."""

    def __init__(self, cfg: Pi0ConfigMM):
        self.mm_config = cfg


def _fake_observation(cfg: Pi0ConfigMM):
    from openpi.models.model import Observation  # noqa: PLC0415

    state = torch.zeros(2, cfg.action_dim)
    images = {
        "base_0_rgb": torch.zeros(2, 3, 224, 224),
        "left_wrist_0_rgb": torch.zeros(2, 3, 224, 224),
        "right_wrist_0_rgb": torch.zeros(2, 3, 224, 224),
    }
    image_masks = {
        "base_0_rgb": torch.zeros(2, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(2, dtype=torch.bool),
    }
    tokenized_prompt = torch.zeros(2, cfg.max_token_len, dtype=torch.int32)
    tokenized_prompt_mask = torch.ones(2, cfg.max_token_len, dtype=torch.bool)
    tactile = {"left": torch.zeros(2, 224, 224, 3), "right": torch.zeros(2, 224, 224, 3)}
    tactile_mask = {"left": torch.ones(2, dtype=torch.bool), "right": torch.ones(2, dtype=torch.bool)}
    force = {"left": torch.zeros(2, cfg.force_window), "right": torch.zeros(2, cfg.force_window)}
    force_mask = {"left": torch.ones(2, dtype=torch.bool), "right": torch.ones(2, dtype=torch.bool)}

    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        tactile=tactile,
        tactile_mask=tactile_mask,
        force=force,
        force_mask=force_mask,
        task_index=torch.tensor([0, 0]),
        frame_index=torch.tensor([10, 50]),
    )
