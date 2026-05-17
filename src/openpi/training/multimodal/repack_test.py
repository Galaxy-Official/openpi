"""Unit tests for MultiModalRepack and the multi-modal collate function.

These tests do not depend on lerobot or the cloud datasets; they exercise the
pure-numpy transform contract using a hand-built fake sample.
"""

from __future__ import annotations

import numpy as np

from openpi.training.multimodal.collate import multimodal_collate_fn
from openpi.training.multimodal.repack import MultiModalRepack


def _fake_sample(*, with_force: bool = True, with_tactile: bool = True, with_wrist: bool = True) -> dict:
    sample = {
        "observation.state": np.zeros((10,), dtype=np.float32),
        "observation.state_phone": np.zeros((6,), dtype=np.float32),
        "action": np.zeros((50, 10), dtype=np.float32),
        "observation.images.phone": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8) if with_wrist else None,
        "observation.tactiles.left": np.zeros((224, 224, 3), dtype=np.uint8) if with_tactile else None,
        "observation.tactiles.right": np.zeros((224, 224, 3), dtype=np.uint8) if with_tactile else None,
        "observation.forces.left": np.zeros((16,), dtype=np.float32) if with_force else None,
        "observation.forces.right": np.zeros((16,), dtype=np.float32) if with_force else None,
        "observation.tactiles.left.mask": np.asarray(with_tactile),
        "observation.tactiles.right.mask": np.asarray(with_tactile),
        "observation.forces.left.mask": np.asarray(with_force),
        "observation.forces.right.mask": np.asarray(with_force),
        "task_index": np.asarray(0, dtype=np.int64),
        "episode_index": np.asarray(3, dtype=np.int64),
        "frame_index": np.asarray(42, dtype=np.int64),
        "index": np.asarray(123, dtype=np.int64),
        "timestamp": np.asarray(1.75, dtype=np.float32),
        "prompt": "do the thing",
        "task_name": "fake_task",
    }
    return {k: v for k, v in sample.items() if v is not None}


def test_repack_produces_expected_structure():
    repack = MultiModalRepack()
    out = repack(_fake_sample())

    assert set(out["image"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert set(out["image_mask"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert set(out["tactile"].keys()) == {"left", "right"}
    assert set(out["tactile_mask"].keys()) == {"left", "right"}
    assert set(out["force"].keys()) == {"left", "right"}
    assert set(out["force_mask"].keys()) == {"left", "right"}
    assert out["state"].shape == (10,)
    assert out["actions"].shape == (50, 10)
    assert out["prompt"] == "do the thing"
    assert out["task_name"] == "fake_task"
    assert bool(out["image_mask"]["base_0_rgb"]) is True
    assert bool(out["image_mask"]["left_wrist_0_rgb"]) is True
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is False  # not present in this dataset


def test_repack_missing_modalities_set_mask_false():
    repack = MultiModalRepack()
    out = repack(_fake_sample(with_force=False, with_tactile=False))
    assert bool(out["tactile_mask"]["left"]) is False
    assert bool(out["tactile_mask"]["right"]) is False
    assert bool(out["force_mask"]["left"]) is False
    assert bool(out["force_mask"]["right"]) is False
    # Data should still be present as zeros.
    assert np.all(out["force"]["left"] == 0.0)
    assert np.all(out["tactile"]["left"] == 0)


def test_repack_dropout_zeros_modality_and_sets_mask():
    # Force always-drop: tactile_drop=1.0
    repack = MultiModalRepack(tactile_drop=1.0)
    out = repack(_fake_sample())
    assert bool(out["tactile_mask"]["left"]) is False
    assert bool(out["tactile_mask"]["right"]) is False
    assert np.all(out["tactile"]["left"] == 0)
    assert np.all(out["tactile"]["right"] == 0)


def test_repack_use_full_state_vs_state_phone():
    out_full = MultiModalRepack(use_full_state=True)(_fake_sample())
    out_phone = MultiModalRepack(use_full_state=False)(_fake_sample())
    assert out_full["state"].shape == (10,)
    assert out_phone["state"].shape == (6,)


def test_collate_stacks_numeric_and_lists_strings():
    samples = [MultiModalRepack()(_fake_sample()) for _ in range(3)]
    batch = multimodal_collate_fn(samples)
    assert batch["state"].shape == (3, 10)
    assert batch["actions"].shape == (3, 50, 10)
    assert batch["image"]["base_0_rgb"].shape == (3, 480, 640, 3)
    assert batch["image_mask"]["base_0_rgb"].shape == (3,)
    assert batch["tactile"]["left"].shape == (3, 224, 224, 3)
    assert batch["force"]["left"].shape == (3, 16)
    # String leaves should remain as lists, not numpy arrays.
    assert isinstance(batch["prompt"], list)
    assert isinstance(batch["task_name"], list)
    assert len(batch["prompt"]) == 3


def test_repack_wrist_only_zeros_base_camera_even_when_phone_present():
    # Even with a real phone image in the sample, wrist_only must drop it.
    sample = _fake_sample()
    sample["observation.images.phone"] = np.full((480, 640, 3), 200, dtype=np.uint8)
    out = MultiModalRepack(wrist_only=True)(sample)
    assert bool(out["image_mask"]["base_0_rgb"]) is False
    assert np.all(out["image"]["base_0_rgb"] == 0)
    # Wrist still flows through and is masked True.
    assert bool(out["image_mask"]["left_wrist_0_rgb"]) is True
    # Right wrist slot continues to be a zero placeholder.
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is False


def test_repack_dropout_is_reproducible_per_sample():
    # Same drop_seed + same index -> same drop decision.
    sample = _fake_sample()
    sample["index"] = np.asarray(7, dtype=np.int64)
    rp = MultiModalRepack(tactile_drop=0.5, drop_seed=1234)
    a = rp(dict(sample))
    b = rp(dict(sample))
    assert bool(a["tactile_mask"]["left"]) == bool(b["tactile_mask"]["left"])
