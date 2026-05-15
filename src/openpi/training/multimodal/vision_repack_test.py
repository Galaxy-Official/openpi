from __future__ import annotations

import numpy as np

from openpi.training.multimodal.repack import VisionOnlyWristRepack


def _sample():
    return {
        "observation.state": np.arange(10, dtype=np.float32),
        "action": np.ones((50, 10), dtype=np.float32),
        "observation.images.phone": np.full((8, 8, 3), 255, dtype=np.uint8),
        "observation.images.wrist": np.full((8, 8, 3), 7, dtype=np.uint8),
        "observation.images.wrist.mask": np.asarray(True, dtype=bool),
        "prompt": "pick up the item",
    }


def test_vision_only_wrist_repack_masks_out_phone_camera():
    out = VisionOnlyWristRepack()(_sample())

    assert not bool(out["image_mask"]["base_0_rgb"])
    assert bool(out["image_mask"]["left_wrist_0_rgb"])
    assert not bool(out["image_mask"]["right_wrist_0_rgb"])
    assert np.all(out["image"]["base_0_rgb"] == 0)
    assert np.all(out["image"]["left_wrist_0_rgb"] == 7)
    assert np.all(out["image"]["right_wrist_0_rgb"] == 0)
    assert "observation.images.phone" not in out


def test_vision_only_wrist_repack_does_not_fallback_to_phone_when_wrist_missing():
    sample = _sample()
    sample.pop("observation.images.wrist")

    out = VisionOnlyWristRepack()(sample)

    assert not bool(out["image_mask"]["base_0_rgb"])
    assert not bool(out["image_mask"]["left_wrist_0_rgb"])
    assert not bool(out["image_mask"]["right_wrist_0_rgb"])
    assert np.all(out["image"]["left_wrist_0_rgb"] == 0)


def test_vision_only_wrist_repack_converts_chw_float_to_hwc_uint8():
    sample = _sample()
    sample["observation.images.wrist"] = np.ones((3, 4, 5), dtype=np.float32)

    out = VisionOnlyWristRepack()(sample)

    assert out["image"]["left_wrist_0_rgb"].shape == (4, 5, 3)
    assert out["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert np.all(out["image"]["left_wrist_0_rgb"] == 255)
    assert out["image"]["base_0_rgb"].shape == (4, 5, 3)
