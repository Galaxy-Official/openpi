"""Unit tests for LeRobotMultiModalDataset helper logic."""

from __future__ import annotations

import types

import numpy as np

from openpi.training.multimodal.lerobot_mm_dataset import ForceLeftKey
from openpi.training.multimodal.lerobot_mm_dataset import LeRobotMultiModalDataset
from openpi.training.multimodal.lerobot_mm_dataset import MultiModalDatasetSpec
from openpi.training.multimodal.lerobot_mm_dataset import _build_task_index_map  # noqa: SLF001
from openpi.training.multimodal.lerobot_mm_dataset import _prompt_from_raw_or_index  # noqa: SLF001
from openpi.training.multimodal.lerobot_mm_dataset import _squeeze_scalar_force  # noqa: SLF001


class _FakeTasksFrame:
    def iterrows(self):
        yield "clamp the seal", {"task_index": 0}
        yield "erase the board", {"task_index": 7}


def test_build_task_index_map_handles_lerobot_v3_tasks_frame():
    assert _build_task_index_map(_FakeTasksFrame()) == {
        0: "clamp the seal",
        7: "erase the board",
    }


def test_prompt_prefers_lerobot_task_string_when_present():
    prompt = _prompt_from_raw_or_index({"task": "tighten the screw"}, {"task_index": np.asarray(3)}, {3: "fallback"})
    assert prompt == "tighten the screw"


def test_prompt_falls_back_to_task_index_map():
    prompt = _prompt_from_raw_or_index({}, {"task_index": np.asarray([7])}, {7: "erase the board"})
    assert prompt == "erase the board"


def test_scalar_force_window_squeezes_trailing_unit_dim():
    arr = _squeeze_scalar_force(np.zeros((16, 1), dtype=np.float32))
    assert arr.shape == (16,)
    assert arr.dtype == np.float32


def test_missing_force_zero_placeholder_uses_temporal_vector_shape():
    dataset = object.__new__(LeRobotMultiModalDataset)
    dataset.spec = MultiModalDatasetSpec(repo_id="fake", force_window=16)
    dataset._meta = types.SimpleNamespace(features={ForceLeftKey: {"shape": [1], "dtype": "float32"}})  # noqa: SLF001

    zeros = dataset._zero_for(ForceLeftKey)  # noqa: SLF001

    assert zeros.shape == (16,)
    assert zeros.dtype == np.float32
