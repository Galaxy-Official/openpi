"""Lightweight tests for the multi-modal norm-stats helpers.

These avoid importing lerobot/torch by exercising the script-local helpers
that operate on numpy arrays only.
"""

from __future__ import annotations

import numpy as np

from . import compute_norm_stats_mm as cns


def test_select_walks_nested_dict():
    batch = {"force": {"left": np.arange(4, dtype=np.float32)}, "state": np.zeros((2, 10))}
    assert np.array_equal(cns._select(batch, "force/left"), np.arange(4, dtype=np.float32))  # noqa: SLF001
    assert cns._select(batch, "state").shape == (2, 10)  # noqa: SLF001


def test_force_reshape_collapses_time_axis_to_one_feature():
    arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
    out = cns._shape_for_running_stats(arr, "force/left")  # noqa: SLF001
    assert out.shape == (8 * 16, 1)
    assert out.dtype == np.float32


def test_state_reshape_keeps_last_dim():
    arr = np.zeros((4, 10), dtype=np.float32)
    out = cns._shape_for_running_stats(arr, "state")  # noqa: SLF001
    assert out.shape == (4, 10)


def test_action_reshape_collapses_horizon_into_samples():
    arr = np.zeros((4, 50, 10), dtype=np.float32)
    out = cns._shape_for_running_stats(arr, "actions")  # noqa: SLF001
    assert out.shape == (200, 10)


def test_one_dim_array_promoted_to_one_feature():
    arr = np.arange(7, dtype=np.float32)
    out = cns._shape_for_running_stats(arr, "scalar_thing")  # noqa: SLF001
    assert out.shape == (7, 1)


def test_default_output_dir_is_repo_debug_outputs():
    assert cns.DEFAULT_OUTPUT_DIR == cns._repo_root() / "debug_outputs"  # noqa: SLF001
