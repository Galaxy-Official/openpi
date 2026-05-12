"""Unit tests for compute_task_weights and the multi-task sampler.

Uses tiny in-memory stub datasets so it does not depend on lerobot or any
real task data on disk.
"""

from __future__ import annotations

import collections

import numpy as np
import torch

from openpi.training.multimodal.mix_sampler import MultiTaskConcatDataset
from openpi.training.multimodal.mix_sampler import compute_task_weights
from openpi.training.multimodal.mix_sampler import make_weighted_sampler


class _StubDataset:
    """Minimal random-access dataset stub for the sampler tests."""

    def __init__(self, name: str, n: int):
        self.task_name = name
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        return idx


def test_compute_task_weights_alpha_zero_is_uniform():
    w = compute_task_weights([10, 1000, 30], alpha=0.0)
    np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3])


def test_compute_task_weights_alpha_one_is_proportional():
    w = compute_task_weights([10, 30, 60], alpha=1.0)
    np.testing.assert_allclose(w, [0.1, 0.3, 0.6])


def test_compute_task_weights_alpha_half_is_sqrt():
    counts = [100, 400, 900]
    w = compute_task_weights(counts, alpha=0.5)
    raw = np.sqrt(counts)
    np.testing.assert_allclose(w, raw / raw.sum())


def test_concat_dataset_metadata():
    ds = MultiTaskConcatDataset([_StubDataset("a", 5), _StubDataset("b", 7), _StubDataset("c", 3)])
    assert list(ds.task_names) == ["a", "b", "c"]
    assert list(ds.task_frame_counts) == [5, 7, 3]
    assert ds.task_index_for(0) == 0
    assert ds.task_index_for(4) == 0
    assert ds.task_index_for(5) == 1
    assert ds.task_index_for(11) == 1
    assert ds.task_index_for(12) == 2
    assert ds.task_index_for(14) == 2


def test_weighted_sampler_approaches_target_distribution():
    # Highly imbalanced task sizes; with alpha=0.5, the sampler should produce a
    # distribution that is much more balanced than the raw frame ratios.
    ds = MultiTaskConcatDataset([_StubDataset("small", 100), _StubDataset("big", 10000)])
    gen = torch.Generator().manual_seed(0)
    sampler = make_weighted_sampler(ds, alpha=0.5, num_samples=20000, generator=gen)
    counts = collections.Counter()
    for flat_idx in sampler:
        counts[ds.task_index_for(int(flat_idx))] += 1

    # Expected weights (alpha=0.5): proportional to sqrt(100)=10 and sqrt(10000)=100 -> 0.0909 / 0.9091
    # But per-frame weight is task_weight / num_frames, and num_samples sums over both tasks, so
    # we expect task frequencies ~ task_weight (not raw).
    total = sum(counts.values())
    freq_small = counts[0] / total
    freq_big = counts[1] / total
    # Tolerate Monte Carlo noise.
    assert 0.05 < freq_small < 0.15
    assert 0.85 < freq_big < 0.95


def test_weighted_sampler_explicit_task_weights_override():
    ds = MultiTaskConcatDataset([_StubDataset("a", 1000), _StubDataset("b", 1000)])
    gen = torch.Generator().manual_seed(0)
    # Heavily bias towards task "a".
    sampler = make_weighted_sampler(ds, alpha=0.5, num_samples=4000, task_weights={"a": 0.9, "b": 0.1}, generator=gen)
    counts = collections.Counter()
    for flat_idx in sampler:
        counts[ds.task_index_for(int(flat_idx))] += 1
    total = sum(counts.values())
    assert counts[0] / total > 0.8
    assert counts[1] / total < 0.2
