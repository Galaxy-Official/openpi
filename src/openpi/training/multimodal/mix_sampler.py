"""Weighted multi-task sampling over a `ConcatDataset` of LeRobot tasks.

The weighted sampler implements `weight_i = frames_i ** alpha`, normalized.
`alpha=0` is uniform sampling across tasks, `alpha=1` is proportional to data
volume. The default `alpha=0.5` is a square-root reweighting that mildly
favors larger tasks while preventing small tasks from being starved.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import pathlib
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import WeightedRandomSampler


@runtime_checkable
class _TaskDataset(Protocol):
    """Structural type: anything with `__len__` and a `task_name` attribute."""

    task_name: str

    def __len__(self) -> int: ...


class MultiTaskConcatDataset(ConcatDataset):
    """`ConcatDataset` that also exposes per-frame task names and offsets.

    The base `ConcatDataset` only stores cumulative sizes; this subclass adds
    helpers so a weighted sampler can compute per-frame sampling weights and
    so debugging can recover which task a flat index belongs to.
    """

    def __init__(self, datasets: Sequence[_TaskDataset]):
        super().__init__(list(datasets))
        self._task_names: list[str] = [d.task_name for d in datasets]
        # Cumulative starts (length == len(datasets)+1).
        self._cum_starts: list[int] = [0]
        for d in datasets:
            self._cum_starts.append(self._cum_starts[-1] + len(d))

    @property
    def task_names(self) -> list[str]:
        return list(self._task_names)

    @property
    def task_frame_counts(self) -> list[int]:
        return [self._cum_starts[i + 1] - self._cum_starts[i] for i in range(len(self._task_names))]

    def task_index_for(self, flat_index: int) -> int:
        # Binary search would be cleaner, but len(datasets) is tiny.
        for i in range(len(self._task_names)):
            if self._cum_starts[i] <= flat_index < self._cum_starts[i + 1]:
                return i
        raise IndexError(flat_index)


def _normalize(weights: np.ndarray) -> np.ndarray:
    s = float(weights.sum())
    if s <= 0:
        return np.full_like(weights, 1.0 / max(len(weights), 1))
    return weights / s


def compute_task_weights(frame_counts: Sequence[int], alpha: float = 0.5) -> np.ndarray:
    """`w_i = frames_i ** alpha`, normalized to sum to 1."""
    counts = np.asarray(frame_counts, dtype=np.float64)
    if (counts <= 0).any():
        raise ValueError(f"All frame counts must be positive, got {counts.tolist()}")
    return _normalize(counts**alpha)


def make_weighted_sampler(
    dataset: MultiTaskConcatDataset,
    *,
    alpha: float = 0.5,
    num_samples: int | None = None,
    replacement: bool = True,
    task_weights: dict[str, float] | None = None,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Build a `WeightedRandomSampler` over the concat dataset.

    - If `task_weights` is provided (keyed by task name), it overrides the
      `alpha`-derived defaults. Missing tasks fall back to the alpha default.
    - Per-frame weight is `task_weight / num_frames_in_task` so each task is
      sampled in expectation proportional to `task_weight` regardless of size.
    """
    counts = np.asarray(dataset.task_frame_counts, dtype=np.float64)
    default_w = compute_task_weights(counts.astype(int), alpha=alpha)

    final_w = default_w.copy()
    if task_weights:
        explicit = np.zeros_like(default_w)
        any_set = False
        for i, name in enumerate(dataset.task_names):
            if name in task_weights:
                explicit[i] = float(task_weights[name])
                any_set = True
            else:
                explicit[i] = default_w[i]
        if any_set:
            final_w = _normalize(explicit)

    # Per-frame weight = task_weight / task_frames
    per_frame = np.empty(int(counts.sum()), dtype=np.float64)
    cursor = 0
    for i, n in enumerate(counts.astype(int)):
        per_frame[cursor : cursor + n] = final_w[i] / max(n, 1)
        cursor += n

    weights_t = torch.as_tensor(per_frame, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights_t,
        num_samples=num_samples if num_samples is not None else int(counts.sum()),
        replacement=replacement,
        generator=generator,
    )


def load_task_weights(path: str | pathlib.Path) -> dict:
    """Load a task-weights JSON file.

    Schema:
        {
            "alpha": 0.5,
            "tasks": {
                "<task_name>": {"frames": 31342, "weight": 0.124},
                ...
            }
        }
    Only the `weight` field is used by the sampler; the rest is for traceability.
    """
    with open(path) as f:
        return json.load(f)
