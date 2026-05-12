"""LeRobot dataset wrapper that exposes force/tactile windows.

Wraps `lerobot.datasets.lerobot_dataset.LeRobotDataset` and adds:
- action horizon window (already supported by LeRobotDataset via `delta_timestamps`)
- force window (e.g. last T frames) for left/right channels
- single-frame or stacked tactile videos for left/right channels
- pass-through of latent features when present in the meta

The output sample uses LeRobot's native key names (e.g. `observation.forces.left`).
Use `MultiModalRepack` afterwards to map those to pi0.5 expected key names.

Data root resolution: see `openpi.training.multimodal.paths.resolve_data_root`.
The resolved value is treated as the parent directory of `{repo_id}/`.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib
from typing import Any

import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np

from openpi.training.multimodal.paths import DATA_ROOT_ENV
from openpi.training.multimodal.paths import resolve_data_root

ActionKey = "action"
ForceLeftKey = "observation.forces.left"
ForceRightKey = "observation.forces.right"
TactileLeftKey = "observation.tactiles.left"
TactileRightKey = "observation.tactiles.right"
ImagePhoneKey = "observation.images.phone"
ImageWristKey = "observation.images.wrist"

__all__ = [
    "DATA_ROOT_ENV",
    "LeRobotMultiModalDataset",
    "MultiModalDatasetSpec",
    "resolve_data_root",
]


@dataclasses.dataclass(frozen=True)
class MultiModalDatasetSpec:
    """Per-task configuration for the multi-modal dataset wrapper."""

    repo_id: str
    action_horizon: int = 50
    # Number of past frames to include for the force channels.
    # The window covers [t - force_window + 1, t] in frame indices.
    force_window: int = 16
    # Number of frames to stack for tactile videos. 1 means a single current frame.
    tactile_frame_stack: int = 1
    # If true, set prompt from the LeRobot tasks.parquet via `task_index`.
    prompt_from_task: bool = True
    # Optional stable name used by samplers/loggers; defaults to repo_id.
    name: str | None = None
    # Parent directory containing `{repo_id}/`. None triggers `resolve_data_root()`.
    root: str | None = None

    @property
    def task_name(self) -> str:
        return self.name or self.repo_id


def _build_delta_timestamps(spec: MultiModalDatasetSpec, fps: float) -> dict[str, list[float]]:
    dt = 1.0 / fps
    delta: dict[str, list[float]] = {}

    # Action horizon: frames [t, t+1, ..., t + action_horizon - 1].
    delta[ActionKey] = [i * dt for i in range(spec.action_horizon)]

    # Force window: frames [t - W + 1, ..., t].
    if spec.force_window > 1:
        window = [(i - spec.force_window + 1) * dt for i in range(spec.force_window)]
        delta[ForceLeftKey] = window
        delta[ForceRightKey] = window

    # Tactile frame stack: frames [t - K + 1, ..., t].
    if spec.tactile_frame_stack > 1:
        stack = [(i - spec.tactile_frame_stack + 1) * dt for i in range(spec.tactile_frame_stack)]
        delta[TactileLeftKey] = stack
        delta[TactileRightKey] = stack

    return delta


class LeRobotMultiModalDataset:
    """Random-access dataset that yields per-frame multi-modal samples.

    Each sample is a flat `dict[str, np.ndarray]` containing at least:
      - `action`                       float32, (action_horizon, action_dim)
      - `observation.state`            float32, (state_dim,)
      - `observation.forces.left`      float32, (force_window,)
      - `observation.forces.right`     float32, (force_window,)
      - `observation.images.phone`     uint8 or float32, (H, W, 3)
      - `observation.images.wrist`     uint8 or float32, (H, W, 3)
      - `observation.tactiles.left`    uint8 or float32, (Ht, Wt, 3) [or with stack dim]
      - `observation.tactiles.right`   uint8 or float32, (Ht, Wt, 3) [or with stack dim]
      - `task_index`                   int
      - `episode_index`                int
      - `frame_index`                  int
      - `timestamp`                    float
      - `prompt`                       str (if `prompt_from_task=True`)
      - `task_name`                    str (the configured task identifier)

    Missing optional features are filled with zeros and a corresponding
    `<key>.mask` field is set to `False`. Present features get a mask of `True`.
    The mask convention matches pi0.5's `image_mask`-style usage.
    """

    # Keys that are conditionally present depending on the task's features.
    _OPTIONAL_KEYS: tuple[str, ...] = (
        ForceLeftKey,
        ForceRightKey,
        TactileLeftKey,
        TactileRightKey,
        ImagePhoneKey,
        ImageWristKey,
    )

    def __init__(self, spec: MultiModalDatasetSpec):
        self.spec = spec
        data_root = resolve_data_root(spec.root)
        task_root = data_root / spec.repo_id
        if not task_root.exists():
            raise FileNotFoundError(
                f"LeRobot task directory not found: {task_root}. "
                f"Set spec.root or ${DATA_ROOT_ENV} to the parent directory of {spec.repo_id}/."
            )
        self._task_root = task_root

        self._meta = lerobot_dataset.LeRobotDatasetMetadata(spec.repo_id, root=task_root)
        fps = float(self._meta.fps)

        delta_timestamps = _build_delta_timestamps(spec, fps)
        self._dataset = lerobot_dataset.LeRobotDataset(
            spec.repo_id,
            root=task_root,
            delta_timestamps=delta_timestamps,
        )
        # Cache the set of feature keys actually declared by this task.
        self._features = set(self._meta.features.keys())
        self._tasks_map: dict[int, str] = dict(self._meta.tasks)

    @property
    def task_root(self) -> pathlib.Path:
        return self._task_root

    @property
    def task_name(self) -> str:
        return self.spec.task_name

    @property
    def num_frames(self) -> int:
        return len(self._dataset)

    def __len__(self) -> int:
        return len(self._dataset)

    def has_feature(self, key: str) -> bool:
        return key in self._features

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self._dataset[int(index)]

        out: dict[str, Any] = {}

        # Copy required scalar / state fields verbatim when present.
        for key in (
            "observation.state",
            "observation.state_phone",
            "task_index",
            "episode_index",
            "frame_index",
            "timestamp",
            "index",
            "start_pose",
            "end_pose",
        ):
            if key in raw:
                out[key] = _to_numpy(raw[key])

        # Action sequence (action_horizon, action_dim).
        if ActionKey in raw:
            out[ActionKey] = _to_numpy(raw[ActionKey]).astype(np.float32, copy=False)

        # Optional modalities -> fill with zeros + mask=False if missing.
        for key in self._OPTIONAL_KEYS:
            if key in raw:
                out[key] = _to_numpy(raw[key])
                out[f"{key}.mask"] = np.array(True, dtype=bool)  # noqa: FBT003
            else:
                out[key] = self._zero_for(key)
                out[f"{key}.mask"] = np.array(False, dtype=bool)  # noqa: FBT003

        # Prompt from task index.
        if self.spec.prompt_from_task:
            tidx = int(out.get("task_index", 0))
            out["prompt"] = self._tasks_map.get(tidx, "")

        out["task_name"] = self.spec.task_name
        return out

    def _zero_for(self, key: str) -> np.ndarray:
        """Build a zero placeholder of the expected shape for a missing modality."""
        feat = self._meta.features.get(key, None)
        shape: Sequence[int]
        if feat is None:
            # Reasonable fallbacks if the feature is not declared at all.
            if key in (ForceLeftKey, ForceRightKey):
                shape = (max(self.spec.force_window, 1),)
            elif key in (TactileLeftKey, TactileRightKey):
                shape = (
                    (224, 224, 3)
                    if self.spec.tactile_frame_stack <= 1
                    else (self.spec.tactile_frame_stack, 224, 224, 3)
                )
            elif key in (ImagePhoneKey, ImageWristKey):
                shape = (480, 640, 3)
            else:
                shape = (1,)
            return np.zeros(shape, dtype=np.float32)

        declared_shape = tuple(feat.get("shape", ()))
        # Prepend a time axis for windowed modalities.
        if key in (ForceLeftKey, ForceRightKey) and self.spec.force_window > 1:
            shape = (self.spec.force_window, *declared_shape)
        elif key in (TactileLeftKey, TactileRightKey) and self.spec.tactile_frame_stack > 1:
            shape = (self.spec.tactile_frame_stack, *declared_shape)
        else:
            shape = declared_shape

        dtype = np.float32
        decl_dtype = feat.get("dtype", "float32")
        if decl_dtype in ("uint8", "image", "video"):
            dtype = np.uint8
        return np.zeros(shape, dtype=dtype)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    # torch.Tensor and similar: rely on the `__array__` protocol.
    try:
        return np.asarray(value)
    except Exception:
        return np.asarray(value, dtype=object)
