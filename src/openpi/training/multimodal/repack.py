"""Multi-modal repack transform: LeRobot keys -> pi0.5 expected keys.

Maps the flat dict produced by `LeRobotMultiModalDataset` into the structure
expected by the pi0.5 model preprocessing:

    {
        "image": {"base_0_rgb": ..., "left_wrist_0_rgb": ..., "right_wrist_0_rgb": ...},
        "image_mask": {"base_0_rgb": ..., "left_wrist_0_rgb": ..., "right_wrist_0_rgb": ...},
        "state": ...,
        "actions": ...,
        "tactile": {"left": ..., "right": ...},
        "tactile_mask": {"left": ..., "right": ...},
        "force": {"left": ..., "right": ...},
        "force_mask": {"left": ..., "right": ...},
        "task_index": ..., "episode_index": ..., "frame_index": ...,
        "prompt": "...",
    }

Image dropout / modality dropout is implemented here so it composes cleanly
with the rest of the transform pipeline.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import openpi.transforms as _transforms


def _scalar_bool(value) -> bool:
    arr = np.asarray(value)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


@dataclasses.dataclass(frozen=True)
class MultiModalRepack(_transforms.DataTransformFn):
    """Repacks LeRobot-flat sample dicts into pi0.5 nested format.

    Drop probabilities apply per-sample (set the corresponding mask to False and
    zero the data). Use a fixed seed in the worker process for reproducibility.
    """

    # Probability of zeroing out a whole modality stream for a given sample.
    tactile_drop: float = 0.0
    force_drop: float = 0.0
    vision_phone_drop: float = 0.0
    vision_wrist_drop: float = 0.0
    # If False, the right_wrist_0_rgb image slot is filled with zeros + mask False
    # (this dataset has only one wrist view).
    has_right_wrist_image: bool = False
    # If True, will use `state` (10-dim) as the main state. Otherwise pulls
    # `state_phone` (6-dim). pi0.5 expects state_dim == action_dim by default.
    use_full_state: bool = True
    # Seed offset used to derive a per-sample RNG when drops are non-zero. The
    # actual seed combines this with `index` / `frame_index` for reproducibility.
    drop_seed: int = 0

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        rng = self._rng_from_sample(data)

        # ---- State + actions --------------------------------------------------
        state_key = "observation.state" if self.use_full_state else "observation.state_phone"
        state = np.asarray(data[state_key], dtype=np.float32)
        actions = np.asarray(data["action"], dtype=np.float32) if "action" in data else None

        # ---- Vision images ----------------------------------------------------
        phone = self._as_image(data.get("observation.images.phone"))
        wrist = self._as_image(data.get("observation.images.wrist"))

        phone_mask = phone is not None and not self._roll_drop(rng, self.vision_phone_drop)
        wrist_mask = wrist is not None and not self._roll_drop(rng, self.vision_wrist_drop)

        image = {
            "base_0_rgb": phone if phone is not None else np.zeros((224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": wrist if wrist is not None else np.zeros((224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        }
        if not phone_mask:
            image["base_0_rgb"] = np.zeros_like(image["base_0_rgb"])
        if not wrist_mask:
            image["left_wrist_0_rgb"] = np.zeros_like(image["left_wrist_0_rgb"])
        image_mask = {
            "base_0_rgb": np.asarray(phone_mask, dtype=bool),
            "left_wrist_0_rgb": np.asarray(wrist_mask, dtype=bool),
            "right_wrist_0_rgb": np.asarray(self.has_right_wrist_image, dtype=bool),
        }

        # ---- Tactile ----------------------------------------------------------
        tac_l = data.get("observation.tactiles.left")
        tac_r = data.get("observation.tactiles.right")
        tac_l_present = _scalar_bool(data.get("observation.tactiles.left.mask", False))
        tac_r_present = _scalar_bool(data.get("observation.tactiles.right.mask", False))
        drop_tac = self._roll_drop(rng, self.tactile_drop)
        tac_l_ok = tac_l_present and not drop_tac and tac_l is not None
        tac_r_ok = tac_r_present and not drop_tac and tac_r is not None

        tactile = {
            "left": np.asarray(tac_l)
            if tac_l_ok
            else np.zeros_like(np.asarray(tac_l))
            if tac_l is not None
            else np.zeros((224, 224, 3), dtype=np.uint8),
            "right": np.asarray(tac_r)
            if tac_r_ok
            else np.zeros_like(np.asarray(tac_r))
            if tac_r is not None
            else np.zeros((224, 224, 3), dtype=np.uint8),
        }
        tactile_mask = {
            "left": np.asarray(tac_l_ok, dtype=bool),
            "right": np.asarray(tac_r_ok, dtype=bool),
        }

        # ---- Force ------------------------------------------------------------
        f_l = data.get("observation.forces.left")
        f_r = data.get("observation.forces.right")
        f_l_present = _scalar_bool(data.get("observation.forces.left.mask", False))
        f_r_present = _scalar_bool(data.get("observation.forces.right.mask", False))
        drop_force = self._roll_drop(rng, self.force_drop)
        f_l_ok = f_l_present and not drop_force and f_l is not None
        f_r_ok = f_r_present and not drop_force and f_r is not None

        force = {
            "left": np.asarray(f_l, dtype=np.float32)
            if f_l_ok
            else np.zeros_like(np.asarray(f_l), dtype=np.float32)
            if f_l is not None
            else np.zeros((1,), dtype=np.float32),
            "right": np.asarray(f_r, dtype=np.float32)
            if f_r_ok
            else np.zeros_like(np.asarray(f_r), dtype=np.float32)
            if f_r is not None
            else np.zeros((1,), dtype=np.float32),
        }
        force_mask = {
            "left": np.asarray(f_l_ok, dtype=bool),
            "right": np.asarray(f_r_ok, dtype=bool),
        }

        # ---- Indices / prompt -------------------------------------------------
        out: dict = {
            "image": image,
            "image_mask": image_mask,
            "state": state,
            "tactile": tactile,
            "tactile_mask": tactile_mask,
            "force": force,
            "force_mask": force_mask,
        }
        if actions is not None:
            out["actions"] = actions
        for k in ("task_index", "episode_index", "frame_index", "index", "timestamp"):
            if k in data:
                out[k] = np.asarray(data[k])
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        if "task_name" in data:
            out["task_name"] = data["task_name"]
        return out

    @staticmethod
    def _as_image(value) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value)

    def _rng_from_sample(self, data) -> np.random.Generator:
        # Combine drop_seed with stable per-sample identifiers when available.
        salt = self.drop_seed
        for k in ("index", "frame_index", "episode_index"):
            if k in data:
                salt = (salt * 1_000_003) ^ int(np.asarray(data[k]).reshape(-1)[0])
        return np.random.default_rng(salt & 0xFFFFFFFF)

    def _roll_drop(self, rng: np.random.Generator, p: float) -> bool:
        if p <= 0.0:
            return False
        return bool(rng.random() < p)
