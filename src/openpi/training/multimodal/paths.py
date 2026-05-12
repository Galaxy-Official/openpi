"""Data root resolution for the multi-modal pipeline.

This module is intentionally free of heavy imports (no lerobot/torch) so it can
be exercised by unit tests in any environment.
"""

from __future__ import annotations

import os
import pathlib

# Env var override for the dataset root directory (parent of `{repo_id}`).
DATA_ROOT_ENV = "OPENPI_DATA_ROOT"


def _repo_root() -> pathlib.Path:
    # This file lives at <repo_root>/src/openpi/training/multimodal/paths.py.
    return pathlib.Path(__file__).resolve().parents[4]


def resolve_data_root(explicit: str | os.PathLike | None = None) -> pathlib.Path:
    """Resolve the dataset root directory.

    The returned path is the *parent* of each task directory (i.e. it should
    contain `429_erase_board_lerobot/`, `430_clamp_seal_lerobot/`, ...).

    Precedence:
        1. `explicit` argument if provided.
        2. `$OPENPI_DATA_ROOT` if set.
        3. `<repo_root>/../openpi/Data` if it exists (cloud server layout).
        4. `<repo_root>/Data` (in-tree fallback, useful for local exploration).
    """
    if explicit is not None:
        return pathlib.Path(explicit).expanduser().resolve()

    env = os.environ.get(DATA_ROOT_ENV)
    if env:
        return pathlib.Path(env).expanduser().resolve()

    repo_root = _repo_root()
    sibling = (repo_root.parent / "openpi" / "Data").resolve()
    if sibling.exists():
        return sibling

    return (repo_root / "Data").resolve()
