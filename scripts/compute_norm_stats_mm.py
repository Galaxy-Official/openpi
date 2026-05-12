"""Compute normalization statistics for the multi-modal pretraining pipeline.

Standalone counterpart to `scripts/compute_norm_stats.py` for the multi-task
multi-modal LeRobot pipeline introduced in PR1 (M0). Iterates the joint
ConcatDataset under the configured weighted sampler and accumulates
`RunningStats` over `state`, `actions`, `force/left`, `force/right` (and
optionally `state_phone`). Results are written as a flat
`{key: NormStats}` dict, matching the format consumed by
`openpi.transforms.Normalize`.

Image and tactile modalities are intentionally excluded: SigLIP and the
tactile encoder will apply their own per-modality normalization downstream.

Examples:
    # Default: all 8 tasks, 50k frames sample, output under `<repo>/debug_outputs/`.
    uv run scripts/compute_norm_stats_mm.py \
        --repo-ids 429_erase_board_lerobot 430_clamp_seal_lerobot 430_towel_hanging_lerobot \
                   501_bread_moving_lerobot 505_screw_lerobot 505_stiring_lerobot \
                   506_open_bottle_lerobot 506_peg_flowers_lerobot \
        --weights-json src/openpi/training/multimodal/task_weights.json \
        --max-frames 50000 \
        --batch-size 32 --num-workers 4

    # Single task smoke test
    uv run scripts/compute_norm_stats_mm.py --repo-id 430_clamp_seal_lerobot \
        --max-frames 1024
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
import pathlib

import numpy as np
import tqdm

import openpi.shared.normalize as normalize
from openpi.training.multimodal import LeRobotMultiModalDataset
from openpi.training.multimodal import MultiModalDatasetSpec
from openpi.training.multimodal import MultiModalRepack
from openpi.training.multimodal import MultiTaskConcatDataset
from openpi.training.multimodal import load_task_weights
from openpi.training.multimodal import make_weighted_sampler
from openpi.training.multimodal import multimodal_collate_fn

# Keys we accumulate stats for, in dotted form (used both as data lookup path and
# as the flat name in the saved norm_stats.json).
_STATE_KEYS_FULL: tuple[str, ...] = ("state", "actions")
_STATE_KEYS_OPTIONAL: tuple[str, ...] = ("state_phone",)
_FORCE_KEYS: tuple[str, ...] = ("force/left", "force/right")


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


DEFAULT_OUTPUT_DIR = _repo_root() / "debug_outputs"


def _select(batch: dict, dotted: str):
    """Resolve a `/`-delimited path through a nested dict and return the leaf."""
    node = batch
    for part in dotted.split("/"):
        node = node[part]
    return np.asarray(node)


def _shape_for_running_stats(arr: np.ndarray, dotted_key: str) -> np.ndarray:
    """Reshape a batched leaf into `(N, last_dim)` for RunningStats.update.

    RunningStats expects a 2-D array where each row is one sample and each
    column is one feature dimension. Force windows are scalar physical signals
    over time, so we collapse the time axis into the sample axis (one stat
    per side), not into the feature axis.
    """
    if dotted_key.startswith("force/"):
        # (B, T) -> (B*T, 1) so we get a single scalar stat per side.
        return arr.reshape(-1, 1).astype(np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(np.float32)
    return arr.reshape(-1, arr.shape[-1]).astype(np.float32)


def _build_loader(
    repo_ids: Sequence[str],
    *,
    data_root: str | None,
    action_horizon: int,
    force_window: int,
    tactile_stack: int,
    use_full_state: bool,
    weights_json: str | None,
    alpha: float,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    import torch  # noqa: PLC0415

    views = []
    for rid in repo_ids:
        spec = MultiModalDatasetSpec(
            repo_id=rid,
            action_horizon=action_horizon,
            force_window=force_window,
            tactile_frame_stack=tactile_stack,
            prompt_from_task=False,
            root=data_root,
        )
        views.append(LeRobotMultiModalDataset(spec))

    concat = MultiTaskConcatDataset(views)

    repack = MultiModalRepack(use_full_state=use_full_state, drop_seed=seed)

    class _Wrapped(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self._base = base
            self._transform = transform

        def __len__(self):
            return len(self._base)

        def __getitem__(self, idx):
            raw = self._base[int(idx)]
            out = self._transform(raw)
            if "observation.state_phone" in raw:
                out["state_phone"] = np.asarray(raw["observation.state_phone"], dtype=np.float32)
            return out

    dataset = _Wrapped(concat, repack)

    task_weights = None
    if weights_json:
        meta = load_task_weights(weights_json)
        task_weights = {k: v["weight"] for k, v in meta["tasks"].items()}

    gen = torch.Generator().manual_seed(seed)
    sampler = make_weighted_sampler(
        concat,
        alpha=alpha,
        task_weights=task_weights,
        num_samples=len(concat),
        generator=gen,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )
    return loader, len(concat)


def _accumulate(loader, max_frames: int | None, batch_size: int, keys: list[str]):
    stats = {key: normalize.RunningStats() for key in keys}
    target_batches = max(1, max_frames // batch_size) if max_frames is not None else None

    seen = 0
    for i, batch in enumerate(tqdm.tqdm(loader, total=target_batches, desc="Computing stats")):
        for key in keys:
            try:
                arr = _select(batch, key)
            except KeyError:
                continue
            arr = _shape_for_running_stats(arr, key)
            stats[key].update(arr)
        seen += batch_size
        if target_batches is not None and i + 1 >= target_batches:
            break

    return stats, seen


def _print_summary(stats: dict[str, normalize.RunningStats]) -> None:
    for key, rs in stats.items():
        try:
            ns = rs.get_statistics()
        except ValueError as exc:  # < 2 vectors
            print(f"  {key}: SKIP ({exc})")
            continue
        print(f"  {key}: mean={_fmt(ns.mean)}, std={_fmt(ns.std)}, q01={_fmt(ns.q01)}, q99={_fmt(ns.q99)}")


def _fmt(arr) -> str:
    arr = np.asarray(arr)
    if arr.size <= 4:
        return np.array2string(arr, precision=4, suppress_small=True)
    return f"<shape={tuple(arr.shape)} mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f}>"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, default=None, help="Single task to load.")
    parser.add_argument("--repo-ids", type=str, nargs="*", default=None, help="Multiple tasks for ConcatDataset.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Where to write norm_stats.json. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Cap the total number of frames sampled.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--force-window", type=int, default=16)
    parser.add_argument("--tactile-stack", type=int, default=1)
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Task sampling exponent (only if --weights-json absent)."
    )
    parser.add_argument("--weights-json", type=str, default=None)
    parser.add_argument(
        "--use-full-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use observation.state (10-d). With --no-use-full-state pulls observation.state_phone (6-d).",
    )
    parser.add_argument(
        "--include-state-phone",
        action="store_true",
        help="Also accumulate stats for state_phone even when use_full_state is True.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.repo_id is None and not args.repo_ids:
        parser.error("Provide --repo-id or --repo-ids.")

    repo_ids = [args.repo_id] if args.repo_id else list(args.repo_ids)
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    print(f"Tasks: {repo_ids}")
    print(f"Output: {output_dir / 'norm_stats.json'}")

    loader, total_frames = _build_loader(
        repo_ids,
        data_root=args.data_root,
        action_horizon=args.action_horizon,
        force_window=args.force_window,
        tactile_stack=args.tactile_stack,
        use_full_state=args.use_full_state,
        weights_json=args.weights_json,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    keys = list(_STATE_KEYS_FULL) + list(_FORCE_KEYS)
    if args.include_state_phone and args.use_full_state:
        keys.append(_STATE_KEYS_OPTIONAL[0])
    print(f"Accumulating stats for: {keys}")
    print(f"ConcatDataset frames: {total_frames}; cap: {args.max_frames or 'all'}")

    stats, seen = _accumulate(loader, args.max_frames, args.batch_size, keys)
    print(f"\nDone. Saw ~{seen} frames.\nSummary:")
    _print_summary(stats)

    norm_stats = {}
    for key, rs in stats.items():
        try:
            norm_stats[key] = rs.get_statistics()
        except ValueError as exc:
            logging.warning("Skipping %s: %s", key, exc)
    normalize.save(output_dir, norm_stats)
    print(f"\nWrote {len(norm_stats)} stats to {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
