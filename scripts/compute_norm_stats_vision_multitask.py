"""Compute norm stats for wrist-camera-only multi-task pi0.5 training."""

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
from openpi.training.multimodal import MultiTaskConcatDataset
from openpi.training.multimodal import VisionOnlyWristRepack
from openpi.training.multimodal import load_task_weights
from openpi.training.multimodal import make_weighted_sampler
from openpi.training.multimodal import multimodal_collate_fn


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


DEFAULT_OUTPUT_DIR = _repo_root() / "assets" / "pi05_umi_vision_multitask" / "umi_multitask_vision"


def _build_loader(
    repo_ids: Sequence[str],
    *,
    data_root: str | None,
    episode_start: int | None,
    episode_end: int | None,
    action_horizon: int,
    use_full_state: bool,
    weights_json: str | None,
    alpha: float,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    import torch  # noqa: PLC0415

    datasets = [
        LeRobotMultiModalDataset(
            MultiModalDatasetSpec(
                repo_id=repo_id,
                action_horizon=action_horizon,
                force_window=1,
                tactile_frame_stack=1,
                prompt_from_task=True,
                root=data_root,
                episode_start=episode_start,
                episode_end=episode_end,
            )
        )
        for repo_id in repo_ids
    ]
    concat = MultiTaskConcatDataset(datasets)
    transform = VisionOnlyWristRepack(use_full_state=use_full_state)

    class _Wrapped(torch.utils.data.Dataset):
        def __len__(self):
            return len(concat)

        def __getitem__(self, idx):
            out = transform(concat[int(idx)])
            return {"state": out["state"], "actions": out["actions"]}

    task_weights = None
    if weights_json:
        meta = load_task_weights(weights_json)
        task_weights = {k: v["weight"] for k, v in meta["tasks"].items()}

    sampler = make_weighted_sampler(
        concat,
        alpha=alpha,
        task_weights=task_weights,
        num_samples=len(concat),
        generator=torch.Generator().manual_seed(seed),
    )
    loader = torch.utils.data.DataLoader(
        _Wrapped(),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )
    return loader, len(concat)


def _accumulate(loader, max_frames: int | None, batch_size: int):
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    target_batches = max(1, max_frames // batch_size) if max_frames is not None else None

    seen = 0
    for i, batch in enumerate(tqdm.tqdm(loader, total=target_batches, desc="Computing stats")):
        stats["state"].update(np.asarray(batch["state"], dtype=np.float32))
        stats["actions"].update(np.asarray(batch["actions"], dtype=np.float32))
        seen += batch_size
        if target_batches is not None and i + 1 >= target_batches:
            break
    return stats, seen


def _stabilize_constant_dims(stats: normalize.NormStats, eps: float = 1e-6) -> normalize.NormStats:
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32).copy()
    q01 = np.asarray(stats.q01, dtype=np.float32).copy()
    q99 = np.asarray(stats.q99, dtype=np.float32).copy()
    constant = (q99 - q01) <= eps
    if constant.any():
        std[constant] = 1.0
        q01[constant] = mean[constant] - 1.0
        q99[constant] = mean[constant] + 1.0
    return normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-ids", type=str, nargs="+", required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--weights-json", type=str, default="src/openpi/training/multimodal/task_weights.json")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-frames", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-full-state", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    print(f"Tasks: {args.repo_ids}")
    print(f"Output: {output_dir / 'norm_stats.json'}")

    loader, total_frames = _build_loader(
        args.repo_ids,
        data_root=args.data_root,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        action_horizon=args.action_horizon,
        use_full_state=args.use_full_state,
        weights_json=args.weights_json,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"ConcatDataset frames: {total_frames}; cap: {args.max_frames or 'all'}")

    stats, seen = _accumulate(loader, args.max_frames, args.batch_size)
    norm_stats = {key: _stabilize_constant_dims(value.get_statistics()) for key, value in stats.items()}
    normalize.save(output_dir, norm_stats)
    print(f"Done. Saw ~{seen} frames. Wrote {len(norm_stats)} stats to {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
