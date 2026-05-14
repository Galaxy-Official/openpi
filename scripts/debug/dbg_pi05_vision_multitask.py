"""Debug the wrist-camera-only multi-task pi0.5 data path."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import dataclasses

import torch

import openpi.training.config as _config
import openpi.training.data_loader as _data


def _shape(x) -> str:
    return f"shape={tuple(x.shape)} dtype={x.dtype}"


def _print_sampler_preview(config: _config.TrainConfig, sample_indices: int) -> None:
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = _data.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    sampler = _data.create_multitask_weighted_sampler(raw, data_config, seed=config.seed)
    if sampler is None or not hasattr(raw, "task_index_for"):
        return

    counts: Counter[str] = Counter()
    for i, idx in enumerate(iter(sampler)):
        if i >= sample_indices:
            break
        task_idx = raw.task_index_for(int(idx))
        counts[raw.task_names[task_idx]] += 1

    print(f"Sampler preview over {sum(counts.values())} draws:")
    for name, count in counts.most_common():
        print(f"  {name}: {count / max(sum(counts.values()), 1):.4f} ({count})")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-ids", type=str, nargs="+", required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--weights-json", type=str, default="src/openpi/training/multimodal/task_weights.json")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--sample-indices", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    base = _config.get_config("pi05_umi_vision_multitask")
    data = dataclasses.replace(
        base.data,
        repo_ids=tuple(args.repo_ids),
        data_root=args.data_root,
        weights_json=args.weights_json,
        alpha=args.alpha,
    )
    config = dataclasses.replace(
        base,
        data=data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_batches,
        seed=args.seed,
        exp_name="debug_pi05_vision_multitask",
        wandb_enabled=False,
    )

    _print_sampler_preview(config, args.sample_indices)
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
        num_batches=args.num_batches,
        skip_norm_stats=True,
    )

    for batch_idx, (obs, actions) in enumerate(loader):
        print(f"\n=== batch {batch_idx} ===")
        print(f"state: {_shape(obs.state)}")
        print(f"actions: {_shape(actions)}")
        print(f"tokenized_prompt: {_shape(obs.tokenized_prompt)}")
        for key, image in obs.images.items():
            print(f"image.{key}: {_shape(image)}")
        for key, mask in obs.image_masks.items():
            print(
                f"image_mask.{key}: {_shape(mask)} "
                f"true={int(mask.to(torch.bool).sum().item())}/{mask.numel()}"
            )

        if obs.image_masks["base_0_rgb"].any():
            raise RuntimeError("base_0_rgb must stay masked out for wrist-only training")
        if obs.image_masks["right_wrist_0_rgb"].any():
            raise RuntimeError("right_wrist_0_rgb must stay masked out for wrist-only training")
        if not obs.image_masks["left_wrist_0_rgb"].any():
            raise RuntimeError("left_wrist_0_rgb has no valid samples in this batch")


if __name__ == "__main__":
    main()
