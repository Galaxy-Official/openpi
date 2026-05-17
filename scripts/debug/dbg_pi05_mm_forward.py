"""Smoke-test PI05Multimodal end-to-end with a synthetic batch.

Builds a real-sized backbone (gemma_2b + gemma_300m by default) so the
forward + backward actually exercises the prefix/suffix path. The data side
is mocked: we hand-build an `Observation` so we can exercise the model
independently of the data pipeline.

Note: the upstream `dummy` paligemma variant has a hard-coded
`vision_config.projection_dim = 2048` in `gemma_pytorch.py:38` that does
not match its `text_config.hidden_size`, so its `forward` does not work
in either `PI0Pytorch` or `PI05Multimodal`. Use the real-size defaults.

Examples:
    # Default real-size run, prints loss_dict for all 3 phases and verifies backward.
    uv run scripts/debug/dbg_pi05_mm_forward.py

    # Override to a smaller LLM if you only want to test the encoder layout.
    uv run scripts/debug/dbg_pi05_mm_forward.py --paligemma-variant gemma_2b --action-expert-variant gemma_300m
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from openpi.models.model import Observation
from openpi.models.pi0_config_mm import Pi0ConfigMM
from openpi.models_pytorch.pi05_multimodal_pytorch import PI05Multimodal


def _build_observation(batch_size: int, action_dim: int, max_token_len: int, force_window: int) -> Observation:
    images = {
        "base_0_rgb": torch.zeros(batch_size, 3, 224, 224),
        "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224) * 0.1,
        "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224),
    }
    image_masks = {
        "base_0_rgb": torch.zeros(batch_size, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool),
    }
    state = torch.randn(batch_size, action_dim) * 0.1
    tokenized_prompt = torch.zeros(batch_size, max_token_len, dtype=torch.int32)
    tokenized_prompt_mask = torch.ones(batch_size, max_token_len, dtype=torch.bool)
    tactile = {
        "left": torch.randn(batch_size, 224, 224, 3) * 0.1,
        "right": torch.randn(batch_size, 224, 224, 3) * 0.1,
    }
    tactile_mask = {
        "left": torch.ones(batch_size, dtype=torch.bool),
        "right": torch.ones(batch_size, dtype=torch.bool),
    }
    force = {
        "left": torch.randn(batch_size, force_window) * 0.5,
        "right": torch.randn(batch_size, force_window) * 0.5,
    }
    force_mask = {
        "left": torch.ones(batch_size, dtype=torch.bool),
        "right": torch.ones(batch_size, dtype=torch.bool),
    }
    # Two different task indices + assorted frame indices so the contrastive
    # head's task-aware negative mask actually does something.
    task_index = torch.tensor([i % 2 for i in range(batch_size)], dtype=torch.int64)
    frame_index = torch.arange(batch_size, dtype=torch.int64) * 20

    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        tactile=tactile,
        tactile_mask=tactile_mask,
        force=force,
        force_mask=force_mask,
        task_index=task_index,
        frame_index=frame_index,
    )


def _format_loss(loss_dict: dict) -> str:
    pieces = [
        f"{k}={loss_dict[k]}"
        for k in ("phase", "global_step", "contrast_warmup")
        if k in loss_dict and loss_dict[k] is not None
    ]
    for k in ("total", "action_loss", "contrast_loss", "pos_sim", "neg_sim", "acc@1"):
        v = loss_dict.get(k)
        if v is None:
            pieces.append(f"{k}=None")
        else:
            pieces.append(f"{k}={float(v):.4f}")
    return "  ".join(pieces)


def _count_trainable(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paligemma-variant", type=str, default="gemma_2b")
    parser.add_argument("--action-expert-variant", type=str, default="gemma_300m")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--max-token-len", type=int, default=12)
    parser.add_argument("--force-window", type=int, default=16)
    parser.add_argument("--encoder-warmup-steps", type=int, default=2)
    parser.add_argument("--contrast-warmup-start", type=int, default=4)
    parser.add_argument("--contrast-warmup-end", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-backward", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    cfg = Pi0ConfigMM(
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        pytorch_compile_mode=None,
        encoder_warmup_steps=args.encoder_warmup_steps,
        contrast_warmup_start=args.contrast_warmup_start,
        contrast_warmup_end=args.contrast_warmup_end,
        force_window=args.force_window,
    )
    print(f"Config: {cfg}")
    model = PI05Multimodal(cfg)
    trainable, total = _count_trainable(model)
    print(f"Params: trainable={trainable:,} total={total:,}")

    # Confirm new modules exist.
    for name in cfg.encoder_warmup_modules:
        module = getattr(model, name, None)
        flag = "OK" if module is not None else "MISSING"
        print(f"  module {name}: {flag}")

    observation = _build_observation(args.batch_size, args.action_dim, args.max_token_len, args.force_window)
    actions = torch.randn(args.batch_size, args.action_horizon, args.action_dim) * 0.1

    print("\n--- forward(observation, actions) ---")
    outputs = model(observation, actions)
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} norm={v.float().norm().item():.4g}")
        else:
            print(f"  {k}: {v!r}")

    print("\n--- compute_total_loss across phases ---")
    for step in (
        0,
        cfg.encoder_warmup_steps - 1 if cfg.encoder_warmup_steps > 0 else 0,
        cfg.encoder_warmup_steps,
        cfg.contrast_warmup_start - 1,
        cfg.contrast_warmup_start,
        (cfg.contrast_warmup_start + cfg.contrast_warmup_end) // 2,
        cfg.contrast_warmup_end + 1,
    ):
        loss_dict = model.compute_total_loss(outputs, step)
        print(f"step={step:6d}  {_format_loss(loss_dict)}")

    print("\n--- freeze_backbone_for_warmup + unfreeze_all ---")
    model.freeze_backbone_for_warmup()
    frozen_trainable, _ = _count_trainable(model)
    print(f"After freeze: trainable={frozen_trainable:,}/{total:,}")
    model.unfreeze_all()
    full_trainable, _ = _count_trainable(model)
    print(f"After unfreeze: trainable={full_trainable:,}/{total:,}")
    assert frozen_trainable < total, "freeze did not reduce trainable param count"
    assert full_trainable == total, "unfreeze did not restore all params"

    if not args.no_backward:
        print("\n--- backward(joint-phase total) ---")
        # Rebuild outputs after toggling requires_grad so leaves are picked up.
        outputs = model(observation, actions)
        loss_dict = model.compute_total_loss(outputs, cfg.contrast_warmup_end)
        loss_dict["total"].backward()
        any_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        print(f"  joint-phase backward: any non-zero grad? {any_grad}")


if __name__ == "__main__":
    main()
