"""Smoke test for TouchForceFusion.

Run:
    uv run scripts/debug/dbg_fusion.py
"""

from __future__ import annotations

import argparse

import torch

from openpi.models_pytorch.encoders import TouchForceFusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--tactile-tokens", type=int, default=32)
    parser.add_argument("--force-tokens", type=int, default=4)
    parser.add_argument("--fused-tokens", type=int, default=32)
    args = parser.parse_args()

    fusion = TouchForceFusion(embed_dim=args.embed_dim, num_fused_tokens=args.fused_tokens, proj_dim=32)
    tactile = torch.randn(args.batch_size, args.tactile_tokens, args.embed_dim, requires_grad=True)
    force = torch.randn(args.batch_size, args.force_tokens, args.embed_dim, requires_grad=True)
    fused, mask, z = fusion(tactile, force)
    loss = fused.square().mean() + z.square().mean()
    loss.backward()

    print(
        f"fused={tuple(fused.shape)} mask={tuple(mask.shape)} z={tuple(z.shape)} "
        f"loss={loss.item():.6f} z_norm={z.norm(dim=-1).mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
