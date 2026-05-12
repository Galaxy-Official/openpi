"""Smoke test for TactileEncoder.

Run:
    uv run scripts/debug/dbg_tactile_encoder.py
"""

from __future__ import annotations

import argparse

import torch

from openpi.models_pytorch.encoders import TactileEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--tokens-per-side", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    encoder = TactileEncoder(embed_dim=args.embed_dim, num_tokens_per_side=args.tokens_per_side)
    left = torch.rand(args.batch_size, 3, args.height, args.width)
    right = torch.rand(args.batch_size, args.height, args.width, 3)
    tokens, mask = encoder(left, right)
    loss = tokens.square().mean()
    loss.backward()

    grad_norm = encoder.stem.patch.weight.grad.norm().item() if encoder.stem is not None else 0.0
    print(f"tokens={tuple(tokens.shape)} mask={tuple(mask.shape)} loss={loss.item():.6f} grad_norm={grad_norm:.6f}")


if __name__ == "__main__":
    main()
