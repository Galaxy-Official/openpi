"""Smoke test for ForceEncoder.

Run:
    uv run scripts/debug/dbg_force_encoder.py
"""

from __future__ import annotations

import argparse

import torch

from openpi.models_pytorch.encoders import ForceEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force-window", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=4)
    args = parser.parse_args()

    encoder = ForceEncoder(embed_dim=args.embed_dim, input_dim=2, num_tokens=args.num_tokens)
    force = torch.randn(args.batch_size, args.force_window, 2)
    tokens, mask = encoder(force)
    loss = tokens.square().mean()
    loss.backward()

    grad_norm = encoder.conv[0].weight.grad.norm().item()
    print(f"tokens={tuple(tokens.shape)} mask={tuple(mask.shape)} loss={loss.item():.6f} grad_norm={grad_norm:.6f}")


if __name__ == "__main__":
    main()
