"""Smoke test for ContrastiveHead.

Run:
    uv run scripts/debug/dbg_contrastive.py
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.heads import ContrastiveHead


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--proj-dim", type=int, default=64)
    args = parser.parse_args()

    head = ContrastiveHead(temperature=0.07, neg_window=8)
    z_vision = F.normalize(torch.randn(args.batch_size, args.proj_dim), dim=-1)
    z_touchforce = F.normalize(z_vision + 0.2 * torch.randn_like(z_vision), dim=-1)
    out = head(
        z_vision,
        z_touchforce,
        task_index=torch.zeros(args.batch_size, dtype=torch.long),
        frame_index=torch.arange(args.batch_size, dtype=torch.long) * 16,
    )
    out["loss"].backward()

    print(
        f"loss={out['loss'].item():.6f} acc@1={out['acc@1'].item():.4f} "
        f"pos_sim={out['pos_sim'].item():.4f} neg_sim={out['neg_sim'].item():.4f} "
        f"temperature={head.temperature.item():.4f}"
    )


if __name__ == "__main__":
    main()
