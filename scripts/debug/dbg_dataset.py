"""Standalone smoke test for the multi-modal data pipeline.

Run on the cloud server where the LeRobot tasks live.

Examples:
    # Single task, 2 batches, print shapes and dump sample artifacts:
    uv run scripts/debug/dbg_dataset.py \
        --repo-id 430_clamp_seal_lerobot \
        --batch-size 4 --num-batches 2 \
        --dump-dir /tmp/dbg_mm

    # All 8 tasks with weighted sampler:
    uv run scripts/debug/dbg_dataset.py \
        --repo-ids 429_erase_board_lerobot 430_clamp_seal_lerobot 430_towel_hanging_lerobot \
                   501_bread_moving_lerobot 505_screw_lerobot 505_stiring_lerobot \
                   506_open_bottle_lerobot 506_peg_flowers_lerobot \
        --weights-json src/openpi/training/multimodal/task_weights.json \
        --batch-size 8 --num-batches 4

This script intentionally avoids importing the full openpi training stack so
that it can be used to triage data issues before the model is wired up.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import logging
import pathlib

import numpy as np

from openpi.training.multimodal import LeRobotMultiModalDataset
from openpi.training.multimodal import MultiModalDatasetSpec
from openpi.training.multimodal import MultiModalRepack
from openpi.training.multimodal import MultiTaskConcatDataset
from openpi.training.multimodal import load_task_weights
from openpi.training.multimodal import make_weighted_sampler
from openpi.training.multimodal import multimodal_collate_fn


def _summary(value, name: str) -> str:
    arr = np.asarray(value)
    if arr.dtype == object or arr.size == 0:
        return f"  {name}: dtype={arr.dtype} size={arr.size}"
    return f"  {name}: shape={tuple(arr.shape)} dtype={arr.dtype} min={float(arr.min()):.4g} max={float(arr.max()):.4g}"


def _print_batch_summary(batch: dict, prefix: str = "") -> None:
    for k, v in batch.items():
        full = f"{prefix}{k}"
        if isinstance(v, dict):
            print(f"{full}:")
            _print_batch_summary(v, prefix=full + ".")
        elif isinstance(v, list):
            sample = v[0] if v else None
            print(f"  {full}: list[{len(v)}] sample={sample!r}")
        else:
            print(_summary(v, full))


def _save_artifacts(batch: dict, dump_dir: pathlib.Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save first sample of each image-like leaf as png if PIL is available.
    try:
        from PIL import Image  # noqa: PLC0415

        for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            arr = batch["image"][cam][0]
            arr = np.asarray(arr)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(dump_dir / f"image_{cam}.png")
        for side in ("left", "right"):
            arr = np.asarray(batch["tactile"][side][0])
            # Squeeze frame-stack dim if present.
            if arr.ndim == 4:
                arr = arr[-1]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(dump_dir / f"tactile_{side}.png")
    except ImportError:
        logging.warning("PIL not available; skipping image dumps.")

    # Save force windows as csv for quick inspection.
    for side in ("left", "right"):
        arr = np.asarray(batch["force"][side][0]).reshape(-1)
        np.savetxt(dump_dir / f"force_{side}.csv", arr, fmt="%.6f")

    # Save first action.
    np.savetxt(dump_dir / "action.csv", np.asarray(batch["actions"][0]), fmt="%.6f")


def build_dataset(
    repo_id: str,
    action_horizon: int,
    force_window: int,
    tactile_stack: int,
    drop_seed: int,
    root: str | None,
):
    spec = MultiModalDatasetSpec(
        repo_id=repo_id,
        action_horizon=action_horizon,
        force_window=force_window,
        tactile_frame_stack=tactile_stack,
        prompt_from_task=True,
        root=root,
    )
    raw = LeRobotMultiModalDataset(spec)
    return _TransformedView(raw, MultiModalRepack(drop_seed=drop_seed))


class _TransformedView:
    """Tiny adapter applying a per-sample transform; no dependency on JAX path."""

    def __init__(self, dataset, transform):
        self.raw_dataset = dataset
        self.transform = transform
        self.task_name = getattr(dataset, "task_name", str(dataset))

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        return self.transform(self.raw_dataset[int(idx)])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default=None, help="Single task to load.")
    parser.add_argument("--repo-ids", type=str, nargs="*", default=None, help="Multiple tasks for ConcatDataset.")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--force-window", type=int, default=16)
    parser.add_argument("--tactile-stack", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Task sampling exponent.")
    parser.add_argument("--weights-json", type=str, default=None, help="Optional weights JSON to override alpha.")
    parser.add_argument("--dump-dir", type=str, default=None, help="If set, write batch artifacts here.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Parent directory containing {repo_id}/. Defaults to $OPENPI_DATA_ROOT or ../openpi/Data.",
    )
    args = parser.parse_args(argv)

    if args.repo_id is None and not args.repo_ids:
        parser.error("Provide --repo-id or --repo-ids.")

    import torch  # noqa: PLC0415  (local so the script imports without torch installed)

    from openpi.training.multimodal.paths import resolve_data_root  # noqa: PLC0415

    repo_ids = [args.repo_id] if args.repo_id else list(args.repo_ids)
    print(f"Loading {len(repo_ids)} task(s): {repo_ids}")

    resolved_root = resolve_data_root(args.data_root)
    print(f"Resolved data root: {resolved_root}  (exists={resolved_root.exists()})")
    for rid in repo_ids:
        task_dir = resolved_root / rid
        meta = task_dir / "meta"
        if not meta.exists():
            print(f"  [WARN] {rid}: meta/ missing under {task_dir}")
            continue
        fmt_files = {
            "tasks.jsonl (v2.x)": (meta / "tasks.jsonl").exists(),
            "tasks.parquet (v3.0)": (meta / "tasks.parquet").exists(),
            "episodes.jsonl (v2.x)": (meta / "episodes.jsonl").exists(),
            "episodes/ (v3.0)": (meta / "episodes").is_dir(),
            "info.json": (meta / "info.json").exists(),
        }
        flags = ", ".join(f"{k}={v}" for k, v in fmt_files.items())
        print(f"  {rid}: {flags}")

    views = [
        build_dataset(
            rid,
            args.action_horizon,
            args.force_window,
            args.tactile_stack,
            drop_seed=args.seed,
            root=args.data_root,
        )
        for rid in repo_ids
    ]
    underlying = [v.raw_dataset for v in views]
    concat = MultiTaskConcatDataset(underlying)
    per_task = dict(zip(concat.task_names, concat.task_frame_counts, strict=True))
    print(f"Total frames: {len(concat)}  per-task: {per_task}")

    task_weights = None
    if args.weights_json:
        meta = load_task_weights(args.weights_json)
        task_weights = {k: v["weight"] for k, v in meta["tasks"].items()}
        print(f"Loaded {len(task_weights)} task weights from {args.weights_json}")

    # The concat dataset returns raw samples; we wrap it to apply the transform.
    class _ConcatTransformed:
        def __init__(self, concat, transforms_by_task):
            self._concat = concat
            self._transforms_by_task = transforms_by_task

        def __len__(self):
            return len(self._concat)

        def __getitem__(self, flat_index):
            ti = self._concat.task_index_for(int(flat_index))
            raw = self._concat[int(flat_index)]
            return self._transforms_by_task[ti](raw)

    transforms_by_task = [v.transform for v in views]
    dataset = _ConcatTransformed(concat, transforms_by_task)

    gen = torch.Generator().manual_seed(args.seed)
    sampler = make_weighted_sampler(
        concat,
        alpha=args.alpha,
        task_weights=task_weights,
        generator=gen,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )

    dump_dir = pathlib.Path(args.dump_dir) if args.dump_dir else None
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        print(f"\n=== batch {i} ===")
        _print_batch_summary(batch)
        if dump_dir is not None and i == 0:
            _save_artifacts(batch, dump_dir / "batch_0")
            print(f"Saved batch_0 artifacts to {dump_dir / 'batch_0'}")
            with open(dump_dir / "batch_0" / "summary.json", "w") as f:
                summary = {
                    "task_names_in_batch": batch.get("task_name", []),
                    "task_indices_in_batch": np.asarray(batch.get("task_index", [])).tolist(),
                }
                json.dump(summary, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
