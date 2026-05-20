"""Compare one fixed pi0.5 forward pass between PyTorch and converted JAX weights."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import gc
import os
import pathlib
import sys
from typing import Literal

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models import model as _model
from openpi.models import pi0 as _pi0_jax
from openpi.models_pytorch import pi0_pytorch as _pi0_torch
import openpi.training.config as _config


DEFAULT_PYTORCH_CKPT = pathlib.Path("checkpoints/pi05_umi_vision_multitask/pi05_vision_9task_200k/200000")
DEFAULT_JAX_CKPT = pathlib.Path("checkpoints/pi05_umi_vision_multitask_jax/pi05_vision_9task_200k/200000_float32")


def _resolve_pytorch_weight_file(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    weight_file = path / "model.safetensors"
    if not weight_file.exists():
        raise FileNotFoundError(f"Missing PyTorch model.safetensors under: {path}")
    return weight_file


def _resolve_jax_params_path(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if (path / "params").exists():
        return path / "params"
    if (path / "_METADATA").exists() or (path / "manifest.ocdbt").exists():
        return path
    raise FileNotFoundError(f"Missing JAX params checkpoint at or under: {path}")


def _normalize_precision(value: str) -> Literal["float32", "bfloat16"]:
    if value == "bfloat32":
        print("precision=bfloat32 is not a real dtype; treating it as float32.")
        return "float32"
    if value not in {"float32", "bfloat16"}:
        raise ValueError(f"Unsupported precision: {value}")
    return value  # type: ignore[return-value]


def _jax_dtype(precision: str):
    return jnp.float32 if precision == "float32" else jnp.bfloat16


def _torch_dtype(precision: str):
    return torch.float32 if precision == "float32" else torch.bfloat16


def _fixed_numpy_batch(model_config, *, seed: int):
    rng = np.random.default_rng(seed)
    batch_size = 1
    height, width = _model.IMAGE_RESOLUTION

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    left = np.stack(
        [
            xx,
            yy,
            np.sin(np.pi * xx) * np.cos(np.pi * yy),
        ],
        axis=-1,
    ).astype(np.float32)
    base = np.zeros_like(left)
    right = np.zeros_like(left)

    images = {
        "base_0_rgb": base[None],
        "left_wrist_0_rgb": left[None],
        "right_wrist_0_rgb": right[None],
    }
    image_masks = {
        "base_0_rgb": np.array([False]),
        "left_wrist_0_rgb": np.array([True]),
        "right_wrist_0_rgb": np.array([False]),
    }
    tokenized_prompt = np.zeros((batch_size, model_config.max_token_len), dtype=np.int32)
    tokenized_prompt[:, :16] = np.arange(16, dtype=np.int32) + 32
    tokenized_prompt_mask = np.zeros((batch_size, model_config.max_token_len), dtype=bool)
    tokenized_prompt_mask[:, :16] = True

    state = np.linspace(-0.5, 0.5, model_config.action_dim, dtype=np.float32)[None]
    actions = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(batch_size, model_config.action_horizon, model_config.action_dim),
    ).astype(np.float32)
    noise = rng.normal(size=actions.shape).astype(np.float32)
    time = np.array([0.37], dtype=np.float32)

    return {
        "images": images,
        "image_masks": image_masks,
        "state": state,
        "tokenized_prompt": tokenized_prompt,
        "tokenized_prompt_mask": tokenized_prompt_mask,
        "actions": actions,
        "noise": noise,
        "time": time,
    }


def _torch_observation(batch: dict, *, device: torch.device, dtype: torch.dtype) -> _model.Observation:
    images = {
        key: torch.from_numpy(np.transpose(value, (0, 3, 1, 2))).to(device=device, dtype=dtype)
        for key, value in batch["images"].items()
    }
    image_masks = {key: torch.from_numpy(value).to(device=device) for key, value in batch["image_masks"].items()}
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=torch.from_numpy(batch["state"]).to(device=device, dtype=dtype),
        tokenized_prompt=torch.from_numpy(batch["tokenized_prompt"]).to(device=device),
        tokenized_prompt_mask=torch.from_numpy(batch["tokenized_prompt_mask"]).to(device=device),
    )


def _jax_observation(batch: dict, *, dtype) -> _model.Observation:
    images = {key: jnp.asarray(value, dtype=dtype) for key, value in batch["images"].items()}
    image_masks = {key: jnp.asarray(value, dtype=jnp.bool_) for key, value in batch["image_masks"].items()}
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.asarray(batch["state"], dtype=dtype),
        tokenized_prompt=jnp.asarray(batch["tokenized_prompt"], dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray(batch["tokenized_prompt_mask"], dtype=jnp.bool_),
    )


@torch.no_grad()
def _torch_flow_prediction(model, observation, actions, noise, time) -> tuple[np.ndarray, np.ndarray]:
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)

    if model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = _pi0_torch.make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

    (_, suffix_out), _ = model.paligemma_with_expert.forward(
        attention_mask=att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = suffix_out[:, -model.config.action_horizon :].to(dtype=torch.float32)
    v_t = model.action_out_proj(suffix_out)
    loss = F.mse_loss(u_t.to(dtype=torch.float32), v_t.to(dtype=torch.float32), reduction="none")
    return v_t.detach().float().cpu().numpy(), loss.detach().float().cpu().numpy()


def _jax_flow_prediction(model, observation, actions, noise, time) -> tuple[np.ndarray, np.ndarray]:
    observation = _model.preprocess_observation(None, observation, train=False)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    with jax.default_matmul_precision("highest"):
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0_jax.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        loss = jnp.square(v_t.astype(jnp.float32) - u_t.astype(jnp.float32))

    return np.asarray(jax.device_get(v_t), dtype=np.float32), np.asarray(jax.device_get(loss), dtype=np.float32)


def _load_and_run_pytorch(config, batch, args, precision: str) -> tuple[np.ndarray, np.ndarray]:
    weight_file = _resolve_pytorch_weight_file(args.pytorch_checkpoint_dir)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")

    print(f"Loading PyTorch weights: {weight_file}")
    model = config.model.load_pytorch(config, str(weight_file))
    model = model.to(device=device)
    model.eval()
    torch.set_float32_matmul_precision("highest")

    dtype = _torch_dtype(precision)
    if precision == "float32":
        model = model.to(dtype=torch.float32)

    obs = _torch_observation(batch, device=device, dtype=dtype)
    actions = torch.from_numpy(batch["actions"]).to(device=device, dtype=dtype)
    noise = torch.from_numpy(batch["noise"]).to(device=device, dtype=dtype)
    time = torch.from_numpy(batch["time"]).to(device=device, dtype=torch.float32)
    pred, loss = _torch_flow_prediction(model, obs, actions, noise, time)

    del model, obs, actions, noise, time
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, loss


def _load_and_run_jax(config, batch, args, precision: str) -> tuple[np.ndarray, np.ndarray]:
    params_path = _resolve_jax_params_path(args.jax_checkpoint_dir)
    dtype = _jax_dtype(precision)

    print(f"Loading JAX params: {params_path}")
    params = _model.restore_params(params_path, restore_type=jax.Array, dtype=dtype)
    model = config.model.load(params)

    obs = _jax_observation(batch, dtype=dtype)
    actions = jnp.asarray(batch["actions"], dtype=dtype)
    noise = jnp.asarray(batch["noise"], dtype=dtype)
    time = jnp.asarray(batch["time"], dtype=jnp.float32)
    return _jax_flow_prediction(model, obs, actions, noise, time)


def _summary(name: str, value: np.ndarray) -> str:
    return (
        f"{name}: shape={value.shape} mean={value.mean():.6g} std={value.std():.6g} "
        f"min={value.min():.6g} max={value.max():.6g} l2={np.linalg.norm(value):.6g}"
    )


def _compare(name: str, torch_value: np.ndarray, jax_value: np.ndarray) -> dict[str, float]:
    diff = torch_value - jax_value
    denom = max(float(np.sqrt(np.mean(torch_value**2))), float(np.sqrt(np.mean(jax_value**2))), 1e-8)
    flat_t = torch_value.reshape(-1).astype(np.float64)
    flat_j = jax_value.reshape(-1).astype(np.float64)
    cosine = float(np.dot(flat_t, flat_j) / max(np.linalg.norm(flat_t) * np.linalg.norm(flat_j), 1e-12))
    metrics = {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs": float(np.max(np.abs(diff))),
        "relative_rmse": float(np.sqrt(np.mean(diff**2)) / denom),
        "cosine": cosine,
    }
    print(f"\n{name}")
    print(_summary("  torch", torch_value))
    print(_summary("  jax  ", jax_value))
    print(
        "  diff : "
        f"mae={metrics['mae']:.6g} rmse={metrics['rmse']:.6g} "
        f"relative_rmse={metrics['relative_rmse']:.6g} max_abs={metrics['max_abs']:.6g} "
        f"cosine={metrics['cosine']:.8f}"
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", type=str, default="pi05_umi_vision_multitask")
    parser.add_argument("--pytorch-checkpoint-dir", type=pathlib.Path, default=DEFAULT_PYTORCH_CKPT)
    parser.add_argument("--jax-checkpoint-dir", type=pathlib.Path, default=DEFAULT_JAX_CKPT)
    parser.add_argument("--precision", type=str, default="float32", choices=("float32", "bfloat16", "bfloat32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--relative-rmse-threshold", type=float, default=0.10)
    parser.add_argument("--max-abs-threshold", type=float, default=0.50)
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    args = parser.parse_args(argv)

    precision = _normalize_precision(args.precision)
    base_config = _config.get_config(args.config_name)
    model_config = dataclasses.replace(base_config.model, dtype=precision, pytorch_compile_mode=None)
    config = dataclasses.replace(base_config, model=model_config, batch_size=1, num_workers=0, wandb_enabled=False)
    batch = _fixed_numpy_batch(config.model, seed=args.seed)

    torch_pred, torch_loss = _load_and_run_pytorch(config, batch, args, precision)
    jax_pred, jax_loss = _load_and_run_jax(config, batch, args, precision)

    if torch_pred.shape != jax_pred.shape:
        raise ValueError(f"Prediction shape mismatch: torch={torch_pred.shape}, jax={jax_pred.shape}")
    if torch_loss.shape != jax_loss.shape:
        raise ValueError(f"Loss shape mismatch: torch={torch_loss.shape}, jax={jax_loss.shape}")

    pred_metrics = _compare("flow_prediction_v_t", torch_pred, jax_pred)
    _compare("elementwise_mse_loss", torch_loss, jax_loss)

    finite = all(np.isfinite(arr).all() for arr in (torch_pred, torch_loss, jax_pred, jax_loss))
    passed = (
        finite
        and pred_metrics["relative_rmse"] <= args.relative_rmse_threshold
        and pred_metrics["max_abs"] <= args.max_abs_threshold
        and pred_metrics["cosine"] >= args.cosine_threshold
    )
    print(
        "\nverdict: "
        f"{'PASS' if passed else 'FAIL'} "
        f"(finite={finite}, relative_rmse<={args.relative_rmse_threshold}, "
        f"max_abs<={args.max_abs_threshold}, cosine>={args.cosine_threshold})"
    )
    print(
        "loss diff is reported as a diagnostic only; the pass/fail decision uses "
        "the direct flow prediction comparison."
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
