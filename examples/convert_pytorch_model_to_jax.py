#!/usr/bin/env python3
# ruff: noqa: E402
"""Convert a PyTorch pi0/pi0.5 checkpoint to a JAX/Orbax params checkpoint.

This script is intended for inference deployment compatibility. It writes the
`params` item expected by `openpi.policies.policy_config.create_trained_policy`
and copies checkpoint assets such as normalization stats. It does not attempt to
reconstruct a JAX `train_state` or Optax optimizer state from PyTorch training.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
from typing import Literal

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import safetensors.torch
import torch
import tyro

import openpi.models.gemma as _gemma
import openpi.models.pi0_config as _pi0_config
import openpi.training.config as _config


_PALIGEMMA_PREFIX = "paligemma_with_expert.paligemma.model"
_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model"


class _PaliGemmaShapeConfig:
    def __init__(self, model_config: _pi0_config.Pi0Config):
        paligemma_config = _gemma.get_config(model_config.paligemma_variant)
        self.vision_hidden_size = 1152
        self.vision_layers = 27
        self.vision_heads = 16
        self.vision_head_dim = self.vision_hidden_size // self.vision_heads
        self.text_hidden_size = paligemma_config.width
        self.text_layers = paligemma_config.depth
        self.text_heads = paligemma_config.num_heads
        self.text_kv_heads = paligemma_config.num_kv_heads
        self.text_head_dim = paligemma_config.head_dim


def _as_jax_array(tensor: torch.Tensor, dtype: jnp.dtype) -> jax.Array:
    """Convert a CPU/GPU torch tensor to a JAX array with a stable dtype."""
    array = tensor.detach().cpu()
    if array.dtype == torch.bfloat16:
        array = array.float()
    return jnp.asarray(array.numpy(), dtype=dtype)


def _dtype_from_precision(precision: Literal["float32", "bfloat16", "float16"]) -> jnp.dtype:
    if precision == "float32":
        return jnp.float32
    if precision == "bfloat16":
        return jnp.bfloat16
    if precision == "float16":
        return jnp.float16
    raise ValueError(f"Unsupported precision: {precision}")


class _StateReader:
    def __init__(self, state_dict: dict[str, torch.Tensor]):
        self._state_dict = state_dict
        self.consumed: set[str] = set()

    def take(self, key: str) -> torch.Tensor:
        if key not in self._state_dict:
            candidates = [k for k in self._state_dict if key.split(".")[-2] in k or key.split(".")[-1] in k]
            preview = "\n".join(candidates[:20])
            raise KeyError(f"Missing PyTorch key: {key}\nSimilar available keys:\n{preview}")
        self.consumed.add(key)
        return self._state_dict[key]


def _put(flat: dict[str, jax.Array], key: str, value: torch.Tensor, dtype: jnp.dtype):
    flat[key] = _as_jax_array(value.contiguous(), dtype)


def _reverse_q_kernel(weight: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    # JAX: [num_heads, hidden, head_dim] -> PyTorch: [num_heads * head_dim, hidden]
    hidden = weight.shape[1]
    return weight.reshape(num_heads, head_dim, hidden).permute(0, 2, 1)


def _reverse_kv_kernel(weight: torch.Tensor, *, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    # JAX: [num_kv_heads, hidden, head_dim] -> PyTorch: [num_kv_heads * head_dim, hidden]
    hidden = weight.shape[1]
    return weight.reshape(num_kv_heads, head_dim, hidden).permute(0, 2, 1)


def _reverse_paligemma_o_kernel(weight: torch.Tensor, *, hidden: int, num_heads: int, head_dim: int) -> torch.Tensor:
    # Inverse of: jax_kernel.transpose(2, 0, 1).reshape(num_heads * head_dim, hidden)
    return weight.reshape(hidden, num_heads, head_dim).permute(1, 2, 0)


def _reverse_expert_o_kernel(weight: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    # Inverse of: jax_kernel.reshape(num_heads * head_dim, hidden).transpose(1, 0)
    hidden = weight.shape[0]
    return weight.transpose(0, 1).reshape(num_heads, head_dim, hidden)


def _convert_pi0_projection_params(
    reader: _StateReader,
    flat: dict[str, jax.Array],
    *,
    model_config: _pi0_config.Pi0Config,
    dtype: jnp.dtype,
):
    names = ["action_in_proj", "action_out_proj"]
    if model_config.pi05:
        names.extend(["time_mlp_in", "time_mlp_out"])
    else:
        names.extend(["state_proj", "action_time_mlp_in", "action_time_mlp_out"])

    for name in names:
        _put(flat, f"{name}/kernel", reader.take(f"{name}.weight").transpose(0, 1), dtype)
        _put(flat, f"{name}/bias", reader.take(f"{name}.bias"), dtype)


def _convert_vision_params(
    reader: _StateReader,
    flat: dict[str, jax.Array],
    *,
    model_config: _pi0_config.Pi0Config,
    dtype: jnp.dtype,
):
    config = _PaliGemmaShapeConfig(model_config)
    vision_prefix = f"{_PALIGEMMA_PREFIX}.vision_tower.vision_model"

    _put(
        flat,
        "PaliGemma/img/embedding/kernel",
        reader.take(f"{vision_prefix}.embeddings.patch_embedding.weight").permute(2, 3, 1, 0),
        dtype,
    )
    _put(flat, "PaliGemma/img/embedding/bias", reader.take(f"{vision_prefix}.embeddings.patch_embedding.bias"), dtype)
    _put(
        flat,
        "PaliGemma/img/pos_embedding",
        reader.take(f"{vision_prefix}.embeddings.position_embedding.weight").reshape(
            1, -1, config.vision_hidden_size
        ),
        dtype,
    )

    layernorm0_scale = []
    layernorm0_bias = []
    layernorm1_scale = []
    layernorm1_bias = []
    dense0_kernel = []
    dense0_bias = []
    dense1_kernel = []
    dense1_bias = []
    key_kernel = []
    key_bias = []
    value_kernel = []
    value_bias = []
    query_kernel = []
    query_bias = []
    out_kernel = []
    out_bias = []

    for i in range(config.vision_layers):
        layer = f"{vision_prefix}.encoder.layers.{i}"
        layernorm0_scale.append(reader.take(f"{layer}.layer_norm1.weight"))
        layernorm0_bias.append(reader.take(f"{layer}.layer_norm1.bias"))
        layernorm1_scale.append(reader.take(f"{layer}.layer_norm2.weight"))
        layernorm1_bias.append(reader.take(f"{layer}.layer_norm2.bias"))
        dense0_kernel.append(reader.take(f"{layer}.mlp.fc1.weight").transpose(0, 1))
        dense0_bias.append(reader.take(f"{layer}.mlp.fc1.bias"))
        dense1_kernel.append(reader.take(f"{layer}.mlp.fc2.weight").transpose(0, 1))
        dense1_bias.append(reader.take(f"{layer}.mlp.fc2.bias"))

        key_kernel.append(
            reader.take(f"{layer}.self_attn.k_proj.weight")
            .transpose(0, 1)
            .reshape(config.vision_hidden_size, config.vision_heads, config.vision_head_dim)
        )
        key_bias.append(
            reader.take(f"{layer}.self_attn.k_proj.bias").reshape(config.vision_heads, config.vision_head_dim)
        )
        value_kernel.append(
            reader.take(f"{layer}.self_attn.v_proj.weight")
            .transpose(0, 1)
            .reshape(config.vision_hidden_size, config.vision_heads, config.vision_head_dim)
        )
        value_bias.append(
            reader.take(f"{layer}.self_attn.v_proj.bias").reshape(config.vision_heads, config.vision_head_dim)
        )
        query_kernel.append(
            reader.take(f"{layer}.self_attn.q_proj.weight")
            .transpose(0, 1)
            .reshape(config.vision_hidden_size, config.vision_heads, config.vision_head_dim)
        )
        query_bias.append(
            reader.take(f"{layer}.self_attn.q_proj.bias").reshape(config.vision_heads, config.vision_head_dim)
        )
        out_kernel.append(
            reader.take(f"{layer}.self_attn.out_proj.weight")
            .transpose(0, 1)
            .reshape(config.vision_heads, config.vision_head_dim, config.vision_hidden_size)
        )
        out_bias.append(
            reader.take(f"{layer}.self_attn.out_proj.bias").reshape(config.vision_heads, config.vision_head_dim)
        )

    _put(flat, "PaliGemma/img/Transformer/encoderblock/LayerNorm_0/scale", torch.stack(layernorm0_scale), dtype)
    _put(flat, "PaliGemma/img/Transformer/encoderblock/LayerNorm_0/bias", torch.stack(layernorm0_bias), dtype)
    _put(flat, "PaliGemma/img/Transformer/encoderblock/LayerNorm_1/scale", torch.stack(layernorm1_scale), dtype)
    _put(flat, "PaliGemma/img/Transformer/encoderblock/LayerNorm_1/bias", torch.stack(layernorm1_bias), dtype)
    _put(flat, "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel", torch.stack(dense0_kernel), dtype)
    _put(flat, "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias", torch.stack(dense0_bias), dtype)
    _put(flat, "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel", torch.stack(dense1_kernel), dtype)
    _put(flat, "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias", torch.stack(dense1_bias), dtype)
    _put(
        flat,
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel",
        torch.stack(key_kernel),
        dtype,
    )
    _put(
        flat,
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias",
        torch.stack(key_bias),
        dtype,
    )
    _put(
        flat,
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel",
        torch.stack(value_kernel),
        dtype,
    )
    _put(
        flat,
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias",
        torch.stack(value_bias),
        dtype,
    )
    _put(
        flat,
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel",
        torch.stack(query_kernel),
        dtype,
    )
    _put(
        flat,
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias",
        torch.stack(query_bias),
        dtype,
    )
    _put(
        flat,
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel",
        torch.stack(out_kernel),
        dtype,
    )
    _put(
        flat,
        "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias",
        torch.stack(out_bias),
        dtype,
    )

    _put(
        flat,
        "PaliGemma/img/Transformer/encoder_norm/scale",
        reader.take(f"{vision_prefix}.post_layernorm.weight"),
        dtype,
    )
    _put(
        flat,
        "PaliGemma/img/Transformer/encoder_norm/bias",
        reader.take(f"{vision_prefix}.post_layernorm.bias"),
        dtype,
    )
    _put(
        flat,
        "PaliGemma/img/head/kernel",
        reader.take(f"{_PALIGEMMA_PREFIX}.multi_modal_projector.linear.weight").transpose(0, 1),
        dtype,
    )
    _put(flat, "PaliGemma/img/head/bias", reader.take(f"{_PALIGEMMA_PREFIX}.multi_modal_projector.linear.bias"), dtype)


def _convert_paligemma_text_params(
    reader: _StateReader,
    flat: dict[str, jax.Array],
    *,
    model_config: _pi0_config.Pi0Config,
    dtype: jnp.dtype,
):
    config = _PaliGemmaShapeConfig(model_config)
    text_prefix = f"{_PALIGEMMA_PREFIX}.language_model"

    _put(flat, "PaliGemma/llm/embedder/input_embedding", reader.take(f"{text_prefix}.embed_tokens.weight"), dtype)

    attn_vec = []
    kv = []
    q = []
    gating = []
    linear = []
    pre_attention_norm = []
    pre_ffw_norm = []

    for i in range(config.text_layers):
        layer = f"{text_prefix}.layers.{i}"
        q.append(
            _reverse_q_kernel(
                reader.take(f"{layer}.self_attn.q_proj.weight"),
                num_heads=config.text_heads,
                head_dim=config.text_head_dim,
            )
        )
        k = _reverse_kv_kernel(
            reader.take(f"{layer}.self_attn.k_proj.weight"),
            num_kv_heads=config.text_kv_heads,
            head_dim=config.text_head_dim,
        )
        v = _reverse_kv_kernel(
            reader.take(f"{layer}.self_attn.v_proj.weight"),
            num_kv_heads=config.text_kv_heads,
            head_dim=config.text_head_dim,
        )
        kv.append(torch.stack([k, v], dim=0))
        attn_vec.append(
            _reverse_paligemma_o_kernel(
                reader.take(f"{layer}.self_attn.o_proj.weight"),
                hidden=config.text_hidden_size,
                num_heads=config.text_heads,
                head_dim=config.text_head_dim,
            )
        )
        gating.append(
            torch.stack(
                [
                    reader.take(f"{layer}.mlp.gate_proj.weight").transpose(0, 1),
                    reader.take(f"{layer}.mlp.up_proj.weight").transpose(0, 1),
                ],
                dim=0,
            )
        )
        linear.append(reader.take(f"{layer}.mlp.down_proj.weight").transpose(0, 1))
        pre_attention_norm.append(reader.take(f"{layer}.input_layernorm.weight"))
        pre_ffw_norm.append(reader.take(f"{layer}.post_attention_layernorm.weight"))

    _put(flat, "PaliGemma/llm/layers/attn/attn_vec_einsum/w", torch.stack(attn_vec), dtype)
    _put(flat, "PaliGemma/llm/layers/attn/kv_einsum/w", torch.stack(kv), dtype)
    _put(flat, "PaliGemma/llm/layers/attn/q_einsum/w", torch.stack(q), dtype)
    _put(flat, "PaliGemma/llm/layers/mlp/gating_einsum", torch.stack(gating), dtype)
    _put(flat, "PaliGemma/llm/layers/mlp/linear", torch.stack(linear), dtype)
    _put(flat, "PaliGemma/llm/layers/pre_attention_norm/scale", torch.stack(pre_attention_norm), dtype)
    _put(flat, "PaliGemma/llm/layers/pre_ffw_norm/scale", torch.stack(pre_ffw_norm), dtype)
    _put(flat, "PaliGemma/llm/final_norm/scale", reader.take(f"{text_prefix}.norm.weight"), dtype)


def _convert_expert_params(
    reader: _StateReader,
    flat: dict[str, jax.Array],
    *,
    expert_config: _gemma.Config,
    pi05: bool,
    dtype: jnp.dtype,
):
    attn_vec = []
    kv = []
    q = []
    gating = []
    linear = []
    pre_attention_norm_scale = []
    pre_ffw_norm_scale = []
    pre_attention_norm_dense_bias = []
    pre_attention_norm_dense_kernel = []
    pre_ffw_norm_dense_bias = []
    pre_ffw_norm_dense_kernel = []

    for i in range(expert_config.depth):
        layer = f"{_EXPERT_PREFIX}.layers.{i}"
        q.append(
            _reverse_q_kernel(
                reader.take(f"{layer}.self_attn.q_proj.weight"),
                num_heads=expert_config.num_heads,
                head_dim=expert_config.head_dim,
            )
        )
        k = _reverse_kv_kernel(
            reader.take(f"{layer}.self_attn.k_proj.weight"),
            num_kv_heads=expert_config.num_kv_heads,
            head_dim=expert_config.head_dim,
        )
        v = _reverse_kv_kernel(
            reader.take(f"{layer}.self_attn.v_proj.weight"),
            num_kv_heads=expert_config.num_kv_heads,
            head_dim=expert_config.head_dim,
        )
        kv.append(torch.stack([k, v], dim=0))
        attn_vec.append(
            _reverse_expert_o_kernel(
                reader.take(f"{layer}.self_attn.o_proj.weight"),
                num_heads=expert_config.num_heads,
                head_dim=expert_config.head_dim,
            )
        )
        gating.append(
            torch.stack(
                [
                    reader.take(f"{layer}.mlp.gate_proj.weight").transpose(0, 1),
                    reader.take(f"{layer}.mlp.up_proj.weight").transpose(0, 1),
                ],
                dim=0,
            )
        )
        linear.append(reader.take(f"{layer}.mlp.down_proj.weight").transpose(0, 1))

        if pi05:
            pre_attention_norm_dense_bias.append(reader.take(f"{layer}.input_layernorm.dense.bias"))
            pre_attention_norm_dense_kernel.append(reader.take(f"{layer}.input_layernorm.dense.weight").transpose(0, 1))
            pre_ffw_norm_dense_bias.append(reader.take(f"{layer}.post_attention_layernorm.dense.bias"))
            pre_ffw_norm_dense_kernel.append(
                reader.take(f"{layer}.post_attention_layernorm.dense.weight").transpose(0, 1)
            )
        else:
            pre_attention_norm_scale.append(reader.take(f"{layer}.input_layernorm.weight"))
            pre_ffw_norm_scale.append(reader.take(f"{layer}.post_attention_layernorm.weight"))

    _put(flat, "PaliGemma/llm/layers/attn/attn_vec_einsum_1/w", torch.stack(attn_vec), dtype)
    _put(flat, "PaliGemma/llm/layers/attn/kv_einsum_1/w", torch.stack(kv), dtype)
    _put(flat, "PaliGemma/llm/layers/attn/q_einsum_1/w", torch.stack(q), dtype)
    _put(flat, "PaliGemma/llm/layers/mlp_1/gating_einsum", torch.stack(gating), dtype)
    _put(flat, "PaliGemma/llm/layers/mlp_1/linear", torch.stack(linear), dtype)

    if pi05:
        _put(
            flat,
            "PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/bias",
            torch.stack(pre_attention_norm_dense_bias),
            dtype,
        )
        _put(
            flat,
            "PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel",
            torch.stack(pre_attention_norm_dense_kernel),
            dtype,
        )
        _put(flat, "PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/bias", torch.stack(pre_ffw_norm_dense_bias), dtype)
        _put(
            flat,
            "PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/kernel",
            torch.stack(pre_ffw_norm_dense_kernel),
            dtype,
        )
        _put(flat, "PaliGemma/llm/final_norm_1/Dense_0/bias", reader.take(f"{_EXPERT_PREFIX}.norm.dense.bias"), dtype)
        _put(
            flat,
            "PaliGemma/llm/final_norm_1/Dense_0/kernel",
            reader.take(f"{_EXPERT_PREFIX}.norm.dense.weight").transpose(0, 1),
            dtype,
        )
    else:
        _put(flat, "PaliGemma/llm/layers/pre_attention_norm_1/scale", torch.stack(pre_attention_norm_scale), dtype)
        _put(flat, "PaliGemma/llm/layers/pre_ffw_norm_1/scale", torch.stack(pre_ffw_norm_scale), dtype)
        _put(flat, "PaliGemma/llm/final_norm_1/scale", reader.take(f"{_EXPERT_PREFIX}.norm.weight"), dtype)


def _convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    model_config: _pi0_config.Pi0Config,
    precision: Literal["float32", "bfloat16", "float16"],
) -> dict[str, jax.Array]:
    dtype = _dtype_from_precision(precision)
    flat: dict[str, jax.Array] = {}
    reader = _StateReader(state_dict)

    _convert_vision_params(reader, flat, model_config=model_config, dtype=dtype)
    _convert_paligemma_text_params(reader, flat, model_config=model_config, dtype=dtype)
    _convert_expert_params(
        reader,
        flat,
        expert_config=_gemma.get_config(model_config.action_expert_variant),
        pi05=model_config.pi05,
        dtype=dtype,
    )
    _convert_pi0_projection_params(reader, flat, model_config=model_config, dtype=dtype)

    return traverse_util.unflatten_dict(flat, sep="/")


def _load_pytorch_state_dict(weight_path: pathlib.Path):
    return safetensors.torch.load_file(weight_path, device="cpu")


def _validate_against_jax_model(params: dict[str, jax.Array], model_config: _pi0_config.Pi0Config):
    abstract_model = nnx.eval_shape(model_config.create, jax.random.key(0))
    _, expected_state = nnx.split(abstract_model)
    expected = expected_state.to_pure_dict()

    flat_expected = traverse_util.flatten_dict(expected)
    flat_actual = traverse_util.flatten_dict(params)
    expected_keys = set(flat_expected)
    actual_keys = set(flat_actual)
    if expected_keys != actual_keys:
        missing = sorted("/".join(k) for k in expected_keys - actual_keys)
        extra = sorted("/".join(k) for k in actual_keys - expected_keys)
        raise ValueError(f"Converted params do not match JAX model keys.\nMissing: {missing[:50]}\nExtra: {extra[:50]}")

    shape_errors = []
    for key in sorted(expected_keys):
        expected_shape = tuple(flat_expected[key].shape)
        actual_shape = tuple(flat_actual[key].shape)
        if expected_shape != actual_shape:
            shape_errors.append(("/".join(key), expected_shape, actual_shape))
    if shape_errors:
        preview = "\n".join(f"{key}: expected {exp}, got {got}" for key, exp, got in shape_errors[:50])
        raise ValueError(f"Converted params do not match JAX model shapes:\n{preview}")


def _copy_assets(pytorch_checkpoint_dir: pathlib.Path, output_dir: pathlib.Path, *, overwrite: bool):
    source = pytorch_checkpoint_dir / "assets"
    if not source.exists():
        return
    dest = output_dir / "assets"
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"Output assets directory already exists: {dest}")
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def main(
    pytorch_checkpoint_dir: pathlib.Path,
    config_name: str,
    output_dir: pathlib.Path,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    overwrite: bool = False,
    skip_verify: bool = False,
):
    """Convert a PyTorch checkpoint step directory to a JAX/Orbax params checkpoint.

    Args:
        pytorch_checkpoint_dir: PyTorch checkpoint step directory containing `model.safetensors`.
        config_name: OpenPI train config name used to construct the pi0/pi0.5 model.
        output_dir: Output checkpoint step directory. The script writes `output_dir/params` and copies assets.
        precision: Array precision to store in the Orbax params checkpoint.
        overwrite: Remove an existing output directory before conversion.
        skip_verify: Skip JAX key/shape validation before saving.
    """
    pytorch_checkpoint_dir = pytorch_checkpoint_dir.resolve()
    output_dir = output_dir.resolve()
    weight_path = pytorch_checkpoint_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing PyTorch checkpoint weights: {weight_path}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(config_name)
    model_config = train_config.model
    if not isinstance(model_config, _pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name!r} is not a Pi0Config")

    print(f"Loading PyTorch checkpoint: {weight_path}")
    state_dict = _load_pytorch_state_dict(weight_path)
    print(f"Loaded {len(state_dict)} PyTorch tensors")

    print(f"Converting to JAX params with precision={precision}")
    params = _convert_state_dict(state_dict, model_config=model_config, precision=precision)

    if not skip_verify:
        print("Validating converted params against the JAX model structure")
        _validate_against_jax_model(params, model_config)

    params_dir = output_dir / "params"
    print(f"Saving Orbax params checkpoint: {params_dir}")
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(params_dir, {"params": params})

    _copy_assets(pytorch_checkpoint_dir, output_dir, overwrite=overwrite)
    metadata = {
        "source_pytorch_checkpoint_dir": str(pytorch_checkpoint_dir),
        "config_name": config_name,
        "precision": precision,
        "contains_train_state": False,
        "intended_use": "JAX inference/deployment params checkpoint",
    }
    (output_dir / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Conversion completed: {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
