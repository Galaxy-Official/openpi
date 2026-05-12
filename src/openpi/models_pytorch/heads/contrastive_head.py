"""Vision-touch/force contrastive projection head."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.encoders._utils import masked_mean_pool


class VisionProjector(nn.Module):
    """Project pooled vision tokens into the contrastive embedding space."""

    def __init__(
        self,
        embed_dim: int,
        *,
        proj_dim: int = 256,
        hidden_dim: int | None = None,
        preferred_keys: tuple[str, ...] = ("left_wrist_0_rgb", "wrist", "base_0_rgb", "phone"),
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.preferred_keys = preferred_keys
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(
        self,
        image_tokens: torch.Tensor | dict[str, torch.Tensor],
        image_masks: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, valid = self._pool_inputs(image_tokens, image_masks)
        z_vision = F.normalize(self.projector(pooled), dim=-1)
        z_vision = torch.where(valid[:, None], z_vision, torch.zeros_like(z_vision))
        return z_vision, valid

    def _pool_inputs(
        self,
        image_tokens: torch.Tensor | dict[str, torch.Tensor],
        image_masks: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(image_tokens, torch.Tensor):
            token_mask = image_masks if isinstance(image_masks, torch.Tensor) and image_masks.ndim == 2 else None
            pooled, token_valid = masked_mean_pool(image_tokens, token_mask)
            sample_valid = (
                _sample_valid(image_masks, image_tokens.shape[0], image_tokens.device)
                if image_masks is not None
                else token_valid
            )
            return torch.where(sample_valid[:, None], pooled, torch.zeros_like(pooled)), sample_valid

        ordered_keys = [k for k in self.preferred_keys if k in image_tokens]
        ordered_keys += [k for k in image_tokens if k not in ordered_keys]
        if not ordered_keys:
            raise ValueError("image_tokens dict is empty")

        pooled_options = []
        valid_options = []
        for key in ordered_keys:
            tokens = image_tokens[key]
            masks_for_key = image_masks.get(key) if isinstance(image_masks, dict) else None
            token_mask = masks_for_key if isinstance(masks_for_key, torch.Tensor) and masks_for_key.ndim == 2 else None
            pooled, token_valid = masked_mean_pool(tokens, token_mask)
            valid = (
                _sample_valid(masks_for_key, tokens.shape[0], tokens.device)
                if masks_for_key is not None
                else token_valid
            )
            pooled_options.append(pooled)
            valid_options.append(valid)

        pooled = torch.zeros_like(pooled_options[0])
        chosen = torch.zeros_like(valid_options[0])
        for option, valid in zip(pooled_options, valid_options, strict=True):
            take = valid & ~chosen
            pooled = torch.where(take[:, None], option, pooled)
            chosen = chosen | take
        return pooled, chosen


class ContrastiveHead(nn.Module):
    """Compute symmetric InfoNCE between vision and touch/force embeddings."""

    def __init__(self, *, temperature: float = 0.07, neg_window: int = 8):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature), dtype=torch.float32))
        self.neg_window = neg_window

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-4, max=1.0)

    def forward(
        self,
        z_vision: torch.Tensor,
        z_touchforce: torch.Tensor,
        *,
        valid_pair_mask: torch.Tensor | None = None,
        task_index: torch.Tensor | None = None,
        frame_index: torch.Tensor | None = None,
        neg_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if neg_mask is None and task_index is not None and frame_index is not None:
            neg_mask = build_task_aware_neg_mask(task_index, frame_index, self.neg_window)
        loss, metrics = info_nce(
            z_vision,
            z_touchforce,
            temperature=self.temperature,
            valid_pair_mask=valid_pair_mask,
            neg_mask=neg_mask,
        )
        return {"loss": loss, **metrics}


def build_task_aware_neg_mask(task_index: torch.Tensor, frame_index: torch.Tensor, window: int) -> torch.Tensor:
    task_index = task_index.reshape(-1)
    frame_index = frame_index.reshape(-1)
    same_task = task_index[:, None] == task_index[None, :]
    near_frame = (frame_index[:, None] - frame_index[None, :]).abs() < window
    eye = torch.eye(task_index.shape[0], dtype=torch.bool, device=task_index.device)
    return same_task & near_frame & ~eye


def info_nce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    temperature: float | torch.Tensor = 0.07,
    valid_pair_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if z_a.shape != z_b.shape:
        raise ValueError(f"Embedding shapes must match, got {tuple(z_a.shape)} and {tuple(z_b.shape)}")
    if z_a.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got {tuple(z_a.shape)}")

    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    if isinstance(temperature, torch.Tensor):
        temperature_t = temperature.to(dtype=z_a.dtype, device=z_a.device)
    else:
        temperature_t = torch.tensor(temperature, dtype=z_a.dtype, device=z_a.device)
    logits = z_a @ z_b.T / temperature_t.clamp_min(1e-6)

    batch_size = z_a.shape[0]
    valid = (
        torch.ones(batch_size, dtype=torch.bool, device=z_a.device)
        if valid_pair_mask is None
        else valid_pair_mask.to(dtype=torch.bool, device=z_a.device).reshape(batch_size)
    )
    if int(valid.sum().item()) == 0:
        zero = logits.sum() * 0.0
        return zero, _empty_metrics(zero)

    neg_mask = torch.zeros_like(logits, dtype=torch.bool) if neg_mask is None else neg_mask.to(torch.bool)
    eye = torch.eye(batch_size, dtype=torch.bool, device=z_a.device)
    neg_mask = neg_mask & ~eye

    loss_ab = _masked_cross_entropy(logits, valid, valid, neg_mask)
    loss_ba = _masked_cross_entropy(logits.T, valid, valid, neg_mask.T)
    loss = 0.5 * (loss_ab + loss_ba)

    with torch.no_grad():
        masked_logits = _mask_logits(logits, valid, neg_mask)
        rows = valid
        preds = masked_logits[rows].argmax(dim=-1)
        targets = torch.arange(batch_size, device=z_a.device)[rows]
        acc = (preds == targets).to(torch.float32).mean()
        pos_sim = (z_a * z_b).sum(dim=-1)[valid].mean()
        valid_pairs = valid[:, None] & valid[None, :] & ~eye & ~neg_mask
        neg_sim = (z_a @ z_b.T)[valid_pairs].mean() if valid_pairs.any() else torch.zeros((), device=z_a.device)

    return loss, {"acc@1": acc, "pos_sim": pos_sim, "neg_sim": neg_sim}


def _masked_cross_entropy(
    logits: torch.Tensor,
    row_valid: torch.Tensor,
    col_valid: torch.Tensor,
    neg_mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = _mask_logits(logits, col_valid, neg_mask)
    rows = row_valid
    targets = torch.arange(logits.shape[0], device=logits.device)[rows]
    return F.cross_entropy(masked_logits[rows], targets)


def _mask_logits(logits: torch.Tensor, col_valid: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~col_valid[None, :], torch.finfo(logits.dtype).min)
    return masked_logits.masked_fill(neg_mask, torch.finfo(logits.dtype).min)


def _sample_valid(mask, batch_size: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if isinstance(mask, bool):
        return torch.full((batch_size,), mask, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if mask.ndim == 2:
        return mask.any(dim=1)
    return mask.reshape(batch_size)


def _empty_metrics(zero: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "acc@1": zero.detach(),
        "pos_sim": zero.detach(),
        "neg_sim": zero.detach(),
    }
