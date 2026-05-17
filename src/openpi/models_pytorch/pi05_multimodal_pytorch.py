"""Multi-modal pi0.5 model: pi0.5 backbone + touch/force fusion + contrastive head.

Subclasses `PI0Pytorch` so the original module is left completely unchanged.
This file pulls together:
- `TactileEncoder` (M1) and `ForceEncoder` (M2) for new modality inputs.
- `TouchForceFusion` (M3) that produces fused prefix tokens + a pooled
  `z_touchforce` projection.
- `VisionProjector` + `ContrastiveHead` (M4) for the vision↔touch-force
  InfoNCE auxiliary loss.

Training is single-run with three implicit phases controlled by `global_step`
and `Pi0ConfigMM.{encoder_warmup_steps, contrast_warmup_start, contrast_warmup_end}`:
  1. `[0, encoder_warmup_steps)`: backbone frozen, only the new modules learn,
     loss = L_contrast only.
  2. `[encoder_warmup_steps, contrast_warmup_start)`: everything trains under
     pure L_action.
  3. `[contrast_warmup_start, ∞)`: L_action + warmup·λ_c·L_contrast, where
     warmup ramps 0→1 over [contrast_warmup_start, contrast_warmup_end].

The class is intentionally tolerant of missing modalities: with `use_tactile`,
`use_force`, `use_fusion`, or `use_contrast` toggled off it degrades to plain
pi0.5 behaviour. When fused tokens are skipped the prefix matches the original
`embed_prefix` shape exactly, so weight loaders that target plain pi0.5 keep
working.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models import pi0_config_mm as _pi0_config_mm
import openpi.models.gemma as _gemma
from openpi.models_pytorch import pi0_pytorch as _pi0_pytorch
from openpi.models_pytorch.encoders import ForceEncoder
from openpi.models_pytorch.encoders import TactileEncoder
from openpi.models_pytorch.encoders import TouchForceFusion
from openpi.models_pytorch.heads import contrastive_head as _contrastive
from openpi.models_pytorch.heads.contrastive_head import ContrastiveHead
from openpi.models_pytorch.heads.contrastive_head import VisionProjector

# Index of the wrist camera inside `IMAGE_KEYS` (preprocessing_pytorch.py:11). The
# tactile/force fusion contrastive head pairs against the wrist image, which is
# also the only camera kept in the wrist-only training path.
_WRIST_IMAGE_INDEX = 1
_IMAGE_NAMES: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class PI05Multimodal(_pi0_pytorch.PI0Pytorch):
    """pi0.5 augmented with touch/force fusion and a vision↔touchforce InfoNCE head."""

    def __init__(self, config: _pi0_config_mm.Pi0ConfigMM):
        if not isinstance(config, _pi0_config_mm.Pi0ConfigMM):
            raise TypeError(f"PI05Multimodal requires Pi0ConfigMM, got {type(config).__name__}")
        if not config.pi05:
            raise ValueError("PI05Multimodal currently targets pi0.5 only (pi05=True)")
        self.validate_toggles(config)
        super().__init__(config)
        self.mm_config = config

        # Embedding width of PaliGemma; encoders project into this space so the
        # fused tokens drop into the prefix unchanged.
        embed_dim = _gemma.get_config(config.paligemma_variant).width

        self.tactile_encoder: TactileEncoder | None = None
        self.force_encoder: ForceEncoder | None = None
        self.touch_force_fusion: TouchForceFusion | None = None
        self.vision_projector: VisionProjector | None = None
        self.contrastive_head: ContrastiveHead | None = None

        if config.use_tactile:
            tactile_embedder = None
            if config.tactile_use_siglip:
                # Reuse PaliGemma's SigLIP weights for tactile patches as
                # described in the plan. Cast back to float32 so the
                # downstream LazyLinear / LayerNorm (kept at float32 for
                # numerical stability) doesn't hit a dtype mismatch.
                def _tactile_siglip(image: Tensor) -> Tensor:
                    return self.paligemma_with_expert.embed_image(image).to(torch.float32)

                tactile_embedder = _tactile_siglip
            self.tactile_encoder = TactileEncoder(
                embed_dim=embed_dim,
                num_tokens_per_side=config.n_tactile_tokens_per_side,
                vision_embedder=tactile_embedder,
            )
        if config.use_force:
            self.force_encoder = ForceEncoder(
                embed_dim=embed_dim,
                input_dim=config.force_input_dim,
                num_tokens=config.n_force_tokens,
            )
        if config.use_fusion:
            self.touch_force_fusion = TouchForceFusion(
                embed_dim=embed_dim,
                num_fused_tokens=config.n_fused_tokens,
                proj_dim=config.contrast_proj_dim,
            )
        if config.use_contrast:
            self.vision_projector = VisionProjector(
                embed_dim=embed_dim,
                proj_dim=config.contrast_proj_dim,
            )
            self.contrastive_head = ContrastiveHead(
                temperature=config.contrast_temperature,
                neg_window=config.contrast_neg_window,
            )

        # State maintained by the train loop; consulted by `freeze_backbone_for_warmup`.
        self._frozen_backbone = False

    @staticmethod
    def validate_toggles(config: _pi0_config_mm.Pi0ConfigMM) -> None:
        """Catch incoherent module toggles before paying the backbone build cost."""
        if config.use_fusion and not (config.use_tactile and config.use_force):
            raise ValueError("use_fusion=True requires use_tactile=True and use_force=True")
        if config.use_contrast and not config.use_fusion:
            raise ValueError("use_contrast=True requires use_fusion=True")
        if config.use_vision_latent_contrast:
            # The field + weight are reserved for a future PR. Fail loudly so
            # nobody silently trains a model that thinks it is computing this
            # auxiliary loss when it isn't.
            raise NotImplementedError(
                "use_vision_latent_contrast is not implemented yet; set it to False (the default)."
            )

    # ------------------------------------------------------------------ utils

    @property
    def _has_fusion(self) -> bool:
        return self.touch_force_fusion is not None

    @property
    def _has_contrast(self) -> bool:
        return self.contrastive_head is not None and self.vision_projector is not None

    def _backbone_params(self) -> Iterable[nn.Parameter]:
        """All parameters that live in the pretrained pi0.5 backbone."""
        yield from self.paligemma_with_expert.parameters()
        yield from self.action_in_proj.parameters()
        yield from self.action_out_proj.parameters()
        if self.mm_config.pi05:
            yield from self.time_mlp_in.parameters()
            yield from self.time_mlp_out.parameters()
        else:
            yield from self.state_proj.parameters()
            yield from self.action_time_mlp_in.parameters()
            yield from self.action_time_mlp_out.parameters()

    def _new_module_params(self) -> Iterable[nn.Parameter]:
        for name in self.mm_config.encoder_warmup_modules:
            module = getattr(self, name, None)
            if module is None:
                continue
            yield from module.parameters()

    def freeze_backbone_for_warmup(self) -> None:
        """Disable grad on the pretrained pi0.5 backbone; keep new modules trainable."""
        for p in self._backbone_params():
            p.requires_grad = False
        for p in self._new_module_params():
            p.requires_grad = True
        self._frozen_backbone = True

    def unfreeze_all(self) -> None:
        """Re-enable grad on every parameter (called at the end of warm-up)."""
        for p in self.parameters():
            p.requires_grad = True
        self._frozen_backbone = False

    # ----------------------------------------------------- modality forward

    def _encode_touch_force(self, observation) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Return (fused_tokens, fused_mask, z_touchforce, valid_per_sample) or Nones."""
        if not self._has_fusion:
            return None, None, None, None

        tactile = getattr(observation, "tactile", None)
        tactile_mask = getattr(observation, "tactile_mask", None)
        force = getattr(observation, "force", None)
        force_mask = getattr(observation, "force_mask", None)

        if tactile is None or force is None:
            return None, None, None, None

        device = self._first_param_device()
        tac_l = _to_tensor(tactile.get("left"), device)
        tac_r = _to_tensor(tactile.get("right"), device)
        tactile_tokens, tactile_pad = self.tactile_encoder(tac_l, tac_r, mask=tactile_mask)

        force_tokens, force_pad = self.force_encoder(
            {"left": _to_tensor(force.get("left"), device), "right": _to_tensor(force.get("right"), device)},
            mask=force_mask,
        )

        fused_tokens, fused_mask, z_touchforce = self.touch_force_fusion(
            tactile_tokens,
            force_tokens,
            tactile_mask=tactile_pad,
            force_mask=force_pad,
        )
        valid = _sample_valid(tactile_mask) & _sample_valid(force_mask)
        return fused_tokens, fused_mask, z_touchforce, valid.to(device=fused_tokens.device)

    def _project_vision(
        self,
        image_tokens_list: list[Tensor],
        image_masks_list: list[Tensor],
    ) -> tuple[Tensor | None, Tensor | None]:
        if not self._has_contrast:
            return None, None
        # The PaliGemma backbone runs in bfloat16; the new modules stay in float32 for
        # numerical stability (LayerNorms in particular). Cast image tokens at the
        # boundary so VisionProjector's `nn.Linear` doesn't hit a dtype mismatch.
        proj_dtype = next(self.vision_projector.parameters()).dtype
        token_dict = dict(zip(_IMAGE_NAMES, [t.to(dtype=proj_dtype) for t in image_tokens_list], strict=True))
        mask_dict = dict(zip(_IMAGE_NAMES, image_masks_list, strict=True))
        z_vision, valid = self.vision_projector(token_dict, image_masks=mask_dict)
        return z_vision, valid

    def _first_param_device(self) -> torch.device:
        return next(self.parameters()).device

    # -------------------------------------------------- prefix construction

    def _embed_prefix_with_fusion(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        fused_tokens: Tensor | None,
        fused_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor], list[Tensor]]:
        """Like `PI0Pytorch.embed_prefix`, but inserts `fused_tokens` between images and language.

        Also returns the per-camera image token / mask streams so the contrastive
        head can re-pool from them without re-running SigLIP.
        """
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks: list[int] = []
        image_token_streams: list[Tensor] = []
        image_mask_streams: list[Tensor] = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            expanded_mask = img_mask[:, None].expand(bsize, num_img_embs)
            pad_masks.append(expanded_mask)
            att_masks += [0] * num_img_embs
            image_token_streams.append(img_emb)
            image_mask_streams.append(expanded_mask)

        if fused_tokens is not None and fused_mask is not None:
            embs.append(fused_tokens)
            pad_masks.append(fused_mask)
            att_masks += [0] * fused_tokens.shape[1]

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs_t = torch.cat(embs, dim=1)
        pad_masks_t = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks_t.device)
        bsize = pad_masks_t.shape[0]
        att_masks_t = att_masks_t[None, :].expand(bsize, len(att_masks))

        return embs_t, pad_masks_t, att_masks_t, image_token_streams, image_mask_streams

    # ------------------------------------------------------------- forward

    def forward(self, observation, actions, noise=None, time=None) -> dict[str, Tensor | None]:
        """Run pi0.5 forward augmented with touch/force fusion + contrastive heads.

        Returns a dict (instead of pi0.5's scalar MSE) so the train loop can both
        log per-component metrics and call `compute_total_loss`. The flow-matching
        residual is preserved unchanged: only `embed_prefix` is extended.
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        fused_tokens, fused_mask, z_touchforce, tf_valid = self._encode_touch_force(observation)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            image_tokens_list,
            image_mask_list,
        ) = self._embed_prefix_with_fusion(images, img_masks, lang_tokens, lang_masks, fused_tokens, fused_mask)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        first_layer_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if first_layer_dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = _pi0_pytorch.make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        action_loss = F.mse_loss(u_t, v_t, reduction="none")

        z_vision, vision_valid = self._project_vision(image_tokens_list, image_mask_list)

        # Prefer the global multi-task index injected by the data pipeline; fall
        # back to LeRobot's per-repo `task_index` (mostly always 0) for legacy
        # single-task setups.
        task_id_for_negmask = getattr(observation, "mm_task_index", None)
        if task_id_for_negmask is None:
            task_id_for_negmask = getattr(observation, "task_index", None)
        return {
            "action_loss": action_loss,
            "z_vision": z_vision,
            "z_touchforce": z_touchforce,
            "vision_valid": vision_valid,
            "touchforce_valid": tf_valid,
            "task_index": task_id_for_negmask,
            "frame_index": getattr(observation, "frame_index", None),
        }

    # ------------------------------------------------------------- loss agg

    def compute_total_loss(
        self,
        outputs: dict[str, Tensor | None],
        global_step: int,
    ) -> dict[str, Tensor | str | int | float]:
        """Three-phase loss aggregation (see module docstring)."""
        cfg = self.mm_config
        phase = self.phase_for(global_step)

        action_loss_raw = outputs.get("action_loss")
        if action_loss_raw is None:
            raise ValueError("Model output is missing `action_loss`")
        action_loss = action_loss_raw.mean()

        contrast_metrics = self._contrastive_metrics(outputs)
        contrast_loss = contrast_metrics["loss"]

        if phase == "encoder_warmup":
            total = contrast_loss if contrast_loss is not None else action_loss * 0.0
            action_for_log = action_loss.detach()
        elif phase == "action_only":
            total = action_loss
            action_for_log = action_loss
        else:  # joint
            warmup = self.contrast_warmup_factor(global_step)
            total = action_loss if contrast_loss is None else action_loss + warmup * cfg.contrast_weight * contrast_loss
            action_for_log = action_loss

        log: dict[str, Tensor | str | int | float] = {
            "total": total,
            "action_loss": action_for_log,
            "contrast_loss": contrast_loss.detach() if contrast_loss is not None else None,
            "pos_sim": contrast_metrics["pos_sim"],
            "neg_sim": contrast_metrics["neg_sim"],
            "acc@1": contrast_metrics["acc@1"],
            "phase": phase,
            "global_step": int(global_step),
            "contrast_warmup": self.contrast_warmup_factor(global_step),
        }
        return log

    def _contrastive_metrics(self, outputs: dict[str, Tensor | None]) -> dict[str, Tensor | None]:
        if not self._has_contrast:
            return {"loss": None, "pos_sim": None, "neg_sim": None, "acc@1": None}
        z_vision = outputs.get("z_vision")
        z_touchforce = outputs.get("z_touchforce")
        if z_vision is None or z_touchforce is None:
            return {"loss": None, "pos_sim": None, "neg_sim": None, "acc@1": None}

        vision_valid = outputs.get("vision_valid")
        tf_valid = outputs.get("touchforce_valid")
        valid_pair = _combine_valid(vision_valid, tf_valid, batch_size=z_vision.shape[0], device=z_vision.device)

        task_index = outputs.get("task_index")
        frame_index = outputs.get("frame_index")
        neg_mask = None
        if task_index is not None and frame_index is not None:
            neg_mask = _contrastive.build_task_aware_neg_mask(
                torch.as_tensor(task_index, device=z_vision.device),
                torch.as_tensor(frame_index, device=z_vision.device),
                window=self.mm_config.contrast_neg_window,
            )

        out = self.contrastive_head(
            z_vision,
            z_touchforce,
            valid_pair_mask=valid_pair,
            neg_mask=neg_mask,
        )
        return {
            "loss": out["loss"],
            "pos_sim": out["pos_sim"],
            "neg_sim": out["neg_sim"],
            "acc@1": out["acc@1"],
        }

    # ----------------------------------------------------- inference override

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Mirror `PI0Pytorch.sample_actions` but use the fused prefix.

        Without this override, the inherited `sample_actions` would call the
        base `embed_prefix(images, img_masks, lang_tokens, lang_masks)` which
        ignores tactile/force entirely; the model would then see a prefix
        distribution that doesn't match what it was trained on.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        fused_tokens, fused_mask, _, _ = self._encode_touch_force(observation)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = self._embed_prefix_with_fusion(
            images, img_masks, lang_tokens, lang_masks, fused_tokens, fused_mask
        )
        prefix_att_2d_masks = _pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        first_layer_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if first_layer_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    # ------------------------------------------------------------ scheduling

    def phase_for(self, step: int) -> str:
        cfg = self.mm_config
        if cfg.encoder_warmup_steps > 0 and step < cfg.encoder_warmup_steps:
            return "encoder_warmup"
        if step < cfg.contrast_warmup_start:
            return "action_only"
        return "joint"

    def contrast_warmup_factor(self, step: int) -> float:
        cfg = self.mm_config
        if step <= cfg.contrast_warmup_start:
            return 0.0
        if step >= cfg.contrast_warmup_end:
            return 1.0
        span = max(cfg.contrast_warmup_end - cfg.contrast_warmup_start, 1)
        return float(step - cfg.contrast_warmup_start) / span


# ------------------------------------------------------------- module helpers


def _to_tensor(value, device: torch.device) -> Tensor | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


_TRUE_SCALAR = torch.tensor(True)  # noqa: FBT003 (module-level singleton, not a flag arg)


def _sample_valid(mask) -> Tensor:
    """Reduce a per-side mask dict (or tensor) into a per-sample boolean."""
    if mask is None:
        return _TRUE_SCALAR
    if isinstance(mask, dict):
        parts = [m for m in mask.values() if m is not None]
        if not parts:
            return _TRUE_SCALAR
        merged = torch.as_tensor(parts[0], dtype=torch.bool)
        for m in parts[1:]:
            merged = merged | torch.as_tensor(m, dtype=torch.bool)
        return merged
    return torch.as_tensor(mask, dtype=torch.bool)


def _combine_valid(
    vision_valid: Tensor | None,
    tf_valid: Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    def _norm(value: Tensor | None) -> Tensor:
        if value is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        v = torch.as_tensor(value, dtype=torch.bool, device=device).reshape(-1)
        if v.numel() != batch_size:
            raise ValueError(f"valid mask must have batch_size={batch_size}, got shape {v.shape}")
        return v

    return _norm(vision_valid) & _norm(tf_valid)
