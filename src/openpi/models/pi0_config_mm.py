"""Multi-modal extension of `Pi0Config` for force/tactile fusion + contrastive pretraining.

Kept in a separate file so the original `Pi0Config` (and the wrist-only vision
training path that ships with the repo) stay untouched.

This config carries:
- Modality toggles (`use_tactile`, `use_force`, `use_fusion`, `use_contrast`)
- Encoder shape hyperparameters
- Contrastive loss hyperparameters
- Encoder warm-up schedule (see PR4 plan)
- Per-modality dropout probabilities

Pi0ConfigMM still reports `ModelType.PI05` so the tokenizer/data transforms used
by pi0.5 continue to apply unchanged. The PI05Multimodal subclass branches on
the toggles to construct extra modules.
"""

from __future__ import annotations

import dataclasses

from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config


@dataclasses.dataclass(frozen=True)
class Pi0ConfigMM(_pi0_config.Pi0Config):
    """Configuration for the multi-modal pi0.5 model."""

    # Force pi05 semantics by default; the multi-modal model targets pi0.5.
    pi05: bool = True

    # ---- Modality toggles ----------------------------------------------------
    use_tactile: bool = True
    use_force: bool = True
    use_fusion: bool = True
    use_contrast: bool = True
    use_vision_latent_contrast: bool = False
    use_depth: bool = False  # placeholder, not consumed yet

    # ---- Encoder shapes ------------------------------------------------------
    n_tactile_tokens_per_side: int = 16
    n_force_tokens: int = 4
    n_fused_tokens: int = 32
    contrast_proj_dim: int = 256
    tactile_frame_stack: int = 1
    force_window: int = 16
    force_input_dim: int = 2  # left + right scalar -> stacked into 2 channels
    # If True, TactileEncoder shares PaliGemma's SigLIP vision tower as its
    # patch embedder (matches the original plan). False uses the lightweight
    # randomly-initialized patch stem from PR3 and keeps the tactile branch
    # independent of the pretrained image features.
    tactile_use_siglip: bool = False

    # ---- Contrastive head ----------------------------------------------------
    contrast_weight: float = 0.1
    contrast_temperature: float = 0.07
    contrast_warmup_start: int = 5_000
    contrast_warmup_end: int = 20_000
    contrast_neg_window: int = 8
    latent_contrast_weight: float = 0.05

    # ---- Encoder warm-up schedule (PR4 plan) --------------------------------
    encoder_warmup_steps: int = 2_000
    # Module attribute names allowed to receive gradients during encoder warm-up.
    encoder_warmup_modules: tuple[str, ...] = (
        "tactile_encoder",
        "force_encoder",
        "touch_force_fusion",
        "vision_projector",
        "contrastive_head",
    )

    # ---- Per-modality dropout (applied in the data pipeline, surfaced here
    # so consumers can plumb the values through a single config object). -------
    tactile_drop: float = 0.10
    force_drop: float = 0.10
    vision_phone_drop: float = 0.05
    vision_wrist_drop: float = 0.05

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Stays as PI05 so the tokenizer + model_transforms pipeline reused by
        # pi0.5 continues to apply without modification.
        return _model.ModelType.PI05
