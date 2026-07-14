#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright 2026 The BaseVLA Authors.
#
# The flow-matching action expert and VLM-prefix scaffolding derive from the
# openpi / LeRobot pi0 policy (Apache-2.0). PelicanVLA05 adds a Qwen3-VL backbone, a
# frozen NVIDIA Cosmos image-tokenizer branch, bottleneck tokens, and split
# backbone/head checkpointing. See PROVENANCE.md for a method-level breakdown.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import torch._dynamo as dynamo
from safetensors.torch import load_file, save_file

from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.basevla_4B.cosmos_tokenizer.image_lib import ImageTokenizer
from lerobot.policies.basevla_4B.configuration_basevla import PelicanVLA05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import format_big_number


BACKBONE_FILE = "backbone.safetensors"
HEADS_FILE = "heads.safetensors"
ACTION_HEAD_FILE = "action_head.safetensors"
GEN_HEAD_FILE = "gen_head.safetensors"
TASK_HEAD_FILE = "task_head.safetensors"

BACKBONE_PREFIXES = (
    "model.vlm.",
    "model.cosmos_in_proj.",
    "model.downsample_conv.",
    "model.bottleneck_tokens",
    # Flow-matching input projections: dimensions are max_state_dim/max_action_dim
    # (universal across robots via zero-padding), so they belong in the backbone.
    "model.state_proj.",
    "model.action_in_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)

ACTION_HEAD_PREFIXES = (
    "model.act_head.",
)

GEN_HEAD_PREFIXES = (
    "model.gen_head.",
    "model.upsample_conv.",
    "model.cosmos_out_layer_norm.",
    "model.cosmos_out_proj.",
    "model.slot_gate.",
)

TASK_HEAD_PREFIXES = (
    "model.task_proj_traj.",
    "model.task_proj_text.",
    "model.logit_scale",
)

HEAD_PREFIXES = ACTION_HEAD_PREFIXES + GEN_HEAD_PREFIXES + TASK_HEAD_PREFIXES

SKIP_PREFIXES = (
    "model.cosmos.",
)


@dataclass
class _ActionContext:
    """Immutable outputs of the observation prefill used during denoising."""

    key_values: object
    valid_tokens: Tensor
    last_position: Tensor
    middle_hidden: Tensor
    slot_hidden: Tensor | None


from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    OBS_PREFIX, 
    OBS_IMAGES, 
    OPENPI_ATTENTION_MASK_VALUE,
)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def _declared_backbone_depth(variant: str) -> int | None:
    """Return the layer count encoded by names such as ``qwen3_vl_36l``."""
    suffix = variant.rpartition("_")[2]
    if suffix.endswith("l") and suffix[:-1].isdigit():
        return int(suffix[:-1])
    return None


def _load_qwen3vl_config(model_path: str, variant: str) -> Qwen3VLConfig:
    """Load architecture data from the Apache-licensed upstream model config.

    Keeping the authoritative dimensions in the Hugging Face config avoids a
    second, hand-maintained copy of Qwen's architecture.  The local variant is
    retained only as a checkpoint compatibility assertion.
    """
    hf_config = Qwen3VLConfig.from_pretrained(model_path)
    expected_depth = _declared_backbone_depth(variant)
    actual_depth = hf_config.text_config.num_hidden_layers
    if expected_depth is not None and expected_depth != actual_depth:
        raise ValueError(
            f"{variant!r} requires {expected_depth} language layers, but "
            f"{model_path!r} declares {actual_depth}"
        )
    return hf_config


def _resolve_cosmos_jit_files(config: PelicanVLA05Config) -> tuple[Path, Path]:
    """Resolve the separately licensed Cosmos encoder and decoder assets."""
    configured_path = config.cosmos_tokenizer_path or os.environ.get(
        "COSMOS_TOKENIZER_PATH"
    )
    if configured_path:
        model_dir = Path(configured_path).expanduser()
    else:
        from huggingface_hub import snapshot_download

        logging.info(
            "Downloading Cosmos tokenizer %s at revision %s",
            config.cosmos_tokenizer_repo_id,
            config.cosmos_tokenizer_revision,
        )
        model_dir = Path(
            snapshot_download(
                repo_id=config.cosmos_tokenizer_repo_id,
                revision=config.cosmos_tokenizer_revision,
                allow_patterns=["encoder.jit", "decoder.jit"],
            )
        )

    encoder_path = model_dir / "encoder.jit"
    decoder_path = model_dir / "decoder.jit"
    missing = [
        str(path) for path in (encoder_path, decoder_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cosmos tokenizer requires both encoder.jit and decoder.jit; missing: "
            + ", ".join(missing)
        )
    return encoder_path, decoder_path


class PelicanVLA05(nn.Module):

    def __init__(self, config: PelicanVLA05Config):
        super().__init__()
        self.config = config

        # ---- Build VLM (Qwen3-VL) ----
        qwen_path = config.qwen3_vl_path or os.environ.get("QWEN3_VL_PATH", "Qwen/Qwen3-VL-4B-Instruct")
        vlm_config = _load_qwen3vl_config(qwen_path, config.qwen3_vl_variant)
        vlm_dim = vlm_config.text_config.hidden_size
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_path,
            config=vlm_config,
        )

        # ---- Cosmos tokenizer (frozen) ----
        cosmos_encoder, cosmos_decoder = _resolve_cosmos_jit_files(config)
        cosmos_device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        cosmos_dtype = (
            torch.bfloat16 if cosmos_device.type == "cuda" else torch.float32
        )
        self.cosmos = ImageTokenizer(
            checkpoint_enc=cosmos_encoder,
            checkpoint_dec=cosmos_decoder,
            device=cosmos_device,
            dtype=cosmos_dtype,
        )

        # ---- Cosmos encode projections (input → vlm_dim) ----
        vae_dim = 16
        ds = self.config.scale_factor
        self.cosmos_in_proj = nn.Conv2d(in_channels=vae_dim, out_channels=vlm_dim, kernel_size=1, stride=1, padding=0)
        self.downsample_conv = nn.Conv2d(in_channels=vlm_dim, out_channels=vlm_dim, kernel_size=ds, stride=ds, padding=0)

        # ---- Gen head MLP + spatial decode pipeline ----
        gen_hidden = config.gen_head_hidden_dim
        self.gen_head = nn.Sequential(
            nn.LayerNorm(vlm_dim),
            nn.Linear(vlm_dim, gen_hidden),
            nn.SiLU(),
            nn.Linear(gen_hidden, gen_hidden),
        )
        self.upsample_conv = nn.ConvTranspose2d(in_channels=gen_hidden, out_channels=gen_hidden, kernel_size=ds, stride=ds, padding=0, output_padding=0)
        self.cosmos_out_layer_norm = nn.LayerNorm(gen_hidden)
        self.cosmos_out_proj = nn.Linear(gen_hidden, vae_dim)

        # ---- Act head MLP ----
        act_hidden = config.act_head_hidden_dim
        self.act_head = nn.Sequential(
            nn.LayerNorm(vlm_dim),
            nn.Linear(vlm_dim, act_hidden),
            nn.SiLU(),
            nn.Linear(act_hidden, config.max_action_dim),
        )

        # ---- State & action input projections (→ vlm_dim) ----
        self.state_proj = nn.Linear(config.max_state_dim, vlm_dim)
        self.action_in_proj = nn.Linear(config.max_action_dim, vlm_dim)
        self.action_time_mlp_in = nn.Linear(2 * vlm_dim, vlm_dim)
        self.action_time_mlp_out = nn.Linear(vlm_dim, vlm_dim)

        # ---- Bottleneck Tokens (Phase 1) ----
        if config.enable_bottleneck_tokens:
            K = config.bottleneck_num_tokens
            self.bottleneck_tokens = nn.Parameter(
                torch.randn(1, K, vlm_dim) * 0.02
            )
            self.slot_dropout = nn.Dropout(p=config.reasoning_dropout)
            self.slot_gate = nn.Sequential(
                nn.Linear(vlm_dim, gen_hidden),
                nn.Sigmoid(),
            )

        # ---- InfoNCE Task Contrastive (Phase 2) ----
        if config.enable_task_contrastive:
            self.task_proj_traj = nn.Sequential(
                nn.Linear(vlm_dim, config.task_embed_dim),
                nn.LayerNorm(config.task_embed_dim),
            )
            self.task_proj_text = nn.Sequential(
                nn.Linear(vlm_dim, config.task_embed_dim),
                nn.LayerNorm(config.task_embed_dim),
            )
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        # ---- Precision ----
        self._apply_precision(config.dtype)

        # ---- Gradient checkpointing ----
        self.gradient_checkpointing_enabled = False

        # ---- Compile ----
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        self.set_requires_grad()

    def _apply_precision(self, precision: str):
        if precision == "bfloat16":
            self.vlm.to(dtype=torch.bfloat16)
            params_to_keep_float32 = [
                "input_layernorm",
                "post_attention_layernorm",
                "model.norm",
            ]
            for name, param in self.vlm.named_parameters():
                if any(selector in name for selector in params_to_keep_float32):
                    param.data = param.data.to(dtype=torch.float32)
        elif precision == "float32":
            pass
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def set_requires_grad(self):
        if self.config.freeze_backbone:
            for name, param in self.named_parameters():
                is_backbone = any(name.startswith(p.removeprefix("model.")) for p in BACKBONE_PREFIXES)
                if is_backbone:
                    param.requires_grad = False
            self.vlm.eval()
            logging.info("Froze all backbone parameters (VLM + projections + slots)")
        elif self.config.freeze_vision_encoder:
            self.vlm.visual.eval()
            for params in self.vlm.visual.parameters():
                params.requires_grad = False

        self.cosmos.eval()
        for params in self.cosmos.parameters():
            params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_backbone:
            self.vlm.eval()
        elif self.config.freeze_vision_encoder:
            self.vlm.visual.eval()

        self.cosmos.eval()
        return self

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.vlm.language_model.gradient_checkpointing = True
        self.vlm.visual.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PelicanVLA05 model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.vlm.language_model.gradient_checkpointing = False
        self.vlm.visual.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PelicanVLA05 model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        vlm_dtype = self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE).to(dtype=vlm_dtype)

    def _cast_embs_for_vlm(self, *embs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Match auxiliary embeddings to the VLM language-model compute dtype."""
        vlm_dtype = self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype
        return tuple(emb.to(dtype=vlm_dtype) if emb.dtype != vlm_dtype else emb for emb in embs)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    @dynamo.disable
    def embed_prefix(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build Qwen's interleaved image-and-language input sequence."""
        patch_matrix = pixel_values.reshape(-1, pixel_values.shape[-1])
        grid = image_grid_thw.reshape(-1, 3)
        visual_tokens, _ = self.vlm.visual(patch_matrix, grid)

        text_tokens = self.vlm.get_input_embeddings()(lang_tokens)
        hidden_size = text_tokens.shape[-1]
        flat_tokens = text_tokens.reshape(-1, hidden_size)
        image_positions = lang_tokens.reshape(-1).eq(self.vlm.config.image_token_id)
        destination_rows = image_positions.nonzero(as_tuple=False).squeeze(1)
        if destination_rows.numel() != visual_tokens.shape[0]:
            raise ValueError(
                "Qwen image placeholders do not match visual tokens: "
                f"{destination_rows.numel()} != {visual_tokens.shape[0]}"
            )
        flat_tokens = flat_tokens.index_copy(0, destination_rows, visual_tokens)
        embs = flat_tokens.reshape_as(text_tokens)

        pad_masks = lang_masks.bool()
        att_masks = torch.zeros_like(pad_masks)
        return embs, pad_masks, att_masks

    def get_cosmos_features(self, images):
        """Encode any leading batch dimensions with the frozen visual tokenizer."""
        leading_shape = tuple(images.shape[:-3])
        frame_batch = images.flatten(0, images.ndim - 4)
        normalized = F.interpolate(
            frame_batch,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).mul(2).sub(1)
        encoded = self.cosmos.encode(normalized)
        return encoded.unflatten(0, leading_shape)

    def embed_middle(self, images, img_masks):
        """Convert temporal camera observations into one VLM token block."""
        latents = self.get_cosmos_features(images)
        batch, views, frames, channels, height, width = latents.shape

        projected = latents.reshape(batch * views * frames, channels, height, width)
        projected = self.downsample_conv(self.cosmos_in_proj(projected))
        token_height, token_width = projected.shape[-2:]
        projected = projected.reshape(
            batch,
            views,
            frames,
            projected.shape[1],
            token_height,
            token_width,
        )
        self.cosmos_feat_shape = projected.shape

        embs = projected.permute(0, 1, 2, 4, 5, 3).reshape(batch, -1, projected.shape[3])
        source_is_valid = img_masks.to(dtype=torch.bool, device=embs.device)
        pad_masks = source_is_valid[:, :, None, None, None].expand(
            batch, views, frames, token_height, token_width
        ).reshape(batch, -1)

        att_masks = torch.zeros_like(pad_masks)
        att_masks[:, 0] = True
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep for the VLM suffix."""
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_slots(self, batch_size, device):
        """Expand learnable bottleneck tokens to batch size with dropout and masks."""
        K = self.bottleneck_tokens.shape[1]
        slot_embs = self.bottleneck_tokens.expand(batch_size, K, -1)
        if self.training:
            slot_embs = self.slot_dropout(slot_embs)

        pad_masks = torch.ones(batch_size, K, dtype=torch.bool, device=device)
        att_masks = torch.zeros(batch_size, K, dtype=torch.bool, device=device)
        att_masks[:, 0] = True
        return slot_embs, pad_masks, att_masks

    def apply_slot_bottleneck(self, att_2d_masks, L_prefix, L_middle, L_slots, current_step):
        """Conditionally block suffix from attending to prefix/middle.

        Uses curriculum strategy: probability of hard mask increases linearly
        from 0 to 1 over slot_bottleneck_warmup_steps.
        """
        mode = self.config.slot_bottleneck_mode

        if mode == "soft":
            return att_2d_masks

        if mode == "hard":
            apply_hard = True
        elif mode == "curriculum":
            warmup = max(1, self.config.slot_bottleneck_warmup_steps)
            p = min(1.0, current_step / warmup)
            apply_hard = (torch.rand(1).item() < p) if self.training else True
        else:
            return att_2d_masks

        if apply_hard:
            L_context = L_prefix + L_middle
            suffix_start = L_context + L_slots
            att_2d_masks = att_2d_masks.clone()
            att_2d_masks[:, suffix_start:, :L_context] = False

        return att_2d_masks

    def compute_slot_regularization(self, slot_out):
        """Orthogonality regularization to prevent slot collapse."""
        s = F.normalize(slot_out.float(), dim=-1)
        gram = torch.bmm(s, s.transpose(1, 2))
        K = slot_out.shape[1]
        eye = torch.eye(K, device=slot_out.device).unsqueeze(0)
        loss = ((gram - eye) ** 2).sum(dim=(1, 2)).mean() / (K * K)
        return loss

    def gated_gen_head_forward(self, middle_out, slot_out):
        """Gen head with slot-conditioned gating for semantic control."""
        slot_pooled = slot_out.mean(dim=1)
        gate = self.slot_gate(slot_pooled.float())
        gate = gate.unsqueeze(1)

        gen_features = self.gen_head(middle_out)
        gated_features = gen_features * gate
        return self.decode_cosmos(gated_features)

    def encode_task_text(self, task_input_ids, task_attention_mask):
        """Extract z_text from pure text task instructions via embedding + mean pool."""
        token_embs = self.vlm.get_input_embeddings()(task_input_ids)
        mask = task_attention_mask.unsqueeze(-1).float()
        pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        z_text = self.task_proj_text(pooled.float())
        return F.normalize(z_text, dim=-1)

    def encode_task_traj(self, traj_feat):
        """Extract z_traj from a trajectory feature (slot/suffix/middle) via mean pool."""
        pooled = traj_feat.mean(dim=1)
        z_traj = self.task_proj_traj(pooled.float())
        return F.normalize(z_traj, dim=-1)

    def compute_task_loss(self, z_traj, z_text, task_input_ids):
        """Symmetric InfoNCE loss (CLIP-style) for task alignment.

        Masks out false negatives: off-diagonal pairs that share the same
        task text are set to -inf so they don't act as negatives.
        """
        B = z_traj.shape[0]
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * z_traj @ z_text.T
        labels = torch.arange(B, device=logits.device)

        task_sig = task_input_ids.sum(dim=-1)
        same_task = task_sig.unsqueeze(0) == task_sig.unsqueeze(1)
        false_neg_mask = same_task & ~torch.eye(B, dtype=torch.bool, device=logits.device)

        logits_masked = logits.clone()
        logits_masked[false_neg_mask] = float('-inf')

        loss_t2t = F.cross_entropy(logits_masked, labels)
        loss_t2i = F.cross_entropy(logits_masked.T, labels)
        loss = 0.5 * (loss_t2t + loss_t2i)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = same_task[torch.arange(B, device=logits.device), preds].float().mean()
        return loss, acc

    def get_position_ids(self, lang_tokens, image_grid_thw, pad_masks):
        """Ask Qwen's public M-RoPE helper to index the complete token stream."""
        language_length = lang_tokens.shape[1]
        pad_token_id = self.vlm.config.text_config.pad_token_id
        neutral_token_id = 0 if pad_token_id is None else pad_token_id
        full_tokens = lang_tokens.new_full(pad_masks.shape, neutral_token_id)
        full_tokens.narrow(1, 0, language_length).copy_(lang_tokens)

        grid_thw = None
        if image_grid_thw is not None:
            grid_thw = image_grid_thw.reshape(-1, 3)
        position_ids, rope_deltas = self.vlm.model.get_rope_index(
            full_tokens,
            grid_thw,
            attention_mask=pad_masks.to(dtype=lang_tokens.dtype),
        )
        return position_ids, rope_deltas

    def decode_cosmos(self, features):
        """Decode gen_head output back to Cosmos feature space."""
        batch, views, frames, _, height, width = self.cosmos_feat_shape
        hidden = features.shape[-1]
        spatial = features.reshape(batch, views, frames, height, width, hidden)
        spatial = spatial.mean(dim=2).permute(0, 1, 4, 2, 3)
        spatial = spatial.reshape(batch * views, hidden, height, width)

        upsampled = self.upsample_conv(spatial)
        out_height, out_width = upsampled.shape[-2:]
        channels = upsampled.shape[1]
        tokens = upsampled.permute(0, 2, 3, 1).reshape(batch * views, -1, channels)
        tokens = self.cosmos_out_layer_norm(tokens)
        tokens = self.cosmos_out_proj(tokens)
        return tokens.reshape(
            batch, views, out_height, out_width, tokens.shape[-1]
        ).permute(0, 1, 4, 2, 3)

    def encode(
        self, images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks,
        state, noisy_actions, timestep,
        current_step=0,
    ) -> dict[str, Tensor]:
        """Run backbone encoding and return all intermediate representations.

        This is the primary interface for downstream users who load a pretrained
        backbone and want access to the compressed representations (especially
        slot_out) without re-implementing the embedding + attention logic.

        Returns a dict with keys:
            - "middle_out":  [B, L_middle, vlm_dim]  — spatial image features after VLM
            - "suffix_out":  [B, chunk_size, vlm_dim] — action token features after VLM
            - "slot_out":    [B, K, vlm_dim] or None  — reasoning slot features after VLM
            - "slot_pooled": [B, vlm_dim] or None     — mean-pooled slot features
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
            images[:, :, :2], img_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, noisy_actions, timestep
        )

        B = prefix_embs.shape[0]
        device = prefix_embs.device
        use_slots = self.config.enable_bottleneck_tokens

        if use_slots:
            slot_embs, slot_pad_masks, slot_att_masks = self.embed_slots(B, device)
            prefix_embs, middle_embs, slot_embs, suffix_embs = self._cast_embs_for_vlm(
                prefix_embs, middle_embs, slot_embs, suffix_embs
            )
            all_embs = torch.cat([prefix_embs, middle_embs, slot_embs, suffix_embs], dim=1)
            pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, slot_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, middle_att_masks, slot_att_masks, suffix_att_masks], dim=1)
        else:
            prefix_embs, middle_embs, suffix_embs = self._cast_embs_for_vlm(
                prefix_embs, middle_embs, suffix_embs
            )
            all_embs = torch.cat([prefix_embs, middle_embs, suffix_embs], dim=1)
            pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        if use_slots:
            L_prefix = prefix_embs.shape[1]
            L_middle = middle_embs.shape[1]
            L_slots = slot_embs.shape[1]
            att_2d_masks = self.apply_slot_bottleneck(
                att_2d_masks, L_prefix, L_middle, L_slots, current_step
            )

        position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, pad_masks)
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        L_prefix = prefix_embs.shape[1]
        L_middle = middle_embs.shape[1]

        def forward_func(all_embs, att_2d_masks_4d, position_ids):
            vlm_output = self.vlm.language_model(
                inputs_embeds=all_embs,
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
            ).last_hidden_state
            return vlm_output

        vlm_output = self._apply_checkpoint(forward_func, all_embs, att_2d_masks_4d, position_ids)

        middle_out = vlm_output[:, L_prefix:L_prefix + L_middle]
        suffix_out = vlm_output[:, -self.config.chunk_size:]

        result = {
            "middle_out": middle_out,
            "suffix_out": suffix_out,
            "slot_out": None,
            "slot_pooled": None,
        }

        if use_slots:
            slot_out = vlm_output[:, L_prefix + L_middle:L_prefix + L_middle + L_slots]
            result["slot_out"] = slot_out
            result["slot_pooled"] = slot_out.mean(dim=1)

        return result

    def forward(
        self, images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks,
        state, actions, noise=None, time=None,
        current_step=0, task_input_ids=None, task_attention_mask=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        encoded = self.encode(
            images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks,
            state, x_t, time, current_step=current_step,
        )

        middle_out = encoded["middle_out"]
        suffix_out = encoded["suffix_out"]
        slot_out = encoded["slot_out"]

        device = middle_out.device
        use_slots = self.config.enable_bottleneck_tokens

        # --- Gen head (slot-gated if bottleneck tokens are active) ---
        if use_slots:
            def gated_gen_func(middle_out, slot_out):
                return self.gated_gen_head_forward(middle_out, slot_out)

            pred_cosmos_features = self._apply_checkpoint(
                gated_gen_func, middle_out.to(dtype=torch.float32), slot_out
            )
            loss_slot_reg = self.compute_slot_regularization(slot_out)
        else:
            def gen_head_func(middle_out):
                gen_features = self.gen_head(middle_out)
                return self.decode_cosmos(gen_features)

            pred_cosmos_features = self._apply_checkpoint(gen_head_func, middle_out.to(dtype=torch.float32))
            loss_slot_reg = torch.tensor(0.0, device=device)

        future_embs = self.get_cosmos_features(images[:, :, 2])
        loss_gen = F.mse_loss(pred_cosmos_features[img_masks], future_embs.to(dtype=torch.float32)[img_masks])

        # --- Action head ---
        suffix_out = suffix_out.to(dtype=torch.float32)

        def act_head_func(suffix_out):
            return self.act_head(suffix_out)

        v_t = self._apply_checkpoint(act_head_func, suffix_out)
        loss_action = F.mse_loss(u_t, v_t, reduction="none")

        # --- InfoNCE task contrastive loss ---
        # 轨迹侧表征 z_traj 的来源由 task_traj_source 决定：
        #   "slot"   -> slot_out（需 bottleneck tokens 开启）
        #   "suffix" -> suffix_out（action expert 输出，与 slot 解耦）
        #   "middle" -> middle_out（视觉中间帧特征，与 slot 解耦）
        traj_source = self.config.task_traj_source
        if traj_source == "slot":
            traj_feat = slot_out
        elif traj_source == "suffix":
            traj_feat = suffix_out
        else:  # "middle"
            traj_feat = middle_out

        if (
            self.config.enable_task_contrastive
            and task_input_ids is not None
            and traj_feat is not None
        ):
            z_traj = self.encode_task_traj(traj_feat)
            z_text = self.encode_task_text(task_input_ids, task_attention_mask)
            loss_task, contrastive_acc = self.compute_task_loss(z_traj, z_text, task_input_ids)
        else:
            loss_task = torch.tensor(0.0, device=device)
            contrastive_acc = torch.tensor(0.0, device=device)

        return loss_action, loss_gen, loss_task, loss_slot_reg, contrastive_acc

    @torch.no_grad()
    def sample_actions(
        self, images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks, state, noise=None, num_steps=None, decode_image=False
    ) -> Tensor:
        """Generate an action trajectory with Euler integration of the flow field."""
        steps = self.config.num_inference_steps if num_steps is None else num_steps
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")

        batch_size = state.shape[0]
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                state.device,
            )

        context = self._prefill_action_context(
            images=images,
            img_masks=img_masks,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        actions = self._integrate_action_flow(state, noise, context, steps)
        reconstruction = self._decode_context_image(context) if decode_image else None
        return actions, reconstruction

    def _prefill_action_context(
        self,
        *,
        images,
        img_masks,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
    ) -> _ActionContext:
        """Cache observation tokens once before the iterative action solve."""
        prefix, prefix_valid, prefix_blocks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        (prefix,) = self._cast_embs_for_vlm(prefix)
        prefix_positions, _ = self.get_position_ids(lang_tokens, image_grid_thw, prefix_valid)
        self.vlm.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        prefix_output = self.vlm.language_model(
            inputs_embeds=prefix,
            attention_mask=self._prepare_attention_masks_4d(
                make_att_2d_masks(prefix_valid, prefix_blocks)
            ),
            position_ids=prefix_positions,
            past_key_values=None,
            use_cache=True,
        )

        middle, middle_valid, middle_blocks = self.embed_middle(images[:, :, :2], img_masks)
        (middle,) = self._cast_embs_for_vlm(middle)
        middle_length = middle.shape[1]

        slot_hidden = None
        if self.config.enable_bottleneck_tokens:
            slots, slot_valid, slot_blocks = self.embed_slots(images.shape[0], images.device)
            (slots,) = self._cast_embs_for_vlm(slots)
            continuation = torch.cat((middle, slots), dim=1)
            continuation_valid = torch.cat((middle_valid, slot_valid), dim=1)
            continuation_blocks = torch.cat((middle_blocks, slot_blocks), dim=1)
        else:
            continuation = middle
            continuation_valid = middle_valid
            continuation_blocks = middle_blocks

        query_count = continuation.shape[1]
        prefix_visibility = prefix_valid.unsqueeze(1).expand(-1, query_count, -1)
        local_visibility = make_att_2d_masks(continuation_valid, continuation_blocks)
        visibility = torch.cat((prefix_visibility, local_visibility), dim=-1)

        prefix_tail = prefix_positions.amax(dim=-1, keepdim=True)
        offsets = torch.arange(
            1,
            query_count + 1,
            dtype=prefix_tail.dtype,
            device=prefix_tail.device,
        ).view(1, 1, -1)
        continuation_positions = prefix_tail + offsets

        continuation_output = self.vlm.language_model(
            inputs_embeds=continuation,
            attention_mask=self._prepare_attention_masks_4d(visibility),
            position_ids=continuation_positions,
            past_key_values=prefix_output.past_key_values,
            use_cache=True,
        )
        middle_hidden = continuation_output.last_hidden_state[:, :middle_length]
        key_values = continuation_output.past_key_values

        if self.config.enable_bottleneck_tokens:
            slot_count = self.config.bottleneck_num_tokens
            slot_hidden = continuation_output.last_hidden_state[:, middle_length:]
            self._retain_cache_tail(key_values, slot_count)
            valid_tokens = slot_valid
            last_position = continuation_positions[:, :, -slot_count:].amax(
                dim=-1, keepdim=True
            )
        else:
            valid_tokens = torch.cat((prefix_valid, middle_valid), dim=1)
            last_position = continuation_positions.amax(dim=-1, keepdim=True)

        return _ActionContext(
            key_values=key_values,
            valid_tokens=valid_tokens,
            last_position=last_position,
            middle_hidden=middle_hidden,
            slot_hidden=slot_hidden,
        )

    @staticmethod
    def _retain_cache_tail(key_values, token_count: int) -> None:
        """Discard context K/V entries preceding the reasoning bottleneck."""
        if hasattr(key_values, "key_cache"):
            for layer_index in range(len(key_values.key_cache)):
                key_values.key_cache[layer_index] = key_values.key_cache[layer_index][
                    :, :, -token_count:
                ]
                key_values.value_cache[layer_index] = key_values.value_cache[layer_index][
                    :, :, -token_count:
                ]
        else:
            for layer in key_values.layers:
                layer.keys = layer.keys[:, :, -token_count:]
                layer.values = layer.values[:, :, -token_count:]
        if hasattr(key_values, "_seen_tokens"):
            key_values._seen_tokens = token_count

    def _integrate_action_flow(
        self,
        state: Tensor,
        initial_noise: Tensor,
        context: _ActionContext,
        steps: int,
    ) -> Tensor:
        """Solve dx/dt=v(x,t) from noise at t=1 to an action at t=0."""
        step = torch.tensor(
            -1.0 / steps,
            dtype=torch.float32,
            device=state.device,
        )
        actions = initial_noise
        time_value = torch.tensor(1.0, dtype=torch.float32, device=state.device)
        while time_value >= -step / 2:
            velocity = self.denoise_step(
                state,
                context.valid_tokens,
                copy.deepcopy(context.key_values),
                context.last_position,
                actions.to(dtype=state.dtype),
                time_value.expand(state.shape[0]).to(dtype=state.dtype),
            )
            actions = actions + step * velocity
            time_value += step
        return actions

    def _decode_context_image(self, context: _ActionContext):
        hidden = context.middle_hidden.float()
        if context.slot_hidden is None:
            latent_prediction = self.decode_cosmos(self.gen_head(hidden))
        else:
            latent_prediction = self.gated_gen_head_forward(hidden, context.slot_hidden)
        return self.cosmos.decode(latent_prediction.squeeze(0).squeeze(0))

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        max_prefix_position_ids, 
        x_t,
        timestep,
    ):
        """Evaluate the action flow field for one solver timestep."""
        suffix, suffix_valid, suffix_blocks = self.embed_suffix(state, x_t, timestep)
        (suffix,) = self._cast_embs_for_vlm(suffix)
        suffix_length = suffix.shape[1]

        memory_visibility = prefix_pad_masks.unsqueeze(1).expand(
            -1, suffix_length, -1
        )
        suffix_visibility = make_att_2d_masks(suffix_valid, suffix_blocks)
        visibility = torch.cat((memory_visibility, suffix_visibility), dim=-1)

        offsets = torch.arange(
            1,
            suffix_length + 1,
            dtype=max_prefix_position_ids.dtype,
            device=max_prefix_position_ids.device,
        ).view(1, 1, -1)
        position_ids = max_prefix_position_ids + offsets

        output = self.vlm.language_model(
            inputs_embeds=suffix,
            attention_mask=self._prepare_attention_masks_4d(visibility),
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
        ).last_hidden_state
        action_hidden = output[:, -self.config.chunk_size:].float()
        return self.act_head(action_hidden)


class PelicanVLA05Policy(PreTrainedPolicy):
    """VLM-centric architecture."""

    config_class = PelicanVLA05Config
    name = "pelican_vla05"

    def __init__(
        self,
        config: PelicanVLA05Config,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = PelicanVLA05(config)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self._current_step = 0
        self.reset()

    def __str__(self) -> str:
        lines = []

        lines.append("=" * 60)
        lines.append(f"Policy: {self.__class__.__name__}")
        lines.append("")

        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        full_sd = self.state_dict()
        bb_sd, _, act_sd, gen_sd, task_sd = self._split_state_dict(full_sd)
        num_backbone = sum(v.numel() for v in bb_sd.values())
        num_act_head = sum(v.numel() for v in act_sd.values())
        num_gen_head = sum(v.numel() for v in gen_sd.values())
        num_task_head = sum(v.numel() for v in task_sd.values())

        lines.append("Parameter statistics:")
        lines.append(f"  - Total params        : {num_total_params} ({format_big_number(num_total_params)})")
        lines.append(f"  - Trainable params    : {num_trainable_params} ({format_big_number(num_trainable_params)})")
        lines.append(f"  - Backbone params     : {num_backbone} ({format_big_number(num_backbone)})")
        lines.append(f"  - Action head params  : {num_act_head} ({format_big_number(num_act_head)})")
        lines.append(f"  - Gen head params     : {num_gen_head} ({format_big_number(num_gen_head)})")

        if self.config.freeze_backbone:
            lines.append(f"  * Backbone FROZEN — only heads are trainable")
        elif self.config.freeze_vision_encoder:
            lines.append(f"  * Vision encoder frozen")

        if self.config.enable_bottleneck_tokens:
            num_slot_params = (
                self.model.bottleneck_tokens.numel()
                + sum(p.numel() for p in self.model.slot_gate.parameters())
            )
            lines.append(f"  - Bottleneck tokens   : K={self.config.bottleneck_num_tokens}, "
                         f"mode={self.config.slot_bottleneck_mode}, params={format_big_number(num_slot_params)}")
        if self.config.enable_task_contrastive:
            lines.append(f"  - Task head params    : {num_task_head} ({format_big_number(num_task_head)}), "
                         f"traj_source={self.config.task_traj_source}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        cosmos_tensors = (
            *self.model.cosmos.parameters(),
            *self.model.cosmos.buffers(),
        )
        cosmos_device = (
            cosmos_tensors[0].device
            if cosmos_tensors
            else torch.device(self.config.device or "cpu")
        )
        cosmos_dtype = (
            torch.bfloat16 if cosmos_device.type == "cuda" else torch.float32
        )
        # ``super().to`` already moved the nested ScriptModules. Match the
        # training implementation's dtype-only conversion on CUDA so the JIT
        # execution plan remains numerically identical.
        self.model.cosmos.to(cosmos_dtype)
        self.model.cosmos._runtime_device = cosmos_device
        self.model.cosmos._runtime_dtype = cosmos_dtype
        self.model.gen_head.to(torch.float32)
        self.model.act_head.to(torch.float32)
        if self.config.enable_bottleneck_tokens:
            self.model.slot_gate.to(torch.float32)
        if self.config.enable_task_contrastive:
            self.model.task_proj_traj.to(torch.float32)
            self.model.task_proj_text.to(torch.float32)
        return self

    # ------------------------------------------------------------------
    # Split save / load: backbone vs heads
    # ------------------------------------------------------------------

    @staticmethod
    def _split_state_dict(state_dict: dict[str, Tensor]) -> tuple[dict, dict, dict, dict, dict]:
        """Partition a full state_dict into backbone, action_head, gen_head, task_head.

        Returns:
            (backbone_sd, heads_sd, action_head_sd, gen_head_sd, task_head_sd)
            where heads_sd = action_head_sd | gen_head_sd | task_head_sd.
        """
        backbone_sd, action_head_sd, gen_head_sd, task_head_sd = {}, {}, {}, {}
        for key, val in state_dict.items():
            if any(key.startswith(p) for p in SKIP_PREFIXES):
                continue
            if any(key.startswith(p) for p in BACKBONE_PREFIXES):
                backbone_sd[key] = val
            elif any(key.startswith(p) for p in ACTION_HEAD_PREFIXES):
                action_head_sd[key] = val
            elif any(key.startswith(p) for p in GEN_HEAD_PREFIXES):
                gen_head_sd[key] = val
            elif any(key.startswith(p) for p in TASK_HEAD_PREFIXES):
                task_head_sd[key] = val
            else:
                backbone_sd[key] = val
        heads_sd = {**action_head_sd, **gen_head_sd, **task_head_sd}
        return backbone_sd, heads_sd, action_head_sd, gen_head_sd, task_head_sd

    @staticmethod
    def _dedup_shared_tensors(sd: dict) -> dict:
        """Clone tensors that share underlying storage so safetensors save_file won't error."""
        seen_data_ptrs: set[int] = set()
        out = {}
        for k, v in sd.items():
            ptr = v.data_ptr()
            if ptr in seen_data_ptrs:
                v = v.clone()
            else:
                seen_data_ptrs.add(ptr)
            out[k] = v
        return out

    def _save_pretrained(self, save_directory: Path) -> None:
        """Override: save full model + split backbone / action_head / gen_head / task_head."""
        super()._save_pretrained(save_directory)

        save_directory = Path(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        full_sd = model_to_save.state_dict()
        backbone_sd, heads_sd, action_head_sd, gen_head_sd, task_head_sd = self._split_state_dict(full_sd)

        backbone_sd = self._dedup_shared_tensors(backbone_sd)
        save_file(backbone_sd, str(save_directory / BACKBONE_FILE))
        save_file(heads_sd, str(save_directory / HEADS_FILE))
        save_file(action_head_sd, str(save_directory / ACTION_HEAD_FILE))
        save_file(gen_head_sd, str(save_directory / GEN_HEAD_FILE))
        if task_head_sd:
            save_file(task_head_sd, str(save_directory / TASK_HEAD_FILE))

        n_bb = sum(v.numel() for v in backbone_sd.values())
        n_act = sum(v.numel() for v in action_head_sd.values())
        n_gen = sum(v.numel() for v in gen_head_sd.values())
        n_task = sum(v.numel() for v in task_head_sd.values())
        logging.info(
            f"Split save → backbone: {len(backbone_sd)} tensors ({n_bb / 1e6:.1f}M params), "
            f"action_head: {len(action_head_sd)} tensors ({n_act / 1e6:.1f}M params), "
            f"gen_head: {len(gen_head_sd)} tensors ({n_gen / 1e6:.1f}M params), "
            f"task_head: {len(task_head_sd)} tensors ({n_task / 1e6:.1f}M params)"
        )

    def save_backbone(self, path: str | Path) -> None:
        """Save only backbone weights to a single file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full_sd = self.state_dict()
        backbone_sd = self._split_state_dict(full_sd)[0]
        save_file(backbone_sd, str(path))
        n = sum(v.numel() for v in backbone_sd.values())
        logging.info(f"Saved backbone: {len(backbone_sd)} tensors ({n / 1e6:.1f}M params) → {path}")

    def save_heads(self, path: str | Path) -> None:
        """Save all head weights (action + gen + task) to a single file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full_sd = self.state_dict()
        heads_sd = self._split_state_dict(full_sd)[1]
        save_file(heads_sd, str(path))
        n = sum(v.numel() for v in heads_sd.values())
        logging.info(f"Saved heads: {len(heads_sd)} tensors ({n / 1e6:.1f}M params) → {path}")

    def save_action_head(self, path: str | Path) -> None:
        """Save only action head weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full_sd = self.state_dict()
        action_head_sd = self._split_state_dict(full_sd)[2]
        save_file(action_head_sd, str(path))
        n = sum(v.numel() for v in action_head_sd.values())
        logging.info(f"Saved action_head: {len(action_head_sd)} tensors ({n / 1e6:.1f}M params) → {path}")

    def save_gen_head(self, path: str | Path) -> None:
        """Save only gen (image) head weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full_sd = self.state_dict()
        gen_head_sd = self._split_state_dict(full_sd)[3]
        save_file(gen_head_sd, str(path))
        n = sum(v.numel() for v in gen_head_sd.values())
        logging.info(f"Saved gen_head: {len(gen_head_sd)} tensors ({n / 1e6:.1f}M params) → {path}")

    def _load_partial(self, path: str | Path, expected_prefixes: tuple[str, ...], label: str) -> tuple[list, list]:
        """Generic helper: load a safetensors file into matching keys only."""
        sd = load_file(str(path))
        missing, unexpected = [], []
        own_sd = self.state_dict()
        for key, val in sd.items():
            if key in own_sd:
                own_sd[key] = val
            else:
                unexpected.append(key)
        for key in own_sd:
            if any(key.startswith(p) for p in expected_prefixes) and key not in sd:
                missing.append(key)
        self.load_state_dict(own_sd, strict=False)
        if missing or unexpected:
            logging.warning(f"load_{label}: missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:
                logging.warning(f"  missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                logging.warning(f"  unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        else:
            logging.info(f"Loaded {label} from {path} ({len(sd)} tensors)")
        return missing, unexpected

    def load_backbone(self, path: str | Path) -> tuple[list, list]:
        """Load backbone weights from a file. Heads remain unchanged."""
        return self._load_partial(path, BACKBONE_PREFIXES, "backbone")

    def load_heads(self, path: str | Path) -> tuple[list, list]:
        """Load all head weights (action + gen + task). Backbone remains unchanged."""
        return self._load_partial(path, HEAD_PREFIXES, "heads")

    def load_action_head(self, path: str | Path) -> tuple[list, list]:
        """Load action head weights only."""
        return self._load_partial(path, ACTION_HEAD_PREFIXES, "action_head")

    def load_gen_head(self, path: str | Path) -> tuple[list, list]:
        """Load gen (image) head weights only."""
        return self._load_partial(path, GEN_HEAD_PREFIXES, "gen_head")

    @classmethod
    def from_pretrained_backbone(
        cls,
        backbone_path: str | Path,
        config: PelicanVLA05Config | None = None,
        **kwargs,
    ):
        """Create a fresh policy, load ONLY backbone weights (heads randomly initialized).

        Useful for downstream adaptation: load universal backbone, train new heads.
        """
        if config is None:
            config_dir = Path(backbone_path).parent
            config = PreTrainedConfig.from_pretrained(config_dir)

        instance = cls(config, **kwargs)
        instance.load_backbone(backbone_path)
        logging.info("Heads are randomly initialized — ready for downstream fine-tuning.")
        instance.eval()
        return instance

    @classmethod
    def from_pretrained_split(
        cls,
        backbone_path: str | Path,
        action_head_path: str | Path | None = None,
        gen_head_path: str | Path | None = None,
        heads_path: str | Path | None = None,
        config: PelicanVLA05Config | None = None,
        **kwargs,
    ):
        """Create a policy from separate backbone and head files.

        Load backbone first, then optionally load action head and/or gen head.
        Any head not loaded stays randomly initialized.
        """
        if config is None:
            config_dir = Path(backbone_path).parent
            config = PreTrainedConfig.from_pretrained(config_dir)

        instance = cls(config, **kwargs)
        instance.load_backbone(backbone_path)

        if heads_path is not None:
            instance.load_heads(heads_path)
        if action_head_path is not None:
            instance.load_action_head(action_head_path)
        if gen_head_path is not None:
            instance.load_gen_head(gen_head_path)

        loaded = ["backbone"]
        if heads_path:
            loaded.append("heads")
        if action_head_path:
            loaded.append("action_head")
        if gen_head_path:
            loaded.append("gen_head")
        logging.info(f"Loaded split model: {', '.join(loaded)}")
        instance.eval()
        return instance

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
        """Override to support backbone_pretrained_path in config.

        If config.backbone_pretrained_path is set, load only backbone (heads
        randomly initialized). Otherwise fall back to standard full-model loading.
        """
        if config is not None and getattr(config, "backbone_pretrained_path", None):
            bb_path = config.backbone_pretrained_path
            if not os.path.isabs(bb_path):
                bb_path = os.path.join(pretrained_name_or_path, bb_path)
            logging.info(f"Loading backbone only from: {bb_path}")
            return cls.from_pretrained_backbone(bb_path, config=config, **kwargs)

        model_dir = str(pretrained_name_or_path)
        if os.path.isdir(model_dir):
            full_model_file = os.path.join(model_dir, "model.safetensors")
            backbone_file = os.path.join(model_dir, BACKBONE_FILE)
            heads_file = os.path.join(model_dir, HEADS_FILE)
            if not os.path.exists(full_model_file) and os.path.exists(backbone_file):
                logging.info("model.safetensors not found; loading from split files")
                return cls.from_pretrained_split(
                    backbone_path=backbone_file,
                    heads_path=heads_file if os.path.exists(heads_file) else None,
                    config=config,
                    **kwargs,
                )

        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset the policy's action queues - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model."""
        images = []
        img_masks = []

        for img_idx in range(3):
            img = batch[f"{OBS_IMAGES}.image{img_idx}"]
            mask = batch[f"{OBS_IMAGES}.image{img_idx}_mask"]

            images.append(img)
            img_masks.append(mask)

        images = torch.stack(images, dim=1)  # B, N_view, T, C, H, W
        img_masks = torch.stack(img_masks, dim=1)

        return images, img_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if len(self._action_queue) == 0:
            actions, _ = self.predict_action_chunk(batch)
            actions = actions[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], decode_image=False) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        state = self.prepare_state(batch)

        images, img_masks = self._preprocess_images(batch)

        actions, recon_images = self.model.sample_actions(images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks, state, decode_image=decode_image)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions, recon_images

    def forward(self, batch: dict[str, Tensor], current_step: int = 0) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""

        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]

        images, img_masks = self._preprocess_images(batch)

        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        task_ids_key = f"{OBS_PREFIX}task.input_ids"
        task_input_ids = batch.get(task_ids_key)
        task_attention_mask = batch.get(f"{OBS_PREFIX}task.attention_mask")

        step = current_step if current_step > 0 else self._current_step

        losses_action, loss_gen, loss_task, loss_slot_reg, contrastive_acc = self.model.forward(
            images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks,
            state, actions,
            current_step=step,
            task_input_ids=task_input_ids,
            task_attention_mask=task_attention_mask,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses_action = losses_action[:, :, :original_action_dim]
        loss_action = losses_action.mean()

        loss = (
            loss_action
            + self.config.lambda_gen * loss_gen
            + self.config.lambda_task * loss_task
            + self.config.lambda_slot_reg * loss_slot_reg
        )

        loss_dict = {
            "loss": loss.item(),
            "loss_action": loss_action.item(), 
            "loss_gen": loss_gen.item(), 
        }

        if self.config.enable_bottleneck_tokens:
            loss_dict["loss_slot_reg"] = loss_slot_reg.item()

        if self.config.enable_task_contrastive:
            loss_dict["loss_task"] = loss_task.item()
            loss_dict["contrastive_acc"] = contrastive_acc.item()

        losses_action = losses_action.mean(dim=[0, 1]).detach().cpu().numpy().tolist()
        loss_dict.update({
            f"loss_action_dim{i}": losses_action[i] for i in range(original_action_dim)
        })

        return loss, loss_dict


# Compatibility aliases preserve existing Python imports without changing any
# module attribute names used by saved state dictionaries.
BaseVLA = PelicanVLA05
BaseVLAPolicy = PelicanVLA05Policy
