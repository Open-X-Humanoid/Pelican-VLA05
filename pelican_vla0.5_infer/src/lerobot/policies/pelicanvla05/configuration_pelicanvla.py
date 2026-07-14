#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright 2026 The BaseVLA Authors.
#
# Derived from the openpi / LeRobot pi0 policy configuration (Apache-2.0), with
# PelicanVLA05-specific fields added. See PROVENANCE.md for details.
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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("basevla")
@PreTrainedConfig.register_subclass("pelican_vla05")
@dataclass
class PelicanVLA05Config(PreTrainedConfig):
    qwen3_vl_variant: str = "qwen3_vl_4b"
    # Local dir or HF hub id of the Qwen3-VL weights used to initialise the
    # backbone. When None, resolved from the QWEN3_VL_PATH env var (see modeling).
    qwen3_vl_path: str | None = None
    # Directory containing NVIDIA Cosmos encoder.jit and decoder.jit. When
    # unset, the pinned public model repository below is downloaded by
    # huggingface_hub.
    cosmos_tokenizer_path: str | None = None
    cosmos_tokenizer_repo_id: str = "nvidia/Cosmos-0.1-Tokenizer-CI8x8"
    cosmos_tokenizer_revision: str = "04792f8b318a26d9e54319887124a8848c3cd3ca"
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"

    # Deprecated fields kept as no-ops so older checkpoints/configs still load.
    action_expert_variant: str = "qwen3_600m"
    train_expert_only: bool = False
    train_vlm_only: bool = False

    # MLP head dimensions (replace the former gen_expert / act_expert Transformers)
    gen_head_hidden_dim: int = 1024
    act_head_hidden_dim: int = 1024

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10  # Number of denoising steps during inference
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Optimizer settings: see openpi `AdamW``
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48  # see openpi `__post_init__`

    freeze_vision_encoder: bool = False
    freeze_backbone: bool = False

    # Split loading: backbone from one checkpoint, heads from another (or random init)
    backbone_pretrained_path: str | None = None
    action_head_pretrained_path: str | None = None
    gen_head_pretrained_path: str | None = None

    scale_factor: int = 8  # param for pixel shuffle / unshuffle
    lambda_gen: float = 0.01

    # ---- Bottleneck Tokens (Phase 1) ----
    enable_bottleneck_tokens: bool = True
    bottleneck_num_tokens: int = 32
    reasoning_dropout: float = 0.1
    slot_bottleneck_mode: str = "curriculum"  # "curriculum" / "hard" / "soft"
    slot_bottleneck_warmup_steps: int = 20000
    lambda_slot_reg: float = 0.01

    # ---- InfoNCE Task Contrastive (Phase 2) ----
    enable_task_contrastive: bool = False
    task_embed_dim: int = 256
    lambda_task: float = 0.1
    # 对比学习轨迹侧表征 z_traj 的来源特征：
    #   "slot"   -> 用 bottleneck tokens 输出（需 enable_bottleneck_tokens=true）
    #   "suffix" -> 用 action expert 的 suffix 输出（与 slot 解耦）
    #   "middle" -> 用视觉中间帧特征 middle_out（与 slot 解耦）
    task_traj_source: str = "slot"

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.task_traj_source not in ["slot", "suffix", "middle"]:
            raise ValueError(
                f"Invalid task_traj_source: {self.task_traj_source} "
                f"(expected 'slot', 'suffix' or 'middle')"
            )

        if (
            self.enable_task_contrastive
            and self.task_traj_source == "slot"
            and not self.enable_bottleneck_tokens
        ):
            raise ValueError(
                "task_traj_source='slot' requires enable_bottleneck_tokens=true. "
                "Set task_traj_source to 'suffix' or 'middle' to decouple contrastive "
                "loss from bottleneck tokens."
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_delta_indices(self) -> list | None: 
        return [-15, 0, 15]


# Backward-compatible import name for existing integrations. The checkpoint
# config type remains ``pelican_vla05`` and state-dict keys are unaffected.
BaseVLAConfig = PelicanVLA05Config