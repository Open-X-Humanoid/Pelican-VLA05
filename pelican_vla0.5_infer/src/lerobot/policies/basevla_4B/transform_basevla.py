# Copyright 2026 The BaseVLA Authors.
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
"""Turn image + task-text samples into the token / pixel tensors the Qwen3-VL
backbone expects.

Clean reimplementation for the inference release. The path to the Qwen3-VL
processor is resolved from (in order) the ``pretrained_model_name_or_path``
argument, the ``QWEN3_VL_PATH`` environment variable, or the default HF hub id.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.models.qwen3_vl import Qwen3VLProcessor

from lerobot.transforms.core import DataDict, DataTransformFn
from lerobot.utils.constants import OBS_IMAGES, OBS_STR

_DEFAULT_QWEN3_VL = "Qwen/Qwen3-VL-4B-Instruct"

# Number of canonical camera slots consumed by the model.
_NUM_IMAGE_SLOTS = 3


def _resolve_processor_path(path: str | None) -> str:
    return path or os.environ.get("QWEN3_VL_PATH") or _DEFAULT_QWEN3_VL


@DataTransformFn.register_subclass("pelicanvla05_processor")
@DataTransformFn.register_subclass("basevla_processor")
@dataclass
class PelicanVLA05ProcessorTransformFn(DataTransformFn):
    """Run the Qwen3-VL processor over the three camera images and the task
    string, producing ``input_ids`` / ``attention_mask`` / ``pixel_values`` /
    ``image_grid_thw`` under the ``observation`` namespace."""

    pretrained_model_name_or_path: str | None = None
    max_length: int = 48
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    spatial_merge_size: int = 2

    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151655

    processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.processor = Qwen3VLProcessor.from_pretrained(
            _resolve_processor_path(self.pretrained_model_name_or_path)
        )
        self.vision_start_token_id = self.processor.vision_start_token_id
        self.vision_end_token_id = self.processor.vision_end_token_id
        self.image_token_id = self.processor.image_token_id

    def __call__(self, data: DataDict) -> DataDict:
        token_ids: list[int] = []
        token_mask: list[int] = []
        pixels: list[torch.Tensor] = []
        grids: list[torch.Tensor] = []

        for slot in range(_NUM_IMAGE_SLOTS):
            key = f"{OBS_IMAGES}.image{slot}"
            processed = self.processor.image_processor(data[key][1], do_rescale=False)
            pixels.append(processed.pixel_values)
            grids.append(processed.image_grid_thw)

            n_patches = int(torch.prod(grids[-1]) // (self.spatial_merge_size ** 2))
            token_ids += [self.vision_start_token_id, *([self.image_token_id] * n_patches), self.vision_end_token_id]
            present = bool(data[f"{key}_mask"])
            token_mask += [1 if present else 0] * (n_patches + 2)

        data[f"{OBS_STR}.pixel_values"] = torch.cat(pixels)
        data[f"{OBS_STR}.image_grid_thw"] = torch.cat(grids)

        prompt = self.processor.tokenizer(
            data[self.task_key],
            max_length=self.max_length,
            padding_side=self.padding_side,
            padding=self.padding,
            truncation=self.truncation,
        )
        token_ids += prompt.input_ids
        token_mask += prompt.attention_mask
        data[f"{OBS_STR}.input_ids"] = torch.tensor(token_ids)
        data[f"{OBS_STR}.attention_mask"] = torch.tensor(token_mask)

        # Separate task-only encoding (used by the optional contrastive head).
        task_only = self.processor.tokenizer(
            data.get(self.task_key, ""),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        data[f"{OBS_STR}.task.input_ids"] = torch.tensor(task_only.input_ids)
        data[f"{OBS_STR}.task.attention_mask"] = torch.tensor(task_only.attention_mask)
        return data


# Compatibility alias for older inference integrations.
Qwen3_VLProcessorTransformFn = PelicanVLA05ProcessorTransformFn


