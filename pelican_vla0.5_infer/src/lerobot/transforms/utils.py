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
"""Small tensor helpers used by the inference-time image transforms.

Clean reimplementation written for the open-source inference release.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_batched(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return ``image`` as a 4D ``(B, C, H, W)`` tensor plus a flag telling the
    caller whether the leading batch axis was added here (and should be undone)."""
    if image.ndim == 4:
        return image, False
    if image.ndim == 3:
        return image.unsqueeze(0), True
    raise ValueError(f"expected a [C,H,W] or [B,C,H,W] tensor, received shape {tuple(image.shape)}")


def resize_with_pad(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize ``image`` to ``(target_h, target_w)`` preserving aspect ratio.

    The image is scaled by the largest factor that still fits inside the target
    box, then symmetrically zero-padded to reach the exact requested size.
    Accepts a single image ``[C, H, W]`` or a batch ``[B, C, H, W]`` with pixel
    values in ``[0, 1]`` and returns the same rank it was given.
    """
    batched, added_axis = _to_batched(image)
    _, _, h, w = batched.shape

    if (h, w) != (target_h, target_w):
        ratio = min(target_h / h, target_w / w)
        scaled_h, scaled_w = round(h * ratio), round(w * ratio)

        align = False if mode in ("bilinear", "bicubic") else None
        batched = F.interpolate(batched, size=(scaled_h, scaled_w), mode=mode, align_corners=align)

        gap_h, gap_w = target_h - scaled_h, target_w - scaled_w
        top, left = gap_h // 2, gap_w // 2
        batched = F.pad(batched, (left, gap_w - left, top, gap_h - top), value=0.0)
        batched = batched.clamp(0.0, 1.0)

    return batched.squeeze(0) if added_axis else batched
