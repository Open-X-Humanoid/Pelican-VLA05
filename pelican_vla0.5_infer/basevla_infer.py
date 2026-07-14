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
"""Compatibility entry point for the PelicanVLA05 inference API.

Example
-------
    from pelicanvla05_infer import PelicanVLA05Inference

    engine = PelicanVLA05Inference(
        model_path="/path/to/checkpoints/040000/pretrained_model",
        camera_map={"front": "image0", "wrist": "image1", "side": "image2"},
        state_stats={"mean": state_mean, "std": state_std},
        action_stats={"mean": action_mean, "std": action_std},
        delta_mask=[True, True, True, True, True, True, False],  # last dim absolute
        action_dim=7,
    )

    action_chunk = engine.infer(
        images={"front": rgb_hwc_uint8, "wrist": ..., "side": ...},
        state=[j1, j2, j3, j4, j5, j6, gripper],
        task="pick up the cup",
    )
    # action_chunk: (chunk_size, action_dim) absolute targets

Set the ``QWEN3_VL_PATH`` env var (or pass ``qwen3_vl_path=...``) to point at the
Qwen3-VL weights used to initialise the backbone.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

# Make the bundled `src/` package importable without installation.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    import sys

    sys.path.insert(0, str(_SRC))

from lerobot.policies.pelicanvla05 import (  # noqa: E402
    PelicanVLA05Policy,
    PelicanVLA05ProcessorTransformFn,
)
from lerobot.transforms.utils import resize_with_pad  # noqa: E402


def _to_chw_float(image: np.ndarray) -> torch.Tensor:
    """Convert an HWC uint8/float RGB image into a CHW float tensor in [0, 1]."""
    tensor = torch.as_tensor(np.asarray(image))
    if tensor.ndim != 3:
        raise ValueError(f"expected an HxWxC image, got shape {tuple(tensor.shape)}")
    tensor = tensor.permute(2, 0, 1).float()
    if tensor.max() > 1.5:  # looks like 0-255 data
        tensor = tensor / 255.0
    return tensor


class PelicanVLA05Inference:
    """Load a trained PelicanVLA05 checkpoint and turn raw robot observations into
    action chunks.

    The class keeps a short rolling buffer per camera so it can assemble the
    temporal image stack the model expects (``config.image_delta_indices``).
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        camera_map: Mapping[str, str],
        state_stats: Mapping[str, Sequence[float]],
        action_stats: Mapping[str, Sequence[float]],
        action_dim: int,
        delta_mask: Sequence[bool] | None = None,
        qwen3_vl_path: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        frame_buffer_size: int = 32,
        norm_eps: float = 0.0,
        strict: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.camera_map = dict(camera_map)
        self.action_dim = action_dim
        self.delta_mask = None if delta_mask is None else np.asarray(delta_mask, dtype=bool)
        self.norm_eps = norm_eps

        self.state_mean = np.asarray(state_stats["mean"], dtype=np.float32)
        self.state_std = np.asarray(state_stats["std"], dtype=np.float32)
        self.action_mean = np.asarray(action_stats["mean"], dtype=np.float32)
        self.action_std = np.asarray(action_stats["std"], dtype=np.float32)

        if qwen3_vl_path:
            os.environ.setdefault("QWEN3_VL_PATH", qwen3_vl_path)

        self.policy = PelicanVLA05Policy.from_pretrained(str(model_path), strict=strict)
        self.policy.to(self.device)
        self.policy.eval()

        self.processor = PelicanVLA05ProcessorTransformFn(
            pretrained_model_name_or_path=qwen3_vl_path
        )

        self.image_delta_indices = self.policy.config.image_delta_indices or [0]
        self.image_resolution = tuple(self.policy.config.image_resolution)
        self.max_state_dim = self.policy.config.max_state_dim

        self._buffers: dict[str, collections.deque] = {
            host: collections.deque(maxlen=frame_buffer_size) for host in self.camera_map
        }

    def reset(self) -> None:
        """Clear the per-camera frame history (call between episodes)."""
        for buf in self._buffers.values():
            buf.clear()
        self.policy.reset()

    def _temporal_stack(self, host_cam: str) -> tuple[torch.Tensor, bool]:
        buf = self._buffers[host_cam]
        target_h, target_w = self.image_resolution
        if not buf:
            blank = torch.zeros(len(self.image_delta_indices), 3, target_h, target_w)
            return blank, False

        last = len(buf) - 1
        frames = []
        for offset in self.image_delta_indices:
            idx = max(0, min(last, last + offset))
            frames.append(buf[idx])
        stack = resize_with_pad(torch.stack(frames), target_h, target_w)
        return stack, True

    def _safe_std(self, std: np.ndarray) -> np.ndarray:
        adjusted = std.copy()
        adjusted[adjusted == 0] = 1.0
        if self.norm_eps:
            adjusted += self.norm_eps
        return adjusted

    def _normalize(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (x - mean) / self._safe_std(std)

    def _unnormalize(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return x * self._safe_std(std) + mean

    @torch.no_grad()
    def infer(
        self,
        images: Mapping[str, np.ndarray],
        state: Sequence[float],
        task: str,
    ) -> np.ndarray:
        """Predict an action chunk for one timestep.

        Args:
            images: mapping of host camera name -> HxWxC RGB image.
            state: current proprioceptive state (raw units).
            task: natural-language instruction.

        Returns:
            ``(chunk_size, action_dim)`` array of absolute action targets.
        """
        for host_cam in self.camera_map:
            if host_cam in images:
                self._buffers[host_cam].append(_to_chw_float(images[host_cam]))

        state_raw = np.asarray(state, dtype=np.float32)
        state_norm = self._normalize(state_raw, self.state_mean, self.state_std)
        state_t = torch.from_numpy(state_norm).float()
        if state_t.shape[-1] < self.max_state_dim:
            state_t = torch.nn.functional.pad(state_t, (0, self.max_state_dim - state_t.shape[-1]))

        sample: dict = {"task": task, "observation.state": state_t}
        for host_cam, slot in self.camera_map.items():
            key = f"observation.images.{slot}"
            stack, present = self._temporal_stack(host_cam)
            sample[key] = stack
            sample[f"{key}_mask"] = torch.tensor(present, dtype=torch.bool)

        sample = self.processor(sample)

        batch = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                v = v.unsqueeze(0)
                batch[k] = v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
            else:
                batch[k] = v

        autocast = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast:
            chunk, _ = self.policy.predict_action_chunk(batch)

        pred_norm = chunk[0, :, : self.action_dim].cpu().float().numpy()
        pred_abs = self._unnormalize(
            pred_norm, self.action_mean[: self.action_dim], self.action_std[: self.action_dim]
        )

        if self.delta_mask is not None:
            pred_abs[:, self.delta_mask] += state_raw[self.delta_mask]

        return pred_abs


# Existing callers may continue importing the historical class name.
BaseVLAInference = PelicanVLA05Inference


def load_stats_json(
    stats_path: str | os.PathLike,
    state_keys: Sequence[str],
    action_keys: Sequence[str],
    robot_type: str | None = None,
) -> tuple[dict, dict]:
    """Assemble state/action ``{mean,std}`` from a checkpoint ``stats.json``.

    ``state_keys`` / ``action_keys`` are concatenated (in order) to form the full
    state and action statistics. Pass ``robot_type`` if the file is nested per
    embodiment.
    """
    with open(stats_path) as f:
        raw = json.load(f)
    if robot_type and robot_type in raw:
        raw = raw[robot_type]

    def gather(keys, field):
        return np.concatenate([np.asarray(raw[k][field], dtype=np.float32) for k in keys])

    state_stats = {"mean": gather(state_keys, "mean"), "std": gather(state_keys, "std")}
    action_stats = {"mean": gather(action_keys, "mean"), "std": gather(action_keys, "std")}
    return state_stats, action_stats


def _demo(args: argparse.Namespace) -> None:
    """Smoke-test the pipeline end to end with random images (no real robot)."""
    state_dim = args.action_dim
    state_stats = {"mean": np.zeros(state_dim, np.float32), "std": np.ones(state_dim, np.float32)}
    action_stats = {"mean": np.zeros(state_dim, np.float32), "std": np.ones(state_dim, np.float32)}

    engine = PelicanVLA05Inference(
        model_path=args.model_path,
        camera_map={"cam0": "image0", "cam1": "image1", "cam2": "image2"},
        state_stats=state_stats,
        action_stats=action_stats,
        action_dim=args.action_dim,
        qwen3_vl_path=args.qwen3_vl_path,
    )
    images = {name: np.random.randint(0, 255, (480, 640, 3), np.uint8) for name in engine.camera_map}
    # Prime the temporal buffer so past frames exist.
    for _ in range(2):
        engine.infer(images, [0.0] * args.action_dim, args.task)
    chunk = engine.infer(images, [0.0] * args.action_dim, args.task)
    print(f"predicted action chunk shape: {chunk.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PelicanVLA05 minimal inference demo")
    parser.add_argument("--model_path", required=True, help="path to a pretrained_model directory")
    parser.add_argument("--qwen3_vl_path", default=os.environ.get("QWEN3_VL_PATH"))
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--task", default="pick up the object")
    _demo(parser.parse_args())


if __name__ == "__main__":
    main()
