"""PelicanVLAAdapter — wraps the PelicanVLA public inference release (currently 0.5).

Upstream: `ATTNVIS_PELICANVLA_SRC` provides `lerobot.policies.basevla_4B`
(the upstream module retains its historical name). Weights:
`ATTNVIS_PELICANVLA_CKPT`. External deps: `QWEN3_VL_PATH` + `COSMOS_TOKENIZER_PATH`.

Two-hop (bottleneck tokens): action doesn't look at image directly — it goes
action → bottleneck slots → image. Capture = monkeypatch
`policy.model.vlm.language_model.forward` (which is called multiple times),
take calls[0] (prefix self-attn), calls[1] (slots look at prefix), calls[-1]
(action looks at slots), then compose:
    einsum("lak,lki->lai", action→slots, slots→image) → action→image.

Camera coverage: 3 native (robotwin); 2 (LIBERO) — replicate the second slot
into slot 3 (cross-embodiment OOD).
Input pipeline aligned with the release's `PelicanVLA05Inference.infer`:
`resize_with_pad` + manual state normalization + `Qwen3_VLProcessorTransformFn`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from attnvis.core.types import AttnResult, Frame, ModelAdapter
from attnvis.config import EXTERNAL, UPSTREAM, WEIGHTS, require


PELICANVLA_ROOT = str(UPSTREAM["pelicanvla"])
DEFAULT_CKPT = str(WEIGHTS["pelicanvla_ckpt"])


def _slots_enabled(config) -> bool:
    """Release config uses enable_bottleneck_tokens; older training checkouts
    called it enable_reasoning_slots — check both."""
    return bool(getattr(config, "enable_bottleneck_tokens",
                        getattr(config, "enable_reasoning_slots", False)))


def _num_slots(config) -> int:
    return int(getattr(config, "bottleneck_num_tokens",
                       getattr(config, "reasoning_num_slots", 32)))


def _tolerant_ckpt(ckpt: str):
    """Tolerant load: if config.json carries training-only fields the current
    Config class does not recognize, strip them and write to a temp dir.
    Release checkpoints are usually clean; this is a fallback for older
    training checkpoints.
    """
    import re as _re
    import tempfile as _tmp

    src = Path(ckpt)
    try:
        from lerobot.configs.policies import PreTrainedConfig
        PreTrainedConfig.from_pretrained(str(src))
        return str(src)
    except Exception:
        pass

    cur = json.loads((src / "config.json").read_text())
    d = Path(_tmp.mkdtemp(prefix="pelican_ckpt_"))
    for f in src.iterdir():
        if f.name != "config.json":
            os.symlink(f, d / f.name)
    for _ in range(30):
        (d / "config.json").write_text(json.dumps(cur))
        try:
            from lerobot.configs.policies import PreTrainedConfig
            PreTrainedConfig.from_pretrained(str(d))
            print(f"[pelicanvla] tolerant load: stripped unrecognized config fields → {d}", flush=True)
            return str(d)
        except Exception as e:
            bad = [k for k in _re.findall(r"`([^`]+)`", str(e)) if k in cur]
            if not bad:
                raise
            for k in bad:
                cur.pop(k, None)
    return str(d)


def _resolve_state_stats(stats: dict, embodiment: str | None = None):
    """Resolve state (mean, std) + dimensionality + source label from the
    release's nested stats.json.

    Priority order:
      1) explicit embodiment key that contains observation.state
      2) aloha / observation.state (14D; RoboTwin-friendly)
      3) AgileX Split Aloha, concatenating the 14D field order defined in
         the embodiment registry (training-time convention)
    Returns (mean, std, dim, source_label).
    """
    sk = ["mean", "std"]

    def _from_obs_state(src: dict, label: str):
        mean = np.asarray(src["mean"], dtype=np.float32)
        std = np.asarray(src["std"], dtype=np.float32)
        return mean, std, int(mean.shape[0]), label

    if embodiment and embodiment in stats and "observation.state" in stats[embodiment]:
        return _from_obs_state(stats[embodiment]["observation.state"],
                               f"{embodiment}/observation.state")
    if "aloha" in stats and "observation.state" in stats["aloha"]:
        return _from_obs_state(stats["aloha"]["observation.state"], "aloha/observation.state")

    from attnvis.registry.embodiments import PELICANVLA_AGILEX_STATE_FIELDS
    agx = stats["AgileX Split Aloha"]
    fields = PELICANVLA_AGILEX_STATE_FIELDS
    mean = np.concatenate([np.asarray(agx[f]["mean"]).reshape(-1) for f in fields]).astype(np.float32)
    std = np.concatenate([np.asarray(agx[f]["std"]).reshape(-1) for f in fields]).astype(np.float32)
    label = "AgileX Split Aloha (14D: " + "+".join(f.split(".")[-2] for f in fields) + ")"
    return mean, std, int(mean.shape[0]), label


class PelicanVLAAdapter(ModelAdapter):
    name = "pelicanvla"

    def __init__(
        self,
        ckpt: str | None = None,
        dtype: str = "bfloat16",
        embodiment: str | None = None,
        qwen3_vl_path: str | None = None,
        cosmos_path: str | None = None,
    ):
        # Weights source: release single checkpoint; override via --ckpt or
        # ATTNVIS_PELICANVLA_CKPT.
        self.ckpt = ckpt if ckpt is not None else DEFAULT_CKPT
        self.dtype = dtype
        self.embodiment = embodiment  # stats.json nested key, e.g. "aloha"/"ur5e"; None=auto
        self.qwen3_vl_path = qwen3_vl_path or str(EXTERNAL["qwen3_vl"])
        self.cosmos_path = cosmos_path or str(EXTERNAL["cosmos"])
        # Capture path: auto (pick two-hop/direct by config.enable_bottleneck_tokens)
        # / twohop / direct.
        self.capture = os.environ.get("PELICANVLA_CAPTURE", "auto")
        self.policy = None
        self.processor = None

    def load(self) -> None:
        """Load the release weights once and build the input transform
        (including state normalization)."""
        import torch

        src = f"{PELICANVLA_ROOT}/src"
        if src not in sys.path:
            sys.path.insert(0, src)

        # The release resolves Qwen / Cosmos paths from these two env vars
        # (constructor args override).
        os.environ.setdefault("QWEN3_VL_PATH", self.qwen3_vl_path)
        os.environ.setdefault("COSMOS_TOKENIZER_PATH", self.cosmos_path)
        require(self.ckpt, "PelicanVLA checkpoint")
        require(self.qwen3_vl_path, "QWEN3_VL_PATH")
        require(self.cosmos_path, "COSMOS_TOKENIZER_PATH")

        from lerobot.policies.basevla_4B.modeling_basevla import BaseVLAPolicy
        from lerobot.policies.basevla_4B.transform_basevla import Qwen3_VLProcessorTransformFn

        dtype = torch.float32 if self.dtype == "float32" else torch.bfloat16
        ck = _tolerant_ckpt(self.ckpt)
        self.policy = BaseVLAPolicy.from_pretrained(ck, strict=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy.to(device=device, dtype=dtype).eval()
        self.config = self.policy.config
        self._dtype = dtype
        self._device = device

        stats = json.loads(Path(ck, "stats.json").read_text())
        mean, std, self._state_dim, self._stats_source = _resolve_state_stats(
            stats, embodiment=self.embodiment)
        self._state_mean = mean
        self._state_std = np.where(std == 0, 1.0, std)

        self.processor = Qwen3_VLProcessorTransformFn(
            pretrained_model_name_or_path=self.qwen3_vl_path)

        deltas = self.config.image_delta_indices or [0]
        self._T = len(deltas)
        self._image_resolution = tuple(self.config.image_resolution)
        self._max_state_dim = int(self.config.max_state_dim)

        _eff = self.capture if self.capture != "auto" else (
            "twohop" if _slots_enabled(self.config) else "direct")
        print(
            f"[pelicanvla] ckpt={self.ckpt} slots={_slots_enabled(self.config)} "
            f"capture={self.capture}→{_eff} stats_source={self._stats_source} "
            f"dim={self._state_dim} T={self._T} qwen={self.qwen3_vl_path}",
            flush=True,
        )
        self._merge = int(getattr(self.policy.model.vlm.config.vision_config,
                                  "spatial_merge_size", 2))
        self._img_token_id = self.policy.model.vlm.config.image_token_id
        print("[pelicanvla] policy loaded.", flush=True)

    def _build_batch(self, frame: Frame) -> dict:
        """Build inputs matching the release's PelicanVLA05Inference.infer."""
        import torch
        from lerobot.transforms.utils import resize_with_pad
        from lerobot.utils.constants import OBS_IMAGES

        if len(frame.cams) >= 3:
            slot_cam = list(frame.cams[:3])
        else:
            slot_cam = [frame.cams[0], frame.cams[1], frame.cams[1]]

        target_h, target_w = self._image_resolution

        def to_thw(img):
            # HWC uint8 -> [T,3,H,W] float[0,1], then resize_with_pad to model resolution
            chw = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            stack = chw.unsqueeze(0).expand(self._T, -1, -1, -1).contiguous()
            return resize_with_pad(stack, target_h, target_w)

        s = frame.state.astype(np.float32)
        n = min(self._state_dim, s.shape[0])
        state_raw = np.zeros(self._state_dim, dtype=np.float32)
        state_raw[:n] = s[:n]
        state_norm = (state_raw - self._state_mean) / self._state_std
        state_t = torch.from_numpy(state_norm).float()
        if state_t.shape[-1] < self._max_state_dim:
            state_t = torch.nn.functional.pad(
                state_t, (0, self._max_state_dim - state_t.shape[-1]))

        sample = {
            f"{OBS_IMAGES}.image{i}": to_thw(frame.rgb[slot_cam[i]]) for i in range(3)
        }
        sample["observation.state"] = state_t
        sample["task"] = frame.instruction
        for i in range(3):
            sample[f"{OBS_IMAGES}.image{i}_mask"] = torch.tensor(True)

        sample = self.processor(sample)

        inputs = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                v = v.unsqueeze(0)
                inputs[k] = (
                    v.to(self._device, self._dtype) if v.is_floating_point()
                    else v.to(self._device)
                )
            else:
                inputs[k] = v
        return inputs

    def attention(self, frame: Frame) -> AttnResult:
        """One frame -> two-hop / direct capture -> per-camera action→image heatmap."""
        import torch
        from scipy.ndimage import zoom

        assert self.policy is not None, "call load() before attention()"

        inputs = self._build_batch(frame)

        input_ids = inputs["observation.input_ids"][0].detach().cpu().numpy()
        img_cols = np.nonzero(input_ids == self._img_token_id)[0]
        grid_thw = inputs["observation.image_grid_thw"][0].detach().cpu().numpy()
        cam_counts = [
            int(t[0]) * (int(t[1]) // self._merge) * (int(t[2]) // self._merge)
            for t in grid_thw
        ]

        # Capture: monkeypatch language_model.forward to record attention
        # from every call (averaged across heads).
        calls = []
        hs_prefix = []
        lm = self.policy.model.vlm.language_model
        orig_forward = lm.forward

        def patched_forward(*fa, **fkw):
            fkw["output_attentions"] = True
            if not hs_prefix:
                fkw["output_hidden_states"] = True
            out = orig_forward(*fa, **fkw)
            attn = getattr(out, "attentions", None)
            calls.append(
                torch.stack(attn, dim=0).float().mean(dim=2)[:, 0].cpu().numpy()
                if attn is not None else None
            )
            if not hs_prefix:
                hs = getattr(out, "hidden_states", None)
                if hs is not None:
                    hs_prefix.append(np.stack(
                        [h[0].float().cpu().numpy() for h in hs]
                    ).astype(np.float16))
            return out

        # Forcing 'direct' on a slot/bottleneck ckpt = temporarily flip the
        # bottleneck flag off (do NOT trim the KV cache) so action attends the
        # prefix image tokens directly during denoising. Off-distribution ablation.
        eff_mode = self.capture if self.capture != "auto" else (
            "twohop" if _slots_enabled(self.config) else "direct")
        mdl_cfg = self.policy.model.config
        force_noslot = (eff_mode == "direct" and _slots_enabled(mdl_cfg))
        _slots_was = getattr(mdl_cfg, "enable_bottleneck_tokens",
                             getattr(mdl_cfg, "enable_reasoning_slots", None))
        if force_noslot:
            if hasattr(mdl_cfg, "enable_bottleneck_tokens"):
                mdl_cfg.enable_bottleneck_tokens = False
            if hasattr(mdl_cfg, "enable_reasoning_slots"):
                mdl_cfg.enable_reasoning_slots = False
            print("[pelicanvla] direct on slot ckpt → temporarily disabling bottleneck "
                  "(off-distribution ablation)", flush=True)

        lm.forward = patched_forward
        try:
            with torch.no_grad():
                self.policy.predict_action_chunk(inputs, decode_image=False)
        finally:
            lm.forward = orig_forward
            if force_noslot:
                if hasattr(mdl_cfg, "enable_bottleneck_tokens"):
                    mdl_cfg.enable_bottleneck_tokens = _slots_was
                if hasattr(mdl_cfg, "enable_reasoning_slots"):
                    mdl_cfg.enable_reasoning_slots = _slots_was

        chunk = int(self.config.chunk_size)
        a_denoise = calls[-1]
        Lprefix = calls[0].shape[2]
        mode = eff_mode

        if mode == "twohop":
            K = _num_slots(self.config)
            a_step2 = calls[1]
            Lmiddle = a_step2.shape[1] - K
            slots_to_image = a_step2[:, Lmiddle:Lmiddle + K, :Lprefix][:, :, img_cols]
            action_to_slots = a_denoise[:, 1:1 + chunk, :K]
            raw_attn = np.einsum("lak,lki->lai", action_to_slots, slots_to_image)
        else:
            raw_attn = a_denoise[:, 1:1 + chunk, :Lprefix][:, :, img_cols]
        vec = raw_attn.mean(axis=1).mean(axis=0)

        self._dbg = {
            "calls": calls, "Lprefix": int(Lprefix), "img_cols": img_cols,
            "chunk": int(chunk), "cam_counts": list(cam_counts), "grid_thw": grid_thw,
            "mode": mode, "merge": self._merge,
            "K": (_num_slots(self.config) if mode == "twohop" else None),
        }

        heat = {}
        off = 0
        for ci in range(3):
            cnt = cam_counts[ci]
            seg = vec[off:off + cnt]
            off += cnt
            if ci >= len(frame.cams):
                continue
            cam = frame.cams[ci]
            t = int(grid_thw[ci][0])
            h = int(grid_thw[ci][1]) // self._merge
            w = int(grid_thw[ci][2]) // self._merge
            grid = seg.reshape(t, h, w).mean(axis=0)
            rgb = frame.rgb[cam]
            heat[cam] = zoom(
                grid,
                (rgb.shape[0] / grid.shape[0], rgb.shape[1] / grid.shape[1]),
                order=1,
            ).astype(np.float32)

        pl = raw_attn.mean(axis=1)
        heat_layers = {}
        for ci in range(3):
            if ci >= len(frame.cams):
                continue
            off2 = sum(cam_counts[:ci])
            cnt = cam_counts[ci]
            cam = frame.cams[ci]
            t = int(grid_thw[ci][0])
            h = int(grid_thw[ci][1]) // self._merge
            w = int(grid_thw[ci][2]) // self._merge
            segL = pl[:, off2:off2 + cnt]
            heat_layers[cam] = segL.reshape(pl.shape[0], t, h, w).mean(axis=1).astype(np.float32)

        raw = {
            "attn": raw_attn.astype(np.float16),
            "capture": np.array(mode),
            "cam_counts": np.array(cam_counts),
            "grid_thw": grid_thw,
            "heat_layers": heat_layers,
        }
        if hs_prefix:
            raw["hidden_prefix"] = hs_prefix[0]
            raw["input_ids"] = input_ids.astype(np.int64)
            raw["img_cols"] = img_cols.astype(np.int64)
        return AttnResult(heat=heat, raw=raw)
