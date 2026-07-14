"""Dense-frame dump driver — writes the archive that the fig utilities read.

Behavior: prefetch frames once → load model once → sweep all scenes/frames →
release. One process dumps exactly one model.

Archive layout (per model × per view):
  outputs/dense/{exp}/{scene_tag}/{model_key}/
      frames_{view}.npz     # heat:(T,H,W) f32, rgb:(T,H,W,3) u8, frames:(T,),
                            # instruction, model, scene
      {view}/frame_{t:04d}.png  # per-frame overlay
      {view}/contact_strip.png  # horizontal strip of all T frames
      manifest.json

Usage:
  python -m attnvis.dump_dense --exp demo \\
      --scenes robotwin:adjust_bottle:0 --nframes 30
  # override the release ckpt:
  python -m attnvis.dump_dense --exp demo \\
      --ckpt /path/to/pretrained_model --label my_run \\
      --scenes robotwin:adjust_bottle:0 --nframes 30
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attnvis.core.render import _overlay
from attnvis.paths import DENSE as OUT


_LEROBOT_SOURCES = {
    "tienkung":       "TienkungSource",
    "ur5e":           "Ur5eSource",
    "ur5e_v3":        "Ur5eV3Source",
    "ur5e_dual":      "Ur5eDualSource",
    "agilex_jd":      "AgilexJdSource",
    "realsource_v30": "RealSourceV30Source",
    "franka_fr3":     "FrankaFr3Source",
    "franka_single":  "FrankaSingleSource",
}


def _make_source(name: str, suite: str, ep: int):
    if name == "libero":
        from attnvis.sources.libero import LiberoSource
        return LiberoSource(suite=suite, task_id=ep)
    if name == "robotwin":
        from attnvis.sources.robotwin import RobotwinSource
        return RobotwinSource(suite=suite)
    if name in _LEROBOT_SOURCES:
        from attnvis.sources import robomind as _rm
        return getattr(_rm, _LEROBOT_SOURCES[name])(suite=suite, task_id=ep)
    raise ValueError(f"unknown source: {name}")


def _make_adapter(ckpt=None):
    # attnvis ships one adapter (PelicanVLA); import lazily so figure-only
    # workflows don't pull in torch.
    from attnvis.adapters.pelicanvla import PelicanVLAAdapter
    return PelicanVLAAdapter(ckpt=ckpt) if ckpt is not None else PelicanVLAAdapter()


def _parse_scene(s: str):
    """'robotwin:adjust_bottle:0' -> ('robotwin','adjust_bottle',0). Episode
    defaults to 0."""
    parts = s.split(":")
    src, suite = parts[0], parts[1]
    ep = int(parts[2]) if len(parts) > 2 else 0
    return src, suite, ep


def _scene_tag(src: str, suite: str, ep: int = 0) -> str:
    return f"{src}__{suite}" if ep == 0 else f"{src}__{suite}__ep{ep}"


def _save_contact_strip(path: Path, rgbs: list, heats: list, frames: list, vmax: float) -> None:
    """Horizontal strip of all T frames with attention overlaid; uses a shared
    vmax so the strip is comparable across frames."""
    n = len(rgbs)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4))
    axes = np.atleast_1d(axes)
    for i in range(n):
        _overlay(axes[i], rgbs[i], heats[i], f"t={frames[i]}", vmax=vmax)
    plt.tight_layout()
    plt.savefig(path, dpi=90, bbox_inches="tight")
    plt.close()


def dump_one(exp: str, scene_specs: list, nframes: int,
             ckpt: str | None = None, label: str | None = None, save_pngs: bool = True) -> None:
    import torch  # lazy: pure CPU paths don't need it

    plan = []
    for (src, suite, ep) in scene_specs:
        srcobj = _make_source(src, suite, ep)
        ts = srcobj.pick_frames(ep, nframes)
        frames = {t: srcobj.get_frame(ep, t) for t in ts}
        plan.append((_scene_tag(src, suite, ep), src, suite, ep, ts, frames))
        print(f"[dense] prefetch {_scene_tag(src,suite,ep)} ep{ep}: {len(ts)} frames {ts}", flush=True)

    model_key = label or ("pelicanvla_custom" if ckpt is not None else "pelicanvla")
    adapter = _make_adapter(ckpt)
    adapter.load()
    print(f"[dense] loaded {model_key}", flush=True)

    for (stag, src, suite, ep, ts, frames) in plan:
        outdir = OUT / exp / stag / model_key
        outdir.mkdir(parents=True, exist_ok=True)
        acc: dict[str, dict[str, list]] = {}
        for t in ts:
            res = adapter.attention(frames[t])
            for cam, heat in res.heat.items():
                d = acc.setdefault(cam, {"heat": [], "rgb": []})
                d["heat"].append(np.asarray(heat, dtype=np.float32))
                d["rgb"].append(frames[t].rgb[cam].astype(np.uint8))
        for cam, d in acc.items():
            heat_stk = np.stack(d["heat"])
            rgb_stk = np.stack(d["rgb"])
            np.savez_compressed(
                outdir / f"frames_{cam}.npz",
                heat=heat_stk, rgb=rgb_stk, frames=np.array(ts),
                instruction=np.array(frames[ts[0]].instruction),
                model=np.array(model_key), scene=np.array(stag))
            if save_pngs:
                vfull = float(np.percentile(heat_stk, 99.5))
                vdir = outdir / cam
                vdir.mkdir(parents=True, exist_ok=True)
                for i, t in enumerate(ts):
                    fig, ax = plt.subplots(figsize=(4, 3.2))
                    _overlay(ax, rgb_stk[i], heat_stk[i], f"{model_key} {cam} t={t}", vmax=vfull)
                    plt.tight_layout()
                    plt.savefig(vdir / f"frame_{t:04d}.png", dpi=100, bbox_inches="tight")
                    plt.close()
                _save_contact_strip(vdir / "contact_strip.png",
                                    list(rgb_stk), list(heat_stk), ts, vfull)
        (outdir / "manifest.json").write_text(json.dumps(
            {"exp": exp, "scene": stag, "source": src, "suite": suite, "episode": ep,
             "model": model_key, "views": list(acc.keys()), "frames": ts},
            ensure_ascii=False, indent=2))
        print(f"[dense] {model_key} {stag}: views={list(acc.keys())} -> {outdir}", flush=True)

    if getattr(adapter, "policy", None) is not None:
        del adapter.policy
        adapter.policy = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[dense] DONE exp={exp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="Experiment name; becomes outputs/dense/<exp>/")
    ap.add_argument("--ckpt", default=None,
                    help="Override the release checkpoint")
    ap.add_argument("--label", default=None,
                    help="With --ckpt: output subdir name (model_key)")
    ap.add_argument("--scenes", nargs="+", required=True, help='"source:suite:episode" list')
    ap.add_argument("--nframes", type=int, default=30)
    ap.add_argument("--no-pngs", action="store_true",
                    help="Store only npz, skip per-frame PNG and contact strip")
    args = ap.parse_args()

    scene_specs = [_parse_scene(s) for s in args.scenes]
    dump_one(args.exp, scene_specs, args.nframes,
             ckpt=args.ckpt, label=args.label, save_pngs=not args.no_pngs)


if __name__ == "__main__":
    main()
