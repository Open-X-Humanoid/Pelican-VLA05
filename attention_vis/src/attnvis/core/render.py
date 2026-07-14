"""Rendering layer — overlay an AttnResult onto the raw image, save the npz,
and stitch multi-model comparison plots. Works for any model x source
combination (guaranteed by the contract in core.types).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attnvis.core.types import AttnResult, Frame


def _overlay(ax, rgb: np.ndarray, heat: np.ndarray | None, title: str,
             vmax: float | None = None) -> None:
    """Overlay a heatmap (semi-transparent, jet colormap) on top of the image.

    vmax=None → each cell self-normalizes (default; used by save_result and
    compare()).
    vmax=<float> → shared color scale (pass a percentile of the full stack
    for cross-cell comparability, as dump_dense does for its contact strips).
    """
    ax.imshow(rgb)
    if heat is not None:
        vm = max(float(heat.max()), 1e-8) if vmax is None else max(float(vmax), 1e-8)
        ax.imshow(heat, cmap="jet", alpha=0.5, vmin=0, vmax=vm)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_result(model_name: str, frame: Frame, result: AttnResult, outdir: Path) -> None:
    """Save one model's result for one frame: an overlay PNG per camera plus
    a single npz containing raw images + heatmaps for later reuse."""
    outdir.mkdir(parents=True, exist_ok=True)
    save = {"cams": np.array(frame.cams), "instruction": np.array(frame.instruction),
            "model": np.array(model_name)}
    for cam, heat in result.heat.items():
        rgb = frame.rgb[cam]
        save[f"rgb_{cam}"] = rgb.astype(np.uint8)
        save[f"heat_{cam}"] = heat.astype(np.float32)
        fig, ax = plt.subplots(figsize=(5, 4))
        _overlay(ax, rgb, heat, f"{model_name} action->image ({cam})")
        plt.tight_layout()
        plt.savefig(outdir / f"{model_name}_{cam}.png", dpi=120, bbox_inches="tight")
        plt.close()
    np.savez_compressed(outdir / f"{model_name}_attn.npz", **save)


def compare(frame: Frame, results: dict[str, AttnResult], outdir: Path, tag: str = "") -> Path:
    """Stitch a comparison plot: rows = cameras, columns = [raw] + one per
    model. results = {model_name: AttnResult}."""
    outdir.mkdir(parents=True, exist_ok=True)
    models = list(results.keys())
    cams = frame.cams
    ncol = 1 + len(models)
    fig, axes = plt.subplots(len(cams), ncol, figsize=(4.2 * ncol, 3.4 * len(cams)))
    axes = np.atleast_2d(axes)
    for ri, cam in enumerate(cams):
        _overlay(axes[ri, 0], frame.rgb[cam], None, f"{cam} raw" if ri == 0 else cam)
        for mi, m in enumerate(models):
            heat = results[m].heat.get(cam)  # missing cam is tolerated (None)
            _overlay(axes[ri, 1 + mi], frame.rgb[cam], heat, m if ri == 0 else "")
    fig.suptitle(f"{' vs '.join(models)} | [{frame.instruction[:55]}]  {tag}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = outdir / f"compare_{tag}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out
