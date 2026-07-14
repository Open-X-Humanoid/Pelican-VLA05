"""Shared figure primitives — the pure functions that would otherwise be
copy-pasted across figs/make_*.py: CJK font setup, heatmap blending, shared
color-range percentile calc, per-cell PNG dump.

Pure numpy / matplotlib. No model / heavy deps; safe to import anywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CJK font candidates. Registered so labels containing CJK characters (which
# can appear in dataset instructions, e.g. RoboTwin prompts) render instead
# of turning into tofu boxes. Cross-platform: Linux + Windows + macOS paths.
_CJK_FONT_PATHS = [
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",       # Microsoft YaHei
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",     # SimHei
    "C:/Windows/Fonts/simsun.ttc",     # SimSun
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

# Family names that mpl may already know about via the OS font list.
_CJK_FONT_NAMES = [
    "Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Micro Hei",
    "Microsoft YaHei", "SimHei", "SimSun", "PingFang SC",
    "Arial Unicode MS",
]


def setup_cjk_font() -> str:
    """Register a CJK font if the system has one; return the resolved family.

    Falls back to DejaVu Sans on machines without any CJK font available
    (CJK strings will render as tofu boxes there, but ASCII stays readable).
    """
    import matplotlib.font_manager as fm
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["axes.unicode_minus"] = False

    # Prefer explicit files (guarantees the font is registered).
    for p in _CJK_FONT_PATHS:
        if Path(p).exists():
            try:
                fm.fontManager.addfont(p)
                name = fm.FontProperties(fname=p).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                return name
            except Exception:
                continue

    # Otherwise trust mpl's OS-provided font list.
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_FONT_NAMES:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            return name

    return "DejaVu Sans"


def blend(rgb, heat, cmap, vmin=0.0, vmax=1.0, alpha=0.5, gamma=1.0):
    """Compose the heatmap into rgb, returning one opaque RGB [H,W,3] in [0,1]
    (PNG and PDF then render identically — layered alpha would wash out the PDF).

    gamma<1 lifts the low end of the stretched heatmap so peaky attention
    maps do not turn the whole image blue. If heat is None, only the
    normalized image is returned.
    """
    base = rgb.astype(np.float64)
    if base.max() > 1.5:
        base = base / 255.0
    base = base[..., :3]
    if heat is None:
        return np.clip(base, 0, 1)
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    h = np.clip((heat.astype(np.float64) - vmin) / (vmax - vmin + 1e-12), 0, 1)
    if gamma != 1.0:
        h = np.power(h, gamma)
    return np.clip((1 - alpha) * base + alpha * cm(h)[..., :3], 0, 1)


def pct_range(arr, lo=2.0, hi=99.5, eps=1e-6):
    """Shared color range [pLo, pHi]; defaults to p2 / p99.5 with hi > lo."""
    a = np.asarray(arr).ravel()
    plo = float(np.percentile(a, lo))
    phi = max(float(np.percentile(a, hi)), plo + eps)
    return plo, phi


def pct_vmax(arr, p=99.0, eps=1e-6):
    """One-sided vmax = p-th percentile (default 99), with an eps floor."""
    return max(float(np.percentile(np.asarray(arr).ravel(), p)), eps)


def save_cell(path, rgb01):
    """Save a [0,1] RGB array as a single-cell PNG."""
    import matplotlib.image as mpimg
    mpimg.imsave(str(path), np.clip(np.asarray(rgb01), 0, 1))
