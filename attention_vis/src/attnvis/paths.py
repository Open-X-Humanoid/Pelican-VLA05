"""Output path single source of truth.

OUT defaults to <repo>/outputs (not under src/). Override with ATTNVIS_OUT.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent          # src/attnvis
REPO_ROOT = PKG_ROOT.parents[1]                     # attention_visualization/

OUT = Path(os.environ["ATTNVIS_OUT"]).resolve() if os.environ.get("ATTNVIS_OUT") \
    else (REPO_ROOT / "outputs").resolve()
DENSE = OUT / "dense"
