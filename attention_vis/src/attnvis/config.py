"""Input path / env single source of truth.

Every dataset root, weight path, upstream checkout, and interpreter is listed
here and overridable via ATTNVIS_*. Output paths live in paths.py.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key) or default


def _p(env_key: str, default: str) -> Path:
    return Path(_env(env_key, default))


DATA_ROOTS = {
    "libero":      _p("ATTNVIS_LIBERO_ROOT",      ""),
    "robotwin":    _p("ATTNVIS_ROBOTWIN_ROOT",    ""),
    # LeRobot 3.0 dataset roots — one per collection layout on disk.
    # See README §④ for the on-disk schema they must follow.
    "unify":       _p("ATTNVIS_UNIFY_ROOT",       ""),  # tienkung / ur5e etc.
    "lerobot_v3":  _p("ATTNVIS_LEROBOT_V3_ROOT",  ""),  # ur5e_v3 / ur5e_dual / franka_fr3
    "vla_lerobot": _p("ATTNVIS_VLA_LEROBOT_ROOT", ""),  # agilex_jd / realsource_world_v30
}

WEIGHTS = {
    "pelicanvla_ckpt": _p("ATTNVIS_PELICANVLA_CKPT", ""),
}

EXTERNAL = {
    "qwen3_vl": _p("QWEN3_VL_PATH", ""),
    "cosmos":   _p("COSMOS_TOKENIZER_PATH", ""),
}

UPSTREAM = {
    "pelicanvla": _p("ATTNVIS_PELICANVLA_SRC", ""),
}

ENVS = {
    "pelicanvla": {"kind": "venv", "name": "pelicanvla_release",
                   "python": _env("ATTNVIS_PELICANVLA_PY", "")},
}


def data_root(key: str) -> Path:
    if key not in DATA_ROOTS:
        raise KeyError(f"unknown data root {key!r}; known {list(DATA_ROOTS)}")
    return DATA_ROOTS[key]


def require(path, what: str = "") -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"missing{(' (' + what + ')') if what else ''}: {p}\n"
            f"  → override via env var (see .env.example / attnvis.config).")
    return p
