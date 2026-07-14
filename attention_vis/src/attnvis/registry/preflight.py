"""Preflight compatibility check — run before a dump so failures show up here
instead of after the model has loaded.

Catches:
  - unknown source
  - camera short name is both third-person and wrist (would mis-sort in adapters)
  - missing weights / interpreter paths (when check_paths=True)

Pure Python, no heavy deps — safe to run anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from attnvis import config
from attnvis.registry import embodiments as E


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        lines = []
        for p in self.problems:
            lines.append(f"  [x block  ] {p}")
        for w in self.warnings:
            lines.append(f"  [! warn   ] {w}")
        for i in self.info:
            lines.append(f"  [. info   ] {i}")
        head = "preflight: OK" if self.ok else "preflight: blocking issues"
        return head + ("\n" + "\n".join(lines) if lines else "")


def parse_scene(s: str) -> tuple[str, str, int]:
    """'robotwin:adjust_bottle:0' → ('robotwin', 'adjust_bottle', 0)."""
    parts = s.split(":")
    if len(parts) < 2:
        raise ValueError(f"scene format is 'source:suite[:episode]', got {s!r}")
    return parts[0], parts[1], (int(parts[2]) if len(parts) > 2 else 0)


def _scene_data_root(src: str, suite: str) -> Path | None:
    """Data root directory for existence checking; None for unknown sources.

    For LeRobot-style sources (`emb.loader == "lerobot"`), pull the concrete
    dataset root registered for the requested suite.
    """
    if src == "libero":
        return config.DATA_ROOTS["libero"] / suite if config.DATA_ROOTS.get("libero") else None
    if src == "robotwin":
        return config.DATA_ROOTS["robotwin"] / suite if config.DATA_ROOTS.get("robotwin") else None
    emb = E.EMBODIMENTS.get(src)
    if emb is not None and emb.loader == "lerobot":
        root = emb.tasks.get(suite)
        return Path(root) if root else None
    return None


def _check_role_consistency() -> list[str]:
    problems = []
    third, wrist = E.camera_role_sets()
    dup = third & wrist
    if dup:
        problems.append(f"camera short name is both third_person and wrist: {sorted(dup)}")
    return problems


def check(scenes: list[str], check_paths: bool = True) -> Report:
    """Validate scenes against the PelicanVLA pipeline. `check_paths=False`
    skips filesystem existence checks (useful when only the fig utilities
    will run downstream)."""
    r = Report()
    r.info.append("model=pelicanvla capture=bottleneck on → two-hop / off → direct (auto by config)")

    for p in _check_role_consistency():
        r.problems.append(f"registry inconsistency: {p}")

    py = os.environ.get("ATTNVIS_PELICANVLA_PY", "")
    if check_paths and py and not Path(py).exists():
        r.warnings.append(
            f"ATTNVIS_PELICANVLA_PY interpreter not found: {py} (unset it to "
            "use the current interpreter, or point it at a valid PelicanVLA venv)")

    for s in scenes:
        try:
            src, suite, ep = parse_scene(s)
        except ValueError as e:
            r.problems.append(str(e))
            continue
        emb = E.EMBODIMENTS.get(src)
        if emb is None:
            r.problems.append(
                f"{s}: unknown source {src!r}; registered: {list(E.EMBODIMENTS)}")
            continue

        r.info.append(
            f"{src}: {emb.arm}-arm state{emb.state_dim}D cams={emb.cams} "
            f"(third_person={emb.third_person()})")

        if check_paths:
            root = _scene_data_root(src, suite)
            if root is not None and not root.exists():
                r.warnings.append(
                    f"{s}: data root not found: {root} (set the corresponding "
                    "ATTNVIS_*_ROOT env var; see .env.example)")

    return r
