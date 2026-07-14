"""RoboMIND / LeRobot real-scene data sources (tienkung dual-arm / ur5e single
arm / ur5e-v3 / ur5e_dual / agilex_jd / realsource_v30 / franka_fr3).

Thin wrapper over `_lerobot_frame`. One Source instance = one dataset × one
suite (task). Camera keys / state layout / task→root mapping all come from
`registry.embodiments` (single source of truth).
"""
from __future__ import annotations

import numpy as np

from attnvis.core.types import DataSource, Frame
from attnvis.registry.embodiments import EMBODIMENTS


def _lf():
    from attnvis.sources import _lerobot_frame as lf
    return lf


class _RoboMINDSource(DataSource):
    """Generic RoboMIND / LeRobot source: pick the dataset root by suite.

    Subclasses only need to set `name` (matches a key in `EMBODIMENTS`);
    tasks / default_suite / cameras come from the registry.
    """
    name: str = "robomind"

    def __init__(self, suite: str | None = None, task_id: int = 0):
        self.lf = _lf()
        emb = EMBODIMENTS[self.name]
        self.dataset = emb.name
        self.tasks = emb.tasks
        self.suite = suite or emb.default_suite
        if self.suite not in self.tasks:
            raise KeyError(
                f"{self.name}: unknown suite {self.suite!r}; available: {list(self.tasks)}")
        self.root = self.tasks[self.suite]

    def _set_root(self):
        self.lf.DATASETS[self.dataset]["root"] = self.root

    def pick_frames(self, episode: int, n: int) -> list[int]:
        self._set_root()
        return self.lf.pick_frames(episode, n, self.dataset)

    def get_frame(self, episode: int, frame: int) -> Frame:
        self._set_root()
        d = self.lf.extract_frame(self.dataset, episode, frame)
        return Frame(
            rgb=d["rgb"],
            state=np.asarray(d["state"], np.float32),
            instruction=d["instruction"],
            cams=list(d["cams"]),
            meta={"source": self.name, "dataset": self.dataset, "suite": self.suite,
                  "episode": episode, "frame": frame},
        )


class TienkungSource(_RoboMINDSource):
    name = "tienkung"


class Ur5eSource(_RoboMINDSource):
    name = "ur5e"


class Ur5eV3Source(_RoboMINDSource):
    name = "ur5e_v3"


class Ur5eDualSource(_RoboMINDSource):
    name = "ur5e_dual"


class AgilexJdSource(_RoboMINDSource):
    name = "agilex_jd"


class RealSourceV30Source(_RoboMINDSource):
    name = "realsource_v30"


class FrankaFr3Source(_RoboMINDSource):
    name = "franka_fr3"


class FrankaSingleSource(_RoboMINDSource):
    name = "franka_single"
