"""Core contracts of attnvis — the three-layer interface.

The whole architecture decouples what varies (data sources, models) from what
does not (rendering, comparison) via three types:

    DataSource ── Frame ─▶ ModelAdapter ── AttnResult ─▶ Core (render / compare)

Add a data source = implement DataSource. Add a model = implement
ModelAdapter. Core never changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """One observation frame — output of a DataSource, input to a ModelAdapter.

    Every data source produces this same structure so adapters do not have
    to care where the data came from.
    """
    rgb: dict[str, np.ndarray]   # {camera short name: HWC uint8 image}
    state: np.ndarray            # robot state vector (float32); dimensionality
                                 # depends on the data source
    instruction: str             # task instruction text
    cams: list[str]              # camera order (e.g. ["agentview","wrist"]
                                 # or ["cam_high","cam_left_wrist","cam_right_wrist"])
    meta: dict = field(default_factory=dict)  # source/episode/frame — records only


@dataclass
class AttnResult:
    """A model's attention result for one frame — output of a ModelAdapter,
    input to Core.

    Keys of `heat` are a subset of Frame.cams (some models do not cover
    every camera; that is fine).
    """
    heat: dict[str, np.ndarray]                       # {camera: HWC float32 action→image heatmap}
    heat_grounding: dict[str, np.ndarray] | None = None  # {camera: text→image heatmap}, optional
    raw: dict = field(default_factory=dict)           # raw attention + token layout for reuse


class DataSource(ABC):
    """DataSource contract: turn some data corpus (LIBERO / RoboTwin / ...)
    into Frames.

    Add a new data source = subclass this and implement the two methods.
    Core and adapters do not change.
    """

    name: str = "datasource"

    @abstractmethod
    def get_frame(self, episode: int, frame: int) -> Frame:
        """Return the requested episode's specified frame as a Frame."""

    @abstractmethod
    def pick_frames(self, episode: int, n: int) -> list[int]:
        """Uniformly sample n frame indices within the episode's length."""


class ModelAdapter(ABC):
    """ModelAdapter contract: wrap one VLA into a black box that consumes a
    Frame and produces an AttnResult.

    The adapter hides three model-specific concerns: (1) the attention
    capture path, (2) token / state layout, (3) single-hop vs two-hop
    composition. Add a new model = subclass this and implement load() +
    attention(). Core never changes.
    """

    #: Model short name; used to name output files and label comparison
    #: columns. Subclasses override.
    name: str = "model"

    @abstractmethod
    def load(self) -> None:
        """Load weights onto GPU once.

        Split out from __init__/attention because loading is expensive
        (multi-GB weights, tens of seconds) and we only want to pay it once
        per process. Contract: attention() must not be called before load();
        an unloaded adapter must fail cleanly if attention() is called.
        """

    @abstractmethod
    def attention(self, frame: Frame) -> AttnResult:
        """Forward + capture attention for one frame -> AttnResult (one
        action→image heatmap per camera).

        The adapter encapsulates the model's capture hook, token layout, and
        single- vs two-hop composition. Contract: the returned heat's camera
        keys must be a subset of frame.cams; a missing camera is tolerated
        by Core.
        """
