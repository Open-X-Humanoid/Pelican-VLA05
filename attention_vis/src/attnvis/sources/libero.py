"""LiberoSource — turn LIBERO HDF5 demos into attnvis Frames.

LIBERO: one hdf5 file = one task; each file holds ~50 demos (= episodes).
Per-demo obs: agentview_rgb / eye_in_hand_rgb (T, 128, 128, 3, uint8),
joint_states (7), gripper_states (2). Two cameras (agentview = third person;
wrist); instruction is derived from the task filename.

Note: LIBERO images are stored upside-down by convention
(macros_image_convention); we flip them on read.
"""
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import numpy as np

from attnvis.core.types import DataSource, Frame
from attnvis.config import DATA_ROOTS

LIBERO_ROOT = str(DATA_ROOTS["libero"])


class LiberoSource(DataSource):
    """LIBERO HDF5 -> Frame. One LiberoSource binds one suite + one task
    (= one hdf5 file)."""

    name = "libero"

    def __init__(self, suite: str = "libero_spatial", task_id: int = 0):
        files = sorted(glob.glob(f"{LIBERO_ROOT}/{suite}/*.hdf5"))
        if not files:
            raise FileNotFoundError(f"no hdf5 in LIBERO suite: {LIBERO_ROOT}/{suite}")
        self.path = files[task_id]
        # Filename "pick_up_the_black_bowl..._demo.hdf5" -> instruction
        # "pick up the black bowl ..."
        stem = Path(self.path).stem.replace("_demo", "")
        self.instruction = stem.replace("_", " ")
        self.suite, self.task_id = suite, task_id

    def _demo(self, episode: int):
        """Return the h5py group for the requested demo.

        Lazy: the file is opened once on first call and the handle is reused
        (avoiding open-and-forget handle leaks).
        """
        if not hasattr(self, "_h"):
            self._h = h5py.File(self.path, "r")
        h = self._h
        demo_keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))
        return h["data"][demo_keys[episode]]

    def pick_frames(self, episode: int, n: int) -> list[int]:
        """Uniformly sample n frame indices across the middle 90% of the demo."""
        demo = self._demo(episode)
        L = demo["actions"].shape[0]
        lo, hi = int(L * 0.05), int(L * 0.95)
        return [round(lo + (hi - lo) * i / (n - 1)) for i in range(n)]

    def get_frame(self, episode: int, frame: int) -> Frame:
        demo = self._demo(episode)
        obs = demo["obs"]

        # Two cameras, flipped back upright.
        agentview = obs["agentview_rgb"][frame][::-1]
        wrist = obs["eye_in_hand_rgb"][frame][::-1]

        # State: 7D joints + 1D gripper (first channel of the 2D gripper pair) = 8D.
        joint = obs["joint_states"][frame]
        grip = obs["gripper_states"][frame][:1]
        state = np.concatenate([joint, grip]).astype(np.float32)

        return Frame(
            rgb={"agentview": agentview, "wrist": wrist},
            state=state,
            instruction=self.instruction,
            cams=["agentview", "wrist"],
            meta={"source": self.name, "suite": self.suite, "task_id": self.task_id,
                  "episode": episode, "frame": frame},
        )
