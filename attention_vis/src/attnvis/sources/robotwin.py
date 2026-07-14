"""RobotwinSource — turn RoboTwin HDF5 episodes into attnvis Frames.

RoboTwin: one task directory holds many hdf5 files
(`clean-episode_*.hdf5`); one hdf5 = one episode. Per-episode contents:
  observations/images/{cam_high, cam_left_wrist, cam_right_wrist}:
      (T,) JPEG byte streams
  observations/qpos: (T, 14) float32 — left arm 7 (xyz + quat) + left grip 1
                     + right arm 7 + right grip 1
  action:            (T, 14) float32
  prompts:           (100,) bytes — instruction variants for the task

Three cameras (head + two wrists); 14D state. Images are cv2-decoded as BGR;
we convert to RGB. Orientation is already upright (unlike LIBERO).
"""
from __future__ import annotations

import glob
from pathlib import Path

import cv2
import h5py
import numpy as np

from attnvis.core.types import DataSource, Frame
from attnvis.config import DATA_ROOTS

ROBOTWIN_ROOT = str(DATA_ROOTS["robotwin"])
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


class RobotwinSource(DataSource):
    """RoboTwin HDF5 -> Frame. One RobotwinSource binds one task (= one
    directory of episode hdf5 files)."""

    name = "robotwin"

    def __init__(self, suite: str = "adjust_bottle", prompt_idx: int = 0):
        # suite = task name (reusing the cli's --suite flag);
        # prompt_idx = which prompt variant to serve as the instruction.
        self.task = suite
        self.prompt_idx = prompt_idx
        self.files = sorted(
            glob.glob(f"{ROBOTWIN_ROOT}/{suite}/clean-episode_*.hdf5"),
            key=lambda p: int(Path(p).stem.split("_")[-1]),
        )
        if not self.files:
            raise FileNotFoundError(f"no hdf5 in RoboTwin task: {ROBOTWIN_ROOT}/{suite}")

    def _h5(self, episode: int):
        """Open the episode file. Handles are cached and reused."""
        if not hasattr(self, "_cache"):
            self._cache = {}
        if episode not in self._cache:
            self._cache[episode] = h5py.File(self.files[episode], "r")
        return self._cache[episode]

    def pick_frames(self, episode: int, n: int) -> list[int]:
        h = self._h5(episode)
        L = h["observations/qpos"].shape[0]
        lo, hi = int(L * 0.05), int(L * 0.95)  # drop the first/last 5%
        return [round(lo + (hi - lo) * i / max(n - 1, 1)) for i in range(n)]

    def get_frame(self, episode: int, frame: int) -> Frame:
        h = self._h5(episode)

        rgb = {}
        for cam in CAMS:
            buf = h[f"observations/images/{cam}"][frame]
            arr = np.frombuffer(
                buf if isinstance(buf, (bytes, bytearray)) else buf.tobytes(),
                dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # (480, 640, 3) BGR uint8
            rgb[cam] = np.ascontiguousarray(img_bgr[..., ::-1])  # BGR -> RGB

        state = np.asarray(h["observations/qpos"][frame], dtype=np.float32)

        prompt = h["prompts"][self.prompt_idx]
        instruction = prompt.decode() if isinstance(prompt, (bytes, bytearray)) else str(prompt)

        return Frame(
            rgb=rgb,
            state=state,
            instruction=instruction,
            cams=CAMS,
            meta={"source": self.name, "task": self.task,
                  "episode": episode, "frame": frame, "file": self.files[episode]},
        )
