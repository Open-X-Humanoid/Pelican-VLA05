"""LeRobot / RoboMIND single-frame reader (shared by all lerobot sources).

Reads three-camera RGB + state + instruction for one frame of one episode.
Config-driven: `DATASETS` (derived from `registry.embodiments.lerobot_datasets`)
picks the root / cameras / state layout for each dataset short name.

Read-only, no side effects; camera decoding via OpenCV + parquet via pyarrow.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import cv2
import pyarrow.parquet as pq

from attnvis.registry.embodiments import lerobot_datasets

DATASETS = lerobot_datasets()   # dataset name → {root, cameras, camera_keys, state_cols, [instruction_source]}
DEFAULT_DATASET = "tienkung"


def _cfg(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {list(DATASETS)}")
    return DATASETS[dataset]


def cams_of(dataset: str = DEFAULT_DATASET) -> list[str]:
    return list(_cfg(dataset)["cameras"])


def _info(root: str) -> dict:
    with open(f"{root}/meta/info.json") as f:
        return json.load(f)


def _first_parquet(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet at {pattern}")
    return files[0]


def _episodes(root: str) -> list[dict]:
    return pq.read_table(_first_parquet(f"{root}/meta/episodes/chunk-000/*.parquet")).to_pylist()


def episode_length(episode_idx: int, dataset: str = DEFAULT_DATASET) -> int:
    return int(_episodes(_cfg(dataset)["root"])[episode_idx]["length"])


def num_episodes(dataset: str = DEFAULT_DATASET) -> int:
    return len(_episodes(_cfg(dataset)["root"]))


def pick_frames(episode_idx: int, n: int, dataset: str = DEFAULT_DATASET) -> list[int]:
    """Uniform sampling inside the [5%, 95%] window of episode length."""
    L = episode_length(episode_idx, dataset)
    lo, hi = int(L * 0.05), int(L * 0.95)
    return [int(round(lo + (hi - lo) * i / max(n - 1, 1))) for i in range(n)]


def _decode_video_frame(mp4_path: str, frame_no: int) -> np.ndarray:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {mp4_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_no} @ {mp4_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_frame(dataset: str = DEFAULT_DATASET, episode_idx: int = 0, frame_idx: int = 0) -> dict:
    cfg = _cfg(dataset)
    root = cfg["root"]
    cams = cfg["cameras"]
    cam_keys = cfg["camera_keys"]
    info = _info(root)
    fps = info["fps"]
    e = _episodes(root)[episode_idx]
    if cfg.get("instruction_source") == "episode_tasks":
        _t = e.get("tasks")
        instruction = (_t[0] if isinstance(_t, list) and _t else "").strip() or "do the task"
    else:
        instruction = info.get("metadata.language_instruction", "").strip() or "do the task"
    assert 0 <= frame_idx < e["length"], f"frame_idx OOB (this ep has {e['length']} frames)"

    rgb = {}
    for c in cams:
        vk = f"videos/{cam_keys[c]}"
        from_ts = e[f"{vk}/from_timestamp"]
        fidx = e[f"{vk}/file_index"]
        cidx = e[f"{vk}/chunk_index"]
        # MP4s often chain multiple episodes back to back — from_timestamp
        # tells us where this episode starts within the file.
        frame_no = round(from_ts * fps) + frame_idx
        mp4 = f"{root}/videos/{cam_keys[c]}/chunk-{cidx:03d}/file-{fidx:03d}.mp4"
        rgb[c] = _decode_video_frame(mp4, frame_no)

    total_dim = sum(d for _, d in cfg["state_cols"])
    parquet = f"{root}/data/chunk-{e['data/chunk_index']:03d}/file-{e['data/file_index']:03d}.parquet"
    try:
        df = pq.read_table(parquet)
        row = e["dataset_from_index"] + frame_idx
        parts = []
        for col, dim in cfg["state_cols"]:
            try:
                v = np.atleast_1d(np.asarray(df.column(col)[row].as_py(), dtype=np.float32))
            except KeyError:
                v = np.zeros(dim, dtype=np.float32)
            if v.shape[0] < dim:
                v = np.pad(v, (0, dim - v.shape[0]))
            elif v.shape[0] > dim:
                v = v[:dim]
            parts.append(v)
        state = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
    except FileNotFoundError:
        # Dataset ships videos + episode metadata but no state parquet.
        # PelicanVLA adapter zero-pads state to its expected dim, so returning
        # zeros here keeps the rgb + instruction pipeline usable.
        state = np.zeros(total_dim, dtype=np.float32)

    return {"rgb": rgb, "state": state, "instruction": instruction, "cams": list(cams),
            "ep": episode_idx, "frame": frame_idx, "dataset": dataset}
