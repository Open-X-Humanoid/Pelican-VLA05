#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright 2026 The BaseVLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trimmed subset of LeRobot's ``datasets.utils`` for the inference release.

Only the lightweight, pure-Python dict/JSON helpers that the optimizer and
scheduler state (de)serialization depend on are kept here. The original module
also pulled in ``pandas`` / ``datasets`` / ``huggingface_hub`` for dataset
construction, none of which is needed to run inference.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dict by joining keys, e.g. ``{'a': {'b': 1}} -> {'a/b': 1}``."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Inverse of :func:`flatten_dict`."""
    out: dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    """Convert a (possibly nested) dict of tensors/arrays into JSON-native types."""
    serialized: dict[str, Any] = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            serialized[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], (int, float, list)):
            serialized[key] = value
        elif isinstance(value, np.generic):
            serialized[key] = value.item()
        elif isinstance(value, (int, float)):
            serialized[key] = value
        else:
            raise NotImplementedError(f"unsupported value {value!r} of type {type(value)}")
    return unflatten_dict(serialized)


def load_json(fpath: Path) -> Any:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
