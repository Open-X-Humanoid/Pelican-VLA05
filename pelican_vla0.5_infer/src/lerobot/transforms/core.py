# Copyright 2026 The BaseVLA Authors.
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
"""Composable observation/action transforms used to turn raw robot samples into
model-ready batches.

This is a clean, inference-oriented reimplementation of the registry-style
transform pattern popularised by openpi (Apache-2.0). Only the transforms that
sit on the PelicanVLA05 inference path are kept here; training/dataset-hydration
helpers have been intentionally dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import abc
import draccus

# A sample flowing through the pipeline is just a mutable string-keyed mapping.
DataDict = dict[str, Any]


class DataTransformFn(draccus.ChoiceRegistry, abc.ABC):
    """Base class for a single in-place-ish transform over a sample dict."""

    @abc.abstractmethod
    def __call__(self, data: DataDict) -> DataDict:
        ...


@dataclass(frozen=True)
class TransformGroup:
    """A pair of ordered transform lists: one for inputs, one for outputs."""

    inputs: list[DataTransformFn] = field(default_factory=list)
    outputs: list[DataTransformFn] = field(default_factory=list)

    def push(
        self,
        *,
        inputs: list[DataTransformFn] | None = None,
        outputs: list[DataTransformFn] | None = None,
    ) -> "TransformGroup":
        """Return a new group with extra input transforms appended and extra
        output transforms prepended (outputs run in reverse of inputs)."""
        return TransformGroup(
            inputs=[*self.inputs, *(inputs or [])],
            outputs=[*(outputs or []), *self.outputs],
        )
