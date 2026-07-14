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
"""Canonical PelicanVLA05 policy API.

The implementation remains in the legacy ``basevla_4B`` module so existing
imports and checkpoint-loading integrations continue to work.
"""

from lerobot.policies.basevla_4B.configuration_basevla import PelicanVLA05Config
from lerobot.policies.basevla_4B.modeling_basevla import PelicanVLA05Policy
from lerobot.policies.basevla_4B.transform_basevla import PelicanVLA05ProcessorTransformFn

__all__ = [
    "PelicanVLA05Config",
    "PelicanVLA05Policy",
    "PelicanVLA05ProcessorTransformFn",
]
