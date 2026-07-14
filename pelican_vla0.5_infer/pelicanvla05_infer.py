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
"""Public PelicanVLA05 inference entry point."""

from basevla_infer import BaseVLAInference, PelicanVLA05Inference, load_stats_json, main

__all__ = [
    "PelicanVLA05Inference",
    "load_stats_json",
    "BaseVLAInference",
]


if __name__ == "__main__":
    main()
