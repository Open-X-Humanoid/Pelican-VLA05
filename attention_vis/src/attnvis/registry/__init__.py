"""attnvis pluggable registries.

- `embodiments`: single source of truth for dataset embodiments (arm layout,
  state dimensionality, camera roles).
- `models`: single source of truth for model environment, weights and capture
  technique.

The contracts in `core/types.py` and the rendering layer do not change; the
registries collect the metadata that varies per model / dataset.
"""
