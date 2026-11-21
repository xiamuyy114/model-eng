"""Tensor-shape and config-facing helpers for the unified data path."""

from __future__ import annotations

from segtrue.preprocess.registry import prior_component_names


def input_channels_for_preprocess(name: str) -> int:
    return 3 + len(prior_component_names(name))
