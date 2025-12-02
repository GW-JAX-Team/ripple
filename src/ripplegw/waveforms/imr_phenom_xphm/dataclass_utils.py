"""Utility functions for dataclass registration with JAX tree utilities."""

from __future__ import annotations

import dataclasses

import jax
import jax.tree_util


def _register_dataclass(cls):
    """Register a dataclass with JAX tree utilities (version-agnostic)."""
    # Get all field names from the dataclass
    field_names = [f.name for f in dataclasses.fields(cls)]

    def flatten_fn(obj):
        values = tuple(getattr(obj, name) for name in field_names)
        return values, field_names

    def unflatten_fn(field_names, values):
        return cls(**dict(zip(field_names, values)))

    jax.tree_util.register_pytree_node(cls, flatten_fn, unflatten_fn)
    return cls
