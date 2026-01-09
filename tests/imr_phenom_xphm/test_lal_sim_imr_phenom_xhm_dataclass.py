"""Unit tests for QNMFits dataclass defined in lal_sim_imr_phenom_xhm_dataclass.py."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_xhm_dataclass import QNMFits


class TestQNMFits:
    """Test suite for QNMFits dataclass."""

    def test_instantiation_and_field_access(self):
        """Test that the dataclass can be instantiated and fields accessed."""
        f_ring = jnp.array([1.0, 2.0, 3.0])
        f_damp = jnp.array([0.1, 0.2, 0.3])
        qnm = QNMFits(f_ring_lm=f_ring, f_damp_lm=f_damp)
        assert jnp.allclose(qnm.f_ring_lm, f_ring)
        assert jnp.allclose(qnm.f_damp_lm, f_damp)

    def test_immutability(self):
        """Test that the dataclass is immutable (frozen=True)."""
        f_ring = jnp.array([1.0, 2.0])
        f_damp = jnp.array([0.1, 0.2])
        qnm = QNMFits(f_ring_lm=f_ring, f_damp_lm=f_damp)
        with pytest.raises(dataclasses.FrozenInstanceError):
            qnm.f_ring_lm = jnp.array([4.0, 5.0])  # Should fail if frozen

    def test_functional_update(self):
        """Test updating fields with dataclasses.replace()."""
        f_ring = jnp.array([1.0, 2.0])
        f_damp = jnp.array([0.1, 0.2])
        qnm = QNMFits(f_ring_lm=f_ring, f_damp_lm=f_damp)
        new_ring = jnp.array([3.0, 4.0])
        updated = dataclasses.replace(qnm, f_ring_lm=new_ring)
        assert jnp.allclose(updated.f_ring_lm, new_ring)
        assert jnp.allclose(updated.f_damp_lm, f_damp)  # Original unchanged
        assert jnp.allclose(qnm.f_ring_lm, f_ring)  # Original unchanged

    def test_jax_tree_operations(self):
        """Test JAX tree flattening and unflattening."""
        f_ring = jnp.array([1.0, 2.0])
        f_damp = jnp.array([0.1, 0.2])
        qnm = QNMFits(f_ring_lm=f_ring, f_damp_lm=f_damp)
        flat, treedef = jax.tree_util.tree_flatten(qnm)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
        assert reconstructed == qnm

    def test_jit_compatibility(self):
        """Test that the dataclass works with jax.jit."""

        @jax.jit
        def sum_frequencies(qnm):
            return jnp.sum(qnm.f_ring_lm) + jnp.sum(qnm.f_damp_lm)

        f_ring = jnp.array([1.0, 2.0])
        f_damp = jnp.array([0.1, 0.2])
        qnm = QNMFits(f_ring_lm=f_ring, f_damp_lm=f_damp)
        result = sum_frequencies(qnm)
        expected = 1.0 + 2.0 + 0.1 + 0.2
        assert jnp.allclose(result, expected)

    def test_equality_and_copy(self):
        """Test equality and copying."""
        f_ring = jnp.array([1.0, 2.0])
        f_damp = jnp.array([0.1, 0.2])
        qnm1 = QNMFits(f_ring_lm=f_ring, f_damp_lm=f_damp)
        qnm2 = QNMFits(f_ring_lm=f_ring, f_damp_lm=f_damp)
        assert qnm1 == qnm2
        copied = dataclasses.replace(qnm1)  # Shallow copy
        assert copied == qnm1
        assert copied is not qnm1  # Different object
