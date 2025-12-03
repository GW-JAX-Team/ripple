"""Unit tests for dataclasses defined in lal_sim_imr_phenom_x_internals_dataclass.py."""

from __future__ import annotations

import dataclasses

import jax
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXUsefulPowersDataClass,
    IMRPhenomXWaveformDataClass,
)


class TestIMRPhenomXWaveformDataClass:
    """Test suite for IMRPhenomXWaveformDataClass."""

    def test_instantiation_and_field_access(self, waveform_data_class_sample):
        """Test that the dataclass can be instantiated and fields accessed."""
        p_wf = IMRPhenomXWaveformDataClass(**waveform_data_class_sample)
        assert p_wf.m1 == 1.0
        assert p_wf.eta == 0.25
        assert p_wf.lal_params == {}
        assert isinstance(p_wf.m1_si, float)

    def test_immutability(self, waveform_data_class_sample):
        """Test that the dataclass is immutable (if frozen=True)."""
        p_wf = IMRPhenomXWaveformDataClass(**waveform_data_class_sample)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p_wf.m1 = 2.0  # Should fail if frozen

    def test_functional_update(self, waveform_data_class_sample):
        """Test updating fields with dataclasses.replace()."""
        p_wf = IMRPhenomXWaveformDataClass(**waveform_data_class_sample)
        updated = dataclasses.replace(p_wf, m1=2.0, eta=0.2)
        assert updated.m1 == 2.0
        assert updated.eta == 0.2
        assert p_wf.m1 == 1.0  # Original unchanged

    def test_jax_tree_operations(self, waveform_data_class_sample):
        """Test JAX tree flattening and unflattening."""
        p_wf = IMRPhenomXWaveformDataClass(**waveform_data_class_sample)
        flat, treedef = jax.tree_util.tree_flatten(p_wf)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
        assert reconstructed == p_wf

    def test_jit_compatibility(self, waveform_data_class_sample):
        """Test that the dataclass works with jax.jit."""

        @jax.jit
        def add_masses(p_wf):
            return p_wf.m1 + p_wf.m2

        p_wf = IMRPhenomXWaveformDataClass(**waveform_data_class_sample)
        result = add_masses(p_wf)
        assert result == 2.0

    def test_equality_and_copy(self, waveform_data_class_sample):
        """Test equality and copying."""
        p_wf1 = IMRPhenomXWaveformDataClass(**waveform_data_class_sample)
        p_wf2 = IMRPhenomXWaveformDataClass(**waveform_data_class_sample)
        assert p_wf1 == p_wf2
        copied = dataclasses.replace(p_wf1)  # Shallow copy
        assert copied == p_wf1
        assert copied is not p_wf1  # Different object


class TestIMRPhenomXUsefulPowersDataClass:
    """Test suite for IMRPhenomXUsefulPowersDataClass."""

    def test_instantiation_and_field_access(self, waveform_useful_powers_sample):
        """Test that the dataclass can be instantiated and fields accessed."""
        p_pow = IMRPhenomXUsefulPowersDataClass(**waveform_useful_powers_sample)
        assert p_pow.seven_sixths == 1.0
        assert p_pow.one_third == 1.0
        assert isinstance(p_pow.five, float)

    def test_immutability(self, waveform_useful_powers_sample):
        """Test that the dataclass is immutable (if frozen=True)."""
        p_pow = IMRPhenomXUsefulPowersDataClass(**waveform_useful_powers_sample)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p_pow.seven_sixths = 2.0  # Should fail if frozen

    def test_functional_update(self, waveform_useful_powers_sample):
        """Test updating fields with dataclasses.replace()."""
        p_pow = IMRPhenomXUsefulPowersDataClass(**waveform_useful_powers_sample)
        updated = dataclasses.replace(p_pow, seven_sixths=2.0, one_third=0.5)
        assert updated.seven_sixths == 2.0
        assert updated.one_third == 0.5
        assert p_pow.seven_sixths == 1.0  # Original unchanged

    def test_jax_tree_operations(self, waveform_useful_powers_sample):
        """Test JAX tree flattening and unflattening."""
        p_pow = IMRPhenomXUsefulPowersDataClass(**waveform_useful_powers_sample)
        flat, treedef = jax.tree_util.tree_flatten(p_pow)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
        assert reconstructed == p_pow

    def test_jit_compatibility(self, waveform_useful_powers_sample):
        """Test that the dataclass works with jax.jit."""

        @jax.jit
        def sum_powers(p_pow):
            return p_pow.seven_sixths + p_pow.one_third

        p_pow = IMRPhenomXUsefulPowersDataClass(**waveform_useful_powers_sample)
        result = sum_powers(p_pow)
        assert result == 2.0

    def test_equality_and_copy(self, waveform_useful_powers_sample):
        """Test equality and copying."""
        p_pow1 = IMRPhenomXUsefulPowersDataClass(**waveform_useful_powers_sample)
        p_pow2 = IMRPhenomXUsefulPowersDataClass(**waveform_useful_powers_sample)
        assert p_pow1 == p_pow2
        copied = dataclasses.replace(p_pow1)  # Shallow copy
        assert copied == p_pow1
        assert copied is not p_pow1  # Different object


if __name__ == "__main__":
    pytest.main([__file__])
