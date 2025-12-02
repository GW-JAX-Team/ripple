"""Unit tests for dataclasses defined in lal_sim_imr_phenom_x_internals_dataclass.py."""

from __future__ import annotations

import dataclasses

import jax
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXUsefulPowersDataClass,
    IMRPhenomXWaveformDataClass,
    IMRPhenomXPrecessionDataClass
)

from data_class_sample_data import (
    WAVEFORM_DATA_CLASS_SAMPLE,
    USEFUL_POWERS_SAMPLE,
    PRECESSION_DATA_CLASS_SAMPLE
)


class TestIMRPhenomXWaveformDataClass:
    """Test suite for IMRPhenomXWaveformDataClass."""

    @pytest.fixture
    def sample_data(self):
        """Fixture for sample data to create the dataclass."""
        return WAVEFORM_DATA_CLASS_SAMPLE
    
    def test_instantiation_and_field_access(self, sample_data):
        """Test that the dataclass can be instantiated and fields accessed."""
        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        assert p_wf.m1 == 1.0
        assert p_wf.eta == 0.25
        assert p_wf.lal_params == {}
        assert isinstance(p_wf.m1_si, float)

    def test_immutability(self, sample_data):
        """Test that the dataclass is immutable (if frozen=True)."""
        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p_wf.m1 = 2.0  # Should fail if frozen

    def test_functional_update(self, sample_data):
        """Test updating fields with dataclasses.replace()."""
        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        updated = dataclasses.replace(p_wf, m1=2.0, eta=0.2)
        assert updated.m1 == 2.0
        assert updated.eta == 0.2
        assert p_wf.m1 == 1.0  # Original unchanged

    def test_jax_tree_operations(self, sample_data):
        """Test JAX tree flattening and unflattening."""
        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        flat, treedef = jax.tree_util.tree_flatten(p_wf)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
        assert reconstructed == p_wf

    def test_jit_compatibility(self, sample_data):
        """Test that the dataclass works with jax.jit."""

        @jax.jit
        def add_masses(p_wf):
            return p_wf.m1 + p_wf.m2

        p_wf = IMRPhenomXWaveformDataClass(**sample_data)
        result = add_masses(p_wf)
        assert result == 2.0

    def test_equality_and_copy(self, sample_data):
        """Test equality and copying."""
        p_wf1 = IMRPhenomXWaveformDataClass(**sample_data)
        p_wf2 = IMRPhenomXWaveformDataClass(**sample_data)
        assert p_wf1 == p_wf2
        copied = dataclasses.replace(p_wf1)  # Shallow copy
        assert copied == p_wf1
        assert copied is not p_wf1  # Different object


class TestIMRPhenomXUsefulPowersDataClass:
    """Test suite for IMRPhenomXUsefulPowersDataClass."""

    @pytest.fixture
    def sample_powers_data(self):
        """Fixture for sample data to create the powers dataclass."""
        return USEFUL_POWERS_SAMPLE

    def test_instantiation_and_field_access(self, sample_powers_data):
        """Test that the dataclass can be instantiated and fields accessed."""
        p_pow = IMRPhenomXUsefulPowersDataClass(**sample_powers_data)
        assert p_pow.seven_sixths == 1.0
        assert p_pow.one_third == 1.0
        assert isinstance(p_pow.five, float)

    def test_immutability(self, sample_powers_data):
        """Test that the dataclass is immutable (if frozen=True)."""
        p_pow = IMRPhenomXUsefulPowersDataClass(**sample_powers_data)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p_pow.seven_sixths = 2.0  # Should fail if frozen

    def test_functional_update(self, sample_powers_data):
        """Test updating fields with dataclasses.replace()."""
        p_pow = IMRPhenomXUsefulPowersDataClass(**sample_powers_data)
        updated = dataclasses.replace(p_pow, seven_sixths=2.0, one_third=0.5)
        assert updated.seven_sixths == 2.0
        assert updated.one_third == 0.5
        assert p_pow.seven_sixths == 1.0  # Original unchanged

    def test_jax_tree_operations(self, sample_powers_data):
        """Test JAX tree flattening and unflattening."""
        p_pow = IMRPhenomXUsefulPowersDataClass(**sample_powers_data)
        flat, treedef = jax.tree_util.tree_flatten(p_pow)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
        assert reconstructed == p_pow

    def test_jit_compatibility(self, sample_powers_data):
        """Test that the dataclass works with jax.jit."""

        @jax.jit
        def sum_powers(p_pow):
            return p_pow.seven_sixths + p_pow.one_third

        p_pow = IMRPhenomXUsefulPowersDataClass(**sample_powers_data)
        result = sum_powers(p_pow)
        assert result == 2.0

    def test_equality_and_copy(self, sample_powers_data):
        """Test equality and copying."""
        p_pow1 = IMRPhenomXUsefulPowersDataClass(**sample_powers_data)
        p_pow2 = IMRPhenomXUsefulPowersDataClass(**sample_powers_data)
        assert p_pow1 == p_pow2
        copied = dataclasses.replace(p_pow1)  # Shallow copy
        assert copied == p_pow1
        assert copied is not p_pow1  # Different object


class TestIMRPhenomXPrecessionDataClass:
    """Test suite for IMRPhenomXPrecessionDataClass."""

    @pytest.fixture
    def sample_data(self):
        """Fixture for sample data to create the dataclass."""
        return PRECESSION_DATA_CLASS_SAMPLE

    def test_initialization(self, sample_data):
        """Test that the dataclass can be initialized with valid data."""
        prec_data = IMRPhenomXPrecessionDataClass(**sample_data)

        # Check a few fields to ensure they are set correctly
        assert prec_data.IMRPhenomXPrecVersion == sample_data["IMRPhenomXPrecVersion"]
        assert prec_data.IMRPhenomXReturnCoPrec == sample_data["IMRPhenomXReturnCoPrec"]
        assert prec_data.debug_prec == sample_data["debug_prec"]
        assert prec_data.A1 == sample_data["A1"]
        assert prec_data.cexp_i_alpha == sample_data["cexp_i_alpha"]

    def test_immutability(self, sample_data):
        """Test that the dataclass is immutable (frozen)."""
        prec_data = IMRPhenomXPrecessionDataClass(**sample_data)

        with pytest.raises(dataclasses.FrozenInstanceError):
            prec_data.IMRPhenomXPrecVersion = 100

    def test_jit_compatibility(self, sample_data):
        """Test that the dataclass can be used in JIT-compiled functions."""

        @jax.jit
        def get_version(data):
            return data.IMRPhenomXPrecVersion

        prec_data = IMRPhenomXPrecessionDataClass(**sample_data)
        version = get_version(prec_data)
        assert version == sample_data["IMRPhenomXPrecVersion"]


if __name__ == "__main__":
    pytest.main([__file__])