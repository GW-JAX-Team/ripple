""" "Tests for IMRPhenomXPrecessionDataClass."""

from __future__ import annotations

import dataclasses

import jax
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
    PhenomXPalphaMRD,
)


class TestIMRPhenomXPrecessionDataClass:
    """Test suite for IMRPhenomXPrecessionDataClass."""

    def test_initialization(self, precession_data_class_sample):
        """Test that the dataclass can be initialized with valid data."""
        prec_data = IMRPhenomXPrecessionDataClass(**precession_data_class_sample)

        # Check a few fields to ensure they are set correctly
        assert prec_data.imr_phenom_x_prec_version == precession_data_class_sample["imr_phenom_x_prec_version"]
        assert prec_data.imr_phenom_x_return_co_prec == precession_data_class_sample["imr_phenom_x_return_co_prec"]
        assert prec_data.debug_prec == precession_data_class_sample["debug_prec"]
        assert prec_data.a1 == precession_data_class_sample["a1"]
        assert prec_data.cexp_i_alpha == precession_data_class_sample["cexp_i_alpha"]

    def test_immutability(self, precession_data_class_sample):
        """Test that the dataclass is immutable (frozen)."""
        prec_data = IMRPhenomXPrecessionDataClass(**precession_data_class_sample)

        with pytest.raises(dataclasses.FrozenInstanceError):
            prec_data.IMRPhenomXPrecVersion = 100

    def test_jit_compatibility(self, precession_data_class_sample):
        """Test that the dataclass can be used in JIT-compiled functions."""

        @jax.jit
        def get_version(data):
            return data.imr_phenom_x_prec_version

        prec_data = IMRPhenomXPrecessionDataClass(**precession_data_class_sample)
        version = get_version(prec_data)
        assert version == precession_data_class_sample["imr_phenom_x_prec_version"]


class TestPhenomXPalphaMRD:
    """Test the PhenomXPalphaMRD dataclass."""

    def test_initialization(self):
        """Test basic initialization with default values."""
        params = PhenomXPalphaMRD(a_rd=1.0, b_rd=2.0, c_rd=3.0)
        assert params.a_rd == 1.0
        assert params.b_rd == 2.0
        assert params.c_rd == 3.0

    def test_immutability(self):
        """Test that the dataclass is immutable (frozen)."""
        params = PhenomXPalphaMRD(a_rd=1.0, b_rd=2.0, c_rd=3.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            params.a_rd = 4.0  # Should raise error

    def test_equality(self):
        """Test equality comparison."""
        params1 = PhenomXPalphaMRD(a_rd=1.0, b_rd=2.0, c_rd=3.0)
        params2 = PhenomXPalphaMRD(a_rd=1.0, b_rd=2.0, c_rd=3.0)
        params3 = PhenomXPalphaMRD(a_rd=1.1, b_rd=2.0, c_rd=3.0)

        assert params1 == params2
        assert params1 != params3

    def test_hashability(self):
        """Test that instances are hashable (for use in sets/dicts)."""
        params = PhenomXPalphaMRD(a_rd=1.0, b_rd=2.0, c_rd=3.0)
        # Should not raise an error
        hash_value = hash(params)
        assert isinstance(hash_value, int)

    def test_field_types(self):
        """Test that fields are floats."""
        params = PhenomXPalphaMRD(a_rd=1.0, b_rd=2.0, c_rd=3.0)
        assert isinstance(params.a_rd, float)
        assert isinstance(params.b_rd, float)
        assert isinstance(params.c_rd, float)


if __name__ == "__main__":
    pytest.main([__file__])
