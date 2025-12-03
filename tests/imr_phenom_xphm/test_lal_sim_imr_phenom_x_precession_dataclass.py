""" "Tests for IMRPhenomXPrecessionDataClass."""

from __future__ import annotations

import dataclasses

import jax
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
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


if __name__ == "__main__":
    pytest.main([__file__])
