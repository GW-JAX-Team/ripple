"""Unit tests for imr_phenom_xphm_setup_mode_array function."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_xphm import (
    imr_phenom_xphm_setup_mode_array,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_create_mode_array,
    xlal_sim_inspiral_mode_array_activate_mode,
    xlal_sim_inspiral_mode_array_is_mode_active,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import (
    IMRPhenomXPHMParameterDataClass,
)


class TestImrPhenomXphmSetupModeArray:
    """Tests for imr_phenom_xphm_setup_mode_array function."""

    @pytest.fixture
    def default_params(self) -> IMRPhenomXPHMParameterDataClass:
        """Create default parameter dataclass for testing."""
        return IMRPhenomXPHMParameterDataClass()

    @pytest.fixture
    def custom_mode_array(self) -> jnp.ndarray:
        """Create a custom mode array with some modes activated."""
        mode_array = xlal_sim_inspiral_create_mode_array()
        # Activate some custom modes (not the default ones)
        mode_array = xlal_sim_inspiral_mode_array_activate_mode(mode_array, 2, 0)
        mode_array = xlal_sim_inspiral_mode_array_activate_mode(mode_array, 3, 1)
        return mode_array

    def test_setup_default_modes_when_none(self, default_params):
        """Test that default modes are set up when mode_array is None."""
        err, result = imr_phenom_xphm_setup_mode_array(default_params)
        err.throw()  # Check for errors

        # Should have created a mode array
        assert result.mode_array is not None
        assert isinstance(result.mode_array, jnp.ndarray)
        assert result.mode_array.dtype == bool

        # Check that default modes are activated
        # Default modes: (2,2), (2,1), (3,3), (3,2), (4,4), and their negative m counterparts
        expected_modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (2, -2), (2, -1), (3, -3), (3, -2), (4, -4)]

        for ell, m in expected_modes:
            assert xlal_sim_inspiral_mode_array_is_mode_active(
                result.mode_array, ell, m
            ), f"Mode ({ell}, {m}) should be active"

    def test_return_unchanged_when_mode_array_exists(self, default_params, custom_mode_array):
        """Test that function returns unchanged when mode_array already exists."""
        params_with_modes = dataclasses.replace(default_params, mode_array=custom_mode_array)

        err, result = imr_phenom_xphm_setup_mode_array(params_with_modes)
        err.throw()  # Check for errors

        # Should return the same object with same mode_array
        assert result is not params_with_modes  # dataclasses.replace creates new instance
        assert jnp.array_equal(result.mode_array, custom_mode_array)

        # Verify custom modes are still active
        assert xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 2, 0)
        assert xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 3, 1)

        # Verify default modes are NOT active (since we used custom array)
        assert not xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 2, 2)
        assert not xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 3, 3)

    def test_jax_jit_compatibility_none_mode_array(self, default_params):
        """Test that function is JAX JIT compatible when mode_array is None."""
        jit_setup_mode_array = jax.jit(imr_phenom_xphm_setup_mode_array)

        # Should not raise any errors
        err, result = jit_setup_mode_array(default_params)
        err.throw()  # Check for errors

        # Should have created default modes
        assert result.mode_array is not None
        assert xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 2, 2)
        assert xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 3, 3)

    def test_jax_jit_compatibility_existing_mode_array(self, default_params, custom_mode_array):
        """Test that function is JAX JIT compatible when mode_array already exists."""
        jit_setup_mode_array = jax.jit(imr_phenom_xphm_setup_mode_array)
        params_with_modes = dataclasses.replace(default_params, mode_array=custom_mode_array)

        # Should not raise any errors
        err, result = jit_setup_mode_array(params_with_modes)
        err.throw()  # Check for errors

        # Should preserve custom modes
        assert jnp.array_equal(result.mode_array, custom_mode_array)
        assert xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 2, 0)
        assert xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 3, 1)

    def test_all_default_modes_activated(self, default_params):
        """Test that all expected default modes are properly activated."""
        err, result = imr_phenom_xphm_setup_mode_array(default_params)
        err.throw()  # Check for errors

        # All expected default modes should be active
        default_mode_list = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (2, -2), (2, -1), (3, -3), (3, -2), (4, -4)]

        for ell, m in default_mode_list:
            assert xlal_sim_inspiral_mode_array_is_mode_active(
                result.mode_array, ell, m
            ), f"Default mode ({ell}, {m}) should be active"

        # Check that some non-default modes are NOT active
        assert not xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 2, 0)
        assert not xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 3, 1)
        assert not xlal_sim_inspiral_mode_array_is_mode_active(result.mode_array, 4, 3)

    def test_mode_array_size(self, default_params):
        """Test that the created mode array has the correct size."""
        _error, result = imr_phenom_xphm_setup_mode_array(default_params)

        # Default max_l is 8, so max_modes = 8^2 + 2*8 + 1 = 64 + 16 + 1 = 81
        expected_size = 8**2 + 2 * 8 + 1  # 81
        assert result.mode_array.shape[0] == expected_size

    def test_immutability(self, default_params):
        """Test that the function doesn't modify the input dataclass in place."""
        original_mode_array = default_params.mode_array

        err, result = imr_phenom_xphm_setup_mode_array(default_params)
        err.throw()  # Check for errors

        # Original should be unchanged
        assert default_params.mode_array is original_mode_array
        assert result is not default_params  # Should be a new instance</content>
