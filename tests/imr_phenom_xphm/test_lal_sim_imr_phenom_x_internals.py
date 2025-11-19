"""Unit tests for lal_sim_imr_phenom_x_internals.py."""

from __future__ import annotations

from typing import ClassVar

import jax
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals import (
    check_input_mode_array,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_create_mode_array,
    xlal_sim_inspiral_mode_array_activate_mode,
)


class TestCheckInputModeArray:
    """Tests for check_input_mode_array function."""

    max_l: ClassVar[int] = 5  # Use 5 for smaller arrays
    allowed_modes: ClassVar[list[tuple[int, int]]] = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]  # Only these are allowed

    def get_mode_index(self, ell, m, max_l):
        """Calculate the index for a mode."""
        return ell * ell + ell + m

    def create_mode_array_with_modes(self, modes_to_activate, max_l):
        """Create a mode array and activate specified modes."""
        mode_array = xlal_sim_inspiral_create_mode_array(max_l)
        for ell, m in modes_to_activate:
            mode_array = xlal_sim_inspiral_mode_array_activate_mode(mode_array, ell, m, max_l)
        return mode_array

    def test_no_mode_array(self):
        """Test that function returns True when mode_array is not present."""
        lal_params = {}
        err, result = check_input_mode_array(lal_params, self.max_l)
        err.throw()
        assert result is True

    def test_valid_modes(self):
        """Test that function returns True for allowed modes."""
        mode_array = self.create_mode_array_with_modes(self.allowed_modes, self.max_l)
        lal_params = {"mode_array": mode_array}
        err, result = check_input_mode_array(lal_params, self.max_l)
        err.throw()
        assert result is True

    def test_invalid_modes(self):
        """Test that function raises error for disallowed modes."""
        # Activate an invalid mode, e.g., (2, 0)
        invalid_modes = [(2, 0)]
        mode_array = self.create_mode_array_with_modes(self.allowed_modes + invalid_modes, self.max_l)
        lal_params = {"mode_array": mode_array}

        with pytest.raises(Exception, match="Invalid modes activated"):
            check_input_mode_array(lal_params, self.max_l)[0].throw()

    def test_mixed_valid_invalid(self):
        """Test that function raises error when both valid and invalid modes are active."""
        invalid_modes = [(3, 1)]  # (3,1) not allowed
        mode_array = self.create_mode_array_with_modes(self.allowed_modes[:2] + invalid_modes, self.max_l)
        lal_params = {"mode_array": mode_array}

        with pytest.raises(Exception, match="Invalid modes activated"):
            check_input_mode_array(lal_params, self.max_l)[0].throw()

    def test_negative_m_modes(self):
        """Test that negative m modes are handled (since |m| is checked)."""
        # (2, -2) should be allowed since (2,2) is
        mode_array = self.create_mode_array_with_modes([(2, -2)], self.max_l)
        lal_params = {"mode_array": mode_array}
        err, result = check_input_mode_array(lal_params, self.max_l)
        err.throw()
        assert result is True

    def test_jit_compatibility(self):
        """Test that the function is JIT-compatible."""
        # JIT compile the function
        jit_check = jax.jit(check_input_mode_array, static_argnames=["max_l"])

        # Test with no mode_array
        lal_params = {}
        err, result = jit_check(lal_params, self.max_l)
        err.throw()
        assert result

        # Test with valid modes
        mode_array = self.create_mode_array_with_modes([(2, 2)], self.max_l)
        lal_params = {"mode_array": mode_array}
        err, result = jit_check(lal_params, self.max_l)
        err.throw()
        assert result

        # Test with invalid modes (should raise)
        invalid_modes = [(2, 0)]
        mode_array = self.create_mode_array_with_modes(invalid_modes, self.max_l)
        lal_params = {"mode_array": mode_array}

        with pytest.raises(Exception, match="Invalid modes activated"):
            jit_check(lal_params, self.max_l)[0].throw()

    def test_max_l_parameter(self):
        """Test that max_l parameter affects the check range."""
        # With max_l=3, (4,4) should be invalid even if activated
        mode_array = xlal_sim_inspiral_create_mode_array(5)  # Create with larger size
        mode_array = xlal_sim_inspiral_mode_array_activate_mode(mode_array, 4, 4, 5)
        lal_params = {"mode_array": mode_array}

        # With max_l=3, it should not check up to 4
        err, result = check_input_mode_array(lal_params, max_l=3)
        err.throw()
        assert result is True  # Since 4 > 3, not checked


if __name__ == "__main__":
    pytest.main([__file__])
