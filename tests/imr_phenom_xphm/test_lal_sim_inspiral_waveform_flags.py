"""Unit tests for lal_sim_inspiral_waveform_flags.py, cross-checking with LAL functions."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

try:
    import lalsimulation as lalsim

    LALSIM_AVAILABLE = True
except ImportError:
    LALSIM_AVAILABLE = False

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_create_mode_array,
    xlal_sim_inspiral_mode_array_activate_mode,
    xlal_sim_inspiral_mode_array_deactivate_mode,
    xlal_sim_inspiral_mode_array_is_mode_active,
)


@pytest.mark.skipif(not LALSIM_AVAILABLE, reason="lalsimulation not available")
class TestModeArrayCrossCheck:
    """Cross-check JAX mode array functions with LAL equivalents."""

    def setup_method(self):
        """Setup method to initialize max_l."""
        self.max_l = 8  # pylint: disable=attribute-defined-outside-init

    def lal_mode_array_to_jax_bool_array(self, lal_modes):
        """Convert LAL mode array (bit-packed string) to JAX bool array."""
        bool_list = []
        for l_index in range(self.max_l + 1):
            for m in range(-l_index, l_index + 1):
                if lalsim.SimInspiralModeArrayIsModeActive(lal_modes, l_index, m):
                    bool_list.append(True)
                else:
                    bool_list.append(False)
        return jnp.array(bool_list, dtype=bool)

    def test_create_mode_array(self):
        """Test create_mode_array matches LAL's empty mode array."""
        lal_modes = lalsim.SimInspiralCreateModeArray()
        # JAX
        jax_modes = xlal_sim_inspiral_create_mode_array(self.max_l)

        # Convert LAL to JAX bool array
        expected_jax = self.lal_mode_array_to_jax_bool_array(lal_modes)

        # Check all inactive
        assert jnp.array_equal(jax_modes, expected_jax)

    def test_activate_mode(self):
        """Test activate_mode matches LAL's activation."""
        # Start with empty
        lal_modes = lalsim.SimInspiralCreateModeArray()
        jax_modes = xlal_sim_inspiral_create_mode_array(self.max_l)

        # Test modes to activate
        test_modes = [(0, 0), (1, -1), (2, 2), (3, 0), (4, -4)]

        for ell, m in test_modes:
            # LAL: Activate
            lalsim.SimInspiralModeArrayActivateMode(lal_modes, ell, m)

            # JAX: Activate
            jax_modes = xlal_sim_inspiral_mode_array_activate_mode(jax_modes, ell, m, self.max_l)

            # Convert LAL to JAX bool array
            expected_jax = self.lal_mode_array_to_jax_bool_array(lal_modes)

            # Check match
            assert jnp.array_equal(jax_modes, expected_jax), f"Mismatch after activating ({ell}, {m})"

    def test_deactivate_mode(self):
        """Test deactivate_mode."""
        # Start with all active
        jax_modes = jnp.ones(self.max_l**2 + 2 * self.max_l + 1, dtype=bool)

        # Test modes to deactivate
        test_modes = [(0, 0), (1, 1), (2, -2)]

        for ell, m in test_modes:
            # JAX: Deactivate
            jax_modes = xlal_sim_inspiral_mode_array_deactivate_mode(jax_modes, ell, m, self.max_l)

            # Check deactivated
            i = ell * ell + ell + m
            assert not jax_modes[i], f"Mode ({ell}, {m}) should be deactivated"

    def test_is_mode_active(self):
        """Test is_mode_active matches LAL's check."""
        # Create mixed mode array
        lal_modes = lalsim.SimInspiralCreateModeArray()
        jax_modes = xlal_sim_inspiral_create_mode_array(self.max_l)

        # Activate some modes
        active_modes = [(1, 0), (2, 1), (3, -3)]
        for ell, m in active_modes:
            lalsim.SimInspiralModeArrayActivateMode(lal_modes, ell, m)
            jax_modes = xlal_sim_inspiral_mode_array_activate_mode(jax_modes, ell, m, self.max_l)

        # Test all possible modes up to max_l
        for ell in range(self.max_l + 1):
            for m in range(-ell, ell + 1):
                lal_active = lalsim.SimInspiralModeArrayIsModeActive(lal_modes, ell, m)
                jax_active = xlal_sim_inspiral_mode_array_is_mode_active(jax_modes, ell, m, self.max_l)
                assert lal_active == jax_active, f"Mismatch for mode ({ell}, {m}): LAL={lal_active}, JAX={jax_active}"

    def test_bounds_and_errors(self):
        """Test error handling for invalid inputs."""
        jax_modes = xlal_sim_inspiral_create_mode_array(self.max_l)

        # Invalid ell
        with pytest.raises(Exception, match="Invalid value of ell"):  # checkify.check raises error
            xlal_sim_inspiral_mode_array_activate_mode(jax_modes, self.max_l + 1, 0, self.max_l)

        # Invalid m
        with pytest.raises(Exception, match="Invalid value of m"):
            xlal_sim_inspiral_mode_array_activate_mode(jax_modes, 2, 3, self.max_l)
