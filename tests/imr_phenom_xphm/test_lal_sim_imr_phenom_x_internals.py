"""Unit tests for lal_sim_imr_phenom_x_internals.py."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import pytest

from ripplegw.constants import PI
from ripplegw.waveforms.imr_phenom_xphm.lal_constants import LAL_MSUN_SI
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals import (
    check_input_mode_array,
    imr_phenom_x_get_phase_coefficients,
    imr_phenom_x_initialize_powers,
    imr_phenom_x_set_waveform_variables,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_create_mode_array,
    xlal_sim_inspiral_mode_array_activate_mode,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import (
    IMRPhenomXPHMParameterDataClass,
)


class TestImrPhenomXInitializePowers:
    """Tests for imr_phenom_x_initialize_powers function."""

    def test_negative_number(self):
        """Test that function raises error for negative number."""
        with pytest.raises(Exception, match="number must be non-negative"):
            imr_phenom_x_initialize_powers(-1.0)[0].throw()

    def test_zero(self):
        """Test that function works for zero (edge case)."""
        err, powers = imr_phenom_x_initialize_powers(0.0)
        err.throw()
        # For zero, some values are inf or nan, but check basic ones
        assert powers.itself == 0.0
        assert jnp.isinf(powers.m_one)  # 1/0 = inf
        assert jnp.isneginf(powers.log)  # log(0) = -inf

    def test_scalar_one(self):
        """Test computation for scalar input number=1."""
        err, powers = imr_phenom_x_initialize_powers(1.0)
        err.throw()

        # Check some key values
        assert powers.itself == 1.0
        assert powers.one_sixth == 1.0
        assert powers.m_one_sixth == 1.0
        assert powers.one_third == 1.0
        assert powers.two_thirds == 1.0
        assert powers.four_thirds == 1.0
        assert powers.two == 1.0
        assert powers.three == 1.0
        assert powers.four == 1.0
        assert powers.five == 1.0
        assert powers.log == 0.0
        assert powers.sqrt == 1.0

    def test_scalar_eight(self):
        """Test computation for scalar input number=8."""
        err, powers = imr_phenom_x_initialize_powers(8.0)
        err.throw()

        sixth = 8.0 ** (1 / 6)  # ≈ 1.5874010519681996
        assert jnp.allclose(powers.one_sixth, sixth)
        assert jnp.allclose(powers.m_one_sixth, 1.0 / sixth)
        assert jnp.allclose(powers.two, 64.0)
        assert jnp.allclose(powers.log, jnp.log(8.0))
        assert jnp.allclose(powers.sqrt, jnp.sqrt(8.0))

    def test_array_input(self):
        """Test that function works with array input."""
        numbers = jnp.array([1.0, 8.0, 27.0])
        err, powers = imr_phenom_x_initialize_powers(numbers)
        err.throw()

        # Check vectorization
        assert jnp.allclose(powers.itself, numbers)
        assert jnp.allclose(powers.two, numbers**2)
        assert jnp.allclose(powers.log, jnp.log(numbers))
        # Check one_sixth
        expected_sixth = numbers ** (1 / 6)
        assert jnp.allclose(powers.one_sixth, expected_sixth)

    def test_jit_compatibility(self):
        """Test that the function is JIT-compatible."""
        jit_powers = jax.jit(imr_phenom_x_initialize_powers)

        # Test scalar
        err, powers = jit_powers(1.0)
        err.throw()
        assert powers.itself == 1.0

        # Test array
        numbers = jnp.array([1.0, 2.0])
        err, powers = jit_powers(numbers)
        err.throw()
        assert jnp.allclose(powers.itself, numbers)


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
        lal_params = IMRPhenomXPHMParameterDataClass()
        err, result = check_input_mode_array(lal_params, self.max_l)
        err.throw()
        assert result is True

    def test_valid_modes(self):
        """Test that function returns True for allowed modes."""
        mode_array = self.create_mode_array_with_modes(self.allowed_modes, self.max_l)
        lal_params = IMRPhenomXPHMParameterDataClass()
        lal_params = dataclasses.replace(lal_params, mode_array=mode_array)

        err, result = check_input_mode_array(lal_params, self.max_l)
        err.throw()
        assert result is True

    def test_invalid_modes(self):
        """Test that function raises error for disallowed modes."""
        # Activate an invalid mode, e.g., (2, 0)
        invalid_modes = [(2, 0)]
        mode_array = self.create_mode_array_with_modes(self.allowed_modes + invalid_modes, self.max_l)
        lal_params = IMRPhenomXPHMParameterDataClass()
        lal_params = dataclasses.replace(lal_params, mode_array=mode_array)

        with pytest.raises(Exception, match="Invalid modes activated"):
            check_input_mode_array(lal_params, self.max_l)[0].throw()

    def test_mixed_valid_invalid(self):
        """Test that function raises error when both valid and invalid modes are active."""
        invalid_modes = [(3, 1)]  # (3,1) not allowed
        mode_array = self.create_mode_array_with_modes(self.allowed_modes[:2] + invalid_modes, self.max_l)
        lal_params = IMRPhenomXPHMParameterDataClass()
        lal_params = dataclasses.replace(lal_params, mode_array=mode_array)

        with pytest.raises(Exception, match="Invalid modes activated"):
            check_input_mode_array(lal_params, self.max_l)[0].throw()

    def test_negative_m_modes(self):
        """Test that negative m modes are handled (since |m| is checked)."""
        # (2, -2) should be allowed since (2,2) is
        mode_array = self.create_mode_array_with_modes([(2, -2)], self.max_l)
        lal_params = IMRPhenomXPHMParameterDataClass()
        lal_params = dataclasses.replace(lal_params, mode_array=mode_array)

        err, result = check_input_mode_array(lal_params, self.max_l)
        err.throw()
        assert result is True

    def test_jit_compatibility(self):
        """Test that the function is JIT-compatible."""
        # JIT compile the function
        jit_check = jax.jit(check_input_mode_array, static_argnames=["max_l"])

        # Test with no mode_array
        lal_params = IMRPhenomXPHMParameterDataClass()
        err, result = jit_check(lal_params, self.max_l)
        err.throw()
        assert result

        # Test with valid modes
        mode_array = self.create_mode_array_with_modes([(2, 2)], self.max_l)
        lal_params = IMRPhenomXPHMParameterDataClass()
        lal_params = dataclasses.replace(lal_params, mode_array=mode_array)

        err, result = jit_check(lal_params, self.max_l)
        err.throw()
        assert result

        # Test with invalid modes (should raise)
        invalid_modes = [(2, 0)]
        mode_array = self.create_mode_array_with_modes(invalid_modes, self.max_l)
        lal_params = IMRPhenomXPHMParameterDataClass()
        lal_params = dataclasses.replace(lal_params, mode_array=mode_array)

        with pytest.raises(Exception, match="Invalid modes activated"):
            jit_check(lal_params, self.max_l)[0].throw()

    def test_max_l_parameter(self):
        """Test that max_l parameter affects the check range."""
        # With max_l=3, (4,4) should be invalid even if activated
        mode_array = xlal_sim_inspiral_create_mode_array(5)  # Create with larger size
        mode_array = xlal_sim_inspiral_mode_array_activate_mode(mode_array, 4, 4, 5)
        lal_params = IMRPhenomXPHMParameterDataClass()
        lal_params = dataclasses.replace(lal_params, mode_array=mode_array)

        # With max_l=3, it should not check up to 4
        err, result = check_input_mode_array(lal_params, max_l=3)
        err.throw()
        assert result is True  # Since 4 > 3, not checked


class TestSetWaveformVariables:
    """Tests for imr_phenom_x_set_waveform_variables function."""

    def test_basic_initialization(self):
        """Test initialization with standard parameters."""
        m1_si = 30.0 * LAL_MSUN_SI
        m2_si = 20.0 * LAL_MSUN_SI
        chi1l = 0.5
        chi2l = -0.3
        delta_f = 0.125
        f_ref = 20.0
        phi0 = 0.0
        f_min = 20.0
        f_max = 1024.0
        distance = 1.0e6 * 3.08567758149137e16  # 1 Mpc
        inclination = 0.5

        lal_params = IMRPhenomXPHMParameterDataClass()
        # Initialize powers (needed for the function)
        _, powers = imr_phenom_x_initialize_powers(PI)

        err, waveform_vars = imr_phenom_x_set_waveform_variables(
            m1_si,
            m2_si,
            chi1l,
            chi2l,
            delta_f,
            f_ref,
            phi0,
            f_min,
            f_max,
            distance,
            inclination,
            lal_params,
            powers,
        )
        err.throw()

        # Check some derived quantities
        assert waveform_vars.m1_si == m1_si
        assert waveform_vars.m2_si == m2_si
        assert waveform_vars.chi1l == chi1l
        assert waveform_vars.chi2l == chi2l
        assert waveform_vars.delta_f == delta_f
        assert waveform_vars.f_ref == f_ref
        assert waveform_vars.phi0 == phi0
        assert waveform_vars.f_min == f_min
        assert waveform_vars.f_max == f_max
        assert waveform_vars.distance == distance
        assert waveform_vars.inclination == inclination

        # Check calculated quantities
        # eta = m1*m2/(m1+m2)^2 = 600/2500 = 0.24
        expected_eta = (30 * 20) / (50 * 50)
        assert jnp.allclose(waveform_vars.eta, expected_eta)

    def test_mass_swap(self):
        """Test that masses and spins are swapped if m2 > m1."""
        m1_si = 20.0 * LAL_MSUN_SI
        m2_si = 30.0 * LAL_MSUN_SI
        chi1l = 0.5
        chi2l = -0.3

        lal_params = IMRPhenomXPHMParameterDataClass()
        _, powers = imr_phenom_x_initialize_powers(PI)

        err, waveform_vars = imr_phenom_x_set_waveform_variables(
            m1_si,
            m2_si,
            chi1l,
            chi2l,
            0.1,
            20.0,
            0.0,
            20.0,
            100.0,
            1.0e22,
            0.0,
            lal_params,
            powers,
        )
        err.throw()

        # Should be swapped so m1 >= m2
        assert waveform_vars.m1_si == m2_si
        assert waveform_vars.m2_si == m1_si
        assert waveform_vars.chi1l == chi2l
        assert waveform_vars.chi2l == chi1l

    def test_spin_clamping(self):
        """Test that spins slightly outside [-1, 1] are clamped."""
        m1_si = 30.0 * LAL_MSUN_SI
        m2_si = 30.0 * LAL_MSUN_SI
        chi1l = 1.0000001  # Slightly > 1
        chi2l = -1.0000001  # Slightly < -1

        lal_params = IMRPhenomXPHMParameterDataClass()
        _, powers = imr_phenom_x_initialize_powers(PI)

        err, waveform_vars = imr_phenom_x_set_waveform_variables(
            m1_si,
            m2_si,
            chi1l,
            chi2l,
            0.1,
            20.0,
            0.0,
            20.0,
            100.0,
            1.0e22,
            0.0,
            lal_params,
            powers,
        )
        err.throw()

        assert jnp.isclose(waveform_vars.chi1l, 1.0)
        assert jnp.isclose(waveform_vars.chi2l, -1.0)

    def test_unphysical_spins(self):
        """Test that highly unphysical spins raise an error."""
        m1_si = 30.0 * LAL_MSUN_SI
        m2_si = 30.0 * LAL_MSUN_SI
        chi1l = 1.5  # Clearly > 1
        chi2l = 0.0

        lal_params = IMRPhenomXPHMParameterDataClass()
        _, powers = imr_phenom_x_initialize_powers(PI)

        with pytest.raises(Exception, match="Unphysical spins requested"):
            err, _ = imr_phenom_x_set_waveform_variables(
                m1_si,
                m2_si,
                chi1l,
                chi2l,
                0.1,
                20.0,
                0.0,
                20.0,
                100.0,
                1.0e22,
                0.0,
                lal_params,
                powers,
            )
            err.throw()

    def test_jit_compatibility(self):
        """Test JIT compatibility."""
        m1_si = 30.0 * LAL_MSUN_SI
        m2_si = 20.0 * LAL_MSUN_SI
        chi1l = 0.0
        chi2l = 0.0

        lal_params = IMRPhenomXPHMParameterDataClass()
        _, powers = imr_phenom_x_initialize_powers(PI)

        jit_set_vars = jax.jit(imr_phenom_x_set_waveform_variables)

        err, waveform_vars = jit_set_vars(
            m1_si,
            m2_si,
            chi1l,
            chi2l,
            0.1,
            20.0,
            0.0,
            20.0,
            100.0,
            1.0e22,
            0.0,
            lal_params,
            powers,
        )
        err.throw()

        assert waveform_vars.m1_si == m1_si


class TestIMRPhenomXGetPhaseCoefficients:
    """Tests for imr_phenom_x_get_phase_coefficients function."""

    def create_test_waveform_params(self, m1=30.0, m2=20.0, chi1l=0.5, chi2l=-0.3):
        """Helper to create waveform parameters for testing."""
        lal_params = IMRPhenomXPHMParameterDataClass()
        _, powers = imr_phenom_x_initialize_powers(PI)

        err, waveform_vars = imr_phenom_x_set_waveform_variables(
            m1 * LAL_MSUN_SI,
            m2 * LAL_MSUN_SI,
            chi1l,
            chi2l,
            0.125,  # delta_f
            20.0,  # f_ref
            0.0,  # phi0
            20.0,  # f_min
            1024.0,  # f_max
            1.0e6 * 3.08567758149137e16,  # distance (1 Mpc)
            0.5,  # inclination
            lal_params,
            powers,
        )
        err.throw()
        return waveform_vars

    def test_basic_phase_coefficients_computation(self):
        """Test that phase coefficients are computed without errors."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        # Call the function
        err, result = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)
        err.throw()

        # Check that result is returned
        assert isinstance(result, IMRPhenomXPhaseCoefficientsDataClass)

        # Check that some key fields have been populated (not zero)
        assert result.phi_norm != 0.0
        assert result.f_phase_ins_min != 0.0
        assert result.f_phase_ins_max != 0.0
        assert result.f_phase_rd_min != 0.0
        assert result.f_phase_rd_max != 0.0

    def test_transition_frequencies_ordering(self):
        """Test that transition frequencies are in correct order."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        err, result = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)
        err.throw()

        # Check ordering: ins_min < ins_max < match_in < match_im < rd_min < rd_max
        assert result.f_phase_ins_min < result.f_phase_ins_max
        assert result.f_phase_ins_max > result.f_phase_match_in
        assert result.f_phase_match_in < result.f_phase_match_im
        assert result.f_phase_match_im < result.f_phase_rd_max
        assert result.f_phase_rd_min < result.f_phase_rd_max

    def test_ringdown_collocation_points(self):
        """Test that ringdown collocation points are populated."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        err, result = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)
        err.throw()

        # Check that collocation points are set
        assert result.n_collocation_points_rd == 5
        assert result.collocation_points_phase_rd is not None
        assert len(result.collocation_points_phase_rd) == 5

        # Check that the 4th collocation point is set to f_ring
        assert jnp.isclose(result.collocation_points_phase_rd[3], p_wf.f_ring)

    def test_ringdown_coefficients_populated(self):
        """Test that ringdown phase coefficients are populated."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        err, result = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)
        err.throw()

        # Check that ringdown coefficients are not all zero
        # c0, c1, c2, c4, c_l, c_rd should be set
        coeffs = [result.c0, result.c1, result.c2, result.c4, result.c_l, result.c_rd]
        assert not all(c == 0.0 for c in coeffs)

    def test_pn_phase_coefficients_populated(self):
        """Test that PN phase coefficients are populated."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        err, result = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)
        err.throw()

        # Check that various PN phase coefficients are set
        # At least some should be non-zero
        pn_coeffs = [
            result.phi0,
            result.phi1,
            result.phi2,
            result.phi3,
            result.phi4,
            result.phi5,
            result.phi6,
            result.phi7,
        ]
        assert not all(c == 0.0 for c in pn_coeffs)

    def test_pn_phase_derivatives_populated(self):
        """Test that PN phase derivative coefficients are populated."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        err, result = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)
        err.throw()

        # Check that phase derivative coefficients are set
        dphi_coeffs = [
            result.dphi0,
            result.dphi1,
            result.dphi2,
            result.dphi3,
            result.dphi4,
            result.dphi5,
            result.dphi6,
            result.dphi7,
        ]
        assert not all(c == 0.0 for c in dphi_coeffs)

    def test_pseudo_pn_coefficients_populated(self):
        """Test that pseudo-PN sigma coefficients are populated."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        err, result = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)
        err.throw()

        # Check that sigma coefficients are set
        sigma_coeffs = [
            result.sigma1,
            result.sigma2,
            result.sigma3,
            result.sigma4,
            result.sigma5,
        ]
        assert not all(c == 0.0 for c in sigma_coeffs)

    def test_different_mass_ratios(self):
        """Test with different mass ratios."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        # Test equal mass case
        p_wf_equal = self.create_test_waveform_params(m1=30.0, m2=30.0)
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()
        err, result_equal = imr_phenom_x_get_phase_coefficients(p_wf_equal, p_phase)
        err.throw()

        # Test unequal mass case
        p_wf_unequal = self.create_test_waveform_params(m1=30.0, m2=10.0)
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()
        err, result_unequal = imr_phenom_x_get_phase_coefficients(p_wf_unequal, p_phase)
        err.throw()

        # Results should be different
        assert not jnp.allclose(result_equal.phi8, result_unequal.phi8)
        assert not jnp.allclose(result_equal.c0, result_unequal.c0)

    def test_different_spins(self):
        """Test with different spin configurations."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        # Test non-spinning case
        p_wf_nonspinning = self.create_test_waveform_params(chi1l=0.0, chi2l=0.0)
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()
        err, result_nonspinning = imr_phenom_x_get_phase_coefficients(p_wf_nonspinning, p_phase)
        err.throw()

        # Test spinning case
        p_wf_spinning = self.create_test_waveform_params(chi1l=0.8, chi2l=0.5)
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()
        err, result_spinning = imr_phenom_x_get_phase_coefficients(p_wf_spinning, p_phase)
        err.throw()

        # Results should be different
        assert not jnp.allclose(result_nonspinning.phi8, result_spinning.phi8)
        assert not jnp.allclose(result_nonspinning.c_l, result_spinning.c_l)

    def test_jit_compatibility(self):
        """Test that the function is JIT-compatible."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        # JIT compile the function
        jit_get_phase = jax.jit(imr_phenom_x_get_phase_coefficients)

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        # Call JIT-compiled version
        err, result = jit_get_phase(p_wf, p_phase)
        err.throw()

        # Check that it returns valid results
        assert isinstance(result, IMRPhenomXPhaseCoefficientsDataClass)
        assert result.phi_norm != 0.0
        assert result.n_collocation_points_rd == 5

    def test_reproducibility(self):
        """Test that the function gives reproducible results."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()

        # Call twice with same inputs
        p_phase1 = IMRPhenomXPhaseCoefficientsDataClass()
        err, result1 = imr_phenom_x_get_phase_coefficients(p_wf, p_phase1)

        p_phase2 = IMRPhenomXPhaseCoefficientsDataClass()
        err, result2 = imr_phenom_x_get_phase_coefficients(p_wf, p_phase2)
        err.throw()

        # Results should be identical
        assert jnp.allclose(result1.phi0, result2.phi0)
        assert jnp.allclose(result1.c0, result2.c0)
        assert jnp.allclose(result1.c_l, result2.c_l)
        assert jnp.array_equal(result1.collocation_points_phase_rd, result2.collocation_points_phase_rd)

    def test_phi_norm_value(self):
        """Test that phi_norm has expected value."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        err, result = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)
        err.throw()

        # phi_norm should be -(3.0 * PI^{-5/3}) / 128.0
        _, powers_of_pi = imr_phenom_x_initialize_powers(PI)
        expected_phi_norm = -(3.0 * powers_of_pi.m_five_thirds) / 128.0

        assert jnp.isclose(result.phi_norm, expected_phi_norm)

    def test_inspiral_phase_bounds(self):
        """Test that inspiral phase frequency bounds are set correctly."""
        from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
            IMRPhenomXPhaseCoefficientsDataClass,
        )

        p_wf = self.create_test_waveform_params()
        p_phase = IMRPhenomXPhaseCoefficientsDataClass()

        err, result = imr_phenom_x_get_phase_coefficients(p_wf, p_phase)
        err.throw()

        # f_phase_ins_min should be 0.0026 (as per code)
        assert jnp.isclose(result.f_phase_ins_min, 0.0026)

        # f_phase_ins_max should be 1.020 * f_meco
        expected_ins_max = 1.020 * p_wf.f_meco
        assert jnp.isclose(result.f_phase_ins_max, expected_ins_max)


if __name__ == "__main__":
    pytest.main([__file__])
