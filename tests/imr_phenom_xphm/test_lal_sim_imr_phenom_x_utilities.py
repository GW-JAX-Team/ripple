"""Unit tests for lal_sim_imr_phenom_x_utilities.py"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    imr_phenom_x_approx_equal,
    imr_phenom_x_internal_nudge,
    xlal_imr_phenom_xp_check_masses_and_spins,
    xlal_sim_imr_phenom_x_chi_eff,
    xlal_sim_imr_phenom_x_chi_pn_hat,
    xlal_sim_imr_phenom_x_unwrap_array,
    xlal_sim_imr_phenom_x_utils_hz_to_mf,
    xlal_sim_imr_phenom_x_utils_mf_to_hz,
)

try:
    from lalsimulation import (
        SimIMRPhenomXchiEff,
        SimIMRPhenomXchiPNHat,
        SimIMRPhenomXUtilsHztoMf,
        SimIMRPhenomXUtilsMftoHz,
    )

    HAS_LAL = True
except ImportError:
    HAS_LAL = False


def test_xlal_imr_phenom_xp_check_masses_and_spins():
    """Test the xlal_imr_phenom_xp_check_masses_and_spins function.

    Remarks: IMRPhenomXPCheckMassesAndSpins does not take arguments,
        and therefore we cannot test it directly. Instead, we test
        our wrapper function against the expected behavior.
    """
    # Test case where m1 > m2
    m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z = (
        30.0,
        20.0,
        0.1,
        0.2,
        0.3,
        -0.1,
        -0.2,
        -0.3,
    )
    result = xlal_imr_phenom_xp_check_masses_and_spins(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
    assert result == (30.0, 20.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3)

    # Test case where m1 < m2
    m1, m2 = 20.0, 30.0
    result = xlal_imr_phenom_xp_check_masses_and_spins(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
    assert result == (30.0, 20.0, -0.1, -0.2, -0.3, 0.1, 0.2, 0.3)


class TestImrPhenomXApproxEqual:
    """Test imr_phenom_x_approx_equal function."""

    def test_identical_values(self):
        """Test that identical values are approximately equal."""
        assert imr_phenom_x_approx_equal(1.0, 1.0, 1e-10)

    def test_different_values_large_epsilon(self):
        """Test values within large epsilon."""
        # 1.0 and 1.1 differ by 0.1; max(|1.0|, |1.1|) = 1.1
        # 0.1 <= 0.2 * 1.1 = 0.22, so they should be approximately equal
        assert imr_phenom_x_approx_equal(1.0, 1.1, 0.2)

    def test_different_values_small_epsilon(self):
        """Test values outside small epsilon."""
        # 1.0 and 1.1 differ by 0.1; relative tolerance 0.01
        # 0.1 <= 0.01 * 1.1 = 0.011 is False
        assert not imr_phenom_x_approx_equal(1.0, 1.1, 0.01)

    def test_zero_values(self):
        """Test with zero values."""
        # 0 and 0 should be approximately equal
        assert imr_phenom_x_approx_equal(0.0, 0.0, 1e-10)

    def test_one_value_zero_within_epsilon(self):
        """Test when one value is zero and the other is within epsilon."""
        # |0.001 - 0| <= 1e-10 * max(|0.001|, 0) = 1e-10 * 0.001 = 1e-13
        # 0.001 > 1e-13, so not approximately equal
        assert not imr_phenom_x_approx_equal(0.001, 0.0, 1e-10)

    def test_one_value_zero_absolute_tolerance(self):
        """Test when one value is zero with relative tolerance."""
        # Uses relative tolerance based on max(|x|, |y|)
        # |1e-15 - 0| <= 1e-10 * max(|1e-15|, 0) = 1e-10 * 1e-15 = 1e-25
        # 1e-15 > 1e-25, so not approximately equal
        assert not imr_phenom_x_approx_equal(1e-15, 0.0, 1e-10)

    def test_negative_values(self):
        """Test with negative values."""
        # |-1.0 - (-1.0)| = 0 <= epsilon * max(|-1.0|, |-1.0|)
        assert imr_phenom_x_approx_equal(-1.0, -1.0, 1e-10)

    def test_mixed_sign_values(self):
        """Test with mixed sign values."""
        # |1.0 - (-1.0)| = 2.0; epsilon * max(|1.0|, |-1.0|) = 1e-10 * 1.0
        # 2.0 > 1e-10, not approximately equal
        assert not imr_phenom_x_approx_equal(1.0, -1.0, 1e-10)

    def test_jit_compatibility(self):
        """Test that imr_phenom_x_approx_equal is JIT-compatible."""
        jitted_func = jax.jit(imr_phenom_x_approx_equal)
        result = jitted_func(1.0, 1.0, 1e-10)
        assert result

    def test_very_small_values(self):
        """Test with very small values."""
        # max(|1e-15|, |1e-15|) = 1e-15
        # |1e-15 - 1e-15| = 0 <= 1e-10 * 1e-15
        assert imr_phenom_x_approx_equal(1e-15, 1e-15, 1e-10)

    def test_very_large_values(self):
        """Test with very large values."""
        # |1e15 - 1e15| = 0 <= 1e-10 * 1e15
        assert imr_phenom_x_approx_equal(1e15, 1e15, 1e-10)


class TestImrPhenomXInternalNudge:
    """Test imr_phenom_x_internal_nudge function."""

    def test_no_nudge_far_values_y_nonzero(self):
        """Test no nudge when x and y are far apart (y != 0)."""
        # x=1.0, y=2.0, epsilon=1e-10
        # |1.0 - 2.0| = 1.0; 1e-10 * max(1.0, 2.0) = 2e-10
        # 1.0 > 2e-10, so not approximately equal, return x
        result = imr_phenom_x_internal_nudge(1.0, 2.0, 1e-10)
        assert float(result) == 1.0

    def test_nudge_close_values_y_nonzero(self):
        """Test nudge when x and y are close (y != 0)."""
        # x=1.0, y=1.01, epsilon=0.02
        # |1.0 - 1.01| = 0.01; 0.02 * max(1.0, 1.01) = 0.0202
        # 0.01 <= 0.0202, so approximately equal, return y
        result = imr_phenom_x_internal_nudge(1.0, 1.01, 0.02)
        assert jnp.isclose(float(result), 1.01, rtol=1e-6, atol=1e-8)

    def test_no_nudge_y_zero_far(self):
        """Test no nudge when y=0 and x is far from zero."""
        # y=0, x=1.0, epsilon=0.1
        # |1.0 - 0| = 1.0; 1.0 < 0.1 is False, so return x
        result = imr_phenom_x_internal_nudge(1.0, 0.0, 0.1)
        assert float(result) == 1.0

    def test_nudge_y_zero_close(self):
        """Test nudge when y=0 and x is close to zero (absolute tolerance)."""
        # y=0, x=0.05, epsilon=0.1
        # |0.05 - 0| = 0.05; 0.05 < 0.1 is True, so return y=0
        result = imr_phenom_x_internal_nudge(0.05, 0.0, 0.1)
        assert float(result) == 0.0

    def test_identical_values_y_nonzero(self):
        """Test nudge when x == y (y != 0)."""
        # x=5.0, y=5.0, any epsilon
        # |5.0 - 5.0| = 0; 0 <= epsilon * 5.0 for any positive epsilon
        result = imr_phenom_x_internal_nudge(5.0, 5.0, 1e-10)
        assert float(result) == 5.0

    def test_identical_values_y_zero(self):
        """Test nudge when x == y == 0."""
        # x=0, y=0
        # Branches to y_zero case; |0 - 0| < epsilon is True, return y=0
        result = imr_phenom_x_internal_nudge(0.0, 0.0, 1e-10)
        assert float(result) == 0.0

    def test_negative_values_y_nonzero(self):
        """Test nudge with negative values (y != 0)."""
        # x=-1.0, y=-1.01, epsilon=0.02
        # |-1.0 - (-1.01)| = 0.01; 0.02 * max(|-1.0|, |-1.01|) = 0.0202
        # 0.01 <= 0.0202, so approximately equal, return y
        result = imr_phenom_x_internal_nudge(-1.0, -1.01, 0.02)
        assert jnp.isclose(float(result), -1.01, rtol=1e-6, atol=1e-8)

    def test_very_small_y_nonzero(self):
        """Test nudge with very small y (still nonzero)."""
        # x=1e-15, y=1e-15 (both very small, y != 0)
        # Uses relative tolerance; they're equal, so return y
        result = imr_phenom_x_internal_nudge(1e-15, 1e-15, 1e-10)
        assert jnp.isclose(float(result), 1e-15, rtol=1e-6, atol=1e-20)

    def test_jit_compatibility(self):
        """Test that imr_phenom_x_internal_nudge is JIT-compatible."""
        jitted_func = jax.jit(imr_phenom_x_internal_nudge)
        result = jitted_func(1.0, 2.0, 1e-10)
        assert float(result) == 1.0

    def test_boundary_epsilon_y_nonzero(self):
        """Test nudge at the boundary of epsilon (y != 0)."""
        # x=1.0, y=1.0001, epsilon=0.0001
        # |1.0 - 1.0001| = 0.0001; 0.0001 * max(1.0, 1.0001) ≈ 0.00010001
        # 0.0001 <= 0.00010001, so approximately equal, return y
        result = imr_phenom_x_internal_nudge(1.0, 1.0001, 0.0001)
        assert jnp.isclose(float(result), 1.0001, rtol=1e-6, atol=1e-8)

    def test_boundary_epsilon_y_zero(self):
        """Test nudge at the boundary of epsilon (y == 0)."""
        # y=0, x=0.1, epsilon=0.1
        # |0.1 - 0| = 0.1; 0.1 < 0.1 is False, so return x
        result = imr_phenom_x_internal_nudge(0.1, 0.0, 0.1)
        assert jnp.isclose(float(result), 0.1, rtol=1e-6, atol=1e-8)

    def test_mixed_sign_values(self):
        """Test nudge with mixed sign values (y != 0)."""
        # x=-1.0, y=1.0, epsilon=1e-10
        # |-1.0 - 1.0| = 2.0; 1e-10 * max(|-1.0|, |1.0|) = 1e-10
        # 2.0 > 1e-10, not approximately equal, return x
        result = imr_phenom_x_internal_nudge(-1.0, 1.0, 1e-10)
        assert float(result) == -1.0


class TestXlalSimImrPhenomXChiEff:
    """Test xlal_sim_imr_phenom_x_chi_eff function."""

    def test_chi_eff_zero_spins(self):
        """Test chi_eff with zero spins."""
        eta = 0.25  # Equal mass: m1 = m2
        chi1l = 0.0
        chi2l = 0.0
        result = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
        assert jnp.isclose(float(result), 0.0, atol=1e-14)

    def test_chi_eff_equal_aligned_spins(self):
        """Test chi_eff with equal aligned spins."""
        eta = 0.25  # Equal mass
        chi1l = 0.5
        chi2l = 0.5
        result = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
        # For equal mass: mm1 = mm2 = 0.5
        # chi_eff = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        assert jnp.isclose(float(result), 0.5, rtol=1e-6)

    def test_chi_eff_opposite_spins(self):
        """Test chi_eff with opposite spins."""
        eta = 0.25  # Equal mass
        chi1l = 0.5
        chi2l = -0.5
        result = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
        # For equal mass: mm1 = mm2 = 0.5
        # chi_eff = 0.5 * 0.5 + 0.5 * (-0.5) = 0
        assert jnp.isclose(float(result), 0.0, atol=1e-6)

    def test_chi_eff_unequal_mass(self):
        """Test chi_eff with unequal mass ratio."""
        # m1 = 10, m2 = 5: eta = 10*5 / (10+5)^2 = 50/225 ≈ 0.222
        eta = 50.0 / 225.0
        chi1l = 0.3
        chi2l = 0.1
        result = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)

        # delta = sqrt(1 - 4*eta) = sqrt(1 - 4*0.222) = sqrt(0.111) ≈ 0.333
        # mm1 = 0.5 * (1 + 0.333) ≈ 0.667
        # mm2 = 0.5 * (1 - 0.333) ≈ 0.333
        # chi_eff = 0.667 * 0.3 + 0.333 * 0.1
        delta = np.sqrt(1.0 - 4.0 * eta)
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        expected = mm1 * 0.3 + mm2 * 0.1
        assert jnp.isclose(float(result), expected, rtol=1e-6)

    def test_chi_eff_negative_spins(self):
        """Test chi_eff with negative spin values."""
        eta = 0.25
        chi1l = -0.4
        chi2l = -0.6
        result = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
        # For equal mass: mm1 = mm2 = 0.5
        # chi_eff = 0.5 * (-0.4) + 0.5 * (-0.6) = -0.5
        assert jnp.isclose(float(result), -0.5, rtol=1e-6)

    def test_chi_eff_jit_compatible(self):
        """Test that xlal_sim_imr_phenom_x_chi_eff is JIT-compatible."""
        jitted_func = jax.jit(xlal_sim_imr_phenom_x_chi_eff)
        eta = 0.25
        chi1l = 0.5
        chi2l = 0.3
        result = jitted_func(eta, chi1l, chi2l)
        expected = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
        assert jnp.isclose(float(result), float(expected), rtol=1e-6)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_chi_eff_cross_validation_equal_mass(self):
        """Cross-validate chi_eff against LAL for equal mass case."""
        eta = 0.25
        chi1l = 0.4
        chi2l = 0.3

        result_jax = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
        result_lal = SimIMRPhenomXchiEff(eta, chi1l, chi2l)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-6)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_chi_eff_cross_validation_unequal_mass(self):
        """Cross-validate chi_eff against LAL for unequal mass case."""
        # m1 = 15, m2 = 5: eta = 15*5 / (15+5)^2 = 75/400 = 0.1875
        eta = 75.0 / 400.0
        chi1l = -0.5
        chi2l = 0.2

        result_jax = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
        result_lal = SimIMRPhenomXchiEff(eta, chi1l, chi2l)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-6)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_chi_eff_cross_validation_zero_spins(self):
        """Cross-validate chi_eff against LAL with zero spins."""
        eta = 0.1875
        chi1l = 0.0
        chi2l = 0.0

        result_jax = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
        result_lal = SimIMRPhenomXchiEff(eta, chi1l, chi2l)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-6)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_chi_eff_cross_validation_extremal_spins(self):
        """Cross-validate chi_eff against LAL with near-extremal spins."""
        eta = 0.25
        chi1l = 0.99
        chi2l = -0.99

        result_jax = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
        result_lal = SimIMRPhenomXchiEff(eta, chi1l, chi2l)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-6)


class TestXlalSimImrPhenomXChiPNHat:
    """Test xlal_sim_imr_phenom_x_chi_pn_hat function."""

    def test_chi_pn_hat_zero_spins(self):
        """Test chi_pn_hat with zero spins."""
        eta = 0.25  # Equal mass: m1 = m2
        chi1l = 0.0
        chi2l = 0.0
        result = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
        assert jnp.isclose(float(result), 0.0, atol=1e-14)

    def test_chi_pn_hat_equal_aligned_spins(self):
        """Test chi_pn_hat with equal aligned spins."""
        eta = 0.25  # Equal mass
        chi1l = 0.5
        chi2l = 0.5
        result = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
        # For equal mass: chi_eff = 0.5, chi1l + chi2l = 1.0
        # chi_pn_hat = (0.5 - (38/113) * 0.25 * 1.0) / (1 - (76/113) * 0.25)
        # = (0.5 - 0.0842...) / (1 - 0.169...)
        chi_eff = 0.5
        sum_spins = 1.0
        expected = (chi_eff - (38.0 / 113.0) * eta * sum_spins) / (1.0 - (76.0 / 113.0) * eta)
        assert jnp.isclose(float(result), expected, rtol=1e-6)

    def test_chi_pn_hat_opposite_spins(self):
        """Test chi_pn_hat with opposite spins."""
        eta = 0.25  # Equal mass
        chi1l = 0.5
        chi2l = -0.5
        result = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
        # For equal mass: chi_eff = 0, chi1l + chi2l = 0
        # chi_pn_hat = (0 - (38/113) * 0.25 * 0) / (1 - (76/113) * 0.25) = 0
        assert jnp.isclose(float(result), 0.0, atol=1e-6)

    def test_chi_pn_hat_unequal_mass(self):
        """Test chi_pn_hat with unequal mass ratio."""
        eta = 50.0 / 225.0  # m1 = 10, m2 = 5
        chi1l = 0.3
        chi2l = 0.1
        result = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)

        # Calculate expected value
        delta = np.sqrt(1.0 - 4.0 * eta)
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        chi_eff = mm1 * chi1l + mm2 * chi2l
        sum_spins = chi1l + chi2l
        expected = (chi_eff - (38.0 / 113.0) * eta * sum_spins) / (1.0 - (76.0 / 113.0) * eta)
        assert jnp.isclose(float(result), expected, rtol=1e-6)

    def test_chi_pn_hat_negative_spins(self):
        """Test chi_pn_hat with negative spin values."""
        eta = 0.25
        chi1l = -0.4
        chi2l = -0.6
        result = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
        # chi_eff = -0.5, chi1l + chi2l = -1.0
        chi_eff = -0.5
        sum_spins = -1.0
        expected = (chi_eff - (38.0 / 113.0) * eta * sum_spins) / (1.0 - (76.0 / 113.0) * eta)
        assert jnp.isclose(float(result), expected, rtol=1e-6)

    def test_chi_pn_hat_jit_compatible(self):
        """Test that xlal_sim_imr_phenom_x_chi_pn_hat is JIT-compatible."""
        jitted_func = jax.jit(xlal_sim_imr_phenom_x_chi_pn_hat)
        eta = 0.25
        chi1l = 0.5
        chi2l = 0.3
        result = jitted_func(eta, chi1l, chi2l)
        expected = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
        assert jnp.isclose(float(result), float(expected), rtol=1e-6)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_chi_pn_hat_cross_validation_equal_mass(self):
        """Cross-validate chi_pn_hat against LAL for equal mass case."""
        eta = 0.25
        chi1l = 0.4
        chi2l = 0.3

        result_jax = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
        result_lal = SimIMRPhenomXchiPNHat(eta, chi1l, chi2l)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-6)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_chi_pn_hat_cross_validation_unequal_mass(self):
        """Cross-validate chi_pn_hat against LAL for unequal mass case."""
        eta = 75.0 / 400.0  # m1 = 15, m2 = 5
        chi1l = -0.5
        chi2l = 0.2

        result_jax = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
        result_lal = SimIMRPhenomXchiPNHat(eta, chi1l, chi2l)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-6)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_chi_pn_hat_cross_validation_zero_spins(self):
        """Cross-validate chi_pn_hat against LAL with zero spins."""
        eta = 0.1875
        chi1l = 0.0
        chi2l = 0.0

        result_jax = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
        result_lal = SimIMRPhenomXchiPNHat(eta, chi1l, chi2l)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-6)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_chi_pn_hat_cross_validation_extremal_spins(self):
        """Cross-validate chi_pn_hat against LAL with near-extremal spins."""
        eta = 0.25
        chi1l = 0.99
        chi2l = -0.99

        result_jax = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
        result_lal = SimIMRPhenomXchiPNHat(eta, chi1l, chi2l)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-6)


class TestXlalSimImrPhenomXUtilsHztoMf:
    """Test xlal_sim_imr_phenom_x_utils_hz_to_mf function."""

    def test_basic_conversion(self):
        """Test basic frequency conversion."""
        f_hz = 100.0  # 100 Hz
        m_tot_msun = 60.0  # 60 solar masses

        result = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_hz, m_tot_msun)

        # Expected: f_hz * (LAL_MTSUN_SI * m_tot_msun)
        # LAL_MTSUN_SI ≈ 4.925490947641267e-06
        expected = f_hz * 4.925490947641267e-06 * m_tot_msun
        assert jnp.isclose(float(result), expected, rtol=1e-10)

    def test_zero_frequency(self):
        """Test conversion with zero frequency."""
        f_hz = 0.0
        m_tot_msun = 50.0

        result = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_hz, m_tot_msun)
        assert jnp.isclose(float(result), 0.0, atol=1e-15)

    def test_zero_mass(self):
        """Test conversion with zero total mass."""
        f_hz = 50.0
        m_tot_msun = 0.0

        result = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_hz, m_tot_msun)
        assert jnp.isclose(float(result), 0.0, atol=1e-15)

    def test_typical_gw_frequency(self):
        """Test conversion with typical gravitational wave frequency."""
        f_hz = 20.0  # 20 Hz, typical LIGO frequency
        m_tot_msun = 100.0  # 100 solar masses

        result = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_hz, m_tot_msun)

        # This should give a reasonable dimensionless frequency
        # For 100 Msun total mass, Mf = f * (4.925e-6 * 100) ≈ f * 4.925e-4
        # At 20 Hz: 20 * 4.925e-4 ≈ 0.00985
        expected = f_hz * 4.925490947641267e-06 * m_tot_msun
        assert jnp.isclose(float(result), expected, rtol=1e-10)
        assert 0.005 < float(result) < 0.02  # Reasonable range check

    def test_jit_compatible(self):
        """Test that xlal_sim_imr_phenom_x_utils_hz_to_mf is JIT-compatible."""
        jitted_func = jax.jit(xlal_sim_imr_phenom_x_utils_hz_to_mf)
        f_hz = 35.0
        m_tot_msun = 75.0

        result = jitted_func(f_hz, m_tot_msun)
        expected = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_hz, m_tot_msun)
        assert jnp.isclose(float(result), float(expected), rtol=1e-10)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_cross_validation_basic(self):
        """Cross-validate xlal_sim_imr_phenom_x_utils_hz_to_mf against LAL."""
        f_hz = 25.0
        m_tot_msun = 80.0

        result_jax = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_hz, m_tot_msun)
        result_lal = SimIMRPhenomXUtilsHztoMf(f_hz, m_tot_msun)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-10)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_cross_validation_range(self):
        """Cross-validate across a range of frequencies and masses."""
        frequencies = [10.0, 20.0, 50.0, 100.0, 200.0]
        masses = [20.0, 50.0, 100.0, 150.0]

        for f_hz in frequencies:
            for m_tot_msun in masses:
                result_jax = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_hz, m_tot_msun)
                result_lal = SimIMRPhenomXUtilsHztoMf(f_hz, m_tot_msun)

                assert jnp.isclose(
                    float(result_jax), result_lal, rtol=1e-10
                ), f"Mismatch at f_hz={f_hz}, m_tot_msun={m_tot_msun}"

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_cross_validation_edge_cases(self):
        """Cross-validate edge cases against LAL."""
        test_cases = [
            (0.0, 50.0),  # Zero frequency
            (100.0, 0.0),  # Zero mass
            (1e-6, 1e6),  # Very small/large values
            (1e6, 1e-6),  # Very large/small values
        ]

        for f_hz, m_tot_msun in test_cases:
            result_jax = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_hz, m_tot_msun)
            result_lal = SimIMRPhenomXUtilsHztoMf(f_hz, m_tot_msun)

            assert jnp.isclose(
                float(result_jax), result_lal, rtol=1e-10
            ), f"Mismatch at f_hz={f_hz}, m_tot_msun={m_tot_msun}"


class TestXlalSimImrPhenomXUtilsMftoHz:
    """Test xlal_sim_imr_phenom_x_utils_mf_to_hz function."""

    def test_basic_conversion(self):
        """Test basic frequency conversion from Mf to Hz."""
        mf = 0.01  # Dimensionless frequency
        m_tot_msun = 60.0  # 60 solar masses

        result = xlal_sim_imr_phenom_x_utils_mf_to_hz(mf, m_tot_msun)

        # Expected: mf / (LAL_MTSUN_SI * m_tot_msun)
        # LAL_MTSUN_SI ≈ 4.925490947641267e-06
        expected = mf / (4.925490947641267e-06 * m_tot_msun)
        assert jnp.isclose(float(result), expected, rtol=1e-10)

    def test_zero_dimensionless_frequency(self):
        """Test conversion with zero dimensionless frequency."""
        mf = 0.0
        m_tot_msun = 50.0

        result = xlal_sim_imr_phenom_x_utils_mf_to_hz(mf, m_tot_msun)
        assert jnp.isclose(float(result), 0.0, atol=1e-15)

    def test_typical_gw_dimensionless_frequency(self):
        """Test conversion with typical gravitational wave dimensionless frequency."""
        mf = 0.01  # Typical dimensionless frequency
        m_tot_msun = 100.0  # 100 solar masses

        result = xlal_sim_imr_phenom_x_utils_mf_to_hz(mf, m_tot_msun)

        # Expected: mf / (LAL_MTSUN_SI * m_tot_msun)
        # For 100 Msun: mf / (4.925e-6 * 100) ≈ mf / 4.925e-4
        # At mf=0.01: 0.01 / 4.925e-4 ≈ 20.3 Hz
        expected = mf / (4.925490947641267e-06 * m_tot_msun)
        assert jnp.isclose(float(result), expected, rtol=1e-10)
        assert 15.0 < float(result) < 30.0  # Reasonable range check

    def test_jit_compatible(self):
        """Test that xlal_sim_imr_phenom_x_utils_mf_to_hz is JIT-compatible."""
        jitted_func = jax.jit(xlal_sim_imr_phenom_x_utils_mf_to_hz)
        mf = 0.005
        m_tot_msun = 75.0

        result = jitted_func(mf, m_tot_msun)
        expected = xlal_sim_imr_phenom_x_utils_mf_to_hz(mf, m_tot_msun)
        assert jnp.isclose(float(result), float(expected), rtol=1e-6)

    def test_inverse_relationship(self):
        """Test that mf_to_hz and hz_to_mf are inverses."""
        f_hz = 25.0
        m_tot_msun = 80.0

        # Convert Hz to Mf, then back to Hz
        mf = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_hz, m_tot_msun)
        f_hz_recovered = xlal_sim_imr_phenom_x_utils_mf_to_hz(mf, m_tot_msun)

        assert jnp.isclose(f_hz_recovered, f_hz, rtol=1e-10)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_cross_validation_basic(self):
        """Cross-validate xlal_sim_imr_phenom_x_utils_mf_to_hz against LAL."""
        mf = 0.02
        m_tot_msun = 80.0

        result_jax = xlal_sim_imr_phenom_x_utils_mf_to_hz(mf, m_tot_msun)
        result_lal = SimIMRPhenomXUtilsMftoHz(mf, m_tot_msun)

        assert jnp.isclose(float(result_jax), result_lal, rtol=1e-10)

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_cross_validation_range(self):
        """Cross-validate across a range of dimensionless frequencies and masses."""
        mfs = [0.001, 0.01, 0.05, 0.1, 0.2]
        masses = [20.0, 50.0, 100.0, 150.0]

        for mf in mfs:
            for m_tot_msun in masses:
                result_jax = xlal_sim_imr_phenom_x_utils_mf_to_hz(mf, m_tot_msun)
                result_lal = SimIMRPhenomXUtilsMftoHz(mf, m_tot_msun)

                assert jnp.isclose(
                    float(result_jax), result_lal, rtol=1e-10
                ), f"Mismatch at mf={mf}, m_tot_msun={m_tot_msun}"

    @pytest.mark.skipif(not HAS_LAL, reason="lalsimulation not available")
    def test_cross_validation_edge_cases(self):
        """Cross-validate edge cases against LAL."""
        test_cases = [
            (0.0, 50.0),  # Zero dimensionless frequency
            (1e-6, 1e6),  # Very small/large values
            (1e-3, 1e-6),  # Very large frequency for small mass
        ]

        for mf, m_tot_msun in test_cases:
            result_jax = xlal_sim_imr_phenom_x_utils_mf_to_hz(mf, m_tot_msun)
            result_lal = SimIMRPhenomXUtilsMftoHz(mf, m_tot_msun)

            assert jnp.isclose(
                float(result_jax), result_lal, rtol=1e-10
            ), f"Mismatch at mf={mf}, m_tot_msun={m_tot_msun}"


class TestUnwrapArray:
    """Test suite for xlal_sim_imr_phenom_x_unwrap_array function."""

    def test_unwrap_array_already_continuous(self):
        """Test that already-continuous phase array is unchanged."""
        phases = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        unwrapped = xlal_sim_imr_phenom_x_unwrap_array(phases)

        assert jnp.allclose(unwrapped, phases, atol=1e-14)

    def test_unwrap_array_single_wrap(self):
        """Test unwrapping with a single 2π discontinuity."""
        # After 3.0, jump to -3.0 (equivalent to 3.0 + 2π ~ 9.28, wraps to -3.0)
        phases = jnp.array([0.0, 1.0, 2.0, 3.0, -3.0, -2.0])
        unwrapped = xlal_sim_imr_phenom_x_unwrap_array(phases)

        # The -3.0 should be unwrapped to ~3.28 (detected as +2π jump)
        # Check that unwrapped is monotonic after first 3 elements
        assert jnp.all(jnp.diff(unwrapped[:4]) > 0), "Phase should increase before wrap"
        assert jnp.all(jnp.diff(unwrapped[3:]) > 0), "Phase should increase after unwrap"

    def test_unwrap_array_oscillating_phase(self):
        """Test unwrapping phase that oscillates around π."""
        # Simulate phase oscillating: 0.1, 3.0, 0.1, 3.0
        # Differences: 2.9, -2.9, 2.9
        # 2.9 > π, so wraps to 2.9 - 2π ~ -3.38
        # -2.9 < -π, so wraps to -2.9 + 2π ~ 3.38
        phases = jnp.array([0.1, 3.0, 0.1, 3.0])
        unwrapped = xlal_sim_imr_phenom_x_unwrap_array(phases)

        # After unwrapping, should show smooth evolution, not oscillation
        # Each element should be monotonic or at least not oscillate
        assert unwrapped[0] == phases[0]

        # Check differences are now continuous (no large jumps)
        diffs = jnp.diff(unwrapped)
        assert jnp.all(jnp.abs(diffs) < jnp.pi), "Unwrapped differences should be < π"

    def test_unwrap_array_atan2_output(self):
        """Test unwrapping typical atan2 output with wrapping."""
        # Simulate output from atan2 in range [-π, π]
        # Create angles that increase but wrap at ±π
        phases = jnp.array(
            [
                -3.0,  # Near -π
                -2.0,  # Decreasing toward -π/2
                -1.0,  #
                0.0,  # Crossing zero
                1.0,  # Increasing
                2.0,  # Near π
                -3.0,  # Wrapped to -π side
                -2.0,  # Continuing
            ]
        )
        unwrapped = xlal_sim_imr_phenom_x_unwrap_array(phases)

        # Should be strictly increasing or at least continuous
        diffs = jnp.diff(unwrapped)
        assert jnp.all(jnp.abs(diffs) < jnp.pi), "All differences should be < π"

    def test_unwrap_array_empty(self):
        """Test empty array."""
        phases = jnp.array([])
        unwrapped = xlal_sim_imr_phenom_x_unwrap_array(phases)

        assert unwrapped.shape == (0,)

    def test_unwrap_array_single_element(self):
        """Test single element array."""
        phases = jnp.array([1.5])
        unwrapped = xlal_sim_imr_phenom_x_unwrap_array(phases)

        assert jnp.allclose(unwrapped, phases)

    def test_unwrap_array_two_elements(self):
        """Test two element array."""
        phases = jnp.array([0.1, 3.5])
        unwrapped = xlal_sim_imr_phenom_x_unwrap_array(phases)

        assert jnp.isclose(unwrapped[0], 0.1)
        # 3.5 - 0.1 = 3.4, which is > π ≈ 3.14159, so wraps to 3.4 - 2π ≈ -2.88
        delta_phase = 3.5 - 0.1
        delta_wrapped = delta_phase - 2 * jnp.pi  # Since 3.4 > π
        expected_second = 0.1 + delta_wrapped
        assert jnp.isclose(unwrapped[1], expected_second)

    def test_unwrap_array_large_jumps(self):
        """Test with large phase jumps."""
        # Large jumps that should all be wrapped to [-π, π]
        phases = jnp.array([0.0, 5.0, 10.0, 15.0, 20.0])
        unwrapped = xlal_sim_imr_phenom_x_unwrap_array(phases)

        # After unwrapping, should be monotonic or continuous
        diffs = jnp.diff(unwrapped)
        assert jnp.all(jnp.abs(diffs) < jnp.pi), "Unwrapped differences should be < π"

    def test_unwrap_array_jittable(self):
        """Test that unwrap_array is JAX JIT-compilable."""
        phases = jnp.array([0.1, 1.0, 2.0, 3.0, -2.0, -1.0])

        jitted_unwrap = jax.jit(xlal_sim_imr_phenom_x_unwrap_array)
        unwrapped_jit = jitted_unwrap(phases)
        unwrapped_regular = xlal_sim_imr_phenom_x_unwrap_array(phases)

        assert jnp.allclose(unwrapped_jit, unwrapped_regular)

    def test_unwrap_array_differentiable(self):
        """Test that unwrap_array is differentiable."""
        phases = jnp.array([0.1, 1.0, 2.0, 3.0])

        def loss_fn(x):
            unwrapped = xlal_sim_imr_phenom_x_unwrap_array(x)
            return jnp.sum(unwrapped**2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(phases)

        # Should produce gradients without error
        assert grads.shape == phases.shape
        assert jnp.all(jnp.isfinite(grads))


if __name__ == "__main__":
    pytest.main([__file__])
