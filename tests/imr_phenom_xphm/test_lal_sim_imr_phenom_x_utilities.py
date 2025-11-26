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
)

try:
    from lalsimulation import SimIMRPhenomXchiEff

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
