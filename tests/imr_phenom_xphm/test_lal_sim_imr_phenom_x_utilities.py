"""Unit tests for lal_sim_imr_phenom_x_utilities.py"""

from __future__ import annotations

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    xlal_imr_phenom_xp_check_masses_and_spins,
)


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
