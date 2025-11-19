"""Utilities for IMRPhenomX waveform model."""

from __future__ import annotations


def xlal_imr_phenom_xp_check_masses_and_spins(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    m1_si: float,
    m2_si: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Check if m1 > m2, swap the bodies otherwise.

    This function checks if the mass of the first body (m1)
    is greater than the mass of the second body (m2).
    If not, it swaps the masses and corresponding spin components
    to ensure that m1 is always the larger mass.

    Args:
        m1_si (float): Mass of the first body in SI units.
        m2_si (float): Mass of the second body in SI units.
        chi1x (float): x-component of the dimensionless spin of the first body.
        chi1y (float): y-component of the dimensionless spin of the first body.
        chi1z (float): z-component of the dimensionless spin of the first body.
        chi2x (float): x-component of the dimensionless spin of the second body.
        chi2y (float): y-component of the dimensionless spin of the second body.
        chi2z (float): z-component of the dimensionless spin of the second body.

    Returns:
        tuple: A tuple containing possibly swapped values of
            (m1_si, m2_si, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z).
    """
    if m1_si < m2_si:
        # Swap masses
        m1_si, m2_si = m2_si, m1_si
        # Swap spins
        chi1x, chi2x = chi2x, chi1x
        chi1y, chi2y = chi2y, chi1y
        chi1z, chi2z = chi2z, chi1z

    return m1_si, m2_si, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z
