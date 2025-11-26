"""Utilities for IMRPhenomX waveform model."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


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
    return lax.cond(
        m1_si < m2_si,
        lambda: (m2_si, m1_si, chi2x, chi2y, chi2z, chi1x, chi1y, chi1z),
        lambda: (m1_si, m2_si, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z),
    )


def imr_phenom_x_approx_equal(x: float, y: float, epsilon: float) -> bool:
    """Check if two floats are approximately equal within epsilon.

    Equivalent to gsl_fcmp returning 0 (approximate equality).

    Args:
        x: First value.
        y: Second value.
        epsilon: Tolerance for comparison.

    Returns:
        True if |x - y| <= epsilon * max(|x|, |y|), False otherwise.
    """
    abs_diff = jnp.abs(x - y)
    max_abs = jnp.maximum(jnp.abs(x), jnp.abs(y))
    return abs_diff <= epsilon * max_abs


def imr_phenom_x_internal_nudge(x: float, y: float, epsilon: float) -> float:
    """Nudge x towards y by epsilon if they're approximately equal.

    If y != 0 and x ≈ y (within relative epsilon), return y.
    If y == 0 and |x - y| < epsilon, return y.
    Otherwise return x unchanged.

    Args:
        x: Value to potentially nudge.
        y: Target value.
        epsilon: Tolerance for comparison.

    Returns:
        Nudged or original value.
    """

    def nudge_branch_y_nonzero(_):
        """Nudge x to y when y != 0 and x ≈ y (relative tolerance)."""
        is_approx = imr_phenom_x_approx_equal(x, y, epsilon)
        return lax.cond(is_approx, lambda _: y, lambda _: x, operand=None)

    def nudge_branch_y_zero(_):
        """Nudge x to y when y == 0 and |x - y| < epsilon (absolute tolerance)."""
        is_close = jnp.abs(x - y) < epsilon
        return lax.cond(is_close, lambda _: y, lambda _: x, operand=None)

    # Check if y is nonzero or zero and apply appropriate branch
    y_is_nonzero = y != 0.0
    return lax.cond(y_is_nonzero, nudge_branch_y_nonzero, nudge_branch_y_zero, operand=None)
