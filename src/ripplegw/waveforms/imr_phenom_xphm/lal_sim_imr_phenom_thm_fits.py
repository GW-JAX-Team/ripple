"""
Set of phenomenological fits employed by IMRPhenomT and IMRPhenomTHM models.
Collocation points and coefficients have been calibrated with 531 BBH
non-precessing NR simulations from the last release of the SXS Catalog,
additional BAM NR simulations at q=4, q=8 and q=18, and numerical Teukolsky waveforms placed
at q=200 and q=1000. Calibration procedure has followed the hierarchical data-driven fitting
approach (Xisco Jimenez-Forteza et al https://arxiv.org/abs/1611.00332) using the symmetric mass
ratio eta, dimensionless effective spin Shat=(m1^2*chi1+m2^2*chi2)/(m1^2+m2^2)
and spin difference dchi=chi1-chi2.
Supplementary material for the fits is available at
https://git.ligo.org/waveforms/reviews/phenomt/-/tree/master/SupplementaryMaterial/Fits3DPhenomTHM
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental import checkify


def evaluate_qnm_fit_fring22(final_dimless_spin: float) -> float:
    """
    Evaluate the QNM fit for the (2,2) mode ringdown frequency.

    Args:
        final_dimless_spin: Dimensionless spin of the final black hole.

    Returns:
        Ringdown frequency for the (2,2) mode.
    """
    checkify.check(jnp.fabs(final_dimless_spin) <= 1.0, "Final dimensionless spin must be in [-1, 1]")
    x2 = final_dimless_spin * final_dimless_spin
    x3 = x2 * final_dimless_spin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    x7 = x4 * x3
    return (
        0.05947169566573468
        - 0.14989771215394762 * final_dimless_spin
        + 0.09535606290986028 * x2
        + 0.02260924869042963 * x3
        - 0.02501704155363241 * x4
        - 0.005852438240997211 * x5
        + 0.0027489038393367993 * x6
        + 0.0005821983163192694 * x7
    ) / (
        1
        - 2.8570126619966296 * final_dimless_spin
        + 2.373335413978394 * x2
        - 0.6036964688511505 * x4
        + 0.0873798215084077 * x6
    )
