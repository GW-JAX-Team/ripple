"""Utilities for IMRPhenomX waveform model."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from ripplegw.constants import PI


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


def xlal_sim_imr_phenom_x_chi_eff(eta: float, chi1l: float, chi2l: float) -> float:
    """Compute the effective spin parameter chi_eff.

    Args:
        eta: Symmetric mass ratio.
        chi1l: Dimensionless spin component of the first body along the orbital angular momentum.
        chi2l: Dimensionless spin component of the second body along the orbital angular momentum.

    Returns:
        Effective spin parameter chi_eff.
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    return mm1 * chi1l + mm2 * chi2l


def xlal_sim_imr_phenom_x_chi_pn_hat(eta: float, chi1l: float, chi2l: float) -> float:
    """Compute the PN hat spin parameter chi_pn_hat.

    Args:
        eta: Symmetric mass ratio.
        chi1l: Dimensionless spin component of the first body along the orbital angular momentum.
        chi2l: Dimensionless spin component of the second body along the orbital angular momentum.

    Returns:
        PN hat spin parameter chi_pn_hat.
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    chi_eff = mm1 * chi1l + mm2 * chi2l
    return (chi_eff - (38.0 / 113.0) * eta * (chi1l + chi2l)) / (1.0 - (76.0 / 113.0) * eta)


# /**
#  * Total spin normalized to [-1,1]
#  */
def xlal_sim_imr_phenom_x_stot_r(eta: float, chi1_l: float, chi2_l: float) -> float:
    """Compute the total spin normalized to [-1, 1].

    Args:
        eta: Symmetric mass ratio.
        chi1_l: Dimensionless spin component of the first body along the orbital angular momentum.
        chi2_l: Dimensionless spin component of the second body along the orbital angular momentum.

    Returns:
        Total spin normalized to [-1, 1].
    """
    # // Convention m1 >= m2 and chi1z is the spin projected along Lz on m1
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1 + delta)
    m2 = 0.5 * (1 - delta)
    m1s = m1 * m1
    m2s = m2 * m2

    return (m1s * chi1_l + m2s * chi2_l) / (m1s + m2s)


# /**
#  * Spin difference
#  */
def xlal_sim_imr_phenom_x_dchi(chi1_l: float, chi2_l: float) -> float:
    """Compute the spin difference dchi = chi1_l - chi2_l.

    Args:
        chi1_l: Dimensionless spin component of the first body along the orbital angular momentum.
        chi2_l: Dimensionless spin component of the second body along the orbital angular momentum.

    Returns:
        Spin difference dchi.
    """
    return chi1_l - chi2_l


# /**
#  * Final Mass = 1 - Energy Radiated,  X Jimenez-Forteza et al, PRD, 95, 064024, (2017), arXiv:1611.00332
#  */
def xlal_sim_imr_phenom_x_final_mass_2017(eta: float, chi1_l: float, chi2_l: float) -> float:
    """Compute the final mass of the merged object.

    Args:
        eta: Symmetric mass ratio.
        chi1_l: Dimensionless spin component of the first body along the orbital angular momentum.
        chi2_l: Dimensionless spin component of the second body along the orbital angular momentum.

    Returns:
        Final mass of the merged object.
    """

    delta = jnp.sqrt(1.0 - 4.0 * eta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    s = xlal_sim_imr_phenom_x_stot_r(eta, chi1_l, chi2_l)
    s2 = s * s
    s3 = s2 * s

    dchi = chi1_l - chi2_l
    dchi2 = dchi * dchi

    no_spin = (
        0.057190958417936644 * eta + 0.5609904135313374 * eta2 - 0.84667563764404 * eta3 + 3.145145224278187 * eta4
    )

    #   /* Because of the way this is written, we need to subtract the noSpin term */
    eq_spin = (
        (0.057190958417936644 * eta + 0.5609904135313374 * eta2 - 0.84667563764404 * eta3 + 3.145145224278187 * eta4)
        * (
            1
            + (-0.13084389181783257 - 1.1387311580238488 * eta + 5.49074464410971 * eta2) * s
            + (-0.17762802148331427 + 2.176667900182948 * eta2) * s2
            + (-0.6320191645391563 + 4.952698546796005 * eta - 10.023747993978121 * eta2) * s3
        )
    ) / (1 + (-0.9919475346968611 + 0.367620218664352 * eta + 4.274567337924067 * eta2) * s)

    eq_spin = eq_spin - no_spin

    uneq_spin = (
        -0.09803730445895877 * dchi * delta * (1 - 3.2283713377939134 * eta) * eta2
        + 0.01118530335431078 * dchi2 * eta3
        - 0.01978238971523653 * dchi * delta * (1 - 4.91667749015812 * eta) * eta * s
    )

    #   /* M_final = 1 - E_rad, assuming that M = m1 + m2 = 1 */
    return 1.0 - (no_spin + eq_spin + uneq_spin)


# /**
#  * Final Dimensionless Spin,  X Jimenez-Forteza et al, PRD, 95, 064024, (2017), arXiv:1611.00332
#  */
def xlal_sim_imr_phenom_x_final_spin_2017(eta, chi1_l, chi2_l) -> float:  # pylint: disable=too-many-locals
    """Compute the final dimensionless spin of the merged object.

    Args:
        eta: Symmetric mass ratio.
        chi1_l: Dimensionless spin component of the first body along the orbital angular momentum.
        chi2_l: Dimensionless spin component of the second body along the orbital angular momentum.

    Returns:
        Final dimensionless spin of the merged object.
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + delta)
    m2 = 0.5 * (1.0 - delta)
    m1_sq = m1 * m1
    m2_sq = m2 * m2

    eta2 = eta * eta
    eta3 = eta2 * eta

    # //S  = (m1Sq * chi1L + m2Sq * chi2L) / (m1Sq + m2Sq)
    s = xlal_sim_imr_phenom_x_stot_r(eta, chi1_l, chi2_l)
    s2 = s * s
    s3 = s2 * s

    dchi = chi1_l - chi2_l
    dchi2 = dchi * dchi

    no_spin = (3.4641016151377544 * eta + 20.0830030082033 * eta2 - 12.333573402277912 * eta3) / (
        1 + 7.2388440419467335 * eta
    )

    eq_spin = (m1_sq + m2_sq) * s + (
        (-0.8561951310209386 * eta - 0.09939065676370885 * eta2 + 1.668810429851045 * eta3) * s
        + (0.5881660363307388 * eta - 2.149269067519131 * eta2 + 3.4768263932898678 * eta3) * s2
        + (0.142443244743048 * eta - 0.9598353840147513 * eta2 + 1.9595643107593743 * eta3) * s3
    ) / (1 + (-0.9142232693081653 + 2.3191363426522633 * eta - 9.710576749140989 * eta3) * s)

    uneq_spin = (
        0.3223660562764661 * dchi * delta * (1 + 9.332575956437443 * eta) * eta2
        - 0.059808322561702126 * dchi2 * eta3
        + 2.3170397514509933 * dchi * delta * (1 - 3.2624649875884852 * eta) * eta3 * s
    )
    return no_spin + eq_spin + uneq_spin


def xlal_sim_imr_phenom_x_f_meco(eta: float, chi1_l: float, chi2_l: float) -> float:
    """Compute the MECO frequency.

    Args:
        eta: Symmetric mass ratio.
        chi1_l: Dimensionless spin component of the first body along the orbital angular momentum.
        chi2_l: Dimensionless spin component of the second body along the orbital angular momentum.

    Returns:
        MECO frequency.
    """

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    delta = jnp.sqrt(1.0 - 4.0 * eta)

    s = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1_l, chi2_l)
    s2 = s * s
    s3 = s2 * s
    # //S4    = (S3*S)

    dchi = chi1_l - chi2_l
    dchi2 = dchi * dchi

    no_spin = (
        0.018744340279608845 + 0.0077903147004616865 * eta + 0.003940354686136861 * eta2 - 0.00006693930988501673 * eta3
    ) / (1.0 - 0.10423384680638834 * eta)

    eq_spin = (
        s
        * (
            0.00027180386951683135
            - 0.00002585252361022052 * s
            + eta4 * (-0.0006807631931297156 + 0.022386313074011715 * s - 0.0230825153005985 * s2)
            + eta2 * (0.00036556167661117023 - 0.000010021140796150737 * s - 0.00038216081981505285 * s2)
            + eta * (0.00024422562796266645 - 0.00001049013062611254 * s - 0.00035182990586857726 * s2)
            + eta3 * (-0.0005418851224505745 + 0.000030679548774047616 * s + 4.038390455349854e-6 * s2)
            - 0.00007547517256664526 * s2
        )
    ) / (
        0.026666543809890402
        + (
            -0.014590539285641243
            - 0.012429476486138982 * eta
            + 1.4861197211952053 * eta4
            + 0.025066696514373803 * eta2
            + 0.005146809717492324 * eta3
        )
        * s
        + (
            -0.0058684526275074025
            - 0.02876774751921441 * eta
            - 2.551566872093786 * eta4
            - 0.019641378027236502 * eta2
            - 0.001956646166089053 * eta3
        )
        * s2
        + (
            0.003507640638496499
            + 0.014176504653145768 * eta
            + 1.0 * eta4
            + 0.012622225233586283 * eta2
            - 0.00767768214056772 * eta3
        )
        * s3
    )

    uneq_spin = dchi2 * (0.00034375176678815234 + 0.000016343732281057392 * eta) * eta2 + dchi * delta * eta * (
        0.08064665214195679 * eta2
        + eta * (-0.028476219509487793 - 0.005746537021035632 * s)
        - 0.0011713735642446144 * s
    )

    return no_spin + eq_spin + uneq_spin


def xlal_sim_imr_phenom_x_f_isco(chi_f: float) -> float:
    """Compute the ISCO frequency.

    Args:
        chi_f: Dimensionless spin of the final black hole.

    Returns:
        ISCO frequency.
    """

    z1 = 1.0 + (1.0 - chi_f * chi_f) ** (1 / 3) * ((1 + chi_f) ** (1 / 3) + (1 - chi_f) ** (1 / 3))
    z1 = lax.select(z1 > 3, 3.0, z1)  # Finite precision may give Z1>3, but this can not happen.
    z2 = jnp.sqrt(3.0 * chi_f * chi_f + z1 * z1)

    r_isco = 3.0 + z2 - xlal_sim_imr_phenom_x_sign(chi_f) * jnp.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    r_isco_sq = jnp.sqrt(r_isco)
    r_isco_3o2 = r_isco_sq * r_isco_sq * r_isco_sq

    omega_isco = 1.0 / (r_isco_3o2 + chi_f)

    return omega_isco / PI


def xlal_sim_imr_phenom_x_sign(x: float) -> float:
    """Return the sign of x: 1.0 if x > 0, -1.0 if x < 0, 0.0 if x == 0.

    Args:
        x: Input value.

    Returns:
        Sign of x.
    """
    return lax.select(  # 1.0 if x > 0. else (-1.0 if x < 0.0 else 0.0)
        x > 0.0,
        1.0,
        lax.select(
            x < 0.0,
            -1.0,
            0.0,
        ),
    )
