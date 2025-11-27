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
#  * Total spin normalised to [-1,1]
#  */
def xlal_sim_imr_phenom_x_stot_r(eta: float, chi1L: float, chi2L: float) -> float:
	# // Convention m1 >= m2 and chi1z is the spin projected along Lz on m1
	delta = jnp.sqrt(1.0 - 4.0*eta)
	m1    = 0.5*(1 + delta)
	m2    = 0.5*(1 - delta)
	m1s   = m1*m1
	m2s   = m2*m2

	return ((m1s * chi1L + m2s * chi2L) / (m1s + m2s))


# /**
#  * Spin difference
#  */
def xlal_sim_imr_phenom_x_dchi(chi1L: float, chi2L: float) -> float:
    return chi1L - chi2L


# /**
#  * Final Mass = 1 - Energy Radiated,  X Jimenez-Forteza et al, PRD, 95, 064024, (2017), arXiv:1611.00332
#  */
def xlal_sim_imr_phenom_x_final_mass_2017(eta, chi1L, chi2L):

    delta  = jnp.sqrt(1.0 - 4.0*eta)
    eta2   =  eta*eta
    eta3   = eta2*eta
    eta4   = eta3*eta

    S      = xlal_sim_imr_phenom_x_stot_r(eta,chi1L,chi2L)
    S2     =  S*S
    S3     = S2*S

    dchi   = chi1L - chi2L
    dchi2  = dchi*dchi

    noSpin = 0.057190958417936644*eta + 0.5609904135313374*eta2 - 0.84667563764404*eta3 + 3.145145224278187*eta4

    #   /* Because of the way this is written, we need to subtract the noSpin term */
    eqSpin = ((0.057190958417936644*eta + 0.5609904135313374*eta2 - 0.84667563764404*eta3 + 3.145145224278187*eta4)* \
    (    1 \
        + (-0.13084389181783257 - 1.1387311580238488*eta + 5.49074464410971*eta2)*S \
        + (-0.17762802148331427 + 2.176667900182948*eta2)*S2 \
        + (-0.6320191645391563 + 4.952698546796005*eta - 10.023747993978121*eta2)*S3)) \
        / (1 + (-0.9919475346968611 + 0.367620218664352*eta + 4.274567337924067*eta2)*S)

    eqSpin = eqSpin - noSpin

    uneqSpin =  - 0.09803730445895877*dchi*delta*(1 - 3.2283713377939134*eta)*eta2 \
                + 0.01118530335431078*dchi2*eta3 \
                - 0.01978238971523653*dchi*delta*(1 - 4.91667749015812*eta)*eta*S

    #   /* Mfinal = 1 - Erad, assuming that M = m1 + m2 = 1 */
    return (1.0 - (noSpin + eqSpin + uneqSpin))


# /**
#  * Final Dimensionless Spin,  X Jimenez-Forteza et al, PRD, 95, 064024, (2017), arXiv:1611.00332
#  */
def xlal_sim_imr_phenom_x_final_spin_2017(eta, chi1L, chi2L):
    delta  = jnp.sqrt(1.0 - 4.0*eta)
    m1     = 0.5 * (1.0 + delta)
    m2     = 0.5 * (1.0 - delta)
    m1Sq   = m1*m1
    m2Sq   = m2*m2

    eta2   = eta*eta
    eta3   = eta2*eta

    # //S  = (m1Sq * chi1L + m2Sq * chi2L) / (m1Sq + m2Sq)
    S  = xlal_sim_imr_phenom_x_stot_r(eta,chi1L,chi2L)
    S2 =  S*S
    S3 = S2*S

    dchi  = chi1L - chi2L
    dchi2 = dchi*dchi


    noSpin = (3.4641016151377544*eta + 20.0830030082033*eta2 - 12.333573402277912*eta3)/(1 + 7.2388440419467335*eta)

    eqSpin = (m1Sq + m2Sq)*S \
    + ((-0.8561951310209386*eta - 0.09939065676370885*eta2 + 1.668810429851045*eta3)*S \
    + (0.5881660363307388*eta - 2.149269067519131*eta2 + 3.4768263932898678*eta3)*S2 \
    + (0.142443244743048*eta - 0.9598353840147513*eta2 + 1.9595643107593743*eta3)*S3) \
    / (1 + (-0.9142232693081653 + 2.3191363426522633*eta - 9.710576749140989*eta3)*S)

    uneqSpin = 0.3223660562764661*dchi*delta*(1 + 9.332575956437443*eta)*eta2 \
    - 0.059808322561702126*dchi2*eta3 \
    + 2.3170397514509933*dchi*delta*(1 - 3.2624649875884852*eta)*eta3*S \

    return (noSpin + eqSpin + uneqSpin)


def xlal_sim_imr_phenom_x_fMECO(eta, chi1L, chi2L):

    eta2  = (eta*eta)
    eta3  = (eta2*eta)
    eta4  = (eta3*eta)

    delta = jnp.sqrt(1.0 - 4.0*eta)

    S     = xlal_sim_imr_phenom_x_chi_pn_hat(eta,chi1L,chi2L)
    S2    = (S*S)
    S3    = (S2*S)
    #//S4    = (S3*S)

    dchi  = chi1L - chi2L
    dchi2 = (dchi*dchi)

    noSpin = (0.018744340279608845 + 0.0077903147004616865*eta + 0.003940354686136861*eta2 - 0.00006693930988501673*eta3)/(1. - 0.10423384680638834*eta)

    eqSpin = (S*(0.00027180386951683135 - 0.00002585252361022052*S + eta4*(-0.0006807631931297156 + 0.022386313074011715*S - 0.0230825153005985*S2) + eta2*(0.00036556167661117023 - 0.000010021140796150737*S - 0.00038216081981505285*S2) + eta*(0.00024422562796266645 - 0.00001049013062611254*S - 0.00035182990586857726*S2) + eta3*(-0.0005418851224505745 + 0.000030679548774047616*S + 4.038390455349854e-6*S2) - 0.00007547517256664526*S2))/(0.026666543809890402 + (-0.014590539285641243 - 0.012429476486138982*eta + 1.4861197211952053*eta4 + 0.025066696514373803*eta2 + 0.005146809717492324*eta3)*S + (-0.0058684526275074025 - 0.02876774751921441*eta - 2.551566872093786*eta4 - 0.019641378027236502*eta2 - 0.001956646166089053*eta3)*S2 + (0.003507640638496499 + 0.014176504653145768*eta + 1.*eta4 + 0.012622225233586283*eta2 - 0.00767768214056772*eta3)*S3)

    uneqSpin = dchi2*(0.00034375176678815234 + 0.000016343732281057392*eta)*eta2 + dchi*delta*eta*(0.08064665214195679*eta2 + eta*(-0.028476219509487793 - 0.005746537021035632*S) - 0.0011713735642446144*S)

    return (noSpin + eqSpin + uneqSpin)


def xlal_sim_imr_phenom_x_fISCO(chif):

    Z1 = 1.0 + (1.0 - chif*chif)**(1/3) * ((1 + chif)**(1/3) + (1 - chif)**(1/3))
    if Z1>3:
        Z1=3. #Finite precission may give Z1>3, but this can not happen.
    Z2 = jnp.sqrt(3.0*chif*chif + Z1*Z1)

    rISCO    = 3.0 + Z2 - xlal_sim_imr_phenom_x_sign(chif)*jnp.sqrt( (3 - Z1) * (3 + Z1 + 2*Z2) )
    rISCOsq  = jnp.sqrt(rISCO)
    rISCO3o2 = rISCOsq * rISCOsq * rISCOsq

    OmegaISCO = 1.0 / ( rISCO3o2 + chif)

    return OmegaISCO / PI 


def xlal_sim_imr_phenom_x_sign(x):
    return 1.0 if x > 0. else (-1.0 if x < 0.0 else 0.0)