import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from .LALSimIMRPhenomX_PNR_coefficients import (
    IMRPhenomX_PNR_beta_B0_coefficient,
    IMRPhenomX_PNR_beta_B1_coefficient,
    IMRPhenomX_PNR_beta_B2_coefficient,
    IMRPhenomX_PNR_beta_B3_coefficient,
    IMRPhenomX_PNR_beta_B4_coefficient,
    IMRPhenomX_PNR_beta_B5_coefficient,
)
from ripplegw.constants import PI


@pytree_dataclass
class IMRPhenomX_PNR_beta_parameters:
    """
    Parameter structure for beta angle coefficients.
    Reference: Sec. 8D in arXiv:2107.08876
    """

    B0: float
    B1: float
    B2: float
    B3: float
    B4: float
    B5: float


def IMRPhenomX_PNR_precompute_beta_coefficients(
    pWF: dict, pPrec
) -> IMRPhenomX_PNR_beta_parameters:
    """
    Precompute beta coefficients for PNR angles.

    JAX-friendly implementation without if-else branches.
    Reference: Sec. 8D in arXiv:2107.08876

    Parameters
    ----------
    pWF : dict
        PhenomX waveform struct
    pPrec : IMRPhenomXGetAndSetPrecessionVariables
        PhenomX precession struct

    Returns
    -------
    IMRPhenomX_PNR_beta_parameters
        Beta parameter struct with B0, B1, B2, B3, B4, B5 coefficients
    """

    # Determine eta based on precession version
    # If version==330 and eta < 0.09, use 0.09 otherwise use pWF['eta']
    eta_330 = jnp.where(pWF["eta"] >= 0.09, pPrec.eta, 0.09)
    eta = jnp.where(pPrec.IMRPhenomXPrecVersion == 330, eta_330, pWF["eta"])

    # Compute chi boundary
    chiboundary = 0.80 - 0.20 * jnp.exp(-jnp.power((pWF["q"] - 6.0) / 1.5, 8))

    # Determine chi based on precession version
    # If version==330 and chi_singleSpin > chiboundary, use chiboundary
    chi_330 = jnp.where(
        pPrec.chi_singleSpin <= chiboundary, pPrec.chi_singleSpin, chiboundary
    )
    chi = jnp.where(pPrec.IMRPhenomXPrecVersion == 330, chi_330, pPrec.chi_singleSpin)

    costheta = pPrec.costheta_singleSpin

    # Approximate orientation of final spin
    costhetaf = pPrec.costheta_final_singleSpin

    # Compute B4 and ensure it's sufficiently large (>= 175.0)
    B4_raw = IMRPhenomX_PNR_beta_B4_coefficient(eta, chi, costheta)
    B4 = jnp.where(B4_raw <= 175.0, 175.0, B4_raw)

    # Compute all beta coefficients
    B0 = jnp.arccos(costhetaf) - IMRPhenomX_PNR_beta_B0_coefficient(eta, chi, costheta)
    B1 = IMRPhenomX_PNR_beta_B1_coefficient(eta, chi, costheta)
    B2 = IMRPhenomX_PNR_beta_B2_coefficient(eta, chi, costheta)
    B3_coeff = IMRPhenomX_PNR_beta_B3_coefficient(eta, chi, costheta)
    B3 = B2 * B3_coeff
    B5 = IMRPhenomX_PNR_beta_B5_coefficient(eta, chi, costheta)

    return IMRPhenomX_PNR_beta_parameters(B0=B0, B1=B1, B2=B2, B3=B3, B4=B4, B5=B5)


def IMRPhenomX_PNR_MR_ddbeta_expression(
    Mf: float, betaParams: IMRPhenomX_PNR_beta_parameters
) -> float:
    """
    Second derivative of beta in merger-ringdown regime.

    Reference: Lines 698-716 in LALSimIMRPhenomX_PNR_beta.c

    Parameters
    ----------
    Mf : float
        Geometric frequency
    betaParams : IMRPhenomX_PNR_beta_parameters
        Beta parameter struct

    Returns
    -------
    float
        Second derivative of MR beta at Mf
    """
    B1 = betaParams.B1
    B2 = betaParams.B2
    B3 = betaParams.B3
    B4 = betaParams.B4
    B5 = betaParams.B5

    a = B2 * B4 * B4 - 2.0 * B3 * B4 * B4 * B5
    b = -3.0 * B3 * B4 + 3.0 * B1 * B4 * B4 - 3.0 * B3 * B4 * B4 * B5 * B5
    c = -3.0 * B2 * B4 + 6.0 * B1 * B4 * B4 * B5 - 3.0 * B2 * B4 * B4 * B5 * B5
    d = (
        B3
        - B1 * B4
        - 2.0 * B2 * B4 * B5
        + 2.0 * B3 * B4 * B5 * B5
        + 3.0 * B1 * B4 * B4 * B5 * B5
        - 2.0 * B2 * B4 * B4 * B5 * B5 * B5
        + B3 * B4 * B4 * B5 * B5 * B5 * B5
    )

    return (
        2.0
        * (a * Mf * Mf * Mf + b * Mf * Mf + c * Mf + d)
        / jnp.power((1.0 + B4 * (B5 + Mf) * (B5 + Mf)), 3)
    )


def IMRPhenomX_PNR_MR_dddbeta_expression(
    Mf: float, betaParams: IMRPhenomX_PNR_beta_parameters
) -> float:
    """
    Third derivative of beta in merger-ringdown regime.

    Reference: Lines 721-740 in LALSimIMRPhenomX_PNR_beta.c

    Parameters
    ----------
    Mf : float
        Geometric frequency
    betaParams : IMRPhenomX_PNR_beta_parameters
        Beta parameter struct

    Returns
    -------
    float
        Third derivative of MR beta at Mf
    """
    B1 = betaParams.B1
    B2 = betaParams.B2
    B3 = betaParams.B3
    B4 = betaParams.B4
    B5 = betaParams.B5

    a = -B2 * B4 * B4 + 2.0 * B3 * B4 * B4 * B5
    b = 4.0 * B3 * B4 - 4.0 * B1 * B4 * B4 + 4.0 * B3 * B4 * B4 * B5 * B5
    c = 6.0 * B2 * B4 - 12.0 * B1 * B4 * B4 * B5 + 6.0 * B2 * B4 * B4 * B5 * B5
    d = (
        -4.0 * B3
        + 4.0 * B1 * B4
        + 8.0 * B2 * B4 * B5
        - 8.0 * B3 * B4 * B5 * B5
        - 12.0 * B1 * B4 * B4 * B5 * B5
        + 8.0 * B2 * B4 * B4 * B5 * B5 * B5
        - 4.0 * B3 * B4 * B4 * B5 * B5 * B5 * B5
    )
    e = (
        -B2
        - 2.0 * B3 * B5
        + 4.0 * B1 * B4 * B5
        + 2.0 * B2 * B4 * B5 * B5
        - 4.0 * B3 * B4 * B5 * B5 * B5
        - 4.0 * B1 * B4 * B4 * B5 * B5 * B5
        + 3.0 * B2 * B4 * B4 * B5 * B5 * B5 * B5
        - 2.0 * B3 * B4 * B4 * B5 * B5 * B5 * B5 * B5
    )

    return (
        6.0
        * B4
        * (a * Mf * Mf * Mf * Mf + b * Mf * Mf * Mf + c * Mf * Mf + d * Mf + e)
        / jnp.power((1.0 + B4 * (B5 + Mf) * (B5 + Mf)), 4)
    )


def IMRPhenomX_PNR_single_inflection_point(
    betaParams: IMRPhenomX_PNR_beta_parameters,
) -> float:
    """
    Compute inflection point for beta connection frequency.

    Simplified JAX implementation that assumes the cubic case (a != 0)
    and selects the appropriate root based on the conditions in the C code.

    Reference: Lines 1014-1119 in LALSimIMRPhenomX_PNR_beta.c

    Parameters
    ----------
    betaParams : IMRPhenomX_PNR_beta_parameters
        Beta parameter struct

    Returns
    -------
    float
        Inflection frequency
    """
    B1 = betaParams.B1
    B2 = betaParams.B2
    B3 = betaParams.B3
    B4 = betaParams.B4
    B5 = betaParams.B5

    # Cubic coefficients: ax^3 + bx^2 + cx + d
    a = 2.0 * (B2 * B4 * B4 - 2.0 * B3 * B4 * B4 * B5)
    b = 6.0 * (-B3 * B4 + B1 * B4 * B4 - B3 * B4 * B4 * B5 * B5)
    c = 6.0 * (-B2 * B4 + 2.0 * B1 * B4 * B4 * B5 - B2 * B4 * B4 * B5 * B5)
    d = 2.0 * (
        B3
        - B1 * B4
        - 2.0 * B2 * B4 * B5
        + 2.0 * B3 * B4 * B5 * B5
        + 3.0 * B1 * B4 * B4 * B5 * B5
        - 2.0 * B2 * B4 * B4 * B5 * B5 * B5
        + B3 * B4 * B4 * B5 * B5 * B5 * B5
    )

    # Convert to depressed cubic: t^3 + rt + s
    r = (3.0 * a * c - b * b) / (3.0 * a * a)
    s = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a)

    phi = jnp.arccos(((3.0 * s) / (2.0 * r)) * jnp.sqrt(-3.0 / r))

    # Compute three roots (using n=0, 1, 2)
    f0 = 2.0 * jnp.sqrt(-r / 3.0) * jnp.cos(phi / 3.0) - b / (3.0 * a)
    f1 = 2.0 * jnp.sqrt(-r / 3.0) * jnp.cos((phi - 2.0 * jnp.pi) / 3.0) - b / (3.0 * a)
    f2 = 2.0 * jnp.sqrt(-r / 3.0) * jnp.cos((phi - 4.0 * jnp.pi) / 3.0) - b / (3.0 * a)

    # Selection logic from C code (lines 1098-1114)
    # If a < 0, take central root (f1)
    # If a > 0, check condition and select f0 or f2
    central_root = f1
    conditional_root = jnp.where(b / (3.0 * a) > B5 / 2.0 - (2141.0 / 90988.0), f0, f2)

    f_inflection = jnp.where(a < 0.0, central_root, conditional_root)

    return f_inflection


def IMRPhenomX_PNR_BetaConnectionFrequencies(
    betaParams: IMRPhenomX_PNR_beta_parameters,
) -> tuple:
    """
    Compute connection frequencies for beta angle.

    JAX-friendly implementation of connection frequency calculation
    outlined in Sec. 8B and 8D of arXiv:2107.08876.

    Reference: Lines 754-927 in LALSimIMRPhenomX_PNR_beta.c

    Parameters
    ----------
    betaParams : IMRPhenomX_PNR_beta_parameters
        Beta parameter struct with B0-B5 coefficients

    Returns
    -------
    tuple
        (Mf_beta_lower, Mf_beta_upper) - lower and upper connection frequencies
    """
    B1 = betaParams.B1
    B2 = betaParams.B2
    B3 = betaParams.B3
    B4 = betaParams.B4
    B5 = betaParams.B5

    # Calculate inflection frequency to get beta_inf
    Mf_inflection = IMRPhenomX_PNR_single_inflection_point(betaParams)

    # Calculate derivative of beta at inflection point
    # Using MR_dbeta_expression (lines 680-693 in C code)
    dbeta_inflection = (
        (2.0 * B3 * B4 * B5 - B2 * B4) * Mf_inflection * Mf_inflection
        + (2.0 * B3 - 2.0 * B1 * B4 + 2.0 * B3 * B4 * B5 * B5) * Mf_inflection
        + (B2 - 2.0 * B1 * B4 * B5 + B2 * B4 * B5 * B5)
    ) / jnp.power((1.0 + B4 * (Mf_inflection + B5) * (Mf_inflection + B5)), 2)

    # Compute extrema of beta (lines 793-796)
    root_term = B4 * (B2 - 2.0 * B3 * B5) * (
        B2 - 2.0 * B1 * B4 * B5 + B2 * B4 * B5 * B5
    ) + (B3 - B1 * B4 + B3 * B4 * B5 * B5) * (B3 - B1 * B4 + B3 * B4 * B5 * B5)

    Mf_plus = ((B3 - B1 * B4 + B3 * B4 * B5 * B5) + jnp.sqrt(root_term)) / (
        B4 * (B2 - 2.0 * B3 * B5)
    )
    Mf_minus = ((B3 - B1 * B4 + B3 * B4 * B5 * B5) - jnp.sqrt(root_term)) / (
        B4 * (B2 - 2.0 * B3 * B5)
    )

    # Classify extrema using second derivative (lines 799-812)
    ddbeta_Mf_plus = IMRPhenomX_PNR_MR_ddbeta_expression(Mf_plus, betaParams)

    # If ddbeta > 0, Mf_plus is minimum otherwise it's maximum
    Mf_at_minimum = jnp.where(ddbeta_Mf_plus > 0.0, Mf_plus, Mf_minus)
    Mf_at_maximum = jnp.where(ddbeta_Mf_plus > 0.0, Mf_minus, Mf_plus)

    # Calculate derivatives at maximum (lines 815-816)
    ddbeta = IMRPhenomX_PNR_MR_ddbeta_expression(Mf_at_maximum, betaParams)
    dddbeta = IMRPhenomX_PNR_MR_dddbeta_expression(Mf_at_maximum, betaParams)

    # Maintain sign of dbeta_inflection in Eq. 56 (lines 818-830)
    sign = jnp.where(dbeta_inflection > 0.0, 1.0, -1.0)
    chosen_dbeta = (
        sign * ((dbeta_inflection / 100.0) * (dbeta_inflection / 100.0)) * 25.0
    )

    # Compute two possible square roots for Eq. 57 (lines 833-834)
    d1 = (
        1.0
        / dddbeta
        * (-ddbeta + jnp.sqrt(ddbeta * ddbeta + 2.0 * dddbeta * chosen_dbeta))
    )
    d2 = (
        1.0
        / dddbeta
        * (-ddbeta - jnp.sqrt(ddbeta * ddbeta + 2.0 * dddbeta * chosen_dbeta))
    )

    # Select delta based on logic from lines 840-855
    # Choose smallest positive if both positive, else d1 if d1>0, else d2
    both_positive = (d1 > 0.0) & (d2 > 0.0)
    delta = jnp.where(
        both_positive, jnp.where(d1 < d2, d1, d2), jnp.where(d1 > 0.0, d1, d2)
    )

    # Specify connection frequency based on Eq. 58 (lines 860-867)
    Mf_low = jnp.where(
        Mf_inflection >= 0.06, Mf_inflection - 0.03, 3.0 * Mf_inflection / 5.0
    )

    # Determine Mf_IM based on shape of beta (lines 875-899)
    # First two panels of Fig. 10: (Mf_at_minimum > Mf_at_maximum) or (Mf_inflection > Mf_at_maximum)
    first_case = (Mf_at_minimum > Mf_at_maximum) | (Mf_inflection > Mf_at_maximum)

    # Within first case: use Eq. 57 if Mf_at_maximum >= Mf_low, else use Eq. 58
    Mf_IM_first = jnp.where(Mf_at_maximum >= Mf_low, Mf_at_maximum + delta, Mf_low)

    # Right-most panel of Fig. 10
    Mf_IM_second = jnp.where(
        Mf_at_minimum > 0.06, Mf_at_minimum - 0.03, 3.0 * Mf_at_minimum / 5.0
    )

    Mf_IM = jnp.where(first_case, Mf_IM_first, Mf_IM_second)

    # Upper connection frequency (lines 905-912)
    Mf_MR = jnp.where(Mf_at_minimum > Mf_inflection, Mf_at_minimum, 100.0)

    # Failsafe: if Mf_IM < 0 or NaN, set both to 100.0 (lines 915-924)
    failsafe = (Mf_IM < 0.0) | jnp.isnan(Mf_IM)
    Mf_beta_lower = jnp.where(failsafe, 100.0, Mf_IM)
    Mf_beta_upper = jnp.where(failsafe, 100.0, Mf_MR)

    return Mf_beta_lower, Mf_beta_upper


def IMRPhenomX_PNR_arctan_window(beta: float) -> float:
    # specify the width as described in Sec. 8C of arXiv:2107.08876
    window_border = 0.01

    # if beta is close to zero or PI, window it
    def beta_close_to_zero_or_pi_branch():
        # precompute some things, then evaluate the arctan
        p = 0.002
        PI_by_2 = 1.570796326794897
        PI_by_2_1mp = 1.569378278348018
        PI_by_2_1oq = 7.308338225719002e97
        sign = jnp.where(beta < PI_by_2, 1.0, -1.0)

        return sign * (
            PI_by_2
            - PI_by_2_1mp
            * jnp.pow(jnp.atan2(jnp.pow(PI_by_2 - beta, 1.0 / p), PI_by_2_1oq), p)
        )

    return jax.lax.cond(
        jnp.logical_or(beta <= window_border, beta >= (PI - window_border)),
        beta_close_to_zero_or_pi_branch,
        lambda: beta,
        operand=None,
    )
