import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from .LALSimIMRPhenomX_PNR_coefficients import *


@pytree_dataclass
class IMRPhenomX_PNR_alpha_parameters:
    """
    Parameter structure for alpha angle coefficients.
    Reference: Sec. 8D in arXiv:2107.08876
    """

    A1: float
    A2: float
    A3: float
    A4: float


def IMRPhenomX_PNR_precompute_alpha_coefficients(
    pWF: dict, pPrec
) -> IMRPhenomX_PNR_alpha_parameters:
    """
    Precompute alpha coefficients for PNR angles.

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
    IMRPhenomX_PNR_alpha_parameters
        Alpha parameter struct with A1, A2, A3, A4 coefficients
    """

    # Determine eta based on precession version
    # If version==330 and eta < 0.09, use 0.09; otherwise use pWF['eta']
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

    # Evaluate coefficients as described in Sec. 8D in arXiv:2107.08876
    # A2: clamp to <= 0
    A2_raw = IMRPhenomX_PNR_alpha_A2_coefficient(eta, chi, costheta)
    A2_clamped = jnp.where(A2_raw > 0.0, 0.0, A2_raw)

    # A3: take absolute value and clamp to >= 0.00001
    A3_raw = jnp.abs(IMRPhenomX_PNR_alpha_A3_coefficient(eta, chi, costheta))
    A3 = jnp.where(A3_raw < 0.00001, 0.00001, A3_raw)

    # Compute maximum allowed A2 based on A3
    maxA2 = jnp.pi * jnp.pi * jnp.sqrt(A3)

    # Clamp A2 to >= -maxA2
    A2 = jnp.where(A2_clamped < -maxA2, -maxA2, A2_clamped)

    # A1: take absolute value
    A1 = jnp.abs(IMRPhenomX_PNR_alpha_A1_coefficient(eta, chi, costheta))

    # A4: no clamping needed
    A4 = IMRPhenomX_PNR_alpha_A4_coefficient(eta, chi, costheta)

    return IMRPhenomX_PNR_alpha_parameters(A1=A1, A2=A2, A3=A3, A4=A4)
