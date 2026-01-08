"""Implementation of LALSimIMRPhenomXPNRBetaCoefficients module in JAX."""

from __future__ import annotations

import jax.numpy as jnp

from ripplegw.typing import Array


def imr_phenom_x_pnr_evaluate_coefficient_array(coeff_array: Array, eta: float, chi: float, costheta: float) -> float:
    """
    Evaluate a polynomial sum over a 3D coefficient array.

    Parameters
    ----------
    coeff_array : Array
        Coefficient array of shape (4, 4, 5)
    eta : float
        Symmetric mass ratio
    chi : float
        Effective spin parameter
    costheta : float
        Cosine of the angle

    Returns
    -------
    float
        The polynomial sum
    """
    # Create power arrays for vectorized computation
    eta_powers = jnp.power(eta, jnp.arange(4))  # [eta^0, eta^1, eta^2, eta^3]
    chi_powers = jnp.power(chi, jnp.arange(4))  # [chi^0, chi^1, chi^2, chi^3]
    costheta_powers = jnp.power(costheta, jnp.arange(5))  # [costheta^0, ..., costheta^4]

    # Compute the polynomial sum using einsum for efficiency
    # coeff_array[i,j,k] * eta^i * chi^j * costheta^k
    poly_sum = jnp.einsum("ijk,i,j,k->", coeff_array, eta_powers, chi_powers, costheta_powers)

    return poly_sum


###############################################################################
############################### Beta coefficients##############################
###############################################################################

# Coefficient array for Bf parameter (pre-computed, constant)
_COEFF_ARRAY_BF = jnp.array(
    [
        [
            [3.09601897e00, 1.34032610e00, -1.45826218e00, -9.37928603e-01, 0.00000000e00],
            [0.00000000e00, -3.88910127e00, 3.71319679e00, -5.50316593e-01, 0.00000000e00],
            [-3.78818904e-01, 3.44727678e00, -3.74449485e00, 8.21928673e-01, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        ],
        [
            [-3.63088945e01, -1.38476321e01, 2.66368579e01, 6.74798082e00, 0.00000000e00],
            [0.00000000e00, 1.89698391e01, -1.58854735e01, -1.11744859e00, 0.00000000e00],
            [3.45434991e00, -1.73678802e01, 1.58282455e01, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        ],
        [
            [2.03218040e02, 6.36061063e01, -1.67985530e02, -8.74477395e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            [-7.62767762e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        ],
        [
            [-4.10212724e02, -1.31527994e02, 3.43405885e02, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        ],
    ]
)


def imr_phenom_x_pnr_beta_bf_coefficient(eta: float, chi: float, costheta: float) -> float:
    """
    Compute the Bf coefficient for IMRPhenomX_PNR beta angle.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio
    chi : float
        Effective spin parameter
    costheta : float
        Cosine of the angle

    Returns
    -------
    float
        The Bf coefficient multiplied by chi * sin(theta)
    """
    theta = jnp.arccos(costheta)

    bf = imr_phenom_x_pnr_evaluate_coefficient_array(_COEFF_ARRAY_BF, eta, chi, costheta)

    return chi * jnp.sin(theta) * bf
