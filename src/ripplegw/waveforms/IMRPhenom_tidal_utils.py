"""
Small utility script for shared functions between tidal waveforms, especially for NRTidalv2
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def universal_relation(coeffs: Float[Array, "5"], x: Float) -> Float:
    """
    Applies the general formula of a universal relationship, which is a quartic polynomial.

    Args:
        coeffs (Array): Array of coefficients for the quartic polynomial, starting from the constant term and going to the fourth order.
        x (Float): Variable of quartic polynomial

    Returns:
        Float: Result of universal relation
    """
    return (
        coeffs[0]
        + coeffs[1] * x
        + coeffs[2] * (x**2)
        + coeffs[3] * (x**3)
        + coeffs[4] * (x**4)
    )


def get_quadparam_octparam(lambda_: Float) -> tuple[Float, Float]:
    """
    Compute the quadrupole and octupole parameter by checking the value of lambda and choosing the right subroutine.
    If lambda is smaller than 1, we make use of the fit formula as given by the LAL source code. Otherwise, we rely on the equations of
    the NRTidalv2 paper to get these parameters.

    Args:
        lambda_ (Float): Tidal deformability of object.

    Returns:
        tuple[Float, Float]: Quadrupole and octupole parameters.
    """
    is_low_lambda = lambda_ < 1
    return jax.lax.cond(
        is_low_lambda,
        _get_quadparam_octparam_low,
        _get_quadparam_octparam_high,
        lambda_,
    )


def get_quadparam_octparam_eos(lambda_: Float) -> tuple[Float, Float]:
    """
    Compute the quadrupole and octupole parameters using LAL's
    XLALSimInspiralEOSQfromLambda relation.
    """

    quad_coeffs = jnp.array([0.194, 0.0936, 0.0474, -0.00421, 0.000123])
    oct_coeffs = jnp.array([0.003131, 2.071, -0.7152, 0.2458, -0.03309])

    quadparam = jax.lax.cond(
        lambda_ < 0.5,
        lambda _: jnp.asarray(1.0),
        lambda lam: jnp.exp(universal_relation(quad_coeffs, jnp.log(lam))),
        lambda_,
    )
    octparam = jnp.exp(universal_relation(oct_coeffs, jnp.log(quadparam)))

    return quadparam, octparam


def _get_quadparam_octparam_low(lambda_: Float) -> tuple[Float, Float]:
    """
    Computes quadparameter following LALSimUniversalRelations.c of lalsuite

    Version for lambdas smaller than 1.

    LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
    Args:
        lambda_: tidal deformability

    Returns:
        quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
        octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
    """

    # Coefficients of universal relation
    oct_coeffs = jnp.array([0.003131, 2.071, -0.7152, 0.2458, -0.03309])

    # Extension of the fit in the range lambda2 = [0,1.] so that the BH limit is enforced, lambda2bar->0 gives quadparam->1. and the junction with the universal relation is smooth, of class C2
    quadparam = 1.0 + lambda_ * (
        0.427688866723244
        + lambda_ * (-0.324336526985068 + lambda_ * 0.1107439432180572)
    )
    log_quadparam = jnp.log(quadparam)

    # Compute octparam:
    log_octparam = universal_relation(oct_coeffs, log_quadparam)
    octparam = jnp.exp(log_octparam)

    return quadparam, octparam


def _get_quadparam_octparam_high(lambda_: Float) -> tuple[Float, Float]:
    """
    Computes quadparameter, following LALSimUniversalRelations.c of lalsuite

    Version for lambdas greater than 1.

    LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
    Args:
        lambda_: tidal deformability

    Returns:
        quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
        octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
    """

    # Coefficients of universal relation
    quad_coeffs = jnp.array([0.1940, 0.09163, 0.04812, -4.283e-3, 1.245e-4])
    oct_coeffs = jnp.array([0.003131, 2.071, -0.7152, 0.2458, -0.03309])

    # High lambda (above 1): use universal relation
    log_lambda = jnp.log(lambda_)
    log_quadparam = universal_relation(quad_coeffs, log_lambda)

    # Compute octparam:
    log_octparam = universal_relation(oct_coeffs, log_quadparam)

    quadparam = jnp.exp(log_quadparam)
    octparam = jnp.exp(log_octparam)

    return quadparam, octparam


def get_kappa(theta: Float[Array, "6"]) -> Float:
    """
    Computes the tidal deformability parameter kappa according to equation (8) of the NRTidalv2 paper.

    Args:
        theta (Array): Intrinsic parameters m1, m2, chi1, chi2, lambda1, lambda2

    Returns:
        Float: kappa_eff^T from equation (8) of NRTidalv2 paper.
    """

    # Auxiliary variables
    m1, m2, _, _, lambda1, lambda2 = theta
    M = m1 + m2
    X1 = m1 / M
    X2 = m2 / M

    # Get kappa
    term1 = (1.0 + 12.0 * X2 / X1) * (X1**5.0) * lambda1
    term2 = (1.0 + 12.0 * X1 / X2) * (X2**5.0) * lambda2
    kappa = (3.0 / 13.0) * (term1 + term2)

    return kappa
