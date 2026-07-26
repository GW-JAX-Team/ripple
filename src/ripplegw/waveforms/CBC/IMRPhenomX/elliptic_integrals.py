"""
JAX implementation of GSL elliptic integral functions.

This module provides JAX-compatible implementations of elliptic integrals,
specifically the incomplete elliptic integral of the first kind (F).
"""

import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from ripplegw.typing import FloatLike


def ellint_F(phi: FloatLike, k: FloatLike, n_points: int = 1000) -> FloatLike:
    """
    Compute the incomplete elliptic integral of the first kind.

    This function computes F(phi, k) using the Legendre form:
    F(phi, k) = integral_0^phi dt / sqrt(1 - k^2 sin^2(t))

    This is equivalent to GSL's gsl_sf_ellint_F(phi, k, GSL_PREC_DOUBLE).

    Args:
        phi: The amplitude (upper limit of integration) in radians
        k: The modulus, where 0 <= k^2 <= 1 (note: this is k, not m = k^2)
        n_points: Number of integration points (default: 1000)

    Returns:
        The value of F(phi, k)

    Notes:
        - For k = 0: F(phi, 0) = phi
        - For |k| = 1 and |phi| < pi/2: F(phi, +/-1) = arctanh(sin(phi))
        - The implementation uses numerical integration via trapezoidal rule
        - JAX-compatible (can be used in jit, grad, vmap, etc.)

    Examples:
        >>> ellint_F(jnp.pi/2, 0.5)  # Complete elliptic integral K(0.5)
        >>> ellint_F(jnp.pi/4, 0.8)  # Incomplete elliptic integral
    """
    phi = jnp.asarray(phi)
    k = jnp.asarray(k)
    k2: FloatLike = k * k

    # Handle special case: k = 0
    # F(phi, 0) = phi
    def case_k_zero():
        return phi

    # Handle special case: |k| = 1
    # F(phi, +/-1) = arctanh(sin(phi)) = 0.5 * ln((1 + sin(phi)) / (1 - sin(phi)))
    def case_k_one():
        sin_phi = jnp.sin(phi)
        # Use arctanh for numerical stability
        return jnp.arctanh(sin_phi)

    # General case: numerical integration
    def case_general():
        # Create integration points from 0 to phi
        t = jnp.linspace(0.0, phi, n_points)

        # Compute integrand: 1 / √(1 - k² sin²(t))
        sin_t = jnp.sin(t)
        integrand = 1.0 / jnp.sqrt(1.0 - k2 * sin_t * sin_t)

        # Integrate using trapezoidal rule
        result = trapezoid(integrand, t)
        return result

    # Select appropriate case based on k value
    abs_k = jnp.abs(k)
    is_k_zero = abs_k < 1e-10
    is_k_one = jnp.abs(abs_k - 1.0) < 1e-10

    # Nested conditional: check k=0 first, then k=1, then general
    inner: FloatLike = jnp.where(is_k_one, case_k_one(), case_general())
    result: FloatLike = jnp.where(is_k_zero, case_k_zero(), inner)

    return result


def gsl_sf_elljac_e(
    u: FloatLike, m: FloatLike, max_iter: int = 16
) -> tuple[FloatLike, FloatLike, FloatLike]:
    """
    Compute the Jacobian elliptic functions sn(u|m), cn(u|m), dn(u|m).

    This function computes the Jacobian elliptic functions using the descending
    Landen transformation, which is the same algorithm used by GSL.

    The Jacobian elliptic functions are defined by the inverse of the elliptic
    integral of the first kind:
        u = F(phi, k) = integral_0^phi dt / sqrt(1 - k^2 sin^2(t))

    Then:
        sn(u|m) = sin(phi)
        cn(u|m) = cos(phi)
        dn(u|m) = sqrt(1 - k^2 sin^2(phi))

    where m = k^2 is the parameter (0 <= m <= 1).

    This is equivalent to GSL's gsl_sf_elljac_e(u, m, &sn, &cn, &dn).

    Args:
        u: The argument
        m: The parameter (m = k^2, where k is the modulus), 0 <= m <= 1
        max_iter: Maximum number of Landen transformations (default: 16)

    Returns:
        A tuple (sn, cn, dn) containing the three Jacobian elliptic functions

    Notes:
        - For m = 0: sn(u|0) = sin(u), cn(u|0) = cos(u), dn(u|0) = 1
        - For m = 1: sn(u|1) = tanh(u), cn(u|1) = sech(u), dn(u|1) = sech(u)
        - The implementation uses the descending Landen transformation for accuracy
        - JAX-compatible (can be used in jit, grad, vmap, etc.)

    References:
        - Abramowitz & Stegun, Chapter 16
        - GSL library: gsl_sf_elljac.c
        - NIST DLMF: https://dlmf.nist.gov/22.20

    Examples:
        >>> sn, cn, dn = gsl_sf_elljac_e(1.0, 0.5)
        >>> sn, cn, dn = gsl_sf_elljac_e(jnp.array([0.5, 1.0]), 0.8)
    """
    u = jnp.asarray(u)
    m = jnp.asarray(m)

    # Descending Landen transformation (Abramowitz & Stegun 16.14.1-2, same as GSL).
    #
    # Special cases (m≈0, m≈1) are intentionally omitted:
    #   - The only caller (IMRPhenomX_Return_SNorm_MSA) gates the call behind
    #     cancel_condition = |Smi2 - Spl2| < 1e-5, which corresponds to m→0;
    #     when that condition is True, sn_jacobi is not used, so m=0 is never
    #     reached here in practice.
    #   - m=1 requires S32 = Smi2 exactly, which cannot occur in float64 for
    #     non-degenerate spin parameters.
    #   - JAX's jnp.where evaluates both branches eagerly; omitting the dead-code
    #     paths (2 sin/cos + 3 tanh/cosh + 6 where/comparison ops per element)
    #     reduces the compiled kernel size and GPU instruction count.
    #
    # Only sn is returned: cn and dn are always discarded by callers (_cn, _dn).

    k = jnp.sqrt(m)
    a, c = jnp.ones_like(k), k

    # Forward Landen — Python for-loop so each (a_i, c_i) pair is a graph-node
    # value rather than a slice of a materialised [max_iter, ...] scan-output
    # tensor.  When vmapped over N samples this keeps the AGM intermediates in
    # GPU registers instead of allocating 2 × max_iter × N_samples × N_freq
    # arrays in HBM — critical for PE runs in Jim where N_samples can be O(1000).
    ac_pairs = []
    for _ in range(max_iter):
        b = jnp.sqrt((a - c) * (a + c))
        a, c = 0.5 * (a + b), 0.5 * (a - b)
        ac_pairs.append((a, c))

    phi_n = (
        (2.0**max_iter) * a * u
    )  # a is a_final here; 2^max_iter is a Python constant

    # Backward Landen — same reasoning: Python loop, no scan output tensor.
    phi = phi_n
    for a_i, c_i in reversed(ac_pairs):
        sin_phi = jnp.sin(phi)
        arg = jnp.clip(c_i * sin_phi / a_i, -1.0, 1.0)
        phi = 0.5 * (phi + jnp.arcsin(arg))

    phi_0 = phi

    sn = jnp.sin(phi_0)
    cn = jnp.cos(phi_0)
    dn = jnp.sqrt(1.0 - m * sn * sn)
    return sn, cn, dn
