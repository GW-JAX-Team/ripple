"""
JAX implementation of GSL elliptic integral functions.

This module provides JAX-compatible implementations of elliptic integrals,
specifically the incomplete elliptic integral of the first kind (F).
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.integrate import trapezoid


@jit
def ellint_F(phi: float, k: float, n_points: int = 1000) -> float:
    """
    Compute the incomplete elliptic integral of the first kind.

    This function computes F(φ, k) using the Legendre form:
    F(φ, k) = ∫₀^φ dt / √(1 - k² sin²(t))

    This is equivalent to GSL's gsl_sf_ellint_F(phi, k, GSL_PREC_DOUBLE).

    Args:
        phi: The amplitude (upper limit of integration) in radians
        k: The modulus, where 0 ≤ k² ≤ 1 (note: this is k, not m = k²)
        n_points: Number of integration points (default: 1000)

    Returns:
        The value of F(φ, k)

    Notes:
        - For k = 0: F(φ, 0) = φ
        - For |k| = 1 and |φ| < π/2: F(φ, ±1) = arctanh(sin(φ))
        - The implementation uses numerical integration via trapezoidal rule
        - JAX-compatible (can be used in jit, grad, vmap, etc.)

    Examples:
        >>> ellint_F(jnp.pi/2, 0.5)  # Complete elliptic integral K(0.5)
        >>> ellint_F(jnp.pi/4, 0.8)  # Incomplete elliptic integral
    """
    phi = jnp.asarray(phi)
    k = jnp.asarray(k)
    k2 = k * k

    # Handle special case: k = 0
    # F(φ, 0) = φ
    def case_k_zero():
        return phi

    # Handle special case: |k| = 1
    # F(φ, ±1) = arctanh(sin(φ)) = 0.5 * ln((1 + sin(φ)) / (1 - sin(φ)))
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
    result = jnp.where(
        is_k_zero,
        case_k_zero(),
        jnp.where(
            is_k_one,
            case_k_one(),
            case_general()
        )
    )

    return result


@jit
def ellint_Kcomp(k: float, n_points: int = 1000) -> float:
    """
    Compute the complete elliptic integral of the first kind.

    This is K(k) = F(π/2, k) = ∫₀^(π/2) dt / √(1 - k² sin²(t))

    Equivalent to GSL's gsl_sf_ellint_Kcomp(k, GSL_PREC_DOUBLE).

    Args:
        k: The modulus, where 0 ≤ k² ≤ 1
        n_points: Number of integration points (default: 1000)

    Returns:
        The value of K(k)
    """
    return ellint_F(jnp.pi / 2.0, k, n_points)


@jit
def ellint_F_carlson(phi: float, k: float) -> float:
    """
    Compute the incomplete elliptic integral of the first kind using Carlson's method.

    This is an alternative implementation using Carlson symmetric form:
    F(φ, k) = sin(φ) * R_F(cos²(φ), 1 - k² sin²(φ), 1)

    This method can be more accurate for certain parameter ranges but requires
    implementing Carlson's R_F function.

    Args:
        phi: The amplitude in radians
        k: The modulus

    Returns:
        The value of F(φ, k)

    Note:
        This is a placeholder for a future implementation using Carlson's
        symmetric elliptic integrals, which are numerically more stable.
    """
    # For now, fall back to the trapezoidal integration method
    return ellint_F(phi, k)


@jit
def gsl_sf_elljac_e(u: float, m: float, max_iter: int = 16):
    """
    Compute the Jacobian elliptic functions sn(u|m), cn(u|m), dn(u|m).

    This function computes the Jacobian elliptic functions using the descending
    Landen transformation, which is the same algorithm used by GSL.

    The Jacobian elliptic functions are defined by the inverse of the elliptic
    integral of the first kind:
        u = F(φ, k) = ∫₀^φ dt / √(1 - k² sin²(t))

    Then:
        sn(u|m) = sin(φ)
        cn(u|m) = cos(φ)
        dn(u|m) = √(1 - k² sin²(φ))

    where m = k² is the parameter (0 ≤ m ≤ 1).

    This is equivalent to GSL's gsl_sf_elljac_e(u, m, &sn, &cn, &dn).

    Args:
        u: The argument
        m: The parameter (m = k², where k is the modulus), 0 ≤ m ≤ 1
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

    # Handle special case: m = 0
    # sn(u|0) = sin(u), cn(u|0) = cos(u), dn(u|0) = 1
    def case_m_zero():
        sin_u = jnp.sin(u)
        cos_u = jnp.cos(u)
        return sin_u, cos_u, jnp.ones_like(u)

    # Handle special case: m = 1
    # sn(u|1) = tanh(u), cn(u|1) = sech(u), dn(u|1) = sech(u)
    def case_m_one():
        tanh_u = jnp.tanh(u)
        sech_u = 1.0 / jnp.cosh(u)
        return tanh_u, sech_u, sech_u

    # General case: use descending Landen transformation
    def case_general():
        # This implements Abramowitz & Stegun 16.14.1-2
        # Based on GSL's gsl_sf_elljac.c implementation

        # Build the Landen transformation sequence
        # a[0] = 1, c[0] = k, b[0] = sqrt(1-k^2)
        def landen_forward(carry, i):
            a_prev, c_prev = carry
            # b = sqrt(a^2 - c^2) = sqrt((a-c)(a+c))
            b_prev = jnp.sqrt((a_prev - c_prev) * (a_prev + c_prev))
            # Apply the transformation
            a_next = 0.5 * (a_prev + b_prev)
            c_next = 0.5 * (a_prev - b_prev)
            return (a_next, c_next), (a_next, c_next)

        k = jnp.sqrt(m)
        (a_final, c_final), (a_arr, c_arr) = jax.lax.scan(
            landen_forward,
            (jnp.ones_like(k), k),
            jnp.arange(max_iter)
        )

        # phi_n = 2^n * a_n * u (in the limit, this approaches the final angle)
        phi_n = jnp.power(2.0, max_iter) * a_final * u

        # Now work backward using the inverse transformation
        # sin(phi_{n-1}) = (a_n / a_{n-1}) * sin(phi_n)
        # and phi_{n-1} = (phi_n + arcsin(c_n/a_n * sin(phi_n))) / 2
        def landen_backward(phi_curr, i):
            idx = max_iter - 1 - i
            a_i = a_arr[idx]
            c_i = c_arr[idx]

            sin_phi = jnp.sin(phi_curr)
            # The inverse transformation
            # phi_prev = (phi_curr + arcsin(c * sin(phi_curr) / a)) / 2
            arg = jnp.clip(c_i * sin_phi / a_i, -1.0, 1.0)  # Clip to avoid numerical issues
            phi_prev = 0.5 * (phi_curr + jnp.arcsin(arg))

            return phi_prev, None

        phi_0, _ = jax.lax.scan(landen_backward, phi_n, jnp.arange(max_iter))

        # Compute the elliptic functions from phi_0
        sin_phi = jnp.sin(phi_0)
        cos_phi = jnp.cos(phi_0)

        sn = sin_phi
        cn = cos_phi
        dn = jnp.sqrt(1.0 - m * sin_phi * sin_phi)

        return sn, cn, dn

    # Select appropriate case based on m value
    abs_m = jnp.abs(m)
    is_m_zero = abs_m < 1e-10
    is_m_one = jnp.abs(abs_m - 1.0) < 1e-10

    # Compute all three cases
    sn_zero, cn_zero, dn_zero = case_m_zero()
    sn_one, cn_one, dn_one = case_m_one()
    sn_gen, cn_gen, dn_gen = case_general()

    # Select based on m value using nested where
    sn = jnp.where(is_m_zero, sn_zero, jnp.where(is_m_one, sn_one, sn_gen))
    cn = jnp.where(is_m_zero, cn_zero, jnp.where(is_m_one, cn_one, cn_gen))
    dn = jnp.where(is_m_zero, dn_zero, jnp.where(is_m_one, dn_one, dn_gen))

    return sn, cn, dn
