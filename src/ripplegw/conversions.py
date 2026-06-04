"""
Parameter conversion utilities for binary systems.

Includes conversions between mass parameterisations and tidal parameters.
"""

import jax.numpy as jnp

from jaxtyping import Array


def Mc_eta_to_ms(m: Array) -> tuple[Array, Array]:
    r"""Convert chirp mass and symmetric mass ratio to binary component masses.

    Args:
        m (Array): Array ``(Mchirp, eta)`` — chirp mass and symmetric mass ratio.

    Returns:
        tuple[Array, Array]: Component masses $(m_1, m_2)$ in the same
            units as the chirp mass, with $m_1 \geq m_2$.
    """
    Mchirp, eta = m
    M = Mchirp / (eta ** (3 / 5))
    m2 = (M - jnp.sqrt(jnp.maximum(M**2 - 4 * M**2 * eta, 0.0))) / 2
    m1 = M - m2
    return m1, m2


def ms_to_Mc_eta(m: Array) -> tuple[Array, Array]:
    r"""Convert binary component masses to chirp mass and symmetric mass ratio.

    Args:
        m (Array): Array ``(m1, m2)`` — component masses.

    Returns:
        tuple[Array, Array]: $(\mathcal{M}, \eta)$, with the chirp mass in
            the same units as the component masses.
    """
    m1, m2 = m
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5), m1 * m2 / (m1 + m2) ** 2


# ---------------------------------------------------------------------------
# Internal helpers (not part of the public API)
# ---------------------------------------------------------------------------


def _compute_lambda_tildes_from_eta(
    eta: Array, lambda_1: Array, lambda_2: Array
) -> tuple[Array, Array]:
    """Core tidal conversion: individual lambdas → (lambda_tilde, delta_lambda_tilde) given eta."""
    lambda_plus = lambda_1 + lambda_2
    lambda_minus = lambda_1 - lambda_2
    sqrt_1m4eta = jnp.sqrt(jnp.maximum(1 - 4 * eta, 0.0))
    lambda_tilde = (
        8
        / 13
        * (
            (1 + 7 * eta - 31 * eta**2) * lambda_plus
            + sqrt_1m4eta * (1 + 9 * eta - 11 * eta**2) * lambda_minus
        )
    )
    delta_lambda_tilde = (
        1
        / 2
        * (
            sqrt_1m4eta * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2) * lambda_plus
            + (1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2 + 3380 / 1319 * eta**3)
            * lambda_minus
        )
    )
    return lambda_tilde, delta_lambda_tilde


def _compute_lambdas_from_eta(
    eta: Array, lambda_tilde: Array, delta_lambda_tilde: Array
) -> tuple[Array, Array]:
    """Core tidal conversion: (lambda_tilde, delta_lambda_tilde) → individual lambdas given eta."""
    sqrt_1m4eta = jnp.sqrt(jnp.maximum(1 - 4 * eta, 0.0))
    coefficient_1 = 1 + 7 * eta - 31 * eta**2
    coefficient_2 = sqrt_1m4eta * (1 + 9 * eta - 11 * eta**2)
    coefficient_3 = sqrt_1m4eta * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2)
    coefficient_4 = (
        1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2 + 3380 / 1319 * eta**3
    )
    lambda_1 = (
        13 * lambda_tilde / 8 * (coefficient_3 - coefficient_4)
        - 2 * delta_lambda_tilde * (coefficient_1 - coefficient_2)
    ) / (
        (coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4)
        - (coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4)
    )
    lambda_2 = (
        13 * lambda_tilde / 8 * (coefficient_3 + coefficient_4)
        - 2 * delta_lambda_tilde * (coefficient_1 + coefficient_2)
    ) / (
        (coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4)
        - (coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4)
    )
    return lambda_1, lambda_2


# ---------------------------------------------------------------------------
# Public tidal conversion functions
# ---------------------------------------------------------------------------


def lambdas_to_lambda_tildes(params: Array) -> tuple[Array, Array]:
    """
    Convert from individual tidal parameters to dominant tidal terms. (Code taken from Bilby)

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Args:
        params: (lambda_1, lambda_2, mass_1, mass_2)

    Returns:
        (lambda_tilde, delta_lambda_tilde)
    """
    lambda_1, lambda_2, mass_1, mass_2 = params
    _, eta = ms_to_Mc_eta(jnp.array([mass_1, mass_2]))
    return _compute_lambda_tildes_from_eta(eta, lambda_1, lambda_2)


def lambdas_to_lambda_tildes_from_q(params: Array) -> tuple[Array, Array]:
    """
    Convert from individual tidal parameters to dominant tidal terms using mass ratio. (Code taken from Bilby)

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Args:
        params: (lambda_1, lambda_2, q)

    Returns:
        (lambda_tilde, delta_lambda_tilde)
    """
    lambda_1, lambda_2, q = params
    eta = q / (1 + q) ** 2
    return _compute_lambda_tildes_from_eta(eta, lambda_1, lambda_2)


def lambda_tildes_to_lambdas(params: Array) -> tuple[Array, Array]:
    """
    Convert from dominant tidal terms to individual tidal parameters. Code taken from bilby.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Args:
        params: (lambda_tilde, delta_lambda_tilde, mass_1, mass_2)

    Returns:
        (lambda_1, lambda_2)
    """
    lambda_tilde, delta_lambda_tilde, mass_1, mass_2 = params
    _, eta = ms_to_Mc_eta(jnp.array([mass_1, mass_2]))
    return _compute_lambdas_from_eta(eta, lambda_tilde, delta_lambda_tilde)


def lambda_tildes_to_lambdas_from_q(params: Array) -> tuple[Array, Array]:
    """
    Convert from dominant tidal terms to individual tidal parameters using mass ratio. Code taken from bilby.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Args:
        params: (lambda_tilde, delta_lambda_tilde, q)

    Returns:
        (lambda_1, lambda_2)
    """
    lambda_tilde, delta_lambda_tilde, q = params
    eta = q / (1 + q) ** 2
    return _compute_lambdas_from_eta(eta, lambda_tilde, delta_lambda_tilde)
