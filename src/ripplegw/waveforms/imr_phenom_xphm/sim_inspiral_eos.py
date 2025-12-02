"""SimInspiral EOS functions."""

from __future__ import annotations

import jax
import jax.lax
import jax.numpy as jnp


def xlal_sim_inspiral_eos_q_from_lambda(lambda_: float) -> float:
    """Compute the quadrupole-monopole parameter calculated from love number.

    References:
        - http://arxiv.org/abs/1303.1528

    Args:
        lambda_: Tidal deformability parameter.

    Returns:
        Mass ratio q (dimensionless).
    """
    tolerance = 5e-1

    def true_branch(_):
        """Return q = 1.0 when lambda_ < tolerance."""
        return 1.0

    def false_branch(_):
        """Compute q from polynomial when lambda_ >= tolerance."""
        log_lam = jnp.log(lambda_)
        q = 0.194 + 0.0936 * log_lam + 0.0474 * log_lam * log_lam
        q -= 0.00421 * log_lam * log_lam * log_lam
        q += 0.000123 * log_lam * log_lam * log_lam * log_lam
        q = jnp.exp(q)
        return q

    q = jax.lax.cond(lambda_ < tolerance, true_branch, false_branch, operand=None)

    return q
