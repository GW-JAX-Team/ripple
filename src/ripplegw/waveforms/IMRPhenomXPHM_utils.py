import jax.numpy as jnp
import jax


def XLALSimIMRPhenomXatan2tol(vy, vx, tol):
    cond = (jnp.abs(vy) < tol) & (jnp.abs(vx) < tol)
    return jax.lax.cond(
        cond,
        lambda _: 0.0,
        lambda _: jnp.atan2(vy, vx),
        operand=None,
    )
