"""
Elliptic integral utilities implemented with JAX.
Some functions are based on https://github.com/tagordon/ellip/tree/main
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# relative error will be "less in magnitude than r"
R = 1.0e-15


@jax.jit
@jnp.vectorize
def rf(x, y, z):
    r"""JAX implementation of Carlson's :math:`R_\mathrm{F}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf
    Code taken from https://github.com/tagordon/ellip/tree/main

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the integral :math:`R_\mathrm{F}`

     Notes:
       ``rf`` does not support complex-valued inputs.
       ``rf`` requires `jax.config.update("jax_enable_x64", True)`
    """

    xyz = jnp.array([x, y, z])
    a0 = jnp.sum(xyz) / 3.0
    v = jnp.max(jnp.abs(a0 - xyz))
    q = (3 * R) ** (-1 / 6) * v

    # cond = lambda s: s["f"] * q > jnp.abs(s["An"])
    def cond(s):
        return s["f"] * q > jnp.abs(s["An"])

    def body(s):

        xyz = s["xyz"]
        lam = jnp.sqrt(xyz[0] * xyz[1]) + jnp.sqrt(xyz[0] * xyz[2]) + jnp.sqrt(xyz[1] * xyz[2])

        s["An"] = 0.25 * (s["An"] + lam)
        s["xyz"] = 0.25 * (s["xyz"] + lam)
        s["f"] = s["f"] * 0.25

        return s

    s = {"f": 1, "An": a0, "xyz": xyz}
    s = jax.lax.while_loop(cond, body, s)

    x = (a0 - x) / s["An"] * s["f"]
    y = (a0 - y) / s["An"] * s["f"]
    z = -(x + y)
    e2 = x * y - z * z
    e3 = x * y * z

    return (1 - 0.1 * e2 + e3 / 14 + e2 * e2 / 24 - 3 * e2 * e3 / 44) / jnp.sqrt(s["An"])


@jax.jit
@jnp.vectorize
def ellipfinc(phi, k):
    r"""JAX implementation of the incomplete elliptic integral of the first kind
        Code taken from https://github.com/tagordon/ellip/tree/main

        .. math::

            \[F\left(\phi,k\right)=\int_{0}^{\phi}\frac{\,\mathrm{d}\theta}{\sqrt{1-k^{2}{%
    \sin}^{2}\theta}}]

         Args:
           phi: arraylike, real valued.
           k: arraylike, real valued.

         Returns:
           The value of the complete elliptic integral of the first kind, :math:`F(\phi, k)`

         Notes:
           ``ellipfinc`` does not support complex-valued inputs.
           ``ellipfinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    c = 1.0 / jnp.sin(phi) ** 2
    return rf(c - 1, c - k**2, c)
