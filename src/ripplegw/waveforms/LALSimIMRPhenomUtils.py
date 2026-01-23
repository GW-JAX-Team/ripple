import jax.numpy as jnp


def XLALSimPhenomUtilsChiP(m1, m2, s1x, s1y, s2x, s2y):
    """
    Compute the effective precession parameter chip.

    This is a JAX translation of LALSimIMRPhenomUtils.c XLALSimPhenomUtilsChiP.

    Parameters
    ----------
    m1 : float or array
        Mass of companion 1 (solar masses)
    m2 : float or array
        Mass of companion 2 (solar masses)
    s1x : float or array
        x-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1)
    s1y : float or array
        y-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1)
    s2x : float or array
        x-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1)
    s2y : float or array
        y-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1)

    Returns
    -------
    chip : float or array
        Effective precession parameter
    """
    m1_2 = m1 * m1
    m2_2 = m2 * m2

    # Magnitude of the spin projections in the orbital plane
    S1_perp = m1_2 * jnp.sqrt(s1x * s1x + s1y * s1y)
    S2_perp = m2_2 * jnp.sqrt(s2x * s2x + s2y * s2y)

    A1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    A2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp

    num = jnp.where(ASp2 > ASp1, ASp2, ASp1)
    den = jnp.where(m2 > m1, A2 * m2_2, A1 * m1_2)
    chip = num / den

    return chip
