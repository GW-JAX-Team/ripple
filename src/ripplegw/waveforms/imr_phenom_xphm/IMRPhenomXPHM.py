import jax
import jax.numpy as jnp
from ripplegw import Mc_eta_to_ms

from ripplegw.constants import gt, MSUN
import numpy as np


from ripplegw.typing import Array


def _gen_IMRPhenomXPHM(

):
    h0 = None
    return h0


def gen_IMRPhenomXPHM(f: Array, params: Array, f_ref: float):
    """

    """


    h0 = _gen_IMRPhenomXPHM(

    )
    return h0


def gen_IMRPhenomXAS_hphc(f: Array, params: Array, f_ref: float):
    """

    """
    iota = params[7]
    h0 = gen_IMRPhenomXPHM(f, params, f_ref)

    # hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    # hc = -1j * h0 * jnp.cos(iota)

    return None, None # hp, hc