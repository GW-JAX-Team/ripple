"""Implementation of LALSimIMRPhenomXPNRBeta module in JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_pnr_beta_coefficients import (
    imr_phenom_x_pnr_beta_bf_coefficient,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)


def imr_phenom_x_pnr_generate_ringdown_pnr_beta(
    p_wf: IMRPhenomXWaveformDataClass, p_prec: IMRPhenomXPrecessionDataClass
) -> float | Array:
    """
    Docstring for imr_phenom_x_pnr_generate_ringdown_pnr_beta

    :param p_wf: Description
    :type p_wf: IMRPhenomXWaveformDataClass
    :param p_prec: Description
    :type p_prec: IMRPhenomXPrecessionDataClass
    :return: Description
    :rtype: float | Array
    """

    # /* get effective single spin parameters */
    eta = p_wf.eta
    chi = p_prec.chi_single_spin
    costheta = p_prec.cos_theta_single_spin

    # /* approximate orientation of final spin */
    costhetaf = p_prec.cos_theta_final_single_spin
    betafinal = imr_phenom_x_pnr_arctan_window(
        jnp.arccos(costhetaf) - imr_phenom_x_pnr_beta_bf_coefficient(eta, chi, costheta)
    )

    return betafinal


def imr_phenom_x_pnr_arctan_window(beta: float) -> float:
    """
    Utility function to compute the arctan windowing described in Eq. 62 of arXiv:2107.08876.
    """

    # /* specify the width as described in Sec. 8C of arXiv:2107.08876*/
    window_border = 0.01

    # /* if beta is close to zero or PI, window it */
    def beta_close_to_border():
        p = 0.002
        pi_by_2 = 1.570796326794897
        pi_by_2_1mp = 1.569378278348018
        pi_by_2_1oq = 7.308338225719002e97
        sign = jax.lax.select(beta < pi_by_2, -1.0, 1.0)

        return sign * pi_by_2_1mp * jnp.power(jnp.arctan2(jnp.power(beta - pi_by_2, 1.0 / p), pi_by_2_1oq), p) + pi_by_2

    return jax.lax.cond(
        jnp.logical_or(beta <= window_border, beta >= jnp.pi - window_border),
        beta_close_to_border,
        lambda: beta,
    )
