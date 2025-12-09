"""
Docstring for ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_pnr_internals
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from ripplegw.constants import PI
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    xlal_sim_imr_phenom_x_chi_eff,
    xlal_sim_imr_phenom_x_final_spin_2017,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import (
    IMRPhenomXPHMParameterDataClass,
)


def imr_phenom_x_pnr_get_and_set_pnr_variables(
    p_wf: IMRPhenomXWaveformDataClass, p_prec: IMRPhenomXPrecessionDataClass
) -> IMRPhenomXPrecessionDataClass:
    """
    Docstring for imr_phenom_x_pnr_get_and_set_pnr_variables

    :param p_wf: Description
    :type p_wf: IMRPhenomXWaveformDataClass
    :param p_prec: Description
    :type p_prec: IMRPhenomXPrecessionDataClass
    """
    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x___p_n_r__internals_8c.html#aabb3d2fee26595c5a632df2df8832836

    # /* get needed quantities */
    m1 = p_wf.m1 * p_wf.m_tot
    m2 = p_wf.m2 * p_wf.m_tot
    q = p_wf.q

    chi1x, chi1y, chi2x, chi2y, chieff = jax.lax.select(
        p_prec.imr_phenom_x_prec_version == 330,
        jnp.array(
            [
                p_prec.chi1x_evolved,
                p_prec.chi1y_evolved,
                p_prec.chi2x_evolved,
                p_prec.chi2y_evolved,
                xlal_sim_imr_phenom_x_chi_eff(p_wf.eta, p_prec.chi1z_evolved, p_prec.chi2z_evolved),
            ]
        ),
        jnp.array(
            [
                p_prec.chi1x,
                p_prec.chi1y,
                p_prec.chi2x,
                p_prec.chi2y,
                xlal_sim_imr_phenom_x_chi_eff(p_wf.eta, p_prec.chi1z, p_prec.chi2z),
            ]
        ),
    )

    chipar = p_wf.m_tot * chieff / m1
    chiperp = 0.0
    costheta = 0.0
    chiperp_antisymmetric = 0.0
    theta_antisymmetric = 0.0

    # /* compute effective in-plane spin contribution from Eq. 17 of arXiv:2107.08876 */
    # /* for XO4a this contribution is only used for mass ratios below 1.5 */
    # /* in versions of the model where we use the evolved spin values, we use this contribution at all mass ratios */

    def prec_version_330_branch():
        chis = jnp.sqrt(
            (m1 * m1 * chi1x + m2 * m2 * chi2x) * (m1 * m1 * chi1x + m2 * m2 * chi2x)
            + (m1 * m1 * chi1y + m2 * m2 * chi2y) * (m1 * m1 * chi1y + m2 * m2 * chi2y)
        ) / (m1 * m1)
        return chis

    def other_prec_version_branch():
        chis = jnp.sqrt(
            (m1 * m1 * chi1x + m2 * m2 * chi2x) * (m1 * m1 * chi1x + m2 * m2 * chi2x)
            + (m1 * m1 * chi1y + m2 * m2 * chi2y) * (m1 * m1 * chi1y + m2 * m2 * chi2y)
        ) / (m1 * m1)

        return jax.lax.select(
            q <= 1.5,
            jnp.sin((q - 1.0) * PI) * jnp.sin((q - 1.0) * PI) * p_prec.chi_p
            + jnp.cos((q - 1.0) * PI) * jnp.cos((q - 1.0) * PI) * chis,
            p_prec.chi_p,
        )

    chiperp = jax.lax.cond(
        p_prec.imr_phenom_x_prec_version == 330,
        lambda _: prec_version_330_branch(),
        lambda _: other_prec_version_branch(),
        operand=None,
    )

    antisymmetric_chis = jnp.sqrt(
        (m1 * m1 * chi1x - m2 * m2 * chi2x) * (m1 * m1 * chi1x - m2 * m2 * chi2x)
        + (m1 * m1 * chi1y - m2 * m2 * chi2y) * (m1 * m1 * chi1y - m2 * m2 * chi2y)
    ) / (m1 * m1)
    chiperp_antisymmetric = jax.lax.select(
        q <= 1.5,
        jnp.sin((q - 1.0) * PI) * jnp.sin((q - 1.0) * PI) * p_prec.chi_p
        + jnp.cos((q - 1.0) * PI) * jnp.cos((q - 1.0) * PI) * antisymmetric_chis,
        p_prec.chi_p,
    )

    # /* get the total magnitude, Eq. 18 of arXiv:2107.08876 */
    chi_mag = jnp.sqrt(chipar * chipar + chiperp * chiperp)
    # pPrec->chi_singleSpin = chi_mag

    chi_mag_antisymmetric = jnp.sqrt(chipar * chipar + chiperp_antisymmetric * chiperp_antisymmetric)
    # pPrec->chi_singleSpin_antisymmetric = chi_mag_antisymmetric

    # /* get the opening angle of the single spin, Eq. 19 of arXiv:2107.08876 */
    costheta = jax.lax.select(
        chi_mag >= 1.0e-6,
        chipar / chi_mag,
        0.0,
    )
    theta_antisymmetric = jax.lax.select(
        chi_mag_antisymmetric >= 1.0e-6,
        jnp.arccos(chipar / chi_mag_antisymmetric),
        0.0,
    )

    # /* compute an approximate final spin using single-spin mapping */
    chi1l = chi_mag * costheta
    chi2l = 0.0

    x_fparr = xlal_sim_imr_phenom_x_final_spin_2017(p_wf.eta, chi1l, chi2l)

    # /* rescale Xfperp to use the final total mass of 1 */
    qfactor = q / (1.0 + q)
    x_fperp = qfactor * qfactor * chi_mag * jnp.sqrt(1.0 - costheta * costheta)
    xf = jnp.sqrt(x_fparr * x_fparr + x_fperp * x_fperp)

    costheta_final_single_spin = jax.lax.select(
        xf > 1.0e-6,
        x_fparr / xf,
        0.0,
    )

    # /* set inspiral scaling flag for HM frequency map */
    scaling_condition = (q > p_prec.pnr_q_window_upper) | (chi_mag > p_prec.pnr_chi_window_upper)
    pnr_inspiral_scaling = jax.lax.select(
        scaling_condition,
        1,
        0,
    )

    p_prec = dataclasses.replace(
        p_prec,
        chi_single_spin=chi_mag,
        chi_single_spin_antisymmetric=chi_mag_antisymmetric,
        cos_theta_single_spin=costheta,
        theta_antisymmetric=theta_antisymmetric,
        cos_theta_final_single_spin=costheta_final_single_spin,
        pnr_hm_m_f_low=0.0,
        pnr_hm_m_f_high=0.0,
        pnr_q_window_lower=8.5,
        pnr_q_window_upper=12.0,
        pnr_chi_window_lower=0.85,
        pnr_chi_window_upper=1.2,
        pnr_inspiral_scaling=pnr_inspiral_scaling,
    )

    return p_prec


def imr_phenom_x_pnr_get_and_set_co_prec_params(
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    lal_params: IMRPhenomXPHMParameterDataClass,
) -> None:
    """
    Docstring for imr_phenom_x_pnr_get_and_set_co_prec_params

    :param pwf: Description
    :type pwf: IMRPhenomXWaveformDataClass
    :param p_prec: Description
    :type p_prec: IMRPhenomXPrecessionDataClass
    :param lal_params: Description
    :type lal_params: IMRPhenomXPHMParameterDataClass
    """

    _ = p_wf
    _ = p_prec
    _ = lal_params
