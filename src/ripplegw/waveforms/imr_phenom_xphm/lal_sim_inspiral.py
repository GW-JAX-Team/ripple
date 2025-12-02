"""LALSimInspiral functions."""

from __future__ import annotations

import dataclasses

import jax
import jax.lax
import jax.numpy as jnp

from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass
from ripplegw.waveforms.imr_phenom_xphm.sim_inspiral_eos import xlal_sim_inspiral_eos_q_from_lambda


def xlal_sim_inspiral_set_quad_mon_params_from_lambdas(
    lal_params: IMRPhenomXPHMParameterDataClass,
) -> IMRPhenomXPHMParameterDataClass:
    """Set quadrupole-monopole parameters from tidal deformabilities.

    Args:
        lal_params: Parameter dataclass with lambda1, lambda2, d_quad_mon1, d_quad_mon2.

    Returns:
        Updated parameter dataclass with computed quadrupole-monopole parameters.
    """

    def update_quad_param1(lal_params):
        """Compute d_quad_mon1 from lambda1 if not already set."""
        quad_param1_ur = xlal_sim_inspiral_eos_q_from_lambda(lal_params.lambda1)
        return dataclasses.replace(lal_params, d_quad_mon1=quad_param1_ur - 1.0)

    def no_update_1(lal_params):
        """Return params unchanged if lambda1 condition not met."""
        return lal_params

    # Check condition for lambda1: lambda1 > 0.0 and d_quad_mon1 == 0.0
    condition1 = (lal_params.lambda1 > 0.0) & (lal_params.d_quad_mon1 == 0.0)
    lal_params = jax.lax.cond(condition1, update_quad_param1, no_update_1, lal_params)

    def update_quad_param2(lal_params):
        """Compute d_quad_mon2 from lambda2 if not already set."""
        quad_param2_ur = xlal_sim_inspiral_eos_q_from_lambda(lal_params.lambda2)
        return dataclasses.replace(lal_params, d_quad_mon2=quad_param2_ur - 1.0)

    def no_update_2(lal_params):
        """Return params unchanged if lambda2 condition not met."""
        return lal_params

    # Check condition for lambda2: lambda2 > 0.0 and d_quad_mon2 == 0.0
    condition2 = (lal_params.lambda2 > 0.0) & (lal_params.d_quad_mon2 == 0.0)
    lal_params = jax.lax.cond(condition2, update_quad_param2, no_update_2, lal_params)

    return lal_params


def xlal_sim_inspiral_chirp_time_bound(fstart: float, m1: float, m2: float, s1: float, s2: float) -> float:
    M = m1 + m2  # total mass
    mu = m1 * m2 / M  # reduced mass
    eta = mu / M  # symmetric mass ratio
    # /* chi = (s1*m1 + s2*m2)/M <= max(|s1|,|s2|) */

    chi = jax.lax.select( # fabs(fabs(s1) > fabs(s2) ? s1 : s2) #// over-estimate of chi
        jnp.fabs(s1) > jnp.fabs(s2),
        s1,
        s2,
    )
    
    # /* note: for some reason these coefficients are named wrong...
    #  * "2PN" should be "1PN", "4PN" should be "2PN", etc. */
    c0 = jnp.fabs(XLALSimInspiralTaylorT2Timing_0PNCoeff(M, eta))
    c2 = XLALSimInspiralTaylorT2Timing_2PNCoeff(eta);
    # /* the 1.5pN spin term is in TaylorT2 is 8*beta/5 [Citation ??]
    #  * where beta = (113/12 + (25/4)(m2/m1))*(s1*m1^2/M^2) + 2 <-> 1
    #  * [Cutler & Flanagan, Physical Review D 49, 2658 (1994), Eq. (3.21)]
    #  * which can be written as (113/12)*chi - (19/6)(s1 + s2)
    #  * and we drop the negative contribution */
    c3 = (226.0/15.0) * chi
    # /* there is also a 1.5PN term with eta, but it is negative so do not include it */
    c4 = XLALSimInspiralTaylorT2Timing_4PNCoeff(eta)
    v = cbrt(LAL_PI * LAL_G_SI * M * fstart) / LAL_C_SI
    return c0 * pow(v, -8) * (1.0 + (c2 + (c3 + c4 * v) * v) * v * v)

