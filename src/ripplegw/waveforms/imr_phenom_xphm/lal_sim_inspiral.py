"""LALSimInspiral functions."""

from __future__ import annotations

import dataclasses

import jax
import jax.lax

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
