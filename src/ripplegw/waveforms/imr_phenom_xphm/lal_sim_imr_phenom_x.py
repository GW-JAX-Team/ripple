from __future__ import annotations

import copy
import dataclasses

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import checkify

from ripplegw.typing import Array
from ripplegw.constants import PI, gt, MSUN
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXPHMParameterDataClass,
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession import (
    get_alphaepsilon_atfref,
    imr_phenom_x_initialize_msa_system,
    imr_phenom_x_return_phi_zeta_costheta_l_msa,
    imr_phenom_x_set_precessing_remnant_params,
    imr_phenom_xp_check_max_opening_angle,
    imr_phenom_x_get_and_set_precession_variables,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_xphm import (
    imr_phenom_x_initialize_powers,
    imr_phenom_x_set_waveform_variables,
    imr_phenom_xphm_setup_mode_array,
    check_input_mode_array,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    xlal_imr_phenom_xp_check_masses_and_spins,
)

import lal
import lalsimulation as lalsim
from lalsimulation import SimIMRPhenomXPMSAAngles

def xlal_imr_phenom_xp_msa_angles(
    freqs: Array,
    m1_si: float,                       
    m2_si: float,                       
    chi1x: float,                       
    chi1y: float,                       
    chi1z: float,                       
    chi2x: float,                       
    chi2y: float,                       
    chi2z: float,                       
    inclination: float,                     
    f_ref_in: float,                   
    mprime: int, 
    lal_params: IMRPhenomXPHMParameterDataClass
):
    # Check if m1 > m2, swap the bodies otherwise.
    m1_si, m2_si, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z = xlal_imr_phenom_xp_check_masses_and_spins(
        m1_si=m1_si,
        m2_si=m2_si,
        chi1x=chi1x,
        chi1y=chi1y,
        chi1z=chi1z,
        chi2x=chi2x,
        chi2y=chi2y,
        chi2z=chi2z,
    )

    # Perform initial sanity checks.
    checkify.check(f_ref_in >= 0, "Error: f_fef_in must be positive or set to 0 to ignore.")
    checkify.check(m1_si > 0, "Error: m1 must be positive.")
    checkify.check(m2_si > 0, "Error: m2 must be positive.")
    
    chi1_l = chi1z
    chi2_l = chi2z

    f_ref = f_ref_in

    # /* Use an auxiliar laldict to not overwrite the input argument */
    # Copy the lal_params
    lal_params_aux = lal_params.copy()

    # Initialize the useful powers of pi.
    _error, powers_of_pi = imr_phenom_x_initialize_powers(jnp.pi)

    lal_params_dataclass = IMRPhenomXPHMParameterDataClass()
    # Initialize IMRPhenomX waveform struct and check that it is initialized correctly.
    _error, p_wf = imr_phenom_x_set_waveform_variables(
        m1_si,
        m2_si,
        chi1_l,
        chi2_l,
        0.0,
        f_ref,
        0.0,
        freqs[0],
        freqs[-1],
        1.0,
        inclination,
        lal_params_dataclass,
        powers_of_pi
    )


    # Initialize IMR PhenomX Precession struct and check that it generated successfully
    # pflag = lal_params_aux.precession_version
    # pflag = jax.lax.select(
    #     pflag == 300,
    #     223,
    #     pflag
    # )

    p_prec = IMRPhenomXPrecessionDataClass()

    _, (p_wf, p_prec, lal_params_aux) = imr_phenom_x_get_and_set_precession_variables(
        p_wf,
        p_prec, 
        m1_si,
        m2_si,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        lal_params_aux
    )

    def compute_angles_at_freq(f):
        """Compute MSA angles at a single frequency."""
        # Convert GW frequency to velocity parameter
        # v = (pi * M * f_gw)^(1/3) where f_gw = m * f_orbital
        # For GW frequency: f_gw = mprime * f_orbital / 2
        v = jnp.cbrt(f * p_prec.pi_gm * (2.0 / mprime))

        # Get MSA angles: returns [phi_z, zeta, cos(theta_L)]
        vangles, _ = imr_phenom_x_return_phi_zeta_costheta_l_msa(v, p_wf, p_prec)

        # Extract and apply offsets
        alpha = vangles[0] - p_prec.alpha_offset
        gamma = -(vangles[1] - p_prec.epsilon_offset)
        cosbeta = vangles[2]

        return alpha, gamma, cosbeta
    
    alphas, gammas, cosbetas = jax.vmap(compute_angles_at_freq)(freqs)
    return alphas, gammas, cosbetas


if __name__ == "__main__":

    freqs=jnp.linspace(20.0, 512.0, num=10)
    m1_si=30.0 * MSUN
    m2_si=20.0 * MSUN
    chi1x=0.3
    chi1y=0.2
    chi1z=0.4
    chi2x=0.1
    chi2y=0.2
    chi2z=0.3
    inclination=0.0
    f_ref_in=20.0
    mprime=2
    lal_params=IMRPhenomXPHMParameterDataClass(precession_version=220)

    print("Computing MSA angles with ripple...")
    
    alphas_ripple, gammas_ripple, cosbetas_ripple = xlal_imr_phenom_xp_msa_angles(
        freqs,
        m1_si,
        m2_si,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        inclination,
        f_ref_in,
        mprime,
        lal_params
    )
    
    laldict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(laldict, 220)

    print("Computing MSA angles with LAL...")

    alphas_lal, gammas_lal, cosbetas_lal = SimIMRPhenomXPMSAAngles(
        freqs,
        m1_si,
        m2_si,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        inclination,
        f_ref_in,
        mprime,
        laldict
    )

    print("#" * 40)
    print("ripple MSA angles")
    print(f"alphas: {alphas_ripple}\ngammas: {gammas_ripple}\ncosbetas: {cosbetas_ripple}")
    print("#" * 40)
    print("lal MSA angles")
    print(f"alphas: {alphas_lal.data}\ngammas: {gammas_lal.data}\ncosbetas: {cosbetas_lal.data}")
    print("#" * 40)