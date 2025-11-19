from __future__ import annotations

import jax.numpy as jnp
import jax
from jax.experimental import checkify

from ripplegw.typing import Array


def xlal_sim_imr_phenom_xphm_frequency_seqeuence_one_mode(
    h_lm_pos: Array,
    h_lm_neg: Array,
    freqs: Array,
    l: int,
    m: int,
    m1_si: float,
    m2_si: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
    distance: float,
    inclination: float,
    phi_ref: float,
    f_ref_in: float,
    lal_params: dict) -> None:
    """
    Function to compute one hlm precessing mode on a custom frequency grid.
    Equivalent options and behaviour to that of XLALSimIMRPhenomXPHMOneMode.
    
    Args:
        freqs: Frequency array where the mode is computed.
        l: Spherical harmonic index l.
        m: Spherical harmonic index m.
        m1_si: Mass 1 in SI units (kg).
        m2_si: Mass 2 in SI units (kg).
        chi1x: Dimensionless spin component of body 1 along x.
        chi1y: Dimensionless spin component of body 1 along y.
        chi1z: Dimensionless spin component of body 1 along z.
        chi2x: Dimensionless spin component of body 2 along x.
        chi2y: Dimensionless spin component of body 2 along y.
        chi2z: Dimensionless spin component of body 2 along z.
        distance: Distance to the source in meters.
        inclination: Inclination angle in radians.
        phi_ref: Reference phase in radians.
        f_ref_in: Reference frequency in Hz.
        lal_params: Additional LAL parameters as a dictionary.
    """
    status = xlal_imr_phenom_xp_check_masses_and_spins(m1_si, m2_si, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)

    checkify.check(status == 0, "Error: xlal_imr_phenom_xp_check_masses_and_spins failed.")

    jax.debug.print("fRef_In: {f_ref_in}", f_ref_in=f_ref_in)
    jax.debug.print("m1_si: {m1_si}", m1_si=m1_si)
    jax.debug.print("m2_si: {m2_si}", m2_si=m2_si)
    jax.debug.print("chi1z: {chi1z}", chi1z=chi1z)
    jax.debug.print("chi2z: {chi2z}", chi2z=chi2z)
    jax.debug.print("phi_ref: {phi_ref}", phi_ref=phi_ref)
    jax.debug.print("Prec V.: {prec_version}", prec_version=xlal_sim_inspiral_waveform_params_lookup_phenom_xprec_version(lal_params))
    jax.debug.print("Performing sanity checks...")
    
    checkify.check(f_ref_in >= 0, "Error: r_ref_in must be positive or set to 0 to ignore.")
    checkify.check(m1_si > 0, "Error: m1 must be positive.")
    checkify.check(m2_si > 0, "Error: m2 must be positive.")
    checkify.check(distance > 0, "Error: Distance must be positive.")
    
    # Get minimum and maximum frequencies.
    f_min = freqs[0]
    f_max = freqs[-1]
        
    # Perform a basic sanity check on the region of the parameter space in which model is evaluated.
    # Behaviour is as follows, consistent with the choices for IMRPhenomXAS/IMRPhenomXHM
    #     - For mass ratios <= 20.0 and spins <= 0.99: no warning messages.
    #     - For 1000 > mass ratio > 20 and spins <= 0.99: print a warning message that we are extrapolating outside of *NR* calibration domain.
    #     - For mass ratios > 1000: throw a hard error that model is not valid.
    #     - For spins > 0.99: throw a warning that we are extrapolating the model to extremal  
    
    mass_ratio = jax.lax.cond(m1_si > m2_si, lambda m1, m2: m1 / m2, lambda m1, m2: m2 / m1, m1_si, m2_si)
    
    # Check the mass ratio.
    def check_mass_ratio(mass_ratio):
        def warn_extrapolation(mass_ratio):
            jax.debug.print("Warning: Mass ratio = {mass_ratio}. Extrapolating outside of Numerical Relativity calibration domain. NNLO angles may become pathological at large mass ratios.", mass_ratio=mass_ratio)
            return mass_ratio

        def error_too_large(mass_ratio):
            checkify.check(False, "Error: Mass ratio = {mass_ratio}. Model not valid at mass ratios beyond 1000.", mass_ratio=mass_ratio)
            return mass_ratio

        mass_ratio = jax.lax.cond(mass_ratio >= 1000.0, error_too_large, lambda x: x, mass_ratio)
        mass_ratio = jax.lax.cond((mass_ratio > 20.0) & (mass_ratio <= 1000.0), warn_extrapolation, lambda x: x, mass_ratio)
        return mass_ratio
    
    mass_ratio = check_mass_ratio(mass_ratio)
    
    # Check the spins
    def check_spins(chi1z, chi2z):
        def warn_extrapolation_spins(args):
            chi1z, chi2z = args
            jax.debug.print("Warning: Spins chi1z = {chi1z}, chi2z = {chi2z}. Extrapolating to extremal spins, model is not trusted.", chi1z=chi1z, chi2z=chi2z)
            return chi1z, chi2z

        def no_warning(args):
            return args

        chi1z, chi2z = jax.lax.cond((jnp.abs(chi1z) > 0.99) | (jnp.abs(chi2z) > 0.99), warn_extrapolation_spins, no_warning, (chi1z, chi2z))
        return chi1z, chi2z
    
    chi1z, chi2z = check_spins(chi1z, chi2z)
    
    # If no reference frequency is given, set it to the starting gravitational wave frequency.
    f_ref = jax.lax.cond(f_ref_in == 0.0, lambda f_min, _f_ref_in: f_min, lambda _f_min, f_ref_in: f_ref_in, f_min, f_ref_in)
    
    # Use an auxiliary lal_dict to not overwrite the input argument
    lal_params_aux = lal_params.copy()
    
    # Check that the modes chosen are available for the model
    checkify.check(check_input_mode_array(lal_params_aux) == 0, "Error: Not available mode chosen.")
    
    jax.debug.print("Initializing waveform dict...")
    
    # Initialize IMR PhenomX waveform struct and check that it initialized correctly.
    # We pass inclination 0 since for the individual modes is not relevant.
    
    p_wf = imr_phenom_x_set_waveform_variables(m1_si, m2_si, chi1z, chi2z, 0.0, f_ref, phi_ref, f_min, f_max, distance, inclination, lal_params_aux)
    
    jax.debug.print("Initializing precession dict...")
    
    # Only relevant for SpinTaylor angles
    p_flag = xlal_sim_inspiral_waveform_params_lookup_phenom_xprec_version(lal_params_aux)
    
    p_prec = jax.lax.cond(p_flag == 310 | p_flag == 311 | p_flag == 320, p_flag == 321 | p_flag == 330,
                          lambda _: {"m_min": jnp.max(1, jnp.abs(m)),
                           "m_max": l,
                           "l_max_pnr": l},
                          lambda _: {},
                          p_flag)
    
    jax.lax.cond(xlal_sim_inspiral_waveform_params_lookup_phenom_xpnr_use_tuned_angles(lal_params_aux)==1,
                 lambda _: xlal_sim_inspiral_waveform_params_insert_phenom_xphm_threshold_mband(lal_params_aux, 0),
                 lambda _: None,
                 None)
    
    imr_phenom_x_get_and_set_precession_variables(p_wf,
                                                  p_prec,
                                                  m1_si,
                                                  m2_si,
                                                  chi1x,
                                                  chi1y,
                                                  chi1z,
                                                  chi2x,
                                                  chi2y,
                                                  chi2z,
                                                  lal_params_aux)
    
    jax.lax.cond(p_prec["precessing_tag"] == 3, 
                 lambda _: imr_phenom_x_initialize_euler_angles(p_wf,
                                                                p_prec,
                                                                lal_params),
                 lambda _: None,
                 None)
    
    # Ensure recovering AS limit when modes are in the L0 frame.
    l0_frame = xlal_sim_inspiral_waveform_params_lookup_phenom_xphm_modes_l0_frame(lal_params_aux)  # placeholder function
    
    def handle_l0_frame(p_wf, phi_ref):
        jax.debug.print("The L0Frame option only works near the AS limit, it should not be used otherwise.")
        convention = xlal_sim_inspiral_waveform_params_lookup_phenom_xp_convention(lal_params_aux)  # placeholder function
        
        def case_0_5(p_wf):
            # pWF->phi0 = pPrec->phi0_aligned;  # commented out in C
            return p_wf
        
        def case_1_6_7(p_wf):
            p_wf_new = p_wf.copy()
            p_wf_new['phi0'] = phi_ref
            return p_wf_new
        
        def default_case(p_wf):
            return p_wf
        
        p_wf = jax.lax.cond(
            (convention == 0) | (convention == 5),
            case_0_5,
            lambda p_wf: jax.lax.cond(
                (convention == 1) | (convention == 6) | (convention == 7),
                case_1_6_7,
                default_case,
                p_wf
            ),
            p_wf
        )
        return p_wf
    
    p_wf = jax.lax.cond(
        l0_frame == 1,
        lambda p_wf: handle_l0_frame(p_wf, phi_ref),
        lambda p_wf: p_wf,
        p_wf
    )

    jax.debug.print("Calling imr_phenom_xphm_one_mode...")
    
    xlal_sim_inspiral_waveform_params_insert_phenom_xhm_threshold_mband(lal_params_aux, 0)
    
    status = imr_phenom_xphm_one_mode(h_lm_pos, h_lm_neg, freqs, p_wf, p_prec, l, m, lal_params_aux)
    checkify.check(status == 0, "Error: imr_phenom_xphm_one_mode failed to generate IMRPhenomXHM waveform.")
    
    # Transform modes to L0-frame if requested. It only works for (near) AS cases.
    l0_frame = xlal_sim_inspiral_waveform_params_lookup_phenom_xphm_modes_l0_frame(lal_params_aux)  # placeholder function
    
    def transform_l0_frame(h_lm_pos, h_lm_neg, m, p_prec):
        convention = xlal_sim_inspiral_waveform_params_lookup_phenom_xp_convention(lal_params_aux)  # placeholder function
        
        def do_nothing(h_lm_pos, h_lm_neg):
            return h_lm_pos, h_lm_neg
        
        def apply_shift(h_lm_pos, h_lm_neg):
            shiftpos = jnp.exp(1j * jnp.abs(m) * (p_prec['epsilon0'] - p_prec['alpha0']))
            shiftneg = 1. / shiftpos
            h_lm_pos_new = h_lm_pos * shiftpos
            h_lm_neg_new = h_lm_neg * shiftneg
            return h_lm_pos_new, h_lm_neg_new
        
        h_lm_pos, h_lm_neg = jax.lax.cond(
            (convention == 0) | (convention == 5),
            do_nothing,
            lambda h_lm_pos, h_lm_neg: jax.lax.cond(
                (convention == 1) | (convention == 6) | (convention == 7),
                apply_shift,
                do_nothing,
                h_lm_pos, h_lm_neg
            ),
            h_lm_pos, h_lm_neg
        )
        return h_lm_pos, h_lm_neg
    
    h_lm_pos, h_lm_neg = jax.lax.cond(
        l0_frame == 1,
        lambda h_lm_pos, h_lm_neg: transform_l0_frame(h_lm_pos, h_lm_neg, m, p_prec),
        lambda h_lm_pos, h_lm_neg: (h_lm_pos, h_lm_neg),
        h_lm_pos, h_lm_neg
    )

    jax.debug.print("Call to imr_phenom_xphm_one_mode complete.")
