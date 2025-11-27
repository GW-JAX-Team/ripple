from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals import (
    check_input_mode_array,
    imr_phenom_x_initialize_powers,
    imr_phenom_x_set_waveform_variables,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    xlal_imr_phenom_xp_check_masses_and_spins,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_create_mode_array,
    xlal_sim_inspiral_mode_array_activate_mode
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass

def IMRPhenomXPHM_setup_mode_array(lalParams: dict):
  ModeArray = lalParams.get("mode_array", None) # ModeArray = XLALSimInspiralWaveformParamsLookupModeArray(lalParams)

  # /* If the mode array is empty, populate using a default choice of modes */
  if not ModeArray:
    # /* Default behaviour */
    jax.debug.print("Using default non-precessing modes for IMRPhenomXPHM: 2|2|, 2|1|, 3|3|, 3|2|, 4|4|.")
    ModeArray = xlal_sim_inspiral_create_mode_array()

    # /* IMRPhenomXHM has the following calibrated modes. 22 mode taken from IMRPhenomXAS */
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 2, 2)
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 2, 1)
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 3, 3)
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 3, 2)
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 4, 4)
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 2, -2)
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 2, -1)
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 3, -3)
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 3, -2)
    xlal_sim_inspiral_mode_array_activate_mode(ModeArray, 4, -4)
    lalParams["mode_array"] = ModeArray

  else:
      jax.debug.print("Using custom non-precessing modes for PhenomXPHM.") 


def check_mass_ratio(mass_ratio: float) -> float:
    """Check the mass ratio for validity within the model's calibration domain.

    Args:
        mass_ratio (float): The mass ratio of the binary system.

    Returns:
        float: The validated mass ratio.
    """

    def warn_extrapolation(mass_ratio: float) -> float:
        """Warn about extrapolation outside of NR calibration domain.

        Args:
            mass_ratio (float): The mass ratio of the binary system.

        Returns:
            float: The same mass ratio.
        """
        jax.debug.print(
            "Warning: Mass ratio = {mass_ratio}. Extrapolating outside of Numerical Relativity calibration domain. NNLO angles may become pathological at large mass ratios.",
            mass_ratio=mass_ratio,
        )
        return mass_ratio

    def error_too_large(mass_ratio: float) -> float:
        """Raise an error for mass ratios that are too large.

        Args:
            mass_ratio (float): The mass ratio of the binary system.

        Returns:
            float: The same mass ratio.
        """
        checkify.check(
            False,
            "Error: Mass ratio = {mass_ratio}. Model not valid at mass ratios beyond 1000.",
            mass_ratio=mass_ratio,
        )
        return mass_ratio

    mass_ratio = jax.lax.cond(mass_ratio >= 1000.0, error_too_large, lambda x: x, mass_ratio)
    mass_ratio = jax.lax.cond(
        (mass_ratio > 20.0) & (mass_ratio <= 1000.0),
        warn_extrapolation,
        lambda x: x,
        mass_ratio,
    )
    return mass_ratio


def check_spins(chi1z: float, chi2z: float) -> tuple[float, float]:
    """Check the spins for validity within the model's calibration domain.

    Args:
        chi1z (float): z-component of the dimensionless spin of the first body.
        chi2z (float): z-component of the dimensionless spin of the second body.

    Returns:
        tuple: A tuple containing possibly modified values of (chi1z, chi2z).
    """

    def warn_extrapolation_spins(args: tuple[float, float]) -> tuple[float, float]:
        """Warn about extrapolation to extremal spins.

        Args:
            args (tuple): A tuple containing (chi1z, chi2z).

        Returns:
            tuple: The same tuple (chi1z, chi2z).
        """
        chi1z, chi2z = args
        jax.debug.print(
            "Warning: Spins chi1z = {chi1z}, chi2z = {chi2z}. Extrapolating to extremal spins, model is not trusted.",
            chi1z=chi1z,
            chi2z=chi2z,
        )
        return chi1z, chi2z

    def no_warning(args: tuple[float, float]) -> tuple[float, float]:
        """No warning, return the spins as is.

        Args:
            args (tuple): A tuple containing (chi1z, chi2z).

        Returns:
            tuple: The same tuple (chi1z, chi2z).
        """
        return args

    chi1z, chi2z = jax.lax.cond(
        (jnp.abs(chi1z) > 0.99) | (jnp.abs(chi2z) > 0.99),
        warn_extrapolation_spins,
        no_warning,
        (chi1z, chi2z),
    )
    return chi1z, chi2z


def xlal_sim_imr_phenom_xphm(
    freqs: Array,
    hp_tilde: Array,
    hc_tilde: Array,
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
    f_min: float,
    f_max: float,
    delta_f: float,
    f_ref_in: float,
    lal_params: dict,
) -> None:
    """A JAX implementation of XLALSimIMRPhenomXPHM.

    It computes the plus and cross polarizations of the multimode precessing waveform for positive frequencies in an equally spaced grid.
    """
    # Set initial values of masses and z-components of spins to pass to imr_phenom_x_set_waveform_variables() so it can swap the matter parameters (masses and spins)
    # appropriately if m1 < m2, since the masses and spin vectors will also be swapped by xlal_imr_phenom_xp_check_masses_and_spins() below.
    m1_si_init = m1_si
    m2_si_init = m2_si
    chi1z_init = chi1z
    chi2z_init = chi2z

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
    checkify.check(delta_f > 0, "Error: delta_f must be positive.")
    checkify.check(m1_si > 0, "Error: m1 must be positive.")
    checkify.check(m2_si > 0, "Error: m2 must be positive.")
    checkify.check(f_min > 0, "Error: f_min must be positive.")
    checkify.check(f_max >= 0, "Error: f_max must be non-negative.")
    checkify.check(distance > 0, "Error: Distance must be positive.")

    # Compute the mass ratio
    mass_ratio = m1_si / m2_si

    # Check the mass ratio.
    mass_ratio = check_mass_ratio(mass_ratio)

    # Check the spins
    chi1z, chi2z = check_spins(chi1z, chi2z)

    # Check whether the modes chosen are available for the mode.
    checkify.check(
        check_input_mode_array(lal_params=lal_params)[1], "Error: Invalid modes selected in mode_array of lal_params."
    )

    # If no reference frequency is specified, set it to f_min.
    f_ref = jax.lax.cond(f_ref_in <= 0, lambda: f_min, lambda: f_ref_in)

    # Copy the lal_params
    lal_params_aux = lal_params.copy()

    # Initialize the useful powers of pi.
    error, powers_of_pi = imr_phenom_x_initialize_powers(jnp.pi)

    lal_params_dataclass = IMRPhenomXPHMParameterDataClass()
    # Initialize IMRPhenomX waveform struct and check that it is initialized correctly.
    waveform_variables = imr_phenom_x_set_waveform_variables(
        m1_si,
        m2_si,
        chi1z,
        chi2z,
        delta_f,
        f_ref,
        phi_ref,
        f_min,
        f_max,
        distance,
        inclination,
        lal_params_dataclass,
        powers_of_pi
    )

    # REAL8Sequence *freqs = XLALCreateREAL8Sequence(2) # To interface with ripple, the frequency array should probably be created outside and passed as an argument
    freqs.at[0].set(waveform_variables.f_min)
    freqs.at[1].set(waveform_variables.f_max_prime)

    # TODO
    # if(XLALSimInspiralWaveformParamsLookupPhenomXPNRUseTunedAngles(lalParams)){ # This is passed as an argument?
    #     XLAL_CHECK(
    #     (fRef >=  pWF->fMin)&&(fRef <= pWF->f_max_prime),
    #     XLAL_EFUNC,
    #     "Error: f_min = %.2f <= fRef = %.2f < f_max = %.2f required when using tuned angles.\n",pWF->fMin,fRef,pWF->f_max_prime)
    # }

    # TODO
    # /* Initialize IMRPhenomX Precession struct and check that it generated successfully */ 
    # IMRPhenomXPrecessionStruct *pPrec
    # pPrec  = XLALMalloc(sizeof(IMRPhenomXPrecessionStruct))

    IMRPhenomXPHM_setup_mode_array(lal_params_aux)

    # TODO
    # status = IMRPhenomXGetAndSetPrecessionVariables(
    #             pWF,
    #             pPrec,
    #             m1_SI,
    #             m2_SI,
    #             chi1x,
    #             chi1y,
    #             chi1z,
    #             chi2x,
    #             chi2y,
    #             chi2z,
    #             lalParams_aux,
    #             PHENOMXDEBUG
    #         )
    # XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Error: IMRPhenomXSetPrecessionVariables failed.\n")

    # TODO
    # /* We now call the core IMRPhenomXPHM waveform generator */
    # status = IMRPhenomXPHM_hplushcross(hptilde, hctilde, freqs, pWF, pPrec, lalParams_aux)
    # XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "IMRPhenomXPHM_hplushcross failed to generate IMRPhenomXHM waveform.\n")

    # TODO
    # /* Resize hptilde, hctilde */
    # REAL8 lastfreq
    # if (pWF->f_max_prime < pWF->fMax)
    # {
    #     /* The user has requested a higher f_max than Mf = fCut.
    #     Resize the frequency series to fill with zeros beyond the cutoff frequency. */
    #     lastfreq = pWF->fMax
    #     XLAL_PRINT_WARNING("The input f_max = %.2f Hz is larger than the internal cutoff of Mf=0.3 (%.2f Hz). Array will be filled with zeroes between these two frequencies.\n", pWF->fMax, pWF->f_max_prime)
    # }
    # else{  // We have to look for a power of 2 anyway.
    #     lastfreq = pWF->f_max_prime
    # }

    # TODO
    # // We want to have the length be a power of 2 + 1
    # size_t n_full = NextPow2(lastfreq / deltaF) + 1
    # size_t n = (*hptilde)->data->length

    # /* Resize the COMPLEX16 frequency series */
    # *hptilde = XLALResizeCOMPLEX16FrequencySeries(*hptilde, 0, n_full)
    # XLAL_CHECK (*hptilde, XLAL_ENOMEM, "Failed to resize h_+ COMPLEX16FrequencySeries of length %zu (for internal fCut=%f) to new length %zu (for user-requested f_max=%f).", n, pWF->fCut, n_full, pWF->fMax )

    # /* Resize the COMPLEX16 frequency series */
    # *hctilde = XLALResizeCOMPLEX16FrequencySeries(*hctilde, 0, n_full)
    # XLAL_CHECK (*hctilde, XLAL_ENOMEM, "Failed to resize h_x COMPLEX16FrequencySeries of length %zu (for internal fCut=%f) to new length %zu (for user-requested f_max=%f).", n, pWF->fCut, n_full, pWF->fMax )

    # /* Free memory */
    # LALFree(pWF)
    # LALFree(pPrec)
    # XLALDestroyREAL8Sequence(freqs)
    # XLALDestroyDict(lalParams_aux)

    # return XLAL_SUCCESS


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
    lal_params: dict,
) -> None:
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
    jax.debug.print(
        "Prec V.: {prec_version}",
        prec_version=xlal_sim_inspiral_waveform_params_lookup_phenom_xprec_version(lal_params),
    )
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
            jax.debug.print(
                "Warning: Mass ratio = {mass_ratio}. Extrapolating outside of Numerical Relativity calibration domain. NNLO angles may become pathological at large mass ratios.",
                mass_ratio=mass_ratio,
            )
            return mass_ratio

        def error_too_large(mass_ratio):
            checkify.check(
                False,
                "Error: Mass ratio = {mass_ratio}. Model not valid at mass ratios beyond 1000.",
                mass_ratio=mass_ratio,
            )
            return mass_ratio

        mass_ratio = jax.lax.cond(mass_ratio >= 1000.0, error_too_large, lambda x: x, mass_ratio)
        mass_ratio = jax.lax.cond(
            (mass_ratio > 20.0) & (mass_ratio <= 1000.0),
            warn_extrapolation,
            lambda x: x,
            mass_ratio,
        )
        return mass_ratio

    mass_ratio = check_mass_ratio(mass_ratio)

    # Check the spins
    def check_spins(chi1z, chi2z):
        def warn_extrapolation_spins(args):
            chi1z, chi2z = args
            jax.debug.print(
                "Warning: Spins chi1z = {chi1z}, chi2z = {chi2z}. Extrapolating to extremal spins, model is not trusted.",
                chi1z=chi1z,
                chi2z=chi2z,
            )
            return chi1z, chi2z

        def no_warning(args):
            return args

        chi1z, chi2z = jax.lax.cond(
            (jnp.abs(chi1z) > 0.99) | (jnp.abs(chi2z) > 0.99),
            warn_extrapolation_spins,
            no_warning,
            (chi1z, chi2z),
        )
        return chi1z, chi2z

    chi1z, chi2z = check_spins(chi1z, chi2z)

    # If no reference frequency is given, set it to the starting gravitational wave frequency.
    f_ref = jax.lax.cond(
        f_ref_in == 0.0,
        lambda f_min, _f_ref_in: f_min,
        lambda _f_min, f_ref_in: f_ref_in,
        f_min,
        f_ref_in,
    )

    # Use an auxiliary lal_dict to not overwrite the input argument
    lal_params_aux = lal_params.copy()

    # Check that the modes chosen are available for the model
    checkify.check(check_input_mode_array(lal_params_aux) == 0, "Error: Not available mode chosen.")

    jax.debug.print("Initializing waveform dict...")

    # Initialize IMR PhenomX waveform struct and check that it initialized correctly.
    # We pass inclination 0 since for the individual modes is not relevant.

    p_wf = imr_phenom_x_set_waveform_variables(
        m1_si,
        m2_si,
        chi1z,
        chi2z,
        0.0,
        f_ref,
        phi_ref,
        f_min,
        f_max,
        distance,
        inclination,
        lal_params_aux,
    )

    jax.debug.print("Initializing precession dict...")

    # Only relevant for SpinTaylor angles
    p_flag = xlal_sim_inspiral_waveform_params_lookup_phenom_xprec_version(lal_params_aux)

    p_prec = jax.lax.cond(
        p_flag == 310 | p_flag == 311 | p_flag == 320,
        p_flag == 321 | p_flag == 330,
        lambda _: {"m_min": jnp.max(1, jnp.abs(m)), "m_max": l, "l_max_pnr": l},
        lambda _: {},
        p_flag,
    )

    jax.lax.cond(
        xlal_sim_inspiral_waveform_params_lookup_phenom_xpnr_use_tuned_angles(lal_params_aux) == 1,
        lambda _: xlal_sim_inspiral_waveform_params_insert_phenom_xphm_threshold_mband(lal_params_aux, 0),
        lambda _: None,
        None,
    )

    imr_phenom_x_get_and_set_precession_variables(
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
        lal_params_aux,
    )

    jax.lax.cond(
        p_prec["precessing_tag"] == 3,
        lambda _: imr_phenom_x_initialize_euler_angles(p_wf, p_prec, lal_params),
        lambda _: None,
        None,
    )

    # Ensure recovering AS limit when modes are in the L0 frame.
    l0_frame = xlal_sim_inspiral_waveform_params_lookup_phenom_xphm_modes_l0_frame(
        lal_params_aux
    )  # placeholder function

    def handle_l0_frame(p_wf, phi_ref):
        jax.debug.print("The L0Frame option only works near the AS limit, it should not be used otherwise.")
        convention = xlal_sim_inspiral_waveform_params_lookup_phenom_xp_convention(
            lal_params_aux
        )  # placeholder function

        def case_0_5(p_wf):
            # pWF->phi0 = pPrec->phi0_aligned  # commented out in C
            return p_wf

        def case_1_6_7(p_wf):
            p_wf_new = p_wf.copy()
            p_wf_new["phi0"] = phi_ref
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
                p_wf,
            ),
            p_wf,
        )
        return p_wf

    p_wf = jax.lax.cond(
        l0_frame == 1,
        lambda p_wf: handle_l0_frame(p_wf, phi_ref),
        lambda p_wf: p_wf,
        p_wf,
    )

    jax.debug.print("Calling imr_phenom_xphm_one_mode...")

    xlal_sim_inspiral_waveform_params_insert_phenom_xhm_threshold_mband(lal_params_aux, 0)

    status = imr_phenom_xphm_one_mode(h_lm_pos, h_lm_neg, freqs, p_wf, p_prec, l, m, lal_params_aux)
    checkify.check(
        status == 0,
        "Error: imr_phenom_xphm_one_mode failed to generate IMRPhenomXHM waveform.",
    )

    # Transform modes to L0-frame if requested. It only works for (near) AS cases.
    l0_frame = xlal_sim_inspiral_waveform_params_lookup_phenom_xphm_modes_l0_frame(
        lal_params_aux
    )  # placeholder function

    def transform_l0_frame(h_lm_pos, h_lm_neg, m, p_prec):
        convention = xlal_sim_inspiral_waveform_params_lookup_phenom_xp_convention(
            lal_params_aux
        )  # placeholder function

        def do_nothing(h_lm_pos, h_lm_neg):
            return h_lm_pos, h_lm_neg

        def apply_shift(h_lm_pos, h_lm_neg):
            shiftpos = jnp.exp(1j * jnp.abs(m) * (p_prec["epsilon0"] - p_prec["alpha0"]))
            shiftneg = 1.0 / shiftpos
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
                h_lm_pos,
                h_lm_neg,
            ),
            h_lm_pos,
            h_lm_neg,
        )
        return h_lm_pos, h_lm_neg

    h_lm_pos, h_lm_neg = jax.lax.cond(
        l0_frame == 1,
        lambda h_lm_pos, h_lm_neg: transform_l0_frame(h_lm_pos, h_lm_neg, m, p_prec),
        lambda h_lm_pos, h_lm_neg: (h_lm_pos, h_lm_neg),
        h_lm_pos,
        h_lm_neg,
    )

    jax.debug.print("Call to imr_phenom_xphm_one_mode complete.")


# def imr_phenom_xphm_hphc(
#    freqs_In,                            #/**< Frequency array to evaluate the model. (fmin, fmax) for equally spaced grids. */
#    IMRPhenomXWaveformStruct *pWF,       #/**< IMRPhenomX Waveform Struct  */
#    IMRPhenomXPrecessionStruct *pPrec,   #/**< IMRPhenomXP Precession Struct  */
#    LALDict *lalParams                   #/**< LAL Dictionary Structure    */
# )
# {

#   if (pWF->f_max_prime <= pWF->fMin)
#   {
#     XLAL_ERROR(XLAL_EDOM, "(fCut = %g Hz) <= f_min = %g\n", pWF->f_max_prime, pWF->fMin);
#   }

#   /* Set LIGOTimeGPS */
#   LIGOTimeGPS ligotimegps_zero = LIGOTIMEGPSZERO; // = {0,0}

#   REAL8 deltaF = pWF->deltaF;

#    LALValue *ModeArray = XLALSimInspiralWaveformParamsLookupModeArray(lalParams);

#    /* At this point ModeArray should contain the list of modes
#    and therefore if NULL then something is wrong and abort. */
#     if (ModeArray == NULL)
#     {
#      XLAL_ERROR(XLAL_EDOM, "ModeArray is NULL when it shouldn't be. Aborting.\n");
#     }

#     INT4 status = 0; //Variable to check correct functions calls.
#     /*
#         Take input/default value for the threshold of the Multibanding for the hlms modes.
#         If = 0 then do not use Multibanding. Default value defined in XLALSimInspiralWaveformParams.c.
#         If the input freqs_In is non-uniform the Multibanding has been already switche off.
#     */
#     REAL8 thresholdMB  = XLALSimInspiralWaveformParamsLookupPhenomXHMThresholdMband(lalParams);

#    if(pPrec->precessing_tag==3){
#         status=IMRPhenomX_Initialize_Euler_Angles(pWF,pPrec,lalParams);
#         XLAL_CHECK(status==XLAL_SUCCESS, XLAL_EDOM, "%s: Error in IMRPhenomX_Initialize_Euler_Angles.\n",__func__);
#       }


#   /* Build the frequency array and initialize hptilde to the length of freqs. */
#   REAL8Sequence *freqs;
#   UINT4 offset = SetupWFArrays(&freqs, hptilde, freqs_In, pWF, ligotimegps_zero);

#   /* Initialize hctilde according to hptilde. */
#   size_t npts = (*hptilde)->data->length;
#   *hctilde = XLALCreateCOMPLEX16FrequencySeries("hctilde: FD waveform", &(*hptilde)->epoch, (*hptilde)->f0, pWF->deltaF, &lalStrainUnit, npts);
#   XLAL_CHECK (*hctilde, XLAL_ENOMEM, "Failed to allocated waveform COMPLEX16FrequencySeries of length %zu.", npts);
#   memset((*hctilde)->data->data, 0, npts * sizeof(COMPLEX16));
#   XLALUnitMultiply(&((*hctilde)->sampleUnits), &((*hctilde)->sampleUnits), &lalSecondUnit);

#   /* Object to store the non-precessing 22 mode waveform and to be recycled when calling the 32 mode in multibanding. */
#   COMPLEX16FrequencySeries *htilde22 = NULL;



#   /* Initialize the power of pi for the HM internal functions. */
#   status = IMRPhenomX_Initialize_Powers(&powers_of_lalpiHM, LAL_PI);
#   XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Failed to initialize useful powers of LAL_PI.");


#   SphHarmFrequencySeries **hlms = XLALMalloc(sizeof(SphHarmFrequencySeries));
#   *hlms = NULL;
#   if (XLALSimInspiralWaveformParamsLookupPhenomXPHMTwistPhenomHM(lalParams)==1)
#   {
#     /* evaluate all hlm modes */
#     status = XLALSimIMRPhenomHMGethlmModes(
#         hlms,
#         freqs,
#         pWF->m1_SI,
#         pWF->m2_SI,
#         pPrec->chi1x,
#         pPrec->chi1y,
#         pWF->chi1L,
#         pPrec->chi2x,
#         pPrec->chi2y,
#         pWF->chi2L,
#         pWF->phi0,
#         //pWF->deltaF,
#         0,
#         pWF->fRef,
#         lalParams);
#     XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "XLALSimIMRPhenomHMGethlmModes failed");
#   }

#   /* Set up code for using PNR tuned angles */
#   int IMRPhenomXPNRUseTunedAngles = pPrec->IMRPhenomXPNRUseTunedAngles;
#   int AntisymmetricWaveform = pPrec->IMRPhenomXAntisymmetricWaveform;

#   IMRPhenomX_PNR_angle_spline *hm_angle_spline = NULL;
#   REAL8 Mf_RD_22 = pWF->fRING;
#   REAL8 Mf_RD_lm = 0.0;

#   if (IMRPhenomXPNRUseTunedAngles)
#   {
#     /* We're using tuned angles! */
#     /* Allocate the spline interpolant struct */

#     hm_angle_spline = (IMRPhenomX_PNR_angle_spline *) XLALMalloc(sizeof(IMRPhenomX_PNR_angle_spline));
#     if (!hm_angle_spline)
#     {
#       XLAL_ERROR(XLAL_EFUNC, "hm_angle_spline struct allocation failed in LALSimIMRPhenomXPHM.c.");
#     }

#     /* Generate interpolant structs for the (2,2) angles */
#     status = IMRPhenomX_PNR_GeneratePNRAngleInterpolants(hm_angle_spline, pWF, pPrec, lalParams);
#     XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Error: IMRPhenomX_PNR_GeneratePNRAngleInterpolants failed.\n");

#     /* Here we assign the reference values of alpha and gamma to their values in the precession struct */
#     /* NOTE: the contribution from pPrec->alpha0 is assigned in IMRPhenomX_PNR_RemapThetaJSF */
#     pPrec->alpha_offset = gsl_spline_eval(hm_angle_spline->alpha_spline, pWF->fRef, hm_angle_spline->alpha_acc);
#     /* NOTE: the sign is flipped between gamma and epsilon */
#     pPrec->epsilon_offset = -gsl_spline_eval(hm_angle_spline->gamma_spline, pWF->fRef, hm_angle_spline->gamma_acc) - pPrec->epsilon0;  // note the sign difference between gamma and epsilon

#     /* Remap the J-frame sky location to use beta instead of ThetaJN */
#     REAL8 betaPNR_ref = gsl_spline_eval(hm_angle_spline->beta_spline, pWF->fRef, hm_angle_spline->beta_acc);
#     if(isnan(betaPNR_ref) || isinf(betaPNR_ref)) XLAL_ERROR(XLAL_EDOM, "Error in %s: gsl_spline_eval for beta returned invalid value.\n",__func__);
#     status = IMRPhenomX_PNR_RemapThetaJSF(betaPNR_ref, pWF, pPrec, lalParams);
#     XLAL_CHECK(
#         XLAL_SUCCESS == status,
#         XLAL_EFUNC,
#         "Error: IMRPhenomX_PNR_RemapThetaJSF failed in IMRPhenomX_PNR_GeneratePNRAngles.");
#   }

#   /******************************************************/
#   /******** Antisymmetric waveform generated here ********/
#   /******************************************************/
#   REAL8Sequence *antiSym_amp = NULL;
#   REAL8Sequence *antiSym_phi = NULL;

#   if (AntisymmetricWaveform && IMRPhenomXPNRUseTunedAngles)
#   {
#     antiSym_amp = XLALCreateREAL8Sequence(freqs->length);
#     antiSym_phi = XLALCreateREAL8Sequence(freqs->length);
#   }



#   /***** Loop over non-precessing modes ******/
#   for (UINT4 ell = 2; ell <= L_MAX; ell++)
#   {
#     for (UINT4 emmprime = 1; emmprime <= ell; emmprime++)
#     {
#       /* Loop over only positive mprime is intentional.
#         The single mode function returns the negative mode h_l-mprime, and the positive
#         is added automatically in during the twisting up in IMRPhenomXPHMTwistUp.
#         First check if (l,m) mode is 'activated' in the ModeArray.
#         If activated then generate the mode, else skip this mode.
#       */
#       if (XLALSimInspiralModeArrayIsModeActive(ModeArray, ell, emmprime) != 1)
#       { /* skip mode */
#         continue;
#       } /* else: generate mode */

#       /* Skip twisting-up if the non-precessing mode is zero. */
#       if((pWF->q == 1) && (pWF->chi1L == pWF->chi2L) && (emmprime % 2 != 0))
#       {
#         continue;
#       }

#       /* Compute and store phase alignment quantities for each
#       non-precessing (ell,emm) multipole moment. Note that the
#       (2,2) moment is handled separately within XAS routines. */
#       if (
#             pWF->APPLY_PNR_DEVIATIONS && pWF->IMRPhenomXPNRForceXHMAlignment && (ell != 2) && (emmprime != 2)
#           )
#       {
#         /* Compute and store phase alignment quantities for each
#         non-precessing (ell,emm) multipole moment */
#         IMRPhenomXHM_PNR_SetPhaseAlignmentParams(ell,emmprime,pWF,pPrec,lalParams);
#       }

#       #if DEBUG == 1
#       printf("\n\n*********************************\n*Non-precessing Mode %i%i \n******************************\n",ell, emmprime);
#       // Save the hlm mode into a file
#       FILE *fileangle;
#       char fileSpec[40];

#       if(pPrec->MBandPrecVersion == 0)
#       {
#           sprintf(fileSpec, "angles_hphc_%i%i.dat", ell, emmprime);
#       }
#       else
#       {
#           sprintf(fileSpec, "angles_hphc_MB_%i%i.dat", ell, emmprime);
#       }
#       printf("\nOutput angle file: %s\r\n", fileSpec);
#       fileangle = fopen(fileSpec,"w");

#       fprintf(fileangle,"# q = %.16e m1 = %.16e m2 = %.16e chi1 = %.16e chi2 = %.16e lm = %i%i Mtot = %.16e distance = %.16e\n", pWF->q, pWF->m1, pWF->m2, pWF->chi1L, pWF->chi2L, ell, emmprime, pWF->Mtot, pWF->distance/LAL_PC_SI/1e6);
#       fprintf(fileangle,"#fHz   cexp_i_alpha(re im)   cexp_i_epsilon(re im)    cexp_i_betah(re im)\n");

#       fclose(fileangle);
#       #endif

#       /* Variable to store the strain of only one (negative) mode: h_l-mprime */
#       COMPLEX16FrequencySeries *htildelm = NULL;

#       if (XLALSimInspiralWaveformParamsLookupPhenomXPHMTwistPhenomHM(lalParams)==1)
#       {
#         INT4 minus1l = 1;
#         if(ell % 2 !=0) minus1l = -1;
#         COMPLEX16FrequencySeries *htildelmPhenomHM = NULL;
#         /* Initialize the htilde frequency series */
#         htildelm = XLALCreateCOMPLEX16FrequencySeries("htildelm: FD waveform", &ligotimegps_zero, 0, pWF->deltaF, &lalStrainUnit, npts);
#         /* Check that frequency series generated okay */
#         XLAL_CHECK(htildelm,XLAL_ENOMEM,"Failed to allocate COMPLEX16FrequencySeries of length %zu for f_max = %f, deltaF = %g.\n", npts, freqs_In->data[freqs_In->length - 1], pWF->deltaF);
#         memset((htildelm)->data->data, 0, npts * sizeof(COMPLEX16));
#         XLALUnitMultiply(&((htildelm)->sampleUnits), &((htildelm)->sampleUnits), &lalSecondUnit);

#         htildelmPhenomHM = XLALSphHarmFrequencySeriesGetMode(*hlms, ell, emmprime);
#         for(UINT4 idx = 0; idx < freqs->length; idx++)
#         {
#           htildelm->data->data[idx+offset] = minus1l * htildelmPhenomHM->data->data[idx] * pWF->amp0;
#         }
#         //XLALDestroyCOMPLEX16FrequencySeries(htildelmPhenomHM);
#       }
#       else
#       {
#         /* Compute non-precessing mode */
#         if (thresholdMB == 0){  // No multibanding
#           if(ell == 2 && emmprime == 2)
#           {
#             status = IMRPhenomXASGenerateFD(&htildelm, freqs, pWF, lalParams);
#             XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "IMRPhenomXASGenerateFD failed to generate IMRPhenomXHM waveform.");
#           }
#           else
#           {
#             status = IMRPhenomXHMGenerateFDOneMode(&htildelm, freqs, pWF, ell, emmprime, lalParams);
#             XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "IMRPhenomXHMGenerateFDOneMode failed to generate IMRPhenomXHM waveform.");
#           }
#         }
#         else{               // With multibanding
#           if(ell==3 && emmprime==2){  // mode with mode-mixing
#             status = IMRPhenomXHMMultiBandOneModeMixing(&htildelm, htilde22, pWF, ell, emmprime, lalParams);
#             XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "IMRPhenomXHMMultiBandOneModeMixing failed to generate IMRPhenomXHM waveform.");
#           }
#           else{                  // modes without mode-mixing including 22 mode
#             status = IMRPhenomXHMMultiBandOneMode(&htildelm, pWF, ell, emmprime, lalParams);
#             XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "IMRPhenomXHMMultiBandOneMode failed to generate IMRPhenomXHM waveform.");
#           }

#           /* IMRPhenomXHMMultiBandOneMode* functions set pWF->deltaF=0 internally, we put it back here. */
#           pWF->deltaF = deltaF;

#           /* If the 22 and 32 modes are active, we recycle the 22 mode for the mixing in the 32 and it is passed to IMRPhenomXHMMultiBandOneModeMixing.
#             The 22 mode is always computed first than the 32, we store the 22 mode in the variable htilde22. */
#           if(ell==2 && emmprime==2 && XLALSimInspiralModeArrayIsModeActive(ModeArray, 3, 2)==1){
#             htilde22 = XLALCreateCOMPLEX16FrequencySeries("hptilde: FD waveform", &(ligotimegps_zero), 0.0, pWF->deltaF, &lalStrainUnit, htildelm->data->length);
#             for(UINT4 idx = 0; idx < htildelm->data->length; idx++){
#               htilde22->data->data[idx] = htildelm->data->data[idx];
#             }
#           }
#         }
#       }

#       if(ell==2 && emmprime==2)
#       {
#         if (AntisymmetricWaveform && IMRPhenomXPNRUseTunedAngles)
#         {
#           IMRPhenomX_PNR_GenerateAntisymmetricWaveform(antiSym_amp,antiSym_phi,freqs,pWF,pPrec,lalParams);
#         }
#       }

#       if (!(htildelm)){ XLAL_ERROR(XLAL_EFUNC); }

#       /*
#          For very special cases of deltaF, it can happen that building htildelm with 'freqs_In' or with 'freqs' gives different lengths.
#          In case that happens we resize here to the correct length. We could also have called GenerateFD passing freqs_In,
#          but in that ways we would be computing the uniform frequency array twice.
#          Alsom doing as here we cover the multibanding and PhenomHM cases.
#       */
#       if(htildelm->data->length != npts)
#       {
#         htildelm = XLALResizeCOMPLEX16FrequencySeries(htildelm, 0, npts);
#         XLAL_CHECK (htildelm, XLAL_ENOMEM, "Failed to resize hlm COMPLEX16FrequencySeries" );
#       }

#       /* htildelm is recomputed every time in the loop. Check that it always comes out with the same length */
#       XLAL_CHECK (    ((*hptilde)->data->length==htildelm->data->length)
#                   && ((*hctilde)->data->length==htildelm->data->length),
#                   XLAL_EBADLEN,
#                   "Inconsistent lengths between frequency series htildelm (%d), hptilde (%d) and hctilde (%d).",
#                   htildelm->data->length, (*hptilde)->data->length, (*hctilde)->data->length
#                 );

#       /*
#                               TWISTING UP
#           Transform modes from the precessing L-frame to inertial J-frame.
#       */


#       /* Variable to store the non-precessing waveform in one frequency point. */
#       COMPLEX16 hlmcoprec=0.0;
#       COMPLEX16 hlmcoprec_antiSym=0.0;

#       /* No Multibanding for the angles. */
#       if(pPrec->MBandPrecVersion == 0)
#       {
#         #if DEBUG == 1
#         printf("\n****************************************************************\n");
#         printf("\n*              NOT USING MBAND FOR ANGLES %i                *\n", offset);
#         printf("\n****************************************************************\n");
#         #endif

#         // Let the people know if twisting up will not take place
#         if( pWF->IMRPhenomXReturnCoPrec == 1 )
#         {
#           #if DEBUG == 1
#             printf("\n** We will not twist up the HM waveforms **\n");
#           #endif
#         }

#         /* set variables for PNR angles if needed */
#         REAL8 Mf_high = 0.0;
#         REAL8 Mf_low = 0.0;
#         UINT4 PNRtoggleInspiralScaling = pPrec->PNRInspiralScaling;

#         // set PNR transition frequencies if needed
#         if (IMRPhenomXPNRUseTunedAngles)
#         {

#           if((ell==2)&&(emmprime==2))
#           {
#             /* the frequency parameters don't matter in this case */
#             Mf_RD_lm = 0.0;
#           }
#           else
#           {
#             /* Get the (l,m) RD frequency */

#             Mf_RD_lm = IMRPhenomXHM_GenerateRingdownFrequency(ell, emmprime, pWF);

#             status = IMRPhenomX_PNR_LinearFrequencyMapTransitionFrequencies(&Mf_low, &Mf_high, emmprime, Mf_RD_22, Mf_RD_lm, pPrec);
#             XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Error: IMRPhenomX_PNR_LinearFrequencyMapTransitionFrequencies failed.\n");
#           }
#         }

#         if(pPrec->precessing_tag==3)
#          pPrec->gamma_in = 0.;

#         for (UINT4 idx = 0; idx < freqs->length; idx++)
#         {
#           double Mf             = pWF->M_sec * freqs->data[idx];

#           /* Do not generate waveform above Mf_max (default Mf = 0.3) */
#           if(Mf <= (pWF->f_max_prime * pWF->M_sec))
#           {
#             hlmcoprec             = htildelm->data->data[idx + offset];  /* Co-precessing waveform for one freq point */
#             COMPLEX16 hplus       = 0.0;  /* h_+ */
#             COMPLEX16 hcross      = 0.0;  /* h_x */

#             if( pWF->IMRPhenomXReturnCoPrec == 1 )
#             {
#               // Do not twist up
#               hplus  =  0.5 * hlmcoprec;
#               hcross = -0.5 * I * hlmcoprec;

#               //
#               if( pWF->PhenomXOnlyReturnPhase ){
#                 // Set hplus to phase (as will be stored in hlmcoprec) and hcross to zero
#                 hplus  = hlmcoprec; // NOTE that here hlmcoprec = waveform_phase (assuming one multipole moment)
#                 hcross = 0;
#               }

#             }
#             else
#             {
#               if(IMRPhenomXPNRUseTunedAngles)
#               {
#                 REAL8 Mf_mapped = IMRPhenomX_PNR_LinearFrequencyMap(Mf, ell, emmprime, Mf_low, Mf_high, Mf_RD_22, Mf_RD_lm, PNRtoggleInspiralScaling);
#                 REAL8 f_mapped = XLALSimIMRPhenomXUtilsMftoHz(Mf_mapped, pWF->Mtot);

#                 pPrec->alphaPNR = gsl_spline_eval(hm_angle_spline->alpha_spline, f_mapped, hm_angle_spline->alpha_acc);
#                 pPrec->betaPNR = gsl_spline_eval(hm_angle_spline->beta_spline, f_mapped, hm_angle_spline->beta_acc);
#                 pPrec->gammaPNR = gsl_spline_eval(hm_angle_spline->gamma_spline, f_mapped, hm_angle_spline->gamma_acc);
#               }

#               // Twist up symmetric strain
#              status = IMRPhenomXPHMTwistUp(Mf, hlmcoprec, pWF, pPrec, ell, emmprime, &hplus, &hcross);
#               XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "Call to IMRPhenomXPHMTwistUp failed.");

#               if(ell == 2 && emmprime == 2 && AntisymmetricWaveform && IMRPhenomXPNRUseTunedAngles)
#               {
#                 COMPLEX16 hplus_antiSym       = 0.0;
#                 COMPLEX16 hcross_antiSym      = 0.0;
#                 hlmcoprec_antiSym = antiSym_amp->data[idx]*cexp(I*antiSym_phi->data[idx]);
#                 pPrec->PolarizationSymmetry = -1.0;
#                 IMRPhenomXPHMTwistUp(Mf, hlmcoprec_antiSym, pWF, pPrec, ell, emmprime, &hplus_antiSym, &hcross_antiSym);
#                 pPrec->PolarizationSymmetry = 1.0;
#                 hplus += hplus_antiSym;
#                 hcross += hcross_antiSym;
#               }
#             }

#             (*hptilde)->data->data[idx + offset] += hplus;
#             (*hctilde)->data->data[idx + offset] += hcross;
#           }
#           else
#           {
#             /* Mf > Mf_max, so return 0 */
#             (*hptilde)->data->data[idx + offset] += 0.0 + I*0.0;
#             (*hctilde)->data->data[idx + offset] += 0.0 + I*0.0;
#           }
#         }

#         if(IMRPhenomXPNRUseTunedAngles)
#         {
#           gsl_interp_accel_reset(hm_angle_spline->alpha_acc);
#           gsl_interp_accel_reset(hm_angle_spline->beta_acc);
#           gsl_interp_accel_reset(hm_angle_spline->gamma_acc);
#         }

#         // If we only want the coprecessing waveform, then exit
#         // if( pWF->IMRPhenomXReturnCoPrec == 1 ) return XLAL_SUCCESS;
#         if( pWF->IMRPhenomXReturnCoPrec == 1 ) {
#           return XLAL_SUCCESS;
#         }

#       }
#       else
#       {
#         /*
#           Multibanding for the angles.

#           - In this first release we use the same coarse grid that is used for computing the non-precessing modes.
#           - This grid is discussed in section II-A of arXiv:2001.10897. See also section D of Precessing paper.
#           - This grid is computed with the function XLALSimIMRPhenomXMultibandingVersion defined in LALSimIMRPhenomXHM_multiband.c.
#           - The version of the coarse grid will be changed with the option 'MBandPrecVersion' defined in LALSimInspiralWaveformParams.c.
#           - Currently there is only one version available and the option value for that is 0, which is the default value.
#         */

#         #if DEBUG == 1
#         printf("\n****************************************************************\n");
#         printf("\n*                 USING MBAND FOR ANGLES                       *\n");
#         printf("\n****************************************************************\n");
#         #endif



#         /* Compute non-uniform coarse frequency grid as 1D array */
#         REAL8Sequence *coarseFreqs;
#         XLALSimIMRPhenomXPHMMultibandingGrid(&coarseFreqs, ell, emmprime, pWF, lalParams);

#         UINT4 lenCoarseArray = coarseFreqs->length;


#         /* Euler angles */
#         REAL8 alpha        = 0.0;
#         REAL8 epsilon      = 0.0;

#         REAL8 cBetah       = 0.0;
#         REAL8 sBetah       = 0.0;

#         /* Variables to store the Euler angles in the coarse frequency grid. */
#         REAL8 *valpha      = (REAL8*)XLALMalloc(lenCoarseArray * sizeof(REAL8));
#         REAL8 *vepsilon    = (REAL8*)XLALMalloc(lenCoarseArray * sizeof(REAL8));
#         REAL8 *vbetah      = (REAL8*)XLALMalloc(lenCoarseArray * sizeof(REAL8));

#         if(IMRPhenomXPNRUseTunedAngles)
#         {
#           REAL8 Mf_high = 0.0;
#           REAL8 Mf_low = 0.0;
#           REAL8 fCut = pWF->fCut;

#           if ((ell==2)&&(emmprime==2))
#           {
#             /* the frequency parameters don't matter here */
#             Mf_RD_lm = 0.0;
#           }
#           else
#           {
#             /* Get the (l,m) RD frequency */

#             Mf_RD_lm = IMRPhenomXHM_GenerateRingdownFrequency(ell, emmprime, pWF);

#             status = IMRPhenomX_PNR_LinearFrequencyMapTransitionFrequencies(&Mf_low, &Mf_high, emmprime, Mf_RD_22, Mf_RD_lm, pPrec);
#             XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Error: IMRPhenomX_PNR_LinearFrequencyMapTransitionFrequencies failed.\n");
#           }

#           UINT4 PNRtoggleInspiralScaling = pPrec->PNRInspiralScaling;
#           #if DEBUG == 1
#             // Save the hlm mode into a file
#             FILE *fileangle0101;
#             char fileSpec0101[40];

#             sprintf(fileSpec0101, "angles_pnr_MB_%i%i.dat", ell, emmprime);

#             fileangle0101 = fopen(fileSpec0101,"w");

#             fprintf(fileangle0101,"#Mf  fHz   alpha   beta    gamma\n");

#             fprintf(fileangle0101,"#Mf_low = %.16e\n",Mf_low);
#             fprintf(fileangle0101,"#Mf_high = %.16e\n",Mf_high);
#             fprintf(fileangle0101,"#Mf_RD_22 = %.16e\n",Mf_RD_22);
#             fprintf(fileangle0101,"#Mf_RD_lm = %.16e\n",Mf_RD_lm);

#           #endif

#           for(UINT4 j=0; j<lenCoarseArray; j++)
#           {
#             REAL8 Mf = coarseFreqs->data[j];
#             REAL8 Mf_mapped = IMRPhenomX_PNR_LinearFrequencyMap(Mf, ell, emmprime, Mf_low, Mf_high, Mf_RD_22, Mf_RD_lm, PNRtoggleInspiralScaling);
#             REAL8 f_mapped = XLALSimIMRPhenomXUtilsMftoHz(Mf_mapped, pWF->Mtot);

#             /* add in security to avoid frequency extrapolation */
#             f_mapped = (f_mapped > fCut) ? fCut : f_mapped;

#             double beta = gsl_spline_eval(hm_angle_spline->beta_spline, f_mapped, hm_angle_spline->beta_acc);

#             valpha[j]   = gsl_spline_eval(hm_angle_spline->alpha_spline, f_mapped, hm_angle_spline->alpha_acc) - pPrec->alpha_offset;
#             vepsilon[j] = -1.0 * gsl_spline_eval(hm_angle_spline->gamma_spline, f_mapped, hm_angle_spline->gamma_acc) - pPrec->epsilon_offset;
#             vbetah[j]   = beta / 2.0;

#             #if DEBUG == 1

#               fprintf(fileangle0101,"%.16e\t%.16e\t%.16e\t%.16e\t%.16e\n",Mf,f_mapped,valpha[j],beta,vepsilon[j]);

#             #endif
#           }


#           #if DEBUG == 1
#             fclose(fileangle0101);
#           #endif

#           gsl_interp_accel_reset(hm_angle_spline->alpha_acc);
#           gsl_interp_accel_reset(hm_angle_spline->beta_acc);
#           gsl_interp_accel_reset(hm_angle_spline->gamma_acc);

#         }
#         else
#         {
#           switch(pPrec->IMRPhenomXPrecVersion)
#           {
#             case 101:
#             case 102:
#             case 103:
#             case 104:
#             {
#               /* Use NNLO PN Euler angles */
#               /* Evaluate angles in coarse freq grid */
#               for(UINT4 j=0; j<lenCoarseArray; j++)
#               {
#                 REAL8 Mf = coarseFreqs->data[j];

#                 /* This function already add the offsets to the angles. */
#                 Get_alpha_beta_epsilon(&alpha, &cBetah, &sBetah, &epsilon, emmprime, Mf, pPrec, pWF);

#                 valpha[j]   = alpha;
#                 vepsilon[j] = epsilon;
#                 vbetah[j]   = acos(cBetah);
#               }
#               break;
#             }
#             case 220:
#             case 221:
#             case 222:
#             case 223:
#             case 224:
#             {
#               /* Use MSA Euler angles. */
#               /* Evaluate angles in coarse freq grid */
#               for(UINT4 j=0; j<lenCoarseArray; j++)
#               {
#                 /* Get Euler angles. */
#                 REAL8 Mf = coarseFreqs->data[j];
#                 const REAL8 v        = cbrt (LAL_PI * Mf * (2.0 / emmprime) );
#                 const vector vangles = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(v,pWF,pPrec);
#                 REAL8 cos_beta  = 0.0;

#                 /* Get the offset for the Euler angles alpha and epsilon. */
#                 REAL8 alpha_offset_mprime = 0, epsilon_offset_mprime = 0;
#                 Get_alpha_epsilon_offset(&alpha_offset_mprime, &epsilon_offset_mprime, emmprime, pPrec);

#                 valpha[j]   = vangles.x - alpha_offset_mprime;
#                 vepsilon[j] = vangles.y - epsilon_offset_mprime;
#                 cos_beta    = vangles.z;

#                 status = IMRPhenomXWignerdCoefficients_cosbeta(&cBetah, &sBetah, cos_beta);
#                 XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "Call to IMRPhenomXWignerdCoefficients_cosbeta failed.");

#                 vbetah[j]   = acos(cBetah);
#               }
#               break;
#             }

#              case 310:
#              case 311:
#              case 320:
#              case 321:
# 	     case 330:

#             {

#                 /* Get the offset for the Euler angles alpha and epsilon. */
#                 REAL8 alpha_offset_mprime = pPrec->alpha_ref- pPrec->alpha0;
#                 REAL8 epsilon_offset_mprime = -pPrec->gamma_ref-pPrec->epsilon0;

#                 REAL8 cos_beta=0., gamma=0., alpha_i=0.;
#                 REAL8 Mf;
#                 int success;


#                 /* Evaluate angles in coarse freq grid */
#                 for(UINT4 j=0; j<lenCoarseArray; j++)
#                 {

#                     success = 0;
#                     Mf = coarseFreqs->data[j]*(2.0/emmprime);


#                 if(Mf< pPrec->ftrans_MRD)
#                  {
#                         success = gsl_spline_eval_e(pPrec->alpha_spline, Mf, pPrec->alpha_acc,&alpha_i);
#                         success = success + gsl_spline_eval_e(pPrec->cosbeta_spline, Mf, pPrec->cosbeta_acc,&cos_beta);
#                         success = success + gsl_spline_eval_e(pPrec->gamma_spline,  Mf, pPrec->gamma_acc, &gamma);

#                         XLAL_CHECK(success == XLAL_SUCCESS, XLAL_EFUNC, "%s: Failed to interpolate Euler angles at f=%.7f. \n",__func__,XLALSimIMRPhenomXUtilsMftoHz(Mf,pWF->Mtot));
#                  }

#                else {

#                     if(pPrec->IMRPhenomXPrecVersion==320 || pPrec->IMRPhenomXPrecVersion==321 ){

#                         alpha_i=alphaMRD(Mf,pPrec->alpha_params);
#                         cos_beta=cos(betaMRD(Mf,pWF,pPrec->beta_params));
#                         success = gsl_spline_eval_e(pPrec->gamma_spline,  Mf, pPrec->gamma_acc, &gamma);
#                         if(success!=XLAL_SUCCESS){

#                         if(j>0)
#                         {
#                             REAL8 dMf=(coarseFreqs->data[j]-coarseFreqs->data[j-1])* (2.0 / emmprime);
#                             REAL8 deltagamma=0.;
#                             success = gamma_from_alpha_cosbeta(&deltagamma, Mf,dMf,pWF,pPrec);
#                             if(success!=XLAL_SUCCESS) gamma = pPrec->gamma_in;
#                             else gamma = pPrec->gamma_in+deltagamma;

#                         }

#                        else

#                         {
#                             success = gsl_spline_eval_e(pPrec->gamma_spline, Mf, pPrec->gamma_acc,&gamma);
#                             if(success!=XLAL_SUCCESS) gamma = pPrec->gamma_in;
#                         }
#                         }

#                             }

#                     else{

#                         alpha_i=pPrec->alpha_ftrans;
#                         cos_beta=pPrec->cosbeta_ftrans;
#                         gamma=pPrec->gamma_ftrans;

#                         }

#                 }

#                 pPrec->gamma_in = gamma;

#                 // make sure |cos(beta)| does not exceed 1 due to roundoff errors
#                 if(fabs(cos_beta)>1)
#                     cos_beta=copysign(1.0, cos_beta);

#                 valpha[j]= alpha_i- alpha_offset_mprime;
#                 vepsilon[j] = -gamma - epsilon_offset_mprime;


#                 status = IMRPhenomXWignerdCoefficients_cosbeta(&cBetah, &sBetah, cos_beta);
#                 XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "Call to IMRPhenomXWignerdCoefficients_cosbeta failed.");
#                 vbetah[j]   = acos(cBetah);

#             }


#             break;
#           }



#            default:
#             {
#               XLAL_ERROR(XLAL_EINVAL,"Error: IMRPhenomXPrecVersion not recognized. Recommended default is 223.\n");
#               break;
#             }
#           }
#         }

#         /*
#             We have the three Euler angles evaluated in the coarse frequency grid.
#             Now we have to carry out the iterative linear interpolation for the complex exponential of each Euler angle. This follows the procedure of eq. 2.32 in arXiv:2001.10897..
#             The result will be three arrays of complex exponential evaluated in the finefreqs.
#         */
#         UINT4 fine_count = 0, ratio;
#         REAL8 Omega_alpha, Omega_epsilon, Omega_betah, Qalpha, Qepsilon, Qbetah;
#         REAL8 Mfhere, Mfnext, evaldMf;
#         Mfnext = coarseFreqs->data[0];
#         evaldMf = XLALSimIMRPhenomXUtilsHztoMf(pWF->deltaF, pWF->Mtot);

#         /*
#             Number of points where the waveform will be computed.
#             It is the same for all the modes and could be computed outside the loop, it is here for clarity since it is not used anywhere else.
#         */
#         size_t iStop  = (size_t) (pWF->f_max_prime / pWF->deltaF) + 1 - offset;

#         UINT4 length_fine_grid = iStop + 3; // This is just to reserve memory, add 3 points of buffer.

#         COMPLEX16 *cexp_i_alpha   = (COMPLEX16*)XLALMalloc(length_fine_grid * sizeof(COMPLEX16));
#         COMPLEX16 *cexp_i_epsilon = (COMPLEX16*)XLALMalloc(length_fine_grid * sizeof(COMPLEX16));
#         COMPLEX16 *cexp_i_betah   = (COMPLEX16*)XLALMalloc(length_fine_grid * sizeof(COMPLEX16));


#         #if DEBUG == 1
#         printf("\n\nLENGTHS fine grid estimate, coarseFreqs->length = %i %i\n", length_fine_grid, lenCoarseArray);
#         printf("fine_count, htildelm->length, offset = %i %i %i\n", fine_count, htildelm->data->length, offset);
#         #endif

#         /* Loop over the coarse freq points */
#         for(UINT4 j = 0; j<lenCoarseArray-1 && fine_count < iStop; j++)
#         {
#           Mfhere = Mfnext;
#           Mfnext = coarseFreqs->data[j+1];

#           Omega_alpha   = (valpha[j + 1]   - valpha[j])  /(Mfnext - Mfhere);
#           Omega_epsilon = (vepsilon[j + 1] - vepsilon[j])/(Mfnext - Mfhere);
#           Omega_betah   = (vbetah[j + 1]   - vbetah[j])  /(Mfnext - Mfhere);

#           cexp_i_alpha[fine_count]   = cexp(I*valpha[j]);
#           cexp_i_epsilon[fine_count] = cexp(I*vepsilon[j]);
#           cexp_i_betah[fine_count]   = cexp(I*vbetah[j]);

#           Qalpha   = cexp(I*evaldMf*Omega_alpha);
#           Qepsilon = cexp(I*evaldMf*Omega_epsilon);
#           Qbetah   = cexp(I*evaldMf*Omega_betah);

#           fine_count++;

#           REAL8 dratio = (Mfnext-Mfhere)/evaldMf;
#           UINT4 ceil_ratio  = ceil(dratio);
#           UINT4 floor_ratio = floor(dratio);

#           /* Make sure the rounding is done correctly. */
#           if(fabs(dratio-ceil_ratio) < fabs(dratio-floor_ratio))
#           {
#             ratio = ceil_ratio;
#           }
#           else
#           {
#             ratio = floor_ratio;
#           }

#           /* Compute complex exponential in fine points between two coarse points */
#           /* This loop carry out the eq. 2.32 in arXiv:2001.10897 */
#           for(UINT4 kk = 1; kk < ratio && fine_count < iStop; kk++){
#             cexp_i_alpha[fine_count]   = Qalpha*cexp_i_alpha[fine_count-1];
#             cexp_i_epsilon[fine_count] = Qepsilon*cexp_i_epsilon[fine_count-1];
#             cexp_i_betah[fine_count]   = Qbetah*cexp_i_betah[fine_count-1];
#             fine_count++;
#           }
#         }// Loop over coarse grid

#         /*
#           Now we have the complex exponentials of the three Euler angles alpha, beta, epsilon evaluated in the fine frequency grid.
#           Next step is do the twisting up with these.
#         */

#         #if DEBUG == 1
#         printf("fine_count, htildelm->length, offset = %i %i %i\n", fine_count, htildelm->data->length, offset);
#         #endif


#         /************** TWISTING UP in the fine grid *****************/
#         for (UINT4 idx = 0; idx < fine_count; idx++)
#         {
#           double Mf   = pWF->M_sec * (idx + offset)*pWF->deltaF;

#           hlmcoprec   = htildelm->data->data[idx + offset];  /* Co-precessing waveform */

#           COMPLEX16 hplus       = 0.0;  /* h_+ */
#           COMPLEX16 hcross      = 0.0;  /* h_x */

#           pPrec->cexp_i_alpha   = cexp_i_alpha[idx];
#           pPrec->cexp_i_epsilon = cexp_i_epsilon[idx];
#           pPrec->cexp_i_betah   = cexp_i_betah[idx];

#            if(pPrec->precessing_tag==3) pPrec->gamma_in = 0.;

#           status = IMRPhenomXPHMTwistUp(Mf, hlmcoprec, pWF, pPrec, ell, emmprime, &hplus, &hcross);
#            XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "Call to IMRPhenomXPHMTwistUp failed.");

#           if(ell == 2 && emmprime == 2 && AntisymmetricWaveform && IMRPhenomXPNRUseTunedAngles)
#           {
#             COMPLEX16 hplus_antiSym       = 0.0;
#             COMPLEX16 hcross_antiSym      = 0.0;
#             hlmcoprec_antiSym = antiSym_amp->data[idx] * cexp(I*antiSym_phi->data[idx]);
#             pPrec->PolarizationSymmetry = -1.0;
#             IMRPhenomXPHMTwistUp(Mf, hlmcoprec_antiSym, pWF, pPrec, ell, emmprime, &hplus_antiSym, &hcross_antiSym);
#             pPrec->PolarizationSymmetry = 1.0;
#             hplus += hplus_antiSym;
#             hcross += hcross_antiSym;
#           }

#           (*hptilde)->data->data[idx + offset] += hplus ;
#           (*hctilde)->data->data[idx + offset] += hcross ;

#         }

#         XLALDestroyREAL8Sequence(coarseFreqs);
#         LALFree(valpha);
#         LALFree(vepsilon);
#         LALFree(vbetah);
#         LALFree(cexp_i_alpha);
#         LALFree(cexp_i_epsilon);
#         LALFree(cexp_i_betah);
#       }// End of Multibanding-specific.

#       XLALDestroyCOMPLEX16FrequencySeries(htildelm);
#     }//Loop over emmprime
#   }//Loop over ell

#   if (AntisymmetricWaveform && IMRPhenomXPNRUseTunedAngles)
#   {
#     XLALDestroyREAL8Sequence(antiSym_amp);
#     XLALDestroyREAL8Sequence(antiSym_phi);
#   }

#   if (IMRPhenomXPNRUseTunedAngles){
#     gsl_spline_free(hm_angle_spline->alpha_spline);
#     gsl_spline_free(hm_angle_spline->beta_spline);
#     gsl_spline_free(hm_angle_spline->gamma_spline);

#     gsl_interp_accel_free(hm_angle_spline->alpha_acc);
#     gsl_interp_accel_free(hm_angle_spline->beta_acc);
#     gsl_interp_accel_free(hm_angle_spline->gamma_acc);

#     LALFree(hm_angle_spline);
#   }

#   // Free memory used to hold non-precesing XHM struct
#   if (pWF->APPLY_PNR_DEVIATIONS && pWF->IMRPhenomXPNRForceXHMAlignment) {
#     // Cleaning up
#     LALFree(pPrec->pWF22AS);
#   }

#   XLALDestroySphHarmFrequencySeries(*hlms);
#   XLALFree(hlms);
#   /*
#       Loop over h+ and hx to rotate waveform by 2 \zeta.
#       See discussion in Appendix C: Frame Transformation and Polarization Basis.
#       The formula for \zeta is given by eq. C26.
#   */
#   if(fabs(pPrec->zeta_polarization) > 0.0)
#   {
#     COMPLEX16 PhPpolp, PhPpolc;
#     REAL8 cosPolFac, sinPolFac;

#     cosPolFac = cos(2.0 * pPrec->zeta_polarization);
#     sinPolFac = sin(2.0 * pPrec->zeta_polarization);

#     for (UINT4 i = offset; i < (*hptilde)->data->length; i++)
#     {
#       PhPpolp = (*hptilde)->data->data[i];
#       PhPpolc = (*hctilde)->data->data[i];

#       (*hptilde)->data->data[i] = cosPolFac * PhPpolp + sinPolFac * PhPpolc;
#       (*hctilde)->data->data[i] = cosPolFac * PhPpolc - sinPolFac * PhPpolp;
#     }
#   }

#   /* Free memory */
#   XLALDestroyCOMPLEX16FrequencySeries(htilde22);
#   XLALDestroyValue(ModeArray);
#   XLALDestroyREAL8Sequence(freqs);


#   if(pPrec->precessing_tag==3)
#   {
#   LALFree(pPrec->alpha_params);
#   LALFree(pPrec->beta_params);

#   gsl_spline_free(pPrec->alpha_spline);
#   gsl_spline_free(pPrec->cosbeta_spline);
#   gsl_spline_free(pPrec->gamma_spline);

#   gsl_interp_accel_free(pPrec->alpha_acc);
#   gsl_interp_accel_free(pPrec->gamma_acc);
#   gsl_interp_accel_free(pPrec->cosbeta_acc);

#   }


#   #if DEBUG == 1
#   printf("\n******Leaving IMRPhenomXPHM_hplushcross*****\n");
#   #endif

#   return XLAL_SUCCESS;
# }