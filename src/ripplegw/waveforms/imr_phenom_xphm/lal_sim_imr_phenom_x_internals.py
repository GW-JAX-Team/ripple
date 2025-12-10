"""Helper functions for IMRPhenomXPHM waveform model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw import ms_to_Mc_eta
from ripplegw.constants import MRSUN, PI, gt
from ripplegw.waveforms.imr_phenom_xphm.lal_constants import LAL_MSUN_SI
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXPhaseCoefficientsDataClass,
    IMRPhenomXUsefulPowersDataClass,
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_qnm import (
    evaluate_QNMfit_fdamp22,
    evaluate_QNMfit_fring22,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    imr_phenom_x_internal_nudge,
    xlal_sim_imr_phenom_x_chi_eff,
    xlal_sim_imr_phenom_x_chi_pn_hat,
    xlal_sim_imr_phenom_x_dchi,
    xlal_sim_imr_phenom_x_f_isco,
    xlal_sim_imr_phenom_x_f_meco,
    xlal_sim_imr_phenom_x_final_mass_2017,
    xlal_sim_imr_phenom_x_final_spin_2017,
    xlal_sim_imr_phenom_x_stot_r,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_mode_array_is_mode_active,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass
from ripplegw.waveforms.IMRPhenom_tidal_utils import get_kappa
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import _get_merger_frequency


@checkify.checkify
def imr_phenom_x_initialize_powers(number: float | jnp.ndarray) -> IMRPhenomXUsefulPowersDataClass:
    """Initialize various powers of the input number.

    Args:
        number: Input number (float or JAX array).

    Returns:
        IMRPhenomXUsefulPowersDataClass containing computed power values.
    """
    # Ensure number is a JAX array for consistent operations
    number = jnp.asarray(number)

    # Sanity check
    checkify.check(jnp.all(number >= 0), "Error: number must be non-negative.")

    # Compute sixth root and its reciprocal
    sixth = jnp.power(number, 1.0 / 6.0)
    m_sixth = 1.0 / sixth

    # Build the powers dataclass
    return IMRPhenomXUsefulPowersDataClass(
        seven_sixths=sixth * number,
        one_sixth=sixth,
        ten_thirds=((((sixth * sixth) ** 2) ** 2) * number) * number,
        eight_thirds=((((sixth * sixth) ** 2) ** 2) * number) * (sixth * sixth),
        seven_thirds=(((sixth * sixth) ** 2) ** 2) * number,
        five_thirds=(((sixth * sixth) ** 2) ** 2) * (sixth * sixth),
        four_thirds=((sixth * sixth) ** 2) ** 2,
        two_thirds=(sixth * sixth) ** 2,
        one_third=sixth * sixth,
        five=number * number * number * number * number,
        four=number * number * number * number,
        three=number * number * number,
        two=number * number,
        sqrt=sixth * sixth * sixth,  # Equivalent to sqrt(number)
        itself=number,
        m_sqrt=1.0 / (sixth * sixth * sixth),  # 1/sqrt(number)
        m_one=1.0 / number,
        m_two=1.0 / (number * number),
        m_three=1.0 / (number * number * number),
        m_four=1.0 / (number * number * number * number),
        m_five=1.0 / (number * number * number * number * number),
        m_six=m_sixth,
        m_one_third=1.0 / (sixth * sixth),
        m_two_thirds=1.0 / ((sixth * sixth) ** 2),
        m_four_thirds=1.0 / (((sixth * sixth) ** 2) ** 2),
        m_five_thirds=1.0 / ((((sixth * sixth) ** 2) ** 2) * (sixth * sixth)),
        m_seven_thirds=1.0 / ((((sixth * sixth) ** 2) ** 2) * number),
        m_eight_thirds=1.0 / (((((sixth * sixth) ** 2) ** 2) * number) * (sixth * sixth)),
        m_ten_thirds=1.0 / (((((sixth * sixth) ** 2) ** 2) * number) * number),
        m_one_sixth=m_sixth,
        m_seven_sixths=m_sixth / number,
        log=jnp.log(number),
    )


@checkify.checkify
def _validate_inspiral_phase_version(ins_phase_version: jnp.ndarray) -> bool:
    """Validate that the inspiral phase version is allowed."""
    allowed = jnp.array([104, 105, 114, 115])
    is_valid = jnp.any(allowed == ins_phase_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid inspiral phase version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_intermediate_phase_version(int_phase_version: jnp.ndarray) -> bool:
    """Validate that the intermediate phase version is allowed."""
    allowed = jnp.array([104, 105])
    is_valid = jnp.any(allowed == int_phase_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid intermediate phase version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_ringdown_phase_version(rd_phase_version: jnp.ndarray) -> bool:
    """Validate that the ringdown phase version is allowed."""
    allowed = jnp.array([105])
    is_valid = jnp.any(allowed == rd_phase_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid ringdown phase version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_inspiral_amplitude_version(ins_amp_version: jnp.ndarray) -> bool:
    """Validate that the inspiral amplitude version is allowed."""
    allowed = jnp.array([103])
    is_valid = jnp.any(allowed == ins_amp_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid inspiral amplitude version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_intermediate_amplitude_version(int_amp_version: jnp.ndarray) -> bool:
    """Validate that the intermediate amplitude version is allowed."""
    allowed = jnp.array([1043, 104, 105])
    is_valid = jnp.any(allowed == int_amp_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid intermediate amplitude version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_ringdown_amplitude_version(rd_amp_version: jnp.ndarray) -> bool:
    """Validate that the ringdown amplitude version is allowed."""
    allowed = jnp.array([103])
    is_valid = jnp.any(allowed == rd_amp_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid ringdown amplitude version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def imr_phenom_x_set_waveform_variables(  # pylint: disable=too-many-statements,too-many-positional-arguments,too-many-locals,too-many-arguments
    m1_si: float,
    m2_si: float,
    chi1l_in: float,
    chi2l_in: float,
    delta_f: float,
    f_ref: float,
    phi0: float,
    f_min: float,
    f_max: float,
    distance: float,
    inclination: float,
    lal_params: IMRPhenomXPHMParameterDataClass,
    powers_of_lalpi: IMRPhenomXUsefulPowersDataClass,
) -> IMRPhenomXWaveformDataClass:
    """Set up the IMRPhenomX waveform dataclass with given parameters.

    Args:
        m1_si: Mass 1 in SI units.
        m2_si: Mass 2 in SI units.
        chi1l_in: Aligned spin of mass 1.
        chi2l_in: Aligned spin of mass 2.
        delta_f: Frequency step size.
        f_ref: Reference frequency.
        phi0: Initial phase.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        distance: Distance to the source.
        inclination: Inclination angle.
        lal_params: Additional LAL parameters.

    Returns:
        IMRPhenomXWaveformDataClass with initialized waveform parameters.
    """

    # Validate the inspiral phase version
    _validate_inspiral_phase_version(lal_params.ins_phase_version)

    # Validate the intermediate phase version
    _validate_intermediate_phase_version(lal_params.int_phase_version)

    # Validate the ringdown phase version
    _validate_ringdown_phase_version(lal_params.rd_phase_version)

    # Validate the inspiral amplitude version
    _validate_inspiral_amplitude_version(lal_params.ins_amp_version)

    # Validate the intermediate amplitude version
    _validate_intermediate_amplitude_version(lal_params.int_amp_version)

    # Validate the ringdown amplitude version
    _validate_ringdown_amplitude_version(lal_params.rd_amp_version)

    imr_phenom_xpnr_use_tuned_coprec = lal_params.pnr_use_tuned_coprec
    # NOTE that the line below means that 33 tuning can only be on IFF 22 tuning is on
    imr_phenom_xpnr_use_tuned_coprec33 = lal_params.pnr_use_tuned_coprec33 * imr_phenom_xpnr_use_tuned_coprec

    # Rescale the mass in solar masses
    m1_in = m1_si / LAL_MSUN_SI
    m2_in = m2_si / LAL_MSUN_SI

    # Set matter parameters
    lambda1_in = 0.0
    lambda2_in = 0.0
    quad_param1_in = 1.0
    quad_param2_in = 1.0

    # def _tidal_branch(lal_params):
    #     lambda1_in = lal_params.lambda1
    #     lambda2_in = lal_params.lambda2
    #     checkify.check((lambda1_in >= 0.0) & (lambda2_in >= 0.0),
    #                   "Tidal deformabilities lambda1 and lambda2 must be non-negative.")
    #     lal_params = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(lal_params)
    #     quad_param1_in = 1.0 + lal_params.d_quad_mon1
    #     quad_param2_in = 1.0 + lal_params.d_quad_mon2
    #     return lambda1_in, lambda2_in, quad_param1_in, quad_param2_in, lal_params

    # def _no_tidal_branch(lal_params):
    #     return 0.0, 0.0, 1.0, 1.0, lal_params

    # lambda1_in, lambda2_in, quad_param1_in, quad_param2_in, lal_params = jax.lax.cond(
    #     lal_params.phen_x_tidal != 0,
    #     _tidal_branch,
    #     _no_tidal_branch,
    #     lal_params
    # )

    def _swap_parameters(operand):
        m1_in, m2_in, chi1l_in, chi2l_in, lambda1_in, lambda2_in, quad_param1_in, quad_param2_in = operand
        return chi2l_in, chi1l_in, m2_in, m1_in, lambda2_in, lambda1_in, quad_param2_in, quad_param1_in

    def _keep_parameters(operand):
        m1_in, m2_in, chi1l_in, chi2l_in, lambda1_in, lambda2_in, quad_param1_in, quad_param2_in = operand
        return chi1l_in, chi2l_in, m1_in, m2_in, lambda1_in, lambda2_in, quad_param1_in, quad_param2_in

    operand = (m1_in, m2_in, chi1l_in, chi2l_in, lambda1_in, lambda2_in, quad_param1_in, quad_param2_in)
    chi1l, chi2l, m1, m2, lambda1, lambda2, quad_param1, quad_param2 = jax.lax.cond(
        m1_in >= m2_in, _keep_parameters, _swap_parameters, operand
    )

    chi1l = jax.lax.cond(chi1l > 1.0, lambda x: imr_phenom_x_internal_nudge(x, 1.0, 1e-6), lambda x: x, chi1l)
    chi2l = jax.lax.cond(chi2l > 1.0, lambda x: imr_phenom_x_internal_nudge(x, 1.0, 1e-6), lambda x: x, chi2l)
    chi1l = jax.lax.cond(chi1l < -1.0, lambda x: imr_phenom_x_internal_nudge(x, -1.0, 1e-6), lambda x: x, chi1l)
    chi2l = jax.lax.cond(chi2l < -1.0, lambda x: imr_phenom_x_internal_nudge(x, -1.0, 1e-6), lambda x: x, chi2l)

    # If spins are still unphysical after checking for small round-off errors, fail.
    # if chi1l > 1.0 or chi1l < -1.0 or chi2l > 1.0 or chi2l < -1.0:
    checkify.check(
        (chi1l <= 1.0) & (chi1l >= -1.0) & (chi2l <= 1.0) & (chi2l >= -1.0),
        "Unphysical spins requested: must obey the Kerr bound [-1,1].",
    )

    # Symmetric mass ratio
    delta = jnp.abs((m1 - m2) / (m1 + m2))
    mc, eta = ms_to_Mc_eta((m1, m2))
    # eta = jnp.abs(0.25 * (1.0 - delta * delta))
    q = m1 / m2

    # eta = jnp.where(eta > 0.25, 0.25, eta)
    # if eta > 0.25 or eta < 0.0:
    checkify.check((eta <= 0.25) & (eta >= 0.0), "Unphysical mass ratio requested.")

    q = jax.lax.select(eta == 0.25, 1.0, q)

    # Check the mass ratio
    checkify.check(q <= 1000.0, "The model is not supported for mass ratios > 1000.")

    m_tot = m1 + m2
    m_sec = m_tot * gt
    eta2 = eta * eta
    kappa2_t = get_kappa(jnp.array([m1, m2, chi1l, chi2l, lambda1, lambda2]))
    f_merger = _get_merger_frequency(jnp.array([m1, m2, chi1l, chi2l, lambda1, lambda2]), kappa2_t)

    # /* Spin parameterisations */
    chi_eff = xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l)
    chi_pn_hat = xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l)
    s_tot_r = xlal_sim_imr_phenom_x_stot_r(eta, chi1l, chi2l)
    dchi = xlal_sim_imr_phenom_x_dchi(chi1l, chi2l)
    dchi_half = dchi * 0.5

    sigma_l = (chi2l * m2) - (chi1l * m1)  # // SigmaL = (M/m2)*(S2.L) - (M/m2)*(S1.L)
    sl = chi1l * (m1 * m1) + chi2l * (m2 * m2)  # // SL = S1.L + S2.L

    phi_ref_in = phi0
    # phi0 = phi0  # // Orbital phase at reference frequency (as passed from lalsimulation)
    beta = PI * 0.5 - phi0  # // Azimuthal angle of binary at reference frequency
    phi_f_ref = 0.0  # // This is calculated later

    # /* Geometric reference frequency */
    m_f_ref = m_sec * f_ref  # XLALSimIMRPhenomXUtilsHztoMf(fRef,Mtot)
    pi_m = PI * m_sec
    v_ref = (pi_m * f_ref) ** (1 / 3)

    delta_mf = m_sec * delta_f  # XLALSimIMRPhenomXUtilsHztoMf(deltaF,Mtot)

    # /* Define the default end of the waveform as: 0.3 Mf.
    #  This value is chosen such that the 44 mode also shows the ringdown part. */
    # // If chieff is very high the ringdown of the 44 is almost cut it out when using 0.3,
    # so we increase a little bit the cut of freq up 0.33.
    f_cut_def = jax.lax.select(
        chi_eff > 0.99,
        0.33,
        0.3,
    )

    # /* Minimum and maximum frequency */
    m_f_max = m_sec * f_max  # XLALSimIMRPhenomXUtilsHztoMf(fMax,Mtot)

    # /* Convert fCut to physical cut-off frequency */
    f_cut = f_cut_def / m_sec

    # /* Sanity check that minimum start frequency is less than cut-off frequency */
    # if (fCut <= fMin):
    #     jax.debug.print(f"(fCut = {fCut} Hz) <= f_min = {fMin}")
    checkify.check(f_cut > f_min, "Error: f_cut must be greater than f_min.")

    # if(debug)
    # {
    # 	printf("fRef : %.6f\n",fRef)
    # 	printf("phi0 : %.6f\n",phi0)
    # 	printf("fCut : %.6f\n",fCut)
    # 	printf("fMin : %.6f\n",fMin)
    # 	printf("fMax : %.6f\n",fMax)
    # }

    # /* By default f_max_prime is f_max. If fCut < fMax, then use fCut, i.e. waveform up to fMax will be zeros */
    f_max_prime = f_max
    # f_max_prime   = jax.lax.select( # fMax if fMax else fCut
    #     fMax,
    #     fMax,
    #     fCut
    # )
    f_max_prime = jax.lax.select(
        f_max_prime > f_cut, f_cut, f_max_prime
    )  # fCut if (f_max_prime > fCut) else f_max_prime

    # if f_max_prime <= fMin:
    #     jax.debug.print("f_max <= f_min")
    checkify.check(f_max_prime > f_min, "Error: f_max_prime must be greater than f_min.")

    # if(debug)
    # {
    # 	printf("fMin        = %.6f\n",fMin)
    # 	printf("fMax        = %.6f\n",fMax)
    # 	printf("f_max_prime = %.6f\n",f_max_prime)
    # }

    # /* Final Mass and Spin */
    # NOTE: These are only default values
    m_final = xlal_sim_imr_phenom_x_final_mass_2017(eta, chi1l, chi2l)
    afinal = xlal_sim_imr_phenom_x_final_spin_2017(eta, chi1l, chi2l)

    # /* (500) Set default values of physically specific final spin parameters for use with PNR/XCP */
    afinal_nonprec = afinal  # // NOTE: This is only a default value see LALSimIMRPhenomX_precession.c
    afinal_prec = afinal  # // NOTE: This is only a default value see LALSimIMRPhenomX_precession.c

    # /* Ringdown and damping frequency of final BH */
    f_ring = evaluate_QNMfit_fring22(afinal) / (m_final)
    f_damp = evaluate_QNMfit_fdamp22(afinal) / (m_final)

    # if(debug)
    # {
    # 	printf("Mf  = %.6f\n",Mfinal)
    # 	printf("af  = %.6f\n",afinal)
    # 	printf("frd = %.6f\n",fRING)
    # 	printf("fda = %.6f\n",fDAMP)
    # }

    # if Mfinal > 1.0:
    #     jax.debug.print("IMRPhenomX_FinalMass2018: Final mass > 1.0 not physical.")
    checkify.check(m_final <= 1.0, "IMRPhenomX_FinalMass2018: Final mass > 1.0 not physical.")
    # if abs(afinal) > 1.0:
    #     jax.debug.print("IMRPhenomX_FinalSpin2018: Final spin > 1.0 is not physical.")
    checkify.check(abs(afinal) <= 1.0, "IMRPhenomX_FinalSpin2018: Final spin > 1.0 is not physical.")

    # /* Fit to the hybrid minimum energy circular orbit (MECO), Cabero et al, Phys.Rev. D95 (2017) */
    f_meco = xlal_sim_imr_phenom_x_f_meco(eta, chi1l, chi2l)

    # /* Innermost stable circular orbit (ISCO), e.g. Ori et al, Phys.Rev. D62 (2000) 124022 */
    f_isco = xlal_sim_imr_phenom_x_f_isco(afinal)

    # if(debug)
    # {
    #     printf("fMECO = %.6f\n",fMECO)
    #     printf("fISCO = %.6f\n",fISCO)
    # }

    # if(fMECO > fISCO):
    #     # /* If MECO > fISCO, throw an error - this may be the case
    #       for very corner cases in the parameter space (e.g. q ~1000, chi ~ 0.999) */
    #     jax.debug.print("Error: f_MECO cannot be greater than f_ISCO.")
    checkify.check(f_meco <= f_isco, "Error: f_MECO cannot be greater than f_ISCO.")

    # /* Distance and inclination */
    # distance = distance
    # inclination = inclination

    # /* Amplitude normalization */
    amp0 = m_tot * MRSUN * m_tot * gt / distance
    amp_norm = jnp.sqrt(2.0 / 3.0) * jnp.sqrt(eta) * powers_of_lalpi.m_one_sixth

    # if(debug)
    # {
    #     printf("\n\neta     = %.6f\n",eta)
    #     printf("\nampNorm = %e\n",ampNorm)
    #     printf("amp0 : %e",amp0)
    # }

    # if(debug)
    # {
    #     printf("\n\n **** Sanity checks complete. Waveform struct has been initialized. **** \n\n")
    # }

    dphase0 = 5.0 / (128.0 * PI ** (5.0 / 3.0))

    # /* Set nonprecessing value of select precession quantities (PNRUseTunedCoprec)*/
    chi_tot_perp = 0.0
    chi_p = 0.0
    theta_ls = 0.0
    a1 = 0.0
    pnr_dev_parameter = 0.0
    pnr_single_spin = 0
    mu1 = 0
    mu2 = 0
    mu3 = 0
    mu4 = 0
    nu0 = 0
    nu4 = 0
    nu5 = 0
    nu6 = 0
    zeta1 = 0
    zeta2 = 0
    f_ring_eff_shift_divided_by_emm = 0

    f_inspiral_align = 0.0
    xas_dphase_at_f_inspiral_align = 0.0
    xas_phase_at_f_inspiral_align = 0.0
    xhm_dphase_at_f_inspiral_align = 0.0
    xhm_phase_at_f_inspiral_align = 0.0

    beta_rd = 0.0
    f_ring22_prec = 0.0
    f_ring_cp = 0.0
    pnr_window = 0.0

    apply_pnr_deviations = 0

    ## Comes from IMRPhenomXGetAndSetPrecessionVariables function in LAL ##
    m_tot_si = m1_si + m2_si  # Total mass in SI units:        m1_SI + m2_SI
    m1_normalised = m1_si / m_tot_si  # Normalized mass of larger companion:   m1_SI / Mtot_SI
    m2_normalised = m2_si / m_tot_si  # Normalized mass of smaller companion:  m2_SI / Mtot_SI
    # Total mass in solar units -> I believe this LAL comment is incorrect,
    m = m1_normalised + m2_normalised
    # but I'm keeping it for reference

    m1_2 = m1_normalised * m1_normalised
    m2_2 = m2_normalised * m2_normalised
    ##############################################################################

    waveform_dataclass = IMRPhenomXWaveformDataClass(
        imr_phenom_x_inspiral_phase_version=lal_params.ins_phase_version,
        imr_phenom_x_intermediate_phase_version=lal_params.int_phase_version,
        imr_phenom_x_ringdown_phase_version=lal_params.rd_phase_version,
        imr_phenom_x_inspiral_amp_version=lal_params.ins_amp_version,
        imr_phenom_x_intermediate_amp_version=lal_params.int_amp_version,
        imr_phenom_x_ringdown_amp_version=lal_params.rd_amp_version,
        imr_phenom_xpnr_use_tuned_coprec=imr_phenom_xpnr_use_tuned_coprec,
        imr_phenom_xpnr_use_tuned_coprec_33=imr_phenom_xpnr_use_tuned_coprec33,
        imr_phenom_xpnr_force_xhm_alignment=lal_params.pnr_force_xhm_alignment,
        imr_phenom_x_return_co_prec=False,
        phenom_x_only_return_phase=lal_params.phenom_x_only_return_phase,
        m1_si=m1 * LAL_MSUN_SI,
        m2_si=m2 * LAL_MSUN_SI,
        q=q,
        eta=eta,
        mc=mc,
        m_tot_si=m_tot_si,
        m_tot=m_tot,
        m1=m1 / m_tot,
        m2=m2 / m_tot,
        m_sec=m_sec,
        delta=delta,
        eta2=eta2,
        eta3=eta * eta2,
        chi1l=chi1l,
        chi2l=chi2l,
        chi1l2l=chi1l * chi2l,
        chi1l2=chi1l * chi1l,
        chi1l3=chi1l * chi1l * chi1l,
        chi2l2=chi2l * chi2l,
        chi2l3=chi2l * chi2l * chi2l,
        chi_eff=chi_eff,
        chi_pn_hat=chi_pn_hat,
        chi_tot_perp=chi_tot_perp,
        chi_p=chi_p,
        theta_ls=theta_ls,
        a1=a1,
        mu1=mu1,
        mu2=mu2,
        mu3=mu3,
        mu4=mu4,
        nu0=nu0,
        nu4=nu4,
        nu5=nu5,
        nu6=nu6,
        zeta1=zeta1,
        zeta2=zeta2,
        pnr_dev_parameter=pnr_dev_parameter,
        pnr_window=pnr_window,
        apply_pnr_deviations=apply_pnr_deviations,
        pnr_single_spin=pnr_single_spin,
        f_ring_eff_shift_divided_by_emm=f_ring_eff_shift_divided_by_emm,
        s_tot_r=s_tot_r,
        dchi=dchi,
        dchi_half=dchi_half,
        sl=sl,
        sigma_l=sigma_l,
        lambda1=lambda1,
        lambda2=lambda2,
        quad_param1=quad_param1,
        quad_param2=quad_param2,
        kappa2_t=kappa2_t,
        f_merger=f_merger,
        f_ref=f_ref,
        phi_ref_in=phi_ref_in,
        phi0=phi0,
        beta=beta,
        v_ref=v_ref,
        delta_f=delta_f,
        f_min=f_min,
        f_max=f_max,
        f_max_prime=f_max_prime,
        f_cut=f_cut,
        m_final=m_final,
        a_final=afinal,
        a_final_prec=afinal_prec,
        f_ring=f_ring,
        f_damp=f_damp,
        distance=distance,
        inclination=inclination,
        amp0=amp0,
        amp_norm=amp_norm,
        dphase0=dphase0,
        eta4=eta2 * eta2,
        f_meco=f_meco,
        f_isco=f_isco,
        beta_rd=beta_rd,
        f_ring22_prec=f_ring22_prec,
        f_ring_cp=f_ring_cp,
        f_inspiral_align=f_inspiral_align,
        xas_dphase_at_f_inspiral_align=xas_dphase_at_f_inspiral_align,
        xas_phase_at_f_inspiral_align=xas_phase_at_f_inspiral_align,
        xhm_dphase_at_f_inspiral_align=xhm_dphase_at_f_inspiral_align,
        xhm_phase_at_f_inspiral_align=xhm_phase_at_f_inspiral_align,
        f_ring21=0.0,
        f_damp21=0.0,
        f_ring32=0.0,
        f_damp32=0.0,
        f_ring33=0.0,
        f_damp33=0.0,
        f_ring44=0.0,
        f_damp44=0.0,
        m_f_max=m_f_max,
        delta_mf=delta_mf,
        f_cut_def=f_cut_def,
        m_f_ref=m_f_ref,
        phi_f_ref=phi_f_ref,
        pi_m=pi_m,
        e_rad=0.0,
        a_final_non_prec=afinal_nonprec,
        lal_params=lal_params,
        m=m,
        m1_2=m1_2,
        m2_2=m2_2,
    )
    return waveform_dataclass


def _generate_valid_modes(max_l: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute valid (ell, emm) pairs statically.

    Args:
        max_l: Maximum ell value.

    Returns:
        Tuple of JAX arrays (ell_array, emm_array) containing valid modes.
    """
    ell_list = []
    emm_list = []
    for ell in range(2, max_l + 1):
        for emm in range(0, ell + 1):
            ell_list.append(ell)
            emm_list.append(emm)
    return jnp.array(ell_list, dtype=int), jnp.array(emm_list, dtype=int)


@checkify.checkify
def _check_input_mode_array_impl(
    mode_array: jnp.ndarray, ell_valid: jnp.ndarray, emm_valid: jnp.ndarray, max_l: int
) -> bool:
    """Internal implementation of mode array checking.

    Args:
        mode_array: JAX boolean array indicating active modes.
        ell_valid: JAX array of valid ell values to check.
        emm_valid: JAX array of valid emm values to check.
        max_l: Maximum ell value for bounds checking.

    Returns:
        True if all active modes are allowed, raises error otherwise.
    """
    # Allowed modes: (l, |m|) pairs
    allowed_l = jnp.array([2, 2, 3, 3, 4])
    allowed_m = jnp.array([2, 1, 3, 2, 4])

    # Check if each valid mode is active (positive or negative m)
    active_pos = jax.vmap(
        lambda ell, emm: xlal_sim_inspiral_mode_array_is_mode_active(modes=mode_array, ell=ell, m=emm, max_l=max_l)
    )(ell_valid, emm_valid)
    active_neg = jax.vmap(
        lambda ell, emm: xlal_sim_inspiral_mode_array_is_mode_active(modes=mode_array, ell=ell, m=-emm, max_l=max_l)
    )(ell_valid, emm_valid)
    active = active_pos | active_neg

    # For active modes, check if they are allowed
    def is_allowed(ell, emm):
        # Check if (ell, emm) is in allowed
        matches = (allowed_l == ell) & (allowed_m == emm)
        return jnp.any(matches)

    allowed_mask = jax.vmap(is_allowed)(ell_valid, emm_valid)

    # Invalid if active but not allowed
    invalid = active & ~allowed_mask

    # Check that no invalid modes are active
    checkify.check(
        ~jnp.any(invalid),
        "Invalid modes activated in mode_array. Only the modes (2,2), (2,1), (3,3), (3,2), and (4,4) are supported.",
    )

    # Always check a trivial condition to ensure err is properly initialized
    checkify.check(True, "")

    return True


def check_input_mode_array(lal_params: IMRPhenomXPHMParameterDataClass, max_l: int = 8) -> tuple[checkify.Error, bool]:
    """Check if mode_array in lal_params contains only allowed modes.

    Args:
        lal_params: Dictionary that may contain a 'mode_array' key.
        max_l: Maximum angular momentum number to check (must be concrete/static).

    Returns:
        Tuple of (error, result) from checkify.
    """
    # Precompute valid modes statically before any JAX operations
    ell_valid, emm_valid = _generate_valid_modes(max_l)

    # Check if mode_array is present
    has_mode_array = lal_params.mode_array is not None

    def check_with_mode_array():
        """Check modes when mode_array is present."""
        mode_array = lal_params.mode_array
        return _check_input_mode_array_impl(mode_array, ell_valid, emm_valid, max_l)

    def check_without_mode_array():
        """Skip check when mode_array is not present."""
        return checkify.checkify(lambda: True)()

    # Use Python if since this is before JIT transforms the whole function
    # (check_input_mode_array itself is not transformed, only called by JIT)
    if has_mode_array:
        return check_with_mode_array()
    return check_without_mode_array()


def imr_phenom_x_get_phase_coefficients(
    p_wf: IMRPhenomXWaveformDataClass, p_phase: IMRPhenomXPhaseCoefficientsDataClass
) -> tuple[IMRPhenomXWaveformDataClass, IMRPhenomXPhaseCoefficientsDataClass]:

    # /* Get LALparams */
    # LALDict *LALparams    = pWF->LALparams;
    # const INT4 debug      = PHENOMXDEBUG;

    # /* GSL objects for solving system of equations via LU decomposition */
    # gsl_vector *b, *x;
    # gsl_matrix *A;
    # gsl_permutation *p;
    # int s;

    # REAL8 deltax;
    # REAL8 xmin;
    # REAL8 fi;

    # REAL8 gpoints4[4]     = {0.0, 1.0/4.0, 3.0/4.0, 1.0};
    # REAL8 gpoints5[5]     = {0.0, 1.0/2 - 1.0/(2*sqrt(2.0)), 1.0/2.0, 1.0/2 + 1.0/(2.0*sqrt(2.0)), 1.0};

    # // Matching regions

    # /* This is Eq. 5.11 in the paper */
    # double fIMmatch = 0.6 * (0.5 * pWF->fRING + pWF->fISCO);

    # /* This is the MECO frequency */
    # double fINmatch = pWF->fMECO;

    # /* This is Eq. 5.10 in the paper */
    # double deltaf   = (fIMmatch - fINmatch) * 0.03;

    # // Transition frequency is just below the MECO frequency and just above the RD fitting region

    # /* These are defined in Eq. 7.7 and the text just below, f_H = fPhaseMatchIM and f_L = fPhaseMatchIN */
    # pPhase->fPhaseMatchIN  = fINmatch - 1.0*deltaf;
    # pPhase->fPhaseMatchIM  = fIMmatch + 0.5*deltaf;

    # /* Defined in Eq. 7.4, this is f_L */
    # pPhase->fPhaseInsMin  = 0.0026;

    # /* Defined in Eq. 7.4, this is f_H */
    # pPhase->fPhaseInsMax  = 1.020 * pWF->fMECO;

    # /* Defined in Eq. 7.12, this is f_L */
    # pPhase->fPhaseRDMin   = fIMmatch;

    # /* Defined in Eq. 7.12, this is f_L */
    # pPhase->fPhaseRDMax   = pWF->fRING + 1.25*pWF->fDAMP;

    # pPhase->phiNorm    		= -(3.0 * powers_of_lalpi.m_five_thirds) / (128.0);

    # /* For convenience, define some variables here */
    # REAL8 chi1L           = pWF->chi1L;
    # REAL8 chi2L           = pWF->chi2L;

    # REAL8 chi1L2L         = chi1L * chi2L;

    # REAL8 chi1L2          = pWF->chi1L * pWF->chi1L;
    # REAL8 chi1L3          = pWF->chi1L * chi1L2;

    # REAL8 chi2L2          = pWF->chi2L * pWF->chi2L;
    # REAL8 chi2L3          = pWF->chi2L * chi2L2;

    # REAL8 eta             = pWF->eta;
    # REAL8 eta2            = eta*eta;
    # REAL8 eta3            = eta*eta2;

    # REAL8 delta           = pWF->delta;

    # /* Pre-initialize all phenomenological coefficients */
    # pPhase->a0 = 0.0;
    # pPhase->a1 = 0.0;
    # pPhase->a2 = 0.0;
    # pPhase->a3 = 0.0;
    # pPhase->a4 = 0.0;

    # pPhase->b0 = 0.0;
    # pPhase->b1 = 0.0;
    # pPhase->b2 = 0.0;
    # pPhase->b3 = 0.0;
    # pPhase->b4 = 0.0;

    # pPhase->c0 = 0.0;
    # pPhase->c1 = 0.0;
    # pPhase->c2 = 0.0;
    # pPhase->c3 = 0.0;
    # pPhase->c4 = 0.0;
    # pPhase->cL = 0.0;

    # pPhase->c2PN_tidal   = 0.;
    # pPhase->c3PN_tidal   = 0.;
    # pPhase->c3p5PN_tidal = 0.;

    # pPhase->sigma0 = 0.0;
    # pPhase->sigma1 = 0.0;
    # pPhase->sigma2 = 0.0;
    # pPhase->sigma3 = 0.0;
    # pPhase->sigma4 = 0.0;
    # pPhase->sigma5 = 0.0;

    # /*
    # 	The general strategy is to initialize a linear system of equations:

    # 	A.x = b

    # 	- A is a matrix with the coefficients of the ansatz evaluated at the collocation nodes.
    # 	- b is a vector of the value of the collocation points
    # 	- x is the solution vector (i.e. the coefficients) that we must solve for.

    # 	We choose to do this using a standard LU decomposition.
    # */

    # /* Generate list of collocation points */
    # /*
    # The Gauss-Chebyshev Points are given by:
    # GCPoints[n] = Table[Cos[i * pi/n] , {i, 0, n}]

    # SamplePoints[xmin,xmax,n] = {
    # 	pWF->delta = xmax - xmin;
    # 	gpoints = 0.5 * (1 + GCPoints[n-1]);

    # 	return {xmin + pWF->delta * gpoints}
    # }

    # gpoints4 = [1.0, 3.0/4, 1.0/4, 0.0];
    # gpoints5 = [1.0, 1.0/2 + 1.0/(2.0*sqrt(2.0)), 1.0/2, 1.0/2 - 1.0/(2*sqrt(2.0)), 0.]

    # */

    # /*
    # Ringdown phase collocation points:
    # Here we only use the first N+1 points in the array where N = the
    # number of pseudo PN terms.

    # The size of the array is controlled by: N_MAX_COLLOCATION_POINTS_PHASE_RD

    # Default is to use 5 collocation points.
    # */
    # deltax = pPhase->fPhaseRDMax - pPhase->fPhaseRDMin;
    # xmin   = pPhase->fPhaseRDMin;
    # int i;

    # double phaseRD; // This is used in intermediate phase reconstruction.

    # if(debug)
    # {
    # 	printf("\n");
    # 	printf("Solving system of equations for RD phase...\n");
    # }

    # // Initialize collocation points
    # for(i = 0; i < 5; i++)
    # {
    # 	pPhase->CollocationPointsPhaseRD[i] = gpoints5[i] * deltax + xmin;
    # }
    # // Collocation point 4 is set to the ringdown frequency ~ dip in Lorentzian
    # pPhase->CollocationPointsPhaseRD[3] = pWF->fRING;

    # if(debug)
    # {
    # 	printf("Rigndown collocation points : \n");
    # 	printf("F1 : %.6f\n",pPhase->CollocationPointsPhaseRD[0]);
    # 	printf("F2 : %.6f\n",pPhase->CollocationPointsPhaseRD[1]);
    # 	printf("F3 : %.6f\n",pPhase->CollocationPointsPhaseRD[2]);
    # 	printf("F4 : %.6f\n",pPhase->CollocationPointsPhaseRD[3]);
    # 	printf("F5 : %.6f\n",pPhase->CollocationPointsPhaseRD[4]);
    # }

    # switch(pWF->IMRPhenomXRingdownPhaseVersion)
    # {
    # 	case 105:
    # 	{
    # 		pPhase->NCollocationPointsRD = 5;
    # 		break;
    # 	}
    # 	default:
    # 	{
    # 		XLAL_ERROR(XLAL_EINVAL, "Error: IMRPhenomXRingdownPhaseVersion is not valid.\n");
    # 	}
    # }

    # if(debug)
    # {
    # 	printf("NCollRD = %d\n",pPhase->NCollocationPointsRD);
    # }

    # // Eq. 7.13 in arXiv:2001.11412
    # double RDv4 = IMRPhenomX_Ringdown_Phase_22_v4(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXRingdownPhaseVersion);

    # /* These are the calibrated collocation points, as per Eq. 7.13 */
    # pPhase->CollocationValuesPhaseRD[0] = IMRPhenomX_Ringdown_Phase_22_d12(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXRingdownPhaseVersion);
    # pPhase->CollocationValuesPhaseRD[1] = IMRPhenomX_Ringdown_Phase_22_d24(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXRingdownPhaseVersion);
    # pPhase->CollocationValuesPhaseRD[2] = IMRPhenomX_Ringdown_Phase_22_d34( pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXRingdownPhaseVersion);
    # pPhase->CollocationValuesPhaseRD[3] = RDv4;
    # pPhase->CollocationValuesPhaseRD[4] = IMRPhenomX_Ringdown_Phase_22_d54(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXRingdownPhaseVersion);

    # /* v_j = d_{j4} + v4 */
    # pPhase->CollocationValuesPhaseRD[4] = pPhase->CollocationValuesPhaseRD[4] + pPhase->CollocationValuesPhaseRD[3]; // v5 = d54  + v4
    # pPhase->CollocationValuesPhaseRD[2] = pPhase->CollocationValuesPhaseRD[2] + pPhase->CollocationValuesPhaseRD[3]; // v3 = d34  + v4
    # pPhase->CollocationValuesPhaseRD[1] = pPhase->CollocationValuesPhaseRD[1] + pPhase->CollocationValuesPhaseRD[3]; // v2 = d24  + v4
    # pPhase->CollocationValuesPhaseRD[0] = pPhase->CollocationValuesPhaseRD[0] + pPhase->CollocationValuesPhaseRD[1]; // v1 = d12  + v2

    # // Debugging information. Leave for convenience later on.
    # if(debug)
    # {
    # 	printf("\n");
    # 	printf("Ringdown Collocation Points: \n");
    # 	printf("v1 : %.6f\n",pPhase->CollocationValuesPhaseRD[0]);
    # 	printf("v2 : %.6f\n",pPhase->CollocationValuesPhaseRD[1]);
    # 	printf("v3 : %.6f\n",pPhase->CollocationValuesPhaseRD[2]);
    # 	printf("v4 : %.6f\n",pPhase->CollocationValuesPhaseRD[3]);
    # 	printf("v5 : %.6f\n",pPhase->CollocationValuesPhaseRD[4]);
    # 	printf("\n");
    # }

    # phaseRD = pPhase->CollocationValuesPhaseRD[0];

    # p = gsl_permutation_alloc(pPhase->NCollocationPointsRD);
    # b = gsl_vector_alloc(pPhase->NCollocationPointsRD);
    # x = gsl_vector_alloc(pPhase->NCollocationPointsRD);
    # A = gsl_matrix_alloc(pPhase->NCollocationPointsRD,pPhase->NCollocationPointsRD);

    # /*
    # Populate the b vector
    # */
    # gsl_vector_set(b,0,pPhase->CollocationValuesPhaseRD[0]);
    # gsl_vector_set(b,1,pPhase->CollocationValuesPhaseRD[1]);
    # gsl_vector_set(b,2,pPhase->CollocationValuesPhaseRD[2]);
    # gsl_vector_set(b,3,pPhase->CollocationValuesPhaseRD[3]);
    # gsl_vector_set(b,4,pPhase->CollocationValuesPhaseRD[4]);

    # /*
    # 		Eq. 7.12 in arXiv:2001.11412

    # 		ansatzRD(f) = a_0 + a_1 f^(-1/3) + a_2 f^(-2) + a_3 f^(-3) + a_4 f^(-4) + ( aRD ) / ( (f_damp^2 + (f - f_ring)^2 ) )

    # 		Canonical ansatz sets a_3 to 0.
    # */

    # /*
    # 	We now set up and solve a linear system of equations.
    # 	First we populate the matrix A_{ij}
    # */

    # /* Note that ff0 is always 1 */
    # REAL8 ff, invff, ff0, ff1, ff2, ff3, ff4;

    # /* A_{0,i} */
    # ff    = pPhase->CollocationPointsPhaseRD[0];
    # invff = 1.0 / ff;
    # ff1   = cbrt(invff);   // f^{-1/3}
    # ff2   = invff * invff; // f^{-2}
    # ff3   = ff2 * ff2;		 // f^{-4}
    # ff4   = -(pWF->dphase0) / (pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # gsl_matrix_set(A,0,0,1.0); // Constant
    # gsl_matrix_set(A,0,1,ff1); // f^(-1/3) term
    # gsl_matrix_set(A,0,2,ff2); // f^(-2) term
    # gsl_matrix_set(A,0,3,ff3); // f^(-4) term
    # gsl_matrix_set(A,0,4,ff4); // Lorentzian term

    # if(debug)
    # {
    # 	printf("For row 0: a0 + a1 %.6f + a2 %.6f + a4 %.6f + aRD %.6f\n",ff1,ff2,ff3,ff4);
    # }

    # /* A_{1,i} */
    # ff    = pPhase->CollocationPointsPhaseRD[1];
    # invff = 1.0 / ff;
    # ff1   = cbrt(invff);
    # ff2   = invff * invff;
    # ff3   = ff2 * ff2;
    # ff4   = -(pWF->dphase0) / (pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # gsl_matrix_set(A,1,0,1.0);
    # gsl_matrix_set(A,1,1,ff1);
    # gsl_matrix_set(A,1,2,ff2);
    # gsl_matrix_set(A,1,3,ff3);
    # gsl_matrix_set(A,1,4,ff4);

    # /* A_{2,i} */
    # ff    =  pPhase->CollocationPointsPhaseRD[2];
    # invff = 1.0 / ff;
    # ff1   = cbrt(invff);
    # ff2   = invff * invff;
    # ff3   = ff2 * ff2;
    # ff4   = -(pWF->dphase0) / (pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # gsl_matrix_set(A,2,0,1.0);
    # gsl_matrix_set(A,2,1,ff1);
    # gsl_matrix_set(A,2,2,ff2);
    # gsl_matrix_set(A,2,3,ff3);
    # gsl_matrix_set(A,2,4,ff4);

    # /* A_{3,i} */
    # ff    = pPhase->CollocationPointsPhaseRD[3];
    # invff = 1.0 / ff;
    # ff1   = cbrt(invff);
    # ff2   = invff * invff;
    # ff3   = ff2 * ff2;
    # ff4   = -(pWF->dphase0) / (pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # gsl_matrix_set(A,3,0,1.0);
    # gsl_matrix_set(A,3,1,ff1);
    # gsl_matrix_set(A,3,2,ff2);
    # gsl_matrix_set(A,3,3,ff3);
    # gsl_matrix_set(A,3,4,ff4);

    # /* A_{4,i} */
    # ff    = pPhase->CollocationPointsPhaseRD[4];
    # invff = 1.0 / ff;
    # ff1   = cbrt(invff);
    # ff2   = invff * invff;
    # ff3   = ff2 * ff2;
    # ff4   = -(pWF->dphase0) / (pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # gsl_matrix_set(A,4,0,1.0);
    # gsl_matrix_set(A,4,1,ff1);
    # gsl_matrix_set(A,4,2,ff2);
    # gsl_matrix_set(A,4,3,ff3);
    # gsl_matrix_set(A,4,4,ff4);

    # /* We now solve the system A x = b via an LU decomposition */
    # gsl_linalg_LU_decomp(A,p,&s);
    # gsl_linalg_LU_solve(A,p,b,x);

    # pPhase->c0  = gsl_vector_get(x,0); // x[0]; 	// a0
    # pPhase->c1  = gsl_vector_get(x,1); // x[1];		// a1
    # pPhase->c2  = gsl_vector_get(x,2); // x[2]; 	// a2
    # pPhase->c4  = gsl_vector_get(x,3); // x[3]; 	// a4
    # pPhase->cRD = gsl_vector_get(x,4);
    # pPhase->cL  = -(pWF->dphase0 * pPhase->cRD); // ~ x[4] // cL = - a_{RD} * dphase0

    # /* Apply NR tuning for precessing cases (500) */
    # pPhase->cL = pPhase->cL + ( pWF->PNR_DEV_PARAMETER * pWF->NU4 );
    # // pPhase->c0 = pPhase->c0 + ( pWF->PNR_DEV_PARAMETER * pWF->NU0 );

    # if(debug)
    # {
    # 	printf("\n");
    # 	printf("Ringdown Coefficients: \n");
    # 	printf("c0  : %.6f\n",pPhase->c0);
    # 	printf("c1  : %.6f\n",pPhase->c1);
    # 	printf("c2  : %.6f\n",pPhase->c2);
    # 	printf("c4  : %e\n",pPhase->c4);
    # 	printf("cRD : %.6f\n",gsl_vector_get(x,4));
    # 	printf("d0  : %.6f\n",pWF->dphase0);
    # 	printf("cL  : %e\n",pPhase->cL);
    # 	printf("\n");

    # 	printf("Freeing arrays...\n");
    # }

    # /* Tidy up in preparation for next GSL solve ... */
    # gsl_vector_free(b);
    # gsl_vector_free(x);
    # gsl_matrix_free(A);
    # gsl_permutation_free(p);

    # /*
    # Inspiral phase collocation points:
    # Here we only use the first N+1 points in the array where N = the
    # number of pseudo PN terms. E.g. for 4 pseudo-PN terms, we will
    # need 4 collocation points. An ansatz with n free coefficients
    # needs n pieces of information in order to constrain the ansatz.

    # The size of the array is controlled by: N_MAX_COLLOCATION_POINTS_PHASE_INS

    # Default is to use 4 pseudo-PN coefficients and hence 4 collocation points.

    # GC points as per Eq. 7.4 and 7.5, where f_L = pPhase->fPhaseInsMin and f_H = pPhase->fPhaseInsMax
    # */
    # deltax      = pPhase->fPhaseInsMax - pPhase->fPhaseInsMin;
    # xmin        = pPhase->fPhaseInsMin;

    # /*
    # 		Set number of pseudo-PN coefficients:
    # 			- If you add a new PN inspiral approximant, update with new version here.
    # */
    # switch(pWF->IMRPhenomXInspiralPhaseVersion)
    # {
    # 	case 104:
    # 	{
    # 		pPhase->NPseudoPN = 4;
    # 		pPhase->NCollocationPointsPhaseIns = 4;
    # 		break;
    # 	}
    # 	case 105:
    # 	{
    # 		pPhase->NPseudoPN = 5;
    # 		pPhase->NCollocationPointsPhaseIns = 5;
    # 		break;
    # 	}
    # 	case 114:
    # 	{
    # 		pPhase->NPseudoPN = 4;
    # 		pPhase->NCollocationPointsPhaseIns = 4;
    # 		break;
    # 	}
    # 	case 115:
    # 	{
    # 		pPhase->NPseudoPN = 5;
    # 		pPhase->NCollocationPointsPhaseIns = 5;
    # 		break;
    # 	}
    # 	default:
    # 	{
    # 		XLAL_ERROR_REAL8(XLAL_EINVAL, "Error: IMRPhenomXInspiralPhaseVersion is not valid.\n");
    # 	}
    # }

    # if(debug)
    # {
    # 	printf("\n");
    # 	printf("NPseudoPN : %d\n",pPhase->NPseudoPN);
    # 	printf("NColl : %d\n",pPhase->NCollocationPointsPhaseIns);
    # 	printf("\n");
    # }

    # p = gsl_permutation_alloc(pPhase->NCollocationPointsPhaseIns);
    # b = gsl_vector_alloc(pPhase->NCollocationPointsPhaseIns);
    # x = gsl_vector_alloc(pPhase->NCollocationPointsPhaseIns);
    # A = gsl_matrix_alloc(pPhase->NCollocationPointsPhaseIns,pPhase->NCollocationPointsPhaseIns);

    # /*
    # If we are using 4 pseudo-PN coefficients, call the routines below.
    # The inspiral phase version is still passed to the individual functions.
    # */
    # if(pPhase->NPseudoPN == 4)
    # {
    # 	// By default all models implemented use the following GC points.
    # 	// If a new model is calibrated with different choice of collocation points, edit this.
    # 	for(i = 0; i < pPhase->NCollocationPointsPhaseIns; i++)
    # 	{
    # 		fi = gpoints4[i] * deltax + xmin;
    # 		pPhase->CollocationPointsPhaseIns[i] = fi;
    # 	}

    # 	// Calculate the value of the differences between the ith and 3rd collocation points at the GC nodes
    # 	pPhase->CollocationValuesPhaseIns[0] = IMRPhenomX_Inspiral_Phase_22_d13(pWF->eta,pWF->chiPNHat,pWF->dchi,pWF->delta,pWF->IMRPhenomXInspiralPhaseVersion);
    # 	pPhase->CollocationValuesPhaseIns[1] = IMRPhenomX_Inspiral_Phase_22_d23(pWF->eta,pWF->chiPNHat,pWF->dchi,pWF->delta,pWF->IMRPhenomXInspiralPhaseVersion);
    # 	pPhase->CollocationValuesPhaseIns[2] = IMRPhenomX_Inspiral_Phase_22_v3( pWF->eta,pWF->chiPNHat,pWF->dchi,pWF->delta,pWF->IMRPhenomXInspiralPhaseVersion);
    # 	pPhase->CollocationValuesPhaseIns[3] = IMRPhenomX_Inspiral_Phase_22_d43(pWF->eta,pWF->chiPNHat,pWF->dchi,pWF->delta,pWF->IMRPhenomXInspiralPhaseVersion);

    # 	// Calculate the value of the collocation points at GC nodes via: v_i = d_i3 + v3
    # 	pPhase->CollocationValuesPhaseIns[0] = pPhase->CollocationValuesPhaseIns[0] + pPhase->CollocationValuesPhaseIns[2];
    # 	pPhase->CollocationValuesPhaseIns[1] = pPhase->CollocationValuesPhaseIns[1] + pPhase->CollocationValuesPhaseIns[2];
    # 	pPhase->CollocationValuesPhaseIns[3] = pPhase->CollocationValuesPhaseIns[3] + pPhase->CollocationValuesPhaseIns[2];

    # 	if(debug)
    # 	{
    # 		printf("\n");
    # 		printf("Inspiral Collocation Points and Values:\n");
    # 		printf("F1 : %.6f\n",pPhase->CollocationPointsPhaseIns[0]);
    # 		printf("F2 : %.6f\n",pPhase->CollocationPointsPhaseIns[1]);
    # 		printf("F3 : %.6f\n",pPhase->CollocationPointsPhaseIns[2]);
    # 		printf("F4 : %.6f\n",pPhase->CollocationPointsPhaseIns[3]);
    # 		printf("\n");
    # 		printf("V1 : %.6f\n",pPhase->CollocationValuesPhaseIns[0]);
    # 		printf("V2 : %.6f\n",pPhase->CollocationValuesPhaseIns[1]);
    # 		printf("V3 : %.6f\n",pPhase->CollocationValuesPhaseIns[2]);
    # 		printf("V4 : %.6f\n",pPhase->CollocationValuesPhaseIns[3]);
    # 		printf("\n");
    # 	}

    # 	gsl_vector_set(b,0,pPhase->CollocationValuesPhaseIns[0]);
    # 	gsl_vector_set(b,1,pPhase->CollocationValuesPhaseIns[1]);
    # 	gsl_vector_set(b,2,pPhase->CollocationValuesPhaseIns[2]);
    # 	gsl_vector_set(b,3,pPhase->CollocationValuesPhaseIns[3]);

    # 	/* A_{0,i} */
    # 	ff  = pPhase->CollocationPointsPhaseIns[0];
    # 	ff1 = cbrt(ff);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff;
    # 	ff0 = 1.0;
    # 	gsl_matrix_set(A,0,0,1.0);
    # 	gsl_matrix_set(A,0,1,ff1);
    # 	gsl_matrix_set(A,0,2,ff2);
    # 	gsl_matrix_set(A,0,3,ff3);

    # 	/* A_{1,i} */
    # 	ff  = pPhase->CollocationPointsPhaseIns[1];
    # 	ff1 = cbrt(ff);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff;
    # 	ff0 = 1.0;
    # 	gsl_matrix_set(A,1,0,1.0);
    # 	gsl_matrix_set(A,1,1,ff1);
    # 	gsl_matrix_set(A,1,2,ff2);
    # 	gsl_matrix_set(A,1,3,ff3);

    # 	/* A_{2,i} */
    # 	ff  = pPhase->CollocationPointsPhaseIns[2];
    # 	ff1 = cbrt(ff);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff;
    # 	ff0 = 1.0;
    # 	gsl_matrix_set(A,2,0,1.0);
    # 	gsl_matrix_set(A,2,1,ff1);
    # 	gsl_matrix_set(A,2,2,ff2);
    # 	gsl_matrix_set(A,2,3,ff3);

    # 	/* A_{3,i} */
    # 	ff  = pPhase->CollocationPointsPhaseIns[3];
    # 	ff1 = cbrt(ff);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff;
    # 	ff0 = 1.0;
    # 	gsl_matrix_set(A,3,0,1.0);
    # 	gsl_matrix_set(A,3,1,ff1);
    # 	gsl_matrix_set(A,3,2,ff2);
    # 	gsl_matrix_set(A,3,3,ff3);

    # 	/* We now solve the system A x = b via an LU decomposition */
    # 	gsl_linalg_LU_decomp(A,p,&s);
    # 	gsl_linalg_LU_solve(A,p,b,x);

    # 	/* Set inspiral phenomenological coefficients from solution to A x = b */
    # 	pPhase->a0 = gsl_vector_get(x,0); // x[0]; // alpha_0
    # 	pPhase->a1 = gsl_vector_get(x,1); // x[1]; // alpha_1
    # 	pPhase->a2 = gsl_vector_get(x,2); // x[2]; // alpha_2
    # 	pPhase->a3 = gsl_vector_get(x,3); // x[3]; // alpha_3
    # 	pPhase->a4 = 0.0;

    # 	/*
    # 			PSEUDO PN TERMS WORK:
    # 				- 104 works.
    # 				- 105 not tested.
    # 				- 114 not tested.
    # 				- 115 not tested.
    # 	*/
    # 	if(debug)
    # 	{
    # 		printf("\n");
    # 		printf("3pPN\n");
    # 		printf("Inspiral Pseudo-PN Coefficients:\n");
    # 		printf("a0 : %.6f\n",pPhase->a0);
    # 		printf("a1 : %.6f\n",pPhase->a1);
    # 		printf("a2 : %.6f\n",pPhase->a2);
    # 		printf("a3 : %.6f\n",pPhase->a3);
    # 		printf("a4 : %.6f\n",pPhase->a4);
    # 		printf("\n");
    # 	}

    # 	/* Tidy up in preparation for next GSL solve ... */
    # 	gsl_vector_free(b);
    # 	gsl_vector_free(x);
    # 	gsl_matrix_free(A);
    # 	gsl_permutation_free(p);

    # }
    # else if(pPhase->NPseudoPN == 5)
    # {
    # 	// Using 5 pseudo-PN coefficients so set 5 collocation points
    # 	for(i = 0; i < 5; i++)
    # 	{
    # 		fi = gpoints5[i] * deltax + xmin;
    # 		pPhase->CollocationPointsPhaseIns[i] = fi;
    # 	}
    # 	pPhase->CollocationValuesPhaseIns[0] = IMRPhenomX_Inspiral_Phase_22_d13(pWF->eta,pWF->chiPNHat,pWF->dchi,pWF->delta,pWF->IMRPhenomXInspiralPhaseVersion);
    # 	pPhase->CollocationValuesPhaseIns[1] = IMRPhenomX_Inspiral_Phase_22_d23(pWF->eta,pWF->chiPNHat,pWF->dchi,pWF->delta,pWF->IMRPhenomXInspiralPhaseVersion);
    # 	pPhase->CollocationValuesPhaseIns[2] = IMRPhenomX_Inspiral_Phase_22_v3( pWF->eta,pWF->chiPNHat,pWF->dchi,pWF->delta,pWF->IMRPhenomXInspiralPhaseVersion);
    # 	pPhase->CollocationValuesPhaseIns[3] = IMRPhenomX_Inspiral_Phase_22_d43(pWF->eta,pWF->chiPNHat,pWF->dchi,pWF->delta,pWF->IMRPhenomXInspiralPhaseVersion);
    # 	pPhase->CollocationValuesPhaseIns[4] = IMRPhenomX_Inspiral_Phase_22_d53(pWF->eta,pWF->chiPNHat,pWF->dchi,pWF->delta,pWF->IMRPhenomXInspiralPhaseVersion);

    # 	/* v_j = d_j3 + v_3 */
    # 	pPhase->CollocationValuesPhaseIns[0] = pPhase->CollocationValuesPhaseIns[0] + pPhase->CollocationValuesPhaseIns[2];
    # 	pPhase->CollocationValuesPhaseIns[1] = pPhase->CollocationValuesPhaseIns[1] + pPhase->CollocationValuesPhaseIns[2];
    # 	pPhase->CollocationValuesPhaseIns[3] = pPhase->CollocationValuesPhaseIns[3] + pPhase->CollocationValuesPhaseIns[2];
    # 	pPhase->CollocationValuesPhaseIns[4] = pPhase->CollocationValuesPhaseIns[4] + pPhase->CollocationValuesPhaseIns[2];

    # 	if(debug)
    # 	{
    # 		printf("\n");
    # 		printf("Inspiral Collocation Points and Values:\n");
    # 		printf("F1 : %.6f\n",pPhase->CollocationPointsPhaseIns[0]);
    # 		printf("F2 : %.6f\n",pPhase->CollocationPointsPhaseIns[1]);
    # 		printf("F3 : %.6f\n",pPhase->CollocationPointsPhaseIns[2]);
    # 		printf("F4 : %.6f\n",pPhase->CollocationPointsPhaseIns[3]);
    # 		printf("F5 : %.6f\n",pPhase->CollocationPointsPhaseIns[4]);
    # 		printf("\n");
    # 		printf("V1 : %.6f\n",pPhase->CollocationValuesPhaseIns[0]);
    # 		printf("V2 : %.6f\n",pPhase->CollocationValuesPhaseIns[1]);
    # 		printf("V3 : %.6f\n",pPhase->CollocationValuesPhaseIns[2]);
    # 		printf("V4 : %.6f\n",pPhase->CollocationValuesPhaseIns[3]);
    # 		printf("V5 : %.6f\n",pPhase->CollocationValuesPhaseIns[4]);
    # 		printf("\n");
    # 	}

    # 	gsl_vector_set(b,0,pPhase->CollocationValuesPhaseIns[0]);
    # 	gsl_vector_set(b,1,pPhase->CollocationValuesPhaseIns[1]);
    # 	gsl_vector_set(b,2,pPhase->CollocationValuesPhaseIns[2]);
    # 	gsl_vector_set(b,3,pPhase->CollocationValuesPhaseIns[3]);
    # 	gsl_vector_set(b,4,pPhase->CollocationValuesPhaseIns[4]);

    # 	/* A_{0,i} */
    # 	ff  = pPhase->CollocationPointsPhaseIns[0];
    # 	ff1 = cbrt(ff);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff;
    # 	ff4 = ff * ff1;
    # 	gsl_matrix_set(A,0,0,1.0);
    # 	gsl_matrix_set(A,0,1,ff1);
    # 	gsl_matrix_set(A,0,2,ff2);
    # 	gsl_matrix_set(A,0,3,ff3);
    # 	gsl_matrix_set(A,0,4,ff4);

    # 	/* A_{1,i} */
    # 	ff  = pPhase->CollocationPointsPhaseIns[1];
    # 	ff1 = cbrt(ff);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff;
    # 	ff4 = ff * ff1;
    # 	gsl_matrix_set(A,1,0,1.0);
    # 	gsl_matrix_set(A,1,1,ff1);
    # 	gsl_matrix_set(A,1,2,ff2);
    # 	gsl_matrix_set(A,1,3,ff3);
    # 	gsl_matrix_set(A,1,4,ff4);

    # 	/* A_{2,i} */
    # 	ff  = pPhase->CollocationPointsPhaseIns[2];
    # 	ff1 = cbrt(ff);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff;
    # 	ff4 = ff * ff1;
    # 	gsl_matrix_set(A,2,0,1.0);
    # 	gsl_matrix_set(A,2,1,ff1);
    # 	gsl_matrix_set(A,2,2,ff2);
    # 	gsl_matrix_set(A,2,3,ff3);
    # 	gsl_matrix_set(A,2,4,ff4);

    # 	/* A_{3,i} */
    # 	ff  = pPhase->CollocationPointsPhaseIns[3];
    # 	ff1 = cbrt(ff);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff;
    # 	ff4 = ff * ff1;
    # 	gsl_matrix_set(A,3,0,1.0);
    # 	gsl_matrix_set(A,3,1,ff1);
    # 	gsl_matrix_set(A,3,2,ff2);
    # 	gsl_matrix_set(A,3,3,ff3);
    # 	gsl_matrix_set(A,3,4,ff4);

    # 	/* A_{4,i} */
    # 	ff  = pPhase->CollocationPointsPhaseIns[4];
    # 	ff1 = cbrt(ff);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff;
    # 	ff4 = ff * ff1;
    # 	gsl_matrix_set(A,4,0,1.0);
    # 	gsl_matrix_set(A,4,1,ff1);
    # 	gsl_matrix_set(A,4,2,ff2);
    # 	gsl_matrix_set(A,4,3,ff3);
    # 	gsl_matrix_set(A,4,4,ff4);

    # 	/* We now solve the system A x = b via an LU decomposition */
    # 	gsl_linalg_LU_decomp(A,p,&s);
    # 	gsl_linalg_LU_solve(A,p,b,x);

    # 	/* Set inspiral phenomenological coefficients from solution to A x = b */
    # 	pPhase->a0 = gsl_vector_get(x,0); // x[0];
    # 	pPhase->a1 = gsl_vector_get(x,1); // x[1];
    # 	pPhase->a2 = gsl_vector_get(x,2); // x[2];
    # 	pPhase->a3 = gsl_vector_get(x,3); // x[3];
    # 	pPhase->a4 = gsl_vector_get(x,4); // x[4];

    # 	if(debug)
    # 	{
    # 		printf("\n");
    # 		printf("4pPN\n");
    # 		printf("Inspiral Pseudo-PN Coefficients:\n");
    # 		printf("a0 : %.6f\n",pPhase->a0);
    # 		printf("a1 : %.6f\n",pPhase->a1);
    # 		printf("a2 : %.6f\n",pPhase->a2);
    # 		printf("a3 : %.6f\n",pPhase->a3);
    # 		printf("a4 : %.6f\n",pPhase->a4);
    # 		printf("\n");
    # 	}

    # 	/* Tidy up in preparation for next GSL solve ... */
    # 	gsl_vector_free(b);
    # 	gsl_vector_free(x);
    # 	gsl_matrix_free(A);
    # 	gsl_permutation_free(p);
    # }
    # else
    # {
    # 	XLALPrintError("Error in ComputeIMRPhenomXWaveformVariables: NPseudoPN requested is not valid.\n");
    # }

    # /* The pseudo-PN coefficients are normalized such that: (dphase0 / eta) * f^{8/3} * a_j */
    # /* So we must re-scale these terms by an extra factor of f^{-8/3} in the PN phasing */
    # pPhase->sigma1 = (-5.0/3.0) * pPhase->a0;
    # pPhase->sigma2 = (-5.0/4.0) * pPhase->a1;
    # pPhase->sigma3 = (-5.0/5.0) * pPhase->a2;
    # pPhase->sigma4 = (-5.0/6.0) * pPhase->a3;
    # pPhase->sigma5 = (-5.0/7.0) * pPhase->a4;

    # /* Initialize TaylorF2 PN coefficients  */
    # pPhase->dphi0  = 0.0;
    # pPhase->dphi1  = 0.0;
    # pPhase->dphi2  = 0.0;
    # pPhase->dphi3  = 0.0;
    # pPhase->dphi4  = 0.0;
    # pPhase->dphi5  = 0.0;
    # pPhase->dphi6  = 0.0;
    # pPhase->dphi7  = 0.0;
    # pPhase->dphi8  = 0.0;
    # pPhase->dphi9  = 0.0;
    # pPhase->dphi10 = 0.0;
    # pPhase->dphi11 = 0.0;
    # pPhase->dphi12 = 0.0;

    # pPhase->dphi5L = 0.0;
    # pPhase->dphi6L = 0.0;
    # pPhase->dphi8L = 0.0;
    # pPhase->dphi9L = 0.0;

    # pPhase->phi0   = 0.0;
    # pPhase->phi1   = 0.0;
    # pPhase->phi2   = 0.0;
    # pPhase->phi3   = 0.0;
    # pPhase->phi4   = 0.0;
    # pPhase->phi5   = 0.0;
    # pPhase->phi6   = 0.0;
    # pPhase->phi7   = 0.0;
    # pPhase->phi8   = 0.0;
    # pPhase->phi9   = 0.0;
    # pPhase->phi10  = 0.0;
    # pPhase->phi11  = 0.0;
    # pPhase->phi12  = 0.0;

    # pPhase->phi5L  = 0.0;
    # pPhase->phi6L  = 0.0;
    # pPhase->phi8L  = 0.0;
    # pPhase->phi9L  = 0.0;

    # /* **** TaylorF2 PN Coefficients: Phase **** */

    # /*
    # 		- These are the PN coefficients normalised by: 3 / (128 * eta * [pi M f]^{5/3} ).
    # 		- We add in powers of (M f)^{N/3} later but add powers of pi^{N/3} here
    # 		- The log terms are *always* in terms of log(v), so we multiply by log(v) when summing PN phasing series.
    # 		- We *do not* overwrite the PN phasing series with pseudo-PN terms. These are added separately.

    # 		PN terms can be found in:
    # 			- Marsat et al, CQG, 32, 085008, (2015)
    # 			- Bohe et al, CQG, 32, 195010, (2015)
    # 			- Bernard et al, PRD, 95, 044026, (2017)
    # 			- Bernard et al, PRD, 93, 084037, (2016)
    # 			- Damour et al, PRD, 89, 064058, (2014)
    # 			- Damour et al, PRD, 95, 084005, (2017)
    # 			- Bernard et al, PRD, 96, 104043, (2017)
    # 			- Marchand et al, PRD, 97, 044023, (2018)
    # 			- Marchand et al, CQG, 33, 244003, (2016)
    # 			- Nagar et al, PRD, 99, 044007, (2019)
    # 			- Messina et al, PRD, 97, 084016, (2018)
    # */

    # /* Split into non-spinning and spin-dependent coefficients */
    # UNUSED REAL8 phi0S = 0.0, phi1S = 0.0, phi2S = 0.0;
    # REAL8 phi0NS  = 0.0,  phi1NS = 0.0,  phi2NS = 0.0;
    # REAL8 phi3NS  = 0.0,  phi3S  = 0.0,  phi4NS = 0.0,  phi4S   = 0.0,  phi5NS  = 0.0,  phi5S  = 0.0;
    # REAL8 phi5LNS = 0.0,  phi5LS = 0.0,  phi6NS = 0.0,  phi6S   = 0.0,  phi6LNS = 0.0,  phi6LS = 0.0;
    # REAL8 phi7NS  = 0.0,  phi7S  = 0.0,  phi8NS = 0.0,  phi8S   = 0.0,  phi8LNS = 0.0;
    # REAL8 phi8LS  = 0.0,  phi9NS = 0.0,  phi9S  = 0.0,  phi9LNS = 0.0,  phi9LS  = 0.0;

    # /* Analytically known PN coefficients */
    # /* Newtonian */
    # phi0NS         = 1.0;

    # /* ~~ 0.5 PN ~~ */
    # phi1NS         = 0.0;

    # /* ~~ 1.0 PN ~~ */
    # /* 1.0PN, Non-Spinning */
    # phi2NS         = (3715/756. + (55*eta)/9.) * powers_of_lalpi.two_thirds;

    # /* ~~ 1.5 PN ~~ */
    # /* 1.5PN, Non-Spinning */
    # phi3NS         = -16.0 * powers_of_lalpi.two;
    # /* 1.5PN, Spin-Orbit */
    # phi3S          = ( (113*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 76*(chi1L + chi2L)*eta)/6. ) * powers_of_lalpi.itself;

    # /* ~~ 2.0 PN ~~ */
    # /* 2.0PN, Non-Spinning */
    # phi4NS         = ( 15293365/508032. + (27145*eta)/504. + (3085*eta2)/72. ) * powers_of_lalpi.four_thirds;
    # /* 2.0PN, Spin-Spin */
    # phi4S          = ( (-5*(81*chi1L2*(1 + delta - 2*eta) + 316*chi1L2L*eta - 81*chi2L2*(-1 + delta + 2*eta)))/16. ) * powers_of_lalpi.four_thirds;

    # /* ~~ 2.5 PN ~~ */
    # phi5NS         = 0.0;
    # phi5S          = 0.0;

    # /* ~~ 2.5 PN, Log Term ~~ */
    # /* 2.5PN, Non-Spinning */
    # phi5LNS        = ( (5*(46374 - 6552*eta)*LAL_PI)/4536. ) * powers_of_lalpi.five_thirds;
    # /* 2.5PN, Spin-Orbit */
    # phi5LS         = ( (-732985*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 560*(-1213*(chi1L + chi2L)
    # 											+ 63*(chi1L - chi2L)*delta)*eta + 85680*(chi1L + chi2L)*eta2)/4536. ) * powers_of_lalpi.five_thirds;

    # /* ~~ 3.0 PN ~~ */
    # /* 3.0 PN, Non-Spinning */
    # phi6NS         = ( 11583231236531/4.69421568e9 - (5*eta*(3147553127 + 588*eta*(-45633 + 102260*eta)))/3.048192e6 - (6848*LAL_GAMMA)/21.
    # 					- (640*powers_of_lalpi.two)/3. + (2255*eta*powers_of_lalpi.two)/12. - (13696*log(2))/21. - (6848*powers_of_lalpi.log)/63. ) * powers_of_lalpi.two;
    # /* 3.0 PN, Spin-Orbit */
    # phi6S          = ( (5*(227*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 156*(chi1L + chi2L)*eta)*LAL_PI)/3. ) * powers_of_lalpi.two;
    # /* 3.0 PN, Spin-Spin */
    # phi6S         += ( (5*(20*chi1L2L*eta*(11763 + 12488*eta) + 7*chi2L2*(-15103*(-1 + delta) + 2*(-21683 + 6580*delta)*eta - 9808*eta2) -
    # 						7*chi1L2*(-15103*(1 + delta) + 2*(21683 + 6580*delta)*eta + 9808*eta2)))/4032. ) * powers_of_lalpi.two;

    # /* ~~ 3.0 PN, Log Term ~~ */
    # phi6LNS        = (-6848/63.) * powers_of_lalpi.two;
    # phi6LS         = 0.0;

    # /* ~~ 3.5 PN ~~ */
    # /* 3.5 PN, Non-Spinning */
    # phi7NS         = ( (5*(15419335 + 168*(75703 - 29618*eta)*eta)*LAL_PI)/254016. ) * powers_of_lalpi.seven_thirds;
    # /* 3.5 PN, Spin-Orbit */
    # phi7S          = ( (5*(-5030016755*(chi1L + chi2L + chi1L*delta - chi2L*delta) + 4*(2113331119*(chi1L + chi2L) + 675484362*(chi1L - chi2L)*delta)*eta - 1008*(208433*(chi1L + chi2L) + 25011*(chi1L - chi2L)*delta)*eta2 + 90514368*(chi1L + chi2L)*eta3))/6.096384e6 ) * powers_of_lalpi.seven_thirds;
    # /* 3.5 PN, Spin-Spin */
    # phi7S         += ( -5*(57*chi1L2*(1 + delta - 2*eta) + 220*chi1L2L*eta - 57*chi2L2*(-1 + delta + 2*eta))*LAL_PI ) * powers_of_lalpi.seven_thirds;
    # /* 3.5 PN, Cubic-in-Spin */
    # phi7S         += ( (14585*(-(chi2L3*(-1 + delta)) + chi1L3*(1 + delta)) - 5*(chi2L3*(8819 - 2985*delta) + 8439*chi1L*chi2L2*(-1 + delta) - 8439*chi1L2*chi2L*(1 + delta) + chi1L3*(8819 + 2985*delta))*eta + 40*(chi1L + chi2L)*(17*chi1L2 - 14*chi1L2L + 17*chi2L2)*eta2)/48. ) * powers_of_lalpi.seven_thirds;

    # 	/* ~~ 4.0 PN ~~ */
    # /* 4.0 PN, Non-Spinning */
    # phi8NS         = 0.0;
    # /* 4.0 PN, Spin-Orbit */
    # phi8S          = ( (-5*(1263141*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 2*(794075*(chi1L + chi2L) + 178533*(chi1L - chi2L)*delta)*eta + 94344*(chi1L + chi2L)*eta2)*LAL_PI*(-1 + powers_of_lalpi.log))/9072. ) * powers_of_lalpi.eight_thirds;

    # /* ~~ 4.0 PN, Log Term ~~ */
    # /* 4.0 PN, log term, Non-Spinning */
    # phi8LNS        = 0.0;
    # /* 4.0 PN, log term, Spin-Orbit */
    # phi8LS         = ((-5*(1263141*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 2*(794075*(chi1L + chi2L) + 178533*(chi1L - chi2L)*delta)*eta
    # 						+ 94344*(chi1L + chi2L)*eta2)*LAL_PI)/9072.) * powers_of_lalpi.eight_thirds;

    # /* ~~ 4.5 PN ~~ */
    # phi9NS         = 0.0;
    # phi9S          = 0.0;

    # 	/* ~~ 4.5 PN, Log Term ~~ */
    # phi9LNS        = 0.0;
    # phi9LS         = 0.0;

    # /* This version of TaylorF2 contains an additional 4.5PN tail term and a LO-SS tail term at 3.5PN */
    # if(pWF->IMRPhenomXInspiralPhaseVersion == 114 || pWF->IMRPhenomXInspiralPhaseVersion == 115)
    # {
    # 		/* 3.5PN, Leading Order Spin-Spin Tail Term */
    # 		phi7S         += ( (5*(65*chi1L2*(1 + delta - 2*eta) + 252*chi1L2L*eta - 65*chi2L2*(-1 + delta + 2*eta))*LAL_PI)/4. ) * powers_of_lalpi.seven_thirds;

    # 		/* 4.5PN, Tail Term */
    # 		phi9NS        += ( (5*(-256 + 451*eta)*powers_of_lalpi.three)/6. + (LAL_PI*(105344279473163 + 700*eta*(-298583452147 + 96*eta*(99645337 + 14453257*eta)) -
    # 																					12246091038720*LAL_GAMMA - 24492182077440*log(2.0)))/1.877686272e10 - (13696*LAL_PI*powers_of_lalpi.log)/63. ) * powers_of_lalpi.three;

    # 		/* 4.5PN, Log Term */
    # 		phi9LNS       += (  (-13696*LAL_PI)/63.0  ) * powers_of_lalpi.three;
    # }

    # /* 0.0 PN */
    # pPhase->phi0   = phi0NS;

    # /* 0.5 PN */
    # pPhase->phi1   = phi1NS;

    # /* 1.0 PN */
    # pPhase->phi2   = phi2NS;

    # /* 1.5 PN */
    # pPhase->phi3   = phi3NS + phi3S;

    # /* 2.0 PN */
    # pPhase->phi4   = phi4NS + phi4S;

    # /* 2.5 PN */
    # pPhase->phi5   = phi5NS + phi5S;

    # /* 2.5 PN, Log Terms */
    # pPhase->phi5L  = phi5LNS + phi5LS;

    # /* 3.0 PN */
    # pPhase->phi6   = phi6NS + phi6S;

    # /* 3.0 PN, Log Term */
    # pPhase->phi6L  = phi6LNS + phi6LS;

    # /* 3.5PN */
    # pPhase->phi7   = phi7NS + phi7S;

    # /* 4.0PN */
    # pPhase->phi8   = phi8NS + phi8S;

    # /* 4.0 PN, Log Terms */
    # pPhase->phi8L  = phi8LNS + phi8LS;

    # /* 4.5 PN */
    # pPhase->phi9   = phi9NS + phi9S;

    # /* 4.5 PN, Log Terms */
    # pPhase->phi9L  = phi9LNS + phi9LS;

    # if(debug)
    # {
    # 	printf("TaylorF2 PN Coefficients: \n");
    # 	printf("phi0   : %.6f\n",pPhase->phi0);
    # 	printf("phi1   : %.6f\n",pPhase->phi1);
    # 	printf("phi2   : %.6f\n",pPhase->phi2);
    # 	printf("phi3   : %.6f\n",pPhase->phi3);
    # 	printf("phi4   : %.6f\n",pPhase->phi4);
    # 	printf("phi5   : %.6f\n",pPhase->phi5);
    # 	printf("phi6   : %.6f\n",pPhase->phi6);
    # 	printf("phi7   : %.6f\n",pPhase->phi7);
    # 	printf("phi8   : %.6f\n",pPhase->phi8);

    # 	printf("phi5L  : %.6f\n",pPhase->phi5L);
    # 	printf("phi6L  : %.6f\n",pPhase->phi6L);
    # 	printf("phi8L  : %.6f\n",pPhase->phi8L);

    # 	printf("phi8P  : %.6f\n",pPhase->sigma1);
    # 	printf("phi9P  : %.6f\n",pPhase->sigma2);
    # 	printf("phi10P : %.6f\n",pPhase->sigma3);
    # 	printf("phi11P : %.6f\n",pPhase->sigma4);
    # 	printf("phi12P : %.6f\n",pPhase->sigma5);
    # }

    # pPhase->phi_initial = - LAL_PI_4;

    # /* **** TaylorF2 PN Coefficients: Normalized Phase Derivative **** */
    # pPhase->dphi0  = pPhase->phi0;
    # pPhase->dphi1  = 4.0 / 5.0 * pPhase->phi1;
    # pPhase->dphi2  = 3.0 / 5.0 * pPhase->phi2;
    # pPhase->dphi3  = 2.0 / 5.0 * pPhase->phi3;
    # pPhase->dphi4  = 1.0 / 5.0 * pPhase->phi4;
    # pPhase->dphi5  = -3.0 / 5.0 * pPhase->phi5L;
    # pPhase->dphi6  = -1.0 / 5.0 * pPhase->phi6 - 3.0 / 5.0 * pPhase->phi6L;
    # pPhase->dphi6L = -1.0 / 5.0 * pPhase->phi6L;
    # pPhase->dphi7  = -2.0 / 5.0 * pPhase->phi7;
    # pPhase->dphi8  = -3.0 / 5.0 * pPhase->phi8 - 3.0 / 5.0 * pPhase->phi8L;
    # pPhase->dphi8L = -3.0 / 5.0 * pPhase->phi8L;
    # pPhase->dphi9  = -4.0 / 5.0 * pPhase->phi9 - 3.0 / 5.0 * pPhase->phi9L;
    # pPhase->dphi9L = -3.0 / 5.0 * pPhase->phi9L;

    # if(debug)
    # {
    # 	printf("\nTaylorF2 PN Derivative Coefficients\n");
    # 	printf("dphi0  : %.6f\n",pPhase->dphi0);
    # 	printf("dphi1  : %.6f\n",pPhase->dphi1);
    # 	printf("dphi2  : %.6f\n",pPhase->dphi2);
    # 	printf("dphi3  : %.6f\n",pPhase->dphi3);
    # 	printf("dphi4  : %.6f\n",pPhase->dphi4);
    # 	printf("dphi5  : %.6f\n",pPhase->dphi5);
    # 	printf("dphi6  : %.6f\n",pPhase->dphi6);
    # 	printf("dphi7  : %.6f\n",pPhase->dphi7);
    # 	printf("dphi8  : %.6f\n",pPhase->dphi8);
    # 	printf("dphi9  : %.6f\n",pPhase->dphi9);
    # 	printf("\n");
    # 	printf("dphi6L : %.6f\n",pPhase->dphi6L);
    # 	printf("dphi8L : %.6f\n",pPhase->dphi8L);
    # 	printf("dphi9L : %.6f\n",pPhase->dphi9L);
    # }

    # /*
    # 		Calculate phase at fmatchIN. This will be used as the collocation point for the intermediate fit.
    # 		In practice, the transition point is just below the MECO frequency.
    # */
    # if(debug)
    # {
    # 	printf("\nTransition frequency for ins to int : %.6f\n",pPhase->fPhaseMatchIN);
    # }

    # IMRPhenomX_UsefulPowers powers_of_fmatchIN;
    # IMRPhenomX_Initialize_Powers(&powers_of_fmatchIN,pPhase->fPhaseMatchIN);

    # double phaseIN;
    # phaseIN  = pPhase->dphi0; 																	// f^{0/3}
    # phaseIN += pPhase->dphi1 	* powers_of_fmatchIN.one_third; 								// f^{1/3}
    # phaseIN += pPhase->dphi2 	* powers_of_fmatchIN.two_thirds; 								// f^{2/3}
    # phaseIN += pPhase->dphi3 	* powers_of_fmatchIN.itself; 									// f^{3/3}
    # phaseIN += pPhase->dphi4 	* powers_of_fmatchIN.four_thirds; 								// f^{4/3}
    # phaseIN += pPhase->dphi5 	* powers_of_fmatchIN.five_thirds; 								// f^{5/3}
    # phaseIN += pPhase->dphi6  	* powers_of_fmatchIN.two;										// f^{6/3}
    # phaseIN += pPhase->dphi6L 	* powers_of_fmatchIN.two * powers_of_fmatchIN.log;				// f^{6/3}, Log[f]
    # phaseIN += pPhase->dphi7  	* powers_of_fmatchIN.seven_thirds;								// f^{7/3}
    # phaseIN += pPhase->dphi8  	* powers_of_fmatchIN.eight_thirds;								// f^{8/3}
    # phaseIN += pPhase->dphi8L 	* powers_of_fmatchIN.eight_thirds * powers_of_fmatchIN.log;		// f^{8/3}
    # phaseIN += pPhase->dphi9  	* powers_of_fmatchIN.three;										// f^{9/3}
    # phaseIN += pPhase->dphi9L 	* powers_of_fmatchIN.three * powers_of_fmatchIN.log;			// f^{9/3}

    # // Add pseudo-PN Coefficient
    # phaseIN += ( 		pPhase->a0 * powers_of_fmatchIN.eight_thirds
    # 							+ pPhase->a1 * powers_of_fmatchIN.three
    # 							+ pPhase->a2 * powers_of_fmatchIN.eight_thirds * powers_of_fmatchIN.two_thirds
    # 							+ pPhase->a3 * powers_of_fmatchIN.eight_thirds * powers_of_fmatchIN.itself
    # 							+ pPhase->a4 * powers_of_fmatchIN.eight_thirds * powers_of_fmatchIN.four_thirds
    # 						);

    # phaseIN  = phaseIN * powers_of_fmatchIN.m_eight_thirds * pWF->dphase0;

    # /*
    # Intermediate phase collocation points:
    # Here we only use the first N points in the array where N = the
    # number of intermediate collocation points.

    # The size of the array is controlled by: N_MAX_COLLOCATION_POINTS_PHASE_INT

    # Default is to use 5 collocation points.

    # See. Eq. 7.7 and 7.8 where f_H = pPhase->fPhaseMatchIM and f_L = pPhase->fPhaseMatchIN
    # */
    # deltax      = pPhase->fPhaseMatchIM - pPhase->fPhaseMatchIN;
    # xmin        = pPhase->fPhaseMatchIN;

    # switch(pWF->IMRPhenomXIntermediatePhaseVersion)
    # {
    # 	case 104:
    # 	{
    # 		// Fourth order polynomial ansatz
    # 		pPhase->NCollocationPointsInt = 4;
    # 		break;
    # 	}
    # 	case 105:
    # 	{
    # 		// Fifth order polynomial ansatz
    # 		pPhase->NCollocationPointsInt = 5;
    # 		break;
    # 	}
    # 	default:
    # 	{
    # 		XLAL_ERROR(XLAL_EINVAL, "Error: IMRPhenomXIntermediatePhaseVersion is not valid.\n");
    # 	}
    # }

    # if(debug)
    # {
    # printf("\nNColPointsInt : %d\n",pPhase->NCollocationPointsInt);
    # }

    # p = gsl_permutation_alloc(pPhase->NCollocationPointsInt);
    # b = gsl_vector_alloc(pPhase->NCollocationPointsInt);
    # x = gsl_vector_alloc(pPhase->NCollocationPointsInt);
    # A = gsl_matrix_alloc(pPhase->NCollocationPointsInt,pPhase->NCollocationPointsInt);

    # // Canonical intermediate model using 4 collocation points
    # if(pWF->IMRPhenomXIntermediatePhaseVersion == 104)
    # {
    # 	// Using 4 collocation points in intermediate region
    # 	for(i = 0; i < 4; i++)
    # 	{
    # 		fi = gpoints4[i] * deltax + xmin;

    # 		pPhase->CollocationPointsPhaseInt[i] = fi;
    # 	}

    # 	// v2IM - v4RD. Using v4RD helps condition the fits with v4RD being very a robust fit.
    # 	double v2IMmRDv4 = IMRPhenomX_Intermediate_Phase_22_v2mRDv4(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXIntermediatePhaseVersion);

    # 	// v3IM - v4RD. Using v4RD helps condition the fits with v4RD being very a robust fit.
    # 	double v3IMmRDv4 = IMRPhenomX_Intermediate_Phase_22_v3mRDv4(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXIntermediatePhaseVersion);

    # 	// Direct fit to the collocation point at F2. We will take a weighted average of the direct and conditioned fit.
    # 	double v2IM      = IMRPhenomX_Intermediate_Phase_22_v2(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXIntermediatePhaseVersion);

    # 	/* Evaluate collocation points */
    # 	pPhase->CollocationValuesPhaseInt[0] = phaseIN;

    # 	// Take a weighted average for these points? Can help condition the fit.
    # 	pPhase->CollocationValuesPhaseInt[1] = 0.75*(v2IMmRDv4 + RDv4) + 0.25*v2IM;

    # 	// Use just v2 - v4RD to reconstruct the fit?
    # 	//pPhase->CollocationValuesPhaseInt[1] = v2IMmRDv4 + RDv4);

    # 	pPhase->CollocationValuesPhaseInt[2] = v3IMmRDv4 + RDv4;

    # 	pPhase->CollocationValuesPhaseInt[3] = phaseRD;

    # 	/* A_{0,i} */
    # 	ff  = pPhase->CollocationPointsPhaseInt[0];
    # 	ff1 = pWF->fRING / ff;
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff1 * ff2;
    # 	ff0 = (4 * pPhase->cL) / (4.0*pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # 	gsl_matrix_set(A,0,0,1.0);
    # 	gsl_matrix_set(A,0,1,ff1);
    # 	gsl_matrix_set(A,0,2,ff2);
    # 	gsl_matrix_set(A,0,3,ff3);
    # 	gsl_vector_set(b,0,pPhase->CollocationValuesPhaseInt[0] - ff0);

    # 	/* A_{1,i} */
    # 	ff  = pPhase->CollocationPointsPhaseInt[1];
    # 	ff1 = 1.0 / (ff / pWF->fRING);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff1 * ff2;
    # 	ff0 = (4 * pPhase->cL) / (4.0*pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # 	gsl_matrix_set(A,1,0,1);
    # 	gsl_matrix_set(A,1,1,ff1);
    # 	gsl_matrix_set(A,1,2,ff2);
    # 	gsl_matrix_set(A,1,3,ff3);
    # 	gsl_vector_set(b,1,pPhase->CollocationValuesPhaseInt[1] - ff0);

    # 	/* A_{2,i} */
    # 	ff  = pPhase->CollocationPointsPhaseInt[2];
    # 	ff1 = 1.0 / (ff / pWF->fRING);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff1 * ff2;
    # 	ff0 = (4 * pPhase->cL) / (4.0*pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # 	gsl_matrix_set(A,2,0,1);
    # 	gsl_matrix_set(A,2,1,ff1);
    # 	gsl_matrix_set(A,2,2,ff2);
    # 	gsl_matrix_set(A,2,3,ff3);
    # 	gsl_vector_set(b,2,pPhase->CollocationValuesPhaseInt[2] - ff0);

    # 	/* A_{3,i} */
    # 	ff  = pPhase->CollocationPointsPhaseInt[3];
    # 	ff1 = 1.0 / (ff / pWF->fRING);
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff1 * ff2;
    # 	ff0 = (4 * pPhase->cL) / (4.0*pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # 	gsl_matrix_set(A,3,0,1);
    # 	gsl_matrix_set(A,3,1,ff1);
    # 	gsl_matrix_set(A,3,2,ff2);
    # 	gsl_matrix_set(A,3,3,ff3);
    # 	gsl_vector_set(b,3,pPhase->CollocationValuesPhaseInt[3] - ff0);

    # 	/* We now solve the system A x = b via an LU decomposition */
    # 	gsl_linalg_LU_decomp(A,p,&s);
    # 	gsl_linalg_LU_solve(A,p,b,x);

    # 	/* Set intermediate phenomenological coefficients from solution to A x = b */
    # 	pPhase->b0 = gsl_vector_get(x,0);                                                        // x[0] // Constant
    # 	pPhase->b1 = gsl_vector_get(x,1) * pWF->fRING;                                           // x[1] // f^{-1}
    # 	pPhase->b2 = gsl_vector_get(x,2) * pWF->fRING * pWF->fRING;                              // x[2] // f^{-2}
    # 	//pPhase->b3 = 0.0;
    # 	pPhase->b4 = gsl_vector_get(x,3) * pWF->fRING * pWF->fRING * pWF->fRING * pWF->fRING;    // x[3] // f^{-4}

    # 	/* Tidy up in preparation for next GSL solve ... */
    # 	gsl_vector_free(b);
    # 	gsl_vector_free(x);
    # 	gsl_matrix_free(A);
    # 	gsl_permutation_free(p);
    # }
    # // Canonical intermediate model using 5 collocation points
    # else if(pWF->IMRPhenomXIntermediatePhaseVersion == 105)
    # {
    # 	// Using 5 collocation points in intermediate region
    # 	for(i = 0; i < 5; i++)
    # 	{
    # 		fi = gpoints5[i] * deltax + xmin;

    # 		pPhase->CollocationPointsPhaseInt[i] = fi;
    # 	}

    # 	/* Evaluate collocation points */

    # 	/* The first and last collocation points for the intermediate region are set from the inspiral fit and ringdown respectively */
    # 	pPhase->CollocationValuesPhaseInt[0] = phaseIN;
    # 	pPhase->CollocationValuesPhaseInt[4] = phaseRD;

    # 	// v2IM - v4RD. Using v4RD helps condition the fits with v4RD being very a robust fit.
    # 	double v2IMmRDv4 = IMRPhenomX_Intermediate_Phase_22_v2mRDv4(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXIntermediatePhaseVersion);

    # 	// v3IM - v4RD. Using v4RD helps condition the fits with v4RD being very a robust fit.
    # 	double v3IMmRDv4 = IMRPhenomX_Intermediate_Phase_22_v3mRDv4(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXIntermediatePhaseVersion);

    # 	// Direct fit to the collocation point at F2. We will take a weighted average of the direct and conditioned fit.
    # 	double v2IM      = IMRPhenomX_Intermediate_Phase_22_v2(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXIntermediatePhaseVersion);

    # 	// Take a weighted average for these points. Helps condition the fit.
    # 	pPhase->CollocationValuesPhaseInt[1] = 0.75*(v2IMmRDv4 + RDv4) + 0.25*v2IM;

    # 	pPhase->CollocationValuesPhaseInt[2] = v3IMmRDv4 + RDv4;
    # 	pPhase->CollocationValuesPhaseInt[3] = IMRPhenomX_Intermediate_Phase_22_d43(pWF->eta,pWF->STotR,pWF->dchi,pWF->delta,pWF->IMRPhenomXIntermediatePhaseVersion);

    # 	// Collocation points: v4 = d43 + v3
    # 	pPhase->CollocationValuesPhaseInt[3] = pPhase->CollocationValuesPhaseInt[3] + pPhase->CollocationValuesPhaseInt[2];

    # 	/* A_{0,i} */
    # 	ff  = pPhase->CollocationPointsPhaseInt[0];
    # 	ff1 = pWF->fRING / ff;
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff1 * ff2;
    # 	ff4 = ff2 * ff2;
    # 	ff0 = (4.0 * pPhase->cL) / ((2.0*pWF->fDAMP)*(2.0*pWF->fDAMP) + (ff - pWF->fRING)*(ff - pWF->fRING));
    # 	gsl_matrix_set(A,0,0,1.0);
    # 	gsl_matrix_set(A,0,1,ff1);
    # 	gsl_matrix_set(A,0,2,ff2);
    # 	gsl_matrix_set(A,0,3,ff3);
    # 	gsl_matrix_set(A,0,4,ff4);
    # 	gsl_vector_set(b,0,pPhase->CollocationValuesPhaseInt[0] - ff0);

    # 	if(debug)
    # 	{
    # 	printf("For row 0: a0 + a1 %.6f + a2 %.6f + a3 %.6f + a4 %.6f = %.6f , ff0 = %.6f, ff = %.6f\n",ff1,ff2,ff3,ff4,pPhase->CollocationValuesPhaseInt[0] - ff0,ff0,ff);
    # 	}

    # 	/* A_{1,i} */
    # 	ff  = pPhase->CollocationPointsPhaseInt[1];
    # 	ff1 = pWF->fRING / ff;
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff1 * ff2;
    # 	ff4 = ff2 * ff2;
    # 	ff0 = (4 * pPhase->cL) / (4.0*pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # 	gsl_matrix_set(A,1,0,1.0);
    # 	gsl_matrix_set(A,1,1,ff1);
    # 	gsl_matrix_set(A,1,2,ff2);
    # 	gsl_matrix_set(A,1,3,ff3);
    # 	gsl_matrix_set(A,1,4,ff4);
    # 	gsl_vector_set(b,1,pPhase->CollocationValuesPhaseInt[1] - ff0);

    # 	if(debug)
    # 	{
    # 	printf("For row 1: a0 + a1 %.6f + a2 %.6f + a3 %.6f + a4 %.6f = %.6f\n",ff1,ff2,ff3,ff4,pPhase->CollocationValuesPhaseInt[1] - ff0);
    # 	}

    # 	/* A_{2,i} */
    # 	ff  = pPhase->CollocationPointsPhaseInt[2];
    # 	ff1 = pWF->fRING / ff;
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff1 * ff2;
    # 	ff4 = ff2 * ff2;
    # 	ff0 = (4 * pPhase->cL) / (4.0*pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # 	gsl_matrix_set(A,2,0,1.0);
    # 	gsl_matrix_set(A,2,1,ff1);
    # 	gsl_matrix_set(A,2,2,ff2);
    # 	gsl_matrix_set(A,2,3,ff3);
    # 	gsl_matrix_set(A,2,4,ff4);
    # 	gsl_vector_set(b,2,pPhase->CollocationValuesPhaseInt[2] - ff0);

    # 	if(debug)
    # 	{
    # 	printf("For row 2: a0 + a1 %.6f + a2 %.6f + a3 %.6f + a4 %.6f = %.6f\n",ff1,ff2,ff3,ff4,pPhase->CollocationValuesPhaseInt[2] - ff0);
    # 	}

    # 	/* A_{3,i} */
    # 	ff  = pPhase->CollocationPointsPhaseInt[3];
    # 	ff1 = pWF->fRING / ff;
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff1 * ff2;
    # 	ff4 = ff2 * ff2;
    # 	ff0 = (4 * pPhase->cL) / (4.0*pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # 	gsl_matrix_set(A,3,0,1.0);
    # 	gsl_matrix_set(A,3,1,ff1);
    # 	gsl_matrix_set(A,3,2,ff2);
    # 	gsl_matrix_set(A,3,3,ff3);
    # 	gsl_matrix_set(A,3,4,ff4);
    # 	gsl_vector_set(b,3,pPhase->CollocationValuesPhaseInt[3] - ff0);

    # 	if(debug)
    # 	{
    # 	printf("For row 3: a0 + a1 %.6f + a2 %.6f + a3 %.6f + a4 %.6f = %.6f\n",ff1,ff2,ff3,ff4,pPhase->CollocationValuesPhaseInt[3] - ff0);
    # 	}

    # 	/* A_{4,i} */
    # 	ff  = pPhase->CollocationPointsPhaseInt[4];
    # 	ff1 = pWF->fRING / ff;
    # 	ff2 = ff1 * ff1;
    # 	ff3 = ff1 * ff2;
    # 	ff4 = ff2 * ff2;
    # 	ff0 = (4 * pPhase->cL) / (4.0*pWF->fDAMP*pWF->fDAMP + (ff - pWF->fRING)*(ff - pWF->fRING));
    # 	gsl_matrix_set(A,4,0,1.0);
    # 	gsl_matrix_set(A,4,1,ff1);
    # 	gsl_matrix_set(A,4,2,ff2);
    # 	gsl_matrix_set(A,4,3,ff3);
    # 	gsl_matrix_set(A,4,4,ff4);
    # 	gsl_vector_set(b,4,pPhase->CollocationValuesPhaseInt[4] - ff0);

    # 	if(debug)
    # 	{
    # 	printf("For row 4: a0 + a1 %.6f + a2 %.6f + a3 %.6f + a4 %.6f = %.6f\n",ff1,ff2,ff3,ff4,pPhase->CollocationValuesPhaseInt[4] - ff0);
    # 	}

    # 	/* We now solve the system A x = b via an LU decomposition */
    # 	gsl_linalg_LU_decomp(A,p,&s);
    # 	gsl_linalg_LU_solve(A,p,b,x);

    # 	/* Set intermediate phenomenological coefficients from solution to A x = b */
    # 	pPhase->b0 = gsl_vector_get(x,0);                                                      // x[0] // Const.
    # 	pPhase->b1 = gsl_vector_get(x,1) * pWF->fRING;                                         // x[1] // f^{-1}
    # 	pPhase->b2 = gsl_vector_get(x,2) * pWF->fRING * pWF->fRING;                            // x[2] // f^{-2}
    # 	pPhase->b3 = gsl_vector_get(x,3) * pWF->fRING * pWF->fRING * pWF->fRING;               // x[3] // f^{-3}
    # 	pPhase->b4 = gsl_vector_get(x,4) * pWF->fRING * pWF->fRING * pWF->fRING * pWF->fRING;  // x[4] // f^{-4}

    # 	if(debug)
    # 	{
    # 	printf("\n");
    # 	printf("Intermediate Collocation Points and Values:\n");
    # 	printf("F1 : %.7f\n",pPhase->CollocationPointsPhaseInt[0]);
    # 	printf("F2 : %.7f\n",pPhase->CollocationPointsPhaseInt[1]);
    # 	printf("F3 : %.7f\n",pPhase->CollocationPointsPhaseInt[2]);
    # 	printf("F4 : %.7f\n",pPhase->CollocationPointsPhaseInt[3]);
    # 	printf("F5 : %.7f\n",pPhase->CollocationPointsPhaseInt[4]);
    # 	printf("\n");
    # 	printf("V's agree with Mathematica...\n");
    # 	printf("V1 : %.7f\n",pPhase->CollocationValuesPhaseInt[0]);
    # 	printf("V2 : %.7f\n",pPhase->CollocationValuesPhaseInt[1]);
    # 	printf("V3 : %.7f\n",pPhase->CollocationValuesPhaseInt[2]);
    # 	printf("V4 : %.7f\n",pPhase->CollocationValuesPhaseInt[3]);
    # 	printf("V5 : %.7f\n",pPhase->CollocationValuesPhaseInt[4]);
    # 	printf("\n");
    # 	printf("g0 : %.7f\n",gsl_vector_get(x,0));
    # 	printf("g1 : %.7f\n",gsl_vector_get(x,1));
    # 	printf("g2 : %.7f\n",gsl_vector_get(x,2));
    # 	printf("g3 : %.7f\n",gsl_vector_get(x,3));
    # 	printf("g4 : %.7f\n",gsl_vector_get(x,4));
    # 	printf("\n");
    # 	printf("b0 : %.7f\n",pPhase->b0);
    # 	printf("b1 : %.7f\n",pPhase->b1);
    # 	printf("b2 : %.7f\n",pPhase->b2);
    # 	printf("b3 : %.7f\n",pPhase->b3);
    # 	printf("b4 : %.7f\n",pPhase->b4);
    # 	printf("\n");
    # 	}

    # 	/* Tidy up */
    # 	gsl_vector_free(b);
    # 	gsl_vector_free(x);
    # 	gsl_matrix_free(A);
    # 	gsl_permutation_free(p);
    # }
    # else
    # {
    # 	XLALPrintError("Error in ComputeIMRPhenomXWaveformVariables: IMRPhenomXIntermediatePhaseVersion is not valid.\n");
    # }

    # /* Ringdown coefficients */
    # REAL8 nonGR_dc1   = XLALSimInspiralWaveformParamsLookupNonGRDC1(LALparams);
    # REAL8 nonGR_dc2   = XLALSimInspiralWaveformParamsLookupNonGRDC2(LALparams);
    # REAL8 nonGR_dc4   = XLALSimInspiralWaveformParamsLookupNonGRDC4(LALparams);
    # REAL8 nonGR_dcl   = XLALSimInspiralWaveformParamsLookupNonGRDCL(LALparams);

    # /* Intermediate coefficients */
    # REAL8 nonGR_db1   = XLALSimInspiralWaveformParamsLookupNonGRDB1(LALparams);
    # REAL8 nonGR_db2   = XLALSimInspiralWaveformParamsLookupNonGRDB2(LALparams);
    # REAL8 nonGR_db3   = XLALSimInspiralWaveformParamsLookupNonGRDB3(LALparams);
    # REAL8 nonGR_db4   = XLALSimInspiralWaveformParamsLookupNonGRDB4(LALparams);

    # /* Inspiral coefficients */
    # REAL8 dchi_minus2 = XLALSimInspiralWaveformParamsLookupNonGRDChiMinus2(LALparams);
    # REAL8 dchi_minus1 = XLALSimInspiralWaveformParamsLookupNonGRDChiMinus1(LALparams);
    # REAL8 dchi0       = XLALSimInspiralWaveformParamsLookupNonGRDChi0(LALparams);
    # REAL8 dchi1       = XLALSimInspiralWaveformParamsLookupNonGRDChi1(LALparams);
    # REAL8 dchi2       = XLALSimInspiralWaveformParamsLookupNonGRDChi2(LALparams);
    # REAL8 dchi3       = XLALSimInspiralWaveformParamsLookupNonGRDChi3(LALparams);
    # REAL8 dchi4       = XLALSimInspiralWaveformParamsLookupNonGRDChi4(LALparams);
    # REAL8 dchi5       = XLALSimInspiralWaveformParamsLookupNonGRDChi5(LALparams);
    # REAL8 dchi5L      = XLALSimInspiralWaveformParamsLookupNonGRDChi5L(LALparams);
    # REAL8 dchi6       = XLALSimInspiralWaveformParamsLookupNonGRDChi6(LALparams);
    # REAL8 dchi6L      = XLALSimInspiralWaveformParamsLookupNonGRDChi6L(LALparams);
    # REAL8 dchi7       = XLALSimInspiralWaveformParamsLookupNonGRDChi7(LALparams);

    # /* Can include these terms in the future as desired... */
    # REAL8 dchi8       = 0.0;
    # REAL8 dchi8L      = 0.0;
    # REAL8 dchi9       = 0.0;
    # REAL8 dchi9L      = 0.0;

    # /* ~~~~ RINGDOWN ~~~~ */
    # pPhase->cLGR  = pPhase->cL; // Store GR value for reference
    # pPhase->c1   *= (1.0 + nonGR_dc1);
    # pPhase->c2   *= (1.0 + nonGR_dc2);
    # pPhase->c4   *= (1.0 + nonGR_dc4);
    # pPhase->cL   *= (1.0 + nonGR_dcl);

    # /* Set pre-cached variables */
    # pPhase->c4ov3   = pPhase->c4 / 3.0;
    # pPhase->cLovfda = pPhase->cL / pWF->fDAMP;

    # /* Apply NR tuning for precessing cases (500) */
    # pPhase->b1 = pPhase->b1  +  ( pWF->PNR_DEV_PARAMETER * pWF->ZETA2 );
    # pPhase->b4 = pPhase->b4  +  ( pWF->PNR_DEV_PARAMETER * pWF->ZETA1 );

    # /* ~~~~ INTERMEDIATE ~~~~ */
    # if(pWF->IMRPhenomXIntermediatePhaseVersion == 104)
    # {
    # 	pPhase->b1 *= (1.0 + nonGR_db1);
    # 	pPhase->b2 *= (1.0 + nonGR_db2);
    # 	pPhase->b4 *= (1.0 + nonGR_db4);
    # }
    # else if(pWF->IMRPhenomXIntermediatePhaseVersion == 105)
    # {
    # 	pPhase->b1 *= (1.0 + nonGR_db1);
    # 	pPhase->b2 *= (1.0 + nonGR_db2);
    # 	pPhase->b3 *= (1.0 + nonGR_db3);
    # 	pPhase->b4 *= (1.0 + nonGR_db4);
    # }
    # else
    # {
    # 	XLALPrintError("Error in ComputeIMRPhenomXWaveformVariables: IMRPhenomXIntermediatePhaseVersion is not valid.\n");
    # }

    # /* ~~~~ INSPIRAL ~~~~ */
    # /* Initialize -1PN coefficient*/
    # pPhase->phi_minus2   = 0.0;
    # pPhase->dphi_minus2  = 0.0;

    # pPhase->phi_minus1   = 0.0;
    # pPhase->dphi_minus1  = 0.0;

    # /*
    # 	If tgr_parameterization = 1, deform complete PN coefficient. This is an FTA-like parameterization.
    # 	If tgr_parameterization = 0, only deform non-spinning coefficient. This is the original TIGER-like implementation.
    # */
    # int tgr_parameterization = 0;
    # tgr_parameterization     = XLALSimInspiralWaveformParamsLookupNonGRParameterization(LALparams);

    # if(tgr_parameterization == 1)
    # {
    # 		/* -1.0 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi_minus2 = dchi_minus2 / powers_of_lalpi.two_thirds;

    # 		/* -0.5 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi_minus1 = dchi_minus1 / powers_of_lalpi.one_third;

    # 		/* 0.0 PN */
    # 		pPhase->phi0       = (phi0NS + phi0S)*(1.0 + dchi0);

    # 		/* 0.5 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi1       = dchi1 * powers_of_lalpi.one_third;

    # 		/* 1.0 PN */
    # 		pPhase->phi2       = (phi2NS + phi2S)*(1.0 + dchi2);

    # 		/* 1.5 PN */
    # 		pPhase->phi3       = (phi3NS + phi3S)*(1.0 + dchi3);

    # 		/* 2.0 PN */
    # 		pPhase->phi4       = (phi4NS + phi4S)*(1.0 + dchi4);

    # 		/* 2.5 PN */
    # 		pPhase->phi5       = (phi5NS + phi5S)*(1.0 + dchi5);

    # 		/* 2.5 PN, Log Terms */
    # 		pPhase->phi5L      = (phi5LNS + phi5LS)*(1.0 + dchi5L);

    # 		/* 3.0 PN */
    # 		pPhase->phi6       = (phi6NS + phi6S)*(1.0 + dchi6);

    # 		/* 3.0 PN, Log Term */
    # 		pPhase->phi6L      = (phi6LNS + phi6LS)*(1.0 + dchi6L);

    # 		/* 3.5PN */
    # 		pPhase->phi7       = (phi7NS + phi7S)*(1.0 + dchi7);

    # 		/* 4.0PN */
    # 		pPhase->phi8       = (phi8NS + phi8S)*(1.0 + dchi8);

    # 		/* 4.0 PN, Log Terms */
    # 		pPhase->phi8L      = (phi8LNS + phi8LS)*(1.0 + dchi8L);

    # 		/* 4.0 PN */
    # 		pPhase->phi9       = (phi9NS + phi9S)*(1.0 + dchi9);

    # 		/* 4.0 PN, Log Terms */
    # 		pPhase->phi9L      = (phi9LNS + phi9LS)*(1.0 + dchi9L);
    # }
    # else if(tgr_parameterization == 0)
    # {
    # 		/* -1.0 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi_minus2 = dchi_minus2 / powers_of_lalpi.two_thirds;

    # 		/* -0.5 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi_minus1 = dchi_minus1 / powers_of_lalpi.one_third;

    # 		/* 0.0 PN */
    # 		pPhase->phi0       = phi0NS*(1.0 + dchi0) + phi0S;

    # 		/* 0.5 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi1       = dchi1 * powers_of_lalpi.one_third;

    # 		/* 1.0 PN */
    # 		pPhase->phi2       = phi2NS*(1.0 + dchi2) + phi2S;

    # 		/* 1.5 PN */
    # 		pPhase->phi3       = phi3NS*(1.0 + dchi3)+ phi3S;

    # 		/* 2.0 PN */
    # 		pPhase->phi4       = phi4NS*(1.0 + dchi4) + phi4S;

    # 		/* 2.5 PN */
    # 		pPhase->phi5       = phi5NS*(1.0 + dchi5) + phi5S;

    # 		/* 2.5 PN, Log Terms */
    # 		pPhase->phi5L      = phi5LNS*(1.0 + dchi5L) + phi5LS;

    # 		/* 3.0 PN */
    # 		pPhase->phi6       = phi6NS*(1.0 + dchi6) + phi6S;

    # 		/* 3.0 PN, Log Term */
    # 		pPhase->phi6L      = phi6LNS*(1.0 + dchi6L) + phi6LS;

    # 		/* 3.5PN */
    # 		pPhase->phi7       = phi7NS*(1.0 + dchi7) + phi7S;

    # 		/* 4.0PN */
    # 		pPhase->phi8       = phi8NS*(1.0 + dchi8) + phi8S;

    # 		/* 4.0 PN, Log Terms */
    # 		pPhase->phi8L      = phi8LNS*(1.0 + dchi8L) + phi8LS;

    # 		/* 4.0 PN */
    # 		pPhase->phi9       = phi9NS*(1.0 + dchi9) + phi9S;

    # 		/* 4.0 PN, Log Terms */
    # 		pPhase->phi9L      = phi9LNS*(1.0 + dchi9L) + phi9LS;
    # }
    # else
    # {
    # 		XLALPrintError("Error in IMRPhenomXGetPhaseCoefficients: TGR Parameterizataion is not valid.\n");
    # }

    # /* Recalculate phase derivatives including TGR corrections */
    # pPhase->dphi_minus2 = +(7.0 / 5.0) * pPhase->phi_minus2;
    # pPhase->dphi_minus1 = +(6.0 / 5.0) * pPhase->phi_minus1;
    # pPhase->dphi0       = +(5.0 / 5.0) * pPhase->phi0;
    # pPhase->dphi1       = +(4.0 / 5.0) * pPhase->phi1;
    # pPhase->dphi2       = +(3.0 / 5.0) * pPhase->phi2;
    # pPhase->dphi3       = +(2.0 / 5.0) * pPhase->phi3;
    # pPhase->dphi4       = +(1.0 / 5.0) * pPhase->phi4;
    # pPhase->dphi5       = -(3.0 / 5.0) * pPhase->phi5L;
    # pPhase->dphi6       = -(1.0 / 5.0) * pPhase->phi6 - (3.0 / 5.0) * pPhase->phi6L;
    # pPhase->dphi6L      = -(1.0 / 5.0) * pPhase->phi6L;
    # pPhase->dphi7       = -(2.0 / 5.0) * pPhase->phi7;
    # pPhase->dphi8       = -(3.0 / 5.0) * pPhase->phi8 - (3.0 / 5.0) * pPhase->phi8L;
    # pPhase->dphi8L      = -(3.0 / 5.0) * pPhase->phi8L;
    # pPhase->dphi9       = -(4.0 / 5.0) * pPhase->phi9 - (3.0 / 5.0) * pPhase->phi9L;
    # pPhase->dphi9L      = -(3.0 / 5.0) * pPhase->phi9L;

    # /* Initialize connection coefficients */
    # pPhase->C1Int = 0;
    # pPhase->C2Int = 0;
    # pPhase->C1MRD = 0;
    # pPhase->C2MRD = 0;

    return p_wf, p_phase
