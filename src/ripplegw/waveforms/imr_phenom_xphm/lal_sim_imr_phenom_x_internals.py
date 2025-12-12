"""Helper functions for IMRPhenomXPHM waveform model."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw import ms_to_Mc_eta
from ripplegw.constants import MRSUN, PI, gt
from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.lal_constants import LAL_GAMMA, LAL_MSUN_SI
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_inspiral import (
    imr_phenom_x_inspiral_phase_22_d13,
    imr_phenom_x_inspiral_phase_22_d23,
    imr_phenom_x_inspiral_phase_22_d43,
    imr_phenom_x_inspiral_phase_22_d53,
    imr_phenom_x_inspiral_phase_22_v3,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_intermediate import (
    imr_phenom_x_intermediate_phase_22_d43,
    imr_phenom_x_intermediate_phase_22_v2,
    imr_phenom_x_intermediate_phase_22_v2m_rd_v4,
    imr_phenom_x_intermediate_phase_22_v3m_rd_v4,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXPhaseCoefficientsDataClass,
    IMRPhenomXUsefulPowersDataClass,
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_qnm import (
    evaluate_QNMfit_fdamp22,
    evaluate_QNMfit_fring22,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_ringdown import (
    imr_phenom_x_ringdown_phase_22_d12,
    imr_phenom_x_ringdown_phase_22_d24,
    imr_phenom_x_ringdown_phase_22_d34,
    imr_phenom_x_ringdown_phase_22_d54,
    imr_phenom_x_ringdown_phase_22_v4,
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
def imr_phenom_x_initialize_powers(number: float | Array) -> IMRPhenomXUsefulPowersDataClass:
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


@checkify.checkify
def imr_phenom_x_get_phase_coefficients(
    p_wf: IMRPhenomXWaveformDataClass, p_phase: IMRPhenomXPhaseCoefficientsDataClass
) -> IMRPhenomXPhaseCoefficientsDataClass:
    """
    Populate IMRPhenomXPhaseCoefficientsDataClass
    https://lscsoft.docs.ligo.org/lalsuite//lalsimulation/_l_a_l_sim_i_m_r_phenom_x__internals_8c.html#aca4208ae52d217747aa723953850765a
    """

    _, powers_of_lalpi = imr_phenom_x_initialize_powers(PI)

    # /* Get LALparams */
    # lal_params = p_wf.lal_params

    # /* GSL objects for solving system of equations via LU decomposition */
    gpoints4 = jnp.array([0.0, 1.0 / 4.0, 3.0 / 4.0, 1.0])
    gpoints5 = jnp.array(
        [0.0, 1.0 / 2 - 1.0 / (2 * jnp.sqrt(2.0)), 1.0 / 2.0, 1.0 / 2 + 1.0 / (2.0 * jnp.sqrt(2.0)), 1.0]
    )

    # // Matching regions

    # /* This is Eq. 5.11 in the paper */
    f_im_match = 0.6 * (0.5 * p_wf.f_ring + p_wf.f_isco)

    # /* This is the MECO frequency */
    f_in_match = p_wf.f_meco

    # /* This is Eq. 5.10 in the paper */
    deltaf = (f_im_match - f_in_match) * 0.03

    # // Transition frequency is just below the MECO frequency and just above the RD fitting region

    # /* These are defined in Eq. 7.7 and the text just below, f_H = fPhaseMatchIM and f_L = fPhaseMatchIN */
    f_phase_match_in = f_in_match - 1.0 * deltaf
    f_phase_match_im = f_im_match + 0.5 * deltaf

    # /* Defined in Eq. 7.4, this is f_L */
    f_phase_ins_min = 0.0026

    # /* Defined in Eq. 7.4, this is f_H */
    f_phase_ins_max = 1.020 * p_wf.f_meco

    # /* Defined in Eq. 7.12, this is f_L */
    f_phase_rd_min = f_im_match

    # /* Defined in Eq. 7.12, this is f_L */
    f_phase_rd_max = p_wf.f_ring + 1.25 * p_wf.f_damp
    # pPhase->phiNorm = -(3.0 * powers_of_lalpi.m_five_thirds) / (128.0)
    phi_norm = -(3.0 * powers_of_lalpi.m_five_thirds) / (128.0)

    # /* For convenience, define some variables here */
    chi1l = p_wf.chi1l
    chi2l = p_wf.chi2l

    chi1l2l = chi1l * chi2l

    chi1l2 = p_wf.chi1l * p_wf.chi1l
    chi1l3 = p_wf.chi1l * chi1l2

    chi2l2 = p_wf.chi2l * p_wf.chi2l
    chi2l3 = p_wf.chi2l * chi2l2

    eta = p_wf.eta
    eta2 = eta * eta
    eta3 = eta * eta2

    delta = p_wf.delta

    # /*
    # 	The general strategy is to initialize a linear system of equations:

    # 	A.x = b

    # 	- A is a matrix with the coefficients of the ansatz evaluated at the collocation nodes.
    # 	- b is a vector of the value of the collocation points
    # 	- x is the solution vector (i.e. the coefficients) that we must solve for.

    # 	We choose to do this using a standard LU decomposition.
    # */

    # Generate list of collocation points
    #
    # The Gauss-Chebyshev Points are given by:
    # GCPoints[n] = Table[Cos[i * pi/n] , {i, 0, n}]

    # SamplePoints[xmin,xmax,n] = {
    # 	pWF->delta = xmax - xmin
    # 	gpoints = 0.5 * (1 + GCPoints[n-1])

    # 	return {xmin + pWF->delta * gpoints}
    # }

    # gpoints4 = [1.0, 3.0/4, 1.0/4, 0.0]
    # gpoints5 = [1.0, 1.0/2 + 1.0/(2.0*sqrt(2.0)), 1.0/2, 1.0/2 - 1.0/(2*sqrt(2.0)), 0.]

    # */

    # /*
    # Ringdown phase collocation points:
    # Here we only use the first N+1 points in the array where N = the
    # number of pseudo PN terms.

    # The size of the array is controlled by: N_MAX_COLLOCATION_POINTS_PHASE_RD

    # Default is to use 5 collocation points.
    # */

    deltax = f_phase_rd_max - f_phase_rd_min
    xmin = f_phase_rd_min

    # // Initialize collocation points
    collocation_points_phase_rd = gpoints5 * deltax + xmin

    # // Collocation point 4 is set to the ringdown frequency ~ dip in Lorentzian
    collocation_points_phase_rd = collocation_points_phase_rd.at[3].set(p_wf.f_ring)

    # switch(pWF->IMRPhenomXRingdownPhaseVersion)
    # {
    # 	case 105:
    # 	{
    # 		pPhase->NCollocationPointsRD = 5
    # 		break
    # 	}
    # 	default:
    # 	{
    # 		XLAL_ERROR(XLAL_EINVAL, "Error: IMRPhenomXRingdownPhaseVersion is not valid.\n")
    # 	}
    # }

    n_collocation_points_rd = jax.lax.select(p_wf.imr_phenom_x_ringdown_phase_version == 105, 5, 0)
    checkify.check(
        jnp.logical_and(p_wf.imr_phenom_x_ringdown_phase_version == 105, n_collocation_points_rd == 5),
        "Error: NCollocationPointsRD must be 5.",
    )

    # // Eq. 7.13 in arXiv:2001.11412
    rd_v4 = imr_phenom_x_ringdown_phase_22_v4(
        p_wf.eta, p_wf.s_tot_r, p_wf.dchi, p_wf.delta, p_wf.imr_phenom_x_ringdown_phase_version
    )

    # /* These are the calibrated collocation points, as per Eq. 7.13 */
    collocation_values_phase_rd = jnp.array(
        [
            imr_phenom_x_ringdown_phase_22_d12(
                p_wf.eta, p_wf.s_tot_r, p_wf.dchi, p_wf.delta, p_wf.imr_phenom_x_ringdown_phase_version
            ),
            imr_phenom_x_ringdown_phase_22_d24(
                p_wf.eta, p_wf.s_tot_r, p_wf.dchi, p_wf.delta, p_wf.imr_phenom_x_ringdown_phase_version
            ),
            imr_phenom_x_ringdown_phase_22_d34(
                p_wf.eta, p_wf.s_tot_r, p_wf.dchi, p_wf.delta, p_wf.imr_phenom_x_ringdown_phase_version
            ),
            rd_v4,
            imr_phenom_x_ringdown_phase_22_d54(
                p_wf.eta, p_wf.s_tot_r, p_wf.dchi, p_wf.delta, p_wf.imr_phenom_x_ringdown_phase_version
            ),
        ]
    )

    # /* v_j = d_{j4} + v4 */
    # pPhase->CollocationValuesPhaseRD[4] = pPhase->CollocationValuesPhaseRD[4] + pPhase->CollocationValuesPhaseRD[3] #// v5 = d54  + v4
    # pPhase->CollocationValuesPhaseRD[2] = pPhase->CollocationValuesPhaseRD[2] + pPhase->CollocationValuesPhaseRD[3] #// v3 = d34  + v4
    # pPhase->CollocationValuesPhaseRD[1] = pPhase->CollocationValuesPhaseRD[1] + pPhase->CollocationValuesPhaseRD[3] #// v2 = d24  + v4
    # pPhase->CollocationValuesPhaseRD[0] = pPhase->CollocationValuesPhaseRD[0] + pPhase->CollocationValuesPhaseRD[1] #// v1 = d12  + v2

    # First update indices 4, 2, 1 by adding value at index 3
    collocation_values_phase_rd = collocation_values_phase_rd.at[4].add(collocation_values_phase_rd[3])
    collocation_values_phase_rd = collocation_values_phase_rd.at[2].add(collocation_values_phase_rd[3])
    collocation_values_phase_rd = collocation_values_phase_rd.at[1].add(collocation_values_phase_rd[3])
    # Then update index 0 by adding the updated value at index 1
    collocation_values_phase_rd = collocation_values_phase_rd.at[0].add(collocation_values_phase_rd[1])

    phase_rd = collocation_values_phase_rd[0]

    # p = gsl_permutation_alloc(pPhase->NCollocationPointsRD)
    # b = gsl_vector_alloc(pPhase->NCollocationPointsRD)
    # x = gsl_vector_alloc(pPhase->NCollocationPointsRD)
    # A = gsl_matrix_alloc(pPhase->NCollocationPointsRD,pPhase->NCollocationPointsRD)

    # /*
    # Populate the b vector
    # */

    b = collocation_values_phase_rd.copy()

    # /*
    # 		Eq. 7.12 in arXiv:2001.11412

    # 		ansatzRD(f) = a_0 + a_1 f^(-1/3) + a_2 f^(-2) + a_3 f^(-3) + a_4 f^(-4) + ( aRD ) / ( (f_damp^2 + (f - f_ring)^2 ) )

    # 		Canonical ansatz sets a_3 to 0.
    # */

    # /*
    # 	We now set up and solve a linear system of equations.
    # 	First we populate the matrix A_{ij}
    # */

    ff = collocation_points_phase_rd
    invff = 1.0 / ff
    ff1 = jnp.cbrt(invff)  # f^{-1/3}
    ff2 = invff * invff  # // f^{-2}
    ff3 = ff2 * ff2  # // f^{-4}
    ff4 = -(p_wf.dphase0) / (p_wf.f_damp * p_wf.f_damp + (ff - p_wf.f_ring) * (ff - p_wf.f_ring))

    a_matrix = jnp.array([ff, ff1, ff2, ff3, ff4]).T

    # /* We now solve the system A x = b via an LU decomposition */
    x = jax.scipy.linalg.lu_solve(jax.scipy.linalg.lu_factor(a_matrix), b)

    c0 = x[0]  # // x[0] 	// a0
    c1 = x[1]  # // x[1]		// a1
    c2 = x[2]  # // x[2] 	// a2
    c4 = x[3]  # // x[3] 	// a4
    c_rd = x[4]
    c_l = -(p_wf.dphase0 * c_rd)  # // cL = - a_{RD} * dphase0

    # /* Apply NR tuning for precessing cases (500) */
    c_l += p_wf.pnr_dev_parameter * p_wf.nu4

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
    deltax = f_phase_ins_max - f_phase_ins_min
    xmin = f_phase_ins_min

    # /*
    # 		Set number of pseudo-PN coefficients:
    # 			- If you add a new PN inspiral approximant, update with new version here.
    # */

    valid_versions = jnp.array([104, 105, 114, 115])
    check_inspiral_phase_version = jnp.isin(p_wf.imr_phenom_x_inspiral_phase_version, valid_versions)
    checkify.check(check_inspiral_phase_version, "Error: IMRPhenomXInspiralPhaseVersion is not valid.")

    # Version 104 and 114 use 4 pseudo-PN coefficients and 4 collocation points
    # Version 105 and 115 use 5 pseudo-PN coefficients and 5 collocation points
    is_version_4 = jnp.logical_or(
        p_wf.imr_phenom_x_inspiral_phase_version == 104, p_wf.imr_phenom_x_inspiral_phase_version == 114
    )
    n_pseudo_pn = jax.lax.select(is_version_4, 4, 5)
    n_collocation_points_phase_ins = jax.lax.select(is_version_4, 4, 5)

    checkify.check(
        jnp.logical_or(n_pseudo_pn == 4, n_pseudo_pn == 5),
        "Error in imr_phenom_x_get_phase_coefficients: NPseudoPN requested is not valid. Number of pseudo PN coefficients must be 4 or 5.",
    )

    # /*
    # If we are using 4 pseudo-PN coefficients, call the routines below.
    # The inspiral phase version is still passed to the individual functions.
    # */
    def n_pseudo_pn_4_branch(p_phase: IMRPhenomXPhaseCoefficientsDataClass) -> IMRPhenomXPhaseCoefficientsDataClass:
        # // By default all models implemented use the following GC points.
        # // If a new model is calibrated with different choice of collocation points, edit this.
        collocation_points_phase_ins = gpoints4 * deltax + xmin

        # // Calculate the value of the differences between the ith and 3rd collocation points at the GC nodes
        collocation_values_phase_ins = jnp.array(
            [
                imr_phenom_x_inspiral_phase_22_d13(eta, chi1l, chi2l, delta, p_wf.imr_phenom_x_inspiral_phase_version),
                imr_phenom_x_inspiral_phase_22_d23(eta, chi1l, chi2l, delta, p_wf.imr_phenom_x_inspiral_phase_version),
                imr_phenom_x_inspiral_phase_22_v3(eta, chi1l, chi2l, delta, p_wf.imr_phenom_x_inspiral_phase_version),
                imr_phenom_x_inspiral_phase_22_d43(eta, chi1l, chi2l, delta, p_wf.imr_phenom_x_inspiral_phase_version),
            ]
        )

        # // Calculate the value of the collocation points at GC nodes via: v_i = d_i3 + v3
        collocation_values_phase_ins = collocation_values_phase_ins.at[0].add(collocation_values_phase_ins[2])
        collocation_values_phase_ins = collocation_values_phase_ins.at[1].add(collocation_values_phase_ins[2])
        collocation_values_phase_ins = collocation_values_phase_ins.at[3].add(collocation_values_phase_ins[2])

        b = collocation_values_phase_ins.copy()

        ff = collocation_points_phase_ins  # jax.lax.dynamic_slice(
        #     collocation_points_phase_ins,
        #     (0,),  # start index
        #     (4,)   # slice size
        # )
        ff1 = jnp.cbrt(ff)
        ff2 = ff1 * ff1
        ff3 = ff

        a_matrix = jnp.array([ff, ff1, ff2, ff3]).T

        # /* We now solve the system A x = b via an LU decomposition */
        x = jax.scipy.linalg.lu_solve(jax.scipy.linalg.lu_factor(a_matrix), b)

        # /* Set inspiral phenomenological coefficients from solution to A x = b */
        a0 = x[0]  # // x[0] // alpha_0
        a1 = x[1]  # // x[1] // alpha_1
        a2 = x[2]  # // x[2] // alpha_2
        a3 = x[3]  # // x[3] // alpha_3
        a4 = 0.0

        # /*
        #         PSEUDO PN TERMS WORK:
        #             - 104 works.
        #             - 105 not tested.
        #             - 114 not tested.
        #             - 115 not tested.
        # */

        p_phase = dataclasses.replace(
            p_phase,
            collocation_points_phase_ins=jnp.pad(
                collocation_points_phase_ins, (0, 1), constant_values=0.0
            ),  # zero pad for jit compatibility
            collocation_values_phase_ins=jnp.pad(
                collocation_values_phase_ins, (0, 1), constant_values=0.0
            ),  # zero pad for jit compatibility
            a0=a0,
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
        )
        return p_phase

    def n_pseudo_pn_5_branch(p_phase: IMRPhenomXPhaseCoefficientsDataClass) -> IMRPhenomXPhaseCoefficientsDataClass:
        # // Using 5 pseudo-PN coefficients so set 5 collocation points
        collocation_points_phase_ins = gpoints5 * deltax + xmin

        collocation_values_phase_ins = jnp.array(
            [
                imr_phenom_x_inspiral_phase_22_d13(eta, chi1l, chi2l, delta, p_wf.imr_phenom_x_inspiral_phase_version),
                imr_phenom_x_inspiral_phase_22_d23(eta, chi1l, chi2l, delta, p_wf.imr_phenom_x_inspiral_phase_version),
                imr_phenom_x_inspiral_phase_22_v3(eta, chi1l, chi2l, delta, p_wf.imr_phenom_x_inspiral_phase_version),
                imr_phenom_x_inspiral_phase_22_d43(eta, chi1l, chi2l, delta, p_wf.imr_phenom_x_inspiral_phase_version),
                imr_phenom_x_inspiral_phase_22_d53(eta, chi1l, chi2l, delta, p_wf.imr_phenom_x_inspiral_phase_version),
            ]
        )

        # /* v_j = d_j3 + v_3 */
        collocation_values_phase_ins = collocation_values_phase_ins.at[0].add(collocation_values_phase_ins[2])
        collocation_values_phase_ins = collocation_values_phase_ins.at[1].add(collocation_values_phase_ins[2])
        collocation_values_phase_ins = collocation_values_phase_ins.at[3].add(collocation_values_phase_ins[2])
        collocation_values_phase_ins = collocation_values_phase_ins.at[4].add(collocation_values_phase_ins[2])

        b = collocation_values_phase_ins.copy()

        ff = collocation_points_phase_ins
        ff1 = jnp.cbrt(ff)
        ff2 = ff1 * ff1
        ff3 = ff
        ff4 = ff * ff1

        a_matrix = jnp.array([ff, ff1, ff2, ff3, ff4]).T

        # /* We now solve the system A x = b via an LU decomposition */
        x = jax.scipy.linalg.lu_solve(jax.scipy.linalg.lu_factor(a_matrix), b)

        # /* Set inspiral phenomenological coefficients from solution to A x = b */
        a0 = x[0]  # // x[0] // alpha_0
        a1 = x[1]  # // x[1] // alpha_1
        a2 = x[2]  # // x[2] // alpha_2
        a3 = x[3]  # // x[3] // alpha_3
        a4 = x[4]

        p_phase = dataclasses.replace(
            p_phase,
            collocation_points_phase_ins=collocation_points_phase_ins,
            collocation_values_phase_ins=collocation_values_phase_ins,
            a0=a0,
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
        )
        return p_phase

    p_phase = jax.lax.cond(
        n_pseudo_pn == 4, lambda x: n_pseudo_pn_4_branch(x), lambda x: n_pseudo_pn_5_branch(x), operand=p_phase
    )

    # Note: When n_pseudo_pn == 4, the arrays are padded with a trailing zero.
    # This padding doesn't affect the computation since the extra element is unused.

    # The pseudo-PN coefficients are normalized such that: (dphase0 / eta) * f^{8/3} * a_j
    # So we must re-scale these terms by an extra factor of f^{-8/3} in the PN phasing
    sigma1 = (-5.0 / 3.0) * p_phase.a0
    sigma2 = (-5.0 / 4.0) * p_phase.a1
    sigma3 = (-5.0 / 5.0) * p_phase.a2
    sigma4 = (-5.0 / 6.0) * p_phase.a3
    sigma5 = (-5.0 / 7.0) * p_phase.a4

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

    # /* Analytically known PN coefficients */
    # /* Newtonian */
    phi0_ns = 1.0

    # /* ~~ 0.5 PN ~~ */
    phi1_ns = 0.0

    # /* ~~ 1.0 PN ~~ */
    # /* 1.0PN, Non-Spinning */
    phi2_ns = (3715 / 756.0 + (55 * eta) / 9.0) * powers_of_lalpi.two_thirds
    # /* ~~ 1.5 PN ~~ */
    # /* 1.5PN, Non-Spinning */
    phi3_ns = -16.0 * powers_of_lalpi.two
    # /* 1.5PN, Spin-Orbit */
    phi3_s = (
        (113 * (chi1l + chi2l + chi1l * delta - chi2l * delta) - 76 * (chi1l + chi2l) * eta) / 6.0
    ) * powers_of_lalpi.itself
    phi4_ns = (15293365 / 508032.0 + (27145 * eta) / 504.0 + (3085 * eta2) / 72.0) * powers_of_lalpi.four_thirds
    # /* 2.0PN, Spin-Spin */
    phi4_s = (
        (-5 * (81 * chi1l2 * (1 + delta - 2 * eta) + 316 * chi1l2l * eta - 81 * chi2l2 * (-1 + delta + 2 * eta))) / 16.0
    ) * powers_of_lalpi.four_thirds

    # /* ~~ 2.5 PN ~~ */
    phi5_ns = 0.0
    phi5_s = 0.0

    # /* ~~ 2.5 PN, Log Term ~~ */
    # /* 2.5PN, Non-Spinning */
    phi5_lns = ((5 * (46374 - 6552 * eta) * powers_of_lalpi.itself) / 4536.0) * powers_of_lalpi.five_thirds
    # /* 2.5PN, Spin-Orbit */
    phi5_ls = (
        (
            -732985 * (chi1l + chi2l + chi1l * delta - chi2l * delta)
            - 560 * (-1213 * (chi1l + chi2l) + 63 * (chi1l - chi2l) * delta) * eta
            + 85680 * (chi1l + chi2l) * eta2
        )
        / 4536.0
    ) * powers_of_lalpi.five_thirds

    # /* ~~ 3.0 PN ~~ */
    # /* 3.0 PN, Non-Spinning */
    phi6_ns = (
        11583231236531 / 4.69421568e9
        - (5 * eta * (3147553127 + 588 * eta * (-45633 + 102260 * eta))) / 3.048192e6
        - (6848 * LAL_GAMMA) / 21.0
        - (640 * powers_of_lalpi.two) / 3.0
        + (2255 * eta * powers_of_lalpi.two) / 12.0
        - (13696 * jnp.log(2)) / 21.0
        - (6848 * powers_of_lalpi.log) / 63.0
    ) * powers_of_lalpi.two
    # /* 3.0 PN, Spin-Orbit */
    phi6_s = (
        (
            5
            * (227 * (chi1l + chi2l + chi1l * delta - chi2l * delta) - 156 * (chi1l + chi2l) * eta)
            * powers_of_lalpi.itself
        )
        / 3.0
    ) * powers_of_lalpi.two
    # /* 3.0 PN, Spin-Spin */
    phi6_s += (
        (
            5
            * (
                20 * chi1l2l * eta * (11763 + 12488 * eta)
                + 7 * chi2l2 * (-15103 * (-1 + delta) + 2 * (-21683 + 6580 * delta) * eta - 9808 * eta2)
                - 7 * chi1l2 * (-15103 * (1 + delta) + 2 * (21683 + 6580 * delta) * eta + 9808 * eta2)
            )
        )
        / 4032.0
    ) * powers_of_lalpi.two

    # /* ~~ 3.0 PN, Log Term ~~ */
    phi6_lns = (-6848 / 63.0) * powers_of_lalpi.two
    phi6_ls = 0.0

    # /* ~~ 3.5 PN ~~ */
    # /* 3.5 PN, Non-Spinning */
    phi7_ns = (
        (5 * (15419335 + 168 * (75703 - 29618 * eta) * eta) * powers_of_lalpi.itself) / 254016.0
    ) * powers_of_lalpi.seven_thirds
    # /* 3.5 PN, Spin-Orbit */
    phi7_s = (
        (
            5
            * (
                -5030016755 * (chi1l + chi2l + chi1l * delta - chi2l * delta)
                + 4 * (2113331119 * (chi1l + chi2l) + 675484362 * (chi1l - chi2l) * delta) * eta
                - 1008 * (208433 * (chi1l + chi2l) + 25011 * (chi1l - chi2l) * delta) * eta2
                + 90514368 * (chi1l + chi2l) * eta3
            )
        )
        / 6.096384e6
    ) * powers_of_lalpi.seven_thirds
    # /* 3.5 PN, Spin-Spin */
    phi7_s += (
        -5
        * (57 * chi1l2 * (1 + delta - 2 * eta) + 220 * chi1l2l * eta - 57 * chi2l2 * (-1 + delta + 2 * eta))
        * powers_of_lalpi.itself
    ) * powers_of_lalpi.seven_thirds
    # /* 3.5 PN, Cubic-in-Spin */
    phi7_s += (
        (
            14585 * (-(chi2l3 * (-1 + delta)) + chi1l3 * (1 + delta))
            - 5
            * (
                chi2l3 * (8819 - 2985 * delta)
                + 8439 * chi1l * chi2l2 * (-1 + delta)
                - 8439 * chi1l2 * chi2l * (1 + delta)
                + chi1l3 * (8819 + 2985 * delta)
            )
            * eta
            + 40 * (chi1l + chi2l) * (17 * chi1l2 - 14 * chi1l2l + 17 * chi2l2) * eta2
        )
        / 48.0
    ) * powers_of_lalpi.seven_thirds

    # /* ~~ 4.0 PN ~~ */
    # /* 4.0 PN, Non-Spinning */
    phi8_ns = 0.0
    # /* 4.0 PN, Spin-Orbit */
    phi8_s = (
        (
            -5
            * (
                1263141 * (chi1l + chi2l + chi1l * delta - chi2l * delta)
                - 2 * (794075 * (chi1l + chi2l) + 178533 * (chi1l - chi2l) * delta) * eta
                + 94344 * (chi1l + chi2l) * eta2
            )
            * powers_of_lalpi.itself
            * (-1 + powers_of_lalpi.log)
        )
        / 9072.0
    ) * powers_of_lalpi.eight_thirds

    # /* ~~ 4.0 PN, Log Term ~~ */
    # /* 4.0 PN, log term, Non-Spinning */
    phi8_lns = 0.0
    # /* 4.0 PN, log term, Spin-Orbit */
    phi8_ls = (
        (
            -5
            * (
                1263141 * (chi1l + chi2l + chi1l * delta - chi2l * delta)
                - 2 * (794075 * (chi1l + chi2l) + 178533 * (chi1l - chi2l) * delta) * eta
                + 94344 * (chi1l + chi2l) * eta2
            )
            * powers_of_lalpi.itself
        )
        / 9072.0
    ) * powers_of_lalpi.eight_thirds

    # /* ~~ 4.5 PN ~~ */
    phi9_ns = 0.0
    phi9_s = 0.0

    # /* ~~ 4.5 PN, Log Term ~~ */
    phi9_lns = 0.0
    phi9_ls = 0.0

    # This version of TaylorF2 contains an additional 4.5PN tail term and a LO-SS tail term at 3.5PN
    def tail_term_branch(phi7_s, phi9_ns, phi9_lns):
        # /* 3.5PN, Leading Order Spin-Spin Tail Term */
        phi7_s += (
            (
                5
                * (65 * chi1l2 * (1 + delta - 2 * eta) + 252 * chi1l2l * eta - 65 * chi2l2 * (-1 + delta + 2 * eta))
                * powers_of_lalpi.itself
            )
            / 4.0
        ) * powers_of_lalpi.seven_thirds

        # /* 4.5PN, Tail Term */
        phi9_ns += (
            (5 * (-256 + 451 * eta) * powers_of_lalpi.three) / 6.0
            + (
                powers_of_lalpi.itself
                * (
                    105344279473163
                    + 700 * eta * (-298583452147 + 96 * eta * (99645337 + 14453257 * eta))
                    - 12246091038720 * LAL_GAMMA
                    - 24492182077440 * jnp.log(2.0)
                )
            )
            / 1.877686272e10
            - (13696 * powers_of_lalpi.itself * powers_of_lalpi.log) / 63.0
        ) * powers_of_lalpi.three

        # /* 4.5PN, Log Term */
        phi9_lns += ((-13696 * powers_of_lalpi.itself) / 63.0) * powers_of_lalpi.three

        return phi7_s, phi9_ns, phi9_lns

    phi7_s, phi9_ns, phi9_lns = jax.lax.cond(
        jnp.logical_or(
            p_wf.imr_phenom_x_inspiral_phase_version == 114, p_wf.imr_phenom_x_inspiral_phase_version == 115
        ),
        lambda args: tail_term_branch(*args),
        lambda args: args,
        operand=(phi7_s, phi9_ns, phi9_lns),
    )

    # /* 0.0 PN */
    phi0 = phi0_ns

    # /* 0.5 PN */
    phi1 = phi1_ns

    # /* 1.0 PN */
    phi2 = phi2_ns

    # /* 1.5 PN */
    phi3 = phi3_ns + phi3_s

    # /* 2.0 PN */
    phi4 = phi4_ns + phi4_s

    # /* 2.5 PN */
    phi5 = phi5_ns + phi5_s

    # /* 2.5 PN, Log Terms */
    phi5l = phi5_lns + phi5_ls

    # /* 3.0 PN */
    phi6 = phi6_ns + phi6_s

    # /* 3.0 PN, Log Term */
    phi6l = phi6_lns + phi6_ls

    # /* 3.5PN */
    phi7 = phi7_ns + phi7_s

    # /* 4.0PN */
    phi8 = phi8_ns + phi8_s

    # /* 4.0 PN, Log Terms */
    phi8l = phi8_lns + phi8_ls

    # /* 4.5 PN */
    phi9 = phi9_ns + phi9_s

    # /* 4.5 PN, Log Terms */
    phi9l = phi9_lns + phi9_ls

    phi_initial = -PI / 4.0

    # /* **** TaylorF2 PN Coefficients: Normalized Phase Derivative **** */
    dphi0 = phi0
    dphi1 = 4.0 / 5.0 * phi1
    dphi2 = 3.0 / 5.0 * phi2
    dphi3 = 2.0 / 5.0 * phi3
    dphi4 = 1.0 / 5.0 * phi4
    dphi5 = -3.0 / 5.0 * phi5l
    dphi6 = -1.0 / 5.0 * phi6 - 3.0 / 5.0 * phi6l
    dphi6l = -1.0 / 5.0 * phi6l
    dphi7 = -2.0 / 5.0 * phi7
    dphi8 = -3.0 / 5.0 * phi8 - 3.0 / 5.0 * phi8l
    dphi8l = -3.0 / 5.0 * phi8l
    dphi9 = -4.0 / 5.0 * phi9 - 3.0 / 5.0 * phi9l
    dphi9l = -3.0 / 5.0 * phi9l

    # /*
    # 		Calculate phase at fmatchIN. This will be used as the collocation point for the intermediate fit.
    # 		In practice, the transition point is just below the MECO frequency.
    # */
    _, powers_of_fmatch_in = imr_phenom_x_initialize_powers(f_phase_match_in)

    phase_in = dphi0  # f^{0/3}
    phase_in += dphi1 * powers_of_fmatch_in.one_third  # f^{1/3}
    phase_in += dphi2 * powers_of_fmatch_in.two_thirds  # f^{2/3}
    phase_in += dphi3 * powers_of_fmatch_in.itself  # f^{3/3}
    phase_in += dphi4 * powers_of_fmatch_in.four_thirds  # f^{4/3}
    phase_in += dphi5 * powers_of_fmatch_in.five_thirds  # f^{5/3}
    phase_in += dphi6 * powers_of_fmatch_in.two  # f^{6/3}
    phase_in += dphi6 * powers_of_fmatch_in.two * powers_of_fmatch_in.log  # f^{6/3}, Log[f]
    phase_in += dphi7 * powers_of_fmatch_in.seven_thirds  # f^{7/3}
    phase_in += dphi8 * powers_of_fmatch_in.eight_thirds  # f^{8/3}
    phase_in += dphi8 * powers_of_fmatch_in.eight_thirds * powers_of_fmatch_in.log  # f^{8/3}
    phase_in += dphi9 * powers_of_fmatch_in.three  # f^{9/3}
    phase_in += dphi9 * powers_of_fmatch_in.three * powers_of_fmatch_in.log  # f^{9/3}

    # // Add pseudo-PN Coefficient
    phase_in += (
        p_phase.a0 * powers_of_fmatch_in.eight_thirds
        + p_phase.a1 * powers_of_fmatch_in.three
        + p_phase.a2 * powers_of_fmatch_in.eight_thirds * powers_of_fmatch_in.two_thirds
        + p_phase.a3 * powers_of_fmatch_in.eight_thirds * powers_of_fmatch_in.itself
        + p_phase.a4 * powers_of_fmatch_in.eight_thirds * powers_of_fmatch_in.four_thirds
    )

    phase_in = phase_in * powers_of_fmatch_in.m_eight_thirds * p_wf.dphase0

    # /*
    # Intermediate phase collocation points:
    # Here we only use the first N points in the array where N = the
    # number of intermediate collocation points.

    # The size of the array is controlled by: N_MAX_COLLOCATION_POINTS_PHASE_INT

    # Default is to use 5 collocation points.

    # See. Eq. 7.7 and 7.8 where f_H = pPhase->fPhaseMatchIM and f_L = pPhase->fPhaseMatchIN
    # */
    deltax = f_phase_match_im - f_phase_match_in
    xmin = f_phase_match_in

    is_valid = jnp.logical_or(
        p_wf.imr_phenom_x_intermediate_phase_version == 104, p_wf.imr_phenom_x_intermediate_phase_version == 105
    )
    checkify.check(is_valid, "Error: IMRPhenomXIntermediatePhaseVersion is not valid.")
    n_collocation_points_int = jax.lax.select(p_wf.imr_phenom_x_intermediate_phase_version == 104, 4, 5)

    # Canonical intermediate model using 4 collocation points
    def branch_104(p_phase: IMRPhenomXPhaseCoefficientsDataClass) -> IMRPhenomXPhaseCoefficientsDataClass:
        # // Using 4 collocation points in intermediate region
        collocation_points_phase_int = gpoints4 * deltax + xmin

        # v2IM - v4RD. Using v4RD helps condition the fits with v4RD being very a robust fit.
        v2_im_m_rd_v4 = imr_phenom_x_intermediate_phase_22_v2m_rd_v4(
            eta, p_wf.s_tot_r, p_wf.dchi, delta, p_wf.imr_phenom_x_intermediate_phase_version
        )

        # v3IM - v4RD. Using v4RD helps condition the fits with v4RD being very a robust fit.
        v3_im_m_rd_v4 = imr_phenom_x_intermediate_phase_22_v3m_rd_v4(
            eta, p_wf.s_tot_r, p_wf.dchi, delta, p_wf.imr_phenom_x_intermediate_phase_version
        )

        # Direct fit to the collocation point at F2. We will take a weighted average of the direct and conditioned fit.
        v2_im = imr_phenom_x_intermediate_phase_22_v2(
            eta, p_wf.s_tot_r, p_wf.dchi, delta, p_wf.imr_phenom_x_intermediate_phase_version
        )

        # /* Evaluate collocation points */
        collocation_values_phase_int = jnp.array(
            [
                phase_in,
                0.75 * (v2_im_m_rd_v4 + rd_v4) + 0.25 * v2_im,
                v3_im_m_rd_v4 + rd_v4,
                phase_rd,
            ]
        )

        ff = collocation_points_phase_int  # jax.lax.dynamic_slice(
        #     collocation_points_phase_int,
        #     (0,),  # start index
        #     (4,)   # slice size
        # )
        ff1 = p_wf.f_ring / ff
        ff2 = ff1 * ff1
        ff3 = ff1 * ff2
        ff0 = (4 * c_l) / (4.0 * p_wf.f_damp * p_wf.f_damp + (ff - p_wf.f_ring) * (ff - p_wf.f_ring))

        b = collocation_values_phase_int - ff0

        a_matrix = jnp.array([jnp.ones(4), ff1, ff2, ff3]).T

        # /* We now solve the system A x = b via an LU decomposition */
        x = jax.scipy.linalg.lu_solve(jax.scipy.linalg.lu_factor(a_matrix), b)

        # /* Set inspiral phenomenological coefficients from solution to A x = b */
        b0 = x[0]  # Constant
        b1 = x[1]  # f^{-1}
        b2 = x[2] * p_wf.f_ring * p_wf.f_ring  # f^{-2}
        # b3 = 0.0
        b4 = x[3] * p_wf.f_ring * p_wf.f_ring * p_wf.f_ring * p_wf.f_ring  # f^{-4}

        p_phase = dataclasses.replace(
            p_phase,
            collocation_points_phase_int=jnp.pad(
                collocation_points_phase_int, (0, 1), constant_values=0.0
            ),  # zero pad for jit compatibility
            collocation_values_phase_int=jnp.pad(
                collocation_values_phase_int, (0, 1), constant_values=0.0
            ),  # zero pad for jit compatibility
            b0=b0,
            b1=b1,
            b2=b2,
            b4=b4,
        )
        return p_phase

    # Canonical intermediate model using 5 collocation points
    def branch_105(p_phase: IMRPhenomXPhaseCoefficientsDataClass) -> IMRPhenomXPhaseCoefficientsDataClass:
        # // Using 5 collocation points in intermediate region
        collocation_points_phase_int = gpoints5 * deltax + xmin

        # /* Evaluate collocation points */
        # /* The first and last collocation points for the intermediate region are set from the inspiral fit and ringdown respectively */
        # // v2IM - v4RD. Using v4RD helps condition the fits with v4RD being very a robust fit.
        v2_im_m_rd_v4 = imr_phenom_x_intermediate_phase_22_v2m_rd_v4(
            eta, p_wf.s_tot_r, p_wf.dchi, delta, p_wf.imr_phenom_x_intermediate_phase_version
        )

        # v3IM - v4RD. Using v4RD helps condition the fits with v4RD being very a robust fit.
        v3_im_m_rd_v4 = imr_phenom_x_intermediate_phase_22_v3m_rd_v4(
            eta, p_wf.s_tot_r, p_wf.dchi, delta, p_wf.imr_phenom_x_intermediate_phase_version
        )

        # Direct fit to the collocation point at F2. We will take a weighted average of the direct and conditioned fit.
        v2_im = imr_phenom_x_intermediate_phase_22_v2(
            eta, p_wf.s_tot_r, p_wf.dchi, delta, p_wf.imr_phenom_x_intermediate_phase_version
        )

        # /* Evaluate collocation points */
        collocation_values_phase_int = jnp.array(
            [
                phase_in,
                0.75 * (v2_im_m_rd_v4 + rd_v4)
                + 0.25 * v2_im,  # Take a weighted average for these points. Helps condition the fit.
                v3_im_m_rd_v4 + rd_v4,
                imr_phenom_x_intermediate_phase_22_d43(
                    eta, p_wf.s_tot_r, p_wf.dchi, delta, p_wf.imr_phenom_x_intermediate_phase_version
                )
                + (v3_im_m_rd_v4 + rd_v4),
                phase_rd,
            ]
        )

        ff = collocation_points_phase_int
        ff1 = p_wf.f_ring / ff
        ff2 = ff1 * ff1
        ff3 = ff1 * ff2
        ff4 = ff2 * ff2
        ff0 = (4 * c_l) / (4.0 * p_wf.f_damp * p_wf.f_damp + (ff - p_wf.f_ring) * (ff - p_wf.f_ring))

        b = collocation_values_phase_int - ff0

        a_matrix = jnp.array([jnp.ones(5), ff1, ff2, ff3, ff4]).T

        # /* We now solve the system A x = b via an LU decomposition */
        x = jax.scipy.linalg.lu_solve(jax.scipy.linalg.lu_factor(a_matrix), b)

        b0 = x[0]  # Constant
        b1 = x[1]  # f^{-1}
        b2 = x[2] * p_wf.f_ring * p_wf.f_ring  # f^{-2}
        b3 = x[3] * p_wf.f_ring * p_wf.f_ring * p_wf.f_ring  # f^{-3}
        b4 = x[4] * p_wf.f_ring * p_wf.f_ring * p_wf.f_ring * p_wf.f_ring  # f^{-4}

        p_phase = dataclasses.replace(
            p_phase,
            collocation_points_phase_int=collocation_points_phase_int,
            collocation_values_phase_int=collocation_values_phase_int,
            b0=b0,
            b1=b1,
            b2=b2,
            b3=b3,
            b4=b4,
        )
        return p_phase

    p_phase = jax.lax.cond(
        p_wf.imr_phenom_x_intermediate_phase_version == 104,
        lambda x: branch_104(x),
        lambda x: branch_105(x),
        operand=p_phase,
    )

    # Note: When collocation_values_phase_int == 4, the arrays are padded with a trailing zero.
    # This padding doesn't affect the computation since the extra element is unused.

    ####################################################################
    ############# Leaving out non-GR modifications for now #############
    ####################################################################

    # # /* Ringdown coefficients */
    # nonGR_dc1   = XLALSimInspiralWaveformParamsLookupNonGRDC1(LALparams)
    # nonGR_dc2   = XLALSimInspiralWaveformParamsLookupNonGRDC2(LALparams)
    # nonGR_dc4   = XLALSimInspiralWaveformParamsLookupNonGRDC4(LALparams)
    # nonGR_dcl   = XLALSimInspiralWaveformParamsLookupNonGRDCL(LALparams)

    # # /* Intermediate coefficients */
    # nonGR_db1   = XLALSimInspiralWaveformParamsLookupNonGRDB1(LALparams)
    # nonGR_db2   = XLALSimInspiralWaveformParamsLookupNonGRDB2(LALparams)
    # nonGR_db3   = XLALSimInspiralWaveformParamsLookupNonGRDB3(LALparams)
    # nonGR_db4   = XLALSimInspiralWaveformParamsLookupNonGRDB4(LALparams)

    # # /* Inspiral coefficients */
    # dchi_minus2 = XLALSimInspiralWaveformParamsLookupNonGRDChiMinus2(LALparams)
    # dchi_minus1 = XLALSimInspiralWaveformParamsLookupNonGRDChiMinus1(LALparams)
    # dchi0       = XLALSimInspiralWaveformParamsLookupNonGRDChi0(LALparams)
    # dchi1       = XLALSimInspiralWaveformParamsLookupNonGRDChi1(LALparams)
    # dchi2       = XLALSimInspiralWaveformParamsLookupNonGRDChi2(LALparams)
    # dchi3       = XLALSimInspiralWaveformParamsLookupNonGRDChi3(LALparams)
    # dchi4       = XLALSimInspiralWaveformParamsLookupNonGRDChi4(LALparams)
    # dchi5       = XLALSimInspiralWaveformParamsLookupNonGRDChi5(LALparams)
    # dchi5L      = XLALSimInspiralWaveformParamsLookupNonGRDChi5L(LALparams)
    # dchi6       = XLALSimInspiralWaveformParamsLookupNonGRDChi6(LALparams)
    # dchi6L      = XLALSimInspiralWaveformParamsLookupNonGRDChi6L(LALparams)
    # dchi7       = XLALSimInspiralWaveformParamsLookupNonGRDChi7(LALparams)

    # # /* Can include these terms in the future as desired... */
    # dchi8       = 0.0
    # dchi8L      = 0.0
    # dchi9       = 0.0
    # dchi9L      = 0.0

    # # /* ~~~~ RINGDOWN ~~~~ */
    # pPhase->cLGR  = pPhase->cL // Store GR value for reference
    # pPhase->c1   *= (1.0 + nonGR_dc1)
    # pPhase->c2   *= (1.0 + nonGR_dc2)
    # pPhase->c4   *= (1.0 + nonGR_dc4)
    # pPhase->cL   *= (1.0 + nonGR_dcl)

    # # /* Set pre-cached variables */
    # pPhase->c4ov3   = pPhase->c4 / 3.0
    # pPhase->cLovfda = pPhase->cL / pWF->fDAMP

    # # /* Apply NR tuning for precessing cases (500) */
    # pPhase->b1 = pPhase->b1  +  ( pWF->PNR_DEV_PARAMETER * pWF->ZETA2 )
    # pPhase->b4 = pPhase->b4  +  ( pWF->PNR_DEV_PARAMETER * pWF->ZETA1 )

    # # /* ~~~~ INTERMEDIATE ~~~~ */
    # if(pWF->IMRPhenomXIntermediatePhaseVersion == 104)
    # {
    # 	pPhase->b1 *= (1.0 + nonGR_db1)
    # 	pPhase->b2 *= (1.0 + nonGR_db2)
    # 	pPhase->b4 *= (1.0 + nonGR_db4)
    # }
    # else if(pWF->IMRPhenomXIntermediatePhaseVersion == 105)
    # {
    # 	pPhase->b1 *= (1.0 + nonGR_db1)
    # 	pPhase->b2 *= (1.0 + nonGR_db2)
    # 	pPhase->b3 *= (1.0 + nonGR_db3)
    # 	pPhase->b4 *= (1.0 + nonGR_db4)
    # }
    # else
    # {
    # 	XLALPrintError("Error in ComputeIMRPhenomXWaveformVariables: IMRPhenomXIntermediatePhaseVersion is not valid.\n")
    # }

    # /* ~~~~ INSPIRAL ~~~~ */
    # /* Initialize -1PN coefficient*/
    # pPhase->phi_minus2   = 0.0
    # pPhase->dphi_minus2  = 0.0

    # pPhase->phi_minus1   = 0.0
    # pPhase->dphi_minus1  = 0.0

    # /*
    # 	If tgr_parameterization = 1, deform complete PN coefficient. This is an FTA-like parameterization.
    # 	If tgr_parameterization = 0, only deform non-spinning coefficient. This is the original TIGER-like implementation.
    # */
    # int tgr_parameterization = 0
    # tgr_parameterization     = XLALSimInspiralWaveformParamsLookupNonGRParameterization(LALparams)

    # if(tgr_parameterization == 1)
    # {
    # 		/* -1.0 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi_minus2 = dchi_minus2 / powers_of_lalpi.two_thirds

    # 		/* -0.5 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi_minus1 = dchi_minus1 / powers_of_lalpi.one_third

    # 		/* 0.0 PN */
    # 		pPhase->phi0       = (phi0NS + phi0S)*(1.0 + dchi0)

    # 		/* 0.5 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi1       = dchi1 * powers_of_lalpi.one_third

    # 		/* 1.0 PN */
    # 		pPhase->phi2       = (phi2NS + phi2S)*(1.0 + dchi2)

    # 		/* 1.5 PN */
    # 		pPhase->phi3       = (phi3NS + phi3S)*(1.0 + dchi3)

    # 		/* 2.0 PN */
    # 		pPhase->phi4       = (phi4NS + phi4S)*(1.0 + dchi4)

    # 		/* 2.5 PN */
    # 		pPhase->phi5       = (phi5NS + phi5S)*(1.0 + dchi5)

    # 		/* 2.5 PN, Log Terms */
    # 		pPhase->phi5L      = (phi5LNS + phi5LS)*(1.0 + dchi5L)

    # 		/* 3.0 PN */
    # 		pPhase->phi6       = (phi6NS + phi6S)*(1.0 + dchi6)

    # 		/* 3.0 PN, Log Term */
    # 		pPhase->phi6L      = (phi6LNS + phi6LS)*(1.0 + dchi6L)

    # 		/* 3.5PN */
    # 		pPhase->phi7       = (phi7NS + phi7S)*(1.0 + dchi7)

    # 		/* 4.0PN */
    # 		pPhase->phi8       = (phi8NS + phi8S)*(1.0 + dchi8)

    # 		/* 4.0 PN, Log Terms */
    # 		pPhase->phi8L      = (phi8LNS + phi8LS)*(1.0 + dchi8L)

    # 		/* 4.0 PN */
    # 		pPhase->phi9       = (phi9NS + phi9S)*(1.0 + dchi9)

    # 		/* 4.0 PN, Log Terms */
    # 		pPhase->phi9L      = (phi9LNS + phi9LS)*(1.0 + dchi9L)
    # }
    # else if(tgr_parameterization == 0)
    # {
    # 		/* -1.0 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi_minus2 = dchi_minus2 / powers_of_lalpi.two_thirds

    # 		/* -0.5 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi_minus1 = dchi_minus1 / powers_of_lalpi.one_third

    # 		/* 0.0 PN */
    # 		pPhase->phi0       = phi0NS*(1.0 + dchi0) + phi0S

    # 		/* 0.5 PN: This vanishes in GR, so is parameterized as an absolute deviation */
    # 		pPhase->phi1       = dchi1 * powers_of_lalpi.one_third

    # 		/* 1.0 PN */
    # 		pPhase->phi2       = phi2NS*(1.0 + dchi2) + phi2S

    # 		/* 1.5 PN */
    # 		pPhase->phi3       = phi3NS*(1.0 + dchi3)+ phi3S

    # 		/* 2.0 PN */
    # 		pPhase->phi4       = phi4NS*(1.0 + dchi4) + phi4S

    # 		/* 2.5 PN */
    # 		pPhase->phi5       = phi5NS*(1.0 + dchi5) + phi5S

    # 		/* 2.5 PN, Log Terms */
    # 		pPhase->phi5L      = phi5LNS*(1.0 + dchi5L) + phi5LS

    # 		/* 3.0 PN */
    # 		pPhase->phi6       = phi6NS*(1.0 + dchi6) + phi6S

    # 		/* 3.0 PN, Log Term */
    # 		pPhase->phi6L      = phi6LNS*(1.0 + dchi6L) + phi6LS

    # 		/* 3.5PN */
    # 		pPhase->phi7       = phi7NS*(1.0 + dchi7) + phi7S

    # 		/* 4.0PN */
    # 		pPhase->phi8       = phi8NS*(1.0 + dchi8) + phi8S

    # 		/* 4.0 PN, Log Terms */
    # 		pPhase->phi8L      = phi8LNS*(1.0 + dchi8L) + phi8LS

    # 		/* 4.0 PN */
    # 		pPhase->phi9       = phi9NS*(1.0 + dchi9) + phi9S

    # 		/* 4.0 PN, Log Terms */
    # 		pPhase->phi9L      = phi9LNS*(1.0 + dchi9L) + phi9LS
    # }
    # else
    # {
    # 		XLALPrintError("Error in IMRPhenomXGetPhaseCoefficients: TGR Parameterizataion is not valid.\n")
    # }

    # /* Recalculate phase derivatives including TGR corrections */
    # pPhase->dphi_minus2 = +(7.0 / 5.0) * pPhase->phi_minus2
    # pPhase->dphi_minus1 = +(6.0 / 5.0) * pPhase->phi_minus1
    # pPhase->dphi0       = +(5.0 / 5.0) * pPhase->phi0
    # pPhase->dphi1       = +(4.0 / 5.0) * pPhase->phi1
    # pPhase->dphi2       = +(3.0 / 5.0) * pPhase->phi2
    # pPhase->dphi3       = +(2.0 / 5.0) * pPhase->phi3
    # pPhase->dphi4       = +(1.0 / 5.0) * pPhase->phi4
    # pPhase->dphi5       = -(3.0 / 5.0) * pPhase->phi5L
    # pPhase->dphi6       = -(1.0 / 5.0) * pPhase->phi6 - (3.0 / 5.0) * pPhase->phi6L
    # pPhase->dphi6L      = -(1.0 / 5.0) * pPhase->phi6L
    # pPhase->dphi7       = -(2.0 / 5.0) * pPhase->phi7
    # pPhase->dphi8       = -(3.0 / 5.0) * pPhase->phi8 - (3.0 / 5.0) * pPhase->phi8L
    # pPhase->dphi8L      = -(3.0 / 5.0) * pPhase->phi8L
    # pPhase->dphi9       = -(4.0 / 5.0) * pPhase->phi9 - (3.0 / 5.0) * pPhase->phi9L
    # pPhase->dphi9L      = -(3.0 / 5.0) * pPhase->phi9L

    # /* Initialize connection coefficients */
    # pPhase->C1Int = 0
    # pPhase->C2Int = 0
    # pPhase->C1MRD = 0
    # pPhase->C2MRD = 0

    p_phase = dataclasses.replace(
        p_phase,
        f_phase_match_in=f_phase_match_in,
        f_phase_match_im=f_phase_match_im,
        f_phase_ins_min=f_phase_ins_min,
        f_phase_ins_max=f_phase_ins_max,
        f_phase_rd_min=f_phase_rd_min,
        f_phase_rd_max=f_phase_rd_max,
        phi_norm=phi_norm,
        collocation_points_phase_rd=collocation_points_phase_rd,
        collocation_values_phase_rd=collocation_values_phase_rd,
        n_collocation_points_rd=n_collocation_points_rd,
        c0=c0,
        c1=c1,
        c2=c2,
        c4=c4,
        c_l=c_l,
        c_rd=c_rd,
        n_pseudo_pn=n_pseudo_pn,
        n_collocation_points_phase_ins=n_collocation_points_phase_ins,
        sigma1=sigma1,
        sigma2=sigma2,
        sigma3=sigma3,
        sigma4=sigma4,
        sigma5=sigma5,
        dphi0=dphi0,
        dphi1=dphi1,
        dphi2=dphi2,
        dphi3=dphi3,
        dphi4=dphi4,
        dphi5=dphi5,
        dphi5l=0.0,
        dphi6=dphi6,
        dphi6l=dphi6l,
        dphi7=dphi7,
        dphi8=dphi8,
        dphi8l=dphi8l,
        dphi9=dphi9,
        dphi9l=dphi9l,
        phi0=phi0,
        phi1=phi1,
        phi2=phi2,
        phi3=phi3,
        phi4=phi4,
        phi5=phi5,
        phi5l=phi5l,
        phi6=phi6,
        phi6l=phi6l,
        phi7=phi7,
        phi8=phi8,
        phi8l=phi8l,
        phi9=phi9,
        phi9l=phi9l,
        phi_initial=phi_initial,
        n_collocation_points_int=n_collocation_points_int,
    )

    return p_phase


# def imr_phenom_x_phase_22_connection_coefficients(
#     p_wf: IMRPhenomXWaveformDataClass, p_phase: IMRPhenomXPhaseCoefficientsDataClass
# ):
# f_ins = p_phase.f_phase_match_in
# f_int = p_phase.f_phase_match_im

# # /*
# #         Assume an ansatz of the form:

# #         phi_Inspiral (f) = phi_Intermediate (f) + C1 + C2 * f

# #         where transition frequency is fIns

# #         phi_Inspiral (fIns) = phi_Intermediate (fIns) + C1 + C2 * fIns
# #         phi_Inspiral'(fIns) = phi_Intermediate'(fIns) + C2

# #         Solving for C1 and C2

# #         C2 = phi_Inspiral'(fIns) - phi_Intermediate'(fIns)
# #         C1 = phi_Inspiral (fIns) - phi_Intermediate (fIns) - C2 * fIns

# # */

# powers_of_f_ins = imr_phenom_x_initialize_powers(f_ins)


#     DPhiIns = IMRPhenomX_Inspiral_Phase_22_Ansatz(fIns,&powers_of_fIns,pPhase)
#     DPhiInt = IMRPhenomX_Intermediate_Phase_22_Ansatz(fIns,&powers_of_fIns,pWF,pPhase)

#     pPhase->C2Int  = DPhiIns - DPhiInt

#     phiIN = IMRPhenomX_Inspiral_Phase_22_AnsatzInt(fIns,&powers_of_fIns,pPhase)
#     phiIM = IMRPhenomX_Intermediate_Phase_22_AnsatzInt(fIns,&powers_of_fIns,pWF,pPhase)

#     if(debug)
#     {
#     printf("\n")
#     printf("dphiIM = %.6f and dphiIN = %.6f\n",DPhiInt,DPhiIns)
#     printf("phiIN(fIns)  : %.7f\n",phiIN)
#     printf("phiIM(fIns)  : %.7f\n",phiIM)
#     printf("fIns         : %.7f\n",fIns)
#     printf("C2           : %.7f\n",pPhase->C2Int)
#     printf("\n")
#     }

#     pPhase->C1Int = phiIN - phiIM - (pPhase->C2Int * fIns)

#     /*
#             Assume an ansatz of the form:

#             phi_Intermediate (f)    = phi_Ringdown (f) + C1 + C2 * f

#             where transition frequency is fIM

#             phi_Intermediate (fIM) = phi_Ringdown (fRD) + C1 + C2 * fIM
#             phi_Intermediate'(fIM) = phi_Ringdown'(fRD) + C2

#             Solving for C1 and C2

#             C2 = phi_Inspiral'(fIM) - phi_Intermediate'(fIM)
#             C1 = phi_Inspiral (fIM) - phi_Intermediate (fIM) - C2 * fIM

#     */
#     IMRPhenomX_UsefulPowers powers_of_fInt
#     IMRPhenomX_Initialize_Powers(&powers_of_fInt,fInt)

#     phiIMC         = IMRPhenomX_Intermediate_Phase_22_AnsatzInt(fInt,&powers_of_fInt,pWF,pPhase) + pPhase->C1Int + pPhase->C2Int*fInt
#     phiRD          = IMRPhenomX_Ringdown_Phase_22_AnsatzInt(fInt,&powers_of_fInt,pWF,pPhase)
#     DPhiIntC       = IMRPhenomX_Intermediate_Phase_22_Ansatz(fInt,&powers_of_fInt,pWF,pPhase) + pPhase->C2Int
#     DPhiRD         = IMRPhenomX_Ringdown_Phase_22_Ansatz(fInt,&powers_of_fInt,pWF,pPhase)

#     pPhase->C2MRD = DPhiIntC - DPhiRD
#     pPhase->C1MRD = phiIMC - phiRD - pPhase->C2MRD*fInt

#     if(debug)
#     {
#     printf("\n")
#     printf("phiIMC(fInt) : %.7f\n",phiIMC)
#     printf("phiRD(fInt)  : %.7f\n",phiRD)
#     printf("fInt         : %.7f\n",fInt)
#     printf("C2           : %.7f\n",pPhase->C2Int)
#     printf("\n")
#     }

#     if(debug)
#     {
#     printf("dphiIM = %.6f and dphiRD = %.6f\n",DPhiIntC,DPhiRD)
#     printf("\nContinuity Coefficients\n")
#     printf("C1Int : %.6f\n",pPhase->C1Int)
#     printf("C2Int : %.6f\n",pPhase->C2Int)
#     printf("C1MRD : %.6f\n",pPhase->C1MRD)
#     printf("C2MRD : %.6f\n",pPhase->C2MRD)
#     }

#     return


# def imr_phenom_x_full_phase_22(
#         phase: float,
#         dphase: float,
#         m_f: float,
#         p_phase: IMRPhenomXPhaseCoefficientsDataClass,
#         p_wf: IMRPhenomXWaveformDataClass
# ):
#     """
#     Function to compute full model phase. This function is designed to be used in in initialization routines, and not for evaluating the phase at many frequencies.
#     """

#     # /*--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--*/
#     # /*            Define useful powers               */
#     # /*--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--*/

#     # // Get useful powers of Mf
#     powers_of_m_f = imr_phenom_x_initialize_powers(m_f)

#     # /* Initialize a struct containing useful powers of Mf at fRef */
#     powers_of_m_fref = imr_phenom_x_initialize_powers(p_wf.m_f_ref)

#     # /*--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--*/
#     # /*           Define needed constants             */
#     # /*--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--*/

#     # /* 1/eta is used to re-scale the pre-phase quantity */
#     inveta    = (1.0 / p_wf.eta)

#     # /* We keep this phase shift to ease comparison with
#     # original phase routines */
#     lina = 0

#     # /* Get phase connection coefficients
#     # and store them to pWF. This step is to make
#     # sure that teh coefficients are up-to-date */
#     IMRPhenomX_Phase_22_ConnectionCoefficients(pWF,pPhase)

#     # /* Compute the timeshift that PhenomXAS uses to align waveforms
#     # with the hybrids used to make their model */
#     linb = IMRPhenomX_TimeShift_22(pPhase, pWF)

#     # Calculate phase at reference frequency: phifRef = 2.0*phi0 + LAL_PI_4 + PhenomXPhase(fRef)
#     phifRef = -(inveta * IMRPhenomX_Phase_22(pWF->MfRef, &powers_of_MfRef, pPhase, pWF) + linb*pWF->MfRef + lina) + 2.0*pWF->phi0 + LAL_PI_4

#     # ~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+
#     # Note that we do not store the value of phifRef to pWF as is done in
#     # IMRPhenomXASGenerateFD. We choose to not do so in order to avoid
#     # potential confusion (e.g. if this function is called within a
#     # workflow that assumes the value defined in IMRPhenomXASGenerateFD).
#     # Note that this concern may not be valid.
#     # ~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+ */

#     # /*--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--*/
#     # /*        Compute the full model phase           */
#     # /*--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--(*)--*/

#     # /* Use previously made function to compute what we call
#     # here the pre-phase, becuase it's not actually a phase! */
#     pre_phase = IMRPhenomX_Phase_22(Mf,&powers_of_Mf,pPhase,pWF)

#     # /* Given the pre-phase, we need to scale and shift according to the
#     # XAS construction */
#     *phase   = pre_phase * inveta
#     *phase  += linb*Mf + lina + phifRef

#     # /* Repeat the excercise above for the phase derivative:
#     # "dphase" is (d/df)phase at Mf */
#     pre_dphase = IMRPhenomX_dPhase_22(Mf,&powers_of_Mf,pPhase,pWF)
#     *dphase  = pre_dphase * inveta
#     *dphase += linb
