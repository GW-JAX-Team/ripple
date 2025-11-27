"""Helper functions for IMRPhenomXPHM waveform model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.constants import PI, gt, MRSUN
from ripplegw.waveforms.imr_phenom_xphm.lal_constants import LAL_MSUN_SI
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXUsefulPowersDataClass,
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    imr_phenom_x_internal_nudge,
    xlal_sim_imr_phenom_x_chi_eff,
    xlal_sim_imr_phenom_x_chi_pn_hat,
    xlal_sim_imr_phenom_x_stot_r,
    xlal_sim_imr_phenom_x_dchi,
    xlal_sim_imr_phenom_x_final_mass_2017,
    xlal_sim_imr_phenom_x_final_spin_2017,
    xlal_sim_imr_phenom_x_fISCO,
    xlal_sim_imr_phenom_x_fMECO,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_qnm import (
    evaluate_QNMfit_fdamp22,
    evaluate_QNMfit_fring22,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral import xlal_sim_inspiral_set_quad_mon_params_from_lambdas
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_mode_array_is_mode_active,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass
from ripplegw import ms_to_Mc_eta
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


# TODO: Not finished
def imr_phenom_x_set_waveform_variables(
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

    if lal_params.phen_x_tidal != 0:
        lambda1_in = lal_params.lambda1
        lambda2_in = lal_params.lambda2
        if lambda1_in < 0.0 or lambda2_in < 0.0:
            checkify.check(False, "Tidal deformabilities lambda1 and lambda2 must be non-negative.")
            # Set quadrupole-monopole parameters from tidal deformabilities
        lal_params = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(lal_params)

        quad_param1_in = 1.0 + lal_params.d_quad_mon1
        quad_param2_in = 1.0 + lal_params.d_quad_mon2

    if m1_in >= m2_in:
        chi1l = chi1l_in
        chi2l = chi2l_in
        m1 = m1_in
        m2 = m2_in
        lambda1 = lambda1_in
        lambda2 = lambda2_in
        quad_param1 = quad_param1_in
        quad_param2 = quad_param2_in
    else:
        chi1l = chi2l_in
        chi2l = chi1l_in
        m1 = m2_in
        m2 = m1_in
        lambda1 = lambda2_in
        lambda2 = lambda1_in
        quad_param1 = quad_param2_in
        quad_param2 = quad_param1_in

    if chi1l > 1.0:
        chi1l = imr_phenom_x_internal_nudge(chi1l, 1.0, 1e-6)
    if chi2l > 1.0:
        chi2l = imr_phenom_x_internal_nudge(chi2l, 1.0, 1e-6)
    if chi1l < -1.0:
        chi1l = imr_phenom_x_internal_nudge(chi1l, -1.0, 1e-6)
    if chi2l < -1.0:
        chi2l = imr_phenom_x_internal_nudge(chi2l, -1.0, 1e-6)

    # If spins are still unphysical after checking for small round-off errors, fail.
    if chi1l > 1.0 or chi1l < -1.0 or chi2l > 1.0 or chi2l < -1.0:
        checkify.check(False, "Unphysical spins requested: must obey the Kerr bound [-1,1].")

    # Symmetric mass ratio
    delta = jnp.abs((m1 - m2) / (m1 + m2))
    Mc, eta = ms_to_Mc_eta(m1, m2)
    # eta = jnp.abs(0.25 * (1.0 - delta * delta))
    q = m1 / m2

    if eta > 0.25:
        eta = 0.25
    if eta > 0.25 or eta < 0.0:
        checkify.check(False, "Unphysical mass ratio %s requested.", eta)

    if eta == 0.25:
        q = 1.0

    # Check the mass ratio
    checkify.check(q > 1000.0, "The model is not supported for mass ratios > 1000.")

    m_tot = m1 + m2
    M_sec = m_tot * gt
    eta2 = eta * eta
    kappa2T = get_kappa(jnp.array([m1, m2, chi1l, chi2l, lambda1, lambda2]))
    f_merger = _get_merger_frequency(jnp.array([m1, m2, chi1l, chi2l, lambda1, lambda2]), kappa2T)

    # /* Spin parameterisations */
    chiEff    = xlal_sim_imr_phenom_x_chi_eff(eta,chi1l,chi2l)
    chiPNHat  = xlal_sim_imr_phenom_x_chi_pn_hat(eta,chi1l,chi2l)
    STotR     = xlal_sim_imr_phenom_x_stot_r(eta,chi1l,chi2l)
    dchi      = xlal_sim_imr_phenom_x_dchi(chi1l,chi2l)
    dchi_half = dchi*0.5

    SigmaL    = (chi2l * m2) - (chi1l * m1) 			# // SigmaL = (M/m2)*(S2.L) - (M/m2)*(S1.L)
    SL        = chi1l * (m1 * m1) + chi2l * (m2 * m2)  # // SL = S1.L + S2.L

    fRef      = f_ref
    phiRef_In = phi0
    phi0      = phi0							#// Orbital phase at reference frequency (as passed from lalsimulation)
    beta      = PI*0.5 - phi0 				#// Azimuthal angle of binary at reference frequency
    phifRef   = 0.0							#// This is calculated later

    # /* Geometric reference frequency */
    MfRef     = M_sec*fRef #XLALSimIMRPhenomXUtilsHztoMf(fRef,Mtot)
    piM       = PI * M_sec
    v_ref     = (piM * fRef)**(1/3)

    deltaF    = delta_f
    deltaMF   = M_sec*delta_f #XLALSimIMRPhenomXUtilsHztoMf(deltaF,Mtot)

    # /* Define the default end of the waveform as: 0.3 Mf. This value is chosen such that the 44 mode also shows the ringdown part. */
    fCutDef = 0.3
    # // If chieff is very high the ringdown of the 44 is almost cut it out when using 0.3, so we increase a little bit the cut of freq up 0.33.
    if chiEff > 0.99: 
        fCutDef = 0.33

    # /* Minimum and maximum frequency */
    fMin      = f_min
    fMax      = f_max
    MfMax     = M_sec*f_max #XLALSimIMRPhenomXUtilsHztoMf(fMax,Mtot)

    # /* Convert fCut to physical cut-off frequency */
    fCut      = fCutDef / M_sec

    # /* Sanity check that minimum start frequency is less than cut-off frequency */
    if (fCut <= fMin):
        jax.debug.print(f"(fCut = {fCut} Hz) <= f_min = {fMin}")

    # if(debug)
    # {
    # 	printf("fRef : %.6f\n",fRef)
    # 	printf("phi0 : %.6f\n",phi0)
    # 	printf("fCut : %.6f\n",fCut)
    # 	printf("fMin : %.6f\n",fMin)
    # 	printf("fMax : %.6f\n",fMax)
    # }

    # /* By default f_max_prime is f_max. If fCut < fMax, then use fCut, i.e. waveform up to fMax will be zeros */
    f_max_prime   = fMax
    f_max_prime   = fMax if fMax else fCut
    f_max_prime   = fCut if (f_max_prime > fCut) else f_max_prime

    if f_max_prime <= fMin:
        jax.debug.print("f_max <= f_min")

    # if(debug)
    # {
    # 	printf("fMin        = %.6f\n",fMin)
    # 	printf("fMax        = %.6f\n",fMax)
    # 	printf("f_max_prime = %.6f\n",f_max_prime)
    # }

    # /* Final Mass and Spin */
    # NOTE: These are only default values
    Mfinal    = xlal_sim_imr_phenom_x_final_mass_2017(eta,chi1l,chi2l)
    afinal    = xlal_sim_imr_phenom_x_final_spin_2017(eta,chi1l,chi2l)

    # /* (500) Set default values of physically specific final spin parameters for use with PNR/XCP */
    afinal_nonprec = afinal     #// NOTE: This is only a default value see LALSimIMRPhenomX_precession.c
    afinal_prec    = afinal     #// NOTE: This is only a default value see LALSimIMRPhenomX_precession.c

    # /* Ringdown and damping frequency of final BH */
    fRING     = evaluate_QNMfit_fring22(afinal) / (Mfinal)
    fDAMP     = evaluate_QNMfit_fdamp22(afinal) / (Mfinal)



    # if(debug)
    # {
    # 	printf("Mf  = %.6f\n",Mfinal)
    # 	printf("af  = %.6f\n",afinal)
    # 	printf("frd = %.6f\n",fRING)
    # 	printf("fda = %.6f\n",fDAMP)
    # }

    if Mfinal > 1.0:
        jax.debug.print("IMRPhenomX_FinalMass2018: Final mass > 1.0 not physical.")
    if abs(afinal) > 1.0:
        jax.debug.print("IMRPhenomX_FinalSpin2018: Final spin > 1.0 is not physical.")

    # /* Fit to the hybrid minimum energy circular orbit (MECO), Cabero et al, Phys.Rev. D95 (2017) */
    fMECO       = xlal_sim_imr_phenom_x_fMECO(eta,chi1l,chi2l)

    # /* Innermost stable circular orbit (ISCO), e.g. Ori et al, Phys.Rev. D62 (2000) 124022 */
    fISCO       = xlal_sim_imr_phenom_x_fISCO(afinal)

    # if(debug)
    # {
    #     printf("fMECO = %.6f\n",fMECO)
    #     printf("fISCO = %.6f\n",fISCO)
    # }

    if(fMECO > fISCO):
        # /* If MECO > fISCO, throw an error - this may be the case for very corner cases in the parameter space (e.g. q ~ 1000, chi ~ 0.999) */
        jax.debug.print("Error: f_MECO cannot be greater than f_ISCO.")


    # /* Distance and inclination */
    distance    = distance
    inclination = inclination

    # /* Amplitude normalization */
    amp0        = m_tot * MRSUN * m_tot * gt / distance
    ampNorm     = jnp.sqrt(2.0/3.0) * jnp.sqrt(eta) * powers_of_lalpi.m_one_sixth

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

    dphase0 = 5.0 / (128.0 * PI**(5.0/3.0))

    # /* Set nonprecessing value of select precession quantities (PNRUseTunedCoprec)*/
    chiTot_perp = 0.0
    chi_p = 0.0
    theta_LS = 0.0
    a1 = 0.0
    PNR_DEV_PARAMETER = 0.0
    PNR_SINGLE_SPIN = 0
    MU1 = 0
    MU2 = 0
    MU3 = 0
    MU4 = 0
    NU0 = 0
    NU4 = 0
    NU5 = 0
    NU6 = 0
    ZETA1 = 0
    ZETA2 = 0
    fRINGEffShiftDividedByEmm = 0

    f_inspiral_align = 0.0
    XAS_dphase_at_f_inspiral_align = 0.0
    XAS_phase_at_f_inspiral_align = 0.0
    XHM_dphase_at_f_inspiral_align = 0.0
    XHM_phase_at_f_inspiral_align = 0.0

    betaRD = 0.0
    fRING22_prec = 0.0
    fRINGCP = 0.0
    pnr_window = 0.0

    APPLY_PNR_DEVIATIONS = 0

    # /* Set nonprecessing value of select precession quantities (PNRUseTunedCoprec)*/

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
        mc=Mc,
        m_tot_si=m1_si+m1_si,
        m_tot=m_tot,
        m1=m1/m_tot,
        m2=m2/m_tot,
        m_sec=M_sec,
        delta=delta,
        eta2=eta2,
        eta3=eta*eta2,
        chi1l=chi1l,
        chi2l=chi2l,
        chi1l2l=chi1l*chi2l,
        chi1l2=chi1l*chi1l,
        chi1l3=chi1l*chi1l*chi1l,
        chi2l2=chi2l*chi2l,
        chi2l3=chi2l*chi2l*chi2l,
        chi_eff=chiEff,
        chi_pn_hat=chiPNHat,
        chi_tot_perp=chiTot_perp,
        chi_p=chi_p,
        theta_ls=theta_LS,
        a1=a1,
        mu1=MU1,
        mu2=MU2,
        mu3=MU3,
        mu4=MU4,
        nu0=NU0,
        nu4=NU4,
        nu5=NU5,
        nu6=NU6,
        zeta1=ZETA1,
        zeta2=ZETA2,
        pnr_dev_parameter=PNR_DEV_PARAMETER,
        pnr_window=pnr_window,
        apply_pnr_deviations=APPLY_PNR_DEVIATIONS,
        pnr_single_spin=PNR_SINGLE_SPIN,
        f_ring_eff_shift_divided_by_emm=fRINGEffShiftDividedByEmm,
        s_tot_r=STotR,
        dchi=dchi,
        dchi_half=dchi_half,
        sl=SL,
        sigma_l=SigmaL,
        lambda1=lambda1,
        lambda2=lambda2,
        quad_param1=quad_param1,
        quad_param2=quad_param2,
        kappa2_t=kappa2T,
        f_merger=f_merger,
        f_ref=fRef,
        phi_ref_in=phiRef_In,
        phi0=phi0,
        beta=beta,
        v_ref=v_ref,
        delta_f=deltaF,
        f_min=fMin,
        f_max=fMax,
        f_max_prime=f_max_prime,
        f_cut=fCut,
        m_final=Mfinal,
        a_final=afinal,
        a_final_prec=afinal_prec,
        f_ring=fRING,
        f_damp=fDAMP,
        distance=distance,
        inclination=inclination,
        amp0=amp0,
        amp_norm=ampNorm,
        dphase0=dphase0,
        eta4=eta2*eta2,
        f_meco=fMECO,
        f_isco=fISCO,
        beta_rd=betaRD,
        f_ring22_prec=fRING22_prec,
        f_ring_cp=fRINGCP,
        f_inspiral_align=f_inspiral_align,
        xas_dphase_at_f_inspiral_align=XAS_dphase_at_f_inspiral_align,
        xas_phase_at_f_inspiral_align=XAS_phase_at_f_inspiral_align,
        xhm_dphase_at_f_inspiral_align=XHM_dphase_at_f_inspiral_align,
        xhm_phase_at_f_inspiral_align=XHM_phase_at_f_inspiral_align,
        f_ring21=0.0,
        f_damp21=0.0,
        f_ring32=0.0,
        f_damp32=0.0,
        f_ring33=0.0,
        f_damp33=0.0,
        f_ring44=0.0,
        f_damp44=0.0,
        m_f_max=MfMax,
        delta_mf=deltaMF,
        f_cut_def=fCutDef,
        m_f_ref=MfRef,
        phi_f_ref=phifRef,
        pi_m=piM,
        e_rad=0.0,
        a_final_non_prec=afinal_nonprec,
        lal_params=lal_params,
        m=0.0,
        m1_2=0.0,
        m2_2=0.0,
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


def check_input_mode_array(lal_params: dict, max_l: int = 8) -> tuple[checkify.Error, bool]:
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
    has_mode_array = "mode_array" in lal_params

    def check_with_mode_array():
        """Check modes when mode_array is present."""
        mode_array = lal_params["mode_array"]
        return _check_input_mode_array_impl(mode_array, ell_valid, emm_valid, max_l)

    def check_without_mode_array():
        """Skip check when mode_array is not present."""
        return checkify.checkify(lambda: True)()

    # Use Python if since this is before JIT transforms the whole function
    # (check_input_mode_array itself is not transformed, only called by JIT)
    if has_mode_array:
        return check_with_mode_array()
    return check_without_mode_array()
