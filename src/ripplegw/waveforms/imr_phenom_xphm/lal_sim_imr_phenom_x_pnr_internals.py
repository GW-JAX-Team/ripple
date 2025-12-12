"""
Docstring for ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_pnr_internals
"""

from __future__ import annotations

import copy
import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.constants import PI
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals import (
    imr_phenom_x_get_phase_coefficients,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXPhaseCoefficientsDataClass,
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


@checkify.checkify
def imr_phenom_x_pnr_get_and_set_co_prec_params(
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    lal_params: IMRPhenomXPHMParameterDataClass,
) -> tuple[IMRPhenomXWaveformDataClass, IMRPhenomXPrecessionDataClass]:
    """
    Docstring for imr_phenom_x_pnr_get_and_set_co_prec_params

    :param pwf: Description
    :type pwf: IMRPhenomXWaveformDataClass
    :param p_prec: Description
    :type p_prec: IMRPhenomXPrecessionDataClass
    :param lal_params: Description
    :type lal_params: IMRPhenomXPHMParameterDataClass
    """

    # status = 0

    # // Get toggle for outputting coprecesing model from LAL dictionary
    # imr_phenom_x_return_co_prec = lal_params.imr_phenom_x_return_co_prec

    # // Get toggle for PNR coprecessing tuning
    pnr_use_tuned_coprec = lal_params.pnr_use_tuned_coprec
    # imr_phenom_x_use_tuned_coprec = pnr_use_tuned_coprec
    # imr_phenom_x_pnr_use_tuned_coprec = pnr_use_tuned_coprec
    # // Same as above but for 33
    imr_phenom_x_pnr_use_tuned_coprec33 = lal_params.pnr_use_tuned_coprec33 * pnr_use_tuned_coprec

    # // Throw error if preferred value of PNRUseTunedCoprec33 is not found
    # Is this an experimental feature in LAL?
    checkify.check(imr_phenom_x_pnr_use_tuned_coprec33 != 0, "Error: Coprecessing tuning for l=|m|=3 must be off.\n")

    # // Get toggle for enforced use of non-precessing spin as is required during tuning of PNR's coprecessing model
    pnr_use_input_coprec_deviations = lal_params.pnr_use_input_coprec_deviations

    # // Get toggle for forcing inspiral phase and phase derivative alignment with XHM/AS
    pnr_force_xhm_alignment = lal_params.pnr_force_xhm_alignment

    # Throw error if preferred value of PNRForceXHMAlignment is not found
    checkify.check(pnr_force_xhm_alignment != 0, "Error: PNRForceXHMAlignment must be off.")

    # /*-~-~-~-~-~-~-~-~-~-~-~-~-~*
    # Validate PNR usage options
    # *-~-~-~-~-~-~-~-~-~-~-~-~-~*/
    simultaneous_deviations_and_tuned_check = (pnr_use_input_coprec_deviations == 1) & (pnr_use_tuned_coprec == 1)
    checkify.check(
        not simultaneous_deviations_and_tuned_check,
        "Error: PNRUseTunedCoprec and PNRUseInputCoprecDeviations must not be enabled simultaneously.\n",
    )

    # // Define high-level toggle for whether to apply deviations. NOTE that this is imposed at the definition of PNR_DEV_PARAMETER, rather than in a series of IF-ELSE conditions.
    # apply_pnr_deviations = pnr_use_tuned_coprec or pnr_use_input_coprec_deviations

    # // If applying PNR deviations, then we want to be able to refer to some non-PNR waveform properties. For that, we must compute the struct for when PNR is off (and specifically, XAS is wanted).
    # if ( APPLY_PNR_DEVIATIONS && PNRForceXHMAlignment ) {

    # # /*<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.<-.
    # # Compute and store the value of the XAS phase derivative at pPrec->f_inspiral_align
    # #     - Note that this routine also copies the current state of pWF to pPrec for use in PNR+XPHM, where we will similarly want to enforce phase alignment with XHM during inspiral.
    # # ->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.->.*/
    # IMRPhenomX_PNR_SetPhaseAlignmentParams(pWF,pPrec)

    # }

    # jax.lax.cond(
    #     jnp.logical_and(
    #         apply_pnr_deviations,
    #         imr_phenom_x_return_co_prec == 0,
    #     ),

    # )

    # /*-~---~---~---~---~---~---~---~---~---~---~---~---~--*/
    # /*  Define single spin parameters for fit evaluation  */
    # /*-~---~---~---~---~---~---~---~---~---~---~---~---~--*/

    # //
    # REAL8 a1 = pPrec->chi_singleSpin
    # pWF->a1 = a1
    # REAL8 cos_theta = pPrec->costheta_singleSpin

    # //
    # double theta_LS = acos( cos_theta )
    # pWF->theta_LS = theta_LS

    # // Use external function to compute window of tuning deviations. The value of the window will only differ from unity if PNRUseTunedCoprec is equipotent to True. NOTE that correct evaluation of this function requires that e,g, pWF->a1 and pWF->theta_LS be defined above.
    # double pnr_window = 0.0 /* Making the defualt to be zero here, meaning that by default tunings will be off regardless of physical case, or other option flags.*/
    # if (PNRUseTunedCoprec) {
    # // Store for output in related XLAL function
    # pnr_window = IMRPhenomX_PNR_CoprecWindow(pWF)
    # }
    # pWF->pnr_window = pnr_window

    # /* Store XCP deviation parameter: NOTE that we only want to apply the window if PNR is being used, not e.g. if we are calibrating the related coprecessing model */
    # pWF->PNR_DEV_PARAMETER = a1 * sin( pWF->theta_LS ) * APPLY_PNR_DEVIATIONS
    # if ( PNRUseTunedCoprec ){
    # pWF->PNR_DEV_PARAMETER = pnr_window * (pWF->PNR_DEV_PARAMETER)
    # // NOTE that PNR_DEV_PARAMETER for l=m=3 is derived from (and directly proportional to) the one defined just above.
    # }

    # /* Store deviations to be used in PhenomXCP (PNRUseInputCoprecDeviations) */
    # // Get them from the laldict (also used as a way to get default values)
    # // For information about how deviations are applied, see code chunk immediately below.
    # /* NOTE the following for the code just below:
    #     - all default values are zero
    #     - we could toggle the code chunk with PNRUseInputCoprecDeviations, but doing so would be non-orthogonal to the comment above about default values.
    #     - In any case, the user must set PNRUseInputCoprecDeviations=True, AND manually set the deviations using the LALDict interface.
    # */
    # pWF->MU1   = XLALSimInspiralWaveformParamsLookupPhenomXCPMU1(lalParams)
    # pWF->MU2   = XLALSimInspiralWaveformParamsLookupPhenomXCPMU2(lalParams)
    # pWF->MU3   = XLALSimInspiralWaveformParamsLookupPhenomXCPMU3(lalParams)
    # pWF->MU4   = XLALSimInspiralWaveformParamsLookupPhenomXCPMU4(lalParams)
    # pWF->NU0   = XLALSimInspiralWaveformParamsLookupPhenomXCPNU0(lalParams)
    # pWF->NU4   = XLALSimInspiralWaveformParamsLookupPhenomXCPNU4(lalParams)
    # pWF->NU5   = XLALSimInspiralWaveformParamsLookupPhenomXCPNU5(lalParams)
    # pWF->NU6   = XLALSimInspiralWaveformParamsLookupPhenomXCPNU6(lalParams)
    # pWF->ZETA1 = XLALSimInspiralWaveformParamsLookupPhenomXCPZETA1(lalParams)
    # pWF->ZETA2 = XLALSimInspiralWaveformParamsLookupPhenomXCPZETA2(lalParams)

    # //
    # #if DEBUG == 1
    # printf("** >>>>>>>>>>>> PhenomXCP Model domain >>>>>>>>>>> **\n")
    # printf("theta : %f\n",theta_LS*180.0/LAL_PI)
    # printf("eta   : %f\n",pWF->eta)
    # printf("a1    : %f\n",a1)
    # printf("** >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> **\n\n")
    # #endif

    # //
    # if( PNRUseTunedCoprec )
    # {

    # /* ------------------------------------------------------ >>
    #     Get them from the stored model fits that define PhenomXCP
    #     within PhenomXPNR. NOTE that most but not all
    #     modifications take place in LALSimIMRPhenomX_internals.c.
    #     For example, fRING and fDAMP are modified in this file.
    #     NOTE that each tuned parameter requires pWF->PNR_DEV_PARAMETER
    #     to be unchanged from the value used during tuning e.g. a1*sin(theta)
    # << ------------------------------------------------------ */

    # double coprec_eta
    # double coprec_a1 = pWF->a1
    # if (pPrec->IMRPhenomXPrecVersion==330){
    #     // Flatten mass-ratio dependence to limit extrapolation artifacts outside of calibration region
    #     coprec_eta = ( pWF->eta >= 0.09876 ) ? pWF->eta : 0.09876 - ( 0.09876 - pWF->eta ) * 0.1641

    #     // Flatten spin dependence to limit extrapolation artifacts outside of calibration region
    #     coprec_a1  = ( coprec_a1  <= 0.8 ) ? coprec_a1  : 0.8 + (coprec_a1 - 0.8) / 12.0
    #     coprec_a1  = ( coprec_a1  >= 0.2 ) ? coprec_a1  : 0.2
    # }
    # else{
    #     // Flatten mass-ratio dependence to limit extrapolation artifacts outside of calibration region
    #     coprec_eta = ( pWF->eta >= 0.09876 ) ? pWF->eta : 0.09876

    #     // Flatten spin dependence to limit extrapolation artifacts outside of calibration region
    #     coprec_a1  = ( coprec_a1  <= 0.8     ) ? coprec_a1  : 0.8
    #     coprec_a1  = ( coprec_a1  >= 0.2     ) ? coprec_a1  : 0.2
    # }

    # /* MU1 modifies pAmp->v1RD */
    # pWF->MU1     = XLALSimIMRPhenomXCP_MU1_l2m2(   theta_LS, coprec_eta, coprec_a1 )

    # // NOTE that the function for MU2 is not defined in the model
    # /* MU2 would modify pAmp->gamma2 */

    # /* MU2  */
    # pWF->MU2     = XLALSimIMRPhenomXCP_MU2_l2m2(   theta_LS, coprec_eta, coprec_a1 )

    # /* MU3 modifies pAmp->gamma3 */
    # pWF->MU3     = XLALSimIMRPhenomXCP_MU3_l2m2(   theta_LS, coprec_eta, coprec_a1 )

    # /* MU4 modifies V2 for the intermediate amplitude
    # for the DEFAULT value of IMRPhenomXIntermediateAmpVersion
    # use in IMRPhenomXPHM */
    # // pWF->MU4     = IMRPhenomXCP_MU4_l2m2(   theta_LS, coprec_eta, coprec_a1 )

    # /* NU0 modifies the output of IMRPhenomX_TimeShift_22() */
    # pWF->NU0     = XLALSimIMRPhenomXCP_NU0_l2m2(   theta_LS, coprec_eta, coprec_a1 )

    # /* NU4 modifies pPhase->cL */
    # pWF->NU4     = XLALSimIMRPhenomXCP_NU4_l2m2(   theta_LS, coprec_eta, coprec_a1 )

    # /* NU5 modifies pWF->fRING [EXTRAP-PASS-TRUE] */
    # pWF->NU5     = XLALSimIMRPhenomXCP_NU5_l2m2(   theta_LS, coprec_eta, coprec_a1 )

    # /* NU6 modifies pWF->fDAMP [EXTRAP-PASS-TRUE] */
    # pWF->NU6     = XLALSimIMRPhenomXCP_NU6_l2m2(   theta_LS, coprec_eta, coprec_a1 )

    # /* ZETA1 modifies pPhase->b4 */
    # pWF->ZETA1   = XLALSimIMRPhenomXCP_ZETA1_l2m2( theta_LS, coprec_eta, coprec_a1 )

    # /* ZETA2 modifies pPhase->b1  */
    # pWF->ZETA2   = XLALSimIMRPhenomXCP_ZETA2_l2m2( theta_LS, coprec_eta, coprec_a1 )

    # }

    # //
    # pWF->NU0 = 0

    return (p_wf, p_prec)


def imr_phenom_x_pnr_set_phase_alignment_params(
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    lal_params: IMRPhenomXPHMParameterDataClass,
) -> tuple[IMRPhenomXWaveformDataClass, IMRPhenomXPrecessionDataClass]:

    # /*
    # Copy the current state of pWF to pPrec for use in PNR+XPHM, where we will similarly want to enforce phase alignment with XHM during inspiral.

    # The code immediately below is very typical of annoying C language code:
    # to copy the structure, one first must allocate memory for the vessel to
    # hold the copy. Then one must copy the struct into that allocated
    # momory. While there are "correct" versions of this code that do not
    # require use of malloc, these versions essentially copy the pointer, so
    # when pWF is changed, so is pWF22AS. We do not want that, so the use of
    # malloc is essential.
    # */
    p_wf_22_as = copy.deepcopy(p_wf)
    # p_prec.p_wf_22_as = p_wf_22_as

    # /* Define alignment frequency in fM (aka Mf). This is the
    # frequency at which PNR coprecessing phase and phase
    # derivaive will be aligned with corresponding XAS and XHM
    # values.  */
    f_inspiral_align = 0.004

    # // NOTE that just below we compute the non-precessing phase parameters
    # // BEFORE any changes are made to pWF -- SO the pWF input must not
    # // contain any changes due to precession.
    p_phase_as = IMRPhenomXPhaseCoefficientsDataClass()
    p_phase_as = imr_phenom_x_get_phase_coefficients(p_wf, p_phase_as)

    # /*
    # Below we use IMRPhenomX_FullPhase_22 to somultaneously compute
    # the XAS phase and phase derivative at the point of interest.
    # */

    # /**********************************************************/
    # // Initialize holders for the phase and phase derivative
    # double phase, dphase
    # // Define the values inside of IMRPhenomX_FullPhase_22
    # IMRPhenomX_FullPhase_22(&phase,&dphase,pWF->f_inspiral_align,pPhaseAS,pWF)
    # // Store the phase and phase derivative for later use
    # pWF->XAS_phase_at_f_inspiral_align = phase
    # pWF->XAS_dphase_at_f_inspiral_align = dphase//full_dphase_value
    # /**********************************************************/

    # /*
    # Now, once all other model changes have been made, but before the
    # final phase is output in IMRPhenomXASGenerateFD, we want to force
    # the PNR CoPrecessing phase and phase derivative to be pWF->XAS_phase_at_f_inspiral_align and pWF->XAS_dphase_at_f_inspiral_align, respectively. This effort
    # is facilitated by IMRPhenomX_PNR_EnforceXASPhaseAlignment below.
    # */

    # // // Printing for development
    # // printf("##>> XAS_phase_at_f_inspiral_align = %f\n",pWF->XAS_phase_at_f_inspiral_align)
    # // printf("##>> XAS_dphase_at_f_inspiral_align = %f\n",pWF->XAS_dphase_at_f_inspiral_align)

    # LALFree(pPhaseAS)
    p_wf = dataclasses.replace(
        p_wf,
        f_inspiral_align=f_inspiral_align,
        p_wf_22_as=p_wf_22_as,
    )

    return p_wf, p_prec
