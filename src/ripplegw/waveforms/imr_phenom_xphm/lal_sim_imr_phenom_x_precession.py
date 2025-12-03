from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.constants import PI, C, G
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import IMRPhenomXPrecessionDataClass
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    xlal_sim_imr_phenom_x_unwrap_array,
    xlal_sim_imr_phenom_x_utils_hz_to_mf,
    xlal_sim_imr_phenom_x_utils_mf_to_hz,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass


def get_deltaF_from_wfstruct(pWF: IMRPhenomXWaveformDataClass) -> float:
    """Compute deltaF from waveform structure parameters.

    Args:
        pWF: Waveform dataclass containing fRef, m1_SI, m2_SI, chi1L, chi2L, Mtot.
    """


#   REAL8 seglen=XLALSimInspiralChirpTimeBound(pWF->fRef, pWF->m1_SI, pWF->m2_SI, pWF->chi1L,pWF->chi2L);
#   REAL8 deltaFv1= 1./MAX(4.,pow(2, ceil(log(seglen)/log(2))));
#   REAL8 deltaF = MIN(deltaFv1,0.1);
#   REAL8 deltaMF = XLALSimIMRPhenomXUtilsHztoMf(deltaF,pWF->Mtot);
#   return(deltaMF);

# }


@checkify.checkify
def imr_phenom_x_set_precession_var(
    pWF: IMRPhenomXWaveformDataClass,
    pPrec: IMRPhenomXPrecessionDataClass,
    m1_SI: float,
    m2_SI: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
    lalParams: IMRPhenomXPHMParameterDataClass,
    debug_flag: int,
) -> None:

    #   /*
    #       Here we assume m1 > m2, q > 1, dm = m1 - m2 = delta = sqrt(1-4eta) > 0
    #   */
    # pWF.lal_params = lalParams

    # /* Pre-cache useful powers here */:
    pPrec.sqrt2 = 1.4142135623730951
    pPrec.sqrt5 = 2.23606797749978981
    pPrec.sqrt6 = 2.44948974278317788
    pPrec.sqrt7 = 2.64575131106459072
    pPrec.sqrt10 = 3.16227766016838
    pPrec.sqrt14 = 3.74165738677394133
    pPrec.sqrt15 = 3.87298334620741702
    pPrec.sqrt70 = 8.36660026534075563
    pPrec.sqrt30 = 5.477225575051661
    pPrec.sqrt2p5 = 1.58113883008419

    #   pPrec->debug_prec = debug_flag

    # /* Sort out version-specific flags */

    # // Get IMRPhenomX precession version from LAL dictionary
    IMRPhenomXPrecVersion = lalParams.precession_version
    IMRPhenomXPrecVersion = jax.lax.select(IMRPhenomXPrecVersion == 300, 223, IMRPhenomXPrecVersion)

    # // default to NNLO angles if in-plane spins are negligible and one of the SpinTaylor options has been selected. The solutions would be dominated by numerical noise.
    chi_in_plane = jnp.sqrt(chi1x * chi1x + chi1y * chi1y + chi2x * chi2x + chi2y * chi2y)

    # if(chi_in_plane<1e-6 && pPrec->IMRPhenomXPrecVersion==330)
    # {
    # pPrec->IMRPhenomXPrecVersion=102
    # }
    IMRPhenomXPrecVersion = jax.lax.select(
        (chi_in_plane < 1e-6) & (IMRPhenomXPrecVersion == 330), 102, IMRPhenomXPrecVersion
    )

    # if(chi_in_plane<1e-7 && (pPrec->IMRPhenomXPrecVersion==320||pPrec->IMRPhenomXPrecVersion==321||pPrec->IMRPhenomXPrecVersion==310||pPrec->IMRPhenomXPrecVersion==311))
    # {
    # pPrec->IMRPhenomXPrecVersion=102
    # }
    IMRPhenomXPrecVersion = jax.lax.select(
        (chi_in_plane < 1e-7)
        & (
            (IMRPhenomXPrecVersion == 320)
            | (IMRPhenomXPrecVersion == 321)
            | (IMRPhenomXPrecVersion == 310)
            | (IMRPhenomXPrecVersion == 311)
        ),
        102,
        IMRPhenomXPrecVersion,
    )

    # // Get expansion order for MSA system of equations. Default is taken to be 5.
    ExpansionOrder = lalParams.expansion_order

    # // Get toggle for PNR angles
    PNRUseTunedAngles = lalParams.pnr_use_tuned_angles
    IMRPhenomXPNRUseTunedAngles = PNRUseTunedAngles

    # Get PNR angle interpolation tolerance
    IMRPhenomXPNRInterpTolerance = lalParams.pnr_interp_tolerance

    # // Get toggle for symmetric waveform
    AntisymmetricWaveform = lalParams.antisymmetric_waveform
    IMRPhenomXAntisymmetricWaveform = AntisymmetricWaveform

    # // Set toggle for polarization calculation: +1 for symmetric waveform (default), -1 for antisymmetric waveform refer to XXXX.YYYYY for details
    PolarizationSymmetry = 1.0

    # /* allow for conditional disabling of precession multibanding given mass ratio and opening angle */
    conditionalPrecMBand = 0
    MBandPrecVersion = lalParams.mband_version
    # if(MBandPrecVersion == 2){
    # MBandPrecVersion = 0 /* current default value is 0 */
    # conditionalPrecMBand = 1
    # }
    is_version_2 = MBandPrecVersion == 2
    conditionalPrecMBand = jax.lax.select(is_version_2, 1, conditionalPrecMBand)
    MBandPrecVersion = jax.lax.select(is_version_2, 0, MBandPrecVersion)

    # /* Define a number of convenient local parameters */
    m1 = m1_SI / pWF.m_tot_si  # Normalized mass of larger companion:   m1_SI / Mtot_SI
    m2 = m2_SI / pWF.m_tot_si  # Normalized mass of smaller companion:  m2_SI / Mtot_SI
    M = m1 + m2  # Total mass in solar units

    # // Useful powers of mass
    m1_2 = m1 * m1
    m1_3 = m1 * m1_2
    m1_4 = m1 * m1_3
    m1_5 = m1 * m1_4
    m1_6 = m1 * m1_5
    m1_7 = m1 * m1_6
    m1_8 = m1 * m1_7

    m2_2 = m2 * m2

    # I'm keeping this here, but note that these three lines have been moved to the setting of IMRPhenomXPHMParameterDataClass
    #   pWF->M = M
    #   pWF->m1_2 = m1_2
    #   pWF->m2_2 = m2_2

    q = m1 / m2  # q = m1 / m2 > 1.0

    # // Powers of eta
    eta = pWF.eta
    eta2 = eta * eta
    eta3 = eta * eta2
    eta4 = eta * eta3
    eta5 = eta * eta4
    eta6 = eta * eta5

    # // \delta in terms of q > 1
    delta = pWF.delta
    delta2 = delta * delta
    delta3 = delta * delta2

    # // Cache these powers, as we use them regularly
    eta = eta
    eta2 = eta2
    eta3 = eta3
    eta4 = eta4

    inveta = 1.0 / eta
    inveta2 = 1.0 / eta2
    inveta3 = 1.0 / eta3
    inveta4 = 1.0 / eta4
    sqrt_inveta = 1.0 / jnp.sqrt(eta)

    chi_eff = pWF.chi_eff

    twopiGM = 2 * PI * G * (m1_SI + m2_SI) / C**3
    piGM = PI * G * (m1_SI + m2_SI) / C**3

    # /* Set spin variables in pPrec struct */
    chi1x = chi1x
    chi1y = chi1y
    chi1_norm = jnp.sqrt(chi1x * chi1x + chi1y * chi1y + chi1z * chi1z)

    chi2x = chi2x
    chi2y = chi2y
    chi2z = chi2z
    chi2_norm = jnp.sqrt(chi2x * chi2x + chi2y * chi2y + chi2z * chi2z)

    # /* Check that spins obey Kerr bound */
    # if((!PNRUseTunedAngles)||(pWF->PNR_SINGLE_SPIN != 1)){ #/*Allow the single-spin mapping for PNR to break the Kerr limit*/
    # XLAL_CHECK(fabs(pPrec->chi1_norm) <= 1.0, XLAL_EDOM, "Error in IMRPhenomXSetPrecessionVariables: |S1/m1^2| must be <= 1.\n")
    # XLAL_CHECK(fabs(pPrec->chi2_norm) <= 1.0, XLAL_EDOM, "Error in IMRPhenomXSetPrecessionVariables: |S2/m2^2| must be <= 1.\n")
    # }
    kerr_check_cond = (not PNRUseTunedAngles) | (pWF.pnr_single_spin != 1)
    checkify.check(
        kerr_check_cond & (jnp.abs(chi1_norm) <= 1.0),
        "Error in IMRPhenomXSetPrecessionVariables: |S1/m1^2| must be <= 1.\n",
    )
    checkify.check(
        kerr_check_cond & (jnp.abs(chi2_norm) <= 1.0),
        "Error in IMRPhenomXSetPrecessionVariables: |S2/m2^2| must be <= 1.\n",
    )

    # /* Calculate dimensionful spins */
    S1x = chi1x * m1_2
    S1y = chi1y * m1_2
    S1z = chi1z * m1_2
    S1_norm = jnp.abs(chi1_norm) * m1_2

    S2x = chi2x * m2_2
    S2y = chi2y * m2_2
    S2z = chi2z * m2_2
    S2_norm = jnp.abs(chi2_norm) * m2_2

    # // Useful powers
    S1_norm_2 = S1_norm * S1_norm
    S2_norm_2 = S2_norm * S2_norm

    chi1_perp = jnp.sqrt(chi1x * chi1x + chi1y * chi1y)
    chi2_perp = jnp.sqrt(chi2x * chi2x + chi2y * chi2y)

    # /* Get spin projections */
    S1_perp = (m1_2) * jnp.sqrt(chi1x * chi1x + chi1y * chi1y)
    S2_perp = (m2_2) * jnp.sqrt(chi2x * chi2x + chi2y * chi2y)

    # /* Norm of in-plane vector sum: Norm[ S1perp + S2perp ] */
    STot_perp = jnp.sqrt((S1x + S2x) * (S1x + S2x) + (S1y + S2y) * (S1y + S2y))

    # /* This is called chiTot_perp to distinguish from Sperp used in contrusction of chi_p. For normalization, see Sec. IV D of arXiv:2004.06503 */
    chiTot_perp = STot_perp * (M * M) / m1_2
    # /* Store chiTot_perp to pWF so that it can be used in XCP modifications (PNRUseTunedCoprec) */
    pWF = dataclasses.replace(pWF, chiTot_perp=chiTot_perp)

    # /* disable tuned PNR angles, tuned coprec and mode asymmetries in low in-plane spin limit */
    # if((chi_in_plane < 1e-7)&&(pPrec->IMRPhenomXPNRUseTunedAngles == 1)&&(pWF->PNR_SINGLE_SPIN != 1)){
    # XLALSimInspiralWaveformParamsInsertPhenomXPNRUseTunedAngles(lalParams, 0)
    # PNRUseTunedAngles = 0
    # pPrec->IMRPhenomXPNRUseTunedAngles = 0
    # pPrec->IMRPhenomXAntisymmetricWaveform = 0
    # AntisymmetricWaveform = 0
    # XLALSimInspiralWaveformParamsInsertPhenomXAntisymmetricWaveform(lalParams, 0)
    # XLALSimInspiralWaveformParamsInsertPhenomXPNRUseTunedCoprec(lalParams, 0)
    # }
    low_spin_cond = (chi_in_plane < 1e-7) & (IMRPhenomXPNRUseTunedAngles == 1) & (pWF.pnr_single_spin != 1)
    lalParams = jax.lax.cond(
        low_spin_cond,
        lambda x: dataclasses.replace(x, pnr_use_tuned_angles=0, antisymmetric_waveform=0, pnr_use_tuned_coprec=0),
        lambda x: x,
        lalParams,
    )
    PNRUseTunedAngles = jax.lax.select(low_spin_cond, 0, PNRUseTunedAngles)
    IMRPhenomXAntisymmetricWaveform = jax.lax.select(low_spin_cond, 0, IMRPhenomXAntisymmetricWaveform)

    # /*
    # Calculate the effective precessing spin parameter (Schmidt et al, PRD 91, 024043, 2015):
    #     - m1 > m2, so body 1 is the larger black hole
    # */
    A1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    A2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp

    # /* S_p = max(A1 S1_perp, A2 S2_perp) */
    num = jax.lax.select(ASp2 > ASp1, ASp2, ASp1)
    den = jax.lax.select(m2 > m1, A2 * (m2_2), A1 * (m1_2))

    # /* chi_p = max(A1 * Sp1 , A2 * Sp2) / (A_i * m_i^2) where i is the index of the larger BH */
    chip = num / den
    chi1L = chi1z
    chi2L = chi2z

    chi_p = chip
    # // (PNRUseTunedCoprec)
    pWF = dataclasses.replace(pWF, chi_p=chi_p)
    phi0_aligned = pWF.phi0

    # /* Effective (dimensionful) aligned spin */
    SL = chi1L * m1_2 + chi2L * m2_2

    # /* Effective (dimensionful) in-plane spin */
    Sperp = chip * m1_2  # /* m1 > m2 */

    MSA_ERROR = 0

    # pWF22AS = NULL

    # // get first digit of precessing version: this tags the method employed to compute the Euler angles
    # // 1: NNLO 2: MSA 3: SpinTaylor (numerical)
    precversionTag = (IMRPhenomXPrecVersion - (IMRPhenomXPrecVersion % 100)) / 100

    # /* start of SpinTaylor code */

    # ######## NOTE if precversionTag==3: ######## -> Spin-Taylor

    # # // allocate memory to store arrays with results of the PN precession equations
    # # status=XLAL_SUCCESS
    # # pPrec->PNarrays = XLALMalloc(sizeof(PhenomXPInspiralArrays))
    # # L_MAX_PNR=pPrec->M_MAX

    # # // check mode array to estimate frequency range over which splines will need to be evaluated
    # ModeArray = lalParams.mode_array

    # # if (ModeArray != NULL)
    # # {
    # # if(XLALSimInspiralModeArrayIsModeActive(ModeArray, 4, 4)){
    # #     LMAX_PNR = 4
    # # }
    # # else if((XLALSimInspiralModeArrayIsModeActive(ModeArray, 3, 3))||(XLALSimInspiralModeArrayIsModeActive(ModeArray, 3, 2))){
    # #     LMAX_PNR = 3
    # # }
    # # pPrec->L_MAX_PNR = LMAX_PNR
    # # IMRPhenomX_GetandSetModes(ModeArray,pPrec)
    # # XLALDestroyValue(ModeArray)
    # # }

    # LMAX_PNR = jax.lax.select(
    #     ModeArray is not None,
    #     jax.lax.select(
    #         xlal_sim_inspiral_mode_array_is_mode_active(ModeArray, 4, 4),
    #         4,
    #         jax.lax.select(
    #             xlal_sim_inspiral_mode_array_is_mode_active(ModeArray, 3, 3) | xlal_sim_inspiral_mode_array_is_mode_active(ModeArray, 3, 2),
    #             3,
    #             2
    #         )
    #     ),
    #     2
    # )
    # L_MAX_PNR = LMAX_PNR

    # # // buffer for GSL interpolation to succeed
    # # // set first to fMin
    # flow = pWF.f_min

    # # if(pWF.deltaF==0.) pWF.deltaMF = get_deltaF_from_wfstruct(pWF)

    # # // if PNR angles are disabled, step back accordingly to the waveform's frequency grid step
    # # if(PNRUseTunedAngles==false)
    # # {

    # # pPrec->integration_buffer = (pWF->deltaF>0.)? 3.*pWF->deltaF: 0.5
    # # flow = (pWF->fMin-pPrec->integration_buffer)*2./pPrec->M_MAX

    # # }
    # # # // if PNR angles are enabled, adjust buffer to the requirements of IMRPhenomX_PNR_GeneratePNRAngleInterpolants
    # # else{

    # # size_t iStart_here

    # # if (pWF->deltaF == 0.) iStart_here = 0
    # # else{
    # #     iStart_here= (size_t)(pWF->fMin / pWF->deltaF)
    # #     flow = iStart_here * pWF->deltaF
    # # }

    # min_HM_inspiral = flow * 2.0 / M_MAX

    # INT4 precVersion = pPrec->IMRPhenomXPrecVersion
    # // fill in a fake value to allow the next code to work
    # pPrec->IMRPhenomXPrecVersion = 223
    # status = IMRPhenomX_PNR_GetAndSetPNRVariables(pWF, pPrec)
    # XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Error: IMRPhenomX_PNR_GetAndSetPNRVariables failed in IMRPhenomXGetAndSetPrecessionVariables.\n")

    # /* generate alpha parameters to catch edge cases */
    # IMRPhenomX_PNR_alpha_parameters *alphaParams = XLALMalloc(sizeof(IMRPhenomX_PNR_alpha_parameters))
    # IMRPhenomX_PNR_beta_parameters *betaParams = XLALMalloc(sizeof(IMRPhenomX_PNR_beta_parameters))
    # status = IMRPhenomX_PNR_precompute_alpha_coefficients(alphaParams, pWF, pPrec)
    # XLAL_CHECK(
    #     XLAL_SUCCESS == status,
    #     XLAL_EFUNC,
    #     "Error: IMRPhenomX_PNR_precompute_alpha_coefficients failed.\n")
    # status = IMRPhenomX_PNR_precompute_beta_coefficients(betaParams, pWF, pPrec)
    # XLAL_CHECK(
    #     XLAL_SUCCESS == status,
    #     XLAL_EFUNC,
    #     "Error: IMRPhenomX_PNR_precompute_beta_coefficients failed.\n")
    # status = IMRPhenomX_PNR_BetaConnectionFrequencies(betaParams)
    # XLAL_CHECK(
    #     XLAL_SUCCESS == status,
    #     XLAL_EFUNC,
    #     "Error: IMRPhenomX_PNR_BetaConnectionFrequencies failed.\n")
    # pPrec->IMRPhenomXPrecVersion = precVersion
    # REAL8 Mf_alpha_upper = alphaParams->A4 / 3.0
    # REAL8 Mf_low_cut = (3.0 / 3.5) * Mf_alpha_upper
    # REAL8 MF_high_cut = betaParams->Mf_beta_lower
    # LALFree(alphaParams)
    # LALFree(betaParams)

    # if((MF_high_cut > pWF->fCutDef) || (MF_high_cut < 0.1 * pWF->fRING)){
    #     MF_high_cut = pWF->fRING
    # }
    # if((Mf_low_cut > pWF->fCutDef) || (MF_high_cut < Mf_low_cut)){
    #     Mf_low_cut = MF_high_cut / 2.0
    # }

    # REAL8 flow_alpha = XLALSimIMRPhenomXUtilsMftoHz(Mf_low_cut * 0.65 * pPrec->M_MAX / 2.0, pWF->Mtot)

    # if(flow_alpha < flow){
    #     // flow is approximately in the intermediate region of the frequency map
    #     // conservatively reduce flow to account for potential problems in this region
    #     flow = fmin_HM_inspiral / 1.5
    # }
    # else{
    #     REAL8 Mf_RD_22 = pWF->fRING
    #     REAL8 Mf_RD_lm = IMRPhenomXHM_GenerateRingdownFrequency(pPrec->L_MAX_PNR, pPrec->M_MAX, pWF)
    #     REAL8 fmin_HM_ringdowm = XLALSimIMRPhenomXUtilsMftoHz(XLALSimIMRPhenomXUtilsHztoMf(flow, pWF->Mtot) - (Mf_RD_lm - Mf_RD_22), pWF->Mtot)
    #     flow = ((fmin_HM_ringdowm < fmin_HM_inspiral)&&(fmin_HM_ringdowm > 0.0)) ? fmin_HM_ringdowm : fmin_HM_inspiral
    # }

    # double pnr_interpolation_deltaf = IMRPhenomX_PNR_HMInterpolationDeltaF(flow, pWF, pPrec)
    # pPrec->integration_buffer = 1.4*pnr_interpolation_deltaf
    # flow = (flow - 2.0 * pnr_interpolation_deltaf < 0) ? flow / 2.0 : flow - 2.0 * pnr_interpolation_deltaf

    # iStart_here = (size_t)(flow / pnr_interpolation_deltaf)
    # flow = iStart_here * pnr_interpolation_deltaf
    # }

    # XLAL_CHECK(flow>0.,XLAL_EDOM,"Error in %s: starting frequency for SpinTaylor angles must be positive!",__func__)
    # status = IMRPhenomX_InspiralAngles_SpinTaylor(pPrec->PNarrays,&pPrec->fmin_integration,chi1x,chi1y,chi1z,chi2x,chi2y,chi2z,flow,pPrec->IMRPhenomXPrecVersion,pWF,lalParams)
    # // convert the min frequency of integration to geometric units for later convenience
    # pPrec->Mfmin_integration = XLALSimIMRPhenomXUtilsHztoMf(pPrec->fmin_integration,pWF->Mtot)

    # if (pPrec->IMRPhenomXPrecVersion == 330)
    # {

    # REAL8 chi1x_evolved = chi1x
    # REAL8 chi1y_evolved = chi1y
    # REAL8 chi1z_evolved = chi1z
    # REAL8 chi2x_evolved = chi2x
    # REAL8 chi2y_evolved = chi2y
    # REAL8 chi2z_evolved = chi2z

    # // in case that SpinTaylor angles generate, overwrite variables with evolved spins
    # if(status!=XLAL_FAILURE)  {
    #     size_t lenPN = pPrec->PNarrays->V_PN->data->length

    #     REAL8 chi1x_temp = pPrec->PNarrays->S1x_PN->data->data[lenPN-1]
    #     REAL8 chi1y_temp = pPrec->PNarrays->S1y_PN->data->data[lenPN-1]
    #     REAL8 chi1z_temp = pPrec->PNarrays->S1z_PN->data->data[lenPN-1]

    #     REAL8 chi2x_temp = pPrec->PNarrays->S2x_PN->data->data[lenPN-1]
    #     REAL8 chi2y_temp = pPrec->PNarrays->S2y_PN->data->data[lenPN-1]
    #     REAL8 chi2z_temp = pPrec->PNarrays->S2z_PN->data->data[lenPN-1]

    #     REAL8 Lx = pPrec->PNarrays->LNhatx_PN->data->data[lenPN-1]
    #     REAL8 Ly = pPrec->PNarrays->LNhaty_PN->data->data[lenPN-1]
    #     REAL8 Lz = pPrec->PNarrays->LNhatz_PN->data->data[lenPN-1]

    #     // orbital separation vector not stored in PN arrays
    #     //REAL8 nx = pPrec->PNarrays->E1x->data->data[lenPN-1]
    #     //REAL8 ny = pPrec->PNarrays->E1y->data->data[lenPN-1]

    #     // rotate to get x,y,z components in L||z frame
    #     REAL8 phi = atan2( Ly, Lx )
    #     REAL8 theta = acos( Lz / sqrt(Lx*Lx + Ly*Ly + Lz*Lz) )
    #     //REAL8 kappa = atan( ny/nx )

    #     IMRPhenomX_rotate_z(-phi, &chi1x_temp, &chi1y_temp, &chi1z_temp)
    #     IMRPhenomX_rotate_y(-theta, &chi1x_temp, &chi1y_temp, &chi1z_temp)
    #     //IMRPhenomX_rotate_z(-kappa, &chi1x_temp, &chi1y_temp, &chi1z_temp)

    #     IMRPhenomX_rotate_z(-phi, &chi2x_temp, &chi2y_temp, &chi2z_temp)
    #     IMRPhenomX_rotate_y(-theta, &chi2x_temp, &chi2y_temp, &chi2z_temp)
    #     //IMRPhenomX_rotate_z(-kappa, &chi2x_temp, &chi2y_temp, &chi2z_temp)

    #     chi1x_evolved = chi1x_temp
    #     chi1y_evolved = chi1y_temp
    #     chi1z_evolved = chi1z_temp

    #     chi2x_evolved = chi2x_temp
    #     chi2y_evolved = chi2y_temp
    #     chi2z_evolved = chi2z_temp
    # }

    # pPrec->chi1x_evolved = chi1x_evolved
    # pPrec->chi1y_evolved = chi1y_evolved
    # pPrec->chi1z_evolved = chi1z_evolved
    # pPrec->chi2x_evolved = chi2x_evolved
    # pPrec->chi2y_evolved = chi2y_evolved
    # pPrec->chi2z_evolved = chi2z_evolved

    # //printf("%f, %f, %f, %f, %f, %f\n", chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
    # //printf("%f, %f, %f, %f, %f, %f\n", chi1x_evolved, chi1y_evolved, chi1z_evolved, chi2x_evolved, chi2y_evolved, chi2z_evolved)
    # //printf("----\n")
    # }

    # // if PN numerical integration fails, default to MSA+fallback to NNLO
    # if(status==XLAL_FAILURE) {
    #                         LALFree(pPrec->PNarrays)
    #                         XLAL_PRINT_WARNING("Warning: due to a failure in the SpinTaylor routines, the model will default to MSA angles.")
    #                         pPrec->IMRPhenomXPrecVersion=223
    #                         }
    # // end of SpinTaylor code


#             }


#   /* update  precessing version to catch possible fallbacks of SpinTaylor angles */
#   precversionTag=(pPrec->IMRPhenomXPrecVersion-(pPrec->IMRPhenomXPrecVersion%100))/100
#   int pflag = pPrec->IMRPhenomXPrecVersion


#   if(pflag != 101 && pflag != 102 && pflag != 103 && pflag != 104 && pflag != 220 && pflag != 221 && pflag != 222 && pflag != 223 && pflag != 224 && pflag!=310 && pflag!=311 && pflag!=320 && pflag!=321 && pflag!=330)
#   {
#     XLAL_ERROR(XLAL_EINVAL, "Error in IMRPhenomXGetAndSetPrecessionVariables: Invalid precession flag. Allowed versions are 101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321 or 330.\n")
#   }

#   switch( pflag )
#     {
#         case 101: // NNLO single spin PNEuler angles + 2PN non-spinning L
#         case 102: // NNLO single spin PNEuler angles + 3PN spinning L
#         case 103: // NNLO single spin PNEuler angles + 4PN spinning L
#         case 104: // NNLO single spin PNEuler angles + 4PN spinning L + LOS terms in L
#     {
#       break
#     }
#     case 220: // MSA using expressions as detailed in arXiv:1703.03967. Defaults to NNLO v102 if MSA fails.
#     case 221: // MSA using expressions as detailed in arXiv:1703.03967. Terminal failure if MSA fails.
#     case 222: // MSA using expressions as implemented in LALSimInspiralFDPrecAngles. Terminal failure if MSA fails.
#     case 223: // MSA using expressions as implemented in LALSimInspiralFDPrecAngles. Defaults to NNLO v102 if MSA fails.
#     case 224: // MSA using expressions as detailed in arXiv:1703.03967, with \zeta_0 and \phi_{z,0} as in LALSimInspiralFDPrecAngles.  Defaults to NNLO v102 if MSA fails.
#     {
#        /*
#           Double-spin model using angles from Chatziioannou et al, PRD, 95, 104004, (2017), arXiv:1703.03967
#           Uses 3PN L
#        */
#        #if DEBUG == 1
#         printf("Initializing MSA system...\n")
#        #endif

#        if(pPrec->ExpansionOrder < -1 || pPrec->ExpansionOrder > 5)
#        {
#          XLAL_ERROR(XLAL_EINVAL, "Error in IMRPhenomXGetAndSetPrecessionVariables: Invalid expansion order for MSA corrections. Default is 5, allowed values are [-1,0,1,2,3,4,5].\n")
#        }
#        break

#     }

#     case 310: // Numerical integration of SpinTaylor equations, constant angles in MRD
#     case 311: // Numerical integration of SpinTaylor equations, constant angles in MRD, BBH precession
#     case 320: // Numerical integration of SpinTaylor equations, analytical continuation in MRD
#     case 321: // Numerical integration of SpinTaylor equations, analytical continuation in MRD, BBH precession
#     case 330: // Numerical integration of SpinTaylor equations, PNR angles, analytic joining
#         {
#            break
#         }


#         default:
#         {
#             XLAL_ERROR(XLAL_EINVAL, "Error in IMRPhenomXGetAndSetPrecessionVariables: IMRPhenomXPrecessionVersion not recognized.\n")
#       break
#         }
#     }


#   pPrec->precessing_tag=precversionTag


#   /* Calculate parameter for two-spin to single-spin map used in PNR and XCP */
#   /* Initialize PNR variables */
#   pPrec->chi_singleSpin = 0.0
#   pPrec->costheta_singleSpin = 0.0
#   pPrec->costheta_final_singleSpin = 0.0
#   pPrec->chi_singleSpin_antisymmetric = 0.0
#   pPrec->theta_antisymmetric = 0.0
#   pPrec->PNR_HM_Mflow = 0.0
#   pPrec->PNR_HM_Mfhigh = 0.0

#   pPrec->PNR_q_window_lower = 0.0
#   pPrec->PNR_q_window_upper = 0.0
#   pPrec->PNR_chi_window_lower = 0.0
#   pPrec->PNR_chi_window_upper = 0.0
#   // pPrec->PNRInspiralScaling = 0

#   UINT4 status = IMRPhenomX_PNR_GetAndSetPNRVariables(pWF, pPrec)
#   XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Error: IMRPhenomX_PNR_GetAndSetPNRVariables failed in IMRPhenomXGetAndSetPrecessionVariables.\n")

#   pPrec->alphaPNR = 0.0
#   pPrec->betaPNR = 0.0
#   pPrec->gammaPNR = 0.0

#   /*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
#   /      Get and/or store CoPrec params into pWF and pPrec     /
#   /...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

#   status = IMRPhenomX_PNR_GetAndSetCoPrecParams(pWF,pPrec,lalParams)
#   XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC,
#   "Error: IMRPhenomX_PNR_GetAndSetCoPrecParams failed \
#   in IMRPhenomXGetAndSetPrecessionVariables.\n")

#   /*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/


#   //
#   if( pflag == 220 || pflag == 221 || pflag == 222 || pflag == 223 || pflag == 224 )
#     {
#       #if DEBUG == 1
#         printf("Evaluating MSA system.\n")
#         printf("Expansion Order : %d\n",pPrec->ExpansionOrder)
#       #endif

#       IMRPhenomX_Initialize_MSA_System(pWF,pPrec,pPrec->ExpansionOrder)

#       if(pPrec->MSA_ERROR == 1)
#       {
#         // In version 220, 223 and 224 if the MSA system fails to initialize we default to the NNLO PN angles using the 3PN aligned-spin orbital angular momentum
#         if(pflag == 220 || pflag == 223 || pflag == 224)
#         {
#           XLAL_PRINT_WARNING("Warning: Initialization of MSA system failed. Defaulting to NNLO angles using 3PN aligned-spin approximation.")
#           pPrec->IMRPhenomXPrecVersion = 102
#           pflag  = pPrec->IMRPhenomXPrecVersion
#         }
#         else // Otherwise, if the MSA system fails to initialize we trigger a terminal error
#         {
#           XLAL_ERROR(XLAL_EDOM,"Error: IMRPhenomX_Initialize_MSA_System failed to initialize. Terminating.\n")
#         }
#       }
#     }

#   #if DEBUG == 1
#     printf("In IMRPhenomXSetPrecessionVariables... \n\n")
#     printf("chi_p   : %e\n",pPrec->chi_p)
#     printf("phic    : %e\n",pPrec->phi0_aligned)
#     printf("SL      : %e\n",pPrec->SL)
#     printf("Sperp   : %e\n\n",pPrec->Sperp)
#   #endif

#   /*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
#   /      Compute and set final spin and RD frequency           /
#   /...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/
#   IMRPhenomX_SetPrecessingRemnantParams(pWF,pPrec,lalParams)
#   /*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

#   /* Useful powers of \chi_p */
#   const REAL8 chip2    = chip * chip

#   /* Useful powers of spins aligned with L */
#   const REAL8 chi1L2   = chi1L * chi1L
#   const REAL8 chi2L2   = chi2L * chi2L

#   const REAL8 log16    = 2.772588722239781

#   /*  Cache the orbital angular momentum coefficients for future use.

#       References:
#         - Kidder, PRD, 52, 821-847, (1995), arXiv:gr-qc/9506022
#         - Blanchet, LRR, 17, 2, (2014), arXiv:1310.1528
#         - Bohe et al, 1212.5520v2
#         - Marsat, CQG, 32, 085008, (2015), arXiv:1411.4118
#   */
#   switch( pflag )
#   {
#     /* 2PN non-spinning orbital angular momentum (as per IMRPhenomPv2) */
#     case 101:
#     {
#       pPrec->L0   = 1.0
#       pPrec->L1   = 0.0
#       pPrec->L2   = ((3.0/2.0) + (eta/6.0))
#       pPrec->L3   = 0.0
#       pPrec->L4   = (81.0 + (-57.0 + eta)*eta)/24.
#       pPrec->L5   = 0.0
#       pPrec->L6   = 0.0
#       pPrec->L7   = 0.0
#       pPrec->L8   = 0.0
#       pPrec->L8L  = 0.0
#       break
#     }
#     /* 3PN orbital angular momentum */
#     case 102:
#     case 220:
#     case 221:
#     case 224:
#     case 310:
#     case 311:
#     case 320:
#     case 321:
#     case 330:
#     {
#       pPrec->L0   = 1.0
#       pPrec->L1   = 0.0
#       pPrec->L2   = 3.0/2. + eta/6.0
#       pPrec->L3   = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
#       pPrec->L4   = (81 + (-57 + eta)*eta)/24.
#       pPrec->L5   = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
#       pPrec->L6   = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*powers_of_lalpi.two))/1296.
#       pPrec->L7   = 0.0
#       pPrec->L8   = 0.0

#       // This is the log(x) term
#       pPrec->L8L  = 0.0
#       break

#     }
#     /* 3PN orbital angular momentum using non-conserved spin norms as per LALSimInspiralFDPrecAngles.c  */
#     case 222:
#     case 223:
#     {
#       pPrec->L0   = 1.0
#       pPrec->L1   = 0.0
#       pPrec->L2   = 3.0/2. + eta/6.0
#       pPrec->L3   = (-7*(chi1L + chi2L + chi1L*delta - chi2L*delta) + 5*(chi1L + chi2L)*eta)/6.
#       pPrec->L4   = (81 + (-57 + eta)*eta)/24.
#       pPrec->L5   = (-1650*(chi1L + chi2L + chi1L*delta - chi2L*delta) + 1336*(chi1L + chi2L)*eta + 511*(chi1L - chi2L)*delta*eta + 28*(chi1L + chi2L)*eta2)/600.
#       pPrec->L6   = (10935 + eta*(-62001 + 1674*eta + 7*eta2 + 2214*powers_of_lalpi.two))/1296.
#       pPrec->L7   = 0.0
#       pPrec->L8   = 0.0

#       // This is the log(x) term
#       pPrec->L8L  = 0.0
#       break
#     }
#     /* 4PN orbital angular momentum */
#     case 103:
#     {
#       pPrec->L0   = 1.0
#       pPrec->L1   = 0.0
#       pPrec->L2   = 3.0/2. + eta/6.0
#       pPrec->L3   = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
#       pPrec->L4   = (81 + (-57 + eta)*eta)/24.
#       pPrec->L5   = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
#       pPrec->L6   = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*powers_of_lalpi.two))/1296.
#       pPrec->L7   = (chi2L*(-324 + eta*(1119 - 2*eta*(172 + eta)) + delta*(324 + eta*(-633 + 14*eta)))
#                           - chi1L*(324 + eta*(-1119 + 2*eta*(172 + eta)) + delta*(324 + eta*(-633 + 14*eta))))/32.
#       pPrec->L8   = 2835/128. - (eta*(-10677852 + 100*eta*(-640863 + eta*(774 + 11*eta))
#                       + 26542080*LAL_GAMMA + 675*(3873 + 3608*eta)*powers_of_lalpi.two))/622080. - (64*eta*log16)/3.

#       pPrec->L8L  = -(64.0/3.0) * eta
#       break
#     }
#     /*
#         4PN orbital angular momentum + leading order in spin at all PN orders terms.
#           - Marsat, CQG, 32, 085008, (2015), arXiv:1411.4118
#           - Siemonsen et al, PRD, 97, 064010, (2018), arXiv:1606.08832
#     */
#     case 104:
#     {
#       pPrec->L0   = 1.0
#       pPrec->L1   = 0.0
#       pPrec->L2   = 3.0/2. + eta/6.0
#       pPrec->L3   = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
#       pPrec->L4   = (81 + (-57 + eta)*eta)/24.
#       pPrec->L5   = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
#       pPrec->L6   = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*powers_of_lalpi.two))/1296.
#       pPrec->L7   = (chi2L*(-324 + eta*(1119 - 2*eta*(172 + eta)) + delta*(324 + eta*(-633 + 14*eta)))
#                           - chi1L*(324 + eta*(-1119 + 2*eta*(172 + eta)) + delta*(324 + eta*(-633 + 14*eta))))/32.
#       pPrec->L8   = 2835/128. - (eta*(-10677852 + 100*eta*(-640863 + eta*(774 + 11*eta))
#                       + 26542080*LAL_GAMMA + 675*(3873 + 3608*eta)*powers_of_lalpi.two))/622080. - (64*eta*log16)/3.

#       // This is the log(x) term at 4PN, x^4/2 * log(x)
#       pPrec->L8L  = -(64.0/3.0) * eta

#       // Leading order in spin at all PN orders, note that the 1.5PN terms are already included. Here we have additional 2PN and 3.5PN corrections.
#       pPrec->L4  += (chi1L2*(1 + delta - 2*eta) + 4*chi1L*chi2L*eta - chi2L2*(-1 + delta + 2*eta))/2.
#       pPrec->L7  +=  (3*(chi1L + chi2L)*eta*(chi1L2*(1 + delta - 2*eta) + 4*chi1L*chi2L*eta - chi2L2*(-1 + delta + 2*eta)))/4.

#       break
#     }

#     default:
#     {
#       XLAL_ERROR(XLAL_EINVAL,"Error: IMRPhenomXPrecVersion not recognized. Requires version 101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321 or 330.\n")
#       break
#     }
#   }

#   /* Reference orbital angular momentum */
#   pPrec->LRef = M * M * XLALSimIMRPhenomXLPNAnsatz(pWF->v_ref, pWF->eta / pWF->v_ref, pPrec->L0, pPrec->L1, pPrec->L2, pPrec->L3, pPrec->L4, pPrec->L5, pPrec->L6, pPrec->L7, pPrec->L8, pPrec->L8L)

#   /*
#     In the following code block we construct the conventions that relate the source frame and the LAL frame.

#     A detailed discussion of the conventions can be found in Appendix C and D of arXiv:2004.06503 and https://dcc.ligo.org/LIGO-T1500602
#   */

#   /* Get source frame (*_Sf) J = L + S1 + S2. This is an instantaneous frame in which L is aligned with z */
#   pPrec->J0x_Sf = (m1_2)*chi1x + (m2_2)*chi2x
#   pPrec->J0y_Sf = (m1_2)*chi1y + (m2_2)*chi2y
#   pPrec->J0z_Sf = (m1_2)*chi1z + (m2_2)*chi2z + pPrec->LRef

#   pPrec->J0     = sqrt(pPrec->J0x_Sf*pPrec->J0x_Sf + pPrec->J0y_Sf*pPrec->J0y_Sf + pPrec->J0z_Sf*pPrec->J0z_Sf)

#   /* Get angle between J0 and LN (z-direction) */
#   if(pPrec->J0 < 1e-10)
#   {
#     XLAL_PRINT_WARNING("Warning: |J0| < 1e-10. Setting thetaJ = 0.\n")
#     pPrec->thetaJ_Sf = 0.0
#   }
#   else
#   {
#     pPrec->thetaJ_Sf = acos(pPrec->J0z_Sf / pPrec->J0)
#   }

#   const double phiRef = pWF->phiRef_In

#   INT4 convention     = XLALSimInspiralWaveformParamsLookupPhenomXPConvention(lalParams)

#   if ( !(convention == 0 || convention == 1 || convention == 5 || convention == 6 || convention == 7))
#   {
#     XLAL_ERROR(XLAL_EINVAL,"Error: IMRPhenomXPConvention not recognized. Requires version 0, 1, 5, 6 or 7.\n")
#   }

#   #if DEBUG == 1
#     printf("\n*** Convention = %i\n", convention)
#   #endif

#   /* Get azimuthal angle of J0 in the source frame */
#   if(fabs(pPrec->J0x_Sf) < MAX_TOL_ATAN && fabs(pPrec->J0y_Sf) < MAX_TOL_ATAN)
#   {
#       #if DEBUG == 1
#         printf("\nAligned spin limit!\n")
#       #endif

#       /* Impose the aligned spin limit */
#       switch(convention)
#       {
#         case 0:
#         case 5:
#         {
#           pPrec->phiJ_Sf = LAL_PI/2.0 - phiRef
#           break
#         }
#         case 1:
#         case 6:
#         case 7:
#         {
#           pPrec->phiJ_Sf = 0
#           break
#         }

#       }
#   }
#   else
#   {
#       pPrec->phiJ_Sf = atan2(pPrec->J0y_Sf, pPrec->J0x_Sf) /* azimuthal angle of J0 in the source frame */
#   }
#   pPrec->phi0_aligned = - pPrec->phiJ_Sf

#   switch(convention)
#   {
#     case 0:
#     {
#       pWF->phi0 = pPrec->phi0_aligned
#       break
#     }
#     case 1:
#     {
#       pWF->phi0 = 0
#       break
#     }
#     case 5:
#     case 6:
#     case 7:
#     {
#       break
#     }
#   }

#   /*
#       Here we follow the same prescription as in IMRPhenomPv2:

#       Now rotate from SF to J frame to compute alpha0, the azimuthal angle of LN, as well as
#       thetaJ, the angle between J and N.

#       The J frame is defined by imposing that J points in the z-direction and the line of sight N is in the xz-plane
#       (with positive projection along x).

#       The components of any vector in the (new) J-frame can be obtained by rotation from the (old) source frame (SF).
#       This is done by multiplying by: RZ[-kappa].RY[-thetaJ].RZ[-phiJ]

#       Note that kappa is determined by rotating N with RY[-thetaJ].RZ[-phiJ], which brings J to the z-axis, and
#       taking the opposite of the azimuthal angle of the rotated N.
#   */

#   /* Determine kappa via rotations, as above */
#   pPrec->Nx_Sf = sin(pWF->inclination)*cos((LAL_PI / 2.0) - phiRef)
#   pPrec->Ny_Sf = sin(pWF->inclination)*sin((LAL_PI / 2.0) - phiRef)
#   pPrec->Nz_Sf = cos(pWF->inclination)

#   REAL8 tmp_x = pPrec->Nx_Sf
#   REAL8 tmp_y = pPrec->Ny_Sf
#   REAL8 tmp_z = pPrec->Nz_Sf

#   IMRPhenomX_rotate_z(-pPrec->phiJ_Sf,   &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_y(-pPrec->thetaJ_Sf, &tmp_x, &tmp_y, &tmp_z)

#   /* Note difference in overall - sign w.r.t PhenomPv2 code */
#   pPrec->kappa = XLALSimIMRPhenomXatan2tol(tmp_y,tmp_x, MAX_TOL_ATAN)

#   /* Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1} */
#   tmp_x = 0.0
#   tmp_y = 0.0
#   tmp_z = 1.0
#   IMRPhenomX_rotate_z(-pPrec->phiJ_Sf,   &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_y(-pPrec->thetaJ_Sf, &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_z(-pPrec->kappa,     &tmp_x, &tmp_y, &tmp_z)

#   if (fabs(tmp_x) < MAX_TOL_ATAN && fabs(tmp_y) < MAX_TOL_ATAN)
#   {
#       /* This is the aligned spin case */
#       #if DEBUG == 1
#         printf("\nAligned-spin case.\n")
#       #endif

#       switch(convention)
#       {
#         case 0:
#         case 5:
#         {
#           pPrec->alpha0 = LAL_PI
#           break
#         }
#         case 1:
#         case 6:
#         case 7:
#         {
#           pPrec->alpha0 = LAL_PI - pPrec->kappa
#           break
#         }
#       }
#   }
#   else
#   {
#       switch(convention)
#       {
#         case 0:
#         case 5:
#         {
#           pPrec->alpha0 = atan2(tmp_y,tmp_x)
#           break
#         }
#         case 1:
#         case 6:
#         case 7:
#         {
#           pPrec->alpha0 = LAL_PI - pPrec->kappa
#           break
#         }
#       }
#   }


#   switch(convention)
#   {
#     case 0:
#     case 5:
#     {
#         /* Now determine thetaJN by rotating N */
#         tmp_x = pPrec->Nx_Sf
#         tmp_y = pPrec->Ny_Sf
#         tmp_z = pPrec->Nz_Sf
#         IMRPhenomX_rotate_z(-pPrec->phiJ_Sf,   &tmp_x, &tmp_y, &tmp_z)
#         IMRPhenomX_rotate_y(-pPrec->thetaJ_Sf, &tmp_x, &tmp_y, &tmp_z)
#         IMRPhenomX_rotate_z(-pPrec->kappa,     &tmp_x, &tmp_y, &tmp_z)

#         /* We don't need the y-component but we will store it anyway */
#         pPrec->Nx_Jf = tmp_x
#         pPrec->Ny_Jf = tmp_y
#         pPrec->Nz_Jf = tmp_z

#         /* This is a unit vector, so no normalization */
#         pPrec->thetaJN = acos(pPrec->Nz_Jf)
#         break
#     }
#     case 1:
#     case 6:
#     case 7:
#     {
#         REAL8 J0dotN     = (pPrec->J0x_Sf * pPrec->Nx_Sf) + (pPrec->J0y_Sf * pPrec->Ny_Sf) + (pPrec->J0z_Sf * pPrec->Nz_Sf)
#         pPrec->thetaJN   = acos( J0dotN / pPrec->J0 )
#         pPrec->Nz_Jf     = cos(pPrec->thetaJN)
#         pPrec->Nx_Jf     = sin(pPrec->thetaJN)
#         break
#     }
#   }


#   /*
#       Define the polarizations used. This follows the conventions adopted for IMRPhenomPv2.

#       The IMRPhenomP polarizations are defined following the conventions in Arun et al (arXiv:0810.5336),
#       i.e. projecting the metric onto the P, Q, N triad defining where: P = (N x J) / |N x J|.

#       However, the triad X,Y,N used in LAL (the "waveframe") follows the definition in the
#       NR Injection Infrastructure (Schmidt et al, arXiv:1703.01076).

#       The triads differ from each other by a rotation around N by an angle \zeta. We therefore need to rotate
#       the polarizations by an angle 2 \zeta.
#   */
#   pPrec->Xx_Sf = -cos(pWF->inclination) * sin(phiRef)
#   pPrec->Xy_Sf = -cos(pWF->inclination) * cos(phiRef)
#   pPrec->Xz_Sf = +sin(pWF->inclination)

#   tmp_x = pPrec->Xx_Sf
#   tmp_y = pPrec->Xy_Sf
#   tmp_z = pPrec->Xz_Sf

#   IMRPhenomX_rotate_z(-pPrec->phiJ_Sf,   &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_y(-pPrec->thetaJ_Sf, &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_z(-pPrec->kappa,     &tmp_x, &tmp_y, &tmp_z)


#   /*
#       The components tmp_i are now the components of X in the J frame.

#       We now need the polar angle of this vector in the P, Q basis of Arun et al:

#           P = (N x J) / |NxJ|

#       Note, that we put N in the (pos x)z half plane of the J frame
#   */

#   switch(convention)
#   {
#     case 0:
#     case 5:
#     {
#       /* Get polar angle of X vector in J frame in the P,Q basis of Arun et al */
#       pPrec->PArunx_Jf = +0.0
#       pPrec->PAruny_Jf = -1.0
#       pPrec->PArunz_Jf = +0.0

#       /* Q = (N x P) by construction */
#       pPrec->QArunx_Jf =  pPrec->Nz_Jf
#       pPrec->QAruny_Jf =  0.0
#       pPrec->QArunz_Jf = -pPrec->Nx_Jf
#       break
#     }
#     case 1:
#     case 6:
#     case 7:
#     {
#       /* Get polar angle of X vector in J frame in the P,Q basis of Arun et al */
#       pPrec->PArunx_Jf = pPrec->Nz_Jf
#       pPrec->PAruny_Jf = 0
#       pPrec->PArunz_Jf = -pPrec->Nx_Jf

#       /* Q = (N x P) by construction */
#       pPrec->QArunx_Jf =  0
#       pPrec->QAruny_Jf =  1
#       pPrec->QArunz_Jf =  0
#       break
#     }
#   }

#   // (X . P)
#   pPrec->XdotPArun = (tmp_x * pPrec->PArunx_Jf) + (tmp_y * pPrec->PAruny_Jf) + (tmp_z * pPrec->PArunz_Jf)

#   // (X . Q)
#   pPrec->XdotQArun = (tmp_x * pPrec->QArunx_Jf) + (tmp_y * pPrec->QAruny_Jf) + (tmp_z * pPrec->QArunz_Jf)

#   /* Now get the angle zeta */
#   pPrec->zeta_polarization = atan2(pPrec->XdotQArun, pPrec->XdotPArun)

#   /* ********** PN Euler Angle Coefficients ********** */
#   /*
#       This uses the single spin PN Euler angles as per IMRPhenomPv2
#   */

#   /* ********** PN Euler Angle Coefficients ********** */
#   switch( pflag )
#   {
#     case 101:
#     case 102:
#     case 103:
#     case 104:
#     {
#       /*
#           This uses the single spin PN Euler angles as per IMRPhenomPv2
#       */

#       /* Post-Newtonian Euler Angles: alpha */
#       REAL8 chiL       = (1.0 + q) * (chi_eff / q)
#       REAL8 chiL2      = chiL * chiL

#       pPrec->alpha1    = -35/192. + (5*delta)/(64.*m1)

#       pPrec->alpha2    = ((15*chiL*delta*m1)/128. - (35*chiL*m1_2)/128.)/eta

#       pPrec->alpha3    = -5515/3072. + eta*(-515/384. - (15*delta2)/(256.*m1_2)
#                           + (175*delta)/(256.*m1)) + (4555*delta)/(7168.*m1)
#                           + ((15*chip2*delta*m1_3)/128. - (35*chip2*m1_4)/128.)/eta2

#       /* This is the term proportional to log(w) */
#       pPrec->alpha4L   = ((5*chiL*delta2)/16. - (5*chiL*delta*m1)/3. + (2545*chiL*m1_2)/1152.
#                           + ((-2035*chiL*delta*m1)/21504.
#                           + (2995*chiL*m1_2)/9216.)/eta + ((5*chiL*chip2*delta*m1_5)/128.
#                           - (35*chiL*chip2*m1_6)/384.)/eta3
#                           - (35*LAL_PI)/48. + (5*delta*LAL_PI)/(16.*m1))

#       pPrec->alpha5    = (5*(-190512*delta3*eta6 + 2268*delta2*eta3*m1*(eta2*(323 + 784*eta)
#                           + 336*(25*chiL2 + chip2)*m1_4) + 7*m1_3*(8024297*eta4 + 857412*eta5
#                           + 3080448*eta6 + 143640*chip2*eta2*m1_4
#                           - 127008*chip2*(-4*chiL2 + chip2)*m1_8
#                           + 6048*eta3*((2632*chiL2 + 115*chip2)*m1_4 - 672*chiL*m1_2*LAL_PI))
#                           + 3*delta*m1_2*(-5579177*eta4 + 80136*eta5 - 3845520*eta6
#                           + 146664*chip2*eta2*m1_4 + 127008*chip2*(-4*chiL2 + chip2)*m1_8
#                           - 42336*eta3*((726*chiL2 + 29*chip2)*m1_4
#                           - 96*chiL*m1_2*LAL_PI))))/(6.5028096e7*eta4*m1_3)

#       /* Post-Newtonian Euler Angles: epsilon */
#       pPrec->epsilon1  = -35/192. + (5*delta)/(64.*m1)

#       pPrec->epsilon2  = ((15*chiL*delta*m1)/128. - (35*chiL*m1_2)/128.)/eta

#       pPrec->epsilon3  = -5515/3072. + eta*(-515/384. - (15*delta2)/(256.*m1_2)
#                           + (175*delta)/(256.*m1)) + (4555*delta)/(7168.*m1)

#       /* This term is proportional to log(w) */
#       pPrec->epsilon4L = (5*chiL*delta2)/16. - (5*chiL*delta*m1)/3. + (2545*chiL*m1_2)/1152.
#                           + ((-2035*chiL*delta*m1)/21504. + (2995*chiL*m1_2)/9216.)/eta - (35*LAL_PI)/48.
#                           + (5*delta*LAL_PI)/(16.*m1)

#       pPrec->epsilon5  = (5*(-190512*delta3*eta3 + 2268*delta2*m1*(eta2*(323 + 784*eta)
#                         + 8400*chiL2*m1_4)
#                         - 3*delta*m1_2*(eta*(5579177 + 504*eta*(-159 + 7630*eta))
#                         + 254016*chiL*m1_2*(121*chiL*m1_2 - 16*LAL_PI))
#                         + 7*m1_3*(eta*(8024297 + 36*eta*(23817 + 85568*eta))
#                         + 338688*chiL*m1_2*(47*chiL*m1_2 - 12*LAL_PI))))/(6.5028096e7*eta*m1_3)

#       break
#     }
#     case 220:
#     case 221:
#     case 222:
#     case 223:
#     case 224:
#     case 310:
#     case 311:
#     case 320:
#     case 321:
#     case 330:
#     {
#       pPrec->alpha1    = 0
#       pPrec->alpha2    = 0
#       pPrec->alpha3    = 0
#       pPrec->alpha4L   = 0
#       pPrec->alpha5    = 0
#       pPrec->epsilon1  = 0
#       pPrec->epsilon2  = 0
#       pPrec->epsilon3  = 0
#       pPrec->epsilon4L = 0
#       pPrec->epsilon5  = 0
#       break
#     }
#     default:
#     {
#       XLAL_ERROR(XLAL_EINVAL,"Error: IMRPhenomXPrecVersion not recognized. Requires version 101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321 or 330.\n")
#       break
#     }
#   }

#   REAL8 alpha_offset = 0, epsilon_offset = 0

#   #if DEBUG == 1
#       printf("thetaJN             : %e\n",   pPrec->thetaJN)
#       printf("phiJ_Sf             : %e\n", pPrec->phiJ_Sf)
#       printf("alpha0              : %e\n", pPrec->alpha0)
#       printf("pi-kappa            : %e\n", LAL_PI-pPrec->kappa)
#       printf("kappa               : %e\n", pPrec->kappa)
#       printf("pi/2 - phiRef       : %e\n", LAL_PI_2 - phiRef)
#       printf("zeta_polarization   : %.16e\n", pPrec->zeta_polarization)
#       printf("zeta_polarization   : %.16e\n", acos(pPrec->XdotPArun))
#       printf("zeta_polarization   : %.16e\n", asin(pPrec->XdotQArun))
#       printf("zeta_polarization   : %.16e\n\n", LAL_PI_2 - acos(pPrec->XdotQArun))
#       printf("alpha1              : %e\n",  pPrec->alpha1)
#       printf("alpha2              : %e\n",  pPrec->alpha2)
#       printf("alpha3              : %e\n",  pPrec->alpha3)
#       printf("alpha4L             : %e\n",  pPrec->alpha4L)
#       printf("alpha5              : %e\n\n",  pPrec->alpha5)
#   #endif


#   switch(convention)
#   {
#     case 0:
#       pPrec->epsilon0 = 0
#       break
#     case 1:
#     case 6:
#       pPrec->epsilon0 = pPrec->phiJ_Sf - LAL_PI
#       break
#     case 5:
#     case 7:
#       pPrec->epsilon0 = 0
#       break
#   }

#   if(convention == 5 || convention == 7)
#   {
#     pPrec->alpha_offset = -pPrec->alpha0
#     pPrec->epsilon_offset = 0
#     pPrec->alpha_offset_1 = -pPrec->alpha0
#     pPrec->epsilon_offset_1 = 0
#     pPrec->alpha_offset_3 = -pPrec->alpha0
#     pPrec->epsilon_offset_3 = 0
#     pPrec->alpha_offset_4 = -pPrec->alpha0
#     pPrec->epsilon_offset_4 = 0
#   }
#   else
#   {
#     /* Get initial Get \alpha and \epsilon offsets at \omega = pi * M * f_{Ref} */
#     Get_alphaepsilon_atfref(&alpha_offset, &epsilon_offset, 2, pPrec, pWF)
#     pPrec->alpha_offset       = alpha_offset
#     pPrec->epsilon_offset     = epsilon_offset
#     pPrec->alpha_offset_1     = alpha_offset
#     pPrec->epsilon_offset_1   = epsilon_offset
#     pPrec->alpha_offset_3     = alpha_offset
#     pPrec->epsilon_offset_3   = epsilon_offset
#     pPrec->alpha_offset_4     = alpha_offset
#     pPrec->epsilon_offset_4   = epsilon_offset
#   }

#   pPrec->cexp_i_alpha   = 0.
#   pPrec->cexp_i_epsilon = 0.
#   pPrec->cexp_i_betah   = 0.

#   /*
#       Check whether maximum opening angle becomes larger than \pi/2 or \pi/4.

#       If (L + S_L) < 0, then Wigner-d Coefficients will not track the angle between J and L, meaning
#       that the model may become pathological as one moves away from the aligned-spin limit.

#       If this does not happen, then max_beta will be the actual maximum opening angle.

#       This function uses a 2PN non-spinning approximation to the orbital angular momentum L, as
#       the roots can be analytically derived.

#       Returns XLAL_PRINT_WARNING if model is in a pathological regime.
#   */


#   // When L + SL < 0 and q>7, we disable multibanding
#   IMRPhenomXPCheckMaxOpeningAngle(pWF,pPrec,lalParams)

#   /* Activate multibanding for Euler angles it threshold !=0. Only for PhenomXPHM. */
#   if(XLALSimInspiralWaveformParamsLookupPhenomXPHMThresholdMband(lalParams)==0.)
#   {
#     /* User switched off multibanding */
#     pPrec->MBandPrecVersion = 0
#   }
#   else
#   {
#     /* User requested multibanding */
#     pPrec->MBandPrecVersion = 1

#     /* Switch off multiband for very high mass as in IMRPhenomXHM. */
#     if(pWF->Mtot > 500)
#     {
#       XLAL_PRINT_WARNING("Very high mass, only merger in frequency band, multibanding not efficient, switching off for non-precessing modes and Euler angles.")
#       pPrec->MBandPrecVersion = 0
#       XLALSimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lalParams, 0.)
#     }
#     if(pPrec->IMRPhenomXPrecVersion == 330 && pWF->q > 7){
#       /* this is here as a safety catch in case */
#       XLAL_PRINT_WARNING("Multibanding may lead to pathological behaviour in this case. Disabling multibanding .\n")
#       XLALSimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lalParams, 0.)
#       pPrec->MBandPrecVersion = 0
#     }

#     else if(pPrec->IMRPhenomXPrecVersion < 200)
#     {
#       /* The NNLO angles can have a worse, even pathological, behaviour for high mass ratio and double spin cases.
#        The waveform will look noisy, we switch off the multibanding for mass ratio above 8 to avoid worsen even more the waveform. */
#       if(pWF->q > 8)
#       {
#         XLAL_PRINT_WARNING("Very high mass ratio, NNLO angles may become pathological, switching off multibanding for angles.\n")
#         XLALSimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lalParams, 0.)
#         pPrec->MBandPrecVersion = 0
#       }
#     }
#     /* The MSA angles give quite 'noisy' waveforms in this corner of parameter space so we switch off multibanding to avoid worsen the waveform. */
#     else if ( pWF->q > 50 && pWF->Mtot > 100 )
#     {
#       XLALSimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lalParams, 0.)
#       pPrec->MBandPrecVersion = 0
#     }

#   }

#   /* At high mass ratios, we find there can be numerical instabilities in the model, although the waveforms continue to be well behaved.
#    * We warn to user of the possibility of these instabilities.
#    */
#   //printf(pWF->q)
#   if( pWF->q > 80 )
#     {
#       XLAL_PRINT_WARNING("Very high mass ratio, possibility of numerical instabilities. Waveforms remain well behaved.\n")
#     }


#   const REAL8 ytheta  = pPrec->thetaJN
#   const REAL8 yphi    = 0.0
#   pPrec->Y2m2         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -2)
#   pPrec->Y2m1         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -1)
#   pPrec->Y20          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  0)
#   pPrec->Y21          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  1)
#   pPrec->Y22          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  2)
#   pPrec->Y3m3         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3, -3)
#   pPrec->Y3m2         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3, -2)
#   pPrec->Y3m1         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3, -1)
#   pPrec->Y30          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3,  0)
#   pPrec->Y31          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3,  1)
#   pPrec->Y32          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3,  2)
#   pPrec->Y33          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3,  3)
#   pPrec->Y4m4         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4, -4)
#   pPrec->Y4m3         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4, -3)
#   pPrec->Y4m2         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4, -2)
#   pPrec->Y4m1         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4, -1)
#   pPrec->Y40          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  0)
#   pPrec->Y41          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  1)
#   pPrec->Y42          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  2)
#   pPrec->Y43          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  3)
#   pPrec->Y44          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  4)

#   pPrec->LALparams = lalParams

#   return XLAL_SUCCESS
# }


def imr_phenom_x_interpolate_alpha_beta_spin_taylor(  # pylint: disable=unused-argument,unused-variable
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    lal_params: IMRPhenomXPHMParameterDataClass,
):
    """
    IMRPhenomX Interpolate alpha and beta using Spin Taylor method.

    This function replicates the C code behavior of looping through PN arrays,
    rotating the orbital angular momentum direction to the J-frame, and computing
    Euler angles and gravitational wave frequencies. The C code includes an early
    break when fgw_Hz <= 0.

    IMPORTANT JAX LIMITATION:
    JAX cannot truly skip computation in a loop due to JIT compilation. All
    iterations execute their computations regardless. What we CAN do is:
    1. Gate all state updates with should_continue flag
    2. When fgw_Hz <= 0, set should_continue = False
    3. All future iterations become no-ops (computed but discarded)

    The "break" is emulated by stopping state updates, not by skipping iterations.
    Mfmax_PN and i_max track the last valid point before the break condition.

    Args:
        p_wf: Waveform data class containing waveform parameters.
        p_prec: Precession data class containing precession parameters.
        lal_params: Parameter data class containing LAL parameters.

    Returns:
        Tuple of (f_gw, alpha_aux, cos_beta, Mfmax_PN, i_max) where:
        - f_gw: Array of GW frequencies in Mf units (padded with zeros after break point)
        - alpha_aux: Array of Euler angle alpha values (padded with zeros after break point)
        - cos_beta: Array of cos(beta) values (padded with zeros after break point)
        - Mfmax_PN: Maximum valid GW frequency (last value before fgw_Hz <= 0)
        - i_max: Index of last valid point (where break occurs)
    """
    len_pn = p_prec.pn_arrays.v_pn.shape[0]

    # Initialize arrays
    f_gw = jnp.zeros(len_pn)
    alpha_aux = jnp.zeros(len_pn)
    cos_beta = jnp.zeros(len_pn)

    # Initial state for scan
    init_state = {
        "f_gw": f_gw,
        "alpha_aux": alpha_aux,
        "cos_beta": cos_beta,
        "m_fmax_pn": 0.0,
        "i_max": 0,
        "should_continue": True,
        "counter": 0,
    }

    def scan_body(state, i):  # pylint: disable=too-many-locals
        """Process one iteration of the loop, with early termination support.

        In JAX, we cannot truly break early, but we can stop updating state after
        a condition is met. This function:
        1. Computes all values (unavoidable in JAX)
        2. Only updates state arrays if should_continue is True AND frequency is valid
        3. Sets should_continue to False when frequency becomes invalid (the "break point")

        Returns:
            Tuple of (new_state, None) - the None is the output (not used here)
        """
        # Check if we should continue processing (once False, stays False)
        should_process = state["should_continue"]

        # Extract current LN components
        ln_hat_x_temp = p_prec.pn_arrays.ln_hat_x_pn[i]
        ln_hat_y_temp = p_prec.pn_arrays.ln_hat_y_pn[i]
        ln_hat_z_temp = p_prec.pn_arrays.ln_hat_z_pn[i]

        # Apply rotations (computed regardless of should_process, but that's unavoidable in JAX)
        ln_hat_x_temp, ln_hat_y_temp, ln_hat_z_temp = imr_phenom_x_rotate_z(
            -p_prec.phi_j_sf, ln_hat_x_temp, ln_hat_y_temp, ln_hat_z_temp
        )
        ln_hat_x_temp, ln_hat_y_temp, ln_hat_z_temp = imr_phenom_x_rotate_y(
            -p_prec.theta_j_sf, ln_hat_x_temp, ln_hat_y_temp, ln_hat_z_temp
        )
        ln_hat_x_temp, ln_hat_y_temp, ln_hat_z_temp = imr_phenom_x_rotate_z(
            -p_prec.kappa, ln_hat_x_temp, ln_hat_y_temp, ln_hat_z_temp
        )

        # Compute frequencies
        f_gw_hz = jnp.pow(p_prec.pn_arrays.v_pn[i], 3.0) / p_prec.pi_gm
        f_gw_mf = xlal_sim_imr_phenom_x_utils_hz_to_mf(f_gw_hz, p_wf.m_tot)

        # Compute angles
        alpha_temp = jnp.arctan2(ln_hat_y_temp, ln_hat_x_temp)
        cos_beta_temp = ln_hat_z_temp

        # Check if frequency is valid (positive)
        freq_valid = f_gw_hz > 0.0

        # Update arrays ONLY if we should continue processing AND frequency is valid
        # This is the key: should_process gates all updates
        update_condition = should_process & freq_valid

        new_f_gw = jnp.where(update_condition, state["f_gw"].at[i].set(f_gw_mf), state["f_gw"])
        new_alpha_aux = jnp.where(update_condition, state["alpha_aux"].at[i].set(alpha_temp), state["alpha_aux"])
        new_cos_beta = jnp.where(update_condition, state["cos_beta"].at[i].set(cos_beta_temp), state["cos_beta"])

        # Update Mfmax_PN and i_max only if update condition is met
        new_m_fmax_pn = jnp.where(update_condition, f_gw_mf, state["m_fmax_pn"])
        new_i_max = jnp.where(update_condition, i, state["i_max"])

        # CRITICAL: Set should_continue to False if we're currently processing AND frequency became invalid
        # This stops all future updates (the "break" behavior)
        # If should_process is already False, keep it False (once broken, always broken)
        new_should_continue = should_process & freq_valid

        new_state = {
            "f_gw": new_f_gw,
            "alpha_aux": new_alpha_aux,
            "cos_beta": new_cos_beta,
            "m_fmax_pn": new_m_fmax_pn,
            "i_max": new_i_max,
            "should_continue": new_should_continue,
            "counter": state["counter"] + 1,
        }

        return new_state, None

    # Use scan for early termination capability
    final_state, _ = jax.lax.scan(scan_body, init_state, jnp.arange(len_pn))

    # Note: Since JAX requires all iterations to complete, we use scan to process all
    # indices but track should_continue to skip processing after the break condition.
    # The first invalid frequency marks where the original C code would break.

    m_fmax_pn = final_state["m_fmax_pn"]

    fmax_inspiral = jax.lax.cond(p_prec.imr_phenom_xpnr_use_tuned_angles, m_fmax_pn, m_fmax_pn - p_wf.delta_mf)

    fmax_inspiral = jax.lax.cond(
        fmax_inspiral > p_wf.f_ring - p_wf.f_damp, lambda _: 1.020 * p_wf.f_meco, lambda x: x, operand=None
    )

    p_prec = p_prec.replace(ftrans_mrd=0.98 * fmax_inspiral, fmax_inspiral=fmax_inspiral)

    # Interpolate alpha
    alpha_unwrapped = xlal_sim_imr_phenom_x_unwrap_array(final_state["alpha_aux"])

    #       REAL8Sequence *fgw =NULL ;
    #       /* Setup sequences for angles*/
    #       REAL8Sequence *alpha = NULL;
    #       REAL8Sequence *alphaaux = NULL;
    #       REAL8Sequence *cosbeta = NULL;

    #       fgw=XLALCreateREAL8Sequence(lenPN);
    #       alpha=XLALCreateREAL8Sequence(lenPN);
    #       alphaaux=XLALCreateREAL8Sequence(lenPN);
    #       cosbeta=XLALCreateREAL8Sequence(lenPN);

    #       REAL8 fgw_Mf, fgw_Hz, Mfmax_PN=0.;
    #       // i_max is used to discard possibly unphysical points in the calculation of the final spin
    #       UINT8 i_max=0;
    #       REAL8 LNhatx_temp,LNhaty_temp,LNhatz_temp;

    #       for(UINT8 i=0; i < lenPN; i++){

    #           LNhatx_temp = (pPrec->PNarrays->LNhatx_PN->data->data[i]);
    #           LNhaty_temp = (pPrec->PNarrays->LNhaty_PN->data->data[i]);
    #           LNhatz_temp = (pPrec->PNarrays->LNhatz_PN->data->data[i]);

    #           IMRPhenomX_rotate_z(-pPrec->phiJ_Sf,  &LNhatx_temp, &LNhaty_temp, &LNhatz_temp);
    #           IMRPhenomX_rotate_y(-pPrec->thetaJ_Sf, &LNhatx_temp, &LNhaty_temp, &LNhatz_temp);
    #           IMRPhenomX_rotate_z(-pPrec->kappa,  &LNhatx_temp, &LNhaty_temp, &LNhatz_temp);

    #           fgw_Hz= pow(pPrec->PNarrays->V_PN->data->data[i],3.)/pPrec->piGM;
    #           fgw_Mf= XLALSimIMRPhenomXUtilsHztoMf(fgw_Hz,pWF->Mtot);

    #           if(fgw_Hz>0.){

    #           /* Compute Euler angles in the J frame */
    #           alphaaux->data[i] = atan2(LNhaty_temp, LNhatx_temp);
    #           cosbeta->data[i] = LNhatz_temp;
    #           fgw->data[i] = fgw_Mf;
    #           Mfmax_PN = fgw_Mf;
    #           i_max = i;
    #           }

    #           else
    #               break;

    #       }

    #     REAL8 fmax_inspiral;
    #     if(pPrec->IMRPhenomXPNRUseTunedAngles)
    #     fmax_inspiral = Mfmax_PN;
    #     else
    #     fmax_inspiral = Mfmax_PN-pWF->deltaMF;

    #     if(fmax_inspiral > pWF->fRING-pWF->fDAMP) fmax_inspiral = 1.020 * pWF->fMECO;

    #     pPrec->ftrans_MRD = 0.98*fmax_inspiral;
    #     pPrec->fmax_inspiral= fmax_inspiral;

    #     // Interpolate alpha
    #     XLALSimIMRPhenomXUnwrapArray(alphaaux->data, alpha->data, lenPN);

    #     pPrec->alpha_acc = gsl_interp_accel_alloc();
    #     pPrec->alpha_spline = gsl_spline_alloc(gsl_interp_cspline, lenPN);

    #     status = gsl_spline_init(pPrec->alpha_spline, fgw->data, alpha->data, lenPN);

    #     if (status != GSL_SUCCESS)
    #     {
    #          XLALPrintError("Error in %s: error in computing gsl spline for alpha.\n",__func__);
    #     }

    #     // Interpolate cosbeta
    #     pPrec->cosbeta_acc = gsl_interp_accel_alloc();
    #     pPrec->cosbeta_spline = gsl_spline_alloc(gsl_interp_cspline, lenPN);
    #     status =gsl_spline_init(pPrec->cosbeta_spline, fgw->data, cosbeta->data, lenPN);

    #     if (status != GSL_SUCCESS)
    #     {
    #          XLALPrintError("Error in %s: error in computing gsl spline for cos(beta).\n",__func__);
    #     }

    #     REAL8 cosbetamax;

    #     status = gsl_spline_eval_e(pPrec->cosbeta_spline, fmax_inspiral, pPrec->cosbeta_acc,&cosbetamax);
    #     if(status != GSL_SUCCESS)
    #     {
    #         XLALPrintError("Error in %s: error in computing cosbeta.\n",__func__);
    #     }

    #     // estimate final spin using spins at the end of the PN integration

    #     if(XLALSimInspiralWaveformParamsLookupPhenomXPFinalSpinMod(LALparams)==4){

    #     REAL8 m1 = pWF->m1_SI / pWF->Mtot_SI;
    #     REAL8 m2 = pWF->m2_SI / pWF->Mtot_SI;

    #     vector Lnf  = {pPrec->PNarrays->LNhatx_PN->data->data[i_max],pPrec->PNarrays->LNhaty_PN->data->data[i_max],pPrec->PNarrays->LNhatz_PN->data->data[i_max]};
    #     REAL8 Lnorm = sqrt(IMRPhenomX_vector_dot_product(Lnf,Lnf));
    #     vector S1f  = {pPrec->PNarrays->S1x_PN->data->data[i_max],pPrec->PNarrays->S1y_PN->data->data[i_max],pPrec->PNarrays->S1z_PN->data->data[i_max]};
    #     vector S2f  = {pPrec->PNarrays->S2x_PN->data->data[i_max],pPrec->PNarrays->S2y_PN->data->data[i_max],pPrec->PNarrays->S2z_PN->data->data[i_max]};

    #     REAL8 dotS1L = IMRPhenomX_vector_dot_product(S1f,Lnf)/Lnorm;
    #     REAL8 dotS2L  = IMRPhenomX_vector_dot_product(S2f,Lnf)/Lnorm;
    #     vector S1_perp = IMRPhenomX_vector_diff(S1f,IMRPhenomX_vector_scalar(Lnf, dotS1L));
    #     S1_perp = IMRPhenomX_vector_scalar(S1_perp,m1*m1);
    #     vector S2_perp = IMRPhenomX_vector_diff(S2f,IMRPhenomX_vector_scalar(Lnf, dotS2L));
    #     S2_perp = IMRPhenomX_vector_scalar(S2_perp,m2*m2);
    #     vector Stot_perp = IMRPhenomX_vector_sum(S1_perp,S2_perp);
    #     REAL8 S_perp_norm = sqrt(IMRPhenomX_vector_dot_product(Stot_perp,Stot_perp));
    #     REAL8 chi_perp_norm = S_perp_norm *pow(m1 + m2,2)/pow(m1,2);

    #     pWF->afinal= copysign(1.0, cosbetamax)* XLALSimIMRPhenomXPrecessingFinalSpin2017(pWF->eta,dotS1L,dotS2L,chi_perp_norm);

    #     pWF->fRING     = evaluate_QNMfit_fring22(pWF->afinal) / (pWF->Mfinal);
    #     pWF->fDAMP     = evaluate_QNMfit_fdamp22(pWF->afinal) / (pWF->Mfinal);
    #     }

    #     // initialize parameters for RD continuation
    #     pPrec->alpha_params    = XLALMalloc(sizeof(PhenomXPalphaMRD));
    #     pPrec->beta_params    = XLALMalloc(sizeof(PhenomXPbetaMRD));

    #     if(pPrec->IMRPhenomXPrecVersion==320 || pPrec->IMRPhenomXPrecVersion==321 || pPrec->IMRPhenomXPrecVersion==330 ){

    #     status = alphaMRD_coeff(*pPrec->alpha_spline, *pPrec->alpha_acc, pPrec->fmax_inspiral, pWF, pPrec->alpha_params);
    #     if(status!=XLAL_SUCCESS) XLALPrintError("XLAL Error in %s: error in computing parameters for MRD continuation of Euler angles.\n",__func__);

    #     status = betaMRD_coeff(*pPrec->cosbeta_spline, *pPrec->cosbeta_acc, pPrec->fmax_inspiral, pWF, pPrec);
    #      if(status!=XLAL_SUCCESS) XLALPrintError("XLAL Error in %s: error in computing parameters for MRD continuation of Euler angles.\n",__func__);

    #     }

    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->V_PN);
    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->S1x_PN);
    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->S1y_PN);
    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->S1z_PN);
    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->S2x_PN);
    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->S2y_PN);
    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->S2z_PN);
    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->LNhatx_PN);
    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->LNhaty_PN);
    #     XLALDestroyREAL8TimeSeries(pPrec->PNarrays->LNhatz_PN);
    #     XLALFree(pPrec->PNarrays);

    #     XLALDestroyREAL8Sequence(fgw);
    #     XLALDestroyREAL8Sequence(alphaaux);
    #     XLALDestroyREAL8Sequence(cosbeta);
    #     XLALDestroyREAL8Sequence(alpha);

    #     if(status != GSL_SUCCESS){

    #     gsl_spline_free(pPrec->alpha_spline);
    #     gsl_spline_free(pPrec->cosbeta_spline);
    #     gsl_interp_accel_free(pPrec->alpha_acc);
    #     gsl_interp_accel_free(pPrec->cosbeta_acc);

    #     }

    #     return status;

    #   }


def imr_phenom_x_spin_taylor_angles_splines_all(  # pylint: disable=unused-argument,unused-variable
    f_min: float,
    f_max: float,
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    lal_params: IMRPhenomXPHMParameterDataClass,
):
    """Compute spin Taylor Euler angles splines for IMRPhenomXPHM waveform model.

    Args:
        f_min: Minimum frequency for angle computation.
        f_max: Maximum frequency for angle computation.
        p_wf: Waveform data class containing waveform parameters.
        p_prec: Precession data class to be initialized.
        lal_params: Parameter data class containing LAL parameters.
    """
    f_ref = p_wf.f_ref

    # Sanity checks
    checkify.check(f_min > 0, "f_min must be positive.")
    checkify.check(f_max > 0, "f_max must be positive.")
    checkify.check(f_max > f_min, "f_max must be greater than f_min.")
    checkify.check(f_ref >= f_min, "f_ref must be >= f_min.")

    # Evaluate the splines for alpha and cosbeta.


def imr_phenom_x_initialize_euler_angles(  # pylint: disable=unused-argument,unused-variable
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    lal_params: IMRPhenomXPHMParameterDataClass,
):
    """Initialize Euler angles for IMRPhenomXPHM waveform model.

    Args:
        p_wf: Waveform data class containing waveform parameters.
        p_prec: Precession data class to be initialized.
        lal_params: Parameter data class containing LAL parameters.
    """
    threshold_pmb = lal_params.threshold_mband

    buffer = p_prec.integration_buffer

    # start below fMin to avoid interpolation artefacts
    f_min_angles = (p_wf.f_min - buffer) * 2 / p_prec.M_MAX

    # check we still pass a meaningful fmin
    checkify.check(
        f_min_angles > 0.0,
        "Error - imr_phenom_x_initialize_euler_angles: fMin is too low and numerical angles could not be computed.",
    )

    # If MB is on, we take advantage of the fact that we can compute angles on an array

    m_fmax_angles = jax.lax.cond(
        threshold_pmb > 0.0,
        lambda: p_wf.f_ring + 4.0 * p_wf.f_damp,
        lambda: (
            jnp.maximum(p_wf.mf_max, p_wf.f_ring + 4.0 * p_wf.f_damp)
            + xlal_sim_imr_phenom_x_utils_hz_to_mf(buffer, p_wf.m_tot)
        )
        * 2
        / p_prec.M_MIN,
    )

    # If MB is on, we take advantage of the fact that we can compute angles on an array

    #   if(thresholdPMB>0.)
    #     pPrec->Mfmax_angles = pWF->fRING+4.*pWF->fDAMP;
    #   else
    #     pPrec->Mfmax_angles = (MAX(pWF->MfMax,pWF->fRING+4.*pWF->fDAMP)+XLALSimIMRPhenomXUtilsHztoMf(buffer,pWF->Mtot))*2./pPrec->M_MIN;

    m_fmax_angles = jax.lax.cond(
        threshold_pmb > 0.0,
        lambda: p_wf.f_ring + 4.0 * p_wf.f_damp,
        lambda: (
            jnp.maximum(p_wf.mf_max, p_wf.f_ring + 4.0 * p_wf.f_damp)
            + xlal_sim_imr_phenom_x_utils_hz_to_mf(buffer, p_wf.m_tot)
        )
        * 2
        / p_prec.M_MIN,
    )

    p_prec = p_prec.replace(Mfmax_angles=m_fmax_angles)

    fmax_angles = xlal_sim_imr_phenom_x_utils_mf_to_hz(p_prec.Mfmax_angles, p_wf.m_tot)

    #   REAL8 fmaxAngles = XLALSimIMRPhenomXUtilsMftoHz(pPrec->Mfmax_angles,pWF->Mtot);

    #   // we add a few bins to fmax to make sure we do not run into interpolation errors
    #   status = IMRPhenomX_SpinTaylorAnglesSplinesAll(fminAngles,fmaxAngles,pWF,pPrec,lalParams);
    #   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "%s: IMRPhenomX_SpinTaylorAnglesSplinesAll failed.",__func__);

    #   status = gsl_spline_eval_e(pPrec->alpha_spline, pPrec->ftrans_MRD, pPrec->alpha_acc,&pPrec->alpha_ftrans);
    #   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "%s: could not compute alpha et the end of inspiral.",__func__);

    #   status = gsl_spline_eval_e(pPrec->cosbeta_spline, pPrec->ftrans_MRD, pPrec->cosbeta_acc,&pPrec->cosbeta_ftrans);
    #   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "%s: could not compute cosbeta et the end of inspiral.",__func__);

    #   status = gsl_spline_eval_e(pPrec->gamma_spline, pPrec->ftrans_MRD, pPrec->gamma_acc,&pPrec->gamma_ftrans);
    #   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "%s: could not compute gamma et the end of inspiral.",__func__);

    #   return status;


def imr_phenom_x_rotate_z(angle: float, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    """Rotate a vector around the z-axis by a given angle.

    Args:
        angle: Rotation angle in radians.
        vx: x-component of the vector.
        vy: y-component of the vector.
        vz: z-component of the vector.

    Returns:
        A tuple containing the rotated vector components (vx', vy', vz').
    """
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    vx_rotated = cos_angle * vx - sin_angle * vy
    vy_rotated = sin_angle * vx + cos_angle * vy
    vz_rotated = vz

    return vx_rotated, vy_rotated, vz_rotated


def imr_phenom_x_rotate_y(angle: float, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    """Rotate a vector around the y-axis by a given angle.

    Args:
        angle: Rotation angle in radians.
        vx: x-component of the vector.
        vy: y-component of the vector.
        vz: z-component of the vector.

    Returns:
        A tuple containing the rotated vector components (vx', vy', vz').
    """
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    vx_rotated = cos_angle * vx + sin_angle * vz
    vy_rotated = vy
    vz_rotated = -sin_angle * vx + cos_angle * vz

    return vx_rotated, vy_rotated, vz_rotated
