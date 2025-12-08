"""IMRPhenomX precession module for gravitational waveform generation."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.constants import PI, C, G
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    xlal_sim_imr_phenom_x_utils_hz_to_mf,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass


def get_delta_f_from_wfstruct(p_wf: IMRPhenomXWaveformDataClass) -> float:
    """Compute deltaF from waveform structure parameters.

    Args:
        p_wf: Waveform dataclass containing fRef, m1_SI, m2_SI, chi1L, chi2L, Mtot.
    """


#   REAL8 seglen=XLALSimInspiralChirpTimeBound(p_wf->fRef, p_wf->m1_SI, p_wf->m2_SI, p_wf->chi1L,p_wf->chi2L);
#   REAL8 deltaFv1= 1./MAX(4.,pow(2, ceil(log(seglen)/log(2))));
#   REAL8 deltaF = MIN(deltaFv1,0.1);
#   REAL8 deltaMF = XLALSimIMRPhenomXUtilsHztoMf(deltaF,p_wf->Mtot);
#   return(deltaMF);

# }


@checkify.checkify
def imr_phenom_x_set_precession_var(
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    m1_si: float,
    m2_si: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
    lal_params: IMRPhenomXPHMParameterDataClass,
    debug_flag: int,
) -> None:

    #   /*
    #       Here we assume m1 > m2, q > 1, dm = m1 - m2 = delta = sqrt(1-4eta) > 0
    #   */
    # p_wf.lal_params = lal_params

    # /* Pre-cache useful powers here */:
    sqrt2 = 1.4142135623730951
    sqrt5 = 2.23606797749978981
    sqrt6 = 2.44948974278317788
    sqrt7 = 2.64575131106459072
    sqrt10 = 3.16227766016838
    sqrt14 = 3.74165738677394133
    sqrt15 = 3.87298334620741702
    sqrt70 = 8.36660026534075563
    sqrt30 = 5.477225575051661
    sqrt2p5 = 1.58113883008419

    #   p_prec->debug_prec = debug_flag

    # /* Sort out version-specific flags */

    # // Get IMRPhenomX precession version from LAL dictionary
    imr_phenom_x_prec_version = lal_params.precession_version
    imr_phenom_x_prec_version = jax.lax.select(imr_phenom_x_prec_version == 300, 223, imr_phenom_x_prec_version)

    # default to NNLO angles if in-plane spins are negligible and one of the
    # SpinTaylor options has been selected. Solutions dominated by numerical noise.
    chi_in_plane = jnp.sqrt(chi1x * chi1x + chi1y * chi1y + chi2x * chi2x + chi2y * chi2y)

    # if(chi_in_plane<1e-6 && p_prec->IMRPhenomXPrecVersion==330)
    # {
    # p_prec->IMRPhenomXPrecVersion=102
    # }
    imr_phenom_x_prec_version = jax.lax.select(
        (chi_in_plane < 1e-6) & (imr_phenom_x_prec_version == 330), 102, imr_phenom_x_prec_version
    )

    # if(chi_in_plane<1e-7 && (p_prec->IMRPhenomXPrecVersion==320||
    #    p_prec->IMRPhenomXPrecVersion==321||p_prec->IMRPhenomXPrecVersion==310||
    #    p_prec->IMRPhenomXPrecVersion==311))
    # {
    # p_prec->IMRPhenomXPrecVersion=102
    # }
    imr_phenom_x_prec_version = jax.lax.select(
        (chi_in_plane < 1e-7)
        & (
            (imr_phenom_x_prec_version == 320)
            | (imr_phenom_x_prec_version == 321)
            | (imr_phenom_x_prec_version == 310)
            | (imr_phenom_x_prec_version == 311)
        ),
        102,
        imr_phenom_x_prec_version,
    )

    # // Get expansion order for MSA system of equations. Default is taken to be 5.
    expansion_order = lal_params.expansion_order

    # // Get toggle for PNR angles
    pnr_use_tuned_angles = lal_params.pnr_use_tuned_angles
    imr_phenom_xpnr_use_tuned_angles = pnr_use_tuned_angles

    # Get PNR angle interpolation tolerance
    imr_phenom_xpnr_interp_tolerance = lal_params.pnr_interp_tolerance
    # // Get toggle for symmetric waveform
    antisymmetric_waveform = lal_params.antisymmetric_waveform
    imr_phenom_x_antisymmetric_waveform = antisymmetric_waveform

    # Set toggle for polarization calculation: +1 for symmetric waveform
    # (default), -1 for antisymmetric waveform refer to XXXX.YYYYY for details
    polarization_symmetry = 1.0

    # /* allow for conditional disabling of precession multibanding given mass ratio and opening angle */
    conditional_prec_mband = 0
    mband_prec_version = lal_params.mband_version
    # if(MBandPrecVersion == 2){
    # MBandPrecVersion = 0 /* current default value is 0 */
    # conditionalPrecMBand = 1
    # }
    is_version_2 = mband_prec_version == 2
    conditional_prec_mband = jax.lax.select(is_version_2, 1, conditional_prec_mband)
    mband_prec_version = jax.lax.select(is_version_2, 0, mband_prec_version)

    # /* Define a number of convenient local parameters */
    m1 = m1_si / p_wf.m_tot_si  # Normalized mass of larger companion:   m1_SI / Mtot_SI
    m2 = m2_si / p_wf.m_tot_si  # Normalized mass of smaller companion:  m2_SI / Mtot_SI
    big_m = m1 + m2  # Total mass in solar units

    # // Useful powers of mass
    m1_2 = m1 * m1
    # m1_3 = m1 * m1_2
    # m1_4 = m1 * m1_3
    # m1_5 = m1 * m1_4
    # m1_6 = m1 * m1_5
    # m1_7 = m1 * m1_6
    # m1_8 = m1 * m1_7

    m2_2 = m2 * m2

    # I'm keeping this here, but note that these three lines have been moved to
    # the setting of IMRPhenomXPHMParameterDataClass
    #   p_wf->M = M
    #   p_wf->m1_2 = m1_2
    #   p_wf->m2_2 = m2_2

    # q = m1 / m2  # q = m1 / m2 > 1.0

    # // Powers of eta
    eta = p_wf.eta
    eta2 = eta * eta
    eta3 = eta * eta2
    eta4 = eta * eta3
    # eta5 = eta * eta4
    # eta6 = eta * eta5

    # // \delta in terms of q > 1
    # delta = p_wf.delta
    # delta2 = delta * delta
    # delta3 = delta * delta2

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

    # chi_eff = p_wf.chi_eff

    twopi_gm = 2 * PI * G * (m1_si + m2_si) / C**3
    pi_gm = PI * G * (m1_si + m2_si) / C**3

    # /* Set spin variables in p_prec struct */
    chi1x = chi1x
    chi1y = chi1y
    chi1_norm = jnp.sqrt(chi1x * chi1x + chi1y * chi1y + chi1z * chi1z)

    chi2x = chi2x
    chi2y = chi2y
    chi2z = chi2z
    chi2_norm = jnp.sqrt(chi2x * chi2x + chi2y * chi2y + chi2z * chi2z)

    # /* Check that spins obey Kerr bound */
    # if((!PNRUseTunedAngles)||(p_wf->PNR_SINGLE_SPIN != 1)){
    #   /*Allow the single-spin mapping for PNR to break the Kerr limit*/
    # XLAL_CHECK(fabs(p_prec->chi1_norm) <= 1.0, XLAL_EDOM,
    #     "Error in IMRPhenomXSetPrecessionVariables: |S1/m1^2| must be <= 1.\n")
    # XLAL_CHECK(fabs(p_prec->chi2_norm) <= 1.0, XLAL_EDOM,
    #     "Error in IMRPhenomXSetPrecessionVariables: |S2/m2^2| must be <= 1.\n")
    # }
    kerr_check_cond = (not pnr_use_tuned_angles) | (p_wf.pnr_single_spin != 1)
    checkify.check(
        kerr_check_cond & (jnp.abs(chi1_norm) <= 1.0),
        "Error in IMRPhenomXSetPrecessionVariables: |S1/m1^2| must be <= 1.\n",
    )
    checkify.check(
        kerr_check_cond & (jnp.abs(chi2_norm) <= 1.0),
        "Error in IMRPhenomXSetPrecessionVariables: |S2/m2^2| must be <= 1.\n",
    )

    # /* Calculate dimensionful spins */
    s1x = chi1x * m1_2
    s1y = chi1y * m1_2
    s1z = chi1z * m1_2
    s1_norm = jnp.abs(chi1_norm) * m1_2

    s2x = chi2x * m2_2
    s2y = chi2y * m2_2
    s2z = chi2z * m2_2
    s2_norm = jnp.abs(chi2_norm) * m2_2

    # // Useful powers
    s1_norm_2 = s1_norm * s1_norm
    s2_norm_2 = s2_norm * s2_norm

    chi1_perp = jnp.sqrt(chi1x * chi1x + chi1y * chi1y)
    chi2_perp = jnp.sqrt(chi2x * chi2x + chi2y * chi2y)

    # /* Get spin projections */
    s1_perp = (m1_2) * jnp.sqrt(chi1x * chi1x + chi1y * chi1y)
    s2_perp = (m2_2) * jnp.sqrt(chi2x * chi2x + chi2y * chi2y)

    # /* Norm of in-plane vector sum: Norm[ S1perp + S2perp ] */
    s_tot_perp = jnp.sqrt((s1x + s2x) * (s1x + s2x) + (s1y + s2y) * (s1y + s2y))

    # /* This is called chiTot_perp to distinguish from Sperp used in contrusction
    # of chi_p. For normalization, see Sec. IV D of arXiv:2004.06503 */
    chi_tot_perp = s_tot_perp * (big_m * big_m) / m1_2
    # Store chiTot_perp to p_wf so that it can be used in XCP modifications
    # (PNRUseTunedCoprec)
    p_wf = dataclasses.replace(p_wf, chi_tot_perp=chi_tot_perp)

    # /* disable tuned PNR angles, tuned coprec and mode asymmetries in low in-plane spin limit */
    # if((chi_in_plane < 1e-7)&&(p_prec->IMRPhenomXPNRUseTunedAngles == 1)&&(p_wf->PNR_SINGLE_SPIN != 1)){
    # XLALSimInspiralWaveformParamsInsertPhenomXPNRUseTunedAngles(lal_params, 0)
    # PNRUseTunedAngles = 0
    # p_prec->IMRPhenomXPNRUseTunedAngles = 0
    # p_prec->IMRPhenomXAntisymmetricWaveform = 0
    # AntisymmetricWaveform = 0
    # XLALSimInspiralWaveformParamsInsertPhenomXAntisymmetricWaveform(lal_params, 0)
    # XLALSimInspiralWaveformParamsInsertPhenomXPNRUseTunedCoprec(lal_params, 0)
    # }
    low_spin_cond = (chi_in_plane < 1e-7) & (imr_phenom_xpnr_use_tuned_angles == 1) & (p_wf.pnr_single_spin != 1)
    lal_params = jax.lax.cond(
        low_spin_cond,
        lambda x: dataclasses.replace(x, pnr_use_tuned_angles=0, antisymmetric_waveform=0, pnr_use_tuned_coprec=0),
        lambda x: x,
        lal_params,
    )
    pnr_use_tuned_angles = jax.lax.select(low_spin_cond, 0, pnr_use_tuned_angles)
    imr_phenom_x_antisymmetric_waveform = jax.lax.select(low_spin_cond, 0, imr_phenom_x_antisymmetric_waveform)

    # /*
    # Calculate the effective precessing spin parameter (Schmidt et al, PRD 91, 024043, 2015):
    #     - m1 > m2, so body 1 is the larger black hole
    # */
    big_a1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    big_a2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    a_sp_1 = big_a1 * s1_perp
    a_sp_2 = big_a2 * s2_perp

    # /* S_p = max(A1 S1_perp, A2 S2_perp) */
    num = jax.lax.select(a_sp_2 > a_sp_1, a_sp_2, a_sp_1)
    den = jax.lax.select(m2 > m1, big_a2 * (m2_2), big_a1 * (m1_2))
    # /* chi_p = max(A1 * Sp1 , A2 * Sp2) / (A_i * m_i^2) where i is the index of the larger BH */
    chip = num / den
    chi1_l = chi1z
    chi2_l = chi2z

    chi_p = chip
    # // (PNRUseTunedCoprec)
    p_wf = dataclasses.replace(p_wf, chi_p=chi_p)
    phi0_aligned = p_wf.phi0

    # /* Effective (dimensionful) aligned spin */
    s_l = chi1_l * m1_2 + chi2_l * m2_2

    # /* Effective (dimensionful) in-plane spin */
    s_perp = chip * m1_2  # /* m1 > m2 */

    msa_error = 0

    # p_wf22AS = NULL

    # // get first digit of precessing version: this tags the method employed to compute the Euler angles
    # // 1: NNLO 2: MSA 3: SpinTaylor (numerical)
    # precversion_tag = (imr_phenom_x_prec_version - (imr_phenom_x_prec_version % 100)) / 100

    # Update variables computed until now in p_prec
    p_prec = dataclasses.replace(
        p_prec,
        sqrt2=sqrt2,
        sqrt5=sqrt5,
        sqrt6=sqrt6,
        sqrt7=sqrt7,
        sqrt10=sqrt10,
        sqrt14=sqrt14,
        sqrt15=sqrt15,
        sqrt70=sqrt70,
        sqrt30=sqrt30,
        sqrt2p5=sqrt2p5,
        imr_phenom_x_prec_version=imr_phenom_x_prec_version,
        expansion_order=expansion_order,
        imr_phenom_x_pnr_use_tuned_angles=imr_phenom_xpnr_use_tuned_angles,
        imr_phenom_x_pnr_interp_tolerance=imr_phenom_xpnr_interp_tolerance,
        imr_phenom_x_antisymmetric_waveform=imr_phenom_x_antisymmetric_waveform,
        polarization_symmetry=polarization_symmetry,
        conditional_prec_mband=conditional_prec_mband,
        m_band_prec_version=mband_prec_version,
        eta=eta,
        eta2=eta2,
        eta3=eta3,
        eta4=eta4,
        inveta=inveta,
        inveta2=inveta2,
        inveta3=inveta3,
        inveta4=inveta4,
        sqrt_inveta=sqrt_inveta,
        two_pi_gm=twopi_gm,
        pi_gm=pi_gm,
        chi1x=chi1x,
        chi1y=chi1y,
        chi1z=chi1z,
        chi1_norm=chi1_norm,
        chi2x=chi2x,
        chi2y=chi2y,
        chi2z=chi2z,
        chi2_norm=chi2_norm,
        s1x=s1x,
        s1y=s1y,
        s1z=s1z,
        s1_norm=s1_norm,
        s2x=s2x,
        s2y=s2y,
        s2z=s2z,
        s2_norm=s2_norm,
        s1_norm_2=s1_norm_2,
        s2_norm_2=s2_norm_2,
        chi1_perp=chi1_perp,
        chi2_perp=chi2_perp,
        s1_perp=s1_perp,
        s2_perp=s2_perp,
        s_tot_perp=s_tot_perp,
        chi_tot_perp=chi_tot_perp,
        big_a1=big_a1,
        big_a2=big_a2,
        a_sp_1=a_sp_1,
        a_sp_2=a_sp_2,
        chi_p=chi_p,
        phi0_aligned=phi0_aligned,
        s_l=s_l,
        s_perp=s_perp,
        msa_error=msa_error,
    )

    # /* start of SpinTaylor code */

    # ######## NOTE if precversionTag==3: ######## -> Spin-Taylor

    # /* update  precessing version to catch possible fallbacks of SpinTaylor angles */
    precversion_tag = (p_prec.imr_phenom_x_prec_version - (p_prec.imr_phenom_x_prec_version % 100)) / 100
    pflag = p_prec.imr_phenom_x_prec_version


#   if(pflag != 101 && pflag != 102 && pflag != 103 && pflag != 104 &&
#      pflag != 220 && pflag != 221 && pflag != 222 && pflag != 223 &&
#      pflag != 224 && pflag!=310 && pflag!=311 && pflag!=320 &&
#      pflag!=321 && pflag!=330)
#   {
#     XLAL_ERROR(XLAL_EINVAL,
#         "Error in IMRPhenomXGetAndSetPrecessionVariables: Invalid precession"
#         " flag. Allowed versions are 101, 102, 103, 104, 220, 221, 222, 223,"
#         " 224, 310, 311, 320, 321 or 330.\n")
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
#     case 222: // MSA using expressions as implemented in
#               // LALSimInspiralFDPrecAngles. Terminal failure if MSA fails.
#     case 223: // MSA using expressions as implemented in
#               // LALSimInspiralFDPrecAngles. Defaults to NNLO v102 if MSA fails.
#     case 224: // MSA using expressions as detailed in arXiv:1703.03967, with
#               // \zeta_0 and \phi_{z,0} as in LALSimInspiralFDPrecAngles.
#               // Defaults to NNLO v102 if MSA fails.
#     {
#        /*
#           Double-spin model using angles from Chatziioannou et al,
#           PRD, 95, 104004, (2017), arXiv:1703.03967
#           Uses 3PN L
#        */
#        #if DEBUG == 1
#         printf("Initializing MSA system...\n")
#        #endif

#        if(p_prec->ExpansionOrder < -1 || p_prec->ExpansionOrder > 5)
#        {
#          XLAL_ERROR(XLAL_EINVAL,
#              "Error in IMRPhenomXGetAndSetPrecessionVariables: Invalid"
#              " expansion order for MSA corrections. Default is 5, allowed"
#              " values are [-1,0,1,2,3,4,5].\n")
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
#             XLAL_ERROR(XLAL_EINVAL,
#                 "Error in IMRPhenomXGetAndSetPrecessionVariables:"
#                 " IMRPhenomXPrecessionVersion not recognized.\n")
#       break
#         }
#     }


#   p_prec->precessing_tag=precversionTag


#   /* Calculate parameter for two-spin to single-spin map used in PNR and XCP */
#   /* Initialize PNR variables */
#   p_prec->chi_singleSpin = 0.0
#   p_prec->costheta_singleSpin = 0.0
#   p_prec->costheta_final_singleSpin = 0.0
#   p_prec->chi_singleSpin_antisymmetric = 0.0
#   p_prec->theta_antisymmetric = 0.0
#   p_prec->PNR_HM_Mflow = 0.0
#   p_prec->PNR_HM_Mfhigh = 0.0

#   p_prec->PNR_q_window_lower = 0.0
#   p_prec->PNR_q_window_upper = 0.0
#   p_prec->PNR_chi_window_lower = 0.0
#   p_prec->PNR_chi_window_upper = 0.0
#   // p_prec->PNRInspiralScaling = 0

#   UINT4 status = IMRPhenomX_PNR_GetAndSetPNRVariables(p_wf, p_prec)
#   XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC,
#             "Error: IMRPhenomX_PNR_GetAndSetPNRVariables failed in "
#             "IMRPhenomXGetAndSetPrecessionVariables.\n")

#   p_prec->alphaPNR = 0.0
#   p_prec->betaPNR = 0.0
#   p_prec->gammaPNR = 0.0

#   /*...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/
#   /*   Get and/or store CoPrec params into p_wf and p_prec    */
#   /*...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

#   status = IMRPhenomX_PNR_GetAndSetCoPrecParams(p_wf,p_prec,lal_params)
#   XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC,
#   "Error: IMRPhenomX_PNR_GetAndSetCoPrecParams failed \
#   in IMRPhenomXGetAndSetPrecessionVariables.\n")

#   /*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/


#   //
#   if( pflag == 220 || pflag == 221 || pflag == 222 || pflag == 223 || pflag == 224 )
#     {
#       #if DEBUG == 1
#         printf("Evaluating MSA system.\n")
#         printf("Expansion Order : %d\n",p_prec->ExpansionOrder)
#       #endif

#       IMRPhenomX_Initialize_MSA_System(p_wf,p_prec,p_prec->ExpansionOrder)

#       if(p_prec->MSA_ERROR == 1)
#       {
#         // In version 220, 223 and 224 if the MSA system fails to initialize
#         // we default to the NNLO PN angles using the 3PN aligned-spin
#         // orbital angular momentum
#         if(pflag == 220 || pflag == 223 || pflag == 224)
#         {
#           XLAL_PRINT_WARNING(
#               "Warning: Initialization of MSA system failed. Defaulting to "
#               "NNLO angles using 3PN aligned-spin approximation.")
#           p_prec->IMRPhenomXPrecVersion = 102
#           pflag  = p_prec->IMRPhenomXPrecVersion
#         }
#         else // Otherwise, if the MSA system fails to initialize we trigger a terminal error
#         {
#           XLAL_ERROR(XLAL_EDOM,"Error: IMRPhenomX_Initialize_MSA_System failed to initialize. Terminating.\n")
#         }
#       }
#     }

#   #if DEBUG == 1
#     printf("In IMRPhenomXSetPrecessionVariables... \n\n")
#     printf("chi_p   : %e\n",p_prec->chi_p)
#     printf("phic    : %e\n",p_prec->phi0_aligned)
#     printf("SL      : %e\n",p_prec->SL)
#     printf("Sperp   : %e\n\n",p_prec->Sperp)
#   #endif

#   /*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
#   /      Compute and set final spin and RD frequency           /
#   /...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/
#   IMRPhenomX_SetPrecessingRemnantParams(p_wf,p_prec,lal_params)
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
#       p_prec->L0   = 1.0
#       p_prec->L1   = 0.0
#       p_prec->L2   = ((3.0/2.0) + (eta/6.0))
#       p_prec->L3   = 0.0
#       p_prec->L4   = (81.0 + (-57.0 + eta)*eta)/24.
#       p_prec->L5   = 0.0
#       p_prec->L6   = 0.0
#       p_prec->L7   = 0.0
#       p_prec->L8   = 0.0
#       p_prec->L8L  = 0.0
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
#       p_prec->L0   = 1.0
#       p_prec->L1   = 0.0
#       p_prec->L2   = 3.0/2. + eta/6.0
#       p_prec->L3   = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
#       p_prec->L4   = (81 + (-57 + eta)*eta)/24.
#       p_prec->L5   = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
#       p_prec->L6   = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*powers_of_lalpi.two))/1296.
#       p_prec->L7   = 0.0
#       p_prec->L8   = 0.0

#       // This is the log(x) term
#       p_prec->L8L  = 0.0
#       break

#     }
#     /* 3PN orbital angular momentum using non-conserved spin norms
#        as per LALSimInspiralFDPrecAngles.c  */
#     case 222:
#     case 223:
#     {
#       p_prec->L0   = 1.0
#       p_prec->L1   = 0.0
#       p_prec->L2   = 3.0/2. + eta/6.0
#       p_prec->L3   = (
#           -7*(chi1L + chi2L + chi1L*delta - chi2L*delta)
#           + 5*(chi1L + chi2L)*eta
#       )/6.
#       p_prec->L4   = (81 + (-57 + eta)*eta)/24.
#       p_prec->L5   = (-1650*(chi1L + chi2L + chi1L*delta - chi2L*delta) +
#                       1336*(chi1L + chi2L)*eta + 511*(chi1L - chi2L)*delta*eta +
#                       28*(chi1L + chi2L)*eta2)/600.
#       p_prec->L6   = (10935 + eta*(-62001 + 1674*eta + 7*eta2 +
#                       2214*powers_of_lalpi.two))/1296.
#       p_prec->L7   = 0.0
#       p_prec->L8   = 0.0

#       // This is the log(x) term
#       p_prec->L8L  = 0.0
#       break
#     }
#     /* 4PN orbital angular momentum */
#     case 103:
#     {
#       p_prec->L0   = 1.0
#       p_prec->L1   = 0.0
#       p_prec->L2   = 3.0/2. + eta/6.0
#       p_prec->L3   = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
#       p_prec->L4   = (81 + (-57 + eta)*eta)/24.
#       p_prec->L5   = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
#       p_prec->L6   = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*powers_of_lalpi.two))/1296.
#       p_prec->L7   = (chi2L*(-324 + eta*(1119 - 2*eta*(172 + eta)) + delta*(324 + eta*(-633 + 14*eta)))
#                           - chi1L*(324 + eta*(-1119 + 2*eta*(172 + eta)) + delta*(324 + eta*(-633 + 14*eta))))/32.
#       p_prec->L8   = 2835/128. - (eta*(-10677852 + 100*eta*(-640863 + eta*(774 + 11*eta))
#                       + 26542080*LAL_GAMMA + 675*(3873 + 3608*eta)*powers_of_lalpi.two))/622080. - (64*eta*log16)/3.

#       p_prec->L8L  = -(64.0/3.0) * eta
#       break
#     }
#     /*
#         4PN orbital angular momentum + leading order in spin at all PN
#         orders terms.
#           - Marsat, CQG, 32, 085008, (2015), arXiv:1411.4118
#           - Siemonsen et al, PRD, 97, 064010, (2018), arXiv:1606.08832
#     */
#     case 104:
#     {
#       p_prec->L0   = 1.0
#       p_prec->L1   = 0.0
#       p_prec->L2   = 3.0/2. + eta/6.0
#       p_prec->L3   = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
#       p_prec->L4   = (81 + (-57 + eta)*eta)/24.
#       p_prec->L5   = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
#       p_prec->L6   = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*powers_of_lalpi.two))/1296.
#       p_prec->L7   = (chi2L*(-324 + eta*(1119 - 2*eta*(172 + eta)) + delta*(324 + eta*(-633 + 14*eta)))
#                           - chi1L*(324 + eta*(-1119 + 2*eta*(172 + eta)) + delta*(324 + eta*(-633 + 14*eta))))/32.
#       p_prec->L8   = 2835/128. - (eta*(-10677852 + 100*eta*(-640863 + eta*(774 + 11*eta))
#                       + 26542080*LAL_GAMMA + 675*(3873 + 3608*eta)*powers_of_lalpi.two))/622080. - (64*eta*log16)/3.

#       // This is the log(x) term at 4PN, x^4/2 * log(x)
#       p_prec->L8L  = -(64.0/3.0) * eta

#       // Leading order in spin at all PN orders, note that the 1.5PN terms
#       // are already included. Here we have additional 2PN and 3.5PN corrections.
#       p_prec->L4  += (chi1L2*(1 + delta - 2*eta) + 4*chi1L*chi2L*eta -
#                       chi2L2*(-1 + delta + 2*eta))/2.
#                       chi2L2*(-1 + delta + 2*eta))/2.
#       p_prec->L7  +=  (3*(chi1L + chi2L)*eta*(chi1L2*(1 + delta - 2*eta) +
#                        4*chi1L*chi2L*eta - chi2L2*(-1 + delta + 2*eta)))/4.

#       break
#     }

#     default:
#     {
#       XLAL_ERROR(XLAL_EINVAL,
#                  "Error: IMRPhenomXPrecVersion not recognized. Requires "
#                  "version 101, 102, 103, 104, 220, 221, 222, 223, 224, "
#                  "310, 311, 320, 321 or 330.\n")
#       break
#     }
#   }

#   /* Reference orbital angular momentum */
#   p_prec->LRef = M * M * XLALSimIMRPhenomXLPNAnsatz(
#       p_wf->v_ref, p_wf->eta / p_wf->v_ref, p_prec->L0, p_prec->L1,
#       p_prec->L2, p_prec->L3, p_prec->L4, p_prec->L5, p_prec->L6,
#       p_prec->L7, p_prec->L8, p_prec->L8L)

#   /*
#     In the following code block we construct the conventions that relate the source frame and the LAL frame.

#     A detailed discussion of the conventions can be found in Appendix C and D of arXiv:2004.06503 and https://dcc.ligo.org/LIGO-T1500602
#   */

#   /* Get source frame (*_Sf) J = L + S1 + S2. This is an instantaneous frame in which L is aligned with z */
#   p_prec->J0x_Sf = (m1_2)*chi1x + (m2_2)*chi2x
#   p_prec->J0y_Sf = (m1_2)*chi1y + (m2_2)*chi2y
#   p_prec->J0z_Sf = (m1_2)*chi1z + (m2_2)*chi2z + p_prec->LRef

#   p_prec->J0     = sqrt(p_prec->J0x_Sf*p_prec->J0x_Sf + p_prec->J0y_Sf*p_prec->J0y_Sf + p_prec->J0z_Sf*p_prec->J0z_Sf)

#   /* Get angle between J0 and LN (z-direction) */
#   if(p_prec->J0 < 1e-10)
#   {
#     XLAL_PRINT_WARNING("Warning: |J0| < 1e-10. Setting thetaJ = 0.\n")
#     p_prec->thetaJ_Sf = 0.0
#   }
#   else
#   {
#     p_prec->thetaJ_Sf = acos(p_prec->J0z_Sf / p_prec->J0)
#   }

#   const double phiRef = p_wf->phiRef_In

#   INT4 convention     = XLALSimInspiralWaveformParamsLookupPhenomXPConvention(lal_params)

#   if ( !(convention == 0 || convention == 1 || convention == 5 || convention == 6 || convention == 7))
#   {
#     XLAL_ERROR(XLAL_EINVAL,"Error: IMRPhenomXPConvention not recognized. Requires version 0, 1, 5, 6 or 7.\n")
#   }

#   #if DEBUG == 1
#     printf("\n*** Convention = %i\n", convention)
#   #endif

#   /* Get azimuthal angle of J0 in the source frame */
#   if(fabs(p_prec->J0x_Sf) < MAX_TOL_ATAN && fabs(p_prec->J0y_Sf) < MAX_TOL_ATAN)
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
#           p_prec->phiJ_Sf = LAL_PI/2.0 - phiRef
#           break
#         }
#         case 1:
#         case 6:
#         case 7:
#         {
#           p_prec->phiJ_Sf = 0
#           break
#         }

#       }
#   }
#   else
#   {
#       p_prec->phiJ_Sf = atan2(p_prec->J0y_Sf, p_prec->J0x_Sf) /* azimuthal angle of J0 in the source frame */
#   }
#   p_prec->phi0_aligned = - p_prec->phiJ_Sf

#   switch(convention)
#   {
#     case 0:
#     {
#       p_wf->phi0 = p_prec->phi0_aligned
#       break
#     }
#     case 1:
#     {
#       p_wf->phi0 = 0
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
#   p_prec->Nx_Sf = sin(p_wf->inclination)*cos((LAL_PI / 2.0) - phiRef)
#   p_prec->Ny_Sf = sin(p_wf->inclination)*sin((LAL_PI / 2.0) - phiRef)
#   p_prec->Nz_Sf = cos(p_wf->inclination)

#   REAL8 tmp_x = p_prec->Nx_Sf
#   REAL8 tmp_y = p_prec->Ny_Sf
#   REAL8 tmp_z = p_prec->Nz_Sf

#   IMRPhenomX_rotate_z(-p_prec->phiJ_Sf,   &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_y(-p_prec->thetaJ_Sf, &tmp_x, &tmp_y, &tmp_z)

#   /* Note difference in overall - sign w.r.t PhenomPv2 code */
#   p_prec->kappa = XLALSimIMRPhenomXatan2tol(tmp_y,tmp_x, MAX_TOL_ATAN)

#   /* Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1} */
#   tmp_x = 0.0
#   tmp_y = 0.0
#   tmp_z = 1.0
#   IMRPhenomX_rotate_z(-p_prec->phiJ_Sf,   &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_y(-p_prec->thetaJ_Sf, &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_z(-p_prec->kappa,     &tmp_x, &tmp_y, &tmp_z)

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
#           p_prec->alpha0 = LAL_PI
#           break
#         }
#         case 1:
#         case 6:
#         case 7:
#         {
#           p_prec->alpha0 = LAL_PI - p_prec->kappa
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
#           p_prec->alpha0 = atan2(tmp_y,tmp_x)
#           break
#         }
#         case 1:
#         case 6:
#         case 7:
#         {
#           p_prec->alpha0 = LAL_PI - p_prec->kappa
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
#         tmp_x = p_prec->Nx_Sf
#         tmp_y = p_prec->Ny_Sf
#         tmp_z = p_prec->Nz_Sf
#         IMRPhenomX_rotate_z(-p_prec->phiJ_Sf,   &tmp_x, &tmp_y, &tmp_z)
#         IMRPhenomX_rotate_y(-p_prec->thetaJ_Sf, &tmp_x, &tmp_y, &tmp_z)
#         IMRPhenomX_rotate_z(-p_prec->kappa,     &tmp_x, &tmp_y, &tmp_z)

#         /* We don't need the y-component but we will store it anyway */
#         p_prec->Nx_Jf = tmp_x
#         p_prec->Ny_Jf = tmp_y
#         p_prec->Nz_Jf = tmp_z

#         /* This is a unit vector, so no normalization */
#         p_prec->thetaJN = acos(p_prec->Nz_Jf)
#         break
#     }
#     case 1:
#     case 6:
#     case 7:
#     {
#         REAL8 J0dotN     = (p_prec->J0x_Sf * p_prec->Nx_Sf) +
#                             (p_prec->J0y_Sf * p_prec->Ny_Sf) +
#                             (p_prec->J0z_Sf * p_prec->Nz_Sf)
#         p_prec->thetaJN   = acos( J0dotN / p_prec->J0 )
#         p_prec->Nz_Jf     = cos(p_prec->thetaJN)
#         p_prec->Nx_Jf     = sin(p_prec->thetaJN)
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
#   p_prec->Xx_Sf = -cos(p_wf->inclination) * sin(phiRef)
#   p_prec->Xy_Sf = -cos(p_wf->inclination) * cos(phiRef)
#   p_prec->Xz_Sf = +sin(p_wf->inclination)

#   tmp_x = p_prec->Xx_Sf
#   tmp_y = p_prec->Xy_Sf
#   tmp_z = p_prec->Xz_Sf

#   IMRPhenomX_rotate_z(-p_prec->phiJ_Sf,   &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_y(-p_prec->thetaJ_Sf, &tmp_x, &tmp_y, &tmp_z)
#   IMRPhenomX_rotate_z(-p_prec->kappa,     &tmp_x, &tmp_y, &tmp_z)


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
#       p_prec->PArunx_Jf = +0.0
#       p_prec->PAruny_Jf = -1.0
#       p_prec->PArunz_Jf = +0.0

#       /* Q = (N x P) by construction */
#       p_prec->QArunx_Jf =  p_prec->Nz_Jf
#       p_prec->QAruny_Jf =  0.0
#       p_prec->QArunz_Jf = -p_prec->Nx_Jf
#       break
#     }
#     case 1:
#     case 6:
#     case 7:
#     {
#       /* Get polar angle of X vector in J frame in the P,Q basis of Arun et al */
#       p_prec->PArunx_Jf = p_prec->Nz_Jf
#       p_prec->PAruny_Jf = 0
#       p_prec->PArunz_Jf = -p_prec->Nx_Jf

#       /* Q = (N x P) by construction */
#       p_prec->QArunx_Jf =  0
#       p_prec->QAruny_Jf =  1
#       p_prec->QArunz_Jf =  0
#       break
#     }
#   }

#   // (X . P)
#   p_prec->XdotPArun = (tmp_x * p_prec->PArunx_Jf) + (tmp_y * p_prec->PAruny_Jf) + (tmp_z * p_prec->PArunz_Jf)

#   // (X . Q)
#   p_prec->XdotQArun = (tmp_x * p_prec->QArunx_Jf) + (tmp_y * p_prec->QAruny_Jf) + (tmp_z * p_prec->QArunz_Jf)

#   /* Now get the angle zeta */
#   p_prec->zeta_polarization = atan2(p_prec->XdotQArun, p_prec->XdotPArun)

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

#       p_prec->alpha1    = -35/192. + (5*delta)/(64.*m1)

#       p_prec->alpha2    = ((15*chiL*delta*m1)/128. - (35*chiL*m1_2)/128.)/eta

#       p_prec->alpha3    = -5515/3072. + eta*(-515/384. - (15*delta2)/(256.*m1_2)
#                           + (175*delta)/(256.*m1)) + (4555*delta)/(7168.*m1)
#                           + ((15*chip2*delta*m1_3)/128. - (35*chip2*m1_4)/128.)/eta2

#       /* This is the term proportional to log(w) */
#       p_prec->alpha4L   = ((5*chiL*delta2)/16. - (5*chiL*delta*m1)/3. + (2545*chiL*m1_2)/1152.
#                           + ((-2035*chiL*delta*m1)/21504.
#                           + (2995*chiL*m1_2)/9216.)/eta + ((5*chiL*chip2*delta*m1_5)/128.
#                           - (35*chiL*chip2*m1_6)/384.)/eta3
#                           - (35*LAL_PI)/48. + (5*delta*LAL_PI)/(16.*m1))

#       p_prec->alpha5    = (5*(-190512*delta3*eta6 + 2268*delta2*eta3*m1*(eta2*(323 + 784*eta)
#                           + 336*(25*chiL2 + chip2)*m1_4) + 7*m1_3*(8024297*eta4 + 857412*eta5
#                           + 3080448*eta6 + 143640*chip2*eta2*m1_4
#                           - 127008*chip2*(-4*chiL2 + chip2)*m1_8
#                           + 6048*eta3*((2632*chiL2 + 115*chip2)*m1_4 - 672*chiL*m1_2*LAL_PI))
#                           + 3*delta*m1_2*(-5579177*eta4 + 80136*eta5 - 3845520*eta6
#                           + 146664*chip2*eta2*m1_4 + 127008*chip2*(-4*chiL2 + chip2)*m1_8
#                           - 42336*eta3*((726*chiL2 + 29*chip2)*m1_4
#                           - 96*chiL*m1_2*LAL_PI))))/(6.5028096e7*eta4*m1_3)

#       /* Post-Newtonian Euler Angles: epsilon */
#       p_prec->epsilon1  = -35/192. + (5*delta)/(64.*m1)

#       p_prec->epsilon2  = ((15*chiL*delta*m1)/128. - (35*chiL*m1_2)/128.)/eta

#       p_prec->epsilon3  = -5515/3072. + eta*(-515/384. - (15*delta2)/(256.*m1_2)
#                           + (175*delta)/(256.*m1)) + (4555*delta)/(7168.*m1)

#       /* This term is proportional to log(w) */
#       p_prec->epsilon4L = (5*chiL*delta2)/16. - (5*chiL*delta*m1)/3. + (2545*chiL*m1_2)/1152.
#                           + ((-2035*chiL*delta*m1)/21504. + (2995*chiL*m1_2)/9216.)/eta - (35*LAL_PI)/48.
#                           + (5*delta*LAL_PI)/(16.*m1)

#       p_prec->epsilon5  = (5*(-190512*delta3*eta3 + 2268*delta2*m1*(eta2*(323 + 784*eta)
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
#       p_prec->alpha1    = 0
#       p_prec->alpha2    = 0
#       p_prec->alpha3    = 0
#       p_prec->alpha4L   = 0
#       p_prec->alpha5    = 0
#       p_prec->epsilon1  = 0
#       p_prec->epsilon2  = 0
#       p_prec->epsilon3  = 0
#       p_prec->epsilon4L = 0
#       p_prec->epsilon5  = 0
#       break
#     }
#     default:
#     {
#       XLAL_ERROR(XLAL_EINVAL,
#                  "Error: IMRPhenomXPrecVersion not recognized. Requires "
#                  "version 101, 102, 103, 104, 220, 221, 222, 223, 224, "
#                  "310, 311, 320, 321 or 330.\n")
#       break
#     }
#   }

#   REAL8 alpha_offset = 0, epsilon_offset = 0

#   #if DEBUG == 1
#       printf("thetaJN             : %e\n",   p_prec->thetaJN)
#       printf("phiJ_Sf             : %e\n", p_prec->phiJ_Sf)
#       printf("alpha0              : %e\n", p_prec->alpha0)
#       printf("pi-kappa            : %e\n", LAL_PI-p_prec->kappa)
#       printf("kappa               : %e\n", p_prec->kappa)
#       printf("pi/2 - phiRef       : %e\n", LAL_PI_2 - phiRef)
#       printf("zeta_polarization   : %.16e\n", p_prec->zeta_polarization)
#       printf("zeta_polarization   : %.16e\n", acos(p_prec->XdotPArun))
#       printf("zeta_polarization   : %.16e\n", asin(p_prec->XdotQArun))
#       printf("zeta_polarization   : %.16e\n\n", LAL_PI_2 - acos(p_prec->XdotQArun))
#       printf("alpha1              : %e\n",  p_prec->alpha1)
#       printf("alpha2              : %e\n",  p_prec->alpha2)
#       printf("alpha3              : %e\n",  p_prec->alpha3)
#       printf("alpha4L             : %e\n",  p_prec->alpha4L)
#       printf("alpha5              : %e\n\n",  p_prec->alpha5)
#   #endif


#   switch(convention)
#   {
#     case 0:
#       p_prec->epsilon0 = 0
#       break
#     case 1:
#     case 6:
#       p_prec->epsilon0 = p_prec->phiJ_Sf - LAL_PI
#       break
#     case 5:
#     case 7:
#       p_prec->epsilon0 = 0
#       break
#   }

#   if(convention == 5 || convention == 7)
#   {
#     p_prec->alpha_offset = -p_prec->alpha0
#     p_prec->epsilon_offset = 0
#     p_prec->alpha_offset_1 = -p_prec->alpha0
#     p_prec->epsilon_offset_1 = 0
#     p_prec->alpha_offset_3 = -p_prec->alpha0
#     p_prec->epsilon_offset_3 = 0
#     p_prec->alpha_offset_4 = -p_prec->alpha0
#     p_prec->epsilon_offset_4 = 0
#   }
#   else
#   {
#     /* Get initial Get \alpha and \epsilon offsets at \omega = pi * M * f_{Ref} */
#     Get_alphaepsilon_atfref(&alpha_offset, &epsilon_offset, 2, p_prec, p_wf)
#     p_prec->alpha_offset       = alpha_offset
#     p_prec->epsilon_offset     = epsilon_offset
#     p_prec->alpha_offset_1     = alpha_offset
#     p_prec->epsilon_offset_1   = epsilon_offset
#     p_prec->alpha_offset_3     = alpha_offset
#     p_prec->epsilon_offset_3   = epsilon_offset
#     p_prec->alpha_offset_4     = alpha_offset
#     p_prec->epsilon_offset_4   = epsilon_offset
#   }

#   p_prec->cexp_i_alpha   = 0.
#   p_prec->cexp_i_epsilon = 0.
#   p_prec->cexp_i_betah   = 0.

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
#   IMRPhenomXPCheckMaxOpeningAngle(p_wf,p_prec,lal_params)

#   /* Activate multibanding for Euler angles it threshold !=0. Only for PhenomXPHM. */
#   if(XLALSimInspiralWaveformParamsLookupPhenomXPHMThresholdMband(lal_params)==0.)
#   {
#     /* User switched off multibanding */
#     p_prec->MBandPrecVersion = 0
#   }
#   else
#   {
#     /* User requested multibanding */
#     p_prec->MBandPrecVersion = 1

#     /* Switch off multiband for very high mass as in IMRPhenomXHM. */
#     if(p_wf->Mtot > 500)
#     {
#       XLAL_PRINT_WARNING(
#           "Very high mass, only merger in frequency band, multibanding not "
#           "efficient, switching off for non-precessing modes and Euler angles.")
#       p_prec->MBandPrecVersion = 0
#       XLALSimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lal_params, 0.)
#     }
#     if(p_prec->IMRPhenomXPrecVersion == 330 && p_wf->q > 7){
#       /* this is here as a safety catch in case */
#       XLAL_PRINT_WARNING("Multibanding may lead to pathological behaviour in this case. Disabling multibanding .\n")
#       XLALSimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lal_params, 0.)
#       p_prec->MBandPrecVersion = 0
#     }

#     else if(p_prec->IMRPhenomXPrecVersion < 200)
#     {
#       /* The NNLO angles can have a worse, even pathological, behaviour for
#        high mass ratio and double spin cases. The waveform will look noisy,
#        we switch off the multibanding for mass ratio above 8 to avoid
#        worsen even more the waveform. */
#       if(p_wf->q > 8)
#       {
#         XLAL_PRINT_WARNING(
#             "Very high mass ratio, NNLO angles may become pathological, "
#             "switching off multibanding for angles.\n")
#         XLALSimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lal_params, 0.)
#         p_prec->MBandPrecVersion = 0
#       }
#     }
#     /* The MSA angles give quite 'noisy' waveforms in this corner of parameter
#      space so we switch off multibanding to avoid worsen the waveform. */
#     else if ( p_wf->q > 50 && p_wf->Mtot > 100 )
#     {
#       XLALSimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lal_params, 0.)
#       p_prec->MBandPrecVersion = 0
#     }

#   }

#   /* At high mass ratios, we find there can be numerical instabilities in
#    * the model, although the waveforms continue to be well behaved.
#    * We warn to user of the possibility of these instabilities.
#    */
#   //printf(p_wf->q)
#   if( p_wf->q > 80 )
#     {
#       XLAL_PRINT_WARNING(
#           "Very high mass ratio, possibility of numerical instabilities. "
#           "Waveforms remain well behaved.\n")
#     }


#   const REAL8 ytheta  = p_prec->thetaJN
#   const REAL8 yphi    = 0.0
#   p_prec->Y2m2         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -2)
#   p_prec->Y2m1         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -1)
#   p_prec->Y20          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  0)
#   p_prec->Y21          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  1)
#   p_prec->Y22          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  2)
#   p_prec->Y3m3         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3, -3)
#   p_prec->Y3m2         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3, -2)
#   p_prec->Y3m1         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3, -1)
#   p_prec->Y30          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3,  0)
#   p_prec->Y31          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3,  1)
#   p_prec->Y32          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3,  2)
#   p_prec->Y33          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 3,  3)
#   p_prec->Y4m4         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4, -4)
#   p_prec->Y4m3         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4, -3)
#   p_prec->Y4m2         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4, -2)
#   p_prec->Y4m1         = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4, -1)
#   p_prec->Y40          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  0)
#   p_prec->Y41          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  1)
#   p_prec->Y42          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  2)
#   p_prec->Y43          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  3)
#   p_prec->Y44          = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 4,  4)

#   p_prec->lal_params = lal_params

#   return XLAL_SUCCESS
# }


def imr_phenom_x_spin_taylor_angles_splines_all(
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
    f_min_angles = (p_wf.f_min - buffer) * 2 / p_prec.m_max

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
            jnp.maximum(p_wf.m_f_max, p_wf.f_ring + 4.0 * p_wf.f_damp)
            + xlal_sim_imr_phenom_x_utils_hz_to_mf(buffer, p_wf.m_tot)
        )
        * 2
        / p_prec.m_min,
    )

    # If MB is on, we take advantage of the fact that we can compute angles on an array

    #   if(thresholdPMB>0.)
    #     p_prec->Mfmax_angles = p_wf->fRING+4.*p_wf->fDAMP;
    #   else
    #     p_prec->Mfmax_angles = (MAX(p_wf->MfMax,p_wf->fRING+4.*p_wf->fDAMP)+XLALSimIMRPhenomXUtilsHztoMf(buffer,p_wf->Mtot))*2./p_prec->M_MIN;

    m_fmax_angles = jax.lax.cond(
        threshold_pmb > 0.0,
        lambda: p_wf.f_ring + 4.0 * p_wf.f_damp,
        lambda: (
            jnp.maximum(p_wf.m_f_max, p_wf.f_ring + 4.0 * p_wf.f_damp)
            + xlal_sim_imr_phenom_x_utils_hz_to_mf(buffer, p_wf.m_tot)
        )
        * 2
        / p_prec.m_min,
    )

    p_prec = dataclasses.replace(p_prec, m_fmax_angles=m_fmax_angles)

    # fmax_angles = xlal_sim_imr_phenom_x_utils_mf_to_hz(p_prec.m_fmax_angles, p_wf.m_tot)

    #   REAL8 fmaxAngles = XLALSimIMRPhenomXUtilsMftoHz(p_prec->Mfmax_angles,p_wf->Mtot);

    #   // we add a few bins to fmax to make sure we do not run into interpolation errors
    #   status = IMRPhenomX_SpinTaylorAnglesSplinesAll(fminAngles,fmaxAngles,p_wf,p_prec,lal_params);
    #   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "%s: IMRPhenomX_SpinTaylorAnglesSplinesAll failed.",__func__);


#   status = gsl_spline_eval_e(p_prec->alpha_spline, p_prec->ftrans_MRD,
#                               p_prec->alpha_acc,&p_prec->alpha_ftrans);
#   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC,
#              "%s: could not compute alpha et the end of inspiral.",__func__);

#   status = gsl_spline_eval_e(p_prec->cosbeta_spline, p_prec->ftrans_MRD,
#                               p_prec->cosbeta_acc,&p_prec->cosbeta_ftrans);
#   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC,
#              "%s: could not compute cosbeta et the end of inspiral.",__func__);    #   status = gsl_spline_eval_e(p_prec->gamma_spline, p_prec->ftrans_MRD, p_prec->gamma_acc,&p_prec->gamma_ftrans);
#   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "%s: could not compute gamma et the end of inspiral.",__func__);

#   return status;
