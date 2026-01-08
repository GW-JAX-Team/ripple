"""IMRPhenomX precession module for gravitational waveform generation."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.constants import PI, C, EulerGamma, G
from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals import (
    imr_phenom_x_initialize_powers,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_pnr_beta import (
    imr_phenom_x_pnr_generate_ringdown_pnr_beta,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_pnr_internals import (
    imr_phenom_x_pnr_get_and_set_co_prec_params,
    imr_phenom_x_pnr_get_and_set_pnr_variables,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_angle_cases import (
    imr_phenom_x_initialize_msa_system,
    imr_phenom_x_return_msa_corrections_msa,
    imr_phenom_x_return_phiz_msa,
    imr_phenom_x_return_roots_msa,
    imr_phenom_x_return_s_norm_msa,
    imr_phenom_x_return_zeta_msa,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_qnm import (
    evaluate_QNMfit_fdamp22,
    evaluate_QNMfit_fring22,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    xlal_sim_imr_phenom_x_atan2tol,
    xlal_sim_imr_phenom_x_final_spin_2017,
    xlal_sim_imr_phenom_x_precessing_final_spin_2017,
    xlal_sim_imr_phenom_x_utils_hz_to_mf,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_xhm_qnm import (
    evaluate_QNMfit_fring21,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass
from ripplegw.waveforms.spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)

############## Useful pre-cached constants ##############
_, powers_of_lalpi = imr_phenom_x_initialize_powers(PI)
MAX_TOL_ATAN = 1.0e-15


# def get_delta_f_from_wfstruct(p_wf: IMRPhenomXWaveformDataClass) -> float:
#     """Compute deltaF from waveform structure parameters.

#     Args:
#         p_wf: Waveform dataclass containing fRef, m1_SI, m2_SI, chi1L, chi2L, Mtot.
#     """


#   seglen=XLALSimInspiralChirpTimeBound(p_wf->fRef, p_wf->m1_SI, p_wf->m2_SI, p_wf->chi1L,p_wf->chi2L)
#   deltaFv1= 1./MAX(4.,pow(2, ceil(log(seglen)/log(2))))
#   deltaF = MIN(deltaFv1,0.1)
#   deltaMF = XLALSimIMRPhenomXUtilsHztoMf(deltaF,p_wf->Mtot)
#   return(deltaMF)

# }


@checkify.checkify
def imr_phenom_x_get_and_set_precession_variables(
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
) -> tuple[IMRPhenomXWaveformDataClass, IMRPhenomXPrecessionDataClass, IMRPhenomXPHMParameterDataClass]:
    """
    Implementation of IMRPhenomXGetAndSetPrecessionVariables function from LALSimulation.
    """

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
    m1_3 = m1 * m1_2
    m1_4 = m1 * m1_3
    m1_5 = m1 * m1_4
    m1_6 = m1 * m1_5
    m1_7 = m1 * m1_6
    m1_8 = m1 * m1_7

    m2_2 = m2 * m2

    # I'm keeping this here, but note that these three lines have been moved to
    # the setting of IMRPhenomXPHMParameterDataClass
    #   p_wf->M = M
    #   p_wf->m1_2 = m1_2
    #   p_wf->m2_2 = m2_2

    q = m1 / m2  # q = m1 / m2 > 1.0

    # // Powers of eta
    eta = p_wf.eta
    eta2 = eta * eta
    eta3 = eta * eta2
    eta4 = eta * eta3
    eta5 = eta * eta4
    eta6 = eta * eta5

    # // \delta in terms of q > 1
    delta = p_wf.delta
    delta2 = delta * delta
    delta3 = delta * delta2

    # // Cache these powers, as we use them regularly
    inveta = 1.0 / eta
    inveta2 = 1.0 / eta2
    inveta3 = 1.0 / eta3
    inveta4 = 1.0 / eta4
    sqrt_inveta = 1.0 / jnp.sqrt(eta)

    chi_eff = p_wf.chi_eff

    twopi_gm = 2 * PI * G * (m1_si + m2_si) / C**3
    pi_gm = PI * G * (m1_si + m2_si) / C**3

    # /* Set spin variables in p_prec struct */
    chi1_norm = jnp.sqrt(chi1x * chi1x + chi1y * chi1y + chi1z * chi1z)

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

    # /* Norm of in-plane sum: Norm[ S1perp + S2perp ] */
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
    # precversion_tag = (p_prec.imr_phenom_x_prec_version - (p_prec.imr_phenom_x_prec_version % 100)) / 100
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
    p_prec = imr_phenom_x_pnr_get_and_set_pnr_variables(p_wf, p_prec)

    alpha_pnr = 0.0
    beta_pnr = 0.0
    gamma_pnr = 0.0

    # /*...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/
    # /*   Get and/or store CoPrec params into p_wf and p_prec    */
    # /*...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

    _, (p_wf, p_prec) = imr_phenom_x_pnr_get_and_set_co_prec_params(p_wf, p_prec, lal_params)

    # /*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

    is_msa = jnp.logical_or(jnp.logical_or(pflag == 220, pflag == 221), jnp.logical_or(pflag == 223, pflag == 224))

    _, p_prec = jax.lax.cond(
        is_msa,
        lambda args: imr_phenom_x_initialize_msa_system(*args)[1],
        lambda args: (args[0], args[1]),
        operand=(p_wf, p_prec, p_prec.expansion_order),
    )
    # If MSA fails to initialize, and we are in a version that allows fallback, set to NNLO PN angles with 3PN L
    imr_phenom_x_prec_version = jax.lax.select(
        jnp.logical_and(jnp.logical_or(jnp.logical_or(pflag == 220, pflag == 223), pflag == 224), is_msa),
        102,  # In version 220, 223 and 224 if the MSA system fails to initialize we default to the NNLO PN angles using the 3PN aligned-spin orbital angular momentum
        imr_phenom_x_prec_version,
    )

    # /*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
    # /      Compute and set final spin and RD frequency           /
    # /...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/
    p_wf = imr_phenom_x_set_precessing_remnant_params(p_wf, p_prec, lal_params)
    # /*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

    # /* Useful powers of \chi_p */
    chip2 = chip * chip

    # /* Useful powers of spins aligned with L */
    chi1l2 = chi1_l * chi1_l
    chi2l2 = chi2_l * chi2_l

    log16 = 2.772588722239781

    # /*  Cache the orbital angular momentum coefficients for future use.

    #     References:
    #     - Kidder, PRD, 52, 821-847, (1995), arXiv:gr-qc/9506022
    #     - Blanchet, LRR, 17, 2, (2014), arXiv:1310.1528
    #     - Bohe et al, 1212.5520v2
    #     - Marsat, CQG, 32, 085008, (2015), arXiv:1411.4118
    # */

    def branch_2_pn_ns():
        """
        2PN non-spinning orbital angular momentum (as per IMRPhenomPv2)
        """
        l0 = 1.0
        l1 = 0.0
        l2 = (3.0 / 2.0) + (eta / 6.0)
        l3 = 0.0
        l4 = (81.0 + (-57.0 + eta) * eta) / 24.0
        l5 = 0.0
        l6 = 0.0
        l7 = 0.0
        l8 = 0.0
        l8l = 0.0
        return l0, l1, l2, l3, l4, l5, l6, l7, l8, l8l

    def branch_3_pn():
        """
        3PN orbital angular momentum
        """
        l0 = 1.0
        l1 = 0.0
        l2 = 3.0 / 2.0 + eta / 6.0
        l3 = (5 * (chi1_l * (-2 - 2 * delta + eta) + chi2_l * (-2 + 2 * delta + eta))) / 6.0
        l4 = (81 + (-57 + eta) * eta) / 24.0
        l5 = (
            -7
            * (
                chi1_l * (72 + delta * (72 - 31 * eta) + eta * (-121 + 2 * eta))
                + chi2_l * (72 + eta * (-121 + 2 * eta) + delta * (-72 + 31 * eta))
            )
        ) / 144.0
        l6 = (10935 + eta * (-62001 + eta * (1674 + 7 * eta) + 2214 * powers_of_lalpi.two)) / 1296.0
        l7 = 0.0
        l8 = 0.0
        l8l = 0.0
        return l0, l1, l2, l3, l4, l5, l6, l7, l8, l8l

    def branch_3_pn_non_conserved_spin_norms():
        """
        3PN orbital angular momentum using non-conserved spin norms as per LALSimInspiralFDPrecAngles.c
        """
        l0 = 1.0
        l1 = 0.0
        l2 = 3.0 / 2.0 + eta / 6.0
        l3 = (-7 * (chi1_l + chi2_l + chi1_l * delta - chi2_l * delta) + 5 * (chi1_l + chi2_l) * eta) / 6.0
        l4 = (81 + (-57 + eta) * eta) / 24.0
        l5 = (
            -1650 * (chi1_l + chi2_l + chi1_l * delta - chi2_l * delta)
            + 1336 * (chi1_l + chi2_l) * eta
            + 511 * (chi1_l - chi2_l) * delta * eta
            + 28 * (chi1_l + chi2_l) * eta2
        ) / 600.0
        l6 = (10935 + eta * (-62001 + 1674 * eta + 7 * eta2 + 2214 * powers_of_lalpi.two)) / 1296.0
        l7 = 0.0
        l8 = 0.0
        l8l = 0.0
        return l0, l1, l2, l3, l4, l5, l6, l7, l8, l8l

    def branch_4_pn():
        """
        4PN orbital angular momentum
        """
        l0 = 1.0
        l1 = 0.0
        l2 = 3.0 / 2.0 + eta / 6.0
        l3 = (5 * (chi1_l * (-2 - 2 * delta + eta) + chi2_l * (-2 + 2 * delta + eta))) / 6.0
        l4 = (81 + (-57 + eta) * eta) / 24.0
        l5 = (
            -7
            * (
                chi1_l * (72 + delta * (72 - 31 * eta) + eta * (-121 + 2 * eta))
                + chi2_l * (72 + eta * (-121 + 2 * eta) + delta * (-72 + 31 * eta))
            )
        ) / 144.0
        l6 = (10935 + eta * (-62001 + eta * (1674 + 7 * eta) + 2214 * powers_of_lalpi.two)) / 1296.0
        l7 = (
            chi2_l * (-324 + eta * (1119 - 2 * eta * (172 + eta)) + delta * (324 + eta * (-633 + 14 * eta)))
            - chi1_l * (324 + eta * (-1119 + 2 * eta * (172 + eta)) + delta * (324 + eta * (-633 + 14 * eta)))
        ) / 32.0
        l8 = (
            2835 / 128.0
            - (
                eta
                * (
                    -10677852
                    + 100 * eta * (-640863 + eta * (774 + 11 * eta))
                    + 26542080 * EulerGamma
                    + 675 * (3873 + 3608 * eta) * powers_of_lalpi.two
                )
            )
            / 622080.0
            - (64 * eta * log16) / 3.0
        )

        l8l = -(64.0 / 3.0) * eta
        return l0, l1, l2, l3, l4, l5, l6, l7, l8, l8l

    def branch_4_pn_lo_spin():
        """
        4PN orbital angular momentum + leading order in spin at all PN orders terms.
        - Marsat, CQG, 32, 085008, (2015), arXiv:1411.4118
        - Siemonsen et al, PRD, 97, 064010, (2018), arXiv:1606.08832
        """
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l8l = branch_4_pn()
        l4 += (chi1l2 * (1 + delta - 2 * eta) + 4 * chi1_l * chi2_l * eta - chi2l2 * (-1 + delta + 2 * eta)) / 2.0
        l7 += (
            3
            * (chi1_l + chi2_l)
            * eta
            * (chi1l2 * (1 + delta - 2 * eta) + 4 * chi1_l * chi2_l * eta - chi2l2 * (-1 + delta + 2 * eta))
        ) / 4.0
        return l0, l1, l2, l3, l4, l5, l6, l7, l8, l8l

    def pflag_to_branch_index(pflag: int) -> int:
        """Convert pflag to branch index using arithmetic encoding."""
        # Create a lookup using where operations

        is_101 = pflag == 101
        is_102_group = jnp.isin(pflag, jnp.array([102, 220, 221, 224, 310, 311, 320, 321, 330]))
        is_222_223 = jnp.isin(pflag, jnp.array([222, 223]))
        is_103 = pflag == 103
        is_104 = pflag == 104

        # Use weighted sum to get index
        index = 1 * is_101 + 2 * is_102_group + 3 * is_222_223 + 4 * is_103 + 5 * is_104
        return int(index)

    index = pflag_to_branch_index(pflag)
    checkify.check(
        index != 0,
        "Error: IMRPhenomXPrecVersion not recognized. Requires version 101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321 or 330.",
    )
    l0, l1, l2, l3, l4, l5, l6, l7, l8, l8l = jax.lax.switch(
        index - 1, [branch_2_pn_ns, branch_3_pn, branch_3_pn_non_conserved_spin_norms, branch_4_pn, branch_4_pn_lo_spin]
    )

    # /* Reference orbital angular momentum */
    l_ref = (
        big_m
        * big_m
        * xlal_sim_imr_phenom_x_lpn_ansatz(p_wf.v_ref, p_wf.eta / p_wf.v_ref, l0, l1, l2, l3, l4, l5, l6, l7, l8, l8l)
    )

    # /*
    # In the following code block we construct the conventions that relate the source frame and the LAL frame.

    # A detailed discussion of the conventions can be found in Appendix C and D of arXiv:2004.06503 and https://dcc.ligo.org/LIGO-T1500602
    # */

    # /* Get source frame (*_Sf) J = L + S1 + S2. This is an instantaneous frame in which L is aligned with z */
    j0x_sf = (m1_2) * chi1x + (m2_2) * chi2x
    j0y_sf = (m1_2) * chi1y + (m2_2) * chi2y
    j0z_sf = (m1_2) * chi1z + (m2_2) * chi2z + l_ref

    j0 = jnp.sqrt(j0x_sf * j0x_sf + j0y_sf * j0y_sf + j0z_sf * j0z_sf)

    # /* Get angle between J0 and LN (z-direction) */
    theta_j_sf = jax.lax.select(jnp.abs(j0) < 1e-10, 0.0, jnp.arccos(j0z_sf / j0))

    phi_ref = p_wf.phi_ref_in

    convention = lal_params.convention

    is_valid_convention = jnp.isin(convention, jnp.array([0, 1, 5, 6, 7]))
    checkify.check(
        is_valid_convention, "Error: IMRPhenomXPConvention not recognized. Requires version 0, 1, 5, 6 or 7."
    )

    # /* Get azimuthal angle of J0 in the source frame */
    phi_j_sf = jax.lax.select(
        jnp.logical_and(jnp.abs(j0x_sf) < MAX_TOL_ATAN, jnp.abs(j0y_sf) < MAX_TOL_ATAN),
        jax.lax.select(  # Impose the aligned spin limit
            jnp.logical_or(convention == 0, convention == 5), (PI / 2.0) - phi_ref, 0.0
        ),
        jnp.arctan2(j0y_sf, j0x_sf),  # azimuthal angle of J0 in the source frame
    )
    phi0_aligned = -phi_j_sf

    phi0 = jax.lax.select(convention == 0, phi0_aligned, jax.lax.select(convention == 1, 0.0, p_wf.phi0))
    p_wf = p_wf = dataclasses.replace(p_wf, phi0=phi0)

    # /*
    #     Here we follow the same prescription as in IMRPhenomPv2:

    #     Now rotate from SF to J frame to compute alpha0, the azimuthal angle of LN, as well as
    #     thetaJ, the angle between J and N.

    #     The J frame is defined by imposing that J points in the z-direction and the line of sight N is in the xz-plane
    #     (with positive projection along x).

    #     The components of any in the (new) J-frame can be obtained by rotation from the (old) source frame (SF).
    #     This is done by multiplying by: RZ[-kappa].RY[-thetaJ].RZ[-phiJ]

    #     Note that kappa is determined by rotating N with RY[-thetaJ].RZ[-phiJ], which brings J to the z-axis, and
    #     taking the opposite of the azimuthal angle of the rotated N.
    # */

    # /* Determine kappa via rotations, as above */
    nx_sf = jnp.sin(p_wf.inclination) * jnp.cos((PI / 2.0) - phi_ref)
    ny_sf = jnp.sin(p_wf.inclination) * jnp.sin((PI / 2.0) - phi_ref)
    nz_sf = jnp.cos(p_wf.inclination)

    tmp_x = nx_sf
    tmp_y = ny_sf
    tmp_z = nz_sf

    tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_z(-phi_j_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_y(-theta_j_sf, tmp_x, tmp_y, tmp_z)

    # /* Note difference in overall - sign w.r.t PhenomPv2 code */
    kappa = xlal_sim_imr_phenom_x_atan2tol(tmp_y, tmp_x, MAX_TOL_ATAN)

    # /* Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1} */
    tmp_x = 0.0
    tmp_y = 0.0
    tmp_z = 1.0
    tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_z(-phi_j_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_y(-theta_j_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_z(-kappa, tmp_x, tmp_y, tmp_z)

    alpha0 = jax.lax.select(
        jnp.logical_and(jnp.abs(tmp_x) < MAX_TOL_ATAN, jnp.abs(tmp_y) < MAX_TOL_ATAN),
        jax.lax.select(jnp.logical_or(convention == 0, convention == 5), PI, PI - kappa),  # Aligned spin case
        jax.lax.select(jnp.logical_or(convention == 0, convention == 5), jnp.arctan2(tmp_y, tmp_x), PI - kappa),
    )

    def convention_05_branch():
        # Now determine thetaJN by rotating N
        tmp_x = nx_sf
        tmp_y = ny_sf
        tmp_z = nz_sf
        tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_z(-phi_j_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_y(-theta_j_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_z(-kappa, tmp_x, tmp_y, tmp_z)

        # /* We don't need the y-component but we will store it anyway */
        nx_jf = tmp_x
        ny_jf = tmp_y
        nz_jf = tmp_z

        # /* This is a unit vector, so no normalization */
        theta_jn = jnp.arccos(nz_jf)
        return nx_jf, ny_jf, nz_jf, theta_jn

    def convention_167_branch():
        theta_jn = theta_j_sf
        nz_jf = jnp.cos(theta_jn)
        nx_jf = jnp.sin(theta_jn)
        ny_jf = 0.0
        return nx_jf, ny_jf, nz_jf, theta_jn

    nx_jf, ny_jf, nz_jf, theta_jn = jax.lax.cond(
        jnp.logical_or(convention == 0, convention == 5),
        lambda _: convention_05_branch(),
        lambda _: convention_167_branch(),
        operand=None,
    )

    # /*
    #     Define the polarizations used. This follows the conventions adopted for IMRPhenomPv2.

    #     The IMRPhenomP polarizations are defined following the conventions in Arun et al (arXiv:0810.5336),
    #     i.e. projecting the metric onto the P, Q, N triad defining where: P = (N x J) / |N x J|.

    #     However, the triad X,Y,N used in LAL (the "waveframe") follows the definition in the
    #     NR Injection Infrastructure (Schmidt et al, arXiv:1703.01076).

    #     The triads differ from each other by a rotation around N by an angle \zeta. We therefore need to rotate
    #     the polarizations by an angle 2 \zeta.
    # */

    xx_sf = -jnp.cos(p_wf.inclination) * jnp.sin(phi_ref)
    xy_sf = -jnp.cos(p_wf.inclination) * jnp.cos(phi_ref)
    xz_sf = +jnp.sin(p_wf.inclination)

    tmp_x = xx_sf
    tmp_y = xy_sf
    tmp_z = xz_sf

    tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_z(-phi_j_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_y(-theta_j_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = imr_phenom_x_rotate_z(-kappa, tmp_x, tmp_y, tmp_z)

    # /*
    #     The components tmp_i are now the components of X in the J frame.

    #     We now need the polar angle of this in the P, Q basis of Arun et al:

    #         P = (N x J) / |NxJ|

    #     Note, that we put N in the (pos x)z half plane of the J frame
    # */

    # Get polar angle of X in J frame in the P,Q basis of Arun et al
    # Q = (N x P) by construction
    pq_arun = jax.lax.select(
        jnp.logical_or(convention == 0, convention == 5),
        (0.0, -1.0, 0.0, nz_jf, 0.0, -nx_jf),
        (nz_jf, 0.0, -nx_jf, 0.0, 1.0, 0.0),
    )
    p_arun_x_jf, p_arun_y_jf, p_arun_z_jf, q_arun_x_jf, q_arun_y_jf, q_arun_z_jf = pq_arun

    # // (X . P)
    x_dot_p_arun = (tmp_x * p_arun_x_jf) + (tmp_y * p_arun_y_jf) + (tmp_z * p_arun_z_jf)

    # // (X . Q)
    x_dot_q_arun = (tmp_x * q_arun_x_jf) + (tmp_y * q_arun_y_jf) + (tmp_z * q_arun_z_jf)

    # /* Now get the angle zeta */
    zeta_polarization = jnp.arctan2(x_dot_q_arun, x_dot_p_arun)

    # /* ********** PN Euler Angle Coefficients ********** */
    # /*
    #     This uses the single spin PN Euler angles as per IMRPhenomPv2
    # */

    # /* ********** PN Euler Angle Coefficients ********** */
    def get_imr_phenom_pv2_pn_euler_angle_coeffs():
        # This uses the single spin PN Euler angles as per IMRPhenomPv2

        # /* Post-Newtonian Euler Angles: alpha */
        chi_l = (1.0 + q) * (chi_eff / q)
        chi_l2 = chi_l * chi_l

        alpha1 = -35 / 192.0 + (5 * delta) / (64.0 * m1)

        alpha2 = ((15 * chi_l * delta * m1) / 128.0 - (35 * chi_l * m1_2) / 128.0) / eta

        alpha3 = (
            -5515 / 3072.0
            + eta * (-515 / 384.0 - (15 * delta2) / (256.0 * m1_2) + (175 * delta) / (256.0 * m1))
            + (4555 * delta) / (7168.0 * m1)
            + ((15 * chip2 * delta * m1_3) / 128.0 - (35 * chip2 * m1_4) / 128.0) / eta2
        )

        # /* This is the term proportional to log(w) */
        alpha4l = (
            (5 * chi_l * delta2) / 16.0
            - (5 * chi_l * delta * m1) / 3.0
            + (2545 * chi_l * m1_2) / 1152.0
            + ((-2035 * chi_l * delta * m1) / 21504.0 + (2995 * chi_l * m1_2) / 9216.0) / eta
            + ((5 * chi_l * chip2 * delta * m1_5) / 128.0 - (35 * chi_l * chip2 * m1_6) / 384.0) / eta3
            - (35 * PI) / 48.0
            + (5 * delta * PI) / (16.0 * m1)
        )

        alpha5 = (
            5
            * (
                -190512 * delta3 * eta6
                + 2268 * delta2 * eta3 * m1 * (eta2 * (323 + 784 * eta) + 336 * (25 * chi_l2 + chip2) * m1_4)
                + 7
                * m1_3
                * (
                    8024297 * eta4
                    + 857412 * eta5
                    + 3080448 * eta6
                    + 143640 * chip2 * eta2 * m1_4
                    - 127008 * chip2 * (-4 * chi_l2 + chip2) * m1_8
                    + 6048 * eta3 * ((2632 * chi_l2 + 115 * chip2) * m1_4 - 672 * chi_l * m1_2 * PI)
                )
                + 3
                * delta
                * m1_2
                * (
                    -5579177 * eta4
                    + 80136 * eta5
                    - 3845520 * eta6
                    + 146664 * chip2 * eta2 * m1_4
                    + 127008 * chip2 * (-4 * chi_l2 + chip2) * m1_8
                    - 42336 * eta3 * ((726 * chi_l2 + 29 * chip2) * m1_4 - 96 * chi_l * m1_2 * PI)
                )
            )
        ) / (6.5028096e7 * eta4 * m1_3)

        # /* Post-Newtonian Euler Angles: epsilon */
        epsilon1 = -35 / 192.0 + (5 * delta) / (64.0 * m1)

        epsilon2 = ((15 * chi_l * delta * m1) / 128.0 - (35 * chi_l * m1_2) / 128.0) / eta

        epsilon3 = (
            -5515 / 3072.0
            + eta * (-515 / 384.0 - (15 * delta2) / (256.0 * m1_2) + (175 * delta) / (256.0 * m1))
            + (4555 * delta) / (7168.0 * m1)
        )

        # /* This term is proportional to log(w) */
        epsilon4l = (
            (5 * chi_l * delta2) / 16.0
            - (5 * chi_l * delta * m1) / 3.0
            + (2545 * chi_l * m1_2) / 1152.0
            + ((-2035 * chi_l * delta * m1) / 21504.0 + (2995 * chi_l * m1_2) / 9216.0) / eta
            - (35 * PI) / 48.0
            + (5 * delta * PI) / (16.0 * m1)
        )

        epsilon5 = (
            5
            * (
                -190512 * delta3 * eta3
                + 2268 * delta2 * m1 * (eta2 * (323 + 784 * eta) + 8400 * chi_l2 * m1_4)
                - 3
                * delta
                * m1_2
                * (
                    eta * (5579177 + 504 * eta * (-159 + 7630 * eta))
                    + 254016 * chi_l * m1_2 * (121 * chi_l * m1_2 - 16 * PI)
                )
                + 7
                * m1_3
                * (
                    eta * (8024297 + 36 * eta * (23817 + 85568 * eta))
                    + 338688 * chi_l * m1_2 * (47 * chi_l * m1_2 - 12 * PI)
                )
            )
        ) / (6.5028096e7 * eta * m1_3)

        return alpha1, alpha2, alpha3, alpha4l, alpha5, epsilon1, epsilon2, epsilon3, epsilon4l, epsilon5

    is_euler_as_pv2 = jnp.isin(pflag, jnp.array([101, 102, 103, 104]))
    alpha1, alpha2, alpha3, alpha4l, alpha5, epsilon1, epsilon2, epsilon3, epsilon4l, epsilon5 = jax.lax.cond(
        is_euler_as_pv2,
        get_imr_phenom_pv2_pn_euler_angle_coeffs,
        lambda: (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),  # I'm not adding the prec version check here, but earlier code should have already done this
        operand=None,
    )

    epsilon0 = jax.lax.select(
        jnp.logical_or(convention == 1, convention == 6),
        phi_j_sf - PI,
        0.0,
    )

    # Update p_prec
    p_prec = dataclasses.replace(
        p_prec,
        alpha_pnr=alpha_pnr,
        beta_pnr=beta_pnr,
        gamma_pnr=gamma_pnr,
        l0=l0,
        l1=l1,
        l2=l2,
        l3=l3,
        l4=l4,
        l5=l5,
        l6=l6,
        l7=l7,
        l8=l8,
        l8_l=l8l,
        l_ref=l_ref,
        j0x_sf=j0x_sf,
        j0y_sf=j0y_sf,
        j0z_sf=j0z_sf,
        j0=j0,
        theta_j_sf=theta_j_sf,
        phi_j_sf=phi_j_sf,
        phi0_aligned=phi0_aligned,
        nx_sf=nx_sf,
        ny_sf=ny_sf,
        nz_sf=nz_sf,
        kappa=kappa,
        alpha0=alpha0,
        theta_jn=theta_jn,
        nx_jf=nx_jf,
        ny_jf=ny_jf,
        nz_jf=nz_jf,
        xx_sf=xx_sf,
        xy_sf=xy_sf,
        xz_sf=xz_sf,
        p_arun_x_jf=p_arun_x_jf,
        p_arun_y_jf=p_arun_y_jf,
        p_arun_z_jf=p_arun_z_jf,
        q_arun_x_jf=q_arun_x_jf,
        q_arun_y_jf=q_arun_y_jf,
        q_arun_z_jf=q_arun_z_jf,
        x_dot_p_arun=x_dot_p_arun,
        x_dot_q_arun=x_dot_q_arun,
        zeta_polarization=zeta_polarization,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        alpha4l=alpha4l,
        alpha5=alpha5,
        epsilon0=epsilon0,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        epsilon3=epsilon3,
        epsilon4l=epsilon4l,
        epsilon5=epsilon5,
    )

    def alpha_epsilon_offsets_57_branch(p_prec, p_wf):
        _, _ = p_prec, p_wf  # Unused
        alpha_offset = -alpha0
        epsilon_offset = 0.0
        alpha_offset_1 = -alpha0
        epsilon_offset_1 = 0.0
        alpha_offset_3 = -alpha0
        epsilon_offset_3 = 0.0
        alpha_offset_4 = -alpha0
        epsilon_offset_4 = 0.0
        return (
            alpha_offset,
            epsilon_offset,
            alpha_offset_1,
            epsilon_offset_1,
            alpha_offset_3,
            epsilon_offset_3,
            alpha_offset_4,
            epsilon_offset_4,
        )

    def alpha_epsilon_offsets_other_branch(p_prec, p_wf):
        # /* Get initial Get \alpha and \epsilon offsets at \omega = pi * M * f_{Ref} */
        alpha_offset, epsilon_offset, p_prec = get_alphaepsilon_atfref(
            2,
            p_prec,
            p_wf,
        )
        alpha_offset_1 = alpha_offset
        epsilon_offset_1 = epsilon_offset
        alpha_offset_3 = alpha_offset
        epsilon_offset_3 = epsilon_offset
        alpha_offset_4 = alpha_offset
        epsilon_offset_4 = epsilon_offset
        return (
            alpha_offset,
            epsilon_offset,
            alpha_offset_1,
            epsilon_offset_1,
            alpha_offset_3,
            epsilon_offset_3,
            alpha_offset_4,
            epsilon_offset_4,
        )

    (
        alpha_offset,
        epsilon_offset,
        alpha_offset_1,
        epsilon_offset_1,
        alpha_offset_3,
        epsilon_offset_3,
        alpha_offset_4,
        epsilon_offset_4,
    ) = jax.lax.cond(
        jnp.logical_or(convention == 5, convention == 7),
        alpha_epsilon_offsets_57_branch,
        alpha_epsilon_offsets_other_branch,
        operand=(p_prec, p_wf),
    )

    cexp_i_alpha = 0.0
    cexp_i_epsilon = 0.0
    cexp_i_betah = 0.0

    # /*
    #     Check whether maximum opening angle becomes larger than \pi/2 or \pi/4.

    #     If (L + S_L) < 0, then Wigner-d Coefficients will not track the angle between J and L, meaning
    #     that the model may become pathological as one moves away from the aligned-spin limit.

    #     If this does not happen, then max_beta will be the actual maximum opening angle.

    #     This function uses a 2PN non-spinning approximation to the orbital angular momentum L, as
    #     the roots can be analytically derived.

    #     Returns XLAL_PRINT_WARNING if model is in a pathological regime.
    # */

    # // When L + SL < 0 and q>7, we disable multibanding
    lal_params, p_prec = imr_phenom_xp_check_max_opening_angle(p_wf, p_prec, lal_params)

    # /* Activate multibanding for Euler angles it threshold !=0. Only for PhenomXPHM. */

    def handle_multibanding_warnings(p_wf, p_prec, lal_params, pflag):
        """Handle multibanding warnings and disable if necessary."""

        def _warn_high_mass():
            print(
                "Very high mass, only merger in frequency band, multibanding not efficient, switching off for non-precessing modes and Euler angles."
            )

        def _warn_multiband_330():
            print("Warning: Multibanding may lead to pathological behaviour in this case. Disabling multibanding.")

        def _warn_high_q_nnlo():
            print("Very high mass ratio, NNLO angles may become pathological, switching off multibanding for angles.")

        # Determine which condition applies (mutually exclusive, checked in order)
        cond_high_mass = p_wf.m_tot > 500.0  # /* Switch off multiband for very high mass as in IMRPhenomXHM. */
        cond_330_high_q = jnp.logical_and(pflag == 330, p_wf.q > 7.0)  # this is here as a safety catch in case
        cond_nnlo_high_q = jnp.logical_and(
            pflag < 200, p_wf.q > 8.0
        )  # The NNLO angles can have a worse, even pathological, behaviour for
        #     high mass ratio and spin cases. The waveform will look noisy,
        #     we switch off the multibanding for mass ratio above 8 to avoid
        #     worsen even more the waveform.
        cond_msa_extreme = jnp.logical_and(
            p_wf.q > 50, p_wf.m_tot > 100
        )  # The MSA angles give quite 'noisy' waveforms in this corner of parameter space so we switch off multibanding to avoid worsen the waveform.

        # Any condition triggers disabling multibanding
        should_disable = jnp.logical_or(
            jnp.logical_or(cond_high_mass, cond_330_high_q), jnp.logical_or(cond_nnlo_high_q, cond_msa_extreme)
        )

        # Compute index for warning selection: 0=none, 1=high_mass, 2=330, 3=nnlo, 4=msa(silent)
        index = jnp.where(
            cond_high_mass,
            1,
            jnp.where(cond_330_high_q, 2, jnp.where(cond_nnlo_high_q, 3, jnp.where(cond_msa_extreme, 4, 0))),
        )

        def no_warning():
            pass

        def warn_high_mass():
            jax.debug.callback(_warn_high_mass)

        def warn_330():
            jax.debug.callback(_warn_multiband_330)

        def warn_nnlo():
            jax.debug.callback(_warn_high_q_nnlo)

        def warn_msa():
            pass  # MSA case has no warning in original code

        # Issue appropriate warning
        jax.lax.switch(index, [no_warning, warn_high_mass, warn_330, warn_nnlo, warn_msa])

        # Disable multibanding if any condition met
        new_lal_params = jax.lax.cond(
            should_disable, lambda lp: dataclasses.replace(lp, threshold_mband=0.0), lambda lp: lp, lal_params
        )

        new_mband_version = jax.lax.select(should_disable, 0, p_prec.m_band_prec_version)

        return new_mband_version, new_lal_params

    m_band_prec_version, lal_params = handle_multibanding_warnings(p_wf, p_prec, lal_params, pflag)

    ytheta = p_prec.theta_jn
    # yphi    = 0.0
    y2m2 = compute_sminus2_l2(ytheta, -2)
    y2m1 = compute_sminus2_l2(ytheta, -1)
    y20 = compute_sminus2_l2(ytheta, 0)
    y21 = compute_sminus2_l2(ytheta, 1)
    y22 = compute_sminus2_l2(ytheta, 2)
    y3m3 = compute_sminus2_l3(ytheta, -3)
    y3m2 = compute_sminus2_l3(ytheta, -2)
    y3m1 = compute_sminus2_l3(ytheta, -1)
    y30 = compute_sminus2_l3(ytheta, 0)
    y31 = compute_sminus2_l3(ytheta, 1)
    y32 = compute_sminus2_l3(ytheta, 2)
    y33 = compute_sminus2_l3(ytheta, 3)
    y4m4 = compute_sminus2_l4(ytheta, -4)
    y4m3 = compute_sminus2_l4(ytheta, -3)
    y4m2 = compute_sminus2_l4(ytheta, -2)
    y4m1 = compute_sminus2_l4(ytheta, -1)
    y40 = compute_sminus2_l4(ytheta, 0)
    y41 = compute_sminus2_l4(ytheta, 1)
    y42 = compute_sminus2_l4(ytheta, 2)
    y43 = compute_sminus2_l4(ytheta, 3)
    y44 = compute_sminus2_l4(ytheta, 4)

    p_prec = dataclasses.replace(
        p_prec,
        m_band_prec_version=m_band_prec_version,
        cexp_i_alpha=cexp_i_alpha,
        cexp_i_epsilon=cexp_i_epsilon,
        cexp_i_betah=cexp_i_betah,
        alpha_offset=alpha_offset,
        epsilon_offset=epsilon_offset,
        alpha_offset_1=alpha_offset_1,
        epsilon_offset_1=epsilon_offset_1,
        alpha_offset_3=alpha_offset_3,
        epsilon_offset_3=epsilon_offset_3,
        alpha_offset_4=alpha_offset_4,
        epsilon_offset_4=epsilon_offset_4,
        y2m2=y2m2,
        y2m1=y2m1,
        y20=y20,
        y21=y21,
        y22=y22,
        y3m3=y3m3,
        y3m2=y3m2,
        y3m1=y3m1,
        y30=y30,
        y31=y31,
        y32=y32,
        y33=y33,
        y4m4=y4m4,
        y4m3=y4m3,
        y4m2=y4m2,
        y4m1=y4m1,
        y40=y40,
        y41=y41,
        y42=y42,
        y43=y43,
        y44=y44,
        lal_params=lal_params,
    )

    return p_wf, p_prec, lal_params


def imr_phenom_x_set_precessing_remnant_params(
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    lal_params: IMRPhenomXPHMParameterDataClass,
) -> IMRPhenomXWaveformDataClass:
    """
    Function to set remnant quantities related to final spin within IMRPhenomXGetAndSetPrecessionVariables
    """

    # /*

    # The strategy for setting the precessing remnant spin in PhenomX
    # is multifaceted because there are not only the various options
    # implemented for the original PhenomXP, but there are also the
    # options needed for PNR's construction, and its turning off of
    # its CoPrecessing model outside of the PNR calibration region.
    # In that latter case, the final spin transitions from the
    # non-precessing final spin, where PNR's CoPrecessing model is
    # tuned, to the precessing final spin, as is needed for the EZH
    # effective ringdown result.

    # A layer on top of this, is the need for the l=m=2 fundamental
    # QNM frequency to be computed, in some way, amid all of these
    # scenarios.

    # This function exists to draw a conceptual circle around all of
    # this mess.

    # Most comments below are cogent, but some have been left for
    # historical record?

    # */

    # /* Define shorthand variables for PNR CoPrec options */

    # // Toggle for PNR coprecessing tuning
    pnr_use_input_coprec_deviations = p_prec.imr_phenom_xpnr_use_input_coprec_deviations

    # // Toggle for enforced use of non-precessing spin as is required during tuning of PNR's coprecessing model
    pnr_use_tuned_coprec = p_prec.imr_phenom_xpnr_use_tuned_coprec

    # // High-level toggle for whether to apply deviations
    apply_pnr_deviations = p_wf.apply_pnr_deviations

    # /*
    # HISTORICAL COMMENT
    #     Update the final spin in pWF to account for precessing spin effects:

    #     XLALSimIMRPhenomXPrecessingFinalSpin2017(eta,chi1L,chi2L,chi_perp)

    #     Note that chi_perp gets weighted by an appropriate mass factor

    #     q_factor      = m1/M (when m1 > m2)
    #     Sperp   = chip * q_factor * q_factor
    #     af      = copysign(1.0, af_parallel) * sqrt(Sperp*Sperp + af_parallel*af_parallel)
    # */
    big_m = p_wf.m
    af_parallel = xlal_sim_imr_phenom_x_final_spin_2017(p_wf.eta, p_prec.chi1z, p_prec.chi2z)
    l_final = big_m * big_m * af_parallel - p_wf.m1_2 * p_prec.chi1z - p_wf.m2_2 * p_prec.chi2z

    fsflag = lal_params.xp_final_spin_mod
    fsflag = jax.lax.select(jnp.logical_and(fsflag == 4, p_prec.precessing_tag == 3), 3, fsflag)

    # /* For PhenomPNR, we wil use the PhenomPv2 final spin function's result, modified such that its sign is given by sign( cos(betaRD) ). See the related fsflag case below. */
    fsflag = jax.lax.select(jnp.logical_and(pnr_use_tuned_coprec, fsflag < 6), 5, fsflag)

    # /* When tuning the coprecessing model, we wish to enforce use of the non-precessing final spin. See the related fsflag case below. */
    fsflag = jax.lax.select(pnr_use_input_coprec_deviations, 6, fsflag)

    # /* Generate and store ringdown value of precession angle beta. This is to be used for e.g. setting the sign of the final spin, and calculating the effective ringdown frequency */
    def call_imr_phenom_x_pnr_generate_ringdown_pnr_beta(operand):
        p_wf, p_prec = operand
        return imr_phenom_x_pnr_generate_ringdown_pnr_beta(p_wf, p_prec)

    beta_rd = jax.lax.cond(
        pnr_use_tuned_coprec,
        call_imr_phenom_x_pnr_generate_ringdown_pnr_beta,
        lambda x: x[0].beta_rd,
        operand=(p_wf, p_prec),
    )

    chi1_l = p_prec.chi1z
    chi2_l = p_prec.chi2z

    pflag = p_prec.imr_phenom_x_prec_version

    def case0_branch(_):
        return xlal_sim_imr_phenom_x_precessing_final_spin_2017(p_wf.eta, chi1_l, chi2_l, p_prec.chi_p)

    def case1_branch(_):
        return xlal_sim_imr_phenom_x_precessing_final_spin_2017(p_wf.eta, chi1_l, chi2_l, p_prec.chi1x)

    def case24_branch(_):
        return xlal_sim_imr_phenom_x_precessing_final_spin_2017(p_wf.eta, chi1_l, chi2_l, p_prec.chi_tot_perp)

    def case3_branch(_):
        def inner_branch(_):
            def standard_branch(_):
                sign = jnp.copysign(1.0, af_parallel)
                return (
                    sign
                    * jnp.sqrt(p_prec.s_av2 + l_final * l_final + 2.0 * l_final * (p_prec.s1_l_pav + p_prec.s2_l_pav))
                    / (big_m * big_m)
                )

            def error_branch(_):
                def _warn_msa_error():
                    print("Initialization of MSA system failed. Defaulting to final spin version 0.")

                jax.debug.callback(_warn_msa_error)
                return xlal_sim_imr_phenom_x_precessing_final_spin_2017(p_wf.eta, chi1_l, chi2_l, p_prec.chi_p)

            return jax.lax.cond(
                p_prec.msa_error == 1,
                error_branch,
                standard_branch,
                operand=None,
            )

        def outer_branch(_):
            def _warn_prec_version_error():
                print(
                    "Error: XLALSimInspiralWaveformParamsLookupPhenomXPFinalSpinMod version 3 requires PrecVersion 220, 221, 222, 223 or 224. Defaulting to version 0."
                )

            jax.debug.callback(_warn_prec_version_error)
            return xlal_sim_imr_phenom_x_precessing_final_spin_2017(p_wf.eta, chi1_l, chi2_l, p_prec.chi_p)

        return jax.lax.cond(
            jnp.isin(pflag, jnp.array([220, 221, 222, 223, 224])),
            inner_branch,
            outer_branch,
            operand=None,
        )

    def case5_branch(_):
        # /*-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~*
        # Implement Pv2 final spin but with sign derived from EZH's model for ringdown beta.
        # *-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~*/

        # // Use these as input into method for effective RD frequency
        # // * get the value of ringdown beta
        sign = jnp.copysign(1.0, jnp.cos(beta_rd))

        # /* Calculate Pv2 final spin without alteration. Below we alter it. NOTE that XLALSimIMRPhenomXPrecessingFinalSpin2017 appears to be an explicit copy of the actual Pv2 final spin function, FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH. The original PhenomX have not referenced this code duplication.  */
        afinal_prec_pv2 = xlal_sim_imr_phenom_x_precessing_final_spin_2017(p_wf.eta, chi1_l, chi2_l, p_prec.chi_p)

        # // The equivalent PhenomPv2 code reference would look like the commented line below.
        # // afinal_prec_Pv2 = FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH( m1, m2, chi1L, chi2L, chi_p )

        # Define the PNR final spin to be the Pv2 final spin magnitude, with direction given by sign of cos betaRD
        afinal_prec = sign * jnp.abs(afinal_prec_pv2)

        # // Experimental version of final spin
        # // pWF->afinal_prec = cos(pWF->betaRD) * fabs(afinal_prec_Pv2)
        return afinal_prec

    def case6_branch(_):
        # /*-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~*
        #     During PNR tuning, we wish to evaluate the coprecessing model with the same final spin that would be used in PhenomXHM. We implement this here.
        # *-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~*/
        return p_wf.a_final_non_prec

    def case7_branch(_):
        sign = jnp.copysign(1.0, jnp.cos(beta_rd))

        # Calculate final spin using the same spin version adopted for XPHM-SpinTaylor
        afinal_prec = xlal_sim_imr_phenom_x_precessing_final_spin_2017(p_wf.eta, chi1_l, chi2_l, p_prec.chi_tot_perp)

        # Define the PNR final spin to be the above final spin magnitude, with direction given by sign of cos betaRD
        return sign * jnp.abs(afinal_prec)

    afinal_prec = jax.lax.switch(
        fsflag,
        [
            case0_branch,
            case1_branch,
            case24_branch,
            case3_branch,
            case24_branch,
            case5_branch,
            case6_branch,
            case7_branch,
        ],
        operand=None,
    )

    # /* (PNRUseTunedCoprec) When not generating PNR make NO distinction between afinal and afinal_prec */
    afinal = jax.lax.select(
        jnp.logical_not(pnr_use_tuned_coprec),
        afinal_prec,
        (p_wf.pnr_window) * p_wf.a_final_non_prec + (1.0 - p_wf.pnr_window) * afinal_prec,
        # /*  ELSE, use the non-precessing final spin defined in IMRPhenomXSetWaveformVariables. XCP uses the non-precessing parameters as a base upon which to add precessing deviations. The line below is added only for clarity: pWF->afinal is already equal to pWF->afinal_nonprec as is set in IMRPhenomXSetWaveformVariables */
        # /* pWF->afinal = pWF->afinal_nonprec */
        # // ABOVE but commented out, we see what the final spin assignment WOULD BE if NO WINDOWING of coprec tuning were used
        # pWF->afinal = (pWF->pnr_window)*pWF->afinal_nonprec + (1.0-pWF->pnr_window)*pWF->afinal_prec
        # // Above: NOTE that as PNR is turned off outside of its calibration window, we want to turn on use of the precessing final spin as defined in the code section above
    )

    def afinal_gt_1_branch(afinal):
        def _warn_afinal_gt_1():
            print(f"Warning: Final spin magnitude {afinal} > 1. Setting final spin magnitude = 1.")

        jax.debug.callback(_warn_afinal_gt_1)
        return jnp.copysign(1.0, afinal)

    afinal = jax.lax.cond(
        jnp.abs(afinal) > 1.0,
        afinal_gt_1_branch,
        lambda x: x,
        operand=afinal,
    )

    def afinal_prec_gt_1_branch(afinal_prec):
        def _warn_afinal_prec_gt_1():
            print(
                f"Warning: Precessing final spin magnitude {afinal_prec} > 1. Setting precessing final spin magnitude = 1."
            )

        jax.debug.callback(_warn_afinal_prec_gt_1)
        return jnp.copysign(1.0, afinal_prec)

    afinal_prec = jax.lax.cond(
        jnp.abs(afinal_prec) > 1.0,
        afinal_prec_gt_1_branch,
        lambda x: x,
        operand=afinal_prec,
    )

    # /* Update ringdown and damping frequency: no precession to be used for PNR tuned deviations */
    f_ring = evaluate_QNMfit_fring22(afinal) / (p_wf.m_final)
    f_damp = evaluate_QNMfit_fdamp22(afinal) / (p_wf.m_final)
    # //pWF->fISCO     = XLALSimIMRPhenomXfISCO(pWF->afinal)

    # // Copy IMRPhenomXReturnCoPrec to pWF
    # pWF->IMRPhenomXReturnCoPrec = pPrec->IMRPhenomXReturnCoPrec

    def apply_pnr_deviations_branch(operand):
        f_ring, f_damp = operand

        # Add an overall deviation to the high-level ringdown frequency (PNRUseTunedCoprec)
        f_ring_dev = f_ring - (p_wf.pnr_dev_parameter * p_wf.nu5)
        f_damp_dev = f_damp + (p_wf.pnr_dev_parameter * p_wf.nu6)
        return f_ring_dev, f_damp_dev

    f_ring, f_damp = jax.lax.cond(
        apply_pnr_deviations,
        apply_pnr_deviations_branch,
        lambda x: x,
        operand=(f_ring, f_damp),
    )

    # // we want to define the quantities below if PNR is used (PNRUseTunedCoprec). In particular,  pWF->fRINGEffShiftDividedByEmm is used by the HMs
    # // Define identifiers for perturbation theory frequencies
    def define_pnr_quantities(operand):
        _, _, f_ring = operand
        f_ring22_prec = evaluate_QNMfit_fring22(afinal_prec) / (p_wf.m_final)
        f_ring21_prec = evaluate_QNMfit_fring21(afinal_prec) / (p_wf.m_final)

        # // * Calculate and store single quantity needed to determine effective ringdown frequencies for all QNMs
        f_ring_eff_shift_divided_by_emm = (1.0 - jnp.abs(jnp.cos(beta_rd))) * (f_ring22_prec - f_ring21_prec)

        # // As we turn off PNR tuning, we want to turn on use of the effective ringdown frequency
        # // NOTE that when pWF->pnr_window=0, this should reduce to the def of pWF->fRINGCP below

        emm = 2
        # NOTE that we use pWF->fRING and not fRING22_prec below because of how pWF->afinal is defined using (1-pWF->pnr_window)
        f_ring_dev = f_ring - (1.0 - p_wf.pnr_window) * emm * f_ring_eff_shift_divided_by_emm
        # pWF->fRING = (pWF->pnr_window)*pWF->fRING  -  (1-pWF->pnr_window) * emm * pWF->fRINGEffShiftDividedByEmm

        return f_ring22_prec, f_ring_eff_shift_divided_by_emm, f_ring_dev

    f_ring22_prec, f_ring_eff_shift_divided_by_emm, f_ring = jax.lax.cond(
        pnr_use_tuned_coprec,
        define_pnr_quantities,
        lambda x: x,
        operand=(p_wf.f_ring22_prec, p_wf.f_ring_eff_shift_divided_by_emm, f_ring),
    )

    p_wf = dataclasses.replace(
        p_wf,
        beta_rd=beta_rd,
        a_final=afinal,
        a_final_prec=afinal_prec,
        f_ring=f_ring,
        f_damp=f_damp,
        imr_phenom_x_return_co_prec=p_prec.imr_phenom_x_return_co_prec,
        f_ring22_prec=f_ring22_prec,
        f_ring_eff_shift_divided_by_emm=f_ring_eff_shift_divided_by_emm,
    )
    return p_wf


def xlal_sim_imr_phenom_x_lpn_ansatz(
    v,  # Input velocity  */
    l_norm,  # Orbital angular momentum normalization */
    l0,  # Newtonian orbital angular momentum (i.e. LN = 1.0*LNorm) */
    l1,  # 0.5PN Orbital angular momentum */
    l2,  # 1.0PN Orbital angular momentum */
    l3,  # 1.5PN Orbital angular momentum */
    l4,  # 2.0PN Orbital angular momentum */
    l5,  # 2.5PN Orbital angular momentum */
    l6,  # 3.0PN Orbital angular momentum */
    l7,  # 3.5PN Orbital angular momentum */
    l8,  # 4.0PN Orbital angular momentum */
    l8l,  # 4.0PN logarithmic orbital angular momentum term */
):
    """
    Docstring for xlal_sim_imr_phenom_x_lpn_ansatz

    :param v: Description
    :param l_norm: Description
    :param l0: Description
    :param l1: Description
    :param l2: Description
    :param l3: Description
    :param l4: Description
    :param l5: Description
    :param l6: Description
    :param l7: Description
    :param l8: Description
    :param l8l: Description
    """

    x = v * v
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    sqx = jnp.sqrt(x)

    #   /*
    #       Here LN is the Newtonian pre-factor: LN = \eta / \sqrt{x} :

    #       L = L_N \sum_a L_a x^{a/2}
    #         = L_N [ L0 + L1 x^{1/2} + L2 x^{2/2} + L3 x^{3/2} + ... ]

    #   */
    return l_norm * (
        l0
        + l1 * sqx
        + l2 * x
        + l3 * (x * sqx)
        + l4 * x2
        + l5 * (x2 * sqx)
        + l6 * x3
        + l7 * (x3 * sqx)
        + l8 * x4
        + l8l * x4 * jnp.log(x)
    )


# def imr_phenom_x_spin_taylor_angles_splines_all(
#     f_min: float,
#     f_max: float,
#     p_wf: IMRPhenomXWaveformDataClass,
#     p_prec: IMRPhenomXPrecessionDataClass,
#     lal_params: IMRPhenomXPHMParameterDataClass,
# ):
#     """Compute spin Taylor Euler angles splines for IMRPhenomXPHM waveform model.

#     Args:
#         f_min: Minimum frequency for angle computation.
#         f_max: Maximum frequency for angle computation.
#         p_wf: Waveform data class containing waveform parameters.
#         p_prec: Precession data class to be initialized.
#         lal_params: Parameter data class containing LAL parameters.
#     """


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
    #     p_prec->Mfmax_angles = p_wf->fRING+4.*p_wf->fDAMP
    #   else
    #     p_prec->Mfmax_angles = (MAX(p_wf->MfMax,p_wf->fRING+4.*p_wf->fDAMP)+XLALSimIMRPhenomXUtilsHztoMf(buffer,p_wf->Mtot))*2./p_prec->M_MIN

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

    #   fmaxAngles = XLALSimIMRPhenomXUtilsMftoHz(p_prec->Mfmax_angles,p_wf->Mtot)

    #   // we add a few bins to fmax to make sure we do not run into interpolation errors
    #   status = IMRPhenomX_SpinTaylorAnglesSplinesAll(fminAngles,fmaxAngles,p_wf,p_prec,lal_params)
    #   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "%s: IMRPhenomX_SpinTaylorAnglesSplinesAll failed.",__func__)


#   status = gsl_spline_eval_e(p_prec->alpha_spline, p_prec->ftrans_MRD,
#                               p_prec->alpha_acc,&p_prec->alpha_ftrans)
#   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC,
#              "%s: could not compute alpha et the end of inspiral.",__func__)

#   status = gsl_spline_eval_e(p_prec->cosbeta_spline, p_prec->ftrans_MRD,
#                               p_prec->cosbeta_acc,&p_prec->cosbeta_ftrans)
#   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC,
#              "%s: could not compute cosbeta et the end of inspiral.",__func__)    #   status = gsl_spline_eval_e(p_prec->gamma_spline, p_prec->ftrans_MRD, p_prec->gamma_acc,&p_prec->gamma_ftrans)
#   XLAL_CHECK(status == XLAL_SUCCESS, XLAL_EFUNC, "%s: could not compute gamma et the end of inspiral.",__func__)

#   return status


def imr_phenom_x_rotate_z(angle: float, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    """Rotate a around the z-axis by a given angle.

    Args:
        angle: Rotation angle in radians.
        vx: x-component of the vector.
        vy: y-component of the vector.
        vz: z-component of the vector.

    Returns:
        A tuple containing the rotated components (vx', vy', vz').
    """
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    vx_rotated = cos_angle * vx - sin_angle * vy
    vy_rotated = sin_angle * vx + cos_angle * vy
    vz_rotated = vz

    return vx_rotated, vy_rotated, vz_rotated


def imr_phenom_x_rotate_y(angle: float, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    """Rotate a around the y-axis by a given angle.

    Args:
        angle: Rotation angle in radians.
        vx: x-component of the vector.
        vy: y-component of the vector.
        vz: z-component of the vector.

    Returns:
        A tuple containing the rotated components (vx', vy', vz').
    """
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    vx_rotated = cos_angle * vx + sin_angle * vz
    vy_rotated = vy
    vz_rotated = -sin_angle * vx + cos_angle * vz

    return vx_rotated, vy_rotated, vz_rotated


def get_alphaepsilon_atfref(
    mprime: int,
    p_prec: IMRPhenomXPrecessionDataClass,
    p_wf: IMRPhenomXWaveformDataClass,
) -> tuple[float, float, IMRPhenomXPrecessionDataClass]:
    """
    Get alpha and epsilon offset depending of the mprime (second index of the non-precessing mode)
    """

    # /* Compute the offsets due to the choice of integration constant in alpha and epsilon PN formula */
    omega_ref = p_wf.pi_m * p_wf.f_ref * 2.0 / mprime

    pflag = p_prec.imr_phenom_x_prec_version

    def msa_branch(operand):
        omega_ref, p_wf, p_prec = operand
        v = jnp.cbrt(omega_ref)
        vangles, p_prec = imr_phenom_x_return_phi_zeta_costheta_l_msa(v, p_wf, p_prec)

        alpha_offset = vangles[0] - p_prec.alpha0
        epsilon_offset = vangles[1] - p_prec.epsilon0
        return alpha_offset, epsilon_offset, p_prec

    def other_branch(operand):
        omega_ref, _, p_prec = operand
        logomega_ref = jnp.log(omega_ref)
        omega_ref_cbrt = jnp.cbrt(omega_ref)
        omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

        alpha_offset = (
            p_prec.alpha1 / omega_ref
            + p_prec.alpha2 / omega_ref_cbrt2
            + p_prec.alpha3 / omega_ref_cbrt
            + p_prec.alpha4_l * logomega_ref
            + p_prec.alpha5 * omega_ref_cbrt
            - p_prec.alpha0
        )

        epsilon_offset = (
            p_prec.epsilon1 / omega_ref
            + p_prec.epsilon2 / omega_ref_cbrt2
            + p_prec.epsilon3 / omega_ref_cbrt
            + p_prec.epsilon4_l * logomega_ref
            + p_prec.epsilon5 * omega_ref_cbrt
            - p_prec.epsilon0
        )
        return alpha_offset, epsilon_offset, p_prec

    return jax.lax.cond(
        jnp.isin(pflag, jnp.array([220, 221, 222, 223, 224])),
        msa_branch,
        other_branch,
        operand=(omega_ref, p_wf, p_prec),
    )


def imr_phenom_x_return_phi_zeta_costheta_l_msa(
    v: float,  # Velocity
    p_wf: IMRPhenomXWaveformDataClass,  # IMRPhenomX waveform struct
    p_prec: IMRPhenomXPrecessionDataClass,  # IMRPhenomX precession struct
) -> tuple[Array, IMRPhenomXPrecessionDataClass]:
    """
    Wrapper to generate \f$\\phi_z\f$, \f$\\zeta\f$ and \f$\\cos \theta_L\f$ at a given frequency
    """

    # /* Change code here to determine PN order passed for L */
    l_norm = p_wf.eta / v
    j_norm = imr_phenom_x_imr_phenom_x_jnorm_msa(l_norm, p_prec)

    # /* Orbital angular momentum at 3PN, coefficients are pre-cached when initializing precession struct */
    def call_xlal_sim_imr_phenom_x_lpn_ansatz(v, l_norm, p_prec):
        return xlal_sim_imr_phenom_x_lpn_ansatz(
            v,
            l_norm,
            p_prec.l0,
            p_prec.l1,
            p_prec.l2,
            p_prec.l3,
            p_prec.l4,
            p_prec.l5,
            p_prec.l6,
            p_prec.l7,
            p_prec.l8,
            p_prec.l8_l,
        )

    l_norm_3_pn = jax.lax.cond(
        jnp.logical_or(p_prec.imr_phenom_x_prec_version == 222, p_prec.imr_phenom_x_prec_version == 223),
        lambda x: imr_phenom_x_imr_phenom_x_l_norm_3pn_of_v(*x),
        lambda x: call_xlal_sim_imr_phenom_x_lpn_ansatz(*x),
        operand=(v, l_norm, p_prec),
    )

    j_norm_3_pn = imr_phenom_x_imr_phenom_x_jnorm_msa(l_norm_3_pn, p_prec)

    # /*
    #     Get roots to S^2 equation :
    #         vroots.x = A1 = S_{3}^2
    #         vroots.y = A2 = S_{-}^2
    #         vroots.z = A3 = S_{+}^2
    # */
    v_roots = imr_phenom_x_return_roots_msa(l_norm, j_norm, p_prec)

    s32 = v_roots[0]
    s_mi2 = v_roots[1]
    s_pl2 = v_roots[2]

    s_pl2_m_s_mi2 = s_pl2 - s_mi2
    s_pl2_p_s_mi2 = s_pl2 + s_mi2
    s_pl = jnp.sqrt(s_pl2)
    s_mi = jnp.sqrt(s_mi2)

    s_norm = imr_phenom_x_return_s_norm_msa(v, p_prec)
    s_norm_2 = s_norm * s_norm

    def zero_vmsa_branch(operand):
        _ = operand
        return jnp.zeros(3)

    def call_imr_phenom_x_return_msa_corrections_msa(operand):
        v, l_norm, j_norm, p_prec = operand
        return imr_phenom_x_return_msa_corrections_msa(v, l_norm, j_norm, p_prec)

    v_msa = jax.lax.cond(
        jnp.abs(s_norm_2 - s_pl2_m_s_mi2) > 1.0e-5,
        call_imr_phenom_x_return_msa_corrections_msa,  # /* Get phiz_0_MSA and zeta_0_MSA */
        zero_vmsa_branch,
        operand=(v, l_norm, j_norm, p_prec),
    )

    phiz_msa = v_msa[0]
    zeta_msa = v_msa[1]

    phiz = imr_phenom_x_return_phiz_msa(v, j_norm, p_prec)
    zeta = imr_phenom_x_return_zeta_msa(v, p_prec)
    cos_theta_l = imr_phenom_x_costheta_lj(l_norm_3_pn, j_norm_3_pn, s_norm)

    vout_x = phiz + phiz_msa
    vout_y = zeta + zeta_msa
    vout_z = cos_theta_l

    p_prec = dataclasses.replace(
        p_prec,
        s32=s32,
        s_mi2=s_mi2,
        s_pl2=s_pl2,
        s_pl2_m_s_mi2=s_pl2_m_s_mi2,
        s_pl2_p_s_mi2=s_pl2_p_s_mi2,
        s_pl=s_pl,
        s_mi=s_mi,
        s_norm=s_norm,
        s_norm_2=s_norm_2,
    )

    return jnp.array([vout_x, vout_y, vout_z]), p_prec


def imr_phenom_x_imr_phenom_x_jnorm_msa(l_norm: float, p_prec: IMRPhenomXPrecessionDataClass) -> float | Array:
    """
    Get norm of J using Eq 41 of Chatziioannou et al, PRD 95, 104004, (2017)
    """
    j_norm2 = l_norm * l_norm + (2.0 * l_norm * p_prec.c1_over_eta) + p_prec.s_av2
    return jnp.sqrt(j_norm2)


def imr_phenom_x_imr_phenom_x_l_norm_3pn_of_v(
    v: float, l_norm: float, p_prec: IMRPhenomXPrecessionDataClass
) -> float | Array:
    """
    Returns the 3PN accurate orbital angular momentum as implemented in LALSimInspiralFDPrecAngles_internals.c
    """
    v2 = v * v
    return l_norm * (
        1.0
        + v2
        * (
            p_prec.constants_l[0]
            + v * p_prec.constants_l[1]
            + v2 * (p_prec.constants_l[2] + v * p_prec.constants_l[3] + v2 * (p_prec.constants_l[4]))
        )
    )


def imr_phenom_x_costheta_lj(l_norm: float, j_norm: float, s_norm: float) -> float | Array:
    """
    Calculate (L dot J)
    """

    costheta_lj = 0.5 * (j_norm * j_norm + l_norm * l_norm - s_norm * s_norm) / (l_norm * j_norm)

    costheta_lj = jax.lax.clamp(-1.0, costheta_lj, 1.0)

    return costheta_lj


def imr_phenom_xp_check_max_opening_angle(
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    lal_params: IMRPhenomXPHMParameterDataClass,
) -> tuple[IMRPhenomXPHMParameterDataClass, IMRPhenomXPrecessionDataClass]:
    """
    Helper function to check if maximum opening angle > pi/2 or pi/4 and issues a warning. See discussion in https://dcc.ligo.org/LIGO-T1500602
    """

    eta = p_wf.eta

    # /* For now, use the 2PN non-spinning maximum opening angle */
    v_at_max_beta = jnp.sqrt(2.0 / 3.0) * jnp.sqrt(
        (-9.0 - eta + jnp.sqrt(1539.0 - 1008.0 * eta + 19.0 * eta * eta)) / (81 - 57 * eta + eta * eta)
    )

    c_betah, _ = imr_phenom_x_wignerd_coefficients(v_at_max_beta, p_wf, p_prec)

    l_min = xlal_sim_imr_phenom_x_l2pnns(v_at_max_beta, eta)
    max_beta = 2.0 * jnp.acos(c_betah)

    # /*
    #     If L + SL becomes < 0, WignerdCoefficients does not track the angle between J and L.
    #     The model may become pathological as one moves away from the aligned spin limit.

    #     If this does not happen, then max_beta is the actual maximum opening angle as predicted by the model.
    # */

    pathological_condition = jnp.logical_and((l_min + p_prec.s_l) < 0.0, p_prec.chi_p > 0.0)

    def _warn_pathological():
        print("Warning: The maximum opening angle exceeds Pi/2.\nThe model may be pathological in this regime.")

    def _warn_multiband():
        print("Warning: Multibanding may lead to pathological behaviour in this case. Disabling multibanding.")

    def _warn_max_beta(max_beta_val):
        print(
            f"Warning: The maximum opening angle {max_beta_val} is larger than Pi/4.\nThe model has not been tested against NR in this regime."
        )

    def handle_pathological(operand):
        lal_params, p_prec = operand
        jax.debug.callback(_warn_pathological)

        def disable_multiband(operand):
            lal_params, p_prec = operand
            jax.debug.callback(_warn_multiband)
            new_lal_params = dataclasses.replace(lal_params, threshold_mband=0.0)
            new_p_prec = dataclasses.replace(p_prec, imr_phenom_x_prec_version=0)
            return new_lal_params, new_p_prec

        new_lal_params, new_p_prec = jax.lax.cond(
            jnp.logical_and(p_wf.q > 7.0, lal_params.threshold_mband == 1.0),
            disable_multiband,
            lambda x: x,
            operand=(lal_params, p_prec),
        )

        return new_lal_params, new_p_prec

    def handle_else(operand):
        lal_params, p_prec = operand

        def warn_max_beta(_):
            jax.debug.callback(_warn_max_beta, max_beta)

        jax.lax.cond(
            max_beta > jnp.pi / 4.0,
            warn_max_beta,
            lambda _: None,
            operand=None,
        )
        return lal_params, p_prec

    lal_params, p_prec = jax.lax.cond(
        pathological_condition,
        handle_pathological,
        handle_else,
        operand=(lal_params, p_prec),
    )

    return lal_params, p_prec


def imr_phenom_x_wignerd_coefficients(
    v: float,  # Cubic root of (Pi * Frequency (geometric))
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
) -> tuple[Array, Array]:
    """
    Docstring for imr_phenom_x_wignerd_coefficients

    :param v: Description
    :type v: float
    :param p_wf: Description
    :type p_wf: IMRPhenomXWaveformDataClass
    :param p_prec: Description
    :type p_prec: IMRPhenomXPrecessionDataClass
    :return: Description
    :rtype: tuple[Array, Array]
    """

    # /* Orbital angular momentum */
    big_l = xlal_sim_imr_phenom_x_lpn_ansatz(
        v,
        p_wf.eta / v,
        p_prec.l0,
        p_prec.l1,
        p_prec.l2,
        p_prec.l3,
        p_prec.l4,
        p_prec.l5,
        p_prec.l6,
        p_prec.l7,
        p_prec.l8,
        p_prec.l8_l,
    )

    # /*
    #     We ignore the sign of L + SL below:
    #     s := Sp / (L + SL)
    # */
    s = p_prec.s_perp / (big_l + p_prec.s_l)
    s2 = s * s
    cos_beta = jnp.copysign(1.0, big_l + p_prec.s_l) / jnp.sqrt(1.0 + s2)

    cos_beta_half = jnp.sqrt(jnp.fabs(1.0 + cos_beta) / 2.0)  # cos(beta/2)
    sin_beta_half = jnp.sqrt(jnp.fabs(1.0 - cos_beta) / 2.0)  # sin(beta/2)

    return cos_beta_half, sin_beta_half


def xlal_sim_imr_phenom_x_l2pnns(v: float, eta: float) -> float | Array:
    """
    2PN non-spinning orbital angular momentum as a function of x = v^2 = (Pi M f)^{2/3}

    - Bohe et al, 1212.5520v2, Eq 4.7
    """
    eta2 = eta * eta
    x = v * v
    x2 = x * x
    sqx = v

    return (eta / sqx) * (1.0 + x * (3 / 2.0 + eta / 6.0) + x2 * (27 / 8.0 - (19 * eta) / 8.0 + eta2 / 24.0))
