from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.constants import PI, gt
from ripplegw.gsl_ellint import ellipfinc
from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.lal_constants import LAL_GAMMA
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession_dataclass import (
    IMRPhenomXPrecessionDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral import (
    xlal_sim_inspiral_chirp_time_bound,
)


def get_delta_f_from_wfstruct(p_wf: IMRPhenomXWaveformDataClass) -> float:
    seglen = xlal_sim_inspiral_chirp_time_bound(p_wf.f_ref, p_wf.m1_si, p_wf.m2_si, p_wf.chi1l, p_wf.chi2l)
    delta_fv1 = 1.0 / jnp.max(4.0, jnp.ceil(jnp.log(seglen) / jnp.log(2)) ** 2)
    delta_f = jnp.min(delta_fv1, 0.1)
    delta_mf = delta_f * p_wf.m_tot * gt
    return delta_mf


#####################################################################################
######################################## MSA ########################################
#####################################################################################


# /**
#  * Internal function to computes the PN spin-orbit couplings. As in LALSimInspiralFDPrecAngles.c
#  * cf https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralFDPrecAngles_internals.c#L798
#  */
def imr_phenom_x_get_pn_beta(a: float, b: float, p_prec: IMRPhenomXPrecessionDataClass) -> float:
    return p_prec.dot_s1_l * (a + b * p_prec.qq) + p_prec.dot_s2_l * (a + b / p_prec.qq)


# /**
#  * Internal function to compute PN spin-spin couplings. As in LALSimInspiralFDPrecAngles.c
#  * cf https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralFDPrecAngles_internals.c#L806
#  */
def imr_phenom_x_get_pn_sigma(a: float, b: float, p_prec: IMRPhenomXPrecessionDataClass) -> float:
    return p_prec.inveta * (a * p_prec.dot_s1_s2 - b * p_prec.dot_s1_l * p_prec.dot_s2_l)


# /**
#  * Internal function to computes PN spin-spin couplings. As in LALSimInspiralFDPrecAngles.c
#  */
def imr_phenom_x_get_pn_tau(a: float, b: float, p_prec: IMRPhenomXPrecessionDataClass) -> float:
    return (
        p_prec.qq * ((p_prec.s1_norm_2 * a) - b * p_prec.dot_s1_l * p_prec.dot_s1_l)
        + (a * p_prec.s2_norm_2 - b * p_prec.dot_s2_l * p_prec.dot_s2_l) / p_prec.qq
    ) / p_prec.eta


@checkify.checkify
def imr_phenom_x_initialize_msa_system(
    p_wf: IMRPhenomXWaveformDataClass,
    p_prec: IMRPhenomXPrecessionDataClass,
    expansion_order: int,
) -> IMRPhenomXPrecessionDataClass:
    # /*
    # Sanity check on the precession version
    # */
    pflag = p_prec.imr_phenom_x_prec_version
    # if(pflag != 220 && pflag != 221 && pflag != 222 && pflag != 223 && pflag != 224)
    # {
    # XLAL_ERROR(XLAL_EINVAL,"Error: MSA system requires IMRPhenomXPrecVersion 220, 221, 222, 223 or 224.\n")
    # }
    is_valid_version = jnp.any(
        jnp.array(
            [
                pflag == 220,
                pflag == 221,
                pflag == 222,
                pflag == 223,
                pflag == 224,
            ]
        )
    )
    checkify.check(
        is_valid_version,
        "Error: MSA system requires IMRPhenomXPrecVersion 220, 221, 222, 223 or 224.",
    )

    # /*
    # First initialize the system of variables needed for Chatziioannou et al, PRD, 88, 063011, (2013), arXiv:1307.4418:
    # - Racine et al, PRD, 80, 044010, (2009), arXiv:0812.4413
    # - Favata, PRD, 80, 024002, (2009), arXiv:0812.0069
    # - Blanchet et al, PRD, 84, 064041, (2011), arXiv:1104.5659
    # - Bohe et al, CQG, 30, 135009, (2013), arXiv:1303.7412
    # */
    eta = p_prec.eta
    eta2 = p_prec.eta2
    eta3 = p_prec.eta3
    eta4 = p_prec.eta4

    m1 = p_wf.m1
    m2 = p_wf.m2

    # /* PN Coefficients for d \omega / d t as per LALSimInspiralFDPrecAngles_internals.c */
    domegadt_constants_ns = jnp.array(
        [
            96.0 / 5.0,
            -1486.0 / 35.0,
            -264.0 / 5.0,
            384.0 * PI / 5.0,
            34103.0 / 945.0,
            13661.0 / 105.0,
            944.0 / 15.0,
            PI * (-4159.0 / 35.0),
            PI * (-2268.0 / 5.0),
            (
                16447322263.0 / 7276500.0
                + PI**2 * 512.0 / 5.0
                - jnp.log(2.0) * 109568.0 / 175.0
                - jnp.euler_gamma * 54784.0 / 175.0
            ),
            (-56198689.0 / 11340.0 + PI**2 * 902.0 / 5.0),
            1623.0 / 140.0,
            -1121.0 / 27.0,
            -54784.0 / 525.0,
            -PI * 883.0 / 42.0,
            PI * 71735.0 / 63.0,
            PI * 73196.0 / 63.0,
        ]
    )

    domegadt_constants_so = jnp.array(
        [
            -904.0 / 5.0,
            -120.0,
            -62638.0 / 105.0,
            4636.0 / 5.0,
            -6472.0 / 35.0,
            3372.0 / 5.0,
            -PI * 720.0,
            -PI * 2416.0 / 5.0,
            -208520.0 / 63.0,
            796069.0 / 105.0,
            -100019.0 / 45.0,
            -1195759.0 / 945.0,
            514046.0 / 105.0,
            -8709.0 / 5.0,
            -PI * 307708.0 / 105.0,
            PI * 44011.0 / 7.0,
            -PI * 7992.0 / 7.0,
            PI * 151449.0 / 35.0,
        ]
    )

    domegadt_constants_ss = jnp.array([-494.0 / 5.0, -1442.0 / 5.0, -233.0 / 5.0, -719.0 / 5.0])

    l_csts_nonspin = jnp.array(
        [
            3.0 / 2.0,
            1.0 / 6.0,
            27.0 / 8.0,
            -19.0 / 8.0,
            1.0 / 24.0,
            135.0 / 16.0,
            -6889.0 / 144.0 + 41.0 / 24.0 * PI**2,
            31.0 / 24.0,
            7.0 / 1296.0,
        ]
    )

    l_csts_spinorbit = jnp.array([-14.0 / 6.0, -3.0 / 2.0, -11.0 / 2.0, 133.0 / 72.0, -33.0 / 8.0, 7.0 / 4.0])

    # /*
    # Note that Chatziioannou et al use q = m2/m1, where m1 > m2 and therefore q < 1
    # IMRPhenomX assumes m1 > m2 and q > 1. For the internal MSA code, flip q and
    # dump this to p_prec->qq, where qq explicitly dentoes that this is 0 < q < 1.
    # */
    q = m2 / m1  # // m2 / m1, q < 1, m1 > m2
    invq = 1.0 / q  # // m2 / m1, q < 1, m1 > m2

    mu = (m1 * m2) / (m1 + m2)

    # #if DEBUG == 1
    # printf("m1                = %.6f\n\n",p_wf->m1)
    # printf("m2                = %.6f\n\n",p_wf->m2)
    # printf("q (<1)            = %.6f\n\n",p_prec->qq)
    # #endif

    # /* \delta and powers of \delta in terms of q < 1, should just be m1 - m2 */
    delta_qq = (1.0 - q) / (1.0 + q)
    delta2_qq = delta_qq * delta_qq
    delta3_qq = delta_qq * delta2_qq
    delta4_qq = delta_qq * delta3_qq

    # Define source frame such that \hat{L} = {0,0,1} with L_z pointing along \hat{z}.
    l_hat = jnp.array([0.0, 0.0, 1.0])

    # Set LHat variables - these are fixed.
    l_hat_cos_theta = 1.0  # Cosine of Polar angle of orbital angular momentum
    l_hat_phi = 0.0  # Azimuthal angle of orbital angular momentum
    l_hat_theta = 0.0  # Polar angle of orbital angular momentum

    # /* Dimensionful spin vectors, note eta = m1 * m2 and q = m2/m1  */
    s1v = jnp.array([p_prec.chi1x * eta / q, p_prec.chi1y * eta / q, p_prec.chi1z * eta / q])  # eta / q = m1^2
    s2v = jnp.array([p_prec.chi2x * eta * q, p_prec.chi2y * eta * q, p_prec.chi2z * eta * q])  # eta / q = m1^2

    s1_0_norm = jnp.linalg.norm(s1v)
    s2_0_norm = jnp.linalg.norm(s2v)
    # /* Initial dimensionful spin vectors at reference frequency */
    # /* S1 = {S1x,S1y,S1z} */
    # p_prec->S1_0.x = S1v.x
    # p_prec->S1_0.y = S1v.y
    # p_prec->S1_0.z = S1v.z
    # s1_0 = s1v

    # /* S2 = {S2x,S2y,S2z} */
    # p_prec->S2_0.x = S2v.x
    # p_prec->S2_0.y = S2v.y
    # p_prec->S2_0.z = S2v.z
    # s2_0 = s2v

    # /* Reference velocity v and v^2 */
    v_0 = (p_prec.pi_gm * p_wf.f_ref) ** (1 / 3)
    v_0_2 = v_0 * v_0

    # /* Reference orbital angular momenta */
    l_0 = l_hat * p_prec.eta / v_0

    # #if DEBUG == 1
    # printf("v_0                = %.6f\n\n",p_prec.v_0)

    # printf("chi1x              = %.6f\n",p_prec->chi1x)
    # printf("chi1y              = %.6f\n",p_prec->chi1y)
    # printf("chi1z              = %.6f\n\n",p_prec->chi1z)

    # printf("chi2x              = %.6f\n",p_prec->chi2x)
    # printf("chi2y              = %.6f\n",p_prec->chi2y)
    # printf("chi2z              = %.6f\n\n",p_prec->chi2z)

    # printf("S1_0.x             = %.6f\n",p_prec->S1_0.x)
    # printf("S1_0.y             = %.6f\n",p_prec->S1_0.y)
    # printf("S1_0.z             = %.6f\n",p_prec->S1_0.z)
    # printf("S1_0               = %.6f\n\n",S1_0_norm)

    # printf("S2_0.x             = %.6f\n",p_prec->S2_0.x)
    # printf("S2_0.y             = %.6f\n",p_prec->S2_0.y)
    # printf("S2_0.z             = %.6f\n",p_prec->S2_0.z)
    # printf("S2_0               = %.6f\n\n",S2_0_norm)
    # #endif

    # /* Inner products used in MSA system */
    dot_s1_l = jnp.inner(s1v, l_hat)
    dot_s2_l = jnp.inner(s2v, l_hat)
    dot_s1_s2 = jnp.inner(s1v, s2v)
    dot_s1_ln = dot_s1_l / s1_0_norm
    dot_s2_ln = dot_s2_l / s2_0_norm

    # #if DEBUG == 1
    # printf("Lhat_0.x               = %.6f\n",Lhat.x)
    # printf("Lhat_0.y               = %.6f\n",Lhat.y)
    # printf("Lhat_0.z               = %.6f\n\n",Lhat.z)

    # printf("dotS1L                 = %.6f\n",p_prec->dotS1L)
    # printf("dotS2L                 = %.6f\n",p_prec->dotS2L)
    # printf("dotS1Ln                = %.6f\n",p_prec->dotS1Ln)
    # printf("dotS2Ln                = %.6f\n",p_prec->dotS2Ln)
    # printf("dotS1S2                = %.6f\n\n",p_prec->dotS1S2)
    # #endif

    # /* Coeffcients for PN orbital angular momentum at 3PN, as per LALSimInspiralFDPrecAngles_internals.c */
    # p_prec->constants_L[0] = (L_csts_nonspin[0] + eta*L_csts_nonspin[1]);
    # p_prec->constants_L[1] = IMRPhenomX_Get_PN_beta(L_csts_spinorbit[0], L_csts_spinorbit[1], p_prec);
    # p_prec->constants_L[2] = (L_csts_nonspin[2] + eta*L_csts_nonspin[3] + eta*eta*L_csts_nonspin[4]);
    # p_prec->constants_L[3] = IMRPhenomX_Get_PN_beta((L_csts_spinorbit[2]+L_csts_spinorbit[3]*eta), (L_csts_spinorbit[4]+L_csts_spinorbit[5]*eta), p_prec);
    # p_prec->constants_L[4] = (L_csts_nonspin[5]+L_csts_nonspin[6]*eta +L_csts_nonspin[7]*eta*eta+L_csts_nonspin[8]*eta*eta*eta);

    constants_l = jnp.array(
        [
            l_csts_nonspin[0] + eta * l_csts_nonspin[1],
            imr_phenom_x_get_pn_beta(l_csts_spinorbit[0], l_csts_spinorbit[1], p_prec),
            l_csts_nonspin[2] + eta * l_csts_nonspin[3] + eta * eta * l_csts_nonspin[4],
            imr_phenom_x_get_pn_beta(
                (l_csts_spinorbit[2] + l_csts_spinorbit[3] * eta),
                (l_csts_spinorbit[4] + l_csts_spinorbit[5] * eta),
                p_prec,
            ),
            l_csts_nonspin[5]
            + l_csts_nonspin[6] * eta
            + l_csts_nonspin[7] * eta * eta
            + l_csts_nonspin[8] * eta * eta * eta,
        ]
    )

    # /* Effective total spin */
    s_eff = (1.0 + q) * dot_s1_l + (1 + (1.0 / q)) * dot_s2_l
    s_eff2 = s_eff * s_eff

    # #if DEBUG == 1
    # printf("Seff             = %.6f\n\n",p_prec->Seff)
    # #endif

    # /* Initial total spin, S = S1 + S2 */
    # s0 = jnp.zeros(3)  # {0.,0.,0.}
    s0 = s1v + s2v

    # /* Cache total spin in the precession struct */
    s_0 = s0

    # #if DEBUG == 1
    # printf("S_0_x             = %.6f\n",p_prec->S_0.x)
    # printf("S_0_y             = %.6f\n",p_prec->S_0.y)
    # printf("S_0_z             = %.6f\n\n",p_prec->S_0.z)
    # #endif

    # /* Initial total angular momentum, J = L + S1 + S2 */
    j_0 = l_0 + s_0

    # #if DEBUG == 1
    # printf("J_0_x             = %.6f\n",p_prec->J_0.x)
    # printf("J_0_y             = %.6f\n",p_prec->J_0.y)
    # printf("J_0_z             = %.6f\n\n",p_prec->J_0.z)
    # #endif

    # /* Norm of total initial spin */
    s_0_norm = jnp.linalg.norm(s_0)
    s_0_norm_2 = s_0_norm * s_0_norm

    # /* Norm of orbital and total angular momenta */
    l_0_norm = jnp.linalg.norm(l_0)
    j_0_norm = jnp.linalg.norm(j_0)

    # l0norm = l_0_norm
    # j0norm = j_0_norm

    # #if DEBUG == 1
    # printf("L_0_norm             = %.6f\n",p_prec->L_0_norm)
    # printf("J_0_norm             = %.6f\n\n",p_prec->J_0_norm)
    # #endif

    # /* Useful powers */
    s_0_norm_2 = s_0_norm * s_0_norm
    j_0_norm_2 = j_0_norm * j_0_norm
    l_0_norm_2 = l_0_norm * l_0_norm

    # /* Vector for obtaining B, C, D coefficients */
    # v_bcd = imr_phenom_x_return_spin_evolution_coefficients_msa(l_0_norm, j_0_norm, p_prec)

    # #if DEBUG == 1
    # printf("B             = %.6f\n",vBCD.x)
    # printf("C             = %.6f\n",vBCD.y)
    # printf("D             = %.6f\n\n",vBCD.z)
    # #endif

    # /*
    # Get roots to S^2 equation : S^2_+, S^2_-, S^2_3
    #     vroots.x = A1 = S_{3}^2
    #     vroots.y = A2 = S_{-}^2
    #     vroots.z = A3 = S_{+}^2
    # */
    # vRoots = jnp.zeros(3) #{0.,0.,0.}

    # Update p_prec with everything we computed so far
    p_prec = dataclasses.replace(
        p_prec,
        qq=q,
        inv_qq=invq,
        delta_qq=delta_qq,
        delta2_qq=delta2_qq,
        delta3_qq=delta3_qq,
        delta4_qq=delta4_qq,
        l_hat_cos_theta=l_hat_cos_theta,
        l_hat_phi=l_hat_phi,
        l_hat_theta=l_hat_theta,
        s1x=s1v[0],
        s1y=s1v[1],
        s1z=s1v[2],
        s2x=s2v[0],
        s2y=s2v[1],
        s2z=s2v[2],
        # S1_0=S1_0,
        # S2_0=S2_0,
        v_0=v_0,
        v_0_2=v_0_2,
        # l_0=l_0,
        dot_s1_l=dot_s1_l,
        dot_s2_l=dot_s2_l,
        dot_s1_s2=dot_s1_s2,
        dot_s1_ln=dot_s1_ln,
        dot_s2_ln=dot_s2_ln,
        constants_l=constants_l,
        s_eff=s_eff,
        s_eff2=s_eff2,
        # S_0=S_0,
        # J_0=J_0,
        s_0_norm=s_0_norm,
        s_0_norm_2=s_0_norm_2,
        l_0_norm=l_0_norm,
        l_0_norm_2=l_0_norm_2,
        j_0_norm=j_0_norm,
        j_0_norm_2=j_0_norm_2,
    )

    v_roots = imr_phenom_x_return_roots_msa(p_prec.l_0_norm, p_prec.j_0_norm, p_prec)

    # // Set roots
    s_pl2 = v_roots[2]
    s_mi2 = v_roots[1]
    s32 = v_roots[0]

    # S^2_+ + S^2_-
    s_pl2_p_s_mi2 = s_pl2 + s_mi2

    # // S^2_+ - S^2_-
    s_pl2_m_s_mi2 = s_pl2 - s_mi2

    # // S_+ and S_-
    s_pl = jnp.sqrt(s_pl2)
    s_mi = jnp.sqrt(s_mi2)
    # /* Eq. 45 of PRD 95, 104004, (2017), arXiv:1703.03967, set from initial conditions */
    s_av2 = 0.5 * (s_pl2_p_s_mi2)
    s_av = jnp.sqrt(s_av2)
    inv_s_av2 = 1.0 / s_av2
    inv_s_av = 1.0 / s_av

    # #if DEBUG == 1
    # printf("From vRoots... \n")
    # printf("Spl2             = %.6f\n",p_prec->Spl2)
    # printf("Smi2             = %.6f\n",p_prec->Smi2)
    # printf("S32              = %.6f\n",p_prec->S32)
    # printf("SAv2             = %.6f\n",p_prec->SAv2)
    # printf("SAv              = %.6f\n\n",p_prec->SAv)
    # #endif

    # /* c_1 is determined by Eq. 41 of PRD, 95, 104004, (2017), arXiv:1703.03967 */
    c_1 = 0.5 * (j_0_norm * j_0_norm - l_0_norm * l_0_norm - s_av2) / l_0_norm * eta
    c1_2 = c_1 * c_1

    # /* Useful powers and combinations of c_1 */
    c1_over_eta = c_1 / eta

    # /* Average spin couplings over one precession cycle: A9 - A14 of arXiv:1703.03967  */
    omqsq = (1.0 - q) * (1.0 - q) + 1e-16
    omq2 = (1.0 - q * q) + 1e-16

    # /* Precession averaged spin couplings, Eq. A9 - A14 of arXiv:1703.03967, note that we only use the initial values  */
    s1_l_pav = (c_1 * (1.0 + q) - q * eta * s_eff) / (eta * omq2)
    s2_l_pav = -q * (c_1 * (1.0 + q) - eta * s_eff) / (eta * omq2)
    s1_s2_pav = 0.5 * s_av2 - 0.5 * (p_prec.s1_norm_2 + p_prec.s2_norm_2)
    s1_l_sq_pav = (s1_l_pav * s1_l_pav) + ((s_pl2_m_s_mi2) * (s_pl2_m_s_mi2) * v_0_2) / (32.0 * eta2 * omqsq)
    s2_l_sq_pav = (s2_l_pav * s2_l_pav) + (q * q * (s_pl2_m_s_mi2) * (s_pl2_m_s_mi2) * v_0_2) / (32.0 * eta2 * omqsq)
    s1_l_s2_l_pav = s1_l_pav * s2_l_pav - q * (s_pl2_m_s_mi2) * (s_pl2_m_s_mi2) * v_0_2 / (32.0 * eta2 * omqsq)
    # Spin couplings in arXiv:1703.03967
    beta3 = ((113.0 / 12.0) + (25.0 / 4.0) * (m2 / m1)) * s1_l_pav + (
        (113.0 / 12.0) + (25.0 / 4.0) * (m1 / m2)
    ) * s2_l_pav
    beta5 = (
        ((31319.0 / 1008.0) - (1159.0 / 24.0) * eta) + (m2 / m1) * ((809.0 / 84) - (281.0 / 8.0) * eta)
    ) * s1_l_pav + (
        ((31319.0 / 1008.0) - (1159.0 / 24.0) * eta) + (m1 / m2) * ((809.0 / 84) - (281.0 / 8.0) * eta)
    ) * s2_l_pav

    beta6 = PI * (
        ((75.0 / 2.0) + (151.0 / 6.0) * (m2 / m1)) * s1_l_pav + ((75.0 / 2.0) + (151.0 / 6.0) * (m1 / m2)) * s2_l_pav
    )

    beta7 = (
        ((130325.0 / 756) - (796069.0 / 2016) * eta + (100019.0 / 864.0) * eta2)
        + (m2 / m1) * ((1195759.0 / 18144) - (257023.0 / 1008.0) * eta + (2903 / 32.0) * eta2) * s1_l_pav
        + ((130325.0 / 756) - (796069.0 / 2016) * eta + (100019.0 / 864.0) * eta2)
        + (m1 / m2) * ((1195759.0 / 18144) - (257023.0 / 1008.0) * eta + (2903 / 32.0) * eta2) * s2_l_pav
    )

    sigma4 = (
        (1.0 / mu) * ((247.0 / 48.0) * s1_s2_pav - (721.0 / 48.0) * s1_l_pav * s2_l_pav)
        + (1.0 / (m1 * m1)) * ((233.0 / 96.0) * p_prec.s1_norm_2 - (719.0 / 96.0) * s1_l_sq_pav)
        + (1.0 / (m2 * m2)) * ((233.0 / 96.0) * p_prec.s2_norm_2 - (719.0 / 96.0) * s2_l_sq_pav)
    )

    # /* Compute PN coefficients using precession-averaged spin couplings */
    a0 = 96.0 * eta / 5.0

    # /* These are all normalized by a factor of a0 */
    a2 = -(743.0 / 336.0) - (11.0 / 4.0) * eta
    a3 = 4.0 * PI - beta3
    a4 = (34103.0 / 18144.0) + (13661.0 / 2016.0) * eta + (59.0 / 18.0) * eta2 - sigma4
    a5 = -(4159.0 / 672.0) * PI - (189.0 / 8.0) * PI * eta - beta5
    a6 = (
        (16447322263.0 / 139708800.0)
        + (16.0 / 3.0) * PI * PI
        - (856.0 / 105) * jnp.log(16.0)
        - (1712.0 / 105.0) * LAL_GAMMA
        - beta6
        + eta * ((451.0 / 48) * PI * PI - (56198689.0 / 217728.0))
        + eta2 * (541.0 / 896.0)
        - eta3 * (5605.0 / 2592.0)
    )
    a7 = -(4415.0 / 4032.0) * PI + (358675.0 / 6048.0) * PI * eta + (91495.0 / 1512.0) * PI * eta2 - beta7

    # // Coefficients are weighted by an additional factor of a_0
    a2 *= a0
    a3 *= a0
    a4 *= a0
    a5 *= a0
    a6 *= a0
    a7 *= a0

    # #if DEBUG == 1
    # printf("a0     = %.6f\n",p_prec->a0)
    # printf("a2     = %.6f\n",p_prec->a2)
    # printf("a3     = %.6f\n",p_prec->a3)
    # printf("a4     = %.6f\n",p_prec->a4)
    # printf("a5     = %.6f\n\n",p_prec->a5)
    # #endif

    # /* For versions 222 and 223, we compute PN coefficients using initial spin couplings, as per LALSimInspiralFDPrecAngles_internals.c  */
    def setup_22_branch():
        a0 = eta * domegadt_constants_ns[0]
        a2 = eta * (domegadt_constants_ns[1] + eta * (domegadt_constants_ns[2]))
        a3 = eta * (
            domegadt_constants_ns[3]
            + imr_phenom_x_get_pn_beta(domegadt_constants_so[0], domegadt_constants_so[1], p_prec)
        )
        a4 = eta * (
            domegadt_constants_ns[4]
            + eta * (domegadt_constants_ns[5] + eta * (domegadt_constants_ns[6]))
            + imr_phenom_x_get_pn_sigma(domegadt_constants_ss[0], domegadt_constants_ss[1], p_prec)
            + imr_phenom_x_get_pn_tau(domegadt_constants_ss[2], domegadt_constants_ss[3], p_prec)
        )
        a5 = eta * (
            domegadt_constants_ns[7]
            + eta * (domegadt_constants_ns[8])
            + imr_phenom_x_get_pn_beta(
                (domegadt_constants_so[2] + eta * (domegadt_constants_so[3])),
                (domegadt_constants_so[4] + eta * (domegadt_constants_so[5])),
                p_prec,
            )
        )
        return a0, a2, a3, a4, a5

    # if(pflag == 222 || pflag == 223)
    # {
    # p_prec->a0 = eta*domegadt_constants_NS[0]
    # p_prec->a2 = eta*(domegadt_constants_NS[1] + eta*(domegadt_constants_NS[2]))
    # p_prec->a3 = eta*(domegadt_constants_NS[3] + IMRPhenomX_Get_PN_beta(domegadt_constants_SO[0], domegadt_constants_SO[1], p_prec))
    # p_prec->a4 = eta*(domegadt_constants_NS[4] + eta*(domegadt_constants_NS[5] + eta*(domegadt_constants_NS[6])) + IMRPhenomX_Get_PN_sigma(domegadt_constants_SS[0], domegadt_constants_SS[1], p_prec) + IMRPhenomX_Get_PN_tau(domegadt_constants_SS[2], domegadt_constants_SS[3], p_prec))
    # p_prec->a5 = eta*(domegadt_constants_NS[7] + eta*(domegadt_constants_NS[8]) + IMRPhenomX_Get_PN_beta((domegadt_constants_SO[2] + eta*(domegadt_constants_SO[3])), (domegadt_constants_SO[4] + eta*(domegadt_constants_SO[5])), p_prec))
    # }
    a0, a2, a3, a4, a5 = jax.lax.cond(
        jnp.logical_or(pflag == 222, pflag == 223),
        setup_22_branch,
        lambda: (a0, a2, a3, a4, a5),
    )

    # /* Debugging */
    # #if DEBUG == 1
    # printf("Using list of coefficients... \n")
    # printf("a0     = %.6f\n",p_prec->a0)
    # printf("a2     = %.6f\n",p_prec->a2)
    # printf("a3     = %.6f\n",p_prec->a3)
    # printf("a4     = %.6f\n",p_prec->a4)
    # printf("a5     = %.6f\n\n",p_prec->a5)
    # #endif

    # /* Useful powers of a_0 */
    a0_2 = a0 * a0
    a0_3 = a0_2 * a0
    a2_2 = a2 * a2

    # /*
    # Calculate g coefficients as in Appendix A of Chatziioannou et al, PRD, 95, 104004, (2017), arXiv:1703.03967.
    # These constants are used in TaylorT2 where domega/dt is expressed as an inverse polynomial
    # */
    g0 = 1.0 / a0

    # // Eq. A2 (1703.03967)
    g2 = -(a2 / a0_2)

    # // Eq. A3 (1703.03967)
    g3 = -(a3 / a0_2)

    # // Eq.A4 (1703.03967)
    g4 = -(a4 * a0 - a2_2) / a0_3

    # // Eq. A5 (1703.03967)
    g5 = -(a5 * a0 - 2.0 * a3 * a2) / a0_3

    # #if DEBUG == 1
    # printf("g0     = %.6f\n",p_prec->g0)
    # printf("g2     = %.6f\n",p_prec->g2)
    # printf("g3     = %.6f\n",p_prec->g3)
    # printf("g4     = %.6f\n",p_prec->g4)
    # printf("g5     = %.6f\n\n",p_prec->g5)
    # #endif

    # // Useful powers of delta
    delta = delta_qq
    delta2 = delta * delta
    delta3 = delta * delta2
    delta4 = delta * delta3

    # // These are the phase coefficients of Eq. 51 of PRD, 95, 104004, (2017), arXiv:1703.03967
    psi0 = 0.0
    psi1 = 0.0
    psi2 = 0.0

    # /* \psi_1 is defined in Eq. C1 of Appendix C in PRD, 95, 104004, (2017), arXiv:1703.03967  */
    psi1 = 3.0 * (2.0 * eta2 * s_eff - c_1) / (eta * delta2)

    c_1_over_nu = c1_over_eta
    c_1_over_nu_2 = c_1_over_nu * c_1_over_nu
    one_p_q_sq = (1.0 + q) * (1.0 + q)
    s_eff_2 = s_eff * s_eff
    q_2 = q * q
    one_m_q_sq = (1.0 - q) * (1.0 - q)
    one_m_q2_2 = (1.0 - q_2) * (1.0 - q_2)
    one_m_q_4 = one_m_q_sq * one_m_q_sq

    # /*  This implements the Delta term as in LALSimInspiralFDPrecAngles.c
    # c.f. https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralFDPrecAngles_internals.c#L145
    # */
    # if(pflag == 222 || pflag == 223)
    # {
    # Del1 = 4. * c_1_over_nu_2 * one_p_q_sq
    # Del2 = 8. * c_1_over_nu * q * (1. + q) * Seff
    # Del3 = 4. * (one_m_q2_2 * p_prec->S1_norm_2 - q_2 * Seff_2)
    # Del4 = 4. * c_1_over_nu_2 * q_2 * one_p_q_sq
    # Del5 = 8. * c_1_over_nu * q_2 * (1. + q) * Seff
    # Del6 = 4. * (one_m_q2_2 * p_prec->S2_norm_2 - q_2 * Seff_2)
    # p_prec->Delta      = sqrt( fabs( (Del1 - Del2 - Del3) * (Del4 - Del5 - Del6) ))
    # }
    # else
    # {
    # /* Coefficients of \Delta as defined in Eq. C3 of Appendix C in PRD, 95, 104004, (2017), arXiv:1703.03967. */
    # term1  = c1_2 * eta / (q * delta4)
    # term2  = -2.0 * c_1 * eta3 * (1.0 + q) * Seff / (q * delta4)
    # term3  = -eta2 * (delta2 * p_prec->S1_norm_2 - eta2 * Seff2) / delta4
    # /*
    #     Is this 1) (c1_2 * q * eta / delta4) or 2) c1_2*eta2/delta4?

    #     - In paper.pdf, the expression 1) is used.

    #     Using eta^2 leads to higher frequency oscillations, use q * eta
    # */
    # term4  = c1_2 * eta * q / delta4
    # term5  = -2.0*c_1*eta3*(1.0 + q)*Seff / delta4
    # term6  = -eta2 * (delta2*p_prec->S2_norm_2 - eta2*Seff2) / delta4

    # /* \Delta as in Eq. C3 of Appendix C in PRD, 95, 104004, (2017) */
    # p_prec->Delta  = sqrt( fabs( (term1 + term2 + term3) * (term4 + term5 + term6) ) )
    # }

    # /*  This implements the Delta term as in LALSimInspiralFDPrecAngles.c
    # c.f. https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralFDPrecAngles_internals.c#L160
    # */
    # if(pflag == 222 || pflag == 223)
    # {
    # u1 = 3. * p_prec->g2 / p_prec->g0
    # u2 = 0.75 * one_p_q_sq / one_m_q_4
    # u3 = -20. * c_1_over_nu_2 * q_2 * one_p_q_sq
    # u4 = 2. * one_m_q2_2 * (q * (2. + q) * p_prec->S1_norm_2 + (1. + 2. * q) * p_prec->S2_norm_2 - 2. * q * p_prec->SAv2)
    # u5 = 2. * q_2 * (7. + 6. * q + 7. * q_2) * 2. * c_1_over_nu * Seff
    # u6 = 2. * q_2 * (3. + 4. * q + 3. * q_2) * Seff_2
    # u7 = q * p_prec->Delta

    # /* Eq. C2 (1703.03967) */
    # p_prec->psi2 = u1 + u2*(u3 + u4 + u5 - u6 + u7)
    # }
    # else
    # {
    # /* \psi_2 is defined in Eq. C2 of Appendix C in PRD, 95, 104004, (2017). Here we implement system of equations as in paper.pdf */
    # term1         = 3.0 * p_prec->g2 / p_prec->g0

    # /* q^2 or no q^2 in term2? Consensus on retaining q^2 term: https://git.ligo.org/waveforms/reviews/phenompv3hm/issues/7 */
    # term2         = 3.0 * q * q / (2.0 * eta3)
    # term3         = 2.0 * p_prec->Delta
    # term4         = -2.0*eta2*p_prec->SAv2 / delta2
    # term5         = -10.*eta*c1_2 / delta4
    # term6         = 2.0 * eta2 * (7.0 + 6.0*q + 7.0*q*q) * c_1 * Seff / (omqsq * delta2)
    # term7         = -eta3 * (3.0 + 4.0*q + 3.0*q*q) * Seff2 / (omqsq * delta2)
    # term8         = eta * (q * (2.0+q)*p_prec->S1_norm_2 + (1.0 + 2.0*q)*p_prec->S2_norm_2) / ( omqsq )

    # /* \psi_2, C2 of Appendix C of PRD, 95, 104004, (2017)  */
    # p_prec->psi2  = term1 + term2 * (term3 + term4 + term5 + term6 + term7 + term8)
    # }

    def flag_222_or_223_branch():
        del1 = 4.0 * c_1_over_nu_2 * one_p_q_sq
        del2 = 8.0 * c_1_over_nu * q * (1.0 + q) * s_eff
        del3 = 4.0 * (one_m_q2_2 * p_prec.s1_norm_2 - q_2 * s_eff_2)
        del4 = 4.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
        del5 = 8.0 * c_1_over_nu * q_2 * (1.0 + q) * s_eff
        del6 = 4.0 * (one_m_q2_2 * p_prec.s2_norm_2 - q_2 * s_eff_2)
        delta = jnp.sqrt(jnp.abs((del1 - del2 - del3) * (del4 - del5 - del6)))

        u1 = 3.0 * g2 / g0
        u2 = 0.75 * one_p_q_sq / one_m_q_4
        u3 = -20.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
        u4 = (
            2.0 * one_m_q2_2 * (q * (2.0 + q) * p_prec.s1_norm_2 + (1.0 + 2.0 * q) * p_prec.s2_norm_2 - 2.0 * q * s_av2)
        )
        u5 = 2.0 * q_2 * (7.0 + 6.0 * q + 7.0 * q_2) * 2.0 * c_1_over_nu * s_eff
        u6 = 2.0 * q_2 * (3.0 + 4.0 * q + 3.0 * q_2) * s_eff_2
        u7 = q * delta

        # /* Eq. C2 (1703.03967) */
        psi2 = u1 + u2 * (u3 + u4 + u5 - u6 + u7)
        return psi2, delta

    def else_branch():
        term1 = c1_2 * eta / (q * delta4)
        term2 = -2.0 * c_1 * eta3 * (1.0 + q) * s_eff / (q * delta4)
        term3 = -eta2 * (delta2 * p_prec.s1_norm_2 - eta2 * s_eff_2) / delta4
        term4 = c1_2 * eta * q / delta4
        term5 = -2.0 * c_1 * eta3 * (1.0 + q) * s_eff / delta4
        term6 = -eta2 * (delta2 * p_prec.s2_norm_2 - eta2 * s_eff_2) / delta4
        delta = jnp.sqrt(jnp.abs((term1 + term2 + term3) * (term4 + term5 + term6)))

        # /* \psi_2 is defined in Eq. C2 of Appendix C in PRD, 95, 104004, (2017). Here we implement system of equations as in paper.pdf */
        term1 = 3.0 * g2 / g0

        # /* q^2 or no q^2 in term2? Consensus on retaining q^2 term: https://git.ligo.org/waveforms/reviews/phenompv3hm/issues/7 */
        term2 = 3.0 * q * q / (2.0 * eta3)
        term3 = 2.0 * delta
        term4 = -2.0 * eta2 * s_av2 / delta2
        term5 = -10.0 * eta * c1_2 / delta4
        term6 = 2.0 * eta2 * (7.0 + 6.0 * q + 7.0 * q * q) * c_1 * s_eff / (omqsq * delta2)
        term7 = -eta3 * (3.0 + 4.0 * q + 3.0 * q * q) * s_eff_2 / (omqsq * delta2)
        term8 = eta * (q * (2.0 + q) * p_prec.s1_norm_2 + (1.0 + 2.0 * q) * p_prec.s2_norm_2) / (omqsq)
        # /* \psi_2, C2 of Appendix C of PRD, 95, 104004, (2017)  */
        psi2 = term1 + term2 * (term3 + term4 + term5 + term6 + term7 + term8)
        return psi2, delta

    psi2, delta = jax.lax.cond(
        jnp.logical_or(pflag == 222, pflag == 223),
        flag_222_or_223_branch,
        else_branch,
    )

    # #if DEBUG == 1
    # printf("psi1     = %.6f\n",p_prec->psi1)
    # printf("psi2     = %.6f\n\n",p_prec->psi2)
    # #endif

    # /* Eq. D1 of PRD, 95, 104004, (2017), arXiv:1703.03967  */
    rm = s_pl2 - s_mi2
    rm_2 = rm * rm

    # /* Eq. D2 and D3 Appendix D of PRD, 95, 104004, (2017), arXiv:1703.03967   */
    cp = s_pl2 * eta2 - c1_2
    cm = s_mi2 * eta2 - c1_2

    # /*
    # Check if cm goes negative, this is likely pathological. If so, set MSA_ERROR to 1, so that waveform generator can handle
    # the error approriately
    # */
    # if(cm < 0.0)
    # {
    #   p_prec.MSA_ERROR = 1
    #   XLAL_PRINT_ERROR("Error, coefficient cm = %.16f, which is negative and likely to be pathological. Triggering MSA failure.\n",cm)
    # }

    # /* fabs is here to help enforce positive definite cpcm */
    cpcm = jnp.abs(cp * cm)
    sqrt_cpcm = jnp.sqrt(cpcm)

    # /* Eq. D4 in PRD, 95, 104004, (2017), arXiv:1703.03967  Note difference to published version.  */
    a1d_d = 0.5 + 0.75 / eta

    # /* Eq. D5 in PRD, 95, 104004, (2017), arXiv:1703.03967  */
    a2d_d = -0.75 * s_eff / eta

    # /* Eq. E3 in PRD, 95, 104004, (2017), arXiv:1703.03967  Note that this is Rm * D2   */
    d2rm_sq = (cp - sqrt_cpcm) / eta2

    # /* Eq. E4 in PRD, 95, 104004, (2017), arXiv:1703.03967  Note that this is Rm^2 * D4  */
    d4rm_sq = -0.5 * rm * sqrt_cpcm / eta2 - cp / eta4 * (sqrt_cpcm - cp)

    s0m = p_prec.s1_norm_2 - p_prec.s2_norm_2

    # /* Difference of spin norms squared, as used in Eq. D6 of PRD, 95, 104004, (2017), arXiv:1703.03967  */
    aw = -3.0 * (1.0 + q) / q * (2.0 * (1.0 + q) * eta2 * s_eff * c_1 - (1.0 + q) * c1_2 + (1.0 - q) * eta2 * s0m)
    cw = 3.0 / 32.0 / eta * rm_2
    dw = 4.0 * cp - 4.0 * d2rm_sq * eta2
    hw = -2.0 * (2.0 * d2rm_sq - rm) * c_1
    fw = rm * d2rm_sq - d4rm_sq - 0.25 * rm_2

    ad_d = aw / dw
    hd_d = hw / dw
    cd_d = cw / dw
    fd_d = fw / dw

    gw = 3.0 / 16.0 / eta2 / eta * rm_2 * (c_1 - eta2 * s_eff)
    gd_d = gw / dw

    # /* Useful powers of the coefficients */
    hd_d_2 = hd_d * hd_d
    ad_dfd_d = ad_d * fd_d
    ad_dfd_dhd_d = ad_dfd_d * hd_d
    ad_dhd_d_2 = ad_d * hd_d_2

    # #if DEBUG == 1
    # printf("\na1dD      = %.6f\n",a1dD)
    # printf("a2dD      = %.6f\n",a2dD)
    # printf("adD       = %.6f\n",adD)
    # printf("cdD       = %.6f\n",cdD)
    # printf("hdD       = %.6f\n",hdD)
    # printf("fdD       = %.6f\n",fdD)
    # printf("Rm        = %.6f\n",Rm)
    # printf("Delta     = %.6f\n",p_prec->Delta)
    # printf("sqrt_cpcm = %.6f\n",sqrt_cpcm)
    # printf("c1        = %.6f\n",p_prec->c1)
    # printf("gdD       = %.6f\n\n",gdD)
    # #endif

    # // Eq. D10 in PRD, 95, 104004, (2017), arXiv:1703.03967
    omegaz0 = a1d_d + ad_d

    # // Eq. D11 in PRD, 95, 104004, (2017), arXiv:1703.03967
    omegaz1 = a2d_d - ad_d * s_eff - ad_d * hd_d

    # // Eq. D12 in PRD, 95, 104004, (2017), arXiv:1703.03967
    omegaz2 = ad_d * hd_d * s_eff + cd_d - ad_d * fd_d + ad_d * hd_d_2

    # // Eq. D13 in PRD, 95, 104004, (2017), arXiv:1703.03967
    omegaz3 = (ad_dfd_d - cd_d - ad_dhd_d_2) * (s_eff + hd_d) + ad_dfd_dhd_d

    # // Eq. D14 in PRD, 95, 104004, (2017), arXiv:1703.03967
    omegaz4 = (cd_d + ad_dhd_d_2 - 2.0 * ad_dfd_d) * (hd_d * s_eff + hd_d_2 - fd_d) - ad_d * fd_d * fd_d

    # // Eq. D15 in PRD, 95, 104004, (2017), arXiv:1703.03967
    omegaz5 = (
        (cd_d - ad_dfd_d + ad_dhd_d_2) * fd_d * (s_eff + 2.0 * hd_d)
        - (cd_d + ad_dhd_d_2 - 2.0 * ad_dfd_d) * hd_d_2 * (s_eff + hd_d)
        - ad_dfd_d * fd_d * hd_d
    )

    # #if DEBUG == 1
    # printf("Omegaz0     = %.6f\n",p_prec->Omegaz0)
    # printf("Omegaz1     = %.6f\n",p_prec->Omegaz1)
    # printf("Omegaz2     = %.6f\n",p_prec->Omegaz2)
    # printf("Omegaz3     = %.6f\n",p_prec->Omegaz3)
    # printf("Omegaz4     = %.6f\n",p_prec->Omegaz4)
    # printf("Omegaz5     = %.6f\n\n",p_prec->Omegaz5)
    # #endif

    # /*
    # If Omegaz5 > 1000, this is larger than we expect and the system may be pathological.
    # - Set MSA_ERROR = 1 to trigger an error
    # */
    # if(fabs(p_prec->Omegaz5) > 1000.0)
    # {
    # p_prec->MSA_ERROR = 1
    # XLAL_PRINT_WARNING("Warning, |Omegaz5| = %.16f, which is larger than expected and may be pathological. Triggering MSA failure.\n",p_prec->Omegaz5)
    # }
    checkify.check(
        jnp.abs(omegaz5) < 1000.0,
        "Warning, |omegaz5| > 1000, which is larger than expected and may be pathological. Triggering MSA failure.",
    )

    # /* Coefficients of Eq. 65, as defined in Equations D16 - D21 of PRD, 95, 104004, (2017), arXiv:1703.03967 */
    omegaz0_coeff = 3.0 * g0 * omegaz0
    omegaz1_coeff = 3.0 * g0 * omegaz1
    omegaz2_coeff = 3.0 * (g0 * omegaz2 + g2 * omegaz0)
    omegaz3_coeff = 3.0 * (g0 * omegaz3 + g2 * omegaz1 + g3 * omegaz0)
    omegaz4_coeff = 3.0 * (g0 * omegaz4 + g2 * omegaz2 + g3 * omegaz1 + g4 * omegaz0)
    omegaz5_coeff = 3.0 * (g0 * omegaz5 + g2 * omegaz3 + g3 * omegaz2 + g4 * omegaz1 + g5 * omegaz0)

    # /* Coefficients of zeta: in Appendix E of PRD, 95, 104004, (2017), arXiv:1703.03967  */
    c1oveta2 = c_1 / eta2
    omega_zeta0 = omegaz0
    omega_zeta1 = omegaz1 + omegaz0 * c1oveta2
    omega_zeta2 = omegaz2 + omegaz1 * c1oveta2
    omega_zeta3 = omegaz3 + omegaz2 * c1oveta2 + gd_d
    omega_zeta4 = omegaz4 + omegaz3 * c1oveta2 - gd_d * s_eff - gd_d * hd_d
    omega_zeta5 = omegaz5 + omegaz4 * c1oveta2 + gd_d * hd_d * s_eff + gd_d * (hd_d_2 - fd_d)
    # #if DEBUG == 1
    # printf("Omegazeta0     = %.6f\n",p_prec->Omegazeta0)
    # printf("Omegazeta1     = %.6f\n",p_prec->Omegazeta1)
    # printf("Omegazeta2     = %.6f\n",p_prec->Omegazeta2)
    # printf("Omegazeta3     = %.6f\n",p_prec->Omegazeta3)
    # printf("Omegazeta4     = %.6f\n",p_prec->Omegazeta4)
    # printf("Omegazeta5     = %.6f\n\n",p_prec->Omegazeta5)
    # #endif

    omega_zeta0_coeff = -g0 * omega_zeta0
    omega_zeta1_coeff = -1.5 * g0 * omega_zeta1
    omega_zeta2_coeff = -3.0 * (g0 * omega_zeta2 + g2 * omega_zeta0)
    omega_zeta3_coeff = 3.0 * (g0 * omega_zeta3 + g2 * omega_zeta1 + g3 * omega_zeta0)
    omega_zeta4_coeff = 3.0 * (g0 * omega_zeta4 + g2 * omega_zeta2 + g3 * omega_zeta1 + g4 * omega_zeta0)
    omega_zeta5_coeff = 1.5 * (
        g0 * omega_zeta5 + g2 * omega_zeta3 + g3 * omega_zeta2 + g4 * omega_zeta1 + g5 * omega_zeta0
    )

    # /* Expansion order of corrections to retain */
    # switch(ExpansionOrder)
    # {
    # /* Generate all orders */
    # case -1:
    # {
    # break
    # }
    # case 1:
    # {
    # p_prec->Omegaz1_coeff    = 0.0
    # p_prec->Omegazeta1_coeff = 0.0
    # #if __GNUC__ >= 7 && !defined __INTEL_COMPILER
    #             __attribute__ ((fallthrough))
    # #endif

    # }
    # case 2:
    # {
    # p_prec->Omegaz2_coeff    = 0.0
    # p_prec->Omegazeta2_coeff = 0.0
    # #if __GNUC__ >= 7 && !defined __INTEL_COMPILER
    #             __attribute__ ((fallthrough))
    # #endif

    # }
    # case 3:
    # {
    # p_prec->Omegaz3_coeff    = 0.0
    # p_prec->Omegazeta3_coeff = 0.0
    # #if __GNUC__ >= 7 && !defined __INTEL_COMPILER
    #             __attribute__ ((fallthrough))
    # #endif

    # }
    # case 4:
    # {
    # p_prec->Omegaz4_coeff    = 0.0
    # p_prec->Omegazeta4_coeff = 0.0
    # #if __GNUC__ >= 7 && !defined __INTEL_COMPILER
    #             __attribute__ ((fallthrough))
    # #endif

    # }
    # case 5:
    # {
    # p_prec->Omegaz5_coeff    = 0.0
    # p_prec->Omegazeta5_coeff = 0.0
    # break
    # }
    # default:
    # {
    # XLAL_ERROR(XLAL_EDOM, "Expansion order for MSA corrections = %i not recognized. Default is 5. Allowed values are: [-1,1,2,3,4,5].",ExpansionOrder)
    # }
    # }

    omegaz_coeffs = jnp.array(
        [omegaz0_coeff, omegaz1_coeff, omegaz2_coeff, omegaz3_coeff, omegaz4_coeff, omegaz5_coeff]
    )
    omega_zeta_coeffs = jnp.array(
        [
            omega_zeta0_coeff,
            omega_zeta1_coeff,
            omega_zeta2_coeff,
            omega_zeta3_coeff,
            omega_zeta4_coeff,
            omega_zeta5_coeff,
        ]
    )

    def _zero_from(idx, vec):
        mask = jnp.arange(vec.shape[0]) >= idx
        return jnp.where(mask, 0.0, vec)

    def _truncate_coeffs(order):
        # if order == -1:
        #     return (Omegaz_coeffs, Omegazeta_coeffs)
        # idx = order  # orders start at 1
        # return (
        #     _zero_from(idx, Omegaz_coeffs),
        #     _zero_from(idx, Omegazeta_coeffs),
        # )
        return jax.lax.cond(
            order == -1,
            lambda: (omegaz_coeffs, omega_zeta_coeffs),
            lambda: (_zero_from(order, omegaz_coeffs), _zero_from(order, omega_zeta_coeffs)),
        )

    # Check if expansion_order is valid using JAX-compatible operations
    is_valid_expansion_order = jnp.any(
        jnp.array(
            [
                expansion_order == -1,
                expansion_order == 1,
                expansion_order == 2,
                expansion_order == 3,
                expansion_order == 4,
                expansion_order == 5,
            ]
        )
    )
    checkify.check(
        is_valid_expansion_order,
        "Expansion order for MSA corrections = %i not recognized. Default is 5. Allowed values are: [-1,1,2,3,4,5].",
    )
    omegaz_coeffs, omega_zeta_coeffs = _truncate_coeffs(expansion_order)
    omegaz0_coeff, omegaz1_coeff, omegaz2_coeff, omegaz3_coeff, omegaz4_coeff, omegaz5_coeff = omegaz_coeffs
    omega_zeta0_coeff, omega_zeta1_coeff, omega_zeta2_coeff, omega_zeta3_coeff, omega_zeta4_coeff, omega_zeta5_coeff = (
        omega_zeta_coeffs
    )

    # #if DEBUG == 1
    # printf("Omegaz0_coeff     = %.6f\n",p_prec->Omegaz0_coeff)
    # printf("Omegaz1_coeff     = %.6f\n",p_prec->Omegaz1_coeff)
    # printf("Omegaz2_coeff     = %.6f\n",p_prec->Omegaz2_coeff)
    # printf("Omegaz3_coeff     = %.6f\n",p_prec->Omegaz3_coeff)
    # printf("Omegaz4_coeff     = %.6f\n",p_prec->Omegaz4_coeff)
    # printf("Omegaz5_coeff     = %.6f\n\n",p_prec->Omegaz5_coeff)

    # printf("Omegazeta0_coeff     = %.6f\n",p_prec->Omegazeta0_coeff)
    # printf("Omegazeta1_coeff     = %.6f\n",p_prec->Omegazeta1_coeff)
    # printf("Omegazeta2_coeff     = %.6f\n",p_prec->Omegazeta2_coeff)
    # printf("Omegazeta3_coeff     = %.6f\n",p_prec->Omegazeta3_coeff)
    # printf("Omegazeta4_coeff     = %.6f\n",p_prec->Omegazeta4_coeff)
    # printf("Omegazeta5_coeff     = %.6f\n\n",p_prec->Omegazeta5_coeff)
    # #endif

    # /* Get psi0 term */
    # psi_of_v0 = 0.0
    # mm = 0.0
    # tmpB = 0.0
    # volume_element = 0.0
    # vol_sign = 0.0

    # Add g0 to p_prec
    p_prec = dataclasses.replace(p_prec, g0=g0)

    # #if DEBUG == 1
    # printf("psi1     = %.6f\n",p_prec->psi1)
    # printf("psi2     = %.6f\n\n",p_prec->psi2)
    # printf("S_0_norm = %.6f\n\n",p_prec->S_0_norm)
    # #endif

    # /* Tolerance chosen to be consistent with implementation in LALSimInspiralFDPrecAngles */
    # if( fabs(p_prec->Smi2 - p_prec->Spl2) < 1.0e-5)
    # {
    # p_prec->psi0  = 0.0
    # }
    # else
    # {
    # mm      = sqrt( (p_prec->Smi2 - p_prec->Spl2) / (p_prec->S32 - p_prec->Spl2) )
    # tmpB    = (p_prec->S_0_norm*p_prec->S_0_norm - p_prec->Spl2) / (p_prec->Smi2 - p_prec->Spl2)

    # volume_element  = IMRPhenomX_vector_dot_product( IMRPhenomX_vector_cross_product(L_0,S1v), S2v)
    # vol_sign        = (volume_element > 0) - (volume_element < 0)

    # psi_of_v0       = IMRPhenomX_psiofv(p_prec->v_0, p_prec->v_0_2, 0.0, p_prec->psi1, p_prec->psi2, p_prec)

    # if( tmpB < 0. || tmpB > 1. )
    # {
    # if(tmpB > 1.0 && (tmpB - 1.) < 0.00001)
    # {
    #     p_prec->psi0 = gsl_sf_ellint_F(asin(vol_sign*sqrt(1.)) , mm, GSL_PREC_) - psi_of_v0
    # }
    # if(tmpB < 0.0 && tmpB > -0.00001)
    # {
    #     p_prec->psi0 = gsl_sf_ellint_F(asin(vol_sign*sqrt(0.)), mm, GSL_PREC_) - psi_of_v0
    # }
    # }
    # else
    # {
    # p_prec->psi0   = gsl_sf_ellint_F(asin( vol_sign * sqrt(tmpB) ), mm, GSL_PREC_DOUBLE) - psi_of_v0
    # }
    # }

    def out_of_tolerance_branch():
        mm = jnp.sqrt((s_mi2 - s_pl2) / (s32 - s_pl2))
        tmp_b = (s_0_norm * s_0_norm - s_pl2) / (s_mi2 - s_pl2)

        volume_element = jnp.inner(jnp.cross(l_0, s1v), s2v)
        vol_sign = jnp.sign(volume_element)  # (volume_element > 0) - (volume_element < 0)

        psi_of_v0 = imr_phenom_x_psiofv(v_0, v_0_2, 0.0, psi1, psi2, p_prec)

        def case1():
            return ellipfinc(jnp.arcsin(vol_sign * jnp.sqrt(1.0)), mm) - psi_of_v0

        def case2():
            return ellipfinc(jnp.arcsin(vol_sign * jnp.sqrt(0.0)), mm) - psi_of_v0

        def case3():
            return ellipfinc(jnp.arcsin(vol_sign * jnp.sqrt(tmp_b)), mm) - psi_of_v0

        def _case3(_: None) -> float:
            return case3()

        def _fallback(_: None) -> float:
            return jax.lax.cond(
                jnp.logical_and(tmp_b > 1.0, (tmp_b - 1.0) < 0.00001),
                lambda __: case1(),
                lambda __: case2(),
                operand=None,
            )

        return jax.lax.cond(
            jnp.logical_and(tmp_b <= 1.0, tmp_b >= 0.0),
            _case3,
            _fallback,
            operand=None,
        )

    psi0 = jax.lax.cond(
        jnp.abs(s_mi2 - s_pl2) < 1.0e-5,
        lambda: 0.0,
        out_of_tolerance_branch,
    )

    # Update p_prec again
    p_prec = dataclasses.replace(
        p_prec,
        s_pl2=s_pl2,
        s_mi2=s_mi2,
        s32=s32,
        s_pl2_p_s_mi2=s_pl2_p_s_mi2,
        s_pl2_m_s_mi2=s_pl2_m_s_mi2,
        s_pl=s_pl,
        s_mi=s_mi,
        s_av2=s_av2,
        s_av=s_av,
        inv_s_av2=inv_s_av2,
        inv_s_av=inv_s_av,
        c1=c_1,
        c12=c1_2,
        c1_over_eta=c1_over_eta,
        s1_l_pav=s1_l_pav,
        s2_l_pav=s2_l_pav,
        s1_s2_pav=s1_s2_pav,
        s1_l_sq_pav=s1_l_sq_pav,
        s2_l_sq_pav=s2_l_sq_pav,
        s1_l_s2_l_pav=s1_l_s2_l_pav,
        beta3=beta3,
        beta5=beta5,
        beta6=beta6,
        beta7=beta7,
        sigma4=sigma4,
        a0=a0,
        a2=a2,
        a3=a3,
        a4=a4,
        a5=a5,
        a6=a6,
        a7=a7,
        a0_2=a0_2,
        a0_3=a0_3,
        a2_2=a2_2,
        g0=g0,
        g2=g2,
        g3=g3,
        g4=g4,
        g5=g5,
        psi0=psi0,
        psi1=psi1,
        psi2=psi2,
        delta=delta,
        omegaz0=omegaz0,
        omegaz1=omegaz1,
        omegaz2=omegaz2,
        omegaz3=omegaz3,
        omegaz4=omegaz4,
        omegaz5=omegaz5,
        omegaz0_coeff=omegaz0_coeff,
        omegaz1_coeff=omegaz1_coeff,
        omegaz2_coeff=omegaz2_coeff,
        omegaz3_coeff=omegaz3_coeff,
        omegaz4_coeff=omegaz4_coeff,
        omegaz5_coeff=omegaz5_coeff,
        omega_zeta0=omega_zeta0,
        omega_zeta1=omega_zeta1,
        omega_zeta2=omega_zeta2,
        omega_zeta3=omega_zeta3,
        omega_zeta4=omega_zeta4,
        omega_zeta5=omega_zeta5,
        omega_zeta0_coeff=omega_zeta0_coeff,
        omega_zeta1_coeff=omega_zeta1_coeff,
        omega_zeta2_coeff=omega_zeta2_coeff,
        omega_zeta3_coeff=omega_zeta3_coeff,
        omega_zeta4_coeff=omega_zeta4_coeff,
        omega_zeta5_coeff=omega_zeta5_coeff,
        phiz_0=0.0,
        zeta_0=0.0,
    )

    # #if DEBUG == 1
    # printf("psi0_of_v0  = %.6f\n",psi_of_v0)
    # printf("tmpB        = %.6f\n",tmpB)
    # printf("psi0        = %.6f\n\n",p_prec->psi0)
    # #endif

    # vector vMSA = {0.,0.,0.}

    phiz_0 = 0.0
    # phiz_0_MSA = 0.0

    zeta_0 = 0.0
    # zeta_0_MSA = 0.0

    # /* Tolerance chosen to be consistent with implementation in LALSimInspiralFDPrecAngles */
    # if( fabs(p_prec->Spl2 - p_prec->Smi2) > 1.e-5 )
    # {
    # vMSA = imr_phenom_x_return_msa_corrections_msa(p_prec->v_0,p_prec->L_0_norm,p_prec->J_0_norm,p_prec)

    # phiz_0_MSA = vMSA.x
    # zeta_0_MSA = vMSA.y
    # }

    v_msa = jax.lax.select(
        jnp.abs(s_pl2 - s_mi2) < 1.0e-5,
        jnp.zeros(3),
        imr_phenom_x_return_msa_corrections_msa(v_0, l_0_norm, j_0_norm, p_prec),
    )
    # phiz_0_MSA = v_msa[0]
    # zeta_0_MSA = v_msa[1]

    # // Initial \phi_z
    phiz_0 = imr_phenom_x_return_phiz_msa(v_0, j_0_norm, p_prec)

    # // Initial \zeta
    zeta_0 = imr_phenom_x_return_zeta_msa(v_0, p_prec)

    # p_prec->phiz_0    = - phiz_0 - vMSA.x
    # p_prec->zeta_0    = - zeta_0 - vMSA.y

    # Update p_prec a final time
    p_prec = dataclasses.replace(
        p_prec,
        phiz_0=-phiz_0 - v_msa[0],
        zeta_0=-zeta_0 - v_msa[1],
    )

    return p_prec


def imr_phenom_x_return_zeta_msa(v: float, p_prec: IMRPhenomXPrecessionDataClass) -> float:
    invv = 1.0 / v
    invv2 = invv * invv
    invv3 = invv * invv2
    v2 = v * v
    logv = jnp.log(v)

    # Compute zeta using precession coefficients
    zeta_out = (
        p_prec.eta
        * (
            p_prec.omega_zeta0_coeff * invv3
            + p_prec.omega_zeta1_coeff * invv2
            + p_prec.omega_zeta2_coeff * invv
            + p_prec.omega_zeta3_coeff * logv
            + p_prec.omega_zeta4_coeff * v
            + p_prec.omega_zeta5_coeff * v2
        )
        + p_prec.zeta_0
    )

    # Replace NaNs with 0 using jnp.nan_to_num
    zeta_out = jnp.nan_to_num(zeta_out, nan=0.0)

    return zeta_out


def imr_phenom_x_return_phiz_msa(v: float, j_norm: float, p_prec: IMRPhenomXPrecessionDataClass) -> float:

    invv = 1.0 / v
    invv2 = invv * invv
    l_newt = p_prec.eta / v

    c1 = p_prec.c1
    c12 = c1 * c1

    s_av2 = p_prec.s_av2
    s_av = p_prec.s_av
    inv_s_av = p_prec.inv_s_av
    inv_s_av2 = p_prec.inv_s_av2

    # These are log functions defined in Eq. D27 and D28 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    log1 = jnp.log(jnp.abs(c1 + j_norm * p_prec.eta + p_prec.eta * l_newt))
    log2 = jnp.log(jnp.abs(c1 + j_norm * s_av * v + s_av2 * v))
    # Eq. D22-D27 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    phiz_0_coeff = (j_norm * p_prec.inveta**4) * (
        0.5 * c12 - (c1 * p_prec.eta2 * invv) / 6.0 - (s_av2 * p_prec.eta2) / 3.0 - (p_prec.eta4 * invv2) / 3.0
    ) - (0.5 * c1 * p_prec.inveta) * (c12 * p_prec.inveta**4 - s_av2 * p_prec.inveta**2) * log1

    phiz_1_coeff = (
        -0.5 * j_norm * p_prec.inveta**2 * (c1 + p_prec.eta * l_newt)
        + 0.5 * p_prec.inveta**3 * (c12 - p_prec.eta2 * s_av2) * log1
    )

    phiz_2_coeff = -j_norm + s_av * log2 - c1 * log1 * p_prec.inveta
    phiz_3_coeff = j_norm * v - p_prec.eta * log1 + c1 * log2 * inv_s_av

    phiz_4_coeff = (
        0.5 * j_norm * inv_s_av2 * v * (c1 + v * s_av2)
        - 0.5 * inv_s_av2 * inv_s_av * (c12 - p_prec.eta2 * s_av2) * log2
    )

    phiz_5_coeff = (
        -j_norm
        * v
        * (0.5 * c12 * inv_s_av2 * inv_s_av2 - c1 * v * inv_s_av2 / 6.0 - v * v / 3.0 - p_prec.eta2 * inv_s_av2 / 3.0)
        + 0.5 * c1 * inv_s_av2 * inv_s_av2 * inv_s_av * (c12 - p_prec.eta2 * s_av2) * log2
    )

    # Eq. 66 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967

    # \phi_{z,-1} = \sum^5_{n=0} <\Omega_z>^(n) \phi_z^(n) + \phi_{z,-1}^0

    # Note that the <\Omega_z>^(n) are given by p_prec->Omegazn_coeff's as in Eqs. D15-D20
    phiz_out = (
        phiz_0_coeff * p_prec.omegaz0_coeff
        + phiz_1_coeff * p_prec.omegaz1_coeff
        + phiz_2_coeff * p_prec.omegaz2_coeff
        + phiz_3_coeff * p_prec.omegaz3_coeff
        + phiz_4_coeff * p_prec.omegaz4_coeff
        + phiz_5_coeff * p_prec.omegaz5_coeff
        + p_prec.phiz_0
    )

    # Ensure no NaN (replace with 0.0 if NaN)
    phiz_out = jnp.nan_to_num(phiz_out, nan=0.0)

    return phiz_out


def imr_phenom_x_return_msa_corrections_msa(
    v: float, l_norm: float, j_norm: float, p_prec: IMRPhenomXPrecessionDataClass
):  # Adapted from Leonardo Ricca's code

    pflag = p_prec.imr_phenom_x_prec_version

    v2 = v * v

    # Sets c0, c2 and c4 in p_prec as per Eq. B6-B8 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    c_vec = imr_phenom_x_return_constants_c_msa(v, j_norm, p_prec)
    # Sets d0, d2 and d4 in p_prec as per Eq. B9-B11 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    d_vec = imr_phenom_x_return_constants_d_msa(l_norm, j_norm, p_prec)

    c0, c2, c4 = c_vec
    d0, d2, d4 = d_vec

    two_d0 = 2.0 * d0

    # Eq. B20 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    sd = jnp.sqrt(jnp.abs(d2 * d2 - 4.0 * d0 * d4))

    # Eq. F20-21 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    a_theta_l = 0.5 * ((j_norm / l_norm) + (l_norm / j_norm) - (p_prec.s_pl2 / (j_norm * l_norm)))
    b_theta_l = 0.5 * p_prec.s_pl2_m_s_mi2 / (j_norm * l_norm)

    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd

    nc = nc_num / nc_denom
    nd = nc_denom / two_d0

    sqrt_nc = jnp.sqrt(jnp.abs(nc))
    sqrt_nd = jnp.sqrt(jnp.abs(nd))

    psi = imr_phenom_x_return_psi_msa(v, v2, p_prec) + p_prec.psi0
    psi_dot = imr_phenom_x_return_psi_dot_msa(v, p_prec)

    tan_psi = jnp.tan(psi)
    atan_psi = jnp.arctan(tan_psi)

    big_c1 = -0.5 * (c0 / d0 - 2.0 * (c0 + c2 + c4) / nc_num)
    big_c2num = c0 * (-2.0 * d0 * d4 + d2 * d2 + d2 * d4) - c2 * d0 * (d2 + 2.0 * d4) + c4 * d0 * (two_d0 + d2)
    big_c2den = 2.0 * d0 * sd * (d0 + d2 + d4)
    big_c2 = big_c2num / big_c2den

    big_cphi = big_c1 + big_c2
    big_dphi = big_c1 - big_c2

    def compute_c_phi_term():
        def _222_223_branch():
            # // As implemented in LALSimInspiralFDPrecAngles.c, c.f. https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralFDPrecAngles_internals.c#L772
            return (
                jnp.abs(
                    (
                        (
                            c4 * d0 * ((2 * d0 + d2) + sd)
                            - c2 * d0 * ((d2 + 2.0 * d4) - sd)
                            - c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - sd))
                        )
                        / big_c2den
                    )
                    * (sqrt_nc / (nc - 1.0))
                    * (atan_psi - jnp.arctan(sqrt_nc * tan_psi))
                )
                / psi_dot
            )

        def non_222_223_branch():
            return ((big_cphi / psi_dot) * sqrt_nc / (nc - 1.0)) * jnp.arctan(
                ((1.0 - sqrt_nc) * tan_psi) / (1.0 + (sqrt_nc * tan_psi * tan_psi))
            )

        return jax.lax.select(
            jnp.logical_or(pflag == 222, pflag == 223),
            _222_223_branch(),
            non_222_223_branch(),
        )

    def compute_d_phi_term():
        def _222_223_branch():
            # // As implemented in LALSimInspiralFDPrecAngles.c, c.f. https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralFDPrecAngles_internals.c#L772
            return (
                jnp.abs(
                    (
                        (
                            -c4 * d0 * ((2 * d0 + d2) - sd)
                            + c2 * d0 * ((d2 + 2.0 * d4) + sd)
                            - c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd))
                        )
                        / big_c2den
                    )
                    * (sqrt_nd / (nd - 1.0))
                    * (atan_psi - jnp.arctan(sqrt_nd * tan_psi))
                )
                / psi_dot
            )

        def non_222_223_branch():
            return ((big_dphi / psi_dot) * sqrt_nd / (nd - 1.0)) * jnp.arctan(
                ((1.0 - sqrt_nd) * tan_psi) / (1.0 + (sqrt_nd * tan_psi * tan_psi))
            )

        return jax.lax.select(
            jnp.logical_or(pflag == 222, pflag == 223),
            _222_223_branch(),
            non_222_223_branch(),
        )

    phiz_0_msa_cphi_term = jax.lax.cond(  # jnp.where(nc == 1.0, 0.0, compute_Cphi_term())
        nc == 1.0,
        lambda: 0.0,
        compute_c_phi_term,
    )

    phiz_0_msa_dphi_term = jax.lax.cond(  # jnp.where(nd == 1.0, 0.0, compute_Dphi_term())
        nd == 1.0,
        lambda: 0.0,
        compute_d_phi_term,
    )

    v_msa_x = phiz_0_msa_cphi_term + phiz_0_msa_dphi_term

    #####  restart from here
    v_msa_y_222_223_224_value = a_theta_l * v_msa_x + 2.0 * b_theta_l * d0 * (
        phiz_0_msa_cphi_term / (sd - d2) - phiz_0_msa_dphi_term / (sd + d2)
    )
    v_msa_y_non_222_223_224_value = (
        (a_theta_l * (big_cphi + big_dphi)) + (2.0 * d0 * b_theta_l) * ((big_cphi / (sd - d2)) - (big_dphi / (sd + d2)))
    ) / psi_dot
    v_msa_y = jax.lax.select(
        jnp.any(jnp.array([pflag == 222, pflag == 223, pflag == 224])),
        v_msa_y_222_223_224_value,
        v_msa_y_non_222_223_224_value,
    )

    v_msa_x = jax.lax.select(  # jnp.where(jnp.isnan(vMSA_x), 0.0, vMSA_x)
        jnp.isnan(v_msa_x),
        0.0,
        v_msa_x,
    )
    v_msa_y = jax.lax.select(  # jnp.where(jnp.isnan(vMSA_y), 0.0, vMSA_y)
        jnp.isnan(v_msa_y),
        0.0,
        v_msa_y,
    )

    return jnp.array([v_msa_x, v_msa_y, 0.0])


def imr_phenom_x_return_constants_c_msa(v: float, j_norm: float, p_prec: IMRPhenomXPrecessionDataClass):
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v6 = v3 * v3
    j_norm2 = j_norm * j_norm
    s_eff = p_prec.s_eff

    x = j_norm * (
        0.75
        * (1.0 - s_eff * v)
        * v2
        * (
            p_prec.eta3
            + 4.0 * p_prec.eta3 * s_eff * v
            - 2.0
            * p_prec.eta
            * (j_norm2 - p_prec.s_pl2 + 2.0 * (p_prec.s1_norm_2 - p_prec.s2_norm_2) * p_prec.delta_qq)
            * v2
            - 4.0 * p_prec.eta * s_eff * (j_norm2 - p_prec.s_pl2) * v3
            + (j_norm2 - p_prec.s_pl2) ** 2 * v4 * p_prec.inveta
        )
    )

    y = j_norm * (
        -1.5
        * p_prec.eta
        * (p_prec.s_pl2 - p_prec.s_mi2)
        * (1.0 + 2.0 * s_eff * v - (j_norm2 - p_prec.s_pl2) * v2 * p_prec.inveta**2)
        * (1.0 - s_eff * v)
        * v4
    )

    z = j_norm * (0.75 * p_prec.inveta * (p_prec.s_pl2 - p_prec.s_mi2) ** 2 * (1.0 - s_eff * v) * v6)
    return jnp.array([x, y, z])


def imr_phenom_x_return_constants_d_msa(l_norm: float, j_norm: float, p_prec: IMRPhenomXPrecessionDataClass):
    l_norm2 = l_norm * l_norm
    j_norm2 = j_norm * j_norm

    x = -((j_norm2 - (l_norm + p_prec.s_pl)) ** 2) * (j_norm2 - (l_norm - p_prec.s_pl)) ** 2
    y = -2.0 * (p_prec.s_pl2 - p_prec.s_mi2) * (j_norm2 + l_norm2 - p_prec.s_pl2)
    z = -((p_prec.s_pl2 - p_prec.s_mi2) ** 2)

    return jnp.array([x, y, z])


def imr_phenom_x_return_psi_msa(v: float, v2: float, p_prec: IMRPhenomXPrecessionDataClass):
    return -0.75 * p_prec.g0 * p_prec.delta_qq * (1.0 + p_prec.psi1 * v + p_prec.psi2 * v2) / (v2 * v)


def imr_phenom_x_return_psi_dot_msa(v: float, p_prec: IMRPhenomXPrecessionDataClass):
    v2 = v * v

    big_a_coeff = -1.5 * v2 * v2 * v2 * (1.0 - v * p_prec.s_eff) * jnp.sqrt(p_prec.inveta)
    psi_dot = 0.5 * big_a_coeff * jnp.sqrt(p_prec.s_pl2 - p_prec.s32)
    return psi_dot


def imr_phenom_x_psiofv(
    v: float, v2: float, psi0: float, psi1: float, psi2: float, p_prec: IMRPhenomXPrecessionDataClass
) -> float:
    """
    Equation 51 in arXiv:1703.03967
    """
    return psi0 - 0.75 * p_prec.g0 * p_prec.delta_qq * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)


def imr_phenom_x_return_spin_evolution_coefficients_msa(
    l_norm: float, j_norm: float, p_prec: IMRPhenomXPrecessionDataClass
) -> Array:
    # // Total angular momenta: J = L + S1 + S2
    j_norm2 = j_norm * j_norm

    # // Orbital angular momenta
    l_norm2 = l_norm * l_norm

    # // Dimensionfull spin angular momenta
    s1_norm2 = p_prec.s1_norm_2
    s2_norm2 = p_prec.s2_norm_2

    q = p_prec.qq
    eta = p_prec.eta

    j2_m_l2 = j_norm2 - l_norm2
    j2_m_l2_sq = j2_m_l2 * j2_m_l2

    delta = p_prec.delta_qq
    delta_sq = delta * delta
    # /*
    #     Note:
    #     S_{eff} \equiv \xi = (1 + q)(S1.L) + (1 + 1/q)(S2.L)
    # */
    s_eff = p_prec.s_eff

    # // Note that we do not evaluate Eq. B1 here as it is v dependent whereas B, C and D are not

    # Set Eq. B2, B_coeff
    vout_x = (
        (l_norm2 + s1_norm2) * q + 2.0 * l_norm * s_eff - 2.0 * j_norm2 - s1_norm2 - s2_norm2 + (l_norm2 + s2_norm2) / q
    )

    # // Set Eq. B3, C_coeff
    vout_y = (
        j2_m_l2_sq
        - 2.0 * l_norm * s_eff * j2_m_l2
        - 2.0 * ((1.0 - q) / q) * l_norm2 * (s1_norm2 - q * s2_norm2)
        + 4.0 * eta * l_norm2 * s_eff * s_eff
        - 2.0 * delta * (s1_norm2 - s2_norm2) * s_eff * l_norm
        + 2.0 * ((1.0 - q) / q) * (q * s1_norm2 - s2_norm2) * j_norm2
    )

    # // Set Eq. B4, D_coeff
    vout_z = (
        ((1.0 - q) / q) * (s2_norm2 - q * s1_norm2) * j2_m_l2_sq
        + delta_sq * (s1_norm2 - s2_norm2) * (s1_norm2 - s2_norm2) * l_norm2 / eta
        + 2.0 * delta * l_norm * s_eff * (s1_norm2 - s2_norm2) * j2_m_l2
    )

    return jnp.array([vout_x, vout_y, vout_z])


def imr_phenom_x_return_roots_msa(l_norm: float, j_norm: float, p_prec: IMRPhenomXPrecessionDataClass) -> Array:
    v_bcd = imr_phenom_x_return_spin_evolution_coefficients_msa(l_norm, j_norm, p_prec)

    # /* Update struct. Note, that this agreed with independent implementation in Mathematica. */
    b = v_bcd[0]
    c = v_bcd[1]
    d = v_bcd[2]

    s1_norm2 = p_prec.s1_norm_2
    s2_norm2 = p_prec.s2_norm_2

    s_0_norm2 = p_prec.s_0_norm_2

    b2 = b * b
    b3 = b2 * b
    bc = b * c

    p = c - b2 / 3
    qc = (2.0 / 27.0) * b3 - bc / 3.0 + d
    sqrtarg = jnp.sqrt(-p / 3.0)
    acosarg = 1.5 * qc / p / sqrtarg

    # // Make sure that acosarg is appropriately bounded
    # if(acosarg < -1)
    # {
    # acosarg = -1
    # }
    # if(acosarg > 1)
    # {
    # acosarg = +1
    # }
    acosarg = jax.lax.select(acosarg < -1.0, -1.0, jax.lax.select(acosarg > 1.0, +1.0, acosarg))

    theta = jnp.arccos(acosarg) / 3.0
    cos_theta = jnp.cos(theta)

    dot_s1_ln = p_prec.dot_s1_ln
    dot_s2_ln = p_prec.dot_s2_ln

    # // tmp1 = S32
    # // tmp2 = Smi2
    # // tmp3 = Spl2

    def perturbation_branch():
        s32 = 0.0
        s_mi2 = s_0_norm2

        # /*
        #     Add a numerical perturbation to prevent azimuthal precession angle
        #     from diverging.
        # */

        # // Smi2 = S02^2 + epsilon perturbation
        spl2 = s_mi2 + 1e-9

        return spl2, s32, s_mi2

    def non_perturbation_branch():
        # /* E.g. see discussion on elliptic functions in arXiv:0711.4064 */
        tmp1 = 2.0 * sqrtarg * jnp.cos(theta - 2.0 * (2 * PI) / 3.0) - b / 3.0
        tmp2 = 2.0 * sqrtarg * jnp.cos(theta - (2 * PI) / 3.0) - b / 3.0
        tmp3 = 2.0 * sqrtarg * cos_theta - b / 3.0

        tmp4 = jnp.maximum(jnp.maximum(tmp1, tmp2), tmp3)
        tmp5 = jnp.minimum(jnp.minimum(tmp1, tmp2), tmp3)

        # // As tmp1 and tmp3 are set by findind the min and max, find the remaining root
        # if( (tmp4 - tmp3) > 0.0 && (tmp5 - tmp3) < 0.0)
        # {
        #     tmp6 = tmp3
        # }
        # else if( (tmp4 - tmp1) > 0.0 && (tmp5 - tmp1) < 0.0)
        # {
        #     tmp6 = tmp1
        # }
        # else
        # {
        #     tmp6 = tmp2
        # }
        tmp6 = jax.lax.select(
            jnp.logical_and((tmp4 - tmp3) > 0.0, (tmp5 - tmp3) < 0.0),
            tmp3,
            jax.lax.select(jnp.logical_and((tmp4 - tmp1) > 0.0, (tmp5 - tmp1) < 0.0), tmp1, tmp2),
        )

        # /*
        #     When Spl2 ~ 0 to numerical roundoff then Smi2 can sometimes be ~ negative causing NaN's.
        #     This occurs in a very limited portion of the parameter space where spins are ~ 0 to numerical roundoff.
        #     We can circumvent by enforcing +ve definite behaviour when tmp4 ~ 0. Note that S32 can often be negative, this is fine.
        # */
        tmp4 = jnp.abs(tmp4)
        tmp6 = jnp.abs(tmp6)

        return tmp4, tmp5, tmp6

    # if(theta != theta || sqrtarg!=sqrtarg || dotS1Ln == 1 || dotS2Ln == 1 || dotS1Ln == -1 || dotS2Ln == -1 || S1Norm2 == 0
    # || S2Norm2 == 0)

    # (theta != theta)
    #     | (sqrtarg != sqrtarg)
    #     | (dot_s1_ln == 1)
    #     | (dot_s2_ln == 1)
    #     | (dot_s1_ln == -1)
    #     | (dot_s2_ln == -1)
    #     | (s1_norm2 == 0)
    #     | (s2_norm2 == 0),
    spl2, s32, s_mi2 = jax.lax.cond(
        jnp.any(
            jnp.array(
                [
                    jnp.isnan(theta),
                    jnp.isnan(sqrtarg),
                    dot_s1_ln == 1,
                    dot_s2_ln == 1,
                    dot_s1_ln == -1,
                    dot_s2_ln == -1,
                    s1_norm2 == 0,
                    s2_norm2 == 0,
                ]
            )
        ),
        perturbation_branch,
        non_perturbation_branch,
    )

    return jnp.array([s32, s_mi2, spl2])


####################################################################################
####################################### NNLO #######################################
####################################################################################


####################################################################################
#################################### SpinTaylor ####################################
####################################################################################

# def SpinTaylor(
#     p_wf: IMRPhenomXWaveformDataClass,
#     p_prec: IMRPhenomXPrecessionDataClass,
#     lalParams: IMRPhenomXPHMParameterDataClass,
# ):

#     # // check mode array to estimate frequency range over which splines will need to be evaluated
#     ModeArray = lalParams.mode_array

#     LMAX_PNR = jax.lax.select(
#         ModeArray is not None,
#         jax.lax.select(
#             xlal_sim_inspiral_mode_array_is_mode_active(ModeArray, 4, 4),
#             4,
#             jax.lax.select(
#                 xlal_sim_inspiral_mode_array_is_mode_active(ModeArray, 3, 3) | xlal_sim_inspiral_mode_array_is_mode_active(ModeArray, 3, 2),
#                 3,
#                 2
#             )
#         ),
#         2
#     )
#     L_MAX_PNR = LMAX_PNR


#     # // buffer for GSL interpolation to succeed
#     # // set first to fMin
#     flow = p_wf.f_min

#     # if(p_wf.delta_f==0.) p_wf.delta_f = get_deltaF_from_wfstruct(p_wf)
#     p_wf = dataclasses.replace(
#         p_wf,
#         delta_f = jax.lax.select(
#             p_wf.delta_f == 0.0,
#             get_deltaF_from_wfstruct(p_wf),
#             p_wf.delta_f
#         )
#     )

#     # // if PNR angles are disabled, step back accordingly to the waveform's frequency grid step
#     if(PNRUseTunedAngles==false)
#     {

#     p_prec->integration_buffer = (p_wf->deltaF>0.)? 3.*p_wf->deltaF: 0.5
#     flow = (p_wf->fMin-p_prec->integration_buffer)*2./p_prec->M_MAX

#     }
#     // if PNR angles are enabled, adjust buffer to the requirements of IMRPhenomX_PNR_GeneratePNRAngleInterpolants
#     else{

#     size_t iStart_here

#     if (p_wf->deltaF == 0.) iStart_here = 0
#     else{
#     iStart_here= (size_t)(p_wf->fMin / p_wf->deltaF)
#     flow = iStart_here * p_wf->deltaF
#     }

#     fmin_HM_inspiral = flow * 2.0 / p_prec->M_MAX

#     INT4 precVersion = p_prec->IMRPhenomXPrecVersion
#     // fill in a fake value to allow the next code to work
#     p_prec->IMRPhenomXPrecVersion = 223
#     status = IMRPhenomX_PNR_GetAndSetPNRVariables(p_wf, p_prec)
#     XLAL_CHECK(XLAL_SUCCESS == status, XLAL_EFUNC, "Error: IMRPhenomX_PNR_GetAndSetPNRVariables failed in IMRPhenomXGetAndSetPrecessionVariables.\n")

#     /* generate alpha parameters to catch edge cases */
#     IMRPhenomX_PNR_alpha_parameters *alphaParams = XLALMalloc(sizeof(IMRPhenomX_PNR_alpha_parameters))
#     IMRPhenomX_PNR_beta_parameters *betaParams = XLALMalloc(sizeof(IMRPhenomX_PNR_beta_parameters))
#     status = IMRPhenomX_PNR_precompute_alpha_coefficients(alphaParams, p_wf, p_prec)
#     XLAL_CHECK(
#     XLAL_SUCCESS == status,
#     XLAL_EFUNC,
#     "Error: IMRPhenomX_PNR_precompute_alpha_coefficients failed.\n")
#     status = IMRPhenomX_PNR_precompute_beta_coefficients(betaParams, p_wf, p_prec)
#     XLAL_CHECK(
#     XLAL_SUCCESS == status,
#     XLAL_EFUNC,
#     "Error: IMRPhenomX_PNR_precompute_beta_coefficients failed.\n")
#     status = IMRPhenomX_PNR_BetaConnectionFrequencies(betaParams)
#     XLAL_CHECK(
#     XLAL_SUCCESS == status,
#     XLAL_EFUNC,
#     "Error: IMRPhenomX_PNR_BetaConnectionFrequencies failed.\n")
#     p_prec->IMRPhenomXPrecVersion = precVersion
#     Mf_alpha_upper = alphaParams->A4 / 3.0
#     Mf_low_cut = (3.0 / 3.5) * Mf_alpha_upper
#     MF_high_cut = betaParams->Mf_beta_lower
#     LALFree(alphaParams)
#     LALFree(betaParams)

#     if((MF_high_cut > p_wf->fCutDef) || (MF_high_cut < 0.1 * p_wf->fRING)){
#     MF_high_cut = p_wf->fRING
#     }
#     if((Mf_low_cut > p_wf->fCutDef) || (MF_high_cut < Mf_low_cut)){
#     Mf_low_cut = MF_high_cut / 2.0
#     }

#     flow_alpha = XLALSimIMRPhenomXUtilsMftoHz(Mf_low_cut * 0.65 * p_prec->M_MAX / 2.0, p_wf->Mtot)

#     if(flow_alpha < flow){
#     // flow is approximately in the intermediate region of the frequency map
#     // conservatively reduce flow to account for potential problems in this region
#     flow = fmin_HM_inspiral / 1.5
#     }
#     else{
#     Mf_RD_22 = p_wf->fRING
#     Mf_RD_lm = IMRPhenomXHM_GenerateRingdownFrequency(p_prec->L_MAX_PNR, p_prec->M_MAX, p_wf)
#     fmin_HM_ringdowm = XLALSimIMRPhenomXUtilsMftoHz(XLALSimIMRPhenomXUtilsHztoMf(flow, p_wf->Mtot) - (Mf_RD_lm - Mf_RD_22), p_wf->Mtot)
#     flow = ((fmin_HM_ringdowm < fmin_HM_inspiral)&&(fmin_HM_ringdowm > 0.0)) ? fmin_HM_ringdowm : fmin_HM_inspiral
#     }


#     pnr_interpolation_deltaf = IMRPhenomX_PNR_HMInterpolationDeltaF(flow, p_wf, p_prec)
#     p_prec->integration_buffer = 1.4*pnr_interpolation_deltaf
#     flow = (flow - 2.0 * pnr_interpolation_deltaf < 0) ? flow / 2.0 : flow - 2.0 * pnr_interpolation_deltaf

#     iStart_here = (size_t)(flow / pnr_interpolation_deltaf)
#     flow = iStart_here * pnr_interpolation_deltaf
#     }

#     XLAL_CHECK(flow>0.,XLAL_EDOM,"Error in %s: starting frequency for SpinTaylor angles must be positive!",__func__)
#     status = IMRPhenomX_InspiralAngles_SpinTaylor(p_prec->PNarrays,&p_prec->fmin_integration,chi1x,chi1y,chi1z,chi2x,chi2y,chi2z,flow,p_prec->IMRPhenomXPrecVersion,p_wf,lalParams)
#     // convert the min frequency of integration to geometric units for later convenience
#     p_prec->Mfmin_integration = XLALSimIMRPhenomXUtilsHztoMf(p_prec->fmin_integration,p_wf->Mtot)

#     if (p_prec->IMRPhenomXPrecVersion == 330)
#     {

#     chi1x_evolved = chi1x
#     chi1y_evolved = chi1y
#     chi1z_evolved = chi1z
#     chi2x_evolved = chi2x
#     chi2y_evolved = chi2y
#     chi2z_evolved = chi2z

#     // in case that SpinTaylor angles generate, overwrite variables with evolved spins
#     if(status!=XLAL_FAILURE)  {
#     size_t lenPN = p_prec->PNarrays->V_PN->data->length

#     chi1x_temp = p_prec->PNarrays->S1x_PN->data->data[lenPN-1]
#     chi1y_temp = p_prec->PNarrays->S1y_PN->data->data[lenPN-1]
#     chi1z_temp = p_prec->PNarrays->S1z_PN->data->data[lenPN-1]

#     chi2x_temp = p_prec->PNarrays->S2x_PN->data->data[lenPN-1]
#     chi2y_temp = p_prec->PNarrays->S2y_PN->data->data[lenPN-1]
#     chi2z_temp = p_prec->PNarrays->S2z_PN->data->data[lenPN-1]

#     Lx = p_prec->PNarrays->LNhatx_PN->data->data[lenPN-1]
#     Ly = p_prec->PNarrays->LNhaty_PN->data->data[lenPN-1]
#     Lz = p_prec->PNarrays->LNhatz_PN->data->data[lenPN-1]

#     // orbital separation vector not stored in PN arrays
#     //nx = p_prec->PNarrays->E1x->data->data[lenPN-1]
#     //ny = p_prec->PNarrays->E1y->data->data[lenPN-1]

#     // rotate to get x,y,z components in L||z frame
#     phi = atan2( Ly, Lx )
#     theta = acos( Lz / sqrt(Lx*Lx + Ly*Ly + Lz*Lz) )
#     //kappa = atan( ny/nx )

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
#     }

#     p_prec->chi1x_evolved = chi1x_evolved
#     p_prec->chi1y_evolved = chi1y_evolved
#     p_prec->chi1z_evolved = chi1z_evolved
#     p_prec->chi2x_evolved = chi2x_evolved
#     p_prec->chi2y_evolved = chi2y_evolved
#     p_prec->chi2z_evolved = chi2z_evolved

#     //printf("%f, %f, %f, %f, %f, %f\n", chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
#     //printf("%f, %f, %f, %f, %f, %f\n", chi1x_evolved, chi1y_evolved, chi1z_evolved, chi2x_evolved, chi2y_evolved, chi2z_evolved)
#     //printf("----\n")
#     }

#     // if PN numerical integration fails, default to MSA+fallback to NNLO
#     if(status==XLAL_FAILURE) {
#                         LALFree(p_prec->PNarrays)
#                         XLAL_PRINT_WARNING("Warning: due to a failure in the SpinTaylor routines, the model will default to MSA angles.")
#                         p_prec->IMRPhenomXPrecVersion=223
#                         }
#     // end of SpinTaylor code
