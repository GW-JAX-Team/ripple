import dataclasses
import jax.numpy as jnp
from jaxtyping import Float
from ..constants import MTSUN
import jax

from .initialize_MSA_system import (
    IMRPhenomX_Initialize_MSA_System,
    IMRPhenomX_Return_phi_zeta_costhetaL_MSA,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CommonConstants:
    sqrt2: Float = 1.4142135623730951
    sqrt5: Float = 2.23606797749978981
    sqrt6: Float = 2.44948974278317788
    sqrt7: Float = 2.64575131106459072
    sqrt10: Float = 3.16227766016838
    sqrt14: Float = 3.74165738677394133
    sqrt15: Float = 3.87298334620741702
    sqrt70: Float = 8.36660026534075563
    sqrt30: Float = 5.477225575051661
    sqrt2p5: Float = 1.58113883008419
    log16: Float = 2.772588722239781
    power_of_lalpi_2: Float = 9.869604401089358
    MAX_TOL_ATAN: Float = 1.0e-15


def compute_evolved_spin_using_msa(
    Mf,
    mass_1,
    mass_2,
    chi1x,
    chi1y,
    chi1z,
    chi2x,
    chi2y,
    chi2z,
    emm,
    reference_frequency,
    kappa,
    phiJ_Sf,
):
    """
    Args:
        Mf: Dimensionless frequency Mf.
        mass_1: Heavier black hole mass in solar mass units.
        mass_2: Lighter black hole mass in solar mass units.
        chi1x: x-component of the dimensionless spin of the first black hole.
        chi1y: y-component of the dimensionless spin of the first black hole.
        chi1z: z-component of the dimensionless spin of the first black hole.
        chi2x: x-component of the dimensionless spin of the second black hole.
        chi2y: y-component of the dimensionless spin of the second black hole.
        chi2z: z-component of the dimensionless spin of the second black hole.
        emm: The index m of the mode (l, m).
        reference_frequency: Reference frequency for the waveform.
        kappa: Precession angle kappa.
        phiJ_Sf: Azimuthal angle between total angular momentum and source frame.
    """
    inspiral_mask = Mf < 0.3
    # Mf = jnp.clip(Mf, a_max=0.3)
    phenom_xp_convention = 1
    eta = mass_1 * mass_2 / jnp.power(mass_1 + mass_2, 2)
    Msec = (mass_1 + mass_2) * MTSUN
    piM = jnp.pi * Msec

    # if pflag in 220, 221, 222, 223, 224...
    # Line 597
    msa_init = IMRPhenomX_Initialize_MSA_System(
        mass_1=mass_1,
        mass_2=mass_2,
        chi1x=chi1x,
        chi1y=chi1y,
        chi1z=chi1z,
        chi2x=chi2x,
        chi2y=chi2y,
        chi2z=chi2z,
        reference_frequency=reference_frequency,
    )

    alpha0 = jnp.pi - kappa  # For phenom_xp_convention = 1

    epsilon0 = set_epsilon0(phenom_xp_convention, phiJ_Sf)

    # LAL computes the offset with mprime=2 for ALL modes during initialization
    # (see LALSimIMRPhenomX_precession.c line 1192), then uses those same offsets
    # for all modes. The actual mode's emm is only used when computing the angles
    # at runtime via v = cbrt(pi * Mf * 2/emm).
    mprime_for_offset = 2

    eta2 = jnp.power(eta, 2)
    eta3 = jnp.power(eta, 3)
    eta4 = jnp.power(eta, 4)
    inveta = jnp.power(eta, -1)

    SAv2 = msa_init[15]
    SAv = jnp.sqrt(SAv2)
    invSAv = jnp.power(SAv, -1)
    invSAv2 = jnp.power(SAv2, -1)

    qq = mass_2 / mass_1
    delta_qq = (1 - qq) / (1 + qq)

    # When convention five or seven are false...
    # NH My understanding is that these are just vanlges at the reference frequency for the 22 mode.

    alpha_offset, epsilon_offset = Get_alphaepsilon_atfref(
        mprime_for_offset,
        piM,
        reference_frequency,
        alpha0,
        epsilon0,
        eta=eta,
        eta2=eta2,
        eta3=eta3,
        eta4=eta4,
        inveta=inveta,
        SAv=SAv,
        SAv2=SAv2,
        invSAv=invSAv,
        invSAv2=invSAv2,
        qq=qq,
        delta_qq=delta_qq,
        Omegaz0_coeff=msa_init[0],
        Omegaz1_coeff=msa_init[1],
        Omegaz2_coeff=msa_init[2],
        Omegaz3_coeff=msa_init[3],
        Omegaz4_coeff=msa_init[4],
        Omegaz5_coeff=msa_init[5],
        Omegazeta0_coeff=msa_init[6],
        Omegazeta1_coeff=msa_init[7],
        Omegazeta2_coeff=msa_init[8],
        Omegazeta3_coeff=msa_init[9],
        Omegazeta4_coeff=msa_init[10],
        Omegazeta5_coeff=msa_init[11],
        g0=msa_init[12],
        c1=msa_init[13],
        c1_over_eta=msa_init[14],
        Seff=msa_init[16],
        dotS1Ln=msa_init[17],
        dotS2Ln=msa_init[18],
        S_0_norm=msa_init[19],
        psi0=msa_init[20],
        psi1=msa_init[21],
        psi2=msa_init[22],
        phiz_0=msa_init[23],
        zeta_0=msa_init[24],
        constants_L_0=msa_init[25],
        constants_L_1=msa_init[26],
        constants_L_2=msa_init[27],
        constants_L_3=msa_init[28],
        constants_L_4=msa_init[29],
        S1_norm_2=msa_init[30],
        S2_norm_2=msa_init[31],
    )

    vangles = compute_vangles(
        Mf=Mf,
        emm=emm,
        eta=eta,
        eta2=eta2,
        eta3=eta3,
        eta4=eta4,
        inveta=inveta,
        SAv=SAv,
        SAv2=SAv2,
        invSAv=invSAv,
        invSAv2=invSAv2,
        qq=qq,
        delta_qq=delta_qq,
        Omegaz0_coeff=msa_init[0],
        Omegaz1_coeff=msa_init[1],
        Omegaz2_coeff=msa_init[2],
        Omegaz3_coeff=msa_init[3],
        Omegaz4_coeff=msa_init[4],
        Omegaz5_coeff=msa_init[5],
        Omegazeta0_coeff=msa_init[6],
        Omegazeta1_coeff=msa_init[7],
        Omegazeta2_coeff=msa_init[8],
        Omegazeta3_coeff=msa_init[9],
        Omegazeta4_coeff=msa_init[10],
        Omegazeta5_coeff=msa_init[11],
        g0=msa_init[12],
        c1=msa_init[13],
        c1_over_eta=msa_init[14],
        Seff=msa_init[16],
        dotS1Ln=msa_init[17],
        dotS2Ln=msa_init[18],
        S_0_norm=msa_init[19],
        psi0=msa_init[20],
        psi1=msa_init[21],
        psi2=msa_init[22],
        phiz_0=msa_init[23],
        zeta_0=msa_init[24],
        constants_L_0=msa_init[25],
        constants_L_1=msa_init[26],
        constants_L_2=msa_init[27],
        constants_L_3=msa_init[28],
        constants_L_4=msa_init[29],
        S1_norm_2=msa_init[30],
        S2_norm_2=msa_init[31],
    )

    alpha_out = jnp.where(inspiral_mask, vangles[0] - alpha_offset, 0.0)
    epsilon_out = jnp.where(inspiral_mask, vangles[1] - epsilon_offset, 0.0)
    cos_beta_out = jnp.where(inspiral_mask, vangles[2], 0.0)

    debug = False
    if debug:
        jax.debug.print("ripple debug msa_init[0]  Omegaz0_coeff    = {}", msa_init[0])
        jax.debug.print("ripple debug msa_init[1]  Omegaz1_coeff    = {}", msa_init[1])
        jax.debug.print("ripple debug msa_init[2]  Omegaz2_coeff    = {}", msa_init[2])
        jax.debug.print("ripple debug msa_init[3]  Omegaz3_coeff    = {}", msa_init[3])
        jax.debug.print("ripple debug msa_init[4]  Omegaz4_coeff    = {}", msa_init[4])
        jax.debug.print("ripple debug msa_init[5]  Omegaz5_coeff    = {}", msa_init[5])
        jax.debug.print("ripple debug msa_init[6]  Omegazeta0_coeff = {}", msa_init[6])
        jax.debug.print("ripple debug msa_init[7]  Omegazeta1_coeff = {}", msa_init[7])
        jax.debug.print("ripple debug msa_init[8]  Omegazeta2_coeff = {}", msa_init[8])
        jax.debug.print("ripple debug msa_init[9]  Omegazeta3_coeff = {}", msa_init[9])
        jax.debug.print("ripple debug msa_init[10] Omegazeta4_coeff = {}", msa_init[10])
        jax.debug.print("ripple debug msa_init[11] Omegazeta5_coeff = {}", msa_init[11])
        jax.debug.print("ripple debug msa_init[12] g0               = {}", msa_init[12])
        jax.debug.print("ripple debug msa_init[13] c1               = {}", msa_init[13])
        jax.debug.print("ripple debug msa_init[14] c1_over_eta      = {}", msa_init[14])
        jax.debug.print("ripple debug msa_init[15] SAv2             = {}", msa_init[15])
        jax.debug.print("ripple debug msa_init[16] Seff             = {}", msa_init[16])
        jax.debug.print("ripple debug msa_init[17] dotS1Ln          = {}", msa_init[17])
        jax.debug.print("ripple debug msa_init[18] dotS2Ln          = {}", msa_init[18])
        jax.debug.print("ripple debug msa_init[19] S_0_norm         = {}", msa_init[19])
        jax.debug.print("ripple debug msa_init[20] psi0             = {}", msa_init[20])
        jax.debug.print("ripple debug msa_init[21] psi1             = {}", msa_init[21])
        jax.debug.print("ripple debug msa_init[22] psi2             = {}", msa_init[22])
        jax.debug.print("ripple debug msa_init[23] phiz_0           = {}", msa_init[23])
        jax.debug.print("ripple debug msa_init[24] zeta_0           = {}", msa_init[24])
        jax.debug.print("ripple debug msa_init[25] constants_L_0    = {}", msa_init[25])
        jax.debug.print("ripple debug msa_init[26] constants_L_1    = {}", msa_init[26])
        jax.debug.print("ripple debug msa_init[27] constants_L_2    = {}", msa_init[27])
        jax.debug.print("ripple debug msa_init[28] constants_L_3    = {}", msa_init[28])
        jax.debug.print("ripple debug msa_init[29] constants_L_4    = {}", msa_init[29])
        jax.debug.print("ripple debug msa_init[30] S1_norm_2        = {}", msa_init[30])
        jax.debug.print("ripple debug msa_init[31] S2_norm_2        = {}", msa_init[31])
        jax.debug.print("ripple debug  alpha_offset        = {}", alpha_offset)
        jax.debug.print("ripple debug  epsilon_offset      = {}", epsilon_offset)
        jax.debug.print("ripple debug  emm                 = {}", emm)
        jax.debug.print("ripple debug  vangles[0] (phi_z)  = {}", vangles[0])
        jax.debug.print("ripple debug  vangles[1] (zeta)   = {}", vangles[1])
        jax.debug.print("ripple debug  vangles[2] (cosbeta)= {}", vangles[2])
        jax.debug.print("ripple debug  alpha_out           = {}", alpha_out)
        jax.debug.print("ripple debug  epsilon_out         = {}", epsilon_out)
        jax.debug.print("ripple debug  cos_beta_out        = {}", cos_beta_out)
        jax.debug.print("ripple debug  Mf        = {}", Mf)

        def _save_vangles(Mf_, va0, va1, va2, a_off, e_off, a_out, e_out, emm_):
            import csv

            fname = f"ripple_debug_vangles_emm{int(emm_)}.dat"
            a_off_f = float(a_off)
            e_off_f = float(e_off)
            with open(fname, "w", newline="") as f:
                f.write(
                    "# Mf  vangles0(phi_z)  vangles1(zeta)  vangles2(cosbeta)  alpha_offset  epsilon_offset  alpha_out  epsilon_out\n"
                )
                writer = csv.writer(f, delimiter=" ")
                for row in zip(
                    Mf_,
                    va0,
                    va1,
                    va2,
                    [a_off_f] * len(Mf_),
                    [e_off_f] * len(Mf_),
                    a_out,
                    e_out,
                ):
                    writer.writerow(row)

        jax.debug.callback(
            _save_vangles,
            Mf,
            vangles[0],
            vangles[1],
            vangles[2],
            alpha_offset,
            epsilon_offset,
            alpha_out,
            epsilon_out,
            emm,
        )

    return alpha_out, epsilon_out, cos_beta_out


def compute_vangles(
    Mf,
    emm,
    eta,
    eta2,
    eta3,
    eta4,
    inveta,
    c1,
    c1_over_eta,
    SAv,
    SAv2,
    invSAv,
    invSAv2,
    constants_L_0,
    constants_L_1,
    constants_L_2,
    constants_L_3,
    constants_L_4,
    S1_norm_2,
    S2_norm_2,
    qq,
    delta_qq,
    Seff,
    dotS1Ln,
    dotS2Ln,
    S_0_norm,
    psi0,
    psi1,
    psi2,
    g0,
    Omegaz0_coeff,
    Omegaz1_coeff,
    Omegaz2_coeff,
    Omegaz3_coeff,
    Omegaz4_coeff,
    Omegaz5_coeff,
    phiz_0,
    Omegazeta0_coeff,
    Omegazeta1_coeff,
    Omegazeta2_coeff,
    Omegazeta3_coeff,
    Omegazeta4_coeff,
    Omegazeta5_coeff,
    zeta_0,
):
    v = jnp.cbrt(jnp.pi * Mf * 2.0 / emm)

    vangles = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(
        v,
        eta,
        eta2,
        eta3,
        eta4,
        inveta,
        c1,
        c1_over_eta,
        SAv,
        SAv2,
        invSAv,
        invSAv2,
        constants_L_0,
        constants_L_1,
        constants_L_2,
        constants_L_3,
        constants_L_4,
        S1_norm_2,
        S2_norm_2,
        qq,
        delta_qq,
        Seff,
        dotS1Ln,
        dotS2Ln,
        S_0_norm,
        psi0,
        psi1,
        psi2,
        g0,
        Omegaz0_coeff,
        Omegaz1_coeff,
        Omegaz2_coeff,
        Omegaz3_coeff,
        Omegaz4_coeff,
        Omegaz5_coeff,
        phiz_0,
        Omegazeta0_coeff,
        Omegazeta1_coeff,
        Omegazeta2_coeff,
        Omegazeta3_coeff,
        Omegazeta4_coeff,
        Omegazeta5_coeff,
        zeta_0,
    )
    return vangles


def compute_thetaJN_and_kappa(
    mass_1,
    mass_2,
    chi1x,
    chi1y,
    chi1z,
    chi2x,
    chi2y,
    chi2z,
    LRef,
    phiRef_In,
    inclination,
):
    mass1_2 = jnp.power(mass_1, 2)
    mass2_2 = jnp.power(mass_2, 2)

    J0x_Sf = (mass1_2) * chi1x + (mass2_2) * chi2x
    J0y_Sf = (mass1_2) * chi1y + (mass2_2) * chi2y
    J0z_Sf = (mass1_2) * chi1z + (mass2_2) * chi2z + LRef

    J0_Sf = jnp.array([J0x_Sf, J0y_Sf, J0z_Sf])
    J0 = jnp.sqrt(J0x_Sf * J0x_Sf + J0y_Sf * J0y_Sf + J0z_Sf * J0z_Sf)

    # Compress line 772 - 781
    # /* Get angle between J0 and LN (z-direction) */
    # thetaJ_Sf = jax.lax.cond(
    #     J0 < 1e-10, lambda _: 0.0, lambda _: jnp.acos(J0z_Sf / J0), operand=None
    # )

    thetaJ_Sf = jax.lax.cond(
        J0 < 1e-10,
        lambda _: 0.0,
        lambda _: jnp.acos(jnp.clip(J0z_Sf / J0, -1.0, 1.0)),
        operand=None,
    )

    # Line 783
    phiRef = phiRef_In
    # Line 785
    MAX_TOL_ATAN = 1.0e-15

    tol_condition = (jnp.abs(J0x_Sf) < MAX_TOL_ATAN) & (jnp.abs(J0y_Sf) < MAX_TOL_ATAN)
    # Compress line 797-825
    # Get azimuthal angle of J0 in the source frame
    phiJ_Sf = get_phiJ_Sf(tol_condition, J0_Sf)

    # Determine kappa via rotations, as above */
    Nx_Sf = jnp.sin(inclination) * jnp.cos((jnp.pi / 2.0) - phiRef)
    Ny_Sf = jnp.sin(inclination) * jnp.sin((jnp.pi / 2.0) - phiRef)
    Nz_Sf = jnp.cos(inclination)
    N_Sf = jnp.array([Nx_Sf, Ny_Sf, Nz_Sf])

    v_in = jnp.array([Nx_Sf, Ny_Sf, Nz_Sf])

    vout = IMRPhenomX_rotate_z(-phiJ_Sf, v_in)
    vout = IMRPhenomX_rotate_y(-thetaJ_Sf, vout)

    # /* Note difference in overall - sign w.r.t PhenomPv2 code */
    kappa = XLALSimIMRPhenomXatan2tol(vout[1], vout[0], MAX_TOL_ATAN)

    # /* Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1} */
    tmp_x = 0.0
    tmp_y = 0.0
    tmp_z = 1.0
    v_in = jnp.array([tmp_x, tmp_y, tmp_z])
    vout = IMRPhenomX_rotate_z(-phiJ_Sf, v_in)
    vout = IMRPhenomX_rotate_y(-thetaJ_Sf, vout)
    vout = IMRPhenomX_rotate_z(-kappa, vout)

    # Compress line 887 - 930
    tol_condition = (jnp.abs(vout[0]) < MAX_TOL_ATAN) & (
        jnp.abs(vout[1]) < MAX_TOL_ATAN
    )

    # Compress line 931-966
    thetaJN, Nz_Jf, Nx_Jf = thetaJN_Nz_Nx_1_6_7(N_Sf, J0_Sf, J0)

    return thetaJN, Nz_Jf, Nx_Jf, phiJ_Sf, kappa


def compute_zeta_polarization(
    mass_1,
    mass_2,
    chi1x,
    chi1y,
    chi1z,
    chi2x,
    chi2y,
    chi2z,
    LRef,
    phiRef_In,
    inclination,
    Nz_Jf,
    Nx_Jf,
    kappa,
):
    mass1_2 = jnp.power(mass_1, 2)
    mass2_2 = jnp.power(mass_2, 2)

    J0x_Sf = (mass1_2) * chi1x + (mass2_2) * chi2x
    J0y_Sf = (mass1_2) * chi1y + (mass2_2) * chi2y
    J0z_Sf = (mass1_2) * chi1z + (mass2_2) * chi2z + LRef

    J0_Sf = jnp.array([J0x_Sf, J0y_Sf, J0z_Sf])
    J0 = jnp.sqrt(J0x_Sf * J0x_Sf + J0y_Sf * J0y_Sf + J0z_Sf * J0z_Sf)

    # Compress line 772 - 781
    # /* Get angle between J0 and LN (z-direction) */
    # thetaJ_Sf = jax.lax.cond(
    #    J0 < 1e-10, lambda _: 0.0, lambda _: jnp.acos(J0z_Sf / J0), operand=None
    # )

    thetaJ_Sf = jax.lax.cond(
        J0 < 1e-10,
        lambda _: 0.0,
        lambda _: jnp.acos(jnp.clip(J0z_Sf / J0, -1.0, 1.0)),
        operand=None,
    )

    # Line 783

    # Line 785
    MAX_TOL_ATAN = 1.0e-15

    tol_condition = (jnp.abs(J0x_Sf) < MAX_TOL_ATAN) & (jnp.abs(J0y_Sf) < MAX_TOL_ATAN)
    # Compress line 797-825
    # Get azimuthal angle of J0 in the source frame
    phiJ_Sf = get_phiJ_Sf(tol_condition, J0_Sf)

    # /* Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1} */
    tmp_x = 0.0
    tmp_y = 0.0
    tmp_z = 1.0

    v_in = jnp.array([tmp_x, tmp_y, tmp_z])
    vout = IMRPhenomX_rotate_z(-phiJ_Sf, v_in)
    vout = IMRPhenomX_rotate_y(-thetaJ_Sf, vout)
    vout = IMRPhenomX_rotate_z(-kappa, vout)

    """
    Define the polarizations used. This follows the conventions adopted for IMRPhenomPv2.

    The IMRPhenomP polarizations are defined following the conventions in Arun et al (arXiv:0810.5336),
    i.e. projecting the metric onto the P, Q, N triad defining where: P = (N x J) / |N x J|.

    However, the triad X,Y,N used in LAL (the "waveframe") follows the definition in the
    NR Injection Infrastructure (Schmidt et al, arXiv:1703.01076).

    The triads differ from each other by a rotation around N by an angle zeta. We therefore need to rotate 
    the polarizations by an angle 2 zeta.
    """

    Xx_Sf = -jnp.cos(inclination) * jnp.sin(phiRef_In)
    Xy_Sf = -jnp.cos(inclination) * jnp.cos(phiRef_In)
    Xz_Sf = +jnp.sin(inclination)

    v = jnp.array([Xx_Sf, Xy_Sf, Xz_Sf])
    vout = IMRPhenomX_rotate_z(-phiJ_Sf, v)
    vout = IMRPhenomX_rotate_y(-thetaJ_Sf, vout)
    vout = IMRPhenomX_rotate_z(-kappa, vout)

    """

        The components tmp_i are now the components of X in the J frame.

        We now need the polar angle of this vector in the P, Q basis of Arun et al:

            P = (N x J) / |NxJ|

        Note, that we put N in the (pos x)z half plane of the J frame 

    """

    # Compress line 1002-1034
    PArun_Jf, QArun_Jf = PQ_Arun_1_6_7(Nx_Jf, Nz_Jf)

    # As it is line 1035-1043
    # (X . P)
    XdotPArun = (
        (vout[0] * PArun_Jf[0]) + (vout[1] * PArun_Jf[1]) + (vout[2] * PArun_Jf[2])
    )

    # (X . Q)
    XdotQArun = (
        (vout[0] * QArun_Jf[0]) + (vout[1] * QArun_Jf[1]) + (vout[2] * QArun_Jf[2])
    )

    # Now get the angle zeta
    zeta_polarization = jnp.atan2(XdotQArun, XdotPArun)
    return zeta_polarization


def compute_thetaJN_kappa_and_zeta(
    mass_1,
    mass_2,
    chi1x,
    chi1y,
    chi1z,
    chi2x,
    chi2y,
    chi2z,
    LRef,
    phiRef_In,
    inclination,
):
    """Compute thetaJN, Nz_Jf, Nx_Jf, phiJ_Sf, kappa, and zeta_polarization.

    Fuses compute_thetaJN_and_kappa with compute_zeta_polarization so that the
    total angular momentum vector J0, thetaJ_Sf, and phiJ_Sf are computed only
    once instead of twice.

    Returns:
        thetaJN, Nz_Jf, Nx_Jf, phiJ_Sf, kappa, zeta_polarization
    """
    mass1_2 = mass_1 * mass_1
    mass2_2 = mass_2 * mass_2

    J0x_Sf = mass1_2 * chi1x + mass2_2 * chi2x
    J0y_Sf = mass1_2 * chi1y + mass2_2 * chi2y
    J0z_Sf = mass1_2 * chi1z + mass2_2 * chi2z + LRef

    J0_Sf = jnp.array([J0x_Sf, J0y_Sf, J0z_Sf])
    J0 = jnp.sqrt(J0x_Sf * J0x_Sf + J0y_Sf * J0y_Sf + J0z_Sf * J0z_Sf)

    thetaJ_Sf = jax.lax.cond(
        J0 < 1e-10,
        lambda _: 0.0,
        lambda _: jnp.acos(jnp.clip(J0z_Sf / J0, -1.0, 1.0)),
        operand=None,
    )

    MAX_TOL_ATAN = 1.0e-15
    tol_condition = (jnp.abs(J0x_Sf) < MAX_TOL_ATAN) & (jnp.abs(J0y_Sf) < MAX_TOL_ATAN)
    phiJ_Sf = get_phiJ_Sf(tol_condition, J0_Sf)

    # --- Part 1: kappa and thetaJN (from compute_thetaJN_and_kappa) ---
    Nx_Sf = jnp.sin(inclination) * jnp.cos((jnp.pi / 2.0) - phiRef_In)
    Ny_Sf = jnp.sin(inclination) * jnp.sin((jnp.pi / 2.0) - phiRef_In)
    Nz_Sf = jnp.cos(inclination)
    N_Sf = jnp.array([Nx_Sf, Ny_Sf, Nz_Sf])

    vout = IMRPhenomX_rotate_z(-phiJ_Sf, N_Sf)
    vout = IMRPhenomX_rotate_y(-thetaJ_Sf, vout)
    kappa = XLALSimIMRPhenomXatan2tol(vout[1], vout[0], MAX_TOL_ATAN)

    thetaJN, Nz_Jf, Nx_Jf = thetaJN_Nz_Nx_1_6_7(N_Sf, J0_Sf, J0)

    # --- Part 2: zeta_polarization (from compute_zeta_polarization) ---
    Xx_Sf = -jnp.cos(inclination) * jnp.sin(phiRef_In)
    Xy_Sf = -jnp.cos(inclination) * jnp.cos(phiRef_In)
    Xz_Sf = +jnp.sin(inclination)
    v = jnp.array([Xx_Sf, Xy_Sf, Xz_Sf])
    vout = IMRPhenomX_rotate_z(-phiJ_Sf, v)
    vout = IMRPhenomX_rotate_y(-thetaJ_Sf, vout)
    vout = IMRPhenomX_rotate_z(-kappa, vout)

    PArun_Jf, QArun_Jf = PQ_Arun_1_6_7(Nx_Jf, Nz_Jf)
    XdotPArun = vout[0] * PArun_Jf[0] + vout[1] * PArun_Jf[1] + vout[2] * PArun_Jf[2]
    XdotQArun = vout[0] * QArun_Jf[0] + vout[1] * QArun_Jf[1] + vout[2] * QArun_Jf[2]
    zeta_polarization = jnp.atan2(XdotQArun, XdotPArun)

    return thetaJN, Nz_Jf, Nx_Jf, phiJ_Sf, kappa, zeta_polarization


def PQ_Arun_1_6_7(Nx_Jf, Nz_Jf):
    # Get polar angle of X vector in J frame in the P,Q basis of Arun et al
    PArunx_Jf = Nz_Jf
    PAruny_Jf = 0.0
    PArunz_Jf = -Nx_Jf

    QArunx_Jf = 0.0
    QAruny_Jf = 1.0
    QArunz_Jf = 0.0

    return jnp.array([PArunx_Jf, PAruny_Jf, PArunz_Jf]), jnp.array(
        [QArunx_Jf, QAruny_Jf, QArunz_Jf]
    )


def thetaJN_Nz_Nx_1_6_7(N_Sf, J0_Sf, J0):
    # Line 957-962

    J0dotN = (J0_Sf[0] * N_Sf[0]) + (J0_Sf[1] * N_Sf[1]) + (J0_Sf[2] * N_Sf[2])
    cos_thetaJN = jax.lax.cond(
        J0 < 1e-10,
        lambda _: 1.0,
        lambda _: jnp.clip(J0dotN / J0, -1.0, 1.0),
        operand=None,
    )
    thetaJN = jnp.acos(cos_thetaJN)
    Nz_Jf = jnp.cos(thetaJN)
    Nx_Jf = jnp.sin(thetaJN)

    return thetaJN, Nz_Jf, Nx_Jf


def get_phiJ_Sf(tol_condition, J0_Sf):
    """
    Compute phiJ_Sf based on tolerance condition.

    Since convention_condition is always False, this simplifies to:
    - If tol_condition is True: return 0.0
    - Otherwise: return atan2(J0_Sf[1], J0_Sf[0])
    """
    phiJ_Sf = jax.lax.cond(
        tol_condition,
        lambda _: 0.0,
        lambda _: jnp.atan2(J0_Sf[1], J0_Sf[0]),
        operand=None,
    )

    return phiJ_Sf


def Get_alphaepsilon_atfref(
    mprime,
    piM,
    fRef,
    alpha0,
    epsilon0,
    eta,
    eta2,
    eta3,
    eta4,
    inveta,
    c1,
    c1_over_eta,
    SAv,
    SAv2,
    invSAv,
    invSAv2,
    constants_L_0,
    constants_L_1,
    constants_L_2,
    constants_L_3,
    constants_L_4,
    S1_norm_2,
    S2_norm_2,
    qq,
    delta_qq,
    Seff,
    dotS1Ln,
    dotS2Ln,
    S_0_norm,
    psi0,
    psi1,
    psi2,
    g0,
    Omegaz0_coeff,
    Omegaz1_coeff,
    Omegaz2_coeff,
    Omegaz3_coeff,
    Omegaz4_coeff,
    Omegaz5_coeff,
    phiz_0,
    Omegazeta0_coeff,
    Omegazeta1_coeff,
    Omegazeta2_coeff,
    Omegazeta3_coeff,
    Omegazeta4_coeff,
    Omegazeta5_coeff,
    zeta_0,
):
    omega_ref = piM * fRef * 2 / mprime

    alpha_offset, epsilon_offset = Get_alphaepsilon_atfref_pflag_true(
        omega_ref,
        alpha0,
        epsilon0,
        eta,
        eta2,
        eta3,
        eta4,
        inveta,
        c1,
        c1_over_eta,
        SAv,
        SAv2,
        invSAv,
        invSAv2,
        constants_L_0,
        constants_L_1,
        constants_L_2,
        constants_L_3,
        constants_L_4,
        S1_norm_2,
        S2_norm_2,
        qq,
        delta_qq,
        Seff,
        dotS1Ln,
        dotS2Ln,
        S_0_norm,
        psi0,
        psi1,
        psi2,
        g0,
        Omegaz0_coeff,
        Omegaz1_coeff,
        Omegaz2_coeff,
        Omegaz3_coeff,
        Omegaz4_coeff,
        Omegaz5_coeff,
        phiz_0,
        Omegazeta0_coeff,
        Omegazeta1_coeff,
        Omegazeta2_coeff,
        Omegazeta3_coeff,
        Omegazeta4_coeff,
        Omegazeta5_coeff,
        zeta_0,
    )

    return alpha_offset, epsilon_offset


def Get_alphaepsilon_atfref_pflag_true(
    omega_ref,
    alpha0,
    epsilon0,
    eta,
    eta2,
    eta3,
    eta4,
    inveta,
    c1,
    c1_over_eta,
    SAv,
    SAv2,
    invSAv,
    invSAv2,
    constants_L_0,
    constants_L_1,
    constants_L_2,
    constants_L_3,
    constants_L_4,
    S1_norm_2,
    S2_norm_2,
    qq,
    delta_qq,
    Seff,
    dotS1Ln,
    dotS2Ln,
    S_0_norm,
    psi0,
    psi1,
    psi2,
    g0,
    Omegaz0_coeff,
    Omegaz1_coeff,
    Omegaz2_coeff,
    Omegaz3_coeff,
    Omegaz4_coeff,
    Omegaz5_coeff,
    phiz_0,
    Omegazeta0_coeff,
    Omegazeta1_coeff,
    Omegazeta2_coeff,
    Omegazeta3_coeff,
    Omegazeta4_coeff,
    Omegazeta5_coeff,
    zeta_0,
):
    v = jnp.cbrt(omega_ref)
    vangles = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(
        v,
        eta,
        eta2,
        eta3,
        eta4,
        inveta,
        c1,
        c1_over_eta,
        SAv,
        SAv2,
        invSAv,
        invSAv2,
        constants_L_0,
        constants_L_1,
        constants_L_2,
        constants_L_3,
        constants_L_4,
        S1_norm_2,
        S2_norm_2,
        qq,
        delta_qq,
        Seff,
        dotS1Ln,
        dotS2Ln,
        S_0_norm,
        psi0,
        psi1,
        psi2,
        g0,
        Omegaz0_coeff,
        Omegaz1_coeff,
        Omegaz2_coeff,
        Omegaz3_coeff,
        Omegaz4_coeff,
        Omegaz5_coeff,
        phiz_0,
        Omegazeta0_coeff,
        Omegazeta1_coeff,
        Omegazeta2_coeff,
        Omegazeta3_coeff,
        Omegazeta4_coeff,
        Omegazeta5_coeff,
        zeta_0,
    )

    alpha_offset = vangles[0] - alpha0
    epsilon_offset = vangles[1] - epsilon0
    return alpha_offset, epsilon_offset


def flag_222_223_twoPN_non_spinning_orbitan_angular_momentum(
    eta, eta2, chi1L, chi2L, delta, power_of_lalpi_2
):
    L0 = 1.0
    L1 = 0.0
    L2 = 3.0 / 2.0 + eta / 6.0
    L3 = (
        -7 * (chi1L + chi2L + chi1L * delta - chi2L * delta) + 5 * (chi1L + chi2L) * eta
    ) / 6.0
    L4 = (81 + (-57 + eta) * eta) / 24.0
    L5 = (
        -1650 * (chi1L + chi2L + chi1L * delta - chi2L * delta)
        + 1336 * (chi1L + chi2L) * eta
        + 511 * (chi1L - chi2L) * delta * eta
        + 28 * (chi1L + chi2L) * eta2
    ) / 600.0
    L6 = (
        10935 + eta * (-62001 + 1674 * eta + 7 * eta2 + 2214 * power_of_lalpi_2)
    ) / 1296.0
    L7 = 0.0
    L8 = 0.0
    L8L = 0.0
    return jnp.array([L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L])


def set_epsilon0(phenom_xp_convention, phiJ_Sf):
    epsilon0 = jax.lax.cond(
        jnp.isin(phenom_xp_convention, jnp.array([1, 6])),
        lambda _: phiJ_Sf - jnp.pi,
        lambda _: 0.0,
        operand=None,
    )

    return epsilon0


# ---------------------------------------------------------------------------
# MSA precession setup helpers – hoist the emm-invariant work out of the
# per-mode vmap to avoid recomputing it five times per waveform call.
# ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MSAPrecessionSetup:
    """Pre-computed MSA precession constants that are invariant to emm and Mf.

    Compute once via compute_msa_precession_setup and pass into
    compute_evolved_spin_given_setup inside any jax.vmap over modes.
    """

    eta: Float
    eta2: Float
    eta3: Float
    eta4: Float
    inveta: Float
    SAv: Float
    SAv2: Float
    invSAv: Float
    invSAv2: Float
    qq: Float
    delta_qq: Float
    alpha_offset: Float
    epsilon_offset: Float
    c1: Float
    c1_over_eta: Float
    Seff: Float
    dotS1Ln: Float
    dotS2Ln: Float
    S_0_norm: Float
    psi0: Float
    psi1: Float
    psi2: Float
    g0: Float
    Omegaz0_coeff: Float
    Omegaz1_coeff: Float
    Omegaz2_coeff: Float
    Omegaz3_coeff: Float
    Omegaz4_coeff: Float
    Omegaz5_coeff: Float
    phiz_0: Float
    Omegazeta0_coeff: Float
    Omegazeta1_coeff: Float
    Omegazeta2_coeff: Float
    Omegazeta3_coeff: Float
    Omegazeta4_coeff: Float
    Omegazeta5_coeff: Float
    zeta_0: Float
    constants_L_0: Float
    constants_L_1: Float
    constants_L_2: Float
    constants_L_3: Float
    constants_L_4: Float
    S1_norm_2: Float
    S2_norm_2: Float


def compute_msa_precession_setup(
    mass_1,
    mass_2,
    chi1x,
    chi1y,
    chi1z,
    chi2x,
    chi2y,
    chi2z,
    reference_frequency,
    kappa,
    phiJ_Sf,
) -> MSAPrecessionSetup:
    """Compute MSA precession constants invariant to emm and Mf.

    Replaces the emm-invariant prefix of compute_evolved_spin_using_msa.
    Call once per waveform evaluation; pass the result to
    compute_evolved_spin_given_setup inside any vmap over modes.
    """
    phenom_xp_convention = 1
    eta = mass_1 * mass_2 / jnp.power(mass_1 + mass_2, 2)
    Msec = (mass_1 + mass_2) * MTSUN
    piM = jnp.pi * Msec

    msa_init = IMRPhenomX_Initialize_MSA_System(
        mass_1=mass_1,
        mass_2=mass_2,
        chi1x=chi1x,
        chi1y=chi1y,
        chi1z=chi1z,
        chi2x=chi2x,
        chi2y=chi2y,
        chi2z=chi2z,
        reference_frequency=reference_frequency,
    )

    alpha0 = jnp.pi - kappa
    epsilon0 = set_epsilon0(phenom_xp_convention, phiJ_Sf)
    mprime_for_offset = 2

    eta2 = jnp.power(eta, 2)
    eta3 = jnp.power(eta, 3)
    eta4 = jnp.power(eta, 4)
    inveta = jnp.power(eta, -1)

    SAv2 = msa_init[15]
    SAv = jnp.sqrt(SAv2)
    invSAv = jnp.power(SAv, -1)
    invSAv2 = jnp.power(SAv2, -1)

    qq = mass_2 / mass_1
    delta_qq = (1 - qq) / (1 + qq)

    alpha_offset, epsilon_offset = Get_alphaepsilon_atfref(
        mprime_for_offset,
        piM,
        reference_frequency,
        alpha0,
        epsilon0,
        eta=eta,
        eta2=eta2,
        eta3=eta3,
        eta4=eta4,
        inveta=inveta,
        SAv=SAv,
        SAv2=SAv2,
        invSAv=invSAv,
        invSAv2=invSAv2,
        qq=qq,
        delta_qq=delta_qq,
        Omegaz0_coeff=msa_init[0],
        Omegaz1_coeff=msa_init[1],
        Omegaz2_coeff=msa_init[2],
        Omegaz3_coeff=msa_init[3],
        Omegaz4_coeff=msa_init[4],
        Omegaz5_coeff=msa_init[5],
        Omegazeta0_coeff=msa_init[6],
        Omegazeta1_coeff=msa_init[7],
        Omegazeta2_coeff=msa_init[8],
        Omegazeta3_coeff=msa_init[9],
        Omegazeta4_coeff=msa_init[10],
        Omegazeta5_coeff=msa_init[11],
        g0=msa_init[12],
        c1=msa_init[13],
        c1_over_eta=msa_init[14],
        Seff=msa_init[16],
        dotS1Ln=msa_init[17],
        dotS2Ln=msa_init[18],
        S_0_norm=msa_init[19],
        psi0=msa_init[20],
        psi1=msa_init[21],
        psi2=msa_init[22],
        phiz_0=msa_init[23],
        zeta_0=msa_init[24],
        constants_L_0=msa_init[25],
        constants_L_1=msa_init[26],
        constants_L_2=msa_init[27],
        constants_L_3=msa_init[28],
        constants_L_4=msa_init[29],
        S1_norm_2=msa_init[30],
        S2_norm_2=msa_init[31],
    )

    return MSAPrecessionSetup(
        eta=eta,
        eta2=eta2,
        eta3=eta3,
        eta4=eta4,
        inveta=inveta,
        SAv=SAv,
        SAv2=SAv2,
        invSAv=invSAv,
        invSAv2=invSAv2,
        qq=qq,
        delta_qq=delta_qq,
        alpha_offset=alpha_offset,
        epsilon_offset=epsilon_offset,
        c1=msa_init[13],
        c1_over_eta=msa_init[14],
        Seff=msa_init[16],
        dotS1Ln=msa_init[17],
        dotS2Ln=msa_init[18],
        S_0_norm=msa_init[19],
        psi0=msa_init[20],
        psi1=msa_init[21],
        psi2=msa_init[22],
        g0=msa_init[12],
        Omegaz0_coeff=msa_init[0],
        Omegaz1_coeff=msa_init[1],
        Omegaz2_coeff=msa_init[2],
        Omegaz3_coeff=msa_init[3],
        Omegaz4_coeff=msa_init[4],
        Omegaz5_coeff=msa_init[5],
        phiz_0=msa_init[23],
        Omegazeta0_coeff=msa_init[6],
        Omegazeta1_coeff=msa_init[7],
        Omegazeta2_coeff=msa_init[8],
        Omegazeta3_coeff=msa_init[9],
        Omegazeta4_coeff=msa_init[10],
        Omegazeta5_coeff=msa_init[11],
        zeta_0=msa_init[24],
        constants_L_0=msa_init[25],
        constants_L_1=msa_init[26],
        constants_L_2=msa_init[27],
        constants_L_3=msa_init[28],
        constants_L_4=msa_init[29],
        S1_norm_2=msa_init[30],
        S2_norm_2=msa_init[31],
    )


def compute_evolved_spin_given_setup(Mf, emm, setup: MSAPrecessionSetup):
    """Compute precession angles for mode emm using pre-computed MSA setup.

    This is the emm-dependent part of compute_evolved_spin_using_msa.
    Suitable for use inside jax.vmap over harmonic modes.

    Args:
        Mf: Dimensionless frequency array.
        emm: Azimuthal mode number m.
        setup: Pre-computed MSAPrecessionSetup from compute_msa_precession_setup.

    Returns:
        alpha, epsilon, cos_beta arrays over Mf.
    """
    inspiral_mask = Mf < 0.3
    vangles = compute_vangles(
        Mf=Mf,
        emm=emm,
        eta=setup.eta,
        eta2=setup.eta2,
        eta3=setup.eta3,
        eta4=setup.eta4,
        inveta=setup.inveta,
        c1=setup.c1,
        c1_over_eta=setup.c1_over_eta,
        SAv=setup.SAv,
        SAv2=setup.SAv2,
        invSAv=setup.invSAv,
        invSAv2=setup.invSAv2,
        constants_L_0=setup.constants_L_0,
        constants_L_1=setup.constants_L_1,
        constants_L_2=setup.constants_L_2,
        constants_L_3=setup.constants_L_3,
        constants_L_4=setup.constants_L_4,
        S1_norm_2=setup.S1_norm_2,
        S2_norm_2=setup.S2_norm_2,
        qq=setup.qq,
        delta_qq=setup.delta_qq,
        Seff=setup.Seff,
        dotS1Ln=setup.dotS1Ln,
        dotS2Ln=setup.dotS2Ln,
        S_0_norm=setup.S_0_norm,
        psi0=setup.psi0,
        psi1=setup.psi1,
        psi2=setup.psi2,
        g0=setup.g0,
        Omegaz0_coeff=setup.Omegaz0_coeff,
        Omegaz1_coeff=setup.Omegaz1_coeff,
        Omegaz2_coeff=setup.Omegaz2_coeff,
        Omegaz3_coeff=setup.Omegaz3_coeff,
        Omegaz4_coeff=setup.Omegaz4_coeff,
        Omegaz5_coeff=setup.Omegaz5_coeff,
        phiz_0=setup.phiz_0,
        Omegazeta0_coeff=setup.Omegazeta0_coeff,
        Omegazeta1_coeff=setup.Omegazeta1_coeff,
        Omegazeta2_coeff=setup.Omegazeta2_coeff,
        Omegazeta3_coeff=setup.Omegazeta3_coeff,
        Omegazeta4_coeff=setup.Omegazeta4_coeff,
        Omegazeta5_coeff=setup.Omegazeta5_coeff,
        zeta_0=setup.zeta_0,
    )
    alpha_out: Float = jnp.where(inspiral_mask, vangles[0] - setup.alpha_offset, 0.0)
    epsilon_out: Float = jnp.where(
        inspiral_mask, vangles[1] - setup.epsilon_offset, 0.0
    )
    cos_beta_out: Float = jnp.where(inspiral_mask, vangles[2], 0.0)
    return alpha_out, epsilon_out, cos_beta_out


def XLALSimIMRPhenomXUtilsHztoMf(fHz: Float, Mtot_Msun: Float) -> Float:
    """
    Convert frequency from Hz to geometric units (Mf).

    Args:
        fHz (Float): Frequency in Hz
        Mtot_Msun (Float): Total mass in solar masses

    Returns:
        Float: Geometric frequency Mf
    """
    return fHz * Mtot_Msun * MTSUN


def XLALSimIMRPhenomXUtilsMftoHz(Mf: Float, Mtot_Msun: Float) -> Float:
    """
    Convert frequency from geometric units (Mf) to Hz.

    Args:
        Mf (Float): Geometric frequency
        Mtot_Msun (Float): Total mass in solar masses

    Returns:
        Float: Frequency in Hz
    """
    return Mf / (Mtot_Msun * MTSUN)


def IMRPhenomX_rotate_z(angle, v):
    """
    Rotate a 3D vector v = (vx, vy, vz) about the z-axis by given angle.
    Args:
        angle: scalar angle in radians (JAX array or float)
        v: array-like of shape (3,) representing [vx, vy, vz]
    Returns:
        rotated vector as a JAX array of shape (3,)
    """
    cosa = jnp.cos(angle)
    sina = jnp.sin(angle)
    vx = v[0]
    vy = v[1]
    vz = v[2]

    vx_rot = vx * cosa - vy * sina
    vy_rot = vx * sina + vy * cosa
    vz_rot = vz  # unchanged

    return jnp.array([vx_rot, vy_rot, vz_rot])


def IMRPhenomX_rotate_y(angle, v):
    """
    Rotate a 3D vector v = (vx, vy, vz) about the y-axis by a given angle.
    Args:
        angle: scalar angle in radians (JAX array or float)
        v: array-like of shape (3,) representing [vx, vy, vz]
    Returns:
        rotated vector as a JAX array of shape (3,)
    """

    cosa = jnp.cos(angle)
    sina = jnp.sin(angle)
    vx = v[0]
    vy = v[1]
    vz = v[2]

    vx_rot = vx * cosa + vz * sina
    vy_rot = vy  # unchanged
    vz_rot = -vx * sina + vz * cosa

    return jnp.array([vx_rot, vy_rot, vz_rot])


def XLALSimIMRPhenomXLPNAnsatz(
    v: Float,
    LNorm: Float,
    L0: Float,
    L1: Float,
    L2: Float,
    L3: Float,
    L4: Float,
    L5: Float,
    L6: Float,
    L7: Float,
    L8: Float,
    L8L: Float,
) -> Float:
    """
    Compute orbital angular momentum using post-Newtonian expansion

    Args:
        v: Input velocity (Float)
        LNorm: Orbital angular momentum normalization (Float)
        L0: Newtonian orbital angular momentum (Float)
        L1: 0.5PN Orbital angular momentum (Float)
        L2: 1.0PN Orbital angular momentum (Float)
        L3: 1.5PN Orbital angular momentum (Float)
        L4: 2.0PN Orbital angular momentum (Float)
        L5: 2.5PN Orbital angular momentum (Float)
        L6: 3.0PN Orbital angular momentum (Float)
        L7: 3.5PN Orbital angular momentum (Float)
        L8: 4.0PN Orbital angular momentum (Float)
        L8L: 4.0PN logarithmic orbital angular momentum term (Float)

    Returns: Float: Orbital angular momentum
    """

    x = v * v
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    sqx = jnp.sqrt(x)

    # Here LN is the Newtonian pre-factor: LN = \eta / \sqrt{x} :
    # L = L_N \sum_a L_a x^{a/2}
    #   = L_N [ L0 + L1 x^{1/2} + L2 x^{2/2} + L3 x^{3/2} + ... ]

    return LNorm * (
        L0
        + L1 * sqx
        + L2 * x
        + L3 * (x * sqx)
        + L4 * x2
        + L5 * (x2 * sqx)
        + L6 * x3
        + L7 * (x3 * sqx)
        + L8 * x4
        + L8L * x4 * jnp.log(x)
    )


def XLALSimIMRPhenomXatan2tol(vy, vx, tol):
    cond = (jnp.abs(vy) < tol) & (jnp.abs(vx) < tol)
    return jax.lax.cond(
        cond,
        lambda _: 0.0,
        lambda _: jnp.atan2(vy, vx),
        operand=None,
    )
