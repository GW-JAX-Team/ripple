import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ripplegw.constants import MTSUN
from ripplegw.typing import FloatLike
from ripplegw.waveforms.CBC.IMRPhenomX.initialize_MSA_system import (
    IMRPhenomX_Initialize_MSA_System,
    IMRPhenomX_Return_phi_zeta_costhetaL_MSA,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CommonConstants:
    sqrt2: FloatLike = 1.4142135623730951
    sqrt5: FloatLike = 2.23606797749978981
    sqrt6: FloatLike = 2.44948974278317788
    sqrt7: FloatLike = 2.64575131106459072
    sqrt10: FloatLike = 3.16227766016838
    sqrt14: FloatLike = 3.74165738677394133
    sqrt15: FloatLike = 3.87298334620741702
    sqrt70: FloatLike = 8.36660026534075563
    sqrt30: FloatLike = 5.477225575051661
    sqrt2p5: FloatLike = 1.58113883008419
    log16: FloatLike = 2.772588722239781
    power_of_lalpi_2: FloatLike = 9.869604401089358
    MAX_TOL_ATAN: FloatLike = 1.0e-15


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


def compute_thetaJN_kappa_and_zeta(
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    LRef: FloatLike,
    phiRef_In: FloatLike,
    inclination: FloatLike,
) -> tuple[FloatLike, FloatLike, FloatLike, FloatLike, FloatLike, FloatLike]:
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
    eta: FloatLike,
    eta2: FloatLike,
    chi1L: FloatLike,
    chi2L: FloatLike,
    delta: FloatLike,
    power_of_lalpi_2: FloatLike,
) -> Float[Array, "10"]:
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

    eta: FloatLike
    eta2: FloatLike
    eta3: FloatLike
    eta4: FloatLike
    inveta: FloatLike
    SAv: FloatLike
    SAv2: FloatLike
    invSAv: FloatLike
    invSAv2: FloatLike
    qq: FloatLike
    delta_qq: FloatLike
    alpha_offset: FloatLike
    epsilon_offset: FloatLike
    c1: FloatLike
    c1_over_eta: FloatLike
    Seff: FloatLike
    dotS1Ln: FloatLike
    dotS2Ln: FloatLike
    S_0_norm: FloatLike
    psi0: FloatLike
    psi1: FloatLike
    psi2: FloatLike
    g0: FloatLike
    Omegaz0_coeff: FloatLike
    Omegaz1_coeff: FloatLike
    Omegaz2_coeff: FloatLike
    Omegaz3_coeff: FloatLike
    Omegaz4_coeff: FloatLike
    Omegaz5_coeff: FloatLike
    phiz_0: FloatLike
    Omegazeta0_coeff: FloatLike
    Omegazeta1_coeff: FloatLike
    Omegazeta2_coeff: FloatLike
    Omegazeta3_coeff: FloatLike
    Omegazeta4_coeff: FloatLike
    Omegazeta5_coeff: FloatLike
    zeta_0: FloatLike
    constants_L_0: FloatLike
    constants_L_1: FloatLike
    constants_L_2: FloatLike
    constants_L_3: FloatLike
    constants_L_4: FloatLike
    S1_norm_2: FloatLike
    S2_norm_2: FloatLike


def compute_msa_precession_setup(
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    reference_frequency: float,
    kappa: FloatLike,
    phiJ_Sf: FloatLike,
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


def compute_evolved_spin_given_setup(
    Mf: Float[Array, " n_freq"] | FloatLike, emm: int | Array, setup: MSAPrecessionSetup
) -> tuple[Float[Array, " n_freq"], Float[Array, " n_freq"], Float[Array, " n_freq"]]:
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
    alpha_out: FloatLike = jnp.where(
        inspiral_mask, vangles[0] - setup.alpha_offset, 0.0
    )
    epsilon_out: FloatLike = jnp.where(
        inspiral_mask, vangles[1] - setup.epsilon_offset, 0.0
    )
    cos_beta_out: FloatLike = jnp.where(inspiral_mask, vangles[2], 0.0)
    return alpha_out, epsilon_out, cos_beta_out


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
    v: FloatLike,
    LNorm: FloatLike,
    L0: FloatLike,
    L1: FloatLike,
    L2: FloatLike,
    L3: FloatLike,
    L4: FloatLike,
    L5: FloatLike,
    L6: FloatLike,
    L7: FloatLike,
    L8: FloatLike,
    L8L: FloatLike,
) -> FloatLike:
    """
    Compute orbital angular momentum using post-Newtonian expansion

    Args:
        v: Input velocity (FloatLike)
        LNorm: Orbital angular momentum normalization (FloatLike)
        L0: Newtonian orbital angular momentum (FloatLike)
        L1: 0.5PN Orbital angular momentum (FloatLike)
        L2: 1.0PN Orbital angular momentum (FloatLike)
        L3: 1.5PN Orbital angular momentum (FloatLike)
        L4: 2.0PN Orbital angular momentum (FloatLike)
        L5: 2.5PN Orbital angular momentum (FloatLike)
        L6: 3.0PN Orbital angular momentum (FloatLike)
        L7: 3.5PN Orbital angular momentum (FloatLike)
        L8: 4.0PN Orbital angular momentum (FloatLike)
        L8L: 4.0PN logarithmic orbital angular momentum term (FloatLike)

    Returns: FloatLike: Orbital angular momentum
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
