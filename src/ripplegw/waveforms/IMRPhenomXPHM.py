import jax
from jax import jit
import jax.numpy as jnp
from .IMRPhenomD_QNMdata import QNMData_a, QNMData_fRD, QNMData_fdamp
from ..constants import C, PI, MSUN, MTSUN_SI, MTSUN, MRSUN, MPC
from ..typing import Array
from .spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)
from dataclasses import dataclass
from . import LALSimIMRPhenomX_precession as pPrec

from .LALSimIMRPhenomUtils import XLALSimPhenomUtilsChiP

# Some pre-XPHM ripple code
from .IMRPhenomD_utils import (
    EradRational0815,
    get_coeffs,
    get_transition_frequencies,
    get_transition_frequencies_from_fRD_fdamp,
)
from .IMRPhenomD import Phase as IMRPhenomD_Phase
from .IMRPhenomD import IMRPhenDAmplitude_NoCut
from .IMRPhenomD import get_IIb_raw_phase
from .IMRPhenomPv2_utils import FinalSpin0815

# from .LALSimIMRPhenomX_precession import Get_alpha_epsilon_offset


uGpc = 3.085677581491367278913937957796471611e25
# 3.085677581491367278913937957796471611e25 # meters
GMsun_over_c2 = MTSUN_SI * C
# 1.476625061404649406193430731479084713e3 # meters
GMsun_over_c2_Gpc = GMsun_over_c2 / uGpc


# MTSUN_SI = 4.925491025543575903411922162094833998e-6

# TF2 coefficient array indices (for JIT-compatible array-based storage)
TF2_ZERO = 0
TF2_ONE = 1
TF2_TWO = 2
TF2_THREE = 3
TF2_FOUR = 4
TF2_FIVE = 5
TF2_FIVE_LOG = 6
TF2_SIX = 7
TF2_SIX_LOG = 8
TF2_SEVEN = 9
TF2_NUM_COEFFS = 10

# PhiInsp coefficient array indices (for JIT-compatible array-based storage)
PHI_INITIAL_PHASING = 0
PHI_TWO_THIRDS = 1
PHI_THIRD = 2
PHI_THIRD_LOG = 3
PHI_LOG = 4
PHI_MIN_THIRD = 5
PHI_MIN_TWO_THIRDS = 6
PHI_MIN_ONE = 7
PHI_MIN_FOUR_THIRDS = 8
PHI_MIN_FIVE_THIRDS = 9
PHI_ONE = 10
PHI_FOUR_THIRDS = 11
PHI_FIVE_THIRDS = 12
PHI_TWO = 13
PHI_NUM_COEFFS = 14

# Amplitude coefficient array indices (for JIT-compatible array-based storage)
AMP_TWO_THIRDS = 0
AMP_ONE = 1
AMP_FOUR_THIRDS = 2
AMP_FIVE_THIRDS = 3
AMP_TWO = 4
AMP_SEVEN_THIRDS = 5
AMP_EIGHT_THIRDS = 6
AMP_THREE = 7
AMP_NUM_COEFFS = 8

# Alpha coefficient array indices (for JIT-compatible array-based storage)
ALPHA_1 = 0
ALPHA_2 = 1
ALPHA_3 = 2
ALPHA_4 = 3
ALPHA_5 = 4
ALPHA_NUM_COEFFS = 5

# Beta coefficient array indices (for JIT-compatible array-based storage)
BETA_1 = 0
BETA_2 = 1
BETA_3 = 2
BETA_NUM_COEFFS = 3

# Sigma coefficient array indices (for JIT-compatible array-based storage)
SIGMA_1 = 0
SIGMA_2 = 1
SIGMA_3 = 2
SIGMA_4 = 3
SIGMA_NUM_COEFFS = 4

# Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
AMP_fJoin_INS = 0.014
# Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
PHI_fJoin_INS = 0.018


# Phase shift due to leading order complex amplitude
# [L.Blancet, arXiv:1310.1528 (Sec. 9.5)]
# "Spherical hrmonic modes for numerical relativity"
# List of phase shifts: the index is the azimuthal number m
CSHIFT = jnp.array([0.0, PI / 2.0, 0.0, -PI / 2.0, PI, PI / 2.0, 0.0])


@jit
@jit
def generate_xphm(
    mass_1,
    mass_2,
    chi1x,
    chi1y,
    chi1z,
    chi2x,
    chi2y,
    chi2z,
    distance,  # in Mpc
    inclination,
    phi0,
    frequency_array,
    reference_frequency,
):
    Mf = XLALSimIMRPhenomXUtilsHztoMf(frequency_array, mass_1 + mass_2)

    m1_SI = mass_1 * MSUN
    m2_SI = mass_2 * MSUN
    Mtot = mass_1 + mass_2

    # Overall amplitude prefactor from LAL's XLALSimPhenomUtilsFDamp0:
    # amp0 = Mtot * MRSUN * Mtot * MTSUN / distance
    # where Mtot is in solar masses and distance is in meters
    dist_m = distance * MPC  # distance in meters
    amp0 = Mtot * MRSUN * Mtot * MTSUN_SI / dist_m

    extra_params = {"ModeArray": jnp.array([[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]])}

    hlm = XLALSimIMRPhenomHMGethlmModes(
        frequency_array,
        m1_SI,
        m2_SI,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        phi0,
        frequency_array[1] - frequency_array[0],
        reference_frequency,
        extra_params,
    )

    ells = extra_params["ModeArray"][:, 0]
    minus1l = jnp.where(ells % 2 != 0, -1, 1)
    hlms = minus1l[:, None] * hlm * amp0

    hp, hc = twistup(
        Mf,
        mass_1,
        mass_2,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        phi0,
        inclination,
        reference_frequency,
        hlms,
    )

    return hp, hc


@jit
def twistup(
    Mf,
    mass_1,
    mass_2,
    chi1x,
    chi1y,
    chi1z,
    chi2x,
    chi2y,
    chi2z,
    phiRef_In,
    inclination,
    reference_frequency,
    hlm,
):
    "Copy of lalsimulation IMRPhenomXPHMTwistUp"
    "Function to twist up hlms"

    # Check if we are using multibanding for angles.
    # Default in lalsimulation is True but I will force it to False

    # Check PrecVersion
    # Available options 101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321, 330
    # I will use 223 which is default in lalsimulation

    # Modes 21, 22, 32, 33, 43, 44 in that order

    bigM = 1
    eta = mass_1 * mass_2 / jnp.power(mass_1 + mass_2, 2)
    eta2 = jnp.power(eta, 2)
    chi1L = chi1z
    chi2L = chi2z
    total_mass = mass_1 + mass_2

    mass_1_fraction = mass_1 / total_mass
    mass_2_fraction = mass_2 / total_mass

    delta = mass_1_fraction - mass_2_fraction

    orbital_angular_momentum = (
        pPrec.flag_222_223_twoPN_non_spinning_orbitan_angular_momentum(
            eta, eta2, chi1L, chi2L, delta, jnp.power(jnp.pi, 2)
        )
    )
    Msec = (mass_1 + mass_2) * MTSUN
    piM = jnp.pi * Msec
    v_ref = jnp.cbrt(piM * reference_frequency)
    LRef = (
        bigM
        * bigM
        * pPrec.XLALSimIMRPhenomXLPNAnsatz(
            v_ref,
            eta / v_ref,
            orbital_angular_momentum[0],
            orbital_angular_momentum[1],
            orbital_angular_momentum[2],
            orbital_angular_momentum[3],
            orbital_angular_momentum[4],
            orbital_angular_momentum[5],
            orbital_angular_momentum[6],
            orbital_angular_momentum[7],
            orbital_angular_momentum[8],
            orbital_angular_momentum[9],
        )
    )

    theta_JN, Nz_Jf, Nx_Jf, phiJ_Sf, kappa = pPrec.compute_thetaJN_and_kappa(
        mass_1_fraction,
        mass_2_fraction,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        LRef,
        phiRef_In,
        inclination,
    )

    zeta_polarisations = pPrec.compute_zeta_polarization(
        mass_1_fraction,
        mass_2_fraction,
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
    )

    def compute_twist_for_mode(mode_idx):
        # mode_idx: 0->21, 1->22, 2->32, 3->33, 4->44
        emms = jnp.array([1, 2, 2, 3, 4])

        emm = emms[mode_idx]

        alpha, epsilon, cos_beta = pPrec.compute_evolved_spin_using_msa(
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
        )

        beta = jnp.arccos(cos_beta)

        def _save_angles(emm_, Mf_, alpha_, epsilon_, beta_):
            import numpy as np

            np.savetxt(
                f"ripple_angles_{int(emm_)}.dat",
                np.column_stack([Mf_, alpha_, epsilon_, beta_]),
                header="Mf alpha epsilon beta",
            )

        jax.debug.callback(_save_angles, emm, Mf, alpha, epsilon, beta)

        cBetah, sBetah = IMRPhenomXWignerdCoefficients_cosbeta(cos_beta)

        cexp_i_alpha = jnp.exp(1j * alpha)

        beta_powers = BetaPowers.from_half_angle_trig(cBetah, sBetah)

        # Select the appropriate twist function based on mode_idx
        # Order: 21, 22, 32, 33, 44
        hp_twist, hc_twist = jax.lax.switch(
            mode_idx,
            [
                lambda: twist_21(cexp_i_alpha, theta_JN, beta_powers),
                lambda: twist_22(cexp_i_alpha, theta_JN, beta_powers),
                lambda: twist_32(cexp_i_alpha, theta_JN, beta_powers),
                lambda: twist_33(cexp_i_alpha, theta_JN, beta_powers),
                # lambda: twist_43(cexp_i_alpha, pPrec.theta_JN, beta_powers),
                lambda: twist_44(cexp_i_alpha, theta_JN, beta_powers),
            ],
        )

        return hp_twist, hc_twist, epsilon * emm

    mode_indices = jnp.arange(5)  # 0 to 4 for modes 21, 22, 32, 33, 44
    hp_twist_all_modes, hc_twist_all_modes, epsilon_all_modes = jax.vmap(
        compute_twist_for_mode
    )(mode_indices)

    _hp = jnp.sum(
        hlm.T * hp_twist_all_modes.T * jnp.exp(-1j * epsilon_all_modes.T) / 2, axis=1
    )
    _hc = jnp.sum(
        hlm.T * hc_twist_all_modes.T * jnp.exp(-1j * epsilon_all_modes.T) / 2, axis=1
    )

    hp, hc = apply_polarization_rotation(zeta_polarisations, _hp, _hc)

    return hp, hc


@dataclass
class BetaPowers:
    """
    Stores powers of cos(beta/2) and sin(beta/2) for Wigner-d coefficient calculations.

    Attributes:
        cBetah: cos(beta/2)
        cBetah2: cos^2(beta/2)
        cBetah3: cos^3(beta/2)
        cBetah4: cos^4(beta/2)
        cBetah5: cos^5(beta/2)
        cBetah6: cos^6(beta/2)
        cBetah7: cos^7(beta/2)
        cBetah8: cos^8(beta/2)
        sBetah: sin(beta/2)
        sBetah2: sin^2(beta/2)
        sBetah3: sin^3(beta/2)
        sBetah4: sin^4(beta/2)
        sBetah5: sin^5(beta/2)
        sBetah6: sin^6(beta/2)
        sBetah7: sin^7(beta/2)
        sBetah8: sin^8(beta/2)
    """

    cBetah: float
    cBetah2: float
    cBetah3: float
    cBetah4: float
    cBetah5: float
    cBetah6: float
    cBetah7: float
    cBetah8: float
    sBetah: float
    sBetah2: float
    sBetah3: float
    sBetah4: float
    sBetah5: float
    sBetah6: float
    sBetah7: float
    sBetah8: float

    @classmethod
    def from_half_angle_trig(cls, cBetah: float, sBetah: float):
        """
        Constructs a BetaPowers instance from cos(beta/2) and sin(beta/2).

        Args:
            cBetah: cos(beta/2)
            sBetah: sin(beta/2)

        Returns:
            BetaPowers instance with all power values computed
        """
        cBetah2 = cBetah * cBetah
        cBetah3 = cBetah * cBetah2
        cBetah4 = cBetah * cBetah3
        cBetah5 = cBetah * cBetah4
        cBetah6 = cBetah * cBetah5
        cBetah7 = cBetah * cBetah6
        cBetah8 = cBetah * cBetah7

        sBetah2 = sBetah * sBetah
        sBetah3 = sBetah * sBetah2
        sBetah4 = sBetah * sBetah3
        sBetah5 = sBetah * sBetah4
        sBetah6 = sBetah * sBetah5
        sBetah7 = sBetah * sBetah6
        sBetah8 = sBetah * sBetah7

        return cls(
            cBetah=cBetah,
            cBetah2=cBetah2,
            cBetah3=cBetah3,
            cBetah4=cBetah4,
            cBetah5=cBetah5,
            cBetah6=cBetah6,
            cBetah7=cBetah7,
            cBetah8=cBetah8,
            sBetah=sBetah,
            sBetah2=sBetah2,
            sBetah3=sBetah3,
            sBetah4=sBetah4,
            sBetah5=sBetah5,
            sBetah6=sBetah6,
            sBetah7=sBetah7,
            sBetah8=sBetah8,
        )

        return None


def twist_22(cexp_i_alpha, theta_JN, beta_powers):
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha

    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha

    cexp_im_alpha_l2 = jnp.stack(
        [
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
        ],
        axis=0,
    )

    Y2m2 = compute_sminus2_l2(theta_JN, m=-2)
    Y2m1 = compute_sminus2_l2(theta_JN, m=-1)
    Y20 = compute_sminus2_l2(theta_JN, m=0)
    Y21 = compute_sminus2_l2(theta_JN, m=1)
    Y22 = compute_sminus2_l2(theta_JN, m=2)
    Y2mA = jnp.array([Y2m2, Y2m1, Y20, Y21, Y22])

    # Wigner-d coefficients
    # d^2_{-2,2}, d^2_{-1,2}, d^2_{0,2}, d^2_{1,2}, d^2_{2,2}

    d22 = jnp.array(
        [
            beta_powers.sBetah4,
            2.0 * beta_powers.cBetah * beta_powers.sBetah3,
            jnp.sqrt(6) * beta_powers.sBetah2 * beta_powers.cBetah2,
            2.0 * beta_powers.cBetah3 * beta_powers.sBetah,
            beta_powers.cBetah4,
        ]
    )

    # Exploit symmetry d^2_{-m,-2} = (-1)^m d^2_{-m,2}. See eq. A2 of Precessing paper
    # d^2_{-2,-2}, d^2_{-1,-2}, d^2_{0,-2}, d^2_{1,-2}, d^2_{2,-2}
    d2m2 = jnp.array([d22[4], -d22[3], d22[2], -d22[1], d22[0]])

    for m in range(-2, 2 + 1):
        A2m2emm = cexp_im_alpha_l2[-m + 2] * d2m2[m + 2] * Y2mA[m + 2]
        A22emmstar = cexp_im_alpha_l2[m + 2] * d22[m + 2] * jnp.conj(Y2mA[m + 2])
        hp_sum += A2m2emm + A22emmstar
        hc_sum += 1j * (A2m2emm - A22emmstar)

    return hp_sum, hc_sum


def twist_21(cexp_i_alpha, theta_JN, beta_powers):
    """
    Compute the twisting contributions for l=2, m'=1 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (2,1) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha

    cexp_im_alpha_l2 = jnp.stack(
        [
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
        ],
        axis=0,
    )

    Y2m2 = compute_sminus2_l2(theta_JN, m=-2)
    Y2m1 = compute_sminus2_l2(theta_JN, m=-1)
    Y20 = compute_sminus2_l2(theta_JN, m=0)
    Y21 = compute_sminus2_l2(theta_JN, m=1)
    Y22 = compute_sminus2_l2(theta_JN, m=2)
    Y2mA = jnp.array([Y2m2, Y2m1, Y20, Y21, Y22])

    # Wigner-d coefficients for m'=1
    # d^2_{-2,1}, d^2_{-1,1}, d^2_{0,1}, d^2_{1,1}, d^2_{2,1}
    d21 = jnp.array(
        [
            2.0 * beta_powers.cBetah * beta_powers.sBetah3,
            3.0 * beta_powers.cBetah2 * beta_powers.sBetah2 - beta_powers.sBetah4,
            jnp.sqrt(6)
            * (
                beta_powers.cBetah3 * beta_powers.sBetah
                - beta_powers.cBetah * beta_powers.sBetah3
            ),
            beta_powers.cBetah2 * (beta_powers.cBetah2 - 3.0 * beta_powers.sBetah2),
            -2.0 * beta_powers.cBetah3 * beta_powers.sBetah,
        ]
    )

    # Exploit symmetry d^2_{-m,-1} = -(-1)^m d^2_{m,1}. See eq. A2 of Precessing paper.
    # d^2_{-2,-1}, d^2_{-1,-1}, d^2_{0,-1}, d^2_{1,-1}, d^2_{2,-1}
    d2m1 = jnp.array([-d21[4], d21[3], -d21[2], d21[1], -d21[0]])

    for m in range(-2, 2 + 1):
        # Transfer functions, see eqs. 3.5-3.7 in Precessing paper.
        A2m1emm = cexp_im_alpha_l2[-m + 2] * d2m1[m + 2] * Y2mA[m + 2]
        A21emmstar = cexp_im_alpha_l2[m + 2] * d21[m + 2] * jnp.conj(Y2mA[m + 2])
        hp_sum += A2m1emm + A21emmstar
        hc_sum += 1j * (A2m1emm - A21emmstar)

    return hp_sum, hc_sum


def twist_33(cexp_i_alpha, theta_JN, beta_powers):
    """
    Compute the twisting contributions for l=3, m'=3 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (3,3) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha

    cexp_im_alpha_l3 = jnp.stack(
        [
            cexp_m3i_alpha,
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
            cexp_3i_alpha,
        ],
        axis=0,
    )

    Y3m3 = compute_sminus2_l3(theta=theta_JN, m=-3)
    Y3m2 = compute_sminus2_l3(theta=theta_JN, m=-2)
    Y3m1 = compute_sminus2_l3(theta=theta_JN, m=-1)
    Y30 = compute_sminus2_l3(theta=theta_JN, m=0)
    Y31 = compute_sminus2_l3(theta=theta_JN, m=1)
    Y32 = compute_sminus2_l3(theta=theta_JN, m=2)
    Y33 = compute_sminus2_l3(theta=theta_JN, m=3)
    Y3mA = jnp.array([Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33])

    # Wigner-d coefficients for m'=3
    # d^3_{-3,3}, d^3_{-2,3}, d^3_{-1,3}, d^3_{0,3}, d^3_{1,3}, d^3_{2,3}, d^3_{3,3}
    sqrt6 = jnp.sqrt(6.0)
    sqrt15 = jnp.sqrt(15.0)
    sqrt5 = jnp.sqrt(5.0)

    d33 = jnp.array(
        [
            beta_powers.sBetah6,
            sqrt6 * beta_powers.cBetah * beta_powers.sBetah5,
            sqrt15 * beta_powers.cBetah2 * beta_powers.sBetah4,
            2.0 * sqrt5 * beta_powers.cBetah3 * beta_powers.sBetah3,
            sqrt15 * beta_powers.cBetah4 * beta_powers.sBetah2,
            sqrt6 * beta_powers.cBetah5 * beta_powers.sBetah,
            beta_powers.cBetah6,
        ]
    )

    # Exploit symmetry d^3_{-m,-3} = -(-1)^m d^3_{m,3}. See eq. A2 of Precessing paper.
    # d^3_{-3,-3}, d^3_{-2,-3}, d^3_{-1,-3}, d^3_{0,-3}, d^3_{1,-3}, d^3_{2,-3}, d^3_{3,-3}
    d3m3 = jnp.array([d33[6], -d33[5], d33[4], -d33[3], d33[2], -d33[1], d33[0]])

    for m in range(-3, 3 + 1):
        # Transfer functions
        A3m3emm = cexp_im_alpha_l3[-m + 3] * d3m3[m + 3] * Y3mA[m + 3]
        A33emmstar = cexp_im_alpha_l3[m + 3] * d33[m + 3] * jnp.conj(Y3mA[m + 3])
        hp_sum += A3m3emm - A33emmstar
        hc_sum += 1j * (A3m3emm + A33emmstar)

    return hp_sum, hc_sum


def twist_32(cexp_i_alpha, theta_JN, beta_powers):
    """
    Compute the twisting contributions for l=3, m'=2 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (3,2) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha

    cexp_im_alpha_l3 = jnp.stack(
        [
            cexp_m3i_alpha,
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
            cexp_3i_alpha,
        ],
        axis=0,
    )

    Y3m3 = compute_sminus2_l3(theta=theta_JN, m=-3)
    Y3m2 = compute_sminus2_l3(theta=theta_JN, m=-2)
    Y3m1 = compute_sminus2_l3(theta=theta_JN, m=-1)
    Y30 = compute_sminus2_l3(theta=theta_JN, m=0)
    Y31 = compute_sminus2_l3(theta=theta_JN, m=1)
    Y32 = compute_sminus2_l3(theta=theta_JN, m=2)
    Y33 = compute_sminus2_l3(theta=theta_JN, m=3)
    Y3mA = jnp.array([Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33])

    # Wigner-d coefficients for m'=2
    # d^3_{-3,2}, d^3_{-2,2}, d^3_{-1,2}, d^3_{0,2}, d^3_{1,2}, d^3_{2,2}, d^3_{3,2}
    sqrt6 = jnp.sqrt(6.0)
    sqrt10 = jnp.sqrt(10.0)
    sqrt30 = jnp.sqrt(30.0)

    cBetah = beta_powers.cBetah
    cBetah2 = beta_powers.cBetah2
    cBetah3 = beta_powers.cBetah3
    cBetah4 = beta_powers.cBetah4
    cBetah5 = beta_powers.cBetah5
    sBetah = beta_powers.sBetah
    sBetah2 = beta_powers.sBetah2
    sBetah3 = beta_powers.sBetah3
    sBetah4 = beta_powers.sBetah4
    sBetah5 = beta_powers.sBetah5

    d32 = jnp.array(
        [
            sqrt6 * cBetah * sBetah5,
            sBetah4 * (5.0 * cBetah2 - sBetah2),
            sqrt10 * sBetah3 * (2.0 * cBetah3 - cBetah * sBetah2),
            sqrt30 * cBetah2 * (cBetah2 - sBetah2) * sBetah2,
            sqrt10 * cBetah3 * (cBetah2 * sBetah - 2.0 * sBetah3),
            cBetah4 * (cBetah2 - 5.0 * sBetah2),
            -1.0 * sqrt6 * cBetah5 * sBetah,
        ]
    )

    # Exploit symmetry d^3_{-m,-2} = (-1)^m d^3_{m,2}. See eq. A2 of Precessing paper.
    # d^3_{-3,-2}, d^3_{-2,-2}, d^3_{-1,-2}, d^3_{0,-2}, d^3_{1,-2}, d^3_{2,-2}, d^3_{3,-2}
    d3m2 = jnp.array([-d32[6], d32[5], -d32[4], d32[3], -d32[2], d32[1], -d32[0]])

    for m in range(-3, 3 + 1):
        # Transfer functions, see eqs. 3.5-3.7 in Precessing paper.
        A3m2emm = cexp_im_alpha_l3[-m + 3] * d3m2[m + 3] * Y3mA[m + 3]
        A32emmstar = cexp_im_alpha_l3[m + 3] * d32[m + 3] * jnp.conj(Y3mA[m + 3])
        hp_sum += A3m2emm - A32emmstar
        hc_sum += 1j * (A3m2emm + A32emmstar)

    return hp_sum, hc_sum


def twist_44(cexp_i_alpha, theta_JN, beta_powers):
    """
    Compute the twisting contributions for l=4, m'=4 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (4,4) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_4i_alpha = cexp_i_alpha * cexp_3i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha
    cexp_m4i_alpha = cexp_mi_alpha * cexp_m3i_alpha

    cexp_im_alpha_l4 = jnp.stack(
        [
            cexp_m4i_alpha,
            cexp_m3i_alpha,
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
            cexp_3i_alpha,
            cexp_4i_alpha,
        ],
        axis=0,
    )

    Y4m4 = compute_sminus2_l4(theta=theta_JN, m=-4)
    Y4m3 = compute_sminus2_l4(theta=theta_JN, m=-3)
    Y4m2 = compute_sminus2_l4(theta=theta_JN, m=-2)
    Y4m1 = compute_sminus2_l4(theta=theta_JN, m=-1)
    Y40 = compute_sminus2_l4(theta=theta_JN, m=0)
    Y41 = compute_sminus2_l4(theta=theta_JN, m=1)
    Y42 = compute_sminus2_l4(theta=theta_JN, m=2)
    Y43 = compute_sminus2_l4(theta=theta_JN, m=3)
    Y44 = compute_sminus2_l4(theta=theta_JN, m=4)
    Y4mA = jnp.array([Y4m4, Y4m3, Y4m2, Y4m1, Y40, Y41, Y42, Y43, Y44])

    # Wigner-d coefficients for m'=4
    # d^4_{-4,4}, d^4_{-3,4}, d^4_{-2,4}, d^4_{-1,4}, d^4_{0,4}, d^4_{1,4}, d^4_{2,4}, d^4_{3,4}, d^4_{4,4}
    sqrt2 = jnp.sqrt(2.0)
    sqrt7 = jnp.sqrt(7.0)
    sqrt14 = jnp.sqrt(14.0)
    sqrt70 = jnp.sqrt(70.0)

    d44 = jnp.array(
        [
            beta_powers.sBetah8,
            2.0 * sqrt2 * beta_powers.cBetah * beta_powers.sBetah7,
            2.0 * sqrt7 * beta_powers.cBetah2 * beta_powers.sBetah6,
            2.0 * sqrt14 * beta_powers.cBetah3 * beta_powers.sBetah5,
            sqrt70 * beta_powers.cBetah4 * beta_powers.sBetah4,
            2.0 * sqrt14 * beta_powers.cBetah5 * beta_powers.sBetah3,
            2.0 * sqrt7 * beta_powers.cBetah6 * beta_powers.sBetah2,
            2.0 * sqrt2 * beta_powers.cBetah7 * beta_powers.sBetah,
            beta_powers.cBetah8,
        ]
    )

    # Exploit symmetry d^4_{-m,-4} = (-1)^m d^4_{m,4}. See eq. A2 of Precessing paper.
    # d^4_{-4,-4}, d^4_{-3,-4}, d^4_{-2,-4}, d^4_{-1,-4}, d^4_{0,-4}, d^4_{1,-4}, d^4_{2,-4}, d^4_{3,-4}, d^4_{4,-4}
    d4m4 = jnp.array(
        [d44[8], -d44[7], d44[6], -d44[5], d44[4], -d44[3], d44[2], -d44[1], d44[0]]
    )

    for m in range(-4, 4 + 1):
        # Transfer functions, see eqs. 3.5-3.7 in Precessing paper.
        A4m4emm = cexp_im_alpha_l4[-m + 4] * d4m4[m + 4] * Y4mA[m + 4]
        A44emmstar = cexp_im_alpha_l4[m + 4] * d44[m + 4] * jnp.conj(Y4mA[m + 4])
        hp_sum += A4m4emm + A44emmstar
        hc_sum += 1j * (A4m4emm - A44emmstar)

    return hp_sum, hc_sum


@jit
def apply_polarization_rotation(zeta_polarization, _hp, _hc):
    """Apply polarization rotation to waveform components.

    Parameters
    ----------
    zeta_polarization : float
        Polarization angle.
    _hp : array_like
        Plus polarization component (unrotated).
    _hc : array_like
        Cross polarization component (unrotated).

    Returns
    -------
    hp : array_like
        Rotated plus polarization.
    hc : array_like
        Rotated cross polarization.
    """
    cosPolFac = jnp.cos(2.0 * zeta_polarization)
    sinPolFac = jnp.sin(2.0 * zeta_polarization)

    hp = cosPolFac * _hp + sinPolFac * _hc
    hc = cosPolFac * _hc - sinPolFac * _hp

    return hp, hc


@jit
def IMRPhenomXWignerdCoefficients_cosbeta(cos_beta):
    """
    Compute cos(beta/2) and sin(beta/2) from cos(beta).

    Uses half-angle formulas:
    - cos(beta/2) = sqrt((1 + cos(beta)) / 2)
    - sin(beta/2) = sqrt((1 - cos(beta)) / 2)

    Parameters
    ----------
    cos_beta : float or array
        cos(beta)

    Returns
    -------
    cos_beta_half : float or array
        cos(beta/2), always non-negative
    sin_beta_half : float or array
        sin(beta/2), always non-negative
    """
    # Note that the results here are indeed always non-negative
    cos_beta_half = jnp.sqrt(jnp.abs(1.0 + cos_beta) / 2.0)  # cos(beta/2)
    sin_beta_half = jnp.sqrt(jnp.abs(1.0 - cos_beta) / 2.0)  # sin(beta/2)

    return cos_beta_half, sin_beta_half


@jit
def XLALSimIMRPhenomXUtilsHztoMf(fHz: float, Mtot_Msun: float) -> float:
    """
    Convert frequency from Hz to geometric units (Mf).

    Parameters
    ----------
    fHz : float
        Frequency in Hz
    Mtot_Msun : float
        Total mass in solar masses

    Returns
    -------
    float
        Geometric frequency Mf
    """
    # Mtot in seconds = Mtot_Msun * MTSUN_SI
    return fHz * Mtot_Msun * MTSUN_SI


def XLALSimIMRPhenomHMGethlmModes(
    freqs: Array,
    m1_SI: float,
    m2_SI: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
    phiRef: float,
    deltaF: float,
    f_ref: float,
    extraParams: dict,
):
    ModeArray = extraParams["ModeArray"]

    pHM = {}
    pHM = init_PhenomHM_Storage(
        pHM,
        m1_SI,
        m2_SI,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        freqs,
        deltaF,
        f_ref,
        phiRef,
        ModeArray,
    )

    # FIXME? LAL does some frequency spacing here, I'm not sure yet whether we need to do this

    # line 1288
    # Might be unused since we use ripple IMRPhenomD, which uses f[Hz]
    freqs_geom = XLALSimIMRPhenomXUtilsHztoMf(freqs, pHM["Mtot"])

    # line 1316
    # compute the reference phase shift need to align the waveform so that
    # the phase is equal to phiRef at the reference frequency f_ref.
    # the phase shift is computed by evaluating the phase of the
    # (l,m)=(2,2) mode.
    # phi0 is the correction we need to add to each mode.

    theta = jnp.array([pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]])
    PhenomD_coeffs = get_coeffs(theta)
    PhenomD_transition_freqs = get_transition_frequencies(
        theta, PhenomD_coeffs[5], PhenomD_coeffs[6]
    )

    phi_22_at_f_ref = IMRPhenomD_Phase(
        f_ref,
        jnp.array([pHM["m1"], pHM["m2"], chi1z, chi2z]),
        PhenomD_coeffs,
        PhenomD_transition_freqs,
    )
    phi0 = 0.5 * phi_22_at_f_ref + phiRef

    vmapped_IMRPhenomHMEvaluateOnehlmMode = jax.vmap(
        IMRPhenomHMEvaluateOnehlmMode,
        in_axes=(
            None,  # freqs_geom
            None,  # pHM
            0,  # ell
            0,  # mm
            None,  # phi0
        ),
    )

    hlms = vmapped_IMRPhenomHMEvaluateOnehlmMode(
        freqs_geom, pHM, pHM["ell_mm_pairs"][:, 0], pHM["ell_mm_pairs"][:, 1], phi0
    )

    return hlms


def IMRPhenomHMEvaluateOnehlmMode(
    freqs_geom: Array, pHM: dict, ell: int, mm: int, phi0: float
):
    """
    Copy of IMRPhenomHMEvaluateOnehlmMode in LALSimIMRPhenomHM.c
    """

    # generate phase and amplitude for single l,m mode
    phase_lm = IMRPhenomHMPhase(freqs_geom, pHM, ell, mm)
    amp_lm = IMRPhenomHMAmplitude(freqs_geom, pHM, ell, mm)

    # compute time shift using pre-computed fRD, fdamp from pHM (with PhenomPv2 final spin)
    theta_intrinsic = jnp.array([pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]])
    coeffs = get_coeffs(theta_intrinsic)

    # IMPORTANT: Use PhenomD-style fRD, fdamp for t0 calculation to match LAL's IMRPhenomDComputet0
    M_s = pHM["Mtot"] * MTSUN
    f_RD = pHM["Mf_RD_22_PhenomD"] / M_s  # Convert from Mf to Hz
    f_damp = pHM["Mf_DM_22_PhenomD"] / M_s

    # Compute f4 (fmaxCalc) using the correct fRD, fdamp
    gamma2, gamma3 = coeffs[5], coeffs[6]
    f4 = jax.lax.cond(
        gamma2 >= 1,
        lambda: jnp.abs(f_RD + (-f_damp * gamma3) / gamma2),
        lambda: jnp.abs(
            f_RD + (f_damp * (-1 + jnp.sqrt(1 - gamma2**2)) * gamma3) / gamma2
        ),
    )

    t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, theta_intrinsic, coeffs, f_RD, f_damp)

    Mf = freqs_geom
    phase_term1 = -t0 * (Mf - pHM["Mf_ref"])
    phase_term2 = phase_lm - (mm * phi0)
    return amp_lm * jnp.exp(-1j * (phase_term1 + phase_term2))


def XLALSimPhenomUtilsPhenomPv2FinalSpin(
    m1: float, m2: float, chi1_l: float, chi2_l: float, chip: float
):
    """
    Copy of XLALSimPhenomUtilsPhenomPv2FinalSpin in LALSimPhenomUtils.c
    Assuming m1 >= m2
    """

    M = m1 + m2
    eta = m1 * m2 / (M * M)

    q_factor = m1 / M

    # # This is needed to stabilize JAX derivatives
    # Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    af_parallel = FinalSpin0815(eta, chi1_l, chi2_l)

    Sperp = chip * q_factor * q_factor

    return jnp.copysign(1.0, af_parallel) * jnp.sqrt(Sperp**2 + af_parallel**2)


def init_PhenomHM_Storage(
    p: dict,
    m1_SI: float,
    m2_SI: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
    freqs: Array,
    deltaF: float,
    f_ref: float,
    phiRef: float,
    ModeArray: Array,
):
    """
    Precompute a bunch of PhenomHM related quantities and store them
    Copy of init_PhenomHM_Storage in LALSimIMRPhenomHM.c
    """

    p["m1"] = m1_SI / MSUN
    p["m2"] = m2_SI / MSUN
    p["Mtot"] = p["m1"] + p["m2"]
    p["eta"] = p["m1"] * p["m2"] / (p["Mtot"] * p["Mtot"])
    p["chi1x"] = chi1x
    p["chi1y"] = chi1y
    p["chi1z"] = chi1z
    p["chi2x"] = chi2x
    p["chi2y"] = chi2y
    p["chi2z"] = chi2z
    p["phiRef"] = phiRef
    p["deltaF"] = deltaF
    p["freqs"] = freqs
    p["f_ref"] = f_ref
    p["Mf_ref"] = XLALSimIMRPhenomXUtilsHztoMf(f_ref, p["Mtot"])

    p["chip"] = XLALSimPhenomUtilsChiP(
        p["m1"], p["m2"], p["chi1x"], p["chi1y"], p["chi2x"], p["chi2y"]
    )

    p["finmass"] = 1.0 - EradRational0815(p["eta"], p["chi1z"], p["chi2z"])
    p["finspin"] = XLALSimPhenomUtilsPhenomPv2FinalSpin(
        p["m1"], p["m2"], p["chi1z"], p["chi2z"], p["chip"]
    )

    # Define the supported modes and their indices
    ell_mm_pairs = ModeArray
    p["ell_mm_pairs"] = ell_mm_pairs

    # Create a mapping from (ell, mm) to array index for JAX-compatible lookup
    # We'll use a 2D array where mode_index_map[ell, mm] gives the index
    # Maximum ell=4, mm=4, so we need a 5x5 array (indices 0-4)
    # IMPORTANT: Build the map dynamically based on the actual ModeArray order
    mode_index_map = jnp.full((5, 5), -1, dtype=jnp.int32)
    ell_vals = ModeArray[:, 0].astype(jnp.int32)
    mm_vals = ModeArray[:, 1].astype(jnp.int32)
    mode_index_map = mode_index_map.at[ell_vals, mm_vals].set(
        jnp.arange(len(ModeArray))
    )
    p["mode_index_map"] = mode_index_map

    vmapped_IMRPhenomHMGetRingdownFrequency = jax.vmap(
        IMRPhenomHMGetRingdownFrequency, in_axes=(0, 0, None, None)
    )
    f_rd_array, f_damp_array = vmapped_IMRPhenomHMGetRingdownFrequency(
        ell_mm_pairs[:, 0], ell_mm_pairs[:, 1], p["finmass"], p["finspin"]
    )

    # Store as 1D arrays indexed by mode order
    p["PhenomHMfring"] = f_rd_array  # shape: (5,)
    p["PhenomHMfdamp"] = f_damp_array  # shape: (5,)
    p["Mf_RD_22"] = f_rd_array[1]
    p["Mf_DM_22"] = f_damp_array[1]

    # Rholm and Taulm as 1D arrays (one per mode)
    p["Rholm"] = p["Mf_RD_22"] / f_rd_array  # shape: (5,)
    p["Taulm"] = f_damp_array / p["Mf_DM_22"]  # shape: (5,)

    # IMPORTANT: For the PhenomD amplitude calculation, LAL's IMRPhenomDAmpFrequencySequence
    # uses the PhenomD QNM data (QNMData_fRD, QNMData_fdamp) rather than SimRingdownCW_CW07102016.
    # We need to store these separately for use in IMRPhenomHMAmplitude.

    p["Mf_RD_22_PhenomD"] = (
        jnp.interp(p["finspin"], QNMData_a, QNMData_fRD) / p["finmass"]
    )
    p["Mf_DM_22_PhenomD"] = (
        jnp.interp(p["finspin"], QNMData_a, QNMData_fdamp) / p["finmass"]
    )

    return p


def IMRPhenomHMGetRingdownFrequency(
    ell: int, mm: int, finalmass: float, finalspin: float
):
    """
    Copy of IMRPhenomHMGetRingdownFrequency in LALSimIMRPhenomHM.c
    """

    inv2Pi = 0.5 / PI
    ZZ = SimRingdownCW_CW07102016(SimRingdownCW_KAPPA(finalspin, ell, mm), ell, mm, 0)
    Mf_RD_tmp = inv2Pi * jnp.real(
        ZZ
    )  # GW ringdown frequency, converted from angular frequency
    fringdown = Mf_RD_tmp / finalmass  # scale by predicted final mass
    # lm mode ringdown damping time (imaginary part of ringdown), geometric units
    f_DAMP_tmp = inv2Pi * jnp.imag(ZZ)  # this is the 1./tau in the complex QNM
    fdamp = f_DAMP_tmp / finalmass  # scale by predicted final mass

    return fringdown, fdamp


def SimRingdownCW_KAPPA(jf: float, ell: int, emm: int):
    """
    Domain mapping for dimnesionless BH spin
    """
    alpha = jnp.log(2.0 - jf) / jnp.log(3)
    beta = 1.0 / (2.0 + ell - jnp.abs(emm))
    return alpha**beta


def SimRingdownCW_CW07102016(kappa: float, ell: int, input_m: int, n: int):
    """
    Dimensionless QNM Frequencies: Note that name encodes date of writing
    """

    kappa2 = kappa * kappa
    kappa3 = kappa2 * kappa
    kappa4 = kappa3 * kappa

    m = jnp.abs(input_m)

    def branch_220():
        # Fit for (l,m,n) == (2,2,0). This is a zero-damped mode in the extremal Kerr limit.
        return 1.0 + kappa * (
            1.557847 * jnp.exp(2.903124 * 1j)
            + 1.95097051 * jnp.exp(5.920970 * 1j) * kappa
            + 2.09971716 * jnp.exp(2.760585 * 1j) * kappa2
            + 1.41094660 * jnp.exp(5.914340 * 1j) * kappa3
            + 0.41063923 * jnp.exp(2.795235 * 1j) * kappa4
        )

    # def branch_221(): # Unused in XPHM
    #     return

    def branch_320():
        kappa5 = kappa4 * kappa
        kappa6 = kappa5 * kappa
        # Fit for (l,m,n) == (3,2,0). This is NOT a zero-damped mode in the extremal Kerr limit.
        return (
            1.022464 * jnp.exp(0.004870 * 1j)
            + 0.24731213 * jnp.exp(0.665292 * 1j) * kappa
            + 1.70468239 * jnp.exp(3.138283 * 1j) * kappa2
            + 0.94604882 * jnp.exp(0.163247 * 1j) * kappa3
            + 1.53189884 * jnp.exp(5.703573 * 1j) * kappa4
            + 2.28052668 * jnp.exp(2.685231 * 1j) * kappa5
            + 0.92150314 * jnp.exp(5.841704 * 1j) * kappa6
        )

    def branch_440():
        # Fit for (l,m,n) == (4,4,0). This is a zero-damped mode in the extremal Kerr limit.
        return 2.0 + kappa * (
            2.658908 * jnp.exp(3.002787 * 1j)
            + 2.97825567 * jnp.exp(6.050955 * 1j) * kappa
            + 3.21842350 * jnp.exp(2.877514 * 1j) * kappa2
            + 2.12764967 * jnp.exp(5.989669 * 1j) * kappa3
            + 0.60338186 * jnp.exp(2.830031 * 1j) * kappa4
        )

    def branch_210():
        kappa5 = kappa4 * kappa
        kappa6 = kappa5 * kappa
        # Fit for (l,m,n) == (2,1,0). This is NOT a zero-damped mode in the extremal Kerr limit.
        return (
            0.589113 * jnp.exp(0.043525 * 1j)
            + 0.18896353 * jnp.exp(2.289868 * 1j) * kappa
            + 1.15012965 * jnp.exp(5.810057 * 1j) * kappa2
            + 6.04585476 * jnp.exp(2.741967 * 1j) * kappa3
            + 11.12627777 * jnp.exp(5.844130 * 1j) * kappa4
            + 9.34711461 * jnp.exp(2.669372 * 1j) * kappa5
            + 3.03838318 * jnp.exp(5.791518 * 1j) * kappa6
        )

    def branch_330():
        # Fit for (l,m,n) == (3,3,0). This is a zero-damped mode in the extremal Kerr limit.
        return 1.5 + kappa * (
            2.095657 * jnp.exp(2.964973 * 1j)
            + 2.46964352 * jnp.exp(5.996734 * 1j) * kappa
            + 2.66552551 * jnp.exp(2.817591 * 1j) * kappa2
            + 1.75836443 * jnp.exp(5.932693 * 1j) * kappa3
            + 0.49905688 * jnp.exp(2.781658 * 1j) * kappa4
        )

    # def branch_331(): # Unused in XPHM
    #     return

    def branch_430():
        # Fit for (l,m,n) == (4,3,0). This is a zero-damped mode in the extremal Kerr limit.
        return 1.5 + kappa * (
            0.205046 * jnp.exp(0.595328 * 1j)
            + 3.10333396 * jnp.exp(3.016200 * 1j) * kappa
            + 4.23612166 * jnp.exp(6.038842 * 1j) * kappa2
            + 3.02890198 * jnp.exp(2.826239 * 1j) * kappa3
            + 0.90843949 * jnp.exp(5.915164 * 1j) * kappa4
        )

    # def branch_550(): # Unused in XPHM
    #     return

    def branch_not_implemented():
        return 0.0 * 1j  # Return complex nr. so pytree structure is preserved

    # Determine index of branch to use. If other modes are added, this will need to be expanded for new modes
    # Create a unique key from l, m, n: key = l * 100 + m * 10 + n
    key = ell * 100 + jnp.abs(m) * 10 + n

    # Map keys to indices
    # 210 → 0, 220 → 1, 320 → 2, 330 → 3, 430 → 4, 440 → 5
    index = jnp.where(
        key == 210,
        0,
        jnp.where(
            key == 220,
            1,
            jnp.where(
                key == 320,
                2,
                jnp.where(
                    key == 330, 3, jnp.where(key == 430, 4, jnp.where(key == 440, 5, 6))
                ),
            ),
        ),
    )

    ans = jax.lax.switch(
        index,
        [
            branch_210,
            branch_220,
            branch_320,
            branch_330,
            branch_430,
            branch_440,
            branch_not_implemented,
        ],
    )

    return jax.lax.select(  # If m<0, then take the *Negative* conjugate
        input_m < 0, -jnp.conj(ans), ans
    )


def IMRPhenomHMFreqDomainMap(Mflm, ell, mm, pHM, AmpFlag):
    # Mflm here has the same meaning as Mf_wf in XLALSimIMRPhenomHMFreqDomainMapHM (old deleted function).
    # Following variables not used in this funciton but are returned in IMRPhenomHMFreqDomainMapParams
    a, b = IMRPhenomHMFreqDomainMapParams(Mflm, ell, mm, pHM, AmpFlag)
    Mf22 = a * Mflm + b
    return Mf22


def IMRPhenomHMAmplitude(freqs_geom: Array, pHM: dict, ell: int, mm: int):
    """
    Returns IMRPhenomHM amplitude evaluated at a set of input frequencies for the l,m mode
    Copy of IMRPhenomHMAmplitude in LALSimIMRPhenomHM.c
    """

    # scale input frequencies according to PhenomHM model
    # LL: Map the input domain (frequencies) for this ell mm multipole
    # to those appropirate for the ell=|mm| multipole
    freqs_amp = IMRPhenomHMFreqDomainMap(freqs_geom, ell, mm, pHM, AmpFlag=True)

    # LL: Compute the PhenomD Amplitude at the mapped l=m=2 fequencies
    # NOTE: Use IMRPhenDAmplitude_NoCut instead of IMRPhenomD_Amp because
    # the mapped frequencies can exceed fM_CUT for higher modes

    # compute time shift using pre-computed fRD, fdamp from pHM (with PhenomPv2 final spin)
    theta = jnp.array([pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]])
    PhenomD_coeffs = get_coeffs(theta)

    # IMPORTANT: Use PhenomD-style fRD, fdamp (from PhenomD QNM data) for the amplitude calculation.
    # LAL's IMRPhenomDAmpFrequencySequence uses fring()/fdamp() from LALSimIMRPhenomD_internals.c
    # which use QNMData_fring/QNMData_fdamp, NOT SimRingdownCW_CW07102016.
    M_s = pHM["Mtot"] * MTSUN
    f_RD = pHM["Mf_RD_22_PhenomD"] / M_s  # Convert from Mf to Hz
    f_damp = pHM["Mf_DM_22_PhenomD"] / M_s

    # Phase transition frequencies
    f1 = 0.018 / (M_s)
    f2 = 0.5 * f_RD

    # Amplitude transition frequencies
    f3 = 0.014 / (M_s)
    # Compute f4 (fmaxCalc) using the correct fRD, fdamp
    gamma2, gamma3 = PhenomD_coeffs[5], PhenomD_coeffs[6]
    f4 = jax.lax.cond(
        gamma2 >= 1,
        lambda: jnp.abs(f_RD + (-f_damp * gamma3) / gamma2),
        lambda: jnp.abs(
            f_RD + (f_damp * (-1 + jnp.sqrt(1 - gamma2**2)) * gamma3) / gamma2
        ),
    )

    PhenomD_transition_freqs = jnp.array([f1, f2, f3, f4, f_RD, f_damp])
    amps_normalized = IMRPhenDAmplitude_NoCut(
        freqs_amp / (pHM["Mtot"] * MTSUN),
        theta,
        PhenomD_coeffs,
        PhenomD_transition_freqs,
    )

    # Apply the Amp0 prefactor: amp0 = sqrt(2/3 * eta) * pi^(-1/6) * f^(-7/6)
    # This matches LAL's IMRPhenDAmplitude which multiplies AmpInsAnsatz by AmpPreFac
    eta = pHM["eta"]
    amp0 = jnp.sqrt(2.0 / 3.0 * eta) * (PI ** (-1.0 / 6.0))
    # The prefactor is applied at the mapped frequencies (freqs_amp)
    amps = amp0 * (freqs_amp ** (-7.0 / 6.0)) * amps_normalized

    # LL: Here we map the ampliude's range using two steps:
    # (1) We divide by the leading order l=m=2 behavior, and then
    # scale in the expected PN behavior for the multipole of interest.
    # NOTE that this step is done at the mapped frequencies,
    # which results in smooth behavior despite the sharp featured of the domain map.
    # There are other (perhaps more intuitive) options for mapping the amplitudes,
    # but these do not have the desired smooth features.
    # (2) An additional scaling is needed to recover the desired PN ampitude.
    # This is needed becuase only frequencies appropriate for the dominant
    # quadrupole have been used thusly, so the current answer does not
    # conform to PN expectations for inspiral.
    # This is trikier than described here, so please give it a deeper think.

    # LL: Calculate the corrective factor for step #2
    beta_term1 = IMRPhenomHMOnePointFiveSpinPN(
        freqs_geom, ell, mm, pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]
    )

    # COMMENT FROM LAL CODE:
    # HACK to fix equal black hole case producing NaNs.
    # More elegant solution needed.
    def beta_term1_nozero():
        beta_term2 = IMRPhenomHMOnePointFiveSpinPN(
            2 * freqs_geom / mm,
            ell,
            mm,
            pHM["m1"],
            pHM["m2"],
            pHM["chi1z"],
            pHM["chi2z"],
        )

        beta = beta_term1 / beta_term2

        # LL: Apply steps #1 and #2
        HMamp_term1 = IMRPhenomHMOnePointFiveSpinPN(
            freqs_amp, ell, mm, pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]
        )
        HMamp_term2 = IMRPhenomHMOnePointFiveSpinPN(
            freqs_amp, 2, 2, pHM["m1"], pHM["m2"], 0.0, 0.0
        )

        return beta * HMamp_term1 / HMamp_term2

    rescaling = jnp.where(
        beta_term1 == 0.0,
        0.0,
        beta_term1_nozero(),
    )
    return amps * rescaling


def IMRPhenomHMOnePointFiveSpinPN(fM, ell, m, M1, M2, X1z, X2z):
    """
    Implementation of IMRPhenomHMOnePointFiveSpinPN from LALSimIMRPhenomHM.c
    Currently supported modes: (2,1), (2,2), (3,2), (3,3), (4,4)
    """

    # LLondon 2017

    # Define effective intinsic parameters
    M_INPUT = M1 + M2
    M1 = M1 / (M_INPUT)
    M2 = M2 / (M_INPUT)
    M = M1 + M2
    eta = M1 * M2 / (M * M)
    delta = jnp.sqrt(1.0 - 4 * eta)
    Xs = 0.5 * (X1z + X2z)
    Xa = 0.5 * (X1z - X2z)

    # Define PN parameter and realed powers
    v = jnp.power(M * 2.0 * PI * fM / m, 1.0 / 3.0)
    v2 = v * v
    v3 = v * v2

    # Define Leading Order Ampitude for each supported multipole

    # (l,m) = (2,2)
    # THIS IS LEADING ORDER
    def lm_22():
        return jnp.full_like(fM, 1.0, dtype=complex)

    def lm_21():
        # (l,m) = (2,1)
        # SPIN TERMS ADDED

        # UP TO 4PN
        v4 = v * v3
        return (jnp.sqrt(2.0) / 3.0) * (
            v * delta
            - v2 * 1.5 * (Xa + delta * Xs)
            + v3 * delta * ((335.0 / 672.0) + (eta * 117.0 / 56.0))
            + v4
            * (
                Xa * (3427.0 / 1344 - eta * 2101.0 / 336)
                + delta * Xs * (3427.0 / 1344 - eta * 965 / 336)
                + delta * (-1j * 0.5 - PI - 2 * 1j * 0.69314718056)
            )
        )

    def lm_33():
        # (l,m) = (3,3)
        # THIS IS LEADING ORDER
        return 0.75 * jnp.sqrt(5.0 / 7.0) * (v * delta) + 0 * 1j

    def lm_32():
        # (l,m) = (3,2)
        # NO SPIN TERMS to avoid roots
        return (1.0 / 3.0) * jnp.sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta)) + 0 * 1j

    def lm_44():
        # (l,m) = (4,4)
        # THIS IS LEADING ORDER
        return (4.0 / 9.0) * jnp.sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta) + 0 * 1j

    key = ell * 10 + jnp.abs(m)

    # Map keys to indices
    index = jnp.where(
        key == 21,
        0,
        jnp.where(key == 22, 1, jnp.where(key == 32, 2, jnp.where(key == 33, 3, 4))),
    )

    Hlm = jax.lax.switch(index, [lm_21, lm_22, lm_32, lm_33, lm_44])

    # Compute the final PN Amplitude at Leading Order in fM
    return M * M * PI * jnp.sqrt(eta * 2.0 / 3) * v ** (-3.5) * jnp.abs(Hlm)


def IMRPhenomHMPhase(freqs_geom: Array, pHM: dict, ell: int, mm: int):
    """
    Returns IMRPhenomHM phase evaluated at a set of input frequencies for the l,m mode
    Copy of IMRPhenomHMPhase in LALSimIMRPhenomHM.c
    """

    q = {}
    q = IMRPhenomHMPhasePreComp(q, ell, mm, pHM)

    # Get mode index for array lookup
    mode_idx = pHM["mode_index_map"][ell, mm]
    Rholm = pHM["Rholm"][mode_idx]
    Taulm = pHM["Taulm"][mode_idx]

    # compute time shift using pre-computed fRD, fdamp from pHM (with PhenomPv2 final spin)
    theta = jnp.array([pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]])
    PhenomD_coeffs = get_coeffs(theta)
    M_s = pHM["Mtot"] * MTSUN
    f_RD = pHM["Mf_RD_22_PhenomD"] / M_s  # Convert from Mf to Hz
    f_damp = pHM["Mf_DM_22_PhenomD"] / M_s

    # Phase transition frequencies
    f1 = 0.018 / (M_s)
    f2 = 0.5 * f_RD

    # Amplitude transition frequencies
    f3 = 0.014 / (M_s)
    # Compute f4 (fmaxCalc) using the correct fRD, fdamp
    gamma2, gamma3 = PhenomD_coeffs[5], PhenomD_coeffs[6]
    f4 = jax.lax.cond(
        gamma2 >= 1,
        lambda: jnp.abs(f_RD + (-f_damp * gamma3) / gamma2),
        lambda: jnp.abs(
            f_RD + (f_damp * (-1 + jnp.sqrt(1 - gamma2**2)) * gamma3) / gamma2
        ),
    )
    PhenomD_transition_freqs = jnp.array([f1, f2, f3, f4, f_RD, f_damp])

    def PhenDPhaseA(freqs_geom):
        Mf = (q["ai"] * freqs_geom + q["bi"]) / M_s  # ripple PhenomD uses f[Hz] here
        return (
            IMRPhenomD_Phase(
                Mf, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
            )
            / q["ai"]
        )

    def PhenDPhaseB(freqs_geom):
        Mf = (q["am"] * freqs_geom + q["bm"]) / M_s
        return (
            IMRPhenomD_Phase(
                Mf, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
            )
            / q["am"]
            - q["PhDBconst"]
            + q["PhDBAterm"]
        )

    def PhenDPhaseC(freqs_geom):
        Mfr = (q["am"] * q["fr"] + q["bm"]) / M_s
        tmpphaseC = (
            IMRPhenomD_Phase(
                Mfr, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
            )
            / q["am"]
            - q["PhDBconst"]
            + q["PhDBAterm"]
        )
        Mf = (q["ar"] * freqs_geom + q["br"]) / M_s
        return (
            IMRPhenomD_Phase(
                Mf, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
            )
            / q["ar"]
            - q["PhDCconst"]
            + tmpphaseC
        )

    phase = CSHIFT[jnp.abs(mm)] + jnp.where(
        freqs_geom <= q["fi"],
        PhenDPhaseA(freqs_geom),
        jnp.where(
            freqs_geom <= q["fr"], PhenDPhaseB(freqs_geom), PhenDPhaseC(freqs_geom)
        ),
    )  # - PI/2.0 # subtract 22-mode shift as this is accounted for in ripple?

    return phase


def IMRPhenomHMPhasePreComp(q: dict, ell: int, emm: int, pHM: dict):
    """
    Copy of IMRPhenomHMPhasePreComp in LALSimIMRPhenomHM.c
    """

    # NOTE: As long as Mfshift isn't >= fr then the value of the shift is arbitrary.
    Mfshift = 0.0001

    # Get mode index for array lookup
    mode_idx = pHM["mode_index_map"][ell, emm]

    # I have moved the computation of f1, fi and fr outside of IMRPhenomHMFreqDomainMapParams
    f1 = 0.018  # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
    fi = f1 / pHM["Rholm"][mode_idx]
    fr = pHM["PhenomHMfring"][mode_idx]

    flm_0 = Mfshift
    flm_i = fi + Mfshift
    flm_r = fr + Mfshift

    ai, bi = IMRPhenomHMFreqDomainMapParams(flm_0, ell, emm, pHM, ampFlag=False)
    am, bm = IMRPhenomHMFreqDomainMapParams(flm_i, ell, emm, pHM, ampFlag=False)
    ar, br = IMRPhenomHMFreqDomainMapParams(flm_r, ell, emm, pHM, ampFlag=False)

    q["ai"] = ai
    q["bi"] = bi
    q["am"] = am
    q["bm"] = bm
    q["ar"] = ar
    q["br"] = br

    q["fi"] = fi
    q["fr"] = fr

    Rholm = pHM["Rholm"][mode_idx]
    Taulm = pHM["Taulm"][mode_idx]

    M_s = pHM["Mtot"] * MTSUN
    theta = jnp.array([pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]])
    PhenomD_coeffs = get_coeffs(theta)

    # IMPORTANT: Use PhenomD-style fRD/fdamp (from PhenomD QNM data with PhenomPv2 final spin)
    # to match LAL's IMRPhenomDSetupAmpAndPhaseCoefficients behavior
    f_RD = pHM["Mf_RD_22_PhenomD"] / M_s  # Convert from Mf to Hz
    f_damp = pHM["Mf_DM_22_PhenomD"] / M_s
    PhenomD_transition_freqs = get_transition_frequencies_from_fRD_fdamp(
        theta, PhenomD_coeffs[5], PhenomD_coeffs[6], f_RD, f_damp
    )

    PhDBMf = q["am"] * fi + q["bm"]
    q["PhDBconst"] = (
        IMRPhenomD_Phase(
            PhDBMf / M_s, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
        )
        / q["am"]
    )

    PhDCMf = q["ar"] * fr + q["br"]
    q["PhDCconst"] = (
        IMRPhenomD_Phase(
            PhDCMf / M_s, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
        )
        / q["ar"]
    )

    PhDBAMf = q["ai"] * fi + q["bi"]
    q["PhDBAterm"] = (
        IMRPhenomD_Phase(
            PhDBAMf / M_s, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
        )
        / q["ai"]
    )
    return q


def IMRPhenomHMFreqDomainMapParams(
    flm: float,  # input waveform frequency
    ell: int,  # spherical harmonics ell mode
    mm: int,  # spherical harmonics m mode
    pHM: dict,
    ampFlag: bool,  # is ==1 then computes for amplitude, if ==0 then computes for phase
):
    """
    Copy of the phase computation of IMRPhenomHMFreqDomainMapParams in LALSimIMRPhenomHM.c
    """

    # Get mode index for array lookup
    mode_idx = pHM["mode_index_map"][ell, mm]

    Mf_1_22 = jax.lax.select(
        ampFlag,
        0.014,  # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        0.018,  # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
    )
    Mf_RD_22 = pHM["Mf_RD_22"]
    Mf_RD_lm = pHM["PhenomHMfring"][mode_idx]

    # Define a ratio of QNM frequencies to be used for scaling various quantities
    Rholm = pHM["Rholm"][mode_idx]

    # Given experiments with the l!=m modes, it appears that the QNM scaling rather than the PN scaling may be optimal for mapping f1
    Mf_1_lm = Mf_1_22 / Rholm

    # Define transition frequencies
    fi = Mf_1_lm
    fr = Mf_RD_lm

    # Define the slope and intercepts of the linear transformation used
    Ai = 2.0 / mm
    Bi = 0.0

    Am, Bm = IMRPhenomHMSlopeAmAndBm(mm, fi, fr, Mf_RD_22, Mf_RD_lm, ampFlag, ell, pHM)

    Ar = jax.lax.select(
        ampFlag,
        1.0,  # For amplitude
        Rholm,  # For phase
    )
    Br = jax.lax.select(
        ampFlag,
        -Mf_RD_lm + Mf_RD_22,  # For amplitude
        0.0,  # For phase
    )

    a, b = IMRPhenomHMMapParams(flm, fi, fr, Ai, Bi, Am, Bm, Ar, Br)

    return a, b


def IMRPhenomHMSlopeAmAndBm(
    mm: int,
    fi: float,
    fr: float,
    Mf_RD_22: float,
    Mf_RD_lm: float,
    AmpFlag: bool,
    ell: int,
    pHM: dict,
):
    """
    Copy of IMRPhenomHMSlopeAmAndBm in LALSimIMRPhenomHM.c
    """
    # Get mode index for array lookup
    mode_idx = pHM["mode_index_map"][ell, mm]

    Trd = IMRPhenomHMTrd(fr, Mf_RD_22, Mf_RD_lm, AmpFlag, mode_idx, pHM)
    Ti = 2.0 * fi / mm  # = IMRPhenomHMTi(fi, mm), line 543

    Am = (Trd - Ti) / (fr - fi)
    Bm = Ti - fi * Am

    return Am, Bm


def IMRPhenomHMTrd(
    Mf: float, Mf_RD_22: float, Mf_RD_lm: float, AmpFlag: bool, mode_idx: int, pHM: dict
):
    """
    Copy of IMRPhenomHMTrd in LALSimIMRPhenomHM.c
    domain mapping function - ringdown
    """

    return jax.lax.select(
        AmpFlag,
        Mf
        - Mf_RD_lm
        + Mf_RD_22,  # Used for the Amplitude as an approx fix for post merger powerlaw slope
        pHM["Rholm"][mode_idx] * Mf,  # Used for the Phase
    )


def IMRPhenomHMMapParams(
    flm: float,
    fi: float,
    fr: float,
    Ai: float,
    Bi: float,
    Am: float,
    Bm: float,
    Ar: float,
    Br: float,
):
    """
    Copy of IMRPhenomHMMapParams in LALSimIMRPhenomHM.c, line 557
    """
    # Define function to output map params used depending on
    a = jnp.where(flm > fi, jnp.where(flm > fr, Ar, Am), Ai)
    b = jnp.where(flm > fi, jnp.where(flm > fr, Br, Bm), Bi)
    return a, b
