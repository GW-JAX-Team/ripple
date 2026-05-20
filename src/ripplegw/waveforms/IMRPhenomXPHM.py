import jax
import jax.numpy as jnp
from .IMRPhenomD_QNMdata import QNMData_a, QNMData_fRD, QNMData_fdamp
from ..constants import PI, MSUN, MTSUN, MRSUN, MPC
from jaxtyping import Array, Float, Integer
from .spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)
from dataclasses import dataclass
from . import LALSimIMRPhenomX_precession as pPrec


# Some pre-XPHM ripple code
from .IMRPhenomD_utils import (
    EradRational0815,
    get_coeffs,
    get_transition_frequencies_from_fRD_fdamp,
)
from .IMRPhenomD import Phase as IMRPhenomD_Phase
from .IMRPhenomD import IMRPhenDAmplitude_NoCut
from .IMRPhenomD import get_IIb_raw_phase
from .IMRPhenomPv2_utils import FinalSpin0815
from .IMRPhenomXHM import XLALSimIMRPhenomXHMGethlmModes, build_pWF22


# Phase shift due to leading order complex amplitude
# [L.Blancet, arXiv:1310.1528 (Sec. 9.5)]
# "Spherical hrmonic modes for numerical relativity"
# List of phase shifts: the index is the azimuthal number m
CSHIFT = jnp.array([0.0, PI / 2.0, 0.0, -PI / 2.0, PI, PI / 2.0, 0.0])


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
    """Generate IMRPhenomXPHM plus and cross polarizations."""
    Mf = XLALSimIMRPhenomXUtilsHztoMf(frequency_array, mass_1 + mass_2)

    Mtot = mass_1 + mass_2

    # Overall amplitude prefactor from LAL's XLALSimPhenomUtilsFDamp0:
    # amp0 = Mtot * MRSUN * Mtot * MTSUN / distance
    # where Mtot is in solar masses and distance is in meters
    dist_m = distance * MPC  # distance in meters
    amp0 = Mtot * MRSUN * Mtot * MTSUN / dist_m

    # Mode order: [(2,1),(2,2),(3,2),(3,3),(4,4)]
    _ell_mm_pairs = [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]
    _mode_array = jnp.array([[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]], dtype=jnp.int32)

    # Build the co-precessing seed used by LAL's current XPHM validation path.
    # The repo's LAL-side helpers explicitly enable TwistPhenomHM=1, which twists
    # up the legacy PhenomHM modes rather than the XHM modes.
    hlm = XLALSimIMRPhenomHMGethlmModes(
        frequency_array,
        mass_1 * MSUN,
        mass_2 * MSUN,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        0.0,
        0.0,
        reference_frequency,
        {"ModeArray": _mode_array},
    )

    ells = _mode_array[:, 0]
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


# TODO this is a temporary function for testing with XHM and should become the standard version once tested
def DEV_generate_xphm_xhm(
    freqs: Array,
    theta: Array,
    f_ref: float,
):
    """Generate IMRPhenomXPHM plus and cross polarizations."""
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, distance, tc, phi_ref, inclination = theta

    Mtot = m1 + m2  # solar masses
    M_s = Mtot * MTSUN  # total mass in seconds
    dist_m = distance * MPC  # distance in metres
    amp0 = Mtot * MRSUN * Mtot * MTSUN / dist_m

    Mf = XLALSimIMRPhenomXUtilsHztoMf(freqs, m1 + m2)

    freqs_geom = freqs * M_s
    pWF22 = build_pWF22(m1, m2, s1z, s2z, f_ref)

    ell_mm_pairs = [(2, 1), (2, 2), (3, 3), (3, 2), (4, 4)]
    _mode_array = jnp.array([[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]], dtype=jnp.int32)

    hlm_dict = XLALSimIMRPhenomXHMGethlmModes(
        freqs_geom, pWF22, phi0=phi_ref, ell_mm_pairs=ell_mm_pairs
    )
    hlm = jnp.stack(
        [
            jnp.zeros_like(hlm_dict[(2, 1)]),  # hlm_dict[(2, 1)],
            hlm_dict[(2, 2)],
            jnp.zeros_like(hlm_dict[(2, 1)]),  # hlm_dict[(3, 3)],
            jnp.zeros_like(hlm_dict[(2, 1)]),  # hlm_dict[(3, 2)],
            jnp.zeros_like(hlm_dict[(2, 1)]),  # hlm_dict[(4, 4)],
        ],
        axis=0,
    )

    ells = _mode_array[:, 0]
    minus1l = jnp.where(ells % 2 != 0, -1, 1)
    hlms = minus1l[:, None] * hlm * amp0

    hp, hc = twistup(
        Mf,
        m1,
        m2,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        phi_ref,
        inclination,
        f_ref,
        hlms,
    )

    return hp, hc


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
    """
    Rotate the co-precessing frame hlm modes into the inertial (J-frame) polarisations hp, hc.
    Implementation follows the lalsimulation function IMRPhenomXPHMTwistUps.

    Computes the precession angles (alpha, beta, epsilon) at each frequency via MSA,
    applies the per-mode Wigner-d rotation and spherical-harmonic projection, sums over
    modes (21, 22, 32, 33, 44), and applies a final polarisation rotation by zeta.

    Args:
        Mf: Dimensionless frequency array (len N).
        mass_1, mass_2: Component masses in solar masses.
        chi1x/y/z, chi2x/y/z: Dimensionless spin components in the L-frame.
        phiRef_In: Reference orbital phase (rad).
        inclination: Inclination angle between J and line of sight (rad).
        reference_frequency: Reference frequency for spin evolution (Hz).
        hlm: Co-precessing frame modes, shape (n_modes, N).

    Returns:
        hp, hc: Plus and cross polarisations, shape (N,).
    """

    # We are not using multibanding for angles.
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

    # Fused call: compute J0, thetaJN, kappa, and zeta_polarization in one pass
    # (avoids recomputing J0/thetaJ_Sf/phiJ_Sf twice).
    theta_JN, Nz_Jf, Nx_Jf, phiJ_Sf, kappa, zeta_polarisations = (
        pPrec.compute_thetaJN_kappa_and_zeta(
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
    )

    # Compute MSA precession constants once (independent of emm and Mf) so they
    # are not redundantly recomputed for each of the 5 modes inside the vmap.
    _msa_setup = pPrec.compute_msa_precession_setup(
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
    )
    # Compute precession angles for all 4 unique emm values in a single batched call.
    # Modes 22 and 32 share emm=2, so we only need emm = 1, 2, 3, 4.
    _batched_angles = jax.vmap(
        pPrec.compute_evolved_spin_given_setup, in_axes=(None, 0, None)
    )(Mf, jnp.array([1, 2, 3, 4]), _msa_setup)
    # _batched_angles is a tuple of 3 arrays, each shape (4, N_freq)
    alpha_all = _batched_angles[0]  # (4, N)
    eps_all = _batched_angles[1]  # (4, N)
    cos_beta_all = _batched_angles[2]  # (4, N)

    # Unpack per-emm results
    alpha_1, alpha_2, alpha_3, alpha_4 = alpha_all
    eps_1, eps_2, eps_3, eps_4 = eps_all

    # Compute Wigner-d coefficients for all 4 emm values at once
    cBetah_all, sBetah_all = IMRPhenomXWignerdCoefficients_cosbeta(cos_beta_all)

    # Mode 21 – emm = 1
    beta_powers_1 = BetaPowers.from_half_angle_trig(cBetah_all[0], sBetah_all[0])
    hp_21, hc_21 = twist_21(jnp.exp(1j * alpha_1), theta_JN, beta_powers_1)

    # Modes 22 and 32 – both use emm = 2
    beta_powers_2 = BetaPowers.from_half_angle_trig(cBetah_all[1], sBetah_all[1])
    cexp_i_alpha_2 = jnp.exp(1j * alpha_2)
    hp_22, hc_22 = twist_22(cexp_i_alpha_2, theta_JN, beta_powers_2)
    hp_32, hc_32 = twist_32(cexp_i_alpha_2, theta_JN, beta_powers_2)

    # Mode 33 – emm = 3
    beta_powers_3 = BetaPowers.from_half_angle_trig(cBetah_all[2], sBetah_all[2])
    hp_33, hc_33 = twist_33(jnp.exp(1j * alpha_3), theta_JN, beta_powers_3)

    # Mode 44 – emm = 4
    beta_powers_4 = BetaPowers.from_half_angle_trig(cBetah_all[3], sBetah_all[3])
    hp_44, hc_44 = twist_44(jnp.exp(1j * alpha_4), theta_JN, beta_powers_4)

    # Stack into (5, N) matching the old vmap output layout
    hp_twist_all_modes = jnp.stack([hp_21, hp_22, hp_32, hp_33, hp_44], axis=0)
    hc_twist_all_modes = jnp.stack([hc_21, hc_22, hc_32, hc_33, hc_44], axis=0)
    epsilon_all_modes = jnp.stack(
        [eps_1 * 1, eps_2 * 2, eps_2 * 2, eps_3 * 3, eps_4 * 4], axis=0
    )

    exp_neg_i_epsilon = jnp.exp(-1j * epsilon_all_modes.T) / 2
    _hp = jnp.sum(hlm.T * hp_twist_all_modes.T * exp_neg_i_epsilon, axis=1)
    _hc = jnp.sum(hlm.T * hc_twist_all_modes.T * exp_neg_i_epsilon, axis=1)

    # LALSim zeros the contribution for Mf >= 0.3 (f_max_prime). Setting
    # cos_beta=0 in compute_evolved_spin_using_msa does NOT produce a null
    # rotation (beta=pi/2 gives cBetah=sBetah=1/sqrt(2)), so we must
    # explicitly zero _hp/_hc here to match LALSim's behavior.
    inspiral_mask = Mf < 0.299999
    _hp = jnp.where(inspiral_mask, _hp, 0.0)
    _hc = jnp.where(inspiral_mask, _hc, 0.0)

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

    cBetah: Float
    cBetah2: Float
    cBetah3: Float
    cBetah4: Float
    cBetah5: Float
    cBetah6: Float
    cBetah7: Float
    cBetah8: Float
    sBetah: Float
    sBetah2: Float
    sBetah3: Float
    sBetah4: Float
    sBetah5: Float
    sBetah6: Float
    sBetah7: Float
    sBetah8: Float

    @classmethod
    def from_half_angle_trig(cls, cBetah: Float, sBetah: Float):
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
    """
    Compute the twisting contributions for l=2, m'=2 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (2,2) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha

    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha

    # shape (5, N): rows indexed by m+2 for m in -2..2
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

    Y2mA = jnp.array(
        [
            compute_sminus2_l2(theta_JN, m=-2),
            compute_sminus2_l2(theta_JN, m=-1),
            compute_sminus2_l2(theta_JN, m=0),
            compute_sminus2_l2(theta_JN, m=1),
            compute_sminus2_l2(theta_JN, m=2),
        ]
    )

    # Wigner-d coefficients – shape (5, N)
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
    d2m2 = jnp.array([d22[4], -d22[3], d22[2], -d22[1], d22[0]])

    # Vectorised sum over m=-2..2.  The -m+2 index pattern equals reversed order.
    # Shapes: cexp_im_alpha_l2 (5,N), d2m2 (5,N), Y2mA (5,) -> broadcast to (5,N)
    A2m2emm = cexp_im_alpha_l2[::-1] * d2m2 * Y2mA[:, None]
    A22emmstar = cexp_im_alpha_l2 * d22 * jnp.conj(Y2mA)[:, None]
    hp_sum = jnp.sum(A2m2emm + A22emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A2m2emm - A22emmstar), axis=0)

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
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha

    # shape (5, N): rows indexed by m+2 for m in -2..2
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

    Y2mA = jnp.array(
        [
            compute_sminus2_l2(theta_JN, m=-2),
            compute_sminus2_l2(theta_JN, m=-1),
            compute_sminus2_l2(theta_JN, m=0),
            compute_sminus2_l2(theta_JN, m=1),
            compute_sminus2_l2(theta_JN, m=2),
        ]
    )

    # Wigner-d coefficients for m'=1 – shape (5, N)
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
    d2m1 = jnp.array([-d21[4], d21[3], -d21[2], d21[1], -d21[0]])

    # Vectorised sum over m=-2..2.  The -m+2 index pattern equals reversed order.
    A2m1emm = cexp_im_alpha_l2[::-1] * d2m1 * Y2mA[:, None]
    A21emmstar = cexp_im_alpha_l2 * d21 * jnp.conj(Y2mA)[:, None]
    hp_sum = jnp.sum(A2m1emm + A21emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A2m1emm - A21emmstar), axis=0)

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
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha

    # shape (7, N): rows indexed by m+3 for m in -3..3
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

    Y3mA = jnp.array(
        [
            compute_sminus2_l3(theta=theta_JN, m=-3),
            compute_sminus2_l3(theta=theta_JN, m=-2),
            compute_sminus2_l3(theta=theta_JN, m=-1),
            compute_sminus2_l3(theta=theta_JN, m=0),
            compute_sminus2_l3(theta=theta_JN, m=1),
            compute_sminus2_l3(theta=theta_JN, m=2),
            compute_sminus2_l3(theta=theta_JN, m=3),
        ]
    )

    # Wigner-d coefficients for m'=3 – shape (7, N)
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
    d3m3 = jnp.array([d33[6], -d33[5], d33[4], -d33[3], d33[2], -d33[1], d33[0]])

    # Vectorised sum over m=-3..3.  The -m+3 index pattern equals reversed order.
    A3m3emm = cexp_im_alpha_l3[::-1] * d3m3 * Y3mA[:, None]
    A33emmstar = cexp_im_alpha_l3 * d33 * jnp.conj(Y3mA)[:, None]
    hp_sum = jnp.sum(A3m3emm - A33emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A3m3emm + A33emmstar), axis=0)

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
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha

    # shape (7, N): rows indexed by m+3 for m in -3..3
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

    Y3mA = jnp.array(
        [
            compute_sminus2_l3(theta=theta_JN, m=-3),
            compute_sminus2_l3(theta=theta_JN, m=-2),
            compute_sminus2_l3(theta=theta_JN, m=-1),
            compute_sminus2_l3(theta=theta_JN, m=0),
            compute_sminus2_l3(theta=theta_JN, m=1),
            compute_sminus2_l3(theta=theta_JN, m=2),
            compute_sminus2_l3(theta=theta_JN, m=3),
        ]
    )

    # Wigner-d coefficients for m'=2 – shape (7, N)
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
    d3m2 = jnp.array([-d32[6], d32[5], -d32[4], d32[3], -d32[2], d32[1], -d32[0]])

    # Vectorised sum over m=-3..3.  The -m+3 index pattern equals reversed order.
    A3m2emm = cexp_im_alpha_l3[::-1] * d3m2 * Y3mA[:, None]
    A32emmstar = cexp_im_alpha_l3 * d32 * jnp.conj(Y3mA)[:, None]
    hp_sum = jnp.sum(A3m2emm - A32emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A3m2emm + A32emmstar), axis=0)

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
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_4i_alpha = cexp_i_alpha * cexp_3i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha
    cexp_m4i_alpha = cexp_mi_alpha * cexp_m3i_alpha

    # shape (9, N): rows indexed by m+4 for m in -4..4
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

    Y4mA = jnp.array(
        [
            compute_sminus2_l4(theta=theta_JN, m=-4),
            compute_sminus2_l4(theta=theta_JN, m=-3),
            compute_sminus2_l4(theta=theta_JN, m=-2),
            compute_sminus2_l4(theta=theta_JN, m=-1),
            compute_sminus2_l4(theta=theta_JN, m=0),
            compute_sminus2_l4(theta=theta_JN, m=1),
            compute_sminus2_l4(theta=theta_JN, m=2),
            compute_sminus2_l4(theta=theta_JN, m=3),
            compute_sminus2_l4(theta=theta_JN, m=4),
        ]
    )

    # Wigner-d coefficients for m'=4 – shape (9, N)
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
    d4m4 = jnp.array(
        [d44[8], -d44[7], d44[6], -d44[5], d44[4], -d44[3], d44[2], -d44[1], d44[0]]
    )

    # Vectorised sum over m=-4..4.  The -m+4 index pattern equals reversed order.
    A4m4emm = cexp_im_alpha_l4[::-1] * d4m4 * Y4mA[:, None]
    A44emmstar = cexp_im_alpha_l4 * d44 * jnp.conj(Y4mA)[:, None]
    hp_sum = jnp.sum(A4m4emm + A44emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A4m4emm - A44emmstar), axis=0)

    return hp_sum, hc_sum


def apply_polarization_rotation(zeta_polarization, _hp, _hc):
    """Apply polarization rotation to waveform components.

    Args:
        zeta_polarization (float): Polarization angle.
        _hp (array_like): Plus polarization component (unrotated).
        _hc (array_like): Cross polarization component (unrotated).

    Returns:
        tuple[array_like, array_like]: Rotated plus (hp) and cross (hc) polarizations.
    """
    cosPolFac = jnp.cos(2.0 * zeta_polarization)
    sinPolFac = jnp.sin(2.0 * zeta_polarization)

    hp = cosPolFac * _hp + sinPolFac * _hc
    hc = cosPolFac * _hc - sinPolFac * _hp

    return hp, hc


def IMRPhenomXWignerdCoefficients_cosbeta(cos_beta):
    """
    Compute cos(beta/2) and sin(beta/2) from cos(beta).

    Uses half-angle formulas:
    - cos(beta/2) = sqrt((1 + cos(beta)) / 2)
    - sin(beta/2) = sqrt((1 - cos(beta)) / 2)

    Args:
        cos_beta (float or array): cos(beta).

    Returns:
        tuple[float or array, float or array]: (cos(beta/2), sin(beta/2)), both always non-negative.
    """
    # Note that the results here are indeed always non-negative
    cos_beta_half = jnp.sqrt(jnp.abs(1.0 + cos_beta) / 2.0)  # cos(beta/2)
    sin_beta_half = jnp.sqrt(jnp.abs(1.0 - cos_beta) / 2.0)  # sin(beta/2)

    return cos_beta_half, sin_beta_half


def XLALSimIMRPhenomXUtilsHztoMf(fHz: Float, Mtot_Msun: Float) -> Float:
    """
    Convert frequency from Hz to geometric units (Mf).

    Args:
        fHz (Float): Frequency in Hz.
        Mtot_Msun (Float): Total mass in solar masses.

    Returns:
        Float: Geometric frequency Mf.
    """
    # Mtot in seconds = Mtot_Msun * MTSUN
    return fHz * Mtot_Msun * MTSUN


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
    """Compute all hlm modes for IMRPhenomXPHM. JAX translation of XLALSimIMRPhenomHMGethlmModes."""
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

    # Pre-compute mode-independent PhenomD quantities once (used by phase, amplitude, t0)
    theta = jnp.array([pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]])
    PhenomD_coeffs = get_coeffs(theta)
    M_s = pHM["Mtot"] * MTSUN
    f_RD = pHM["Mf_RD_22_PhenomD"] / M_s
    f_damp = pHM["Mf_DM_22_PhenomD"] / M_s
    PhenomD_transition_freqs = get_transition_frequencies_from_fRD_fdamp(
        theta, PhenomD_coeffs[5], PhenomD_coeffs[6], f_RD, f_damp
    )

    # Compute f4 and t0 once (mode-independent)
    gamma2, gamma3 = PhenomD_coeffs[5], PhenomD_coeffs[6]
    f4 = jnp.where(
        gamma2 >= 1,
        jnp.abs(f_RD + (-f_damp * gamma3) / gamma2),
        jnp.abs(f_RD + (f_damp * (-1 + jnp.sqrt(1 - gamma2**2)) * gamma3) / gamma2),
    )
    t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, theta, PhenomD_coeffs, f_RD, f_damp)

    # Store pre-computed quantities in pHM for per-mode functions
    pHM["_theta"] = theta
    pHM["_PhenomD_coeffs"] = PhenomD_coeffs
    pHM["_M_s"] = M_s
    pHM["_f_RD"] = f_RD
    pHM["_f_damp"] = f_damp
    pHM["_PhenomD_transition_freqs"] = PhenomD_transition_freqs
    pHM["_t0"] = t0

    # line 1316
    # compute the reference phase shift need to align the waveform so that
    # the phase is equal to phiRef at the reference frequency f_ref.
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
    freqs_geom: Float, pHM: dict, ell: int, mm: int, phi0: Float
):
    """
    Implementation of IMRPhenomHMEvaluateOnehlmMode in LALSimIMRPhenomHM.c
    """

    # generate phase and amplitude for single l,m mode
    phase_lm = IMRPhenomHMPhase(freqs_geom, pHM, ell, mm)
    amp_lm = IMRPhenomHMAmplitude(freqs_geom, pHM, ell, mm)

    # Use pre-computed t0 from pHM (mode-independent)
    t0 = pHM["_t0"]

    Mf = freqs_geom
    phase_term1 = -t0 * (Mf - pHM["Mf_ref"])
    phase_term2 = phase_lm - (mm * phi0)
    return amp_lm * jnp.exp(-1j * (phase_term1 + phase_term2))


def XLALSimPhenomUtilsPhenomPv2FinalSpin(
    m1: Float, m2: Float, chi1_l: Float, chi2_l: Float, chip: Float
):
    """
    Implementation of XLALSimPhenomUtilsPhenomPv2FinalSpin in LALSimPhenomUtils.c
    Assuming m1 >= m2
    """

    M = m1 + m2
    eta = m1 * m2 / (M * M)

    q_factor = m1 / M

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
    Implementation of init_PhenomHM_Storage in LALSimIMRPhenomHM.c
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
        jnp.arange(len(ModeArray), dtype=jnp.int32)
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
    ell: Integer, mm: Integer, finalmass: Float, finalspin: Float
):
    """
    Implementation of IMRPhenomHMGetRingdownFrequency in LALSimIMRPhenomHM.c
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


def SimRingdownCW_KAPPA(jf: Float, ell: Integer, emm: Integer):
    """
    Domain mapping for dimnesionless BH spin
    """
    alpha = jnp.log(2.0 - jf) / jnp.log(3)
    beta = 1.0 / (2.0 + ell - jnp.abs(emm))
    return alpha**beta


def SimRingdownCW_CW07102016(kappa: Float, ell: Integer, input_m: Integer, n: int):
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
    """Map input frequency Mflm to the effective 22-mode frequency Mf22 for the (ell, mm) mode."""
    # Mflm here has the same meaning as Mf_wf in XLALSimIMRPhenomHMFreqDomainMapHM (old deleted function).
    # Following variables not used in this funciton but are returned in IMRPhenomHMFreqDomainMapParams
    a, b = IMRPhenomHMFreqDomainMapParams(Mflm, ell, mm, pHM, AmpFlag)
    Mf22 = a * Mflm + b
    return Mf22


def IMRPhenomHMAmplitude(freqs_geom: Array, pHM: dict, ell: int, mm: int):
    """
    Returns IMRPhenomHM amplitude evaluated at a set of input frequencies for the l,m mode
    Implementation of IMRPhenomHMAmplitude in LALSimIMRPhenomHM.c
    """

    # scale input frequencies according to PhenomHM model
    # LL: Map the input domain (frequencies) for this ell mm multipole
    # to those appropirate for the ell=|mm| multipole
    freqs_amp = IMRPhenomHMFreqDomainMap(freqs_geom, ell, mm, pHM, AmpFlag=True)

    # LL: Compute the PhenomD Amplitude at the mapped l=m=2 fequencies
    # NOTE: Use IMRPhenDAmplitude_NoCut instead of IMRPhenomD_Amp because
    # the mapped frequencies can exceed fM_CUT for higher modes

    # Use pre-computed quantities from pHM
    theta = pHM["_theta"]
    PhenomD_coeffs = pHM["_PhenomD_coeffs"]
    PhenomD_transition_freqs = pHM["_PhenomD_transition_freqs"]

    amps_normalized = IMRPhenDAmplitude_NoCut(
        freqs_amp / pHM["_M_s"],
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
    delta = jnp.sqrt(jnp.maximum(1.0 - 4 * eta, 0.0))
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
    Implementation of IMRPhenomHMPhase in LALSimIMRPhenomHM.c
    """

    q = {}
    q = IMRPhenomHMPhasePreComp(q, ell, mm, pHM)

    # Get mode index for array lookup
    mode_idx = pHM["mode_index_map"][ell, mm]
    Rholm = pHM["Rholm"][mode_idx]
    Taulm = pHM["Taulm"][mode_idx]

    # Use pre-computed quantities from pHM
    theta = pHM["_theta"]
    PhenomD_coeffs = pHM["_PhenomD_coeffs"]
    M_s = pHM["_M_s"]
    PhenomD_transition_freqs = pHM["_PhenomD_transition_freqs"]

    # Fused piecewise phase: single IMRPhenomD_Phase call with per-region frequency mapping
    is_A = freqs_geom <= q["fi"]
    is_B = (freqs_geom > q["fi"]) & (freqs_geom <= q["fr"])
    # is_C = ~is_A & ~is_B

    # Map frequencies: each region uses its own (a, b) pair
    Mf_fused = jnp.where(
        is_A,
        (q["ai"] * freqs_geom + q["bi"]) / M_s,
        jnp.where(
            is_B,
            (q["am"] * freqs_geom + q["bm"]) / M_s,
            (q["ar"] * freqs_geom + q["br"]) / M_s,
        ),
    )

    # Single vectorized IMRPhenomD_Phase call
    phi_raw = IMRPhenomD_Phase(
        Mf_fused, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
    )

    # Apply per-region scaling and offsets
    inv_a = jnp.where(
        is_A, 1.0 / q["ai"], jnp.where(is_B, 1.0 / q["am"], 1.0 / q["ar"])
    )
    offset = jnp.where(
        is_A,
        0.0,
        jnp.where(
            is_B,
            -q["PhDBconst"] + q["PhDBAterm"],
            -q["PhDCconst"] + q["tmpphaseC"],
        ),
    )

    phase = CSHIFT[jnp.abs(mm)] + phi_raw * inv_a + offset

    return phase


def IMRPhenomHMPhasePreComp(q: dict, ell: int, emm: int, pHM: dict):
    """
    Implementation of IMRPhenomHMPhasePreComp in LALSimIMRPhenomHM.c
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

    # Use pre-computed quantities from pHM
    M_s = pHM["_M_s"]
    theta = pHM["_theta"]
    PhenomD_coeffs = pHM["_PhenomD_coeffs"]
    PhenomD_transition_freqs = pHM["_PhenomD_transition_freqs"]

    # Batch all 4 scalar Phase evaluations into a single vectorized call
    PhDBMf = q["am"] * fi + q["bm"]
    PhDCMf = q["ar"] * fr + q["br"]
    PhDBAMf = q["ai"] * fi + q["bi"]
    Mfr_mapped_Mf = q["am"] * fr + q["bm"]

    boundary_freqs = jnp.array([PhDBMf, PhDCMf, PhDBAMf, Mfr_mapped_Mf]) / M_s
    boundary_phases = IMRPhenomD_Phase(
        boundary_freqs, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
    )

    q["PhDBconst"] = boundary_phases[0] / q["am"]
    q["PhDCconst"] = boundary_phases[1] / q["ar"]
    q["PhDBAterm"] = boundary_phases[2] / q["ai"]
    q["tmpphaseC"] = boundary_phases[3] / q["am"] - q["PhDBconst"] + q["PhDBAterm"]
    return q


def IMRPhenomHMFreqDomainMapParams(
    flm: float,  # input waveform frequency
    ell: int,  # spherical harmonics ell mode
    mm: int,  # spherical harmonics m mode
    pHM: dict,
    ampFlag: bool,  # is ==1 then computes for amplitude, if ==0 then computes for phase
):
    """
    Implementation of the phase computation of IMRPhenomHMFreqDomainMapParams in LALSimIMRPhenomHM.c
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
    Implementation of IMRPhenomHMSlopeAmAndBm in LALSimIMRPhenomHM.c
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
    Implementation of IMRPhenomHMTrd in LALSimIMRPhenomHM.c
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
    flm: Float,
    fi: Float,
    fr: Float,
    Ai: Float,
    Bi: Float,
    Am: Float,
    Bm: Float,
    Ar: Float,
    Br: Float,
):
    """
    Implementation of IMRPhenomHMMapParams in LALSimIMRPhenomHM.c, line 557
    """
    # Define function to output map params used depending on
    a = jnp.where(flm > fi, jnp.where(flm > fr, Ar, Am), Ai)
    b = jnp.where(flm > fi, jnp.where(flm > fr, Br, Bm), Bi)
    return a, b


def XLALSimPhenomUtilsChiP(m1, m2, s1x, s1y, s2x, s2y):
    """
    Compute the effective precession parameter chip.

    This is a JAX translation of LALSimIMRPhenomUtils.c XLALSimPhenomUtilsChiP.

    Args:
        m1 (float or array): Mass of companion 1 (solar masses).
        m2 (float or array): Mass of companion 2 (solar masses).
        s1x (float or array): x-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1).
        s1y (float or array): y-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1).
        s2x (float or array): x-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1).
        s2y (float or array): y-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1).

    Returns:
        float or array: Effective precession parameter chip.
    """
    m1_2 = m1 * m1
    m2_2 = m2 * m2

    # Magnitude of the spin projections in the orbital plane
    S1_perp = m1_2 * jnp.sqrt(s1x * s1x + s1y * s1y)
    S2_perp = m2_2 * jnp.sqrt(s2x * s2x + s2y * s2y)

    A1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    A2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp

    num = jnp.where(ASp2 > ASp1, ASp2, ASp1)
    den = jnp.where(m2 > m1, A2 * m2_2, A1 * m1_2)
    chip = num / den

    return chip
