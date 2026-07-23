import jax
from ripplegw.interfaces import Waveform
from ripplegw.registry import register
from ripplegw.conversions import Mc_eta_to_ms
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex
from ripplegw.typing import FloatLike
from ripplegw.constants import MTSUN, MRSUN, MPC
from ripplegw.waveforms.spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)
from dataclasses import dataclass
from ripplegw.waveforms import LALSimIMRPhenomX_precession as pPrec
from ripplegw.waveforms.initialize_MSA_system import IMRPhenomX_Initialize_MSA_System
from ripplegw.waveforms.IMRPhenomXHM import XLALSimIMRPhenomXHMGethlmModes, build_pWF22


def compute_chip(
    m1: FloatLike,
    m2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
) -> FloatLike:
    """
    Effective precession spin parameter chi_p.

    LALSimIMRPhenomX_precession.c lines 260-275:
      A1 = 2 + 3*m2/(2*m1), A2 = 2 + 3*m1/(2*m2)
      chip = max(A1*m1^2*chi1_perp, A2*m2^2*chi2_perp) / (A1*m1^2)
    """
    chi1_perp = jnp.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = jnp.sqrt(chi2x**2 + chi2y**2)
    q = m2 / m1  # assumes m1 >= m2
    A1 = 2.0 + 1.5 * q
    A2 = 2.0 + 1.5 / q
    return jnp.maximum(A1 * chi1_perp * m1**2, A2 * chi2_perp * m2**2) / (A1 * m1**2)


def compute_chiTot_perp(
    m1: FloatLike,
    m2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
) -> FloatLike:
    """
    Total perpendicular spin parameter for afinal_prec (LAL PhenomXPFinalSpinMod=4, the default).

    LALSimIMRPhenomX_precession.c lines 237-241:
      STot_perp = |S1_perp + S2_perp|  where S_i = chi_i * (mi/M)^2
      chiTot_perp = STot_perp * M^2 / m1^2

    This is used by build_pWF22 (as its 'chip' argument) to compute afinal_prec via
      Sperp = chiTot_perp * mm1^2 = STot_perp
      afinal_prec = sign(a_aln) * sqrt(STot_perp^2 + a_aln^2)
    matching LAL's default fsflag=4 convention.
    """
    mm1 = m1 / (m1 + m2)
    mm2 = m2 / (m1 + m2)
    Sx = chi1x * mm1**2 + chi2x * mm2**2
    Sy = chi1y * mm1**2 + chi2y * mm2**2
    return jnp.sqrt(Sx**2 + Sy**2) / mm1**2


def generate_xphm(
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    distance: FloatLike,
    inclination: FloatLike,
    phi0: FloatLike,
    frequency_array: Float[Array, " n_freq"],
    reference_frequency: float,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """Generate IMRPhenomXPHM plus and cross polarizations."""
    Mf: Float[Array, " n_freq"] = pPrec.XLALSimIMRPhenomXUtilsHztoMf(
        frequency_array, mass_1 + mass_2
    )  # type: ignore[assignment]

    Mtot = mass_1 + mass_2

    # Overall amplitude prefactor from LAL's XLALSimPhenomUtilsFDamp0:
    # amp0 = Mtot * MRSUN * Mtot * MTSUN / distance
    # where Mtot is in solar masses and distance is in meters
    dist_m = distance * MPC  # distance in meters
    amp0 = Mtot * MRSUN * Mtot * MTSUN / dist_m

    # Mode order: [(2,1),(2,2),(3,2),(3,3),(4,4)]
    _ell_mm_pairs = [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]

    # Use XHM (PhenomXAS-based) modes as the co-precessing seed — matches LAL
    # default (TwistPhenomHM=0) and Bilby. XHM uses IMRPhenomX_TimeShift_22 for
    # the t0 convention, which is the correct PE coalescence-time reference.
    # phi0=0.0 follows LAL convention=1 (pWF->phi0 = 0 for the co-precessing modes).
    # Use MSA-averaged afinal_prec (fsflag=3) for PrecVersion=222, matching LAL default.
    _msa_init = IMRPhenomX_Initialize_MSA_System(
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
    pWF22 = build_pWF22(
        mass_1,
        mass_2,
        chi1z,
        chi2z,
        reference_frequency,
        msa_SAv2=_msa_init[15],
        msa_S1L_pav=_msa_init[32],
        msa_S2L_pav=_msa_init[33],
    )
    hlm_dict = XLALSimIMRPhenomXHMGethlmModes(
        Mf, pWF22, phi0=0.0, ell_mm_pairs=_ell_mm_pairs
    )

    hlms = (
        jnp.stack(
            [
                (-1 if ell % 2 != 0 else 1) * hlm_dict[(ell, mm)]
                for (ell, mm) in _ell_mm_pairs
            ]
        )
        * amp0
    )

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


def twistup(
    Mf: Float[Array, " n_freq"] | FloatLike,
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    phiRef_In: FloatLike,
    inclination: FloatLike,
    reference_frequency: float,
    hlm: Complex[Array, "n_modes n_freq"],
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
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
    theta_JN, _Nz_Jf, _Nx_Jf, phiJ_Sf, kappa, zeta_polarisations = (
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

    cBetah: FloatLike
    cBetah2: FloatLike
    cBetah3: FloatLike
    cBetah4: FloatLike
    cBetah5: FloatLike
    cBetah6: FloatLike
    cBetah7: FloatLike
    cBetah8: FloatLike
    sBetah: FloatLike
    sBetah2: FloatLike
    sBetah3: FloatLike
    sBetah4: FloatLike
    sBetah5: FloatLike
    sBetah6: FloatLike
    sBetah7: FloatLike
    sBetah8: FloatLike

    @classmethod
    def from_half_angle_trig(cls, cBetah: FloatLike, sBetah: FloatLike):
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


def twist_22(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
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


def twist_21(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
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


def twist_33(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
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


def twist_32(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
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


def twist_44(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
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


def apply_polarization_rotation(
    zeta_polarization: FloatLike, _hp, _hc
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
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


def IMRPhenomXWignerdCoefficients_cosbeta(
    cos_beta: Float[Array, " n_freq"],
) -> tuple[Float[Array, " n_freq"], Float[Array, " n_freq"]]:
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


@register("IMRPhenomXPHM", domain="FD", is_tidal=False, is_precessing=True)
class IMRPhenomXPHM(Waveform):
    """IMRPhenomXPHM frequency-domain waveform (precessing spins, higher-order modes).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_x",
            "s1_y",
            "s1_z",
            "s2_x",
            "s2_y",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXPHM waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_x``, ``s1_y``, ``s1_z``,
                ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        hp, hc = generate_xphm(
            m1,
            m2,
            params["s1_x"],
            params["s1_y"],
            params["s1_z"],
            params["s2_x"],
            params["s2_y"],
            params["s2_z"],
            params["d_L"],
            params["iota"],
            params["phase_c"],
            frequency,
            self.f_ref,
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXPHM(f_ref={self.f_ref})"
