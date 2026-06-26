import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex
from ripplegw.constants import PI, MTSUN, MRSUN, MPC
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.waveforms import LALSimIMRPhenomX_precession as pPrec
from ripplegw.waveforms.initialize_MSA_system import IMRPhenomX_Initialize_MSA_System


from ripplegw.waveforms.IMRPhenomXHM import XLALSimIMRPhenomXHMGethlmModes, build_pWF22
from ripplegw.waveforms.IMRPhenomXPHM import (
    IMRPhenomXWignerdCoefficients_cosbeta,
    twist_22,
    BetaPowers,
    apply_polarization_rotation,
)  # spaghetti code! FIXME

jax.config.update("jax_enable_x64", True)


def gen_IMRPhenomXP_hphc(
    f: Float[Array, " n_freq"],
    theta: Float[Array, "12"],
    f_ref: float,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Generate PhenomXP frequency domain waveform.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, s1x, s1y, s1z, s2x, s2y, s2z, D, tc, phic, iota]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence
    iota: Inclination angle of the binary [between 0 and PI]

    f_ref: Reference frequency for the waveform

    Returns:
      hp (array): Strain of the plus polarization
      hc (array): Strain of the cross polarization
    """

    Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, D, tc, phic, iota = theta

    m1, m2 = Mc_eta_to_ms(jnp.array([Mc, eta]))

    ### Geometry first: need phiJ_Sf for co-precessing phase convention ###
    bigM = 1
    eta2 = jnp.power(eta, 2)
    chi1L = s1z
    chi2L = s2z
    total_mass = m1 + m2

    mass_1_fraction = m1 / total_mass
    mass_2_fraction = m2 / total_mass

    delta = mass_1_fraction - mass_2_fraction

    orbital_angular_momentum = (
        pPrec.flag_222_223_twoPN_non_spinning_orbitan_angular_momentum(
            eta, eta2, chi1L, chi2L, delta, jnp.power(jnp.pi, 2)
        )
    )
    Msec = (m1 + m2) * MTSUN
    piM = jnp.pi * Msec
    v_ref = jnp.cbrt(piM * f_ref)
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

    # Fused call: compute J0, thetaJN, kappa, and zeta_polarization in one pass.
    theta_JN, _Nz_Jf, _Nx_Jf, phiJ_Sf, kappa, zeta_polarisations = (
        pPrec.compute_thetaJN_kappa_and_zeta(
            mass_1_fraction,
            mass_2_fraction,
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
            LRef,
            phic,
            iota,
        )
    )

    # Co-precessing (2,2) waveform via XHM using MSA-averaged afinal_prec for fRING/fDAMP.
    # For PrecVersion=222 LAL uses fsflag=3 (MSA formula), not chiTot_perp (fsflag=4).
    # We call IMRPhenomX_Initialize_MSA_System to get SAv2, S1L_pav, S2L_pav and pass
    # them to build_pWF22 which computes afinal_prec_MSA internally.
    _msa_init = IMRPhenomX_Initialize_MSA_System(
        mass_1=m1,
        mass_2=m2,
        chi1x=s1x,
        chi1y=s1y,
        chi1z=s1z,
        chi2x=s2x,
        chi2y=s2y,
        chi2z=s2z,
        reference_frequency=f_ref,
    )
    pWF22_prec = build_pWF22(
        m1,
        m2,
        s1z,
        s2z,
        f_ref,
        msa_SAv2=_msa_init[15],
        msa_S1L_pav=_msa_init[32],
        msa_S2L_pav=_msa_init[33],
    )
    Mf = (m1 + m2) * f * MTSUN
    hlm_22 = XLALSimIMRPhenomXHMGethlmModes(
        Mf, pWF22_prec, phi0=0.0, ell_mm_pairs=[(2, 2)]
    )[(2, 2)]
    dist_m = D * MPC
    amp0 = (m1 + m2) * MRSUN * (m1 + m2) * MTSUN / dist_m
    h0_bare = hlm_22 * amp0 * jnp.exp(2j * PI * f * tc)

    # Compute MSA precession constants once (independent of emm and Mf) so they
    # are not redundantly recomputed for each of the 5 modes inside the vmap.
    _msa_setup = pPrec.compute_msa_precession_setup(
        m1,
        m2,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        f_ref,
        kappa,
        phiJ_Sf,
    )
    # Compute precession angles for all 4 unique emm values in a single batched call.
    # Modes 22 and 32 share emm=2, so we only need emm = 1, 2, 3, 4.
    _angles = pPrec.compute_evolved_spin_given_setup(Mf, 2, _msa_setup)

    # _angles is a tuple of 3 arrays, each shape (N_freq)
    alpha, eps, cos_beta = _angles
    # eps *= -1

    # Compute Wigner-d coefficients
    cBetah, sBetah = IMRPhenomXWignerdCoefficients_cosbeta(cos_beta)

    # Modes 22 and 32 – both use emm = 2
    beta_powers_2 = BetaPowers.from_half_angle_trig(cBetah, sBetah)
    cexp_i_alpha_2 = jnp.exp(1j * alpha)
    hp_twist_22, hc_twist_22 = twist_22(cexp_i_alpha_2, theta_JN, beta_powers_2)

    # epsilon is returned from compute_evolved_spin_given_setup; apply e^{-2i*epsilon}/2 scaling
    # (hc_twist_22 already includes the factor of i from the transfer function construction)
    # Factor of 2 matches LAL's cexp(-2.0*I*epsilon); XPHM applies this via eps_2 * 2.
    exp_neg_i_2epsilon = jnp.exp(-2j * eps) / 2.0
    _hp = h0_bare * hp_twist_22 * exp_neg_i_2epsilon
    _hc = h0_bare * hc_twist_22 * exp_neg_i_2epsilon

    # LALSim zeros the contribution for Mf >= 0.3 (f_max_prime). Setting
    # cos_beta=0 in compute_evolved_spin_using_msa does NOT produce a null
    # rotation (beta=pi/2 gives cBetah=sBetah=1/sqrt(2)), so we must
    # explicitly zero _hp/_hc here to match LALSim's behavior.
    inspiral_mask = Mf < 0.299999
    _hp = jnp.where(inspiral_mask, _hp, 0.0)
    _hc = jnp.where(inspiral_mask, _hc, 0.0)

    hp, hc = apply_polarization_rotation(zeta_polarisations, _hp, _hc)

    return hp, hc
