"""by Robin Chan"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

import ripplegw.waveforms.cbc.IMRPhenomX.LALSimIMRPhenomX_precession as pPrec
from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import Mc_eta_to_ms, lambda_tildes_to_lambdas
from ripplegw.interfaces import DistanceScaledWaveform, FrequencyDomainWaveform
from ripplegw.registry import register
from ripplegw.typing import FloatLike
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomXAS_NRTidalv3 import (
    _amplitude_of,
    _bbh_amp_psi,
    _phase_of,
)
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXHM import build_pWF22
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM import (
    BetaPowers,
    IMRPhenomXWignerdCoefficients_cosbeta,
    apply_polarization_rotation,
    twist_22,
)
from ripplegw.waveforms.cbc.IMRPhenomX.initialize_MSA_system import (
    IMRPhenomX_Initialize_MSA_System,
    lal_M_sec,
)

jax.config.update("jax_enable_x64", True)


def gen_IMRPhenomXP_NRTidalv3(
    f: Float[Array, " n_freq"],
    theta: Float[Array, "14"],
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """Generate IMRPhenomXP_NRTidalv3 frequency-domain plus and cross polarizations.

    ``theta`` = [Mchirp, eta, s1x, s1y, s1z, s2x, s2y, s2z,
    lambda_tilde/lambda1, delta_lambda_tilde/lambda2, D, tc, phic, iota].
    """

    # --- Set up precession variables ---
    Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, l1, l2, D, tc, phic, iota = theta

    m1, m2 = Mc_eta_to_ms(jnp.array([Mc, eta]))

    if use_lambda_tildes:
        l1, l2 = lambda_tildes_to_lambdas(jnp.array([l1, l2, m1, m2]))

    theta_intrinsic_XAS = jnp.array([m1, m2, s1z, s2z, l1, l2])
    theta_extrinsic = jnp.array([D, tc, phic])

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
    # LAL: Mf = pWF->M_sec * f with M_sec built from round-tripped solar masses;
    # (m1+m2)*f*MTSUN differs by up to 1 ULP, which the near-degenerate MSA S^2
    # cubic amplifies into the Euler angles.
    Msec_lal = lal_M_sec(m1, m2)
    Mf = f * Msec_lal

    chip = pWF22_prec.get("chip", 0.0)
    # Use afinal_prec directly for fRING/fDAMP to avoid the chip roundtrip losing
    # information when afinal_prec < a_aln (clamped chip=0 case, common at low chi_p).
    a_prec_override = pWF22_prec.get("afinal", None)

    # Co-precessing frame: phic must be 0 here. In LAL's XP convention the
    # co-precessing mode uses phi0=0 (phase zeroed at f_ref); phic enters only
    # through the precession geometry (phiJ_Sf -> epsilon_0).  Including 2*phic
    # in the phase would double-count it, producing a 2*phic offset vs LAL.
    theta_extrinsic_coprec = jnp.array([D, tc, 0.0])
    bbh_amp, bbh_psi = _bbh_amp_psi(
        f,
        theta_intrinsic_XAS,
        theta_extrinsic,
        chip=chip,
        a_prec_override=a_prec_override,
    )
    amp_22 = _amplitude_of(
        f, theta_intrinsic_XAS, theta_extrinsic, bbh_amp, no_taper=no_taper
    )
    phase_22 = _phase_of(
        f,
        f_ref,
        theta_intrinsic_XAS,
        theta_extrinsic_coprec,
        bbh_psi,
        no_taper=no_taper,
        chip=chip,
        a_prec_override=a_prec_override,
    )

    h0_coprec = amp_22 * jnp.exp(1j * phase_22)

    # XAS uses amp0_XAS = 2*sqrt(5/(64*π)) * M_s²*C/(D*MPC); twist_22 expects the XP/XHM
    # convention amp0_XP = M_s²*C/(D*MPC).  Divide out the extra prefactor so the mode
    # amplitude matches what the non-tidal gen_IMRPhenomXP passes to twist_22.
    h0_coprec = h0_coprec / (2.0 * jnp.sqrt(5.0 / (64.0 * PI)))

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
    # XP_NRTidalv3 carries the (2,2) mode only, so the precession angles are
    # needed at emm=2.
    _angles = pPrec.compute_evolved_spin_given_setup(Mf, 2, _msa_setup)

    # alpha, eps, cos_beta are arrays of shape (N_freq).
    alpha, eps, cos_beta = _angles

    # Compute Wigner-d coefficients
    cBetah, sBetah = IMRPhenomXWignerdCoefficients_cosbeta(cos_beta)

    # (2,2) twist: emm = 2
    beta_powers_2 = BetaPowers.from_half_angle_trig(cBetah, sBetah)
    cexp_i_alpha_2 = jnp.exp(1j * alpha)
    hp_twist_22, hc_twist_22 = twist_22(cexp_i_alpha_2, theta_JN, beta_powers_2)

    # LAL grouping (IMRPhenomXPHMTwistUp, LALSimIMRPhenomXPHM.c:2192, 2204-2205):
    # eps_phase = (cexp(-mprime*i*eps) * hlm) / 2, then hp = eps_phase * hp_sum.
    # Complex multiplication is not associative, so mirror the order exactly.
    eps_phase_hP = (jnp.exp(-2j * eps) * h0_coprec) / 2.0
    _hp = eps_phase_hP * hp_twist_22
    _hc = eps_phase_hP * hc_twist_22

    # LALSim zeroes the contribution above Mf = f_max_prime*M_sec — inclusive
    # comparison, fCutDef in {0.3, 0.33}; see mf_twist_cutoff.  Setting
    # cos_beta=0 in compute_evolved_spin_given_setup does NOT produce a null
    # rotation (beta=pi/2 gives cBetah=sBetah=1/sqrt(2)), so we must
    # explicitly zero _hp/_hc here to match LALSim's behavior.
    inspiral_mask = Mf <= pPrec.mf_twist_cutoff(eta, s1z, s2z, Msec_lal)
    _hp = jnp.where(inspiral_mask, _hp, 0.0)
    _hc = jnp.where(inspiral_mask, _hc, 0.0)

    hp, hc = apply_polarization_rotation(zeta_polarisations, _hp, _hc)

    return hp, hc


def gen_IMRPhenomXP_NRTidalv3_hphc(
    f: Float[Array, " n_freq"],
    theta: Float[Array, "14"],
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """Alias of ``gen_IMRPhenomXP_NRTidalv3`` for the ``*_hphc`` naming the
    other precessing generators use."""
    return gen_IMRPhenomXP_NRTidalv3(f, theta, f_ref, use_lambda_tildes, no_taper)


@register("IMRPhenomXP_NRTidalv3", is_tidal=True, is_precessing=True)
class IMRPhenomXP_NRTidalv3(FrequencyDomainWaveform, DistanceScaledWaveform):
    """IMRPhenomXP_NRTidalv3 frequency-domain waveform (precessing spins, NRTidalv3 tides).

    Attributes:
        f_ref (float): Reference frequency in Hz.
        use_lambda_tildes (bool): If True, expects ``lambda_tilde`` /
            ``delta_lambda_tilde``; otherwise ``lambda_1`` / ``lambda_2``.
        no_taper (bool): If True, the Planck taper in the amplitude is disabled.
    """

    f_ref: float
    use_lambda_tildes: bool
    no_taper: bool

    def __init__(
        self,
        f_ref: float = 20.0,
        use_lambda_tildes: bool = False,
        no_taper: bool = False,
    ) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool): Whether to parameterise tidal deformability
                via ``lambda_tilde`` / ``delta_lambda_tilde`` rather than
                ``lambda_1`` / ``lambda_2``. Defaults to False.
            no_taper (bool): Whether to disable tapering (useful for relative
                binning runs). Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self.no_taper = no_taper

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
            *(
                ("lambda_tilde", "delta_lambda_tilde")
                if self.use_lambda_tildes
                else ("lambda_1", "lambda_2")
            ),
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXP_NRTidalv3 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params: Source parameters with keys ``M_c``, ``eta``, ``s1_x``,
                ``s1_y``, ``s1_z``, ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``,
                ``phase_c``, ``iota``, plus tidal keys depending on
                ``use_lambda_tildes``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXP_NRTidalv3_hphc(
            frequency,
            theta,
            self.f_ref,
            use_lambda_tildes=self.use_lambda_tildes,
            no_taper=self.no_taper,
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return (
            f"IMRPhenomXP_NRTidalv3(f_ref={self.f_ref}, "
            f"use_lambda_tildes={self.use_lambda_tildes}, no_taper={self.no_taper})"
        )
