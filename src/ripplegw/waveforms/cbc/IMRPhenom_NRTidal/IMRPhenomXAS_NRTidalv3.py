"""by Robin Chan"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import Mc_eta_to_ms, lambda_tildes_to_lambdas
from ripplegw.interfaces import AmplitudePhaseWaveform, DistanceScaledWaveform
from ripplegw.registry import register
from ripplegw.typing import FloatLike
from ripplegw.utils.tidal import get_kappa
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    get_tidal_amplitude,
)  # Same between v2 and v3
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.NRTidalv3_utils import (
    _get_merger_frequency,
    _get_phenomx_spin_coefficients,
    changePhase_if_min,
    fullTidalPhaseCorrection,
    general_planck_taper,
    get_NRTidalv3_coefficients,
    get_tidal_phase,
    get_tidal_phase_PN,
    get_tidalphasePN_coeffs,
)
from ripplegw.waveforms.cbc.IMRPhenomX import IMRPhenomX_utils
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXAS import Amp, Phase, PhaseDerivative


def _amplitude_of(
    f: Float[Array, " n_freq"],
    theta_intrinsic: Float[Array, "6"],
    theta_extrinsic: Float[Array, "3"],
    bbh_amp: Float[Array, " n_freq"],
    no_taper: bool = False,
) -> Float[Array, " n_freq"]:
    """Amplitude of ``h0 = amplitude * exp(1j * phase)``.

    ``bbh_amp`` is the underlying BBH baseline amplitude (see ``_bbh_amp_psi``);
    the tidal amplitude and its own taper are added/applied on top of it.
    """
    m1, m2, _, _, _, _ = theta_intrinsic
    M_s = (m1 + m2) * MTSUN
    x = PI * f * M_s
    x_23 = x ** (2.0 / 3.0)
    f_Ms = f * M_s

    kappa = get_kappa(theta=theta_intrinsic)
    A_T = get_tidal_amplitude(x_23, theta_intrinsic, kappa, distance=theta_extrinsic[0])
    f_merger = _get_merger_frequency(theta_intrinsic)

    if no_taper:
        A_P = jnp.ones_like(f)
    else:
        A_P = 1 - general_planck_taper(f_Ms, f_merger * M_s, 1.2 * f_merger * M_s)

    return A_P * (bbh_amp + A_T)


def _phase_of(
    f: Float[Array, " n_freq"],
    f_ref: float,
    theta_intrinsic: Float[Array, "6"],
    theta_extrinsic: Float[Array, "3"],
    bbh_psi: Float[Array, " n_freq"],
    no_taper: bool = False,
) -> Float[Array, " n_freq"]:
    """Phase of ``h0 = amplitude * exp(1j * phase)`` (the exponent).

    ``bbh_psi`` is the underlying BBH baseline phase (see ``_bbh_amp_psi``);
    tidal and spin-correction phase terms are added on top of it. Note this
    phase's taper (``P_P``, windows 1.15-1.35x the merger frequency) differs
    from the amplitude's (``A_P`` in ``_amplitude_of``, windows 1.0-1.2x).
    """
    m1, m2, _, _, lambda1, lambda2 = theta_intrinsic
    M_s = (m1 + m2) * MTSUN
    Xa = m1 / (m1 + m2)
    Xb = m2 / (m1 + m2)
    x = PI * f * M_s
    f_Ms = f * M_s

    f_merger = _get_merger_frequency(theta_intrinsic)

    # Tidal phase offset #
    df = jax.lax.cond(
        f.shape[0] > 1,
        lambda _: f[1] - f[0],
        lambda _: jnp.asarray(0.0, dtype=f.dtype),
        operand=None,
    )
    f_final = f[-1] + df
    f_final = jax.lax.select(f_merger < f_final, f_merger, f_final)

    if no_taper:
        P_P = jnp.ones_like(f)
        P_P_fref = jnp.asarray(1.0)
        dphiT = jax.grad(fullTidalPhaseCorrection)(
            f_final * M_s, theta_intrinsic, jnp.asarray(1.0)
        )
    else:
        P_P = general_planck_taper(f_Ms, 1.15 * f_merger * M_s, 1.35 * f_merger * M_s)
        P_P_fref = general_planck_taper(
            f_ref * M_s, 1.15 * f_merger * M_s, 1.35 * f_merger * M_s
        )
        dphiT = jax.grad(
            lambda fMs: fullTidalPhaseCorrection(
                fMs,
                theta_intrinsic,
                general_planck_taper(fMs, 1.15 * f_merger * M_s, 1.35 * f_merger * M_s),
            )
        )(f_final * M_s)

    bbh_phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table

    phiTfRef = fullTidalPhaseCorrection(  # This is part of the tidal correction to the phase alignment and takes into account spin-tidal interactions
        f_ref * M_s, theta_intrinsic, P_P_fref
    )

    # dphiXAS = jax.grad(Phase, argnums=0)(f_final, theta_intrinsic[:4], bbh_phase_coeffs) / M_s
    dphiXAS = PhaseDerivative(f_final, theta_intrinsic[:4], bbh_phase_coeffs) / M_s
    linb = dphiT - dphiXAS
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] + 2 * theta_extrinsic[2]
    phase_shift = (
        linb * (f_Ms - f_ref * M_s)
        - Phase(f_ref, theta_intrinsic[:4], bbh_phase_coeffs)
        + phiTfRef
        + PI / 4.0
        - PI
        + ext_phase_contrib
    )

    # Get tidal phase and spin corrections for BNS
    PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
    NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)
    NRTidalv3_phase = get_tidal_phase(x, NRTidalv3_coeffs, PN_coeffs)

    # Check for local minimum post-merger (Sec. IV G of arXiv:2311.07456).
    fHzmrgcheck = 0.9 * f_merger
    increasing = jnp.concatenate(
        [jnp.array([False]), NRTidalv3_phase[1:] >= NRTidalv3_phase[:-1]]
    )
    valid = (f >= fHzmrgcheck) & increasing
    x_lax = (f, NRTidalv3_phase, valid)
    NRTidalv3_phase = jax.lax.cond(
        jnp.any(valid), lambda arr: changePhase_if_min(*arr), lambda arr: arr[1], x_lax
    )

    psi_T = (
        NRTidalv3_phase * (1 - P_P)
        + get_tidal_phase_PN(x, Xa, lambda1, lambda2, PN_coeffs) * P_P
    )

    c2pn, c3pn, c3p5pn = _get_phenomx_spin_coefficients(theta_intrinsic)

    pfaN = 3.0 / (128.0 * Xa * Xb)
    psi_SS = (
        pfaN * c2pn / ((PI * f_Ms) ** (1.0 / 3.0))
        + pfaN * c3pn * ((PI * f_Ms) ** (1.0 / 3.0))
        + pfaN * c3p5pn * ((PI * f_Ms) ** (2.0 / 3.0))
    )

    return (bbh_psi + phase_shift) - (psi_T + psi_SS)


def _gen_IMRPhenomXAS_NRTidalv3(
    f: Float[Array, " n_freq"],
    f_ref: float,
    theta_intrinsic: Float[Array, "6"],
    theta_extrinsic: Float[Array, "3"],
    bbh_amp: Float[Array, " n_freq"],
    bbh_psi: Float[Array, " n_freq"],
    no_taper: bool = False,
) -> Complex[Array, " n_freq"]:
    """
    Master internal function to get the GW strain for given parameters.

    The function takes a BBH strain, computed from an underlying BBH approximant,
    e.g. IMRPhenomD, and applies the tidal corrections to it afterwards.

    Args:
        f (Array): Frequencies in Hz.
        f_ref (float): Reference frequency for the waveform.
        theta_intrinsic (Array): Intrinsic parameters of the system: [m1, m2, chi1, chi2, lambda1, lambda2].
        theta_extrinsic (Array): Extrinsic parameters of the system: [d_L, tc, phi_c].
        bbh_amp (Array): The BBH amplitude of the underlying model (before applying tidal corrections).
        bbh_psi (Array): The BBH phase of the underlying model (before applying tidal corrections).
        no_taper (bool, optional): Whether to disable tapering. Default is False.

    Returns:
        Array: Final complex-valued strain of GW.
    """
    # Reconstruct waveform with NRTidal terms included: h(f) = [A(f) + A_tidal(f)] * Exp{I [phi(f) - phi_tidal(f)]} * window(f)
    A = _amplitude_of(f, theta_intrinsic, theta_extrinsic, bbh_amp, no_taper=no_taper)
    phi = _phase_of(
        f, f_ref, theta_intrinsic, theta_extrinsic, bbh_psi, no_taper=no_taper
    )
    h0 = A * jnp.exp(1.0j * phi)
    return h0


def _bbh_amp_psi(
    f: Float[Array, " n_freq"],
    theta_intrinsic: Float[Array, "6"],
    theta_extrinsic: Float[Array, "3"],
) -> tuple[Float[Array, " n_freq"], Float[Array, " n_freq"]]:
    """BBH-baseline amplitude and phase that the tidal corrections are applied on top of."""
    m1, m2, chi1, chi2, _, _ = theta_intrinsic
    bbh_theta_intrinsic = jnp.array([m1, m2, chi1, chi2])
    phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table
    amp_coeffs = IMRPhenomX_utils.PhenomX_amp_coeff_table
    Psi = Phase(f, bbh_theta_intrinsic, phase_coeffs)
    A = Amp(f, bbh_theta_intrinsic, amp_coeffs, D=theta_extrinsic[0])
    return A, Psi


def gen_IMRPhenomXAS_NRTidalv3(
    f: Float[Array, " n_freq"],
    params: Float[Array, "9"],
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> Complex[Array, " n_freq"]:
    """
    Generate NRTidalv3 frequency domain waveform following 2311.07456.

    Args:
        f (Array): Frequencies in Hz.
        params (Array): Array containing both intrinsic and extrinsic variables:
            theta = [Mchirp, eta, chi1, chi2, lambda1, lambda2, D, tc, phic]:

            - Mchirp: Chirp mass of the system [solar masses]
            - eta: Symmetric mass ratio [between 0.0 and 0.25]
            - chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
            - chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
            - lambda1: Dimensionless tidal deformability of primary object
            - lambda2: Dimensionless tidal deformability of secondary object
            - D: Luminosity distance to source [Mpc]
            - tc: Time of coalescence. This only appears as an overall linear in f
            contribution to the phase
            - phic: Phase of coalescence
        f_ref (float): Reference frequency for the waveform.
        use_lambda_tildes (bool, optional): Use lambda tilde and delta lambda instead of lambda1 and lambda2. Default is True.
        no_taper (bool, optional): Whether to disable tapering. Default is False.

    Returns:
        h0 (Array): Strain.
    """

    # Get component masses
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    if use_lambda_tildes:
        lambda1, lambda2 = lambda_tildes_to_lambdas(
            jnp.array([params[4], params[5], m1, m2])
        )
    else:
        lambda1, lambda2 = params[4], params[5]
    chi1, chi2 = params[2], params[3]

    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    theta_extrinsic = params[6:]

    bbh_amp, bbh_psi = _bbh_amp_psi(f, theta_intrinsic, theta_extrinsic)

    # Use BBH waveform and add tidal corrections
    return _gen_IMRPhenomXAS_NRTidalv3(
        f, f_ref, theta_intrinsic, theta_extrinsic, bbh_amp, bbh_psi, no_taper=no_taper
    )


def gen_IMRPhenomXAS_NRTidalv3_hphc(
    f: Float[Array, " n_freq"],
    params: Float[Array, "10"],
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Generate NRTidalv3 frequency domain waveform with plus and cross polarizations.

    IMRPhenom denotes the name of the underlying BBH approximant used, before
    applying tidal corrections.

    Args:
        f (Array): Frequencies in Hz.
        params (Array): Array containing both intrinsic and extrinsic variables:
            theta = [Mchirp, eta, chi1, chi2, lambda1, lambda2, D, tc, phic, inclination]:

            - Mchirp: Chirp mass of the system [solar masses]
            - eta: Symmetric mass ratio [between 0.0 and 0.25]
            - chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
            - chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
            - lambda1: Dimensionless tidal deformability of primary object
            - lambda2: Dimensionless tidal deformability of secondary object
            - D: Luminosity distance to source [Mpc]
            - tc: Time of coalescence. This only appears as an overall linear in f
            contribution to the phase
            - phic: Phase of coalescence
            - inclination: Inclination angle of the binary [between 0 and PI]
        f_ref (float): Reference frequency for the waveform.
        use_lambda_tildes (bool, optional): Use lambda tilde and delta lambda instead of lambda1 and lambda2. Default is True.
        no_taper (bool, optional): Whether to disable tapering. Default is False.

    Returns:
        hp (Array): Strain of the plus polarization.
        hc (Array): Strain of the cross polarization.
    """
    iota = params[-1]
    h0 = gen_IMRPhenomXAS_NRTidalv3(
        f, params[:-1], f_ref, use_lambda_tildes=use_lambda_tildes, no_taper=no_taper
    )

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc


def _split_params(
    params: Mapping[str, FloatLike], use_lambda_tildes: bool
) -> tuple[Float[Array, "6"], Float[Array, "3"]]:
    """Build ``(theta_intrinsic, theta_extrinsic)`` from a params mapping.

    ``tc`` (time of coalescence) is fixed at 0; it is not yet reachable
    through the class API.
    """
    if use_lambda_tildes:
        first_lambda_param = params["lambda_tilde"]
        second_lambda_param = params["delta_lambda_tilde"]
    else:
        first_lambda_param = params["lambda_1"]
        second_lambda_param = params["lambda_2"]

    m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
    if use_lambda_tildes:
        lambda1, lambda2 = lambda_tildes_to_lambdas(
            jnp.array([first_lambda_param, second_lambda_param, m1, m2])
        )
    else:
        lambda1, lambda2 = first_lambda_param, second_lambda_param

    theta_intrinsic = jnp.array(
        [m1, m2, params["s1_z"], params["s2_z"], lambda1, lambda2]
    )
    theta_extrinsic = jnp.array([params["d_L"], 0.0, params["phase_c"]])
    return theta_intrinsic, theta_extrinsic


@register("IMRPhenomXAS_NRTidalv3", is_tidal=True, is_precessing=False)
class IMRPhenomXAS_NRTidalv3(AmplitudePhaseWaveform, DistanceScaledWaveform):
    """IMRPhenomXAS_NRTidalv3 frequency-domain waveform (non-precessing, NRTidalv3 tides).

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
            "s1_z",
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

    def amplitude(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> Float[Array, " n_freq"]:
        """Amplitude of ``h0``."""
        theta_intrinsic, theta_extrinsic = _split_params(params, self.use_lambda_tildes)
        bbh_amp, _ = _bbh_amp_psi(frequency, theta_intrinsic, theta_extrinsic)
        return _amplitude_of(
            frequency, theta_intrinsic, theta_extrinsic, bbh_amp, no_taper=self.no_taper
        )

    def phase(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> Float[Array, " n_freq"]:
        """Phase of ``h0``."""
        theta_intrinsic, theta_extrinsic = _split_params(params, self.use_lambda_tildes)
        _, bbh_psi = _bbh_amp_psi(frequency, theta_intrinsic, theta_extrinsic)
        return _phase_of(
            frequency,
            self.f_ref,
            theta_intrinsic,
            theta_extrinsic,
            bbh_psi,
            no_taper=self.no_taper,
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXAS_NRTidalv3 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params: Source parameters with keys ``M_c``, ``eta``, ``s1_z``,
                ``s2_z``, ``d_L``, ``phase_c``, ``iota``, plus tidal keys
                depending on ``use_lambda_tildes``.

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
                params["s1_z"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXAS_NRTidalv3_hphc(
            frequency,
            theta,
            self.f_ref,
            use_lambda_tildes=self.use_lambda_tildes,
            no_taper=self.no_taper,
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return (
            f"IMRPhenomXAS_NRTidalv3(f_ref={self.f_ref}, "
            f"use_lambda_tildes={self.use_lambda_tildes}, no_taper={self.no_taper})"
        )
