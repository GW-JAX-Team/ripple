"""by Robin Chan"""

import jax
import jax.numpy as jnp
from ..constants import MTSUN, PI
from jaxtyping import Array, Float, Complex
from ..conversions import Mc_eta_to_ms, lambda_tildes_to_lambdas
from .IMRPhenom_tidal_utils import get_kappa
from .IMRPhenomD_NRTidalv2 import (
    get_tidal_amplitude,
)  # Same between v2 and v3
from .NRTidalv3_utils import (
    _get_merger_frequency,
    _get_phenomx_spin_coefficients,
    get_tidal_phase,
    get_NRTidalv3_coefficients,
    get_tidalphasePN_coeffs,
    get_tidal_phase_PN,
    general_planck_taper,
    fullTidalPhaseCorrection,
    changePhase_if_min,
)
from . import IMRPhenomX_utils
from .IMRPhenomXAS import Amp, Phase, PhaseDerivative


def IMRPhenomXAS_NRTidalv3_Phase(
    f: Array,
    f_ref: float,
    theta_intrinsic: Array,
    theta_extrinsic: Array,
    no_taper: bool = False,
    chip: float = 0.0,
    a_prec_override=None,
) -> Array:
    """
    Currently a helper function for XP_NRTidalv3. We could use this later for XAS too
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

    phi_XAS = Phase(f, theta_intrinsic[:4], bbh_phase_coeffs, chip, a_prec_override)
    # dphiXAS = jax.grad(Phase, argnums=0)(f_final, theta_intrinsic[:4], bbh_phase_coeffs) / M_s
    dphiXAS = (
        PhaseDerivative(
            f_final, theta_intrinsic[:4], bbh_phase_coeffs, chip, a_prec_override
        )
        / M_s
    )
    linb = dphiT - dphiXAS
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] + 2 * theta_extrinsic[2]
    phase_shift = (
        linb * (f_Ms - f_ref * M_s)
        - Phase(f_ref, theta_intrinsic[:4], bbh_phase_coeffs, chip, a_prec_override)
        + phiTfRef
        + PI / 4.0
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

    return phi_XAS + phase_shift - (psi_T + psi_SS)


def IMRPhenomXAS_NRTidalv3_Amp(
    f: Array,
    theta_intrinsic: Array,
    theta_extrinsic: Array,
    no_taper: bool = False,
    chip: float = 0.0,
    a_prec_override=None,
) -> Array:
    """
    TODO
    """

    m1, m2, _, _, lambda1, lambda2 = theta_intrinsic
    M_s = (m1 + m2) * MTSUN
    x = PI * f * M_s
    x_23 = x ** (2.0 / 3.0)
    f_Ms = f * M_s

    # Compute kappa
    kappa = get_kappa(theta=theta_intrinsic)

    # Compute amplitudes
    A_T = get_tidal_amplitude(x_23, theta_intrinsic, kappa, distance=theta_extrinsic[0])
    f_merger = _get_merger_frequency(theta_intrinsic)

    XAS_amplitude = Amp(
        f,
        theta_intrinsic[:4],
        IMRPhenomX_utils.PhenomX_amp_coeff_table,
        theta_extrinsic[0],
        chip=chip,
        a_prec_override=a_prec_override,
    )

    if no_taper:
        A_P = jnp.ones_like(f)
    else:
        A_P = 1 - general_planck_taper(f_Ms, f_merger * M_s, 1.2 * f_merger * M_s)

    return A_P * (XAS_amplitude + A_T)


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

    m1, m2, _, _, lambda1, lambda2 = theta_intrinsic
    M_s = (m1 + m2) * MTSUN
    Xa = m1 / (m1 + m2)
    Xb = m2 / (m1 + m2)
    x = PI * f * M_s
    x_23 = x ** (2.0 / 3.0)
    f_Ms = f * M_s

    # Compute kappa
    kappa = get_kappa(theta=theta_intrinsic)

    # Compute amplitudes
    A_T = get_tidal_amplitude(x_23, theta_intrinsic, kappa, distance=theta_extrinsic[0])
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
        A_P = jnp.ones_like(f)
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
        A_P = 1 - general_planck_taper(f_Ms, f_merger * M_s, 1.2 * f_merger * M_s)

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

    # Reconstruct waveform with NRTidal terms included: h(f) = [A(f) + A_tidal(f)] * Exp{I [phi(f) - phi_tidal(f)]} * window(f)
    h0 = (
        A_P
        * (bbh_amp + A_T)
        * jnp.exp(1.0j * ((bbh_psi + phase_shift) - (psi_T + psi_SS)))
    )

    return h0


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
    phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table
    amp_coeffs = IMRPhenomX_utils.PhenomX_amp_coeff_table

    # Generate the BBH part:
    bbh_theta_intrinsic = jnp.array([m1, m2, chi1, chi2])
    Psi = Phase(f, bbh_theta_intrinsic, phase_coeffs)
    A = Amp(f, bbh_theta_intrinsic, amp_coeffs, D=theta_extrinsic[0])

    bbh_amp = A
    bbh_psi = Psi

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

    # -1 prefactor in hp (and the corresponding sign in hc) comes from
    # Ylmfactor = e^(i*PI) in Y_{-2}^{22}, which LAL evaluates at phi=PI/2
    # after generating the h22 mode:
    #   hp = pfac  * Ylmfactor * h22 = -(1+cos^2 iota)/2 * h22
    #   hc = -i    * cfac * Ylmfactor * h22 = i * cos(iota) * h22
    hp = -h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = 1j * h0 * jnp.cos(iota)

    return hp, hc
