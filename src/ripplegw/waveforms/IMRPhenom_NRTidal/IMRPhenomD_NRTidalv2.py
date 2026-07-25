"""
This file implements the NRTidalv2 corrections that can be applied to any BBH baseline, see http://arxiv.org/abs/1905.06011 for equations used.
"""

import jax
from ripplegw.interfaces import AmplitudePhaseWaveform, DistanceScaledWaveform
from ripplegw.registry import register
import jax.numpy as jnp
from ripplegw.constants import MTSUN, MPC, PI, TWO_PI, MRSUN
from jaxtyping import Array, Float, Complex
from ripplegw.typing import FloatLike
from typing import Any, Mapping, Optional
from ripplegw.conversions import Mc_eta_to_ms, lambda_tildes_to_lambdas
from ripplegw.utils.tidal import get_quadparam_octparam, get_kappa
from ripplegw.waveforms.IMRPhenomD.IMRPhenomD import (
    Amp,
    get_IIa_raw_phase,
    get_IIb_raw_phase,
    get_inspiral_phase,
)
from ripplegw.waveforms.IMRPhenomD.IMRPhenomD_utils import (
    get_coeffs,
    get_transition_frequencies,
)
from ripplegw.waveforms.IMRPhenomD.IMRPhenomD_QNMdata import fM_CUT
from ripplegw.waveforms.TaylorF2.TaylorF2 import (
    get_4PNQM2SCoeff,
    get_4PNQM2SOCoeff,
    get_6PNQM2SCoeff,
)

#################
### AMPLITUDE ###
#################


# The code below to compute the Planck taper is obtained from gwfast (https://github.com/CosmoStatGW/gwfast/blob/ccde00e644682639aa8c9cbae323e42718fd61ca/gwfast/waveforms.py#L1332)
@jax.custom_jvp
def get_planck_taper(x: Float[Array, " n_freq"], y: float) -> Float[Array, " n_freq"]:
    """
    Compute the Planck taper function.

    Args:
        x (Array): Array of frequencies
        y (float): Point at which the Planck taper starts. The taper ends at 1.2 times y.

    Returns:
        Array: Planck taper function.
    """
    a = 1.2
    yp = a * y
    return jnp.where(
        x < y,
        1.0,
        jnp.where(
            x > yp,
            0.0,
            1.0 - 1.0 / (jnp.exp((yp - y) / (x - y) + (yp - y) / (x - yp)) + 1.0),
        ),
    )


def get_planck_taper_der(
    x: Float[Array, " n_freq"], y: float
) -> Float[Array, " n_freq"]:
    """
    Derivative of the Planck taper function.

    Args:
        x (Array): Array of frequencies
        y (float): Starting point of the Planck taper.

    Returns:
        Array: Array of derivative of Planck taper.
    """
    a = 1.2
    yp = a * y
    tangent_out = jnp.where(
        x < y,
        0.0,
        jnp.where(
            x > yp,
            0.0,
            jnp.exp((yp - y) / (x - y) + (yp - y) / (x - yp))
            * (
                (-1.0 + a) / (x - y)
                + (-1.0 + a) / (x - yp)
                + (-y + yp) / ((x - y) ** 2)
                + 1.2 * (-y + yp) / ((x - yp) ** 2)
            )
            / ((jnp.exp((yp - y) / (x - y) + (yp - y) / (x - yp)) + 1.0) ** 2),
        ),
    )
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out


get_planck_taper.defjvps(
    None, lambda y_dot, primal_out, x, y: get_planck_taper_der(x, y) * y_dot
)


def get_amp0_lal(M: FloatLike, distance: FloatLike) -> FloatLike:
    """
    Get the amp0 prefactor as defined in LAL in LALSimIMRPhenomD, line 331.

    Args:
        M (FloatLike): Total mass in solar masses
        distance (FloatLike): Distance to the source in meters.

    Returns:
        FloatLike: amp0 from LAL.
    """
    amp0 = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M * MRSUN * M * MTSUN / distance
    return amp0


def get_tidal_amplitude(
    x: Float[Array, " n_freq"],
    theta: Float[Array, "6"],
    kappa: FloatLike,
    distance: FloatLike = 1,
) -> Float[Array, " n_freq"]:
    """
    Get the tidal amplitude corrections as given in equation (24) of the NRTidal paper.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)
        kappa (FloatLike): Tidal parameter kappa
        distance (FloatLike, optional): Distance to the source in Mpc.

    Returns:
        Array: Tidal amplitude corrections A_T from NRTidalv2 paper.
    """

    # Mass variables
    m1, m2, _, _, _, _ = theta
    M = m1 + m2

    # Convert distance to meters
    distance *= MPC

    # Pade approximant
    n1 = 4.157407407407407
    n289 = 2519.111111111111
    d = 13477.8073677
    num = 1.0 + n1 * x + n289 * x**2.89
    den = 1.0 + d * x**4.0
    poly = num / den

    # Prefactors are taken from lal source code
    prefac = -9.0 * kappa
    ampT = prefac * x ** (13.0 / 4.0) * poly
    amp0 = get_amp0_lal(M, distance)
    ampT *= amp0 * 2 * jnp.sqrt(PI / 5)

    return ampT


#############
### PHASE ###
#############


def get_tidal_phase(
    x: Float[Array, " n_freq"], theta: Float[Array, "6"], kappa: FloatLike
) -> Float[Array, " n_freq"]:
    """
    Computes the tidal phase psi_T from equation (17) of the NRTidalv2 paper.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters in the order (mass1, mass2, chi1, chi2, lambda1, lambda2)
        kappa (float): Tidal parameter kappa, precomputed in the main function.

    Returns:
        Array: Tidal phase correction.
    """

    # Compute auxiliary quantities
    m1, m2, _, _, _, _ = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    # eta = m1_s * m2_s / (M_s**2.0)

    X1 = m1_s / M_s
    X2 = m2_s / M_s

    # Compute powers
    x_2 = x ** (2.0)
    x_3 = x ** (3.0)
    x_3over2 = x ** (3.0 / 2.0)
    x_5over2 = x ** (5.0 / 2.0)

    # Initialize the coefficients
    c_Newt = 2.4375
    n_1 = -12.615214237993088
    n_3over2 = 19.0537346970349
    n_2 = -21.166863146081035
    n_5over2 = 90.55082156324926
    n_3 = -60.25357801943598
    d_1 = -15.111207827736678
    d_3over2 = 22.195327350624694
    d_2 = 8.064109635305156

    # Pade approximant
    num = (
        1.0
        + (n_1 * x)
        + (n_3over2 * x_3over2)
        + (n_2 * x_2)
        + (n_5over2 * x_5over2)
        + (n_3 * x_3)
    )
    den = 1.0 + (d_1 * x) + (d_3over2 * x_3over2) + (d_2 * x_2)
    ratio = num / den

    # Assemble everything
    psi_T = -kappa * c_Newt / (X1 * X2) * x_5over2
    psi_T *= ratio

    return psi_T


def get_spin_phase_correction(
    x: Float[Array, " n_freq"], theta: Float[Array, "6"]
) -> Float[Array, " n_freq"]:
    """
    Get the higher order spin corrections (3.5PN only).

    LAL's IMRPhenomD_NRTidalv2 uses XLALSimInspiralGetHOSpinTerms which only
    computes the 3.5PN spin-squared and spin-cubed terms. The 2PN and 3PN terms
    described in the NRTidalv2 paper are NOT included in LAL's implementation.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)

    Returns:
        Array: Higher order spin corrections to the phase (3.5PN only).
    """

    # Compute auxiliary quantities
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    # Compute the auxiliary variables
    X1 = m1_s / M_s
    X1sq = X1 * X1
    chi1_sq = chi1 * chi1

    X2 = m2_s / M_s
    X2sq = X2 * X2
    chi2_sq = chi2 * chi2

    # Compute quadrupole parameters
    quadparam1, octparam1 = get_quadparam_octparam(lambda1)
    quadparam2, octparam2 = get_quadparam_octparam(lambda2)

    # Remove 1 for the BBH baseline, from here on, quadparam is "quadparam hat" as referred to in the NRTidalv2 paper etc
    quadparam1 -= 1
    quadparam2 -= 1
    octparam1 -= 1
    octparam2 -= 1

    # 3.5PN spin-squared and spin-cubed terms (matching LAL's XLALSimInspiralGetHOSpinTerms)
    SS_3p5 = (
        -400.0 * PI * quadparam1 * chi1_sq * X1sq
        - 400.0 * PI * quadparam2 * chi2_sq * X2sq
    )
    SSS_3p5 = (
        10.0
        * ((X1sq + 308.0 / 3.0 * X1) * chi1 + (X2sq - 89.0 / 3.0 * X2) * chi2)
        * quadparam1
        * X1sq
        * chi1_sq
        + 10.0
        * ((X2sq + 308.0 / 3.0 * X2) * chi2 + (X1sq - 89.0 / 3.0 * X1) * chi1)
        * quadparam2
        * X2sq
        * chi2_sq
        - 440.0 * octparam1 * X1 * X1sq * chi1_sq * chi1
        - 440.0 * octparam2 * X2 * X2sq * chi2_sq * chi2
    )

    prefac = 3.0 / (128.0 * eta)
    # Only 3.5PN term, matching LAL's implementation
    psi_SS = prefac * (SS_3p5 + SSS_3p5) * x

    return psi_SS


def get_qm_phase_correction(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "6"],
) -> Float[Array, " n_freq"]:
    """
    Return the residual quadrupole-monopole phase correction hidden inside
    LAL's IMRPhenomD baseline when it is called in NRTidalv2 mode.

    LAL passes dQuadMon{1,2} into TaylorF2AlignedPhasing while constructing the
    IMRPhenomD inspiral phase. This shifts the 2PN and 3PN spin-squared pieces
    of the BBH baseline before the usual phase-reference alignment is applied.
    """

    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    X1 = m1_s / M_s
    X2 = m2_s / M_s
    quadparam1, _ = get_quadparam_octparam(lambda1)
    quadparam2, _ = get_quadparam_octparam(lambda2)
    dquadmon1 = quadparam1 - 1.0
    dquadmon2 = quadparam2 - 1.0

    delta_phi4 = (
        get_4PNQM2SOCoeff(X1) + get_4PNQM2SCoeff(X1)
    ) * dquadmon1 * chi1 * chi1 + (
        get_4PNQM2SOCoeff(X2) + get_4PNQM2SCoeff(X2)
    ) * dquadmon2 * chi2 * chi2
    delta_phi6 = (
        get_6PNQM2SCoeff(X1) * dquadmon1 * chi1 * chi1
        + get_6PNQM2SCoeff(X2) * dquadmon2 * chi2 * chi2
    )

    v = (PI * fM_s) ** (1.0 / 3.0)
    prefac = 3.0 / (128.0 * eta)
    return prefac * (delta_phi4 / v + delta_phi6 * v)


def Phase_with_qm_correction(
    f: Float[Array, " n_freq"],
    theta_bbh: Float[Array, "4"],
    theta_intrinsic: Float[Array, "6"],
    coeffs: Float[Array, "19"],
    transition_freqs: tuple[
        FloatLike, FloatLike, FloatLike, FloatLike, FloatLike, FloatLike
    ],
) -> Float[Array, " n_freq"]:
    """
    Compute the IMRPhenomD BBH phase with the hidden NRTidalv2 quadrupole
    correction included before the region-I/IIa matching.
    """

    m1, m2, _, _ = theta_bbh
    M_s = (m1 + m2) * MTSUN
    f1, f2, _, _, f_RD, f_damp = transition_freqs

    def inspiral_phase(fM_s: Array) -> Array:
        return get_inspiral_phase(fM_s, theta_bbh, coeffs) + get_qm_phase_correction(
            fM_s, theta_intrinsic
        )

    phi_Ins = inspiral_phase(f * M_s)

    phi_Ins_f1, dphi_Ins_f1 = jax.value_and_grad(inspiral_phase)(f1 * M_s)
    phi_IIa_f1, dphi_IIa_f1 = jax.value_and_grad(get_IIa_raw_phase)(
        f1 * M_s, theta_bbh, coeffs
    )

    beta1_correction = dphi_Ins_f1 - dphi_IIa_f1
    beta0 = phi_Ins_f1 - beta1_correction * (f1 * M_s) - phi_IIa_f1

    phi_IIa_func = lambda fM_s: (
        get_IIa_raw_phase(fM_s, theta_bbh, coeffs) + beta1_correction * fM_s
    )
    phi_IIa = phi_IIa_func(f * M_s) + beta0

    phi_IIa_f2, dphi_IIa_f2 = jax.value_and_grad(phi_IIa_func)(f2 * M_s)
    phi_IIb_f2, dphi_IIb_f2 = jax.value_and_grad(get_IIb_raw_phase)(
        f2 * M_s, theta_bbh, coeffs, f_RD, f_damp
    )

    a1_correction = dphi_IIa_f2 - dphi_IIb_f2
    a0 = phi_IIa_f2 + beta0 - a1_correction * (f2 * M_s) - phi_IIb_f2

    phi_IIb = (
        get_IIb_raw_phase(f * M_s, theta_bbh, coeffs, f_RD, f_damp)
        + a0
        + a1_correction * (f * M_s)
    )

    return (
        phi_Ins * jnp.heaviside(f1 - f, 0.5)
        + jnp.heaviside(f - f1, 0.5) * phi_IIa * jnp.heaviside(f2 - f, 0.5)
        + phi_IIb * jnp.heaviside(f - f2, 0.5)
    )


def _get_merger_frequency(
    theta: Float[Array, "6"], kappa: Optional[FloatLike] = None
) -> FloatLike:
    """
    Computes the merger frequency in Hz of the given system. This is defined in equation (11) in https://arxiv.org/abs/1804.02235 and the lal source code.

    Args:
        theta (Array): Intrinsic parameters with order (m1, m2, chi1, chi2, lambda1, lambda2)
        kappa (Optional[FloatLike]): Tidal parameter kappa. Defaults to None, so that it is computed from the given parameters theta.

    Returns:
        FloatLike: The merger frequency in Hz.
    """

    # Compute auxiliary quantities
    m1, m2, _, _, _, _ = theta
    M = m1 + m2
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    q = m1_s / m2_s

    if kappa is None:
        kappa = get_kappa(theta)
    assert kappa is not None
    kappa_2 = kappa * kappa

    # Initialize coefficients
    a_0 = 0.3586
    n_1 = 3.35411203e-2
    n_2 = 4.31460284e-5
    d_1 = 7.54224145e-2
    d_2 = 2.23626859e-4

    # Get ratio and prefactor
    num = 1.0 + n_1 * kappa + n_2 * kappa_2
    den = 1.0 + d_1 * kappa + d_2 * kappa_2
    Q_0 = a_0 * (q) ** (-1.0 / 2.0)

    # Dimensionless angular frequency of merger
    Momega_merger = Q_0 * (num / den)

    # Convert from angular frequency to frequency (divide by 2*pi) and then convert from dimensionless frequency to Hz
    fHz_merger = Momega_merger / (M * MTSUN) / (TWO_PI)

    return fHz_merger


def _amplitude_of(
    f: Float[Array, " n_freq"],
    theta_intrinsic: Float[Array, "6"],
    theta_extrinsic: Float[Array, "3"],
    bbh_amp: Float[Array, " n_freq"],
    no_taper: bool = False,
) -> Float[Array, " n_freq"]:
    """Amplitude of ``h0 = amplitude * exp(1j * phase)``.

    ``bbh_amp`` is the underlying BBH baseline amplitude (see ``_bbh_amp_psi``);
    the tidal amplitude and Planck taper are added/applied on top of it.
    """
    m1, m2, _, _, _, _ = theta_intrinsic
    M_s = (m1 + m2) * MTSUN
    x = (PI * M_s * f) ** (2.0 / 3.0)

    kappa = get_kappa(theta=theta_intrinsic)
    A_T = get_tidal_amplitude(x, theta_intrinsic, kappa, distance=theta_extrinsic[0])

    if no_taper:
        A_P = jnp.ones_like(f)
    else:
        f_merger = _get_merger_frequency(theta_intrinsic, kappa)
        A_P = get_planck_taper(f, f_merger)

    # LAL's IMRPhenomD (when called with NRTidalv2_V) adds tidal amplitude as:
    #   amp0 * (amp + 2*sqrt(PI/5)*ampT) * exp(-i*phi)
    # where ampT is the dimensionless tidal amplitude from XLALSimNRTunedTidesFDTidalAmplitudeFrequencySeries.
    # Our get_tidal_amplitude already includes the amp0 * 2*sqrt(PI/5) factor,
    # so we add A_T directly to bbh_amp.
    # See LALSimIMRPhenomD.c lines 428-452 and LALSimIMRPhenomD_NRTidal.c.
    return A_P * (bbh_amp + A_T)


def _phase_of(
    f: Float[Array, " n_freq"],
    theta_intrinsic: Float[Array, "6"],
    bbh_psi: Float[Array, " n_freq"],
) -> Float[Array, " n_freq"]:
    """Phase of ``h0 = amplitude * exp(1j * phase)`` (the exponent).

    ``bbh_psi`` is the underlying BBH baseline phase (see ``_bbh_amp_psi``);
    tidal and spin-correction phase terms are added on top of it.
    """
    m1, m2, _, _, _, _ = theta_intrinsic
    M_s = (m1 + m2) * MTSUN
    x = (PI * M_s * f) ** (2.0 / 3.0)

    kappa = get_kappa(theta=theta_intrinsic)
    psi_T = get_tidal_phase(x, theta_intrinsic, kappa)
    psi_SS = get_spin_phase_correction(x, theta_intrinsic)

    return -(bbh_psi + psi_T + psi_SS)


def _gen_IMRPhenomD_NRTidalv2(
    f: Float[Array, " n_freq"],
    theta_intrinsic: Float[Array, "6"],
    theta_extrinsic: Float[Array, "3"],
    bbh_amp: Float[Array, " n_freq"],
    bbh_psi: Float[Array, " n_freq"],
    no_taper: bool = False,
) -> Complex[Array, " n_freq"]:
    """
    Master internal function to get the GW strain for given parameters. The function takes
    a BBH strain, computed from an underlying BBH approximant, e.g. IMRPhenomD, and applies the
    tidal corrections to it afterwards, according to equation (25) of the NRTidalv2 paper.

    Args:
        f (Array): Frequencies in Hz
        theta_intrinsic (Array): Internal parameters of the system: m1, m2, chi1, chi2, lambda1, lambda2
        theta_extrinsic (Array): Extrinsic parameters of the system: d_L, tc and phi_c
        bbh_amp (Array): BBH baseline amplitude, before tidal corrections.
        bbh_psi (Array): BBH baseline phase, before tidal corrections.

    Returns:
        Array: Final complex-valued strain of GW.
    """
    A = _amplitude_of(f, theta_intrinsic, theta_extrinsic, bbh_amp, no_taper=no_taper)
    phi = _phase_of(f, theta_intrinsic, bbh_psi)
    h0 = A * jnp.exp(1.0j * phi)
    return h0


def _bbh_amp_psi(
    f: Float[Array, " n_freq"],
    theta_intrinsic: Float[Array, "6"],
    theta_extrinsic: Float[Array, "3"],
    f_ref: float,
) -> tuple[Float[Array, " n_freq"], Float[Array, " n_freq"]]:
    """BBH-baseline amplitude and phase that the tidal corrections are applied on top of."""
    m1, m2, chi1, chi2, _, _ = theta_intrinsic
    bbh_theta_intrinsic = jnp.array([m1, m2, chi1, chi2])
    coeffs = get_coeffs(bbh_theta_intrinsic)
    M_s = (bbh_theta_intrinsic[0] + bbh_theta_intrinsic[1]) * MTSUN

    # Shift phase so that peak amplitude matches t = 0
    transition_freqs = get_transition_frequencies(
        bbh_theta_intrinsic, coeffs[5], coeffs[6]
    )
    _, _, _, f4, f_RD, f_damp = transition_freqs
    t0 = jax.grad(get_IIb_raw_phase)(
        f4 * M_s, bbh_theta_intrinsic, coeffs, f_RD, f_damp
    )

    Psi = Phase_with_qm_correction(
        f, bbh_theta_intrinsic, theta_intrinsic, coeffs, transition_freqs
    )
    Psi_ref = Phase_with_qm_correction(
        jnp.array([f_ref]),
        bbh_theta_intrinsic,
        theta_intrinsic,
        coeffs,
        transition_freqs,
    )[0]
    Mf_ref = f_ref * M_s
    Psi -= t0 * ((f * M_s) - Mf_ref) + Psi_ref
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] - 2 * theta_extrinsic[2]
    Psi += ext_phase_contrib
    fcut_true = jnp.floor(fM_CUT / M_s / (f[1] - f[0])) * (f[1] - f[0])
    Psi = Psi * jnp.heaviside(fcut_true - f, 0.0) + 2.0 * PI * jnp.heaviside(
        f - fcut_true, 1.0
    )

    A = Amp(f, bbh_theta_intrinsic, coeffs, transition_freqs, D=theta_extrinsic[0])
    return A, Psi


def gen_IMRPhenomD_NRTidalv2(
    f: Float[Array, " n_freq"],
    params: Float[Array, "9"],
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> Complex[Array, " n_freq"]:
    """
    Generate NRTidalv2 frequency domain waveform following NRTidalv2 paper.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    lambda1: Dimensionless tidal deformability of primary object
    lambda2: Dimensionless tidal deformability of secondary object
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence

    f_ref: Reference frequency for the waveform

    Returns:
        h0 (array): Strain
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

    bbh_amp, bbh_psi = _bbh_amp_psi(f, theta_intrinsic, theta_extrinsic, f_ref)

    # Use BBH waveform and add tidal corrections
    return _gen_IMRPhenomD_NRTidalv2(
        f, theta_intrinsic, theta_extrinsic, bbh_amp, bbh_psi, no_taper=no_taper
    )


def gen_IMRPhenomD_NRTidalv2_hphc(
    f: Float[Array, " n_freq"],
    params: Float[Array, "10"],
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    vars array contains both intrinsic and extrinsic variables

    IMRphenom denotes the name of the underlying BBH approximant used, before applying tidal corrections.

    theta = [Mchirp, eta, chi1, chi2, lambda1, lambda2, D, tc, phic, inclination]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence
    inclination: Inclination angle of the binary [between 0 and PI]

    f_ref: Reference frequency for the waveform

    Returns:
        hp (array): Strain of the plus polarization
        hc (array): Strain of the cross polarization
    """
    iota = params[-1]
    h0 = gen_IMRPhenomD_NRTidalv2(
        f, params[:-1], f_ref, use_lambda_tildes=use_lambda_tildes, no_taper=no_taper
    )

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc


def _split_params(
    params: Mapping[str, Any], use_lambda_tildes: bool
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


@register("IMRPhenomD_NRTidalv2", is_tidal=True, is_precessing=False)
class IMRPhenomD_NRTidalv2(AmplitudePhaseWaveform, DistanceScaledWaveform):
    """IMRPhenomD_NRTidalv2 frequency-domain waveform (non-precessing, NRTidalv2 tides).

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
                via ``lambda_tilde`` / ``delta_lambda_tilde`` (Eq. 5-6 of
                arXiv:1402.5156) instead of ``lambda_1`` / ``lambda_2``.
                Defaults to False.
            no_taper (bool): Whether to remove the Planck taper in the amplitude
                (useful for relative binning runs). Defaults to False.
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
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, Any]
    ) -> Float[Array, " n_freq"]:
        """Amplitude of ``h0``. Requires a grid with at least 2 uniformly spaced points."""
        theta_intrinsic, theta_extrinsic = _split_params(params, self.use_lambda_tildes)
        bbh_amp, _ = _bbh_amp_psi(
            frequency, theta_intrinsic, theta_extrinsic, self.f_ref
        )
        return _amplitude_of(
            frequency, theta_intrinsic, theta_extrinsic, bbh_amp, no_taper=self.no_taper
        )

    def phase(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, Any]
    ) -> Float[Array, " n_freq"]:
        """Phase of ``h0``. Forced to ``2*pi`` above the BBH baseline's high-frequency cutoff."""
        theta_intrinsic, theta_extrinsic = _split_params(params, self.use_lambda_tildes)
        _, bbh_psi = _bbh_amp_psi(
            frequency, theta_intrinsic, theta_extrinsic, self.f_ref
        )
        return _phase_of(frequency, theta_intrinsic, bbh_psi)

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, Any]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomD_NRTidalv2 waveform.

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
        hp, hc = gen_IMRPhenomD_NRTidalv2_hphc(
            frequency,
            theta,
            self.f_ref,
            use_lambda_tildes=self.use_lambda_tildes,
            no_taper=self.no_taper,
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return (
            f"IMRPhenomD_NRTidalv2(f_ref={self.f_ref}, "
            f"use_lambda_tildes={self.use_lambda_tildes}, no_taper={self.no_taper})"
        )
