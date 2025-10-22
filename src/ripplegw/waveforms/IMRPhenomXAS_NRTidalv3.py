"""by Robin Chan"""

import jax
import jax.numpy as jnp
from ..constants import gt, m_per_Mpc, PI, TWO_PI, MRSUN, C
from ..typing import Array
from ripplegw import Mc_eta_to_ms, lambda_tildes_to_lambdas
from .IMRPhenom_tidal_utils import get_quadparam_octparam, get_kappa
from .IMRPhenomD_NRTidalv2 import get_tidal_amplitude, get_spin_phase_correction, get_planck_taper # Same between v2 and v3
from .NRTidalv3_utils import _get_merger_frequency, get_tidal_phase, get_NRTidalv3_coefficients, get_tidalphasePN_coeffs, get_tidal_phase_PN, general_planck_taper
from ripplegw.waveforms import IMRPhenomX_utils
from .IMRPhenomXAS import Amp, Phase


# This could be a general utils function
def _gen_IMRPhenomXAS_NRTidalv3(
    f: Array,
    theta_intrinsic: Array,
    theta_extrinsic: Array,
    bbh_amp: Array,
    bbh_psi: Array,
    no_taper: bool = False,
):
    """
    Master internal function to get the GW strain for given parameters. The function takes
    a BBH strain, computed from an underlying BBH approximant, e.g. IMRPhenomD, and applies the
    tidal corrections to it afterwards

    Args:
        f (Array): Frequencies in Hz
        theta_intrinsic (Array): Internal parameters of the system: m1, m2, chi1, chi2, lambda1, lambda2
        theta_extrinsic (Array): Extrinsic parameters of the system: d_L, tc and phi_c
        h0_bbh (Array): The BBH strain of the underlying model (i.e. before applying tidal corrections).

    Returns:
        Array: Final complex-valued strain of GW.
    """

    m1, m2, _, _, lambda1, lambda2 = theta_intrinsic
    M_s = (m1 + m2) * gt
    Xa = m1 / (m1 + m2)
    x = PI * f * M_s
    x_23 = x**(2.0/3.0)

    # Compute kappa
    kappa = get_kappa(theta=theta_intrinsic)

    # Compute amplitudes
    A_T = get_tidal_amplitude(x_23, theta_intrinsic, kappa, distance=theta_extrinsic[0])
    f_merger = _get_merger_frequency(theta_intrinsic)

    # Decide whether to include the Planck taper or not
    if no_taper:
        A_P = jnp.ones_like(f)
        AA_P = jnp.ones_like(f)
    else:
        A_P = general_planck_taper(f, 1.15*f_merger, 1.35*f_merger)
        AA_P = get_planck_taper(f, f_merger)

    # Get tidal phase and spin corrections for BNS
    PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
    NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)
    NRTidalv3_phase = get_tidal_phase(x, NRTidalv3_coeffs, PN_coeffs)

    # Check for local minimum TODO
    fHzmrgcheck = 0.9 * f_merger
    increasing = jnp.concatenate([jnp.array([False]), NRTidalv3_phase[1:] >= NRTidalv3_phase[:-1]])
    valid = (f >= fHzmrgcheck) & increasing


    psi_T = NRTidalv3_phase * (1 - A_P) + get_tidal_phase_PN(x, Xa, lambda1, lambda2, PN_coeffs) * A_P
    psi_SS = get_spin_phase_correction(x_23, theta_intrinsic)

    # Reconstruct waveform with NRTidal terms included: h(f) = [A(f) + A_tidal(f)] * Exp{I [phi(f) - phi_tidal(f)]} * window(f)
    h0 = AA_P * (bbh_amp + A_T) * jnp.exp(1.0j * -(bbh_psi + psi_T + psi_SS))

    return h0


def gen_IMRPhenomXAS_NRTidalv3(
    f: Array,
    params: Array,
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> Array:
    """
    Generate NRTidalv3 frequency domain waveform following NRTidalv3 paper.
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
    --------
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
    phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table
    amp_coeffs = IMRPhenomX_utils.PhenomX_amp_coeff_table

    # Generate the BBH part:
    bbh_theta_intrinsic = jnp.array([m1, m2, chi1, chi2])
    m1_s = m1 * gt
    m2_s = m2 * gt

    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)

    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)
    chia = chi1 - chi2

    fM_s = f * M_s
    fMs_RD, fMs_damp, _, _ = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    Psi = Phase(f, bbh_theta_intrinsic, phase_coeffs)

    # Generate the linear in f and constant contribution to the phase in order
    # to roll the waveform such that the peak is at the input tc and phic
    lina, linb, psi4tostrain = IMRPhenomX_utils.calc_phaseatpeak(
        eta, StotR, chia, delta
    )
    dphi22Ref = (
        jax.grad(Phase)((fMs_RD - fMs_damp) / M_s, bbh_theta_intrinsic, phase_coeffs) / M_s
    )
    linb = linb - dphi22Ref - 2.0 * PI * (500.0 + psi4tostrain)
    phifRef = (
        -(Phase(f_ref, bbh_theta_intrinsic, phase_coeffs) + linb * (f_ref * M_s) + lina)
        + PI / 4.0
        + PI
    )
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] - theta_extrinsic[2]
    Psi = Psi + (linb * fM_s) + lina + phifRef - 2 * PI + ext_phase_contrib

    A = Amp(f, bbh_theta_intrinsic, amp_coeffs, D=theta_extrinsic[0])

    bbh_amp = A
    bbh_psi = Psi

    # Use BBH waveform and add tidal corrections
    return _gen_IMRPhenomXAS_NRTidalv3(
        f, theta_intrinsic, theta_extrinsic, bbh_amp, bbh_psi, no_taper=no_taper
    )


def gen_IMRPhenomXAS_NRTidalv3_hphc(
    f: Array,
    params: Array,
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
):
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
    --------
        hp (array): Strain of the plus polarization
        hc (array): Strain of the cross polarization
    """
    iota = params[-1]
    h0 = gen_IMRPhenomXAS_NRTidalv3(
        f, params[:-1], f_ref, use_lambda_tildes=use_lambda_tildes, no_taper=no_taper
    )

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc
