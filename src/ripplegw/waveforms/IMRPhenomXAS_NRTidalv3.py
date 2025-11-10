"""by Robin Chan"""

import jax
import jax.numpy as jnp
from ..constants import gt, PI
from ..typing import Array
from ripplegw import Mc_eta_to_ms, lambda_tildes_to_lambdas
from .IMRPhenom_tidal_utils import get_kappa
from .IMRPhenomD_NRTidalv2 import get_spin_phase_correction, get_planck_taper, get_tidal_amplitude # Same between v2 and v3
from .NRTidalv3_utils import _get_merger_frequency, get_tidal_phase, get_NRTidalv3_coefficients, get_tidalphasePN_coeffs, get_tidal_phase_PN, general_planck_taper, fullTidalPhaseCorrection, changePhase_if_min
from ripplegw.waveforms import IMRPhenomX_utils
from .IMRPhenomXAS import Amp, Phase


def _gen_IMRPhenomXAS_NRTidalv3(
    f: Array,
    f_ref: float,
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
    f_Ms = f * M_s

    # Compute kappa
    kappa = get_kappa(theta=theta_intrinsic)

    # Compute amplitudes
    A_T = get_tidal_amplitude(x_23, theta_intrinsic, kappa, distance=theta_extrinsic[0])
    f_merger = _get_merger_frequency(theta_intrinsic)

    # Tidal phase offset #
    f_final = f[-1]
    f_final = jax.lax.select(
        f_merger < f_final,
        f_merger,
        f_final
    )

    if no_taper:
        P_P = jnp.ones_like(f)
        P_P_fref = 1.0
        P_P_ffinal = 1.0
        A_P = jnp.ones_like(f)
    else:
        P_P = general_planck_taper(f, 1.15*f_merger, 1.35*f_merger)
        P_P_fref = general_planck_taper(f_ref, 1.15*f_merger, 1.35*f_merger)
        P_P_ffinal = general_planck_taper(f_final, 1.15*f_merger, 1.35*f_merger)
        A_P = get_planck_taper(f, f_merger)

    bbh_phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table
    # Note: the π shift from Y22 has been moved to the calculation of h0
    phifRef = (
        Phase(f_ref, theta_intrinsic[:4], bbh_phase_coeffs)
        - PI / 4.0 
    ) # This is part of the BBH phase alignment
    phiTfRef = fullTidalPhaseCorrection(f_ref, theta_intrinsic, P_P_fref)  # This is part of the tidal correction to the phase alignment

    dphiXAS = jax.grad(Phase)(f_final, theta_intrinsic[:4], bbh_phase_coeffs)
    dphiT = jax.grad(fullTidalPhaseCorrection)(f_final, theta_intrinsic, P_P_ffinal)
    dphi_merger = -(dphiXAS - dphiT) / M_s # linb from LAL

    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] + 2 * theta_extrinsic[2]

    phase_shift = -(phifRef - phiTfRef + dphi_merger*(f_ref*M_s)) + dphi_merger*f_Ms + ext_phase_contrib

    # Get tidal phase and spin corrections for BNS
    PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
    NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)
    NRTidalv3_phase = get_tidal_phase(x, NRTidalv3_coeffs, PN_coeffs)
    
    # TODO: Check for local minimum -> this doesn't seem to work correctly at the moment
    fHzmrgcheck = 0.9 * f_merger
    increasing = jnp.concatenate([jnp.array([False]), NRTidalv3_phase[1:] >= NRTidalv3_phase[:-1]])
    valid = (f >= fHzmrgcheck) & increasing

    # if local minimum found: set NRTidalv3 phase to this value afterwards
    x_lax = (f, NRTidalv3_phase, valid)
    NRTidalv3_phase = jax.lax.cond(
        jnp.any(valid),
        lambda arr: changePhase_if_min(*arr),
        lambda arr: arr[1],
        x_lax
    )

    
    psi_T = NRTidalv3_phase * (1 - P_P) + get_tidal_phase_PN(x, Xa, lambda1, lambda2, PN_coeffs) * P_P
    psi_SS = get_spin_phase_correction(x_23, theta_intrinsic)

    # Reconstruct waveform with NRTidal terms included: h(f) = [A(f) + A_tidal(f)] * Exp{I [phi(f) - phi_tidal(f)]} * window(f)
    h0 = A_P * (bbh_amp + A_T) * jnp.exp(1.0j * ((bbh_psi + phase_shift + PI) - (psi_T + psi_SS))) # The additional π shift comes from Y22

    return h0


def gen_IMRPhenomXAS_NRTidalv3(
    f: Array,
    params: Array,
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> Array:
    """
    Generate NRTidalv3 frequency domain waveform following 2311.07456.

    params array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]:
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
    Psi = Phase(f, bbh_theta_intrinsic, phase_coeffs)
    A = Amp(f, bbh_theta_intrinsic, amp_coeffs, D=theta_extrinsic[0])

    bbh_amp = A
    bbh_psi = Psi

    # Use BBH waveform and add tidal corrections
    return _gen_IMRPhenomXAS_NRTidalv3(
        f, f_ref, theta_intrinsic, theta_extrinsic, bbh_amp, bbh_psi, no_taper=no_taper
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

    IMRPhenom denotes the name of the underlying BBH approximant used, before applying tidal corrections.

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
