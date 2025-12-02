"""by Robin Chan"""

import jax
import jax.numpy as jnp
from ..constants import gt, PI
from ..typing import Array
from ripplegw import Mc_eta_to_ms, lambda_tildes_to_lambdas
from .IMRPhenom_tidal_utils import get_kappa, get_quadparam_octparam
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
        # P_P_ffinal = 1.0
        dphiT = jax.grad(fullTidalPhaseCorrection)(f_final*M_s, theta_intrinsic, 1.0)
        A_P = jnp.ones_like(f)
    else:
        P_P = general_planck_taper(f, 1.15*f_merger, 1.35*f_merger)
        P_P_fref = general_planck_taper(f_ref*M_s, 1.15*f_merger*M_s, 1.35*f_merger*M_s)
        # P_P_ffinal = general_planck_taper(f_final, 1.15*f_merger, 1.35*f_merger)
        dphiT = jax.grad(lambda fMs: fullTidalPhaseCorrection(
            fMs, theta_intrinsic, general_planck_taper(fMs, 1.15*f_merger*M_s, 1.35*f_merger*M_s)
        ))(f_final*M_s)
        A_P = get_planck_taper(f, f_merger)

    bbh_phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table
    # Note: the π shift from Y22 has been moved to the calculation of h0
    phifRef = (
        Phase(f_ref, theta_intrinsic[:4], bbh_phase_coeffs)
        - PI / 4.0 
    ) # This is part of the BBH phase alignment
    phiTfRef = fullTidalPhaseCorrection(f_ref*M_s, theta_intrinsic, P_P_fref)  # This is part of the tidal correction to the phase alignment

    dphiXAS = jax.grad(Phase)(f_final, theta_intrinsic[:4], bbh_phase_coeffs) / M_s
    # dphiT = IMRPhenomX_TidalPhaseDerivative(f_final, theta_intrinsic) # Analytical derivative for testing purposes
    dphi_merger = -(dphiXAS - dphiT)  # linb from LAL

    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] + 2 * theta_extrinsic[2]

    phase_shift = -(phifRef - phiTfRef + dphi_merger*(f_ref*M_s)) + dphi_merger*f_Ms + ext_phase_contrib

    # Get tidal phase and spin corrections for BNS
    PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
    NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)
    NRTidalv3_phase = get_tidal_phase(x, NRTidalv3_coeffs, PN_coeffs)
    
    # Check for local minimum
    fHzmrgcheck = 0.9 * f_merger
    increasing = jnp.concatenate([jnp.array([False]), NRTidalv3_phase[1:] >= NRTidalv3_phase[:-1]])
    valid = (f >= fHzmrgcheck) & increasing

    # If local minimum found: set NRTidalv3 phase to this value afterwards
    # See discussion in Sec. IV G of arXiv:2311.07456
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
    theta = [Mchirp, eta, chi1, chi2, lambda1, lambda2, D, tc, phic]:
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


################################### ANALYTICAL DERIVATIVE FOR TESTING ###################################

def IMRPhenomX_TidalPhaseDerivative(f, theta):
    """Calculate tidal phase derivative at a single frequency f."""
    theta_intrinsic = theta
    m1, m2, chi1, chi2, lambda1, lambda2 = theta_intrinsic
    M = m1 + m2
    M_s = (m1 + m2) * gt
    Mf = f * M_s
    x = PI * f * M_s

    X_A = m1/M
    X_B = m2/M

    PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
    NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)

    # Compute quadrupole parameters
    quadparam1, octparam1 = get_quadparam_octparam(lambda1)
    quadparam2, octparam2 = get_quadparam_octparam(lambda2)

    # Remove 1 for the BBH baseline, from here on, quadparam is "quadparam hat" as referred to in the NRTidalv2 paper etc
    quadparam1 -= 1
    quadparam2 -= 1
    octparam1 -= 1
    octparam2 -= 1

    c2pn = -50.0 * quadparam1 * chi1**2 * X_A**2 - 50.0 * quadparam2 * chi2**2 * X_B**2
    c3pn = (
        5.0
        / 84.0
        * (9407.0 + 8218.0 * X_A - 2016.0 * X_A**2)
        * quadparam1
        * X_A**2
        * chi1**2
        + 5.0
        / 84.0
        * (9407.0 + 8218.0 * X_B - 2016.0 * X_B**2)
        * quadparam2
        * X_B**2
        * chi2**2
    )
    c3p5pn = (
        -400.0 * PI * quadparam1 * chi1**2 * X_A**2
        - 400.0 * PI * quadparam2 * chi2**2 * X_B**2
    ) + (
        10.0
        * ((X_A**2 + 308.0 / 3.0 * X_A) * chi1 + (X_B**2 - 89.0 / 3.0 * X_B) * chi2)
        * quadparam1
        * X_A**2
        * chi1**2
        + 10.0
        * ((X_B**2 + 308.0 / 3.0 * X_B) * chi2 + (X_A**2 - 89.0 / 3.0 * X_A) * chi1)
        * quadparam2
        * X_B**2
        * chi2**2
        - 440.0 * octparam1 * X_A * X_A**2 * chi1**2 * chi1
        - 440.0 * octparam2 * X_B * X_B**2 * chi2**2 * chi2
    )

    NRTuned_dphase=0.

    pfaN=3./(128.*X_A*X_B)

    threePN_dphase=(pfaN*(-c2pn + c3pn*x**(2.0/3.0)))/(3.0*Mf**(4.0/3.0) * PI**(1.0/3.0))

    dphase=threePN_dphase

    dphase+=(2.*c3p5pn*pfaN*PI**(2.0/3.0))/(3.0*Mf**(1.0/3.0))

    PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
    NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)

    s1 = NRTidalv3_coeffs[0]
    s2 = NRTidalv3_coeffs[1]
    s3 = NRTidalv3_coeffs[2]

    s2s3 = s2*s3
    s2Mf = -s2*Mf*PI*2.0
    exps2s3 = jnp.cosh(s2s3) + jnp.sinh(s2s3)
    exps2Mf = jnp.cosh(s2Mf) + jnp.sinh(s2Mf)

    # Rewriting Eq. (27) of arXiv:2311.07456 
    dynk2barfunc = 1.0 + ((s1) - 1)*(1.0/(1.0 + exps2Mf*exps2s3)) - ((s1-1.0)/(1.0 + exps2s3)) - 2.0*(Mf*PI)*((s1) - 1)*s2*exps2s3/((1.0 + exps2s3)*(1.0 + exps2s3))

    kappaA = NRTidalv3_coeffs[4]
    kappaB = NRTidalv3_coeffs[5]

    dynkappaA = kappaA*dynk2barfunc
    dynkappaB = kappaB*dynk2barfunc

    # Pade Coefficients, Table II of arXiv:2311.07456 */
    n_5over2A = NRTidalv3_coeffs[6]
    n_3A = NRTidalv3_coeffs[7]
    d_1A = NRTidalv3_coeffs[8]

    n_5over2B = NRTidalv3_coeffs[9]
    n_3B = NRTidalv3_coeffs[10]
    d_1B = NRTidalv3_coeffs[11]

    # 7.5PN Coefficients */
    c_NewtA = PN_coeffs[0]
    c_1A = PN_coeffs[1]
    c_3over2A = PN_coeffs[2]
    c_2A = PN_coeffs[3]
    c_5over2A = PN_coeffs[4]

    c_NewtB = PN_coeffs[5]
    c_1B = PN_coeffs[6]
    c_3over2B = PN_coeffs[7]
    c_2B = PN_coeffs[8]
    c_5over2B = PN_coeffs[9]

    # Pade Coefficients constrained with PN, see Eq. (33) in arXiv:2311.07456 */
    n_1A = NRTidalv3_coeffs[12]
    n_3over2A = NRTidalv3_coeffs[13]
    n_2A = NRTidalv3_coeffs[14]
    d_3over2A = NRTidalv3_coeffs[15]

    n_1B = NRTidalv3_coeffs[16]
    n_3over2B = NRTidalv3_coeffs[17]
    n_2B = NRTidalv3_coeffs[18]
    d_3over2B = NRTidalv3_coeffs[19]

    # Rewriting Eq. (30) and (32) in arXiv:2311.07456 in terms of Mf instead of x */
    NRphasetermA=-((c_NewtA*dynkappaA*x**(5.0/3.0)*(1 + x**(2.0/3.0)*n_1A + Mf*n_3over2A*PI + x**(4.0/3.0)*n_2A + x**(5.0/3.0)*n_5over2A + x**2 * n_3A))/((1 + d_1A*x**(2.0/3.0) + d_3over2A*Mf*PI )))
    NRphasetermB=-((c_NewtB*dynkappaB*x**(5.0/3.0)*(1 + x**(2.0/3.0)*n_1B + Mf*n_3over2B*PI + x**(4.0/3.0)*n_2B + x**(5.0/3.0)*n_5over2B + x**2 * n_3B))/((1 + d_1B*x**(2.0/3.0) + d_3over2B*Mf*PI )))

    NRphaseNRT = NRphasetermA + NRphasetermB

    PNtidalphaseA = -((c_NewtA*kappaA*x**(5.0/3.0)*(1 + x**(2.0/3.0)*c_1A + Mf*c_3over2A*PI + x**(4.0/3.0)*c_2A + x**(5.0/3.0)*c_5over2A)))
    PNtidalphaseB = -((c_NewtB*kappaB*x**(5.0/3.0)*(1 + x**(2.0/3.0)*c_1B + Mf*c_3over2B*PI + x**(4.0/3.0)*c_2B + x**(5.0/3.0)*c_5over2B)))

    PNtidalphase = PNtidalphaseA + PNtidalphaseB

    #DERIVATIVES

    s2s3b = -s2*s3
    s2Mfb = s2*Mf*PI*2.0
    exps2s3b = jnp.cosh(s2s3b) + jnp.sinh(s2s3b)
    exps2Mfb = jnp.cosh(s2Mfb) + jnp.sinh(s2Mfb)

    dynk2barfunc_deriv = (s1-1.0)*(2.0*PI)*s2*exps2Mfb*exps2s3b*(1/((1 + exps2Mfb*exps2s3b)*(1 + exps2Mfb*exps2s3b))) -2.0*PI*((s1) - 1.0)*s2*exps2s3/((1.0 + exps2s3)*(1.0 + exps2s3))

    NRTuned_dphaseA1=-((c_NewtA*kappaA*dynk2barfunc_deriv*x**(5.0/3.0)*(1 +  x**(2.0/3.0)*n_1A + Mf*n_3over2A*PI + x**(4.0/3.0)*n_2A + x**(5.0/3.0)*n_5over2A + x**2 * n_3A))/((1 + d_1A*x**(2.0/3.0) + d_3over2A*Mf*PI)))
    NRTuned_dphaseB1=-((c_NewtB*kappaB*dynk2barfunc_deriv*x**(5.0/3.0)*(1 + x**(2.0/3.0)*n_1B + Mf*n_3over2B*PI + x**(4.0/3.0)*n_2B + x**(5.0/3.0)*n_5over2B + x**2 * n_3B))/((1 + d_1B*x**(2.0/3.0) + d_3over2B*Mf*PI)))

    NRTuned_dphaseA2=((c_NewtA*kappaA*dynk2barfunc*Mf**(2.0/3.0) * PI**(5.0/3.0) * (-5 - x**(2.0/3.0)*(3*d_1A + 7*n_1A) - 2*Mf*(d_3over2A + 4*n_3over2A)*PI - x**(4.0/3.0)*(5*d_1A*n_1A + 9*n_2A) - 2*x**(5.0/3.0)*(2*d_3over2A*n_1A + 3*d_1A*n_3over2A + 5*n_5over2A) - x**2*(7*d_1A*n_2A + 11*n_3A + 5*d_3over2A*n_3over2A) - 2*x**(7.0/3.0)*(3*d_3over2A*n_2A + 4*d_1A*n_5over2A) - x**(8.0/3.0)*(9*d_1A*n_3A + 7*d_3over2A*n_5over2A) - 2*x**3 *(4*d_3over2A*n_3A)))/(3.*jnp.pow(1 + d_1A*x**(2.0/3.0) + d_3over2A*Mf*PI, 2)))
    NRTuned_dphaseB2=((c_NewtB*kappaB*dynk2barfunc*Mf**(2.0/3.0) * PI**(5.0/3.0) * (-5 - x**(2.0/3.0)*(3*d_1B + 7*n_1B) - 2*Mf*(d_3over2B + 4*n_3over2B)*PI - x**(4.0/3.0)*(5*d_1B*n_1B + 9*n_2B) - 2*x**(5.0/3.0)*(2*d_3over2B*n_1B + 3*d_1B*n_3over2B + 5*n_5over2B) - x**2*(7*d_1B*n_2B + 11*n_3B + 5*d_3over2B*n_3over2B) - 2*x**(7.0/3.0)*(3*d_3over2B*n_2B + 4*d_1B*n_5over2B) - x**(8.0/3.0)*(9*d_1B*n_3B + 7*d_3over2B*n_5over2B) - 2*x**3 *(4*d_3over2B*n_3B)))/(3.*jnp.pow(1 + d_1B*x**(2.0/3.0) + d_3over2B*Mf*PI, 2)))

    factorA = -c_NewtA*kappaA
    factorB = -c_NewtB*kappaB

    NRTuned_dphaseNRT = NRTuned_dphaseA1 + NRTuned_dphaseA2 + NRTuned_dphaseB1 + NRTuned_dphaseB2

    PNpolyA = (1 + x**(2.0/3.0)*c_1A + Mf*c_3over2A*PI + x**(4.0/3.0)*c_2A + x**(5.0/3.0)*c_5over2A)
    PNpolyB = (1 + x**(2.0/3.0)*c_1B + Mf*c_3over2B*PI + x**(4.0/3.0)*c_2B + x**(5.0/3.0)*c_5over2B)


    dPNpolyA = (2./3.)*Mf**(-1.0/3.0)*c_1A*PI**(2.0/3.0) + c_3over2A*PI + (4./3.)*Mf**(1.0/3.0)*c_2A*PI**(4.0/3.0) + (5./3.)*Mf**(2.0/3.0)*c_5over2A*PI**(5.0/3.0)
    dPNpolyB = (2./3.)*Mf**(-1.0/3.0)*c_1B*PI**(2.0/3.0) + c_3over2B*PI + (4./3.)*Mf**(1.0/3.0)*c_2B*PI**(4.0/3.0) + (5./3.)*Mf**(2.0/3.0)*c_5over2B*PI**(5.0/3.0)

    PNTuned_dphaseA = factorA*((5./3.)*Mf**(2.0/3.0)*PI**(5.0/3.0)*PNpolyA + x**(5.0/3.0)*dPNpolyA)
    PNTuned_dphaseB = factorB*((5./3.)*Mf**(2.0/3.0)*PI**(5.0/3.0)*PNpolyB + x**(5.0/3.0)*dPNpolyB)

    PNTuned_dphase = PNTuned_dphaseA + PNTuned_dphaseB

    Mfmerger = _get_merger_frequency(theta) * M_s

    Mftaperstart = 1.15*Mfmerger
    Mftaperend = 1.35*Mfmerger


    # Eq. (46) in arXiv:2311.07456 */
    plancktaperfn = general_planck_taper(Mf, Mftaperstart, Mftaperend)
    dPlancktaper = dGeneral_planck_taper(Mf, Mftaperstart, Mftaperend) * plancktaperfn**2

    NRTuned_dphase = NRTuned_dphaseNRT*(1.0 - plancktaperfn) + PNTuned_dphase*plancktaperfn - (NRphaseNRT - PNtidalphase)*dPlancktaper


    dphase+=NRTuned_dphase

    return dphase * M_s


def dGeneral_planck_taper(x, y1, y2):
    exp_arg = (y2 - y1)/(x - y1) + (y2 - y1)/(x - y2)
    exp_fn = jnp.cosh(exp_arg) + jnp.sinh(exp_arg)
    dfactor_taper = jnp.where(
        x < y1,
        0,
        jnp.where(
            x > y2,
            0,
            -(exp_fn)*(-(y2 - y1)/((x - y1) * (x - y1)) - (y2 - y1)/((x - y2) * (x - y2))),
        ),
    )

    return dfactor_taper