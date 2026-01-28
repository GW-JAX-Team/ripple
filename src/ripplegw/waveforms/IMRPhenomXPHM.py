
import jax
from jax import jit
import jax.numpy as jnp
from jax import vmap
import numpy as np
from .IMRPhenomD_QNMdata import fM_CUT, QNMData_a, QNMData_fRD, QNMData_fdamp
from ..constants import C, PI, MSUN, MTSUN_SI, MTSUN
from ..typing import Array
from ripplegw import Mc_eta_to_ms
from .spherical_harmonics import (compute_sminus2_l2, compute_sminus2_l3, compute_sminus2_l4)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from . import LALSimIMRPhenomX_precession as pPrec
from .initialise_MSA_system import IMRPhenomX_Return_phi_zeta_costhetaL_MSA
from .LALSimIMRPhenomX_internals import IMRPhenomXSetWaveformVariables

from .LALSimIMRPhenomD_internals import DPhiMRD
from .LALSimIMRPhenomUtils import XLALSimPhenomUtilsChiP

#from .LALSimIMRPhenomX_precession import Get_alpha_epsilon_offset



uGpc = 3.085677581491367278913937957796471611e25
#3.085677581491367278913937957796471611e25 # meters
GMsun_over_c2 = MTSUN_SI * C
#1.476625061404649406193430731479084713e3 # meters
GMsun_over_c2_Gpc = GMsun_over_c2/uGpc


#MTSUN_SI = 4.925491025543575903411922162094833998e-6

# TF2 coefficient array indices (for JIT-compatible array-based storage)
TF2_ZERO = 0
TF2_ONE = 1
TF2_TWO = 2
TF2_THREE = 3
TF2_FOUR = 4
TF2_FIVE = 5
TF2_FIVE_LOG = 6
TF2_SIX = 7
TF2_SIX_LOG = 8
TF2_SEVEN = 9
TF2_NUM_COEFFS = 10

# PhiInsp coefficient array indices (for JIT-compatible array-based storage)
PHI_INITIAL_PHASING = 0
PHI_TWO_THIRDS = 1
PHI_THIRD = 2
PHI_THIRD_LOG = 3
PHI_LOG = 4
PHI_MIN_THIRD = 5
PHI_MIN_TWO_THIRDS = 6
PHI_MIN_ONE = 7
PHI_MIN_FOUR_THIRDS = 8
PHI_MIN_FIVE_THIRDS = 9
PHI_ONE = 10
PHI_FOUR_THIRDS = 11
PHI_FIVE_THIRDS = 12
PHI_TWO = 13
PHI_NUM_COEFFS = 14

# Amplitude coefficient array indices (for JIT-compatible array-based storage)
AMP_TWO_THIRDS = 0
AMP_ONE = 1
AMP_FOUR_THIRDS = 2
AMP_FIVE_THIRDS = 3
AMP_TWO = 4
AMP_SEVEN_THIRDS = 5
AMP_EIGHT_THIRDS = 6
AMP_THREE = 7
AMP_NUM_COEFFS = 8

# Alpha coefficient array indices (for JIT-compatible array-based storage)
ALPHA_1 = 0
ALPHA_2 = 1
ALPHA_3 = 2
ALPHA_4 = 3
ALPHA_5 = 4
ALPHA_NUM_COEFFS = 5

# Beta coefficient array indices (for JIT-compatible array-based storage)
BETA_1 = 0
BETA_2 = 1
BETA_3 = 2
BETA_NUM_COEFFS = 3

# Sigma coefficient array indices (for JIT-compatible array-based storage)
SIGMA_1 = 0
SIGMA_2 = 1
SIGMA_3 = 2
SIGMA_4 = 3
SIGMA_NUM_COEFFS = 4


@jit
def SpinWeighted_SphericalHarmonic(theta, modes, phi=0.):
            # Taken from arXiv:0709.0093v3 eq. (II.7), (II.8) and LALSimulation for the s=-2 case and up to l=4.
            # We assume already phi=0 and s=-2 to simplify the function

            Ylm    = jnp.where(modes==21, jnp.sqrt( 5.0 / ( 16.0 * jnp.pi ) ) * jnp.sin( theta )*( 1.0 + jnp.cos( theta )), jnp.where(modes==22, jnp.sqrt( 5.0 / ( 64.0 * jnp.pi ) ) * ( 1.0 + jnp.cos( theta ))*( 1.0 + jnp.cos( theta )), jnp.where(modes==32, jnp.sqrt(7.0/jnp.pi)*((jnp.cos(theta*0.5))**(4.0))*(-2.0 + 3.0*jnp.cos(theta))*0.5, jnp.where(modes==33, -jnp.sqrt(21.0/(2.0*jnp.pi))*((jnp.cos(theta/2.0))**(5.0))*jnp.sin(theta*0.5), jnp.where(modes==43, -3.0*jnp.sqrt(7.0/(2.0*jnp.pi))*((jnp.cos(theta*0.5))**5.0)*(-1.0 + 2.0*jnp.cos(theta))*jnp.sin(theta*0.5), 3.0*jnp.sqrt(7.0/jnp.pi)*((jnp.cos(theta*0.5))**6.0)*(jnp.sin(theta*0.5)*jnp.sin(theta*0.5)))))))
            Ylminm = jnp.where(modes==21, jnp.sqrt( 5.0 / ( 16.0 * jnp.pi ) ) * jnp.sin( theta )*( 1.0 - jnp.cos( theta )), jnp.where(modes==22, jnp.sqrt( 5.0 / ( 64.0 * jnp.pi ) ) * ( 1.0 - jnp.cos( theta ))*( 1.0 - jnp.cos( theta )), jnp.where(modes==32, jnp.sqrt(7.0/(4.0*jnp.pi))*(2.0 + 3.0*jnp.cos(theta))*((jnp.sin(theta*0.5))**(4.0)), jnp.where(modes==33, jnp.sqrt(21.0/(2.0*jnp.pi))*jnp.cos(theta*0.5)*((jnp.sin(theta*0.5))**(5.)), jnp.where(modes==43, 3.0*jnp.sqrt(7.0/(2.0*jnp.pi))*jnp.cos(theta*0.5)*(1.0 + 2.0*jnp.cos(theta))*((jnp.sin(theta*0.5))**5.0), 3.0*jnp.sqrt(7.0/jnp.pi)*(jnp.cos(theta*0.5)*jnp.cos(theta*0.5))*((jnp.sin(theta*0.5))**6.0))))))

            return Ylm, Ylminm

@jit
def OnePointFiveSpinPN(infreqs, ChiS, ChiA, mms, modes, eta, Seta):
            # PN amplitudes function, needed to scale

            v  = jnp.moveaxis((2.*jnp.pi*infreqs/mms)**(1./3.), len(infreqs.shape)-1, len(infreqs.shape) - 2)
            v2 = v*v
            v3 = v2*v

            reshModes = jnp.expand_dims(modes, len(modes.shape))
            Hlm = jnp.where(reshModes==21, (jnp.sqrt(2.0) / 3.0) * (v * Seta - v2 * 1.5 * (ChiA + Seta * ChiS) + v3 * Seta * ((335.0 / 672.0) + (eta * 117.0 / 56.0)) + v3*v * (ChiA * (3427.0 / 1344. - eta * 2101.0 / 336.) + Seta * ChiS * (3427.0 / 1344 - eta * 965 / 336) + Seta * (-1j * 0.5 - jnp.pi - 2 * 1j * 0.69314718056))), jnp.where(reshModes==22, 1., jnp.where(reshModes==32, (1.0 / 3.0) * jnp.sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta)), jnp.where(reshModes==33, 0.75 * jnp.sqrt(5.0 / 7.0) * (v * Seta), jnp.where(reshModes==43, 0.75 * jnp.sqrt(3.0 / 35.0) * v3 * Seta * (1.0 - 2.0 * eta), (4.0 / 9.0) * jnp.sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta))))))

            # Compute the final PN Amplitude at Leading Order in Mf

            return jnp.pi * jnp.sqrt(eta * 2. / 3.) * (v**(-3.5)) * abs(Hlm)


@jit
def _radiatednrg(eta, chi1, chi2):
        """
        Compute the total radiated energy, as in `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_ eq. (3.7) and (3.8).
        
        :param array or float eta: Symmetric mass ratio of the objects.
        :param array or float chi1: Spin of the primary object.
        :param array or float chi2: Spin of the secondary object.
        :return: Total energy radiated by the system.
        :rtype: array or float
        
        """
        # This is needed to stabilize JAX derivatives
        Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)

        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)

        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    

@jit
def _finalspin(eta, chi1, chi2):
        """
        Compute the spin of the final object, as in LALSimIMRPhenomD_internals.c line 161 and 142, which is taken from `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_ eq. (3.6).
        
        :param array or float eta: Symmetric mass ratio of the objects.
        :param array or float chi1: Spin of the primary object.
        :param array or float chi2: Spin of the secondary object.
        :return: The spin of the final object.
        :rtype: array or float
        
        """
        # This is needed to stabilize JAX derivatives
        Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2)
        af1 = eta*(3.4641016151377544 - 4.399247300629289*eta + 9.397292189321194*eta*eta - 13.180949901606242*eta*eta*eta)
        af2 = eta*(s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) + (0.1014665242971878 - 2.0967746996832157*eta)*s))
        af3 = eta*(s*((-1.3546806617824356 + 4.108962025369336*eta)*s*s + (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
        return af1 + af2 + af3

@jit
def compute_fring_and_fdamp(
    aeff: Array,
    finMass: Array,
    modes: Array,
    chip: Array,
    m1ByM: Array,
) -> tuple[Array, Array]:
    """
    Compute the ringdown and damping frequencies for different (l,m) modes.

    Based on fits from LALSimIMRPhenomHM.c for quasi-normal mode frequencies.

    Parameters
    ----------
    aeff : Array
        Effective final spin of the remnant black hole.
    finMass : Array
        Final mass of the remnant (1 - E_rad), dimensionless.
    modes : Array, optional
        Array of mode identifiers (e.g., [21, 22, 32, 33, 44]).
        Defaults to [21, 22, 32, 33, 44].
    chip : Array, optional
        In-plane spin parameter. If provided along with m1ByM, applies
        PhenomD correction to the (2,2) mode.
    m1ByM : Array, optional
        Mass ratio m1/M. Required if chip is provided.

    Returns
    -------
    fringlm : Array
        Ringdown frequencies for each mode, shape (n_modes, ...).
    fdamplm : Array
        Damping frequencies for each mode, shape (n_modes, ...).
    """

    # Domain mapping for dimensionless BH spin
    alphaRDfr = jnp.log(2. - aeff) / jnp.log(3.)

    # beta = 1. / (2. + l - abs(m)) for each mode
    betaRDfr = jnp.where(
        modes == 21, 1./3.,
        jnp.where(modes == 22, 0.5,
        jnp.where(modes == 32, 1./3.,
        jnp.where(modes == 33, 0.5,
        jnp.where(modes == 43, 1./3., 0.5)))))

    # Compute kappa powers
    kappaRDfr = jnp.expand_dims(alphaRDfr, len(alphaRDfr.shape)) ** betaRDfr
    kappaRDfr2 = kappaRDfr * kappaRDfr
    kappaRDfr3 = kappaRDfr * kappaRDfr2
    kappaRDfr4 = kappaRDfr * kappaRDfr3
    kappaRDfr5 = kappaRDfr4 * kappaRDfr
    kappaRDfr6 = kappaRDfr4 * kappaRDfr2

    # Fit coefficients for each mode (complex exponential form: coeff * exp(phase * 1j))
    tmpRDfr = jnp.where(
        modes == 21,
        (0.589113 * jnp.exp(0.043525 * 1j) +
         0.18896353 * jnp.exp(2.289868 * 1j) * kappaRDfr +
         1.15012965 * jnp.exp(5.810057 * 1j) * kappaRDfr2 +
         6.04585476 * jnp.exp(2.741967 * 1j) * kappaRDfr3 +
         11.12627777 * jnp.exp(5.844130 * 1j) * kappaRDfr4 +
         9.34711461 * jnp.exp(2.669372 * 1j) * kappaRDfr5 +
         3.03838318 * jnp.exp(5.791518 * 1j) * kappaRDfr6),
    jnp.where(
        modes == 22,
        (1.0 + kappaRDfr * (
            1.557847 * jnp.exp(2.903124 * 1j) +
            1.95097051 * jnp.exp(5.920970 * 1j) * kappaRDfr +
            2.09971716 * jnp.exp(2.760585 * 1j) * kappaRDfr2 +
            1.41094660 * jnp.exp(5.914340 * 1j) * kappaRDfr3 +
            0.41063923 * jnp.exp(2.795235 * 1j) * kappaRDfr4)),
    jnp.where(
        modes == 32,
        (1.022464 * jnp.exp(0.004870 * 1j) +
         0.24731213 * jnp.exp(0.665292 * 1j) * kappaRDfr +
         1.70468239 * jnp.exp(3.138283 * 1j) * kappaRDfr2 +
         0.94604882 * jnp.exp(0.163247 * 1j) * kappaRDfr3 +
         1.53189884 * jnp.exp(5.703573 * 1j) * kappaRDfr4 +
         2.28052668 * jnp.exp(2.685231 * 1j) * kappaRDfr5 +
         0.92150314 * jnp.exp(5.841704 * 1j) * kappaRDfr6),
    jnp.where(
        modes == 33,
        (1.5 + kappaRDfr * (
            2.095657 * jnp.exp(2.964973 * 1j) +
            2.46964352 * jnp.exp(5.996734 * 1j) * kappaRDfr +
            2.66552551 * jnp.exp(2.817591 * 1j) * kappaRDfr2 +
            1.75836443 * jnp.exp(5.932693 * 1j) * kappaRDfr3 +
            0.49905688 * jnp.exp(2.781658 * 1j) * kappaRDfr4)),
        # Default case (mode 44)
        (2.0 + kappaRDfr * (
            2.658908 * jnp.exp(3.002787 * 1j) +
            2.97825567 * jnp.exp(6.050955 * 1j) * kappaRDfr +
            3.21842350 * jnp.exp(2.877514 * 1j) * kappaRDfr2 +
            2.12764967 * jnp.exp(5.989669 * 1j) * kappaRDfr3 +
            0.60338186 * jnp.exp(2.830031 * 1j) * kappaRDfr4))))))

    # Extract real and imaginary parts, normalized by 2*pi*finMass
    finMass_expanded = jnp.expand_dims(finMass, len(finMass.shape))
    fringlm = jnp.real(tmpRDfr) / (2. * jnp.pi * finMass_expanded)
    fdamplm = jnp.imag(tmpRDfr) / (2. * jnp.pi * finMass_expanded)

    # Apply PhenomD correction to (2,2) mode if chip is provided

    Sperp = chip * jnp.power(m1ByM, 2)
    finspin_phenomD = jnp.sign(aeff) * jnp.sqrt(Sperp * Sperp + aeff * aeff)

    fring_phenomD = jnp.interp(finspin_phenomD, jnp.array(QNMData_a), jnp.array(QNMData_fRD)) / finMass
    fdamp_phenomD = jnp.interp(finspin_phenomD, jnp.array(QNMData_a), jnp.array(QNMData_fdamp)) / finMass

    # Overwrite index 1 (the 22 mode) with PhenomD values
    fringlm = fringlm.at[1].set(fring_phenomD)
    fdamplm = fdamplm.at[1].set(fdamp_phenomD)

    return fringlm, fdamplm


@jit
def compute_DPhiIns(
    f: Array,
    TF2coeffs: Array,
    TF2OverallAmpl: Array,
    sigma1: Array,
    sigma2: Array,
    sigma3: Array,
    sigma4: Array,
    eta: Array,
) -> Array:
    """
    Compute the derivative of the inspiral phase with respect to frequency.

    This computes d(PhiIns)/df, which includes the TaylorF2 post-Newtonian
    contribution plus higher-order calibrated sigma terms.

    Parameters
    ----------
    f : Array
        Frequency at which to evaluate the derivative.
    TF2coeffs : Array
        Array of TF2 phase coefficients with shape (..., TF2_NUM_COEFFS).
    TF2OverallAmpl : Array
        Overall TF2 amplitude factor, typically 3/(128*eta).
    sigma1, sigma2, sigma3, sigma4 : Array
        Higher-order calibration coefficients for the inspiral phase.
    eta : Array
        Symmetric mass ratio.

    Returns
    -------
    DPhiIns : Array
        Derivative of the inspiral phase at frequency f.
    """
    pif = jnp.pi * f
    pif_1_3 = pif ** (1. / 3.)
    pif_2_3 = pif_1_3 * pif_1_3
    pif_4_3 = pif_2_3 * pif_2_3
    pif_5_3 = pif_4_3 * pif_1_3
    pif_7_3 = pif_5_3 * pif_2_3
    pif_8_3 = pif_7_3 * pif_1_3

    f_1_3 = f ** (1. / 3.)
    f_2_3 = f_1_3 * f_1_3

    # TF2 (post-Newtonian) contribution
    DPhiIns_TF2 = (
        2.0 * TF2coeffs[..., TF2_SEVEN] * TF2OverallAmpl * pif_7_3
        + (TF2coeffs[..., TF2_SIX] * TF2OverallAmpl
           + TF2coeffs[..., TF2_SIX_LOG] * TF2OverallAmpl * (1.0 + jnp.log(pif) / 3.)) * (pif ** 2.)
        + TF2coeffs[..., TF2_FIVE_LOG] * TF2OverallAmpl * pif_5_3
        - TF2coeffs[..., TF2_FOUR] * TF2OverallAmpl * pif_4_3
        - 2. * TF2coeffs[..., TF2_THREE] * TF2OverallAmpl * pif
        - 3. * TF2coeffs[..., TF2_TWO] * TF2OverallAmpl * pif_2_3
        - 4. * TF2coeffs[..., TF2_ONE] * TF2OverallAmpl * pif_1_3
        - 5. * TF2coeffs[..., TF2_ZERO] * TF2OverallAmpl
    ) * jnp.pi / (3. * pif_8_3)

    # Higher-order calibrated sigma terms
    DPhiIns_sigma = (sigma1 + sigma2 * f_1_3 + sigma3 * f_2_3 + sigma4 * f) / eta

    return DPhiIns_TF2 + DPhiIns_sigma


@jit
def compute_PhiInsJoin(
    f: Array,
    PhiInspcoeffs: Array,
    eta: Array,
) -> Array:
    """
    Compute the inspiral phase at a given frequency.

    This evaluates the inspiral phase function using the pre-computed
    PhiInspcoeffs array, which includes TaylorF2 and higher-order terms.

    Parameters
    ----------
    f : Array
        Frequency at which to evaluate the inspiral phase.
    PhiInspcoeffs : Array
        Array of inspiral phase coefficients with shape (..., PHI_NUM_COEFFS).
    eta : Array
        Symmetric mass ratio.

    Returns
    -------
    PhiIns : Array
        Inspiral phase at frequency f.
    """
    # Pre-compute powers of f
    f_1_3 = f ** (1. / 3.)
    f_2_3 = f_1_3 * f_1_3
    f_4_3 = f_2_3 * f_2_3
    f_5_3 = f_4_3 * f_1_3
    f_m1_3 = 1. / f_1_3
    f_m2_3 = f_m1_3 * f_m1_3
    f_m4_3 = f_m2_3 * f_m2_3
    f_m5_3 = f_m4_3 * f_m1_3

    log_pi_f = jnp.log(jnp.pi * f)

    # Inspiral phase with positive and negative powers of f
    PhiIns = (
        PhiInspcoeffs[..., PHI_INITIAL_PHASING]
        # Positive powers of f
        + PhiInspcoeffs[..., PHI_TWO_THIRDS] * f_2_3
        + PhiInspcoeffs[..., PHI_THIRD] * f_1_3
        + PhiInspcoeffs[..., PHI_THIRD_LOG] * f_1_3 * log_pi_f / 3.
        + PhiInspcoeffs[..., PHI_LOG] * log_pi_f / 3.
        # Negative powers of f
        + PhiInspcoeffs[..., PHI_MIN_THIRD] * f_m1_3
        + PhiInspcoeffs[..., PHI_MIN_TWO_THIRDS] * f_m2_3
        + PhiInspcoeffs[..., PHI_MIN_ONE] / f
        + PhiInspcoeffs[..., PHI_MIN_FOUR_THIRDS] * f_m4_3
        + PhiInspcoeffs[..., PHI_MIN_FIVE_THIRDS] * f_m5_3
        # Higher order terms (divided by eta)
        + (PhiInspcoeffs[..., PHI_ONE] * f
           + PhiInspcoeffs[..., PHI_FOUR_THIRDS] * f_4_3
           + PhiInspcoeffs[..., PHI_FIVE_THIRDS] * f_5_3
           + PhiInspcoeffs[..., PHI_TWO] * f * f) / eta
    )

    return PhiIns


@jit
def compute_gamma_coefficients(
    eta: Array,
    eta2: Array,
    xi: Array,
) -> tuple[Array, Array, Array]:
    """
    Compute gamma coefficients for the merger-ringdown amplitude.

    These coefficients appear in arXiv:1508.07253 eq. (19), with numerical
    values from Table 5. They are used to compute the peak frequency fpeak.

    Parameters
    ----------
    eta : Array
        Symmetric mass ratio.
    eta2 : Array
        Symmetric mass ratio squared.
    xi : Array
        Spin parameter, defined as xi = -1 + chiPN.

    Returns
    -------
    gamma1 : Array
        First gamma coefficient.
    gamma2 : Array
        Second gamma coefficient.
    gamma3 : Array
        Third gamma coefficient.
    """
    xi2 = xi * xi
    xi3 = xi2 * xi

    gamma1 = (
        0.006927402739328343
        + 0.03020474290328911 * eta
        + (0.006308024337706171
           - 0.12074130661131138 * eta
           + 0.26271598905781324 * eta2
           + (0.0034151773647198794
              - 0.10779338611188374 * eta
              + 0.27098966966891747 * eta2) * xi
           + (0.0007374185938559283
              - 0.02749621038376281 * eta
              + 0.0733150789135702 * eta2) * xi2) * xi
    )

    gamma2 = (
        1.010344404799477
        + 0.0008993122007234548 * eta
        + (0.283949116804459
           - 4.049752962958005 * eta
           + 13.207828172665366 * eta2
           + (0.10396278486805426
              - 7.025059158961947 * eta
              + 24.784892370130475 * eta2) * xi
           + (0.03093202475605892
              - 2.6924023896851663 * eta
              + 9.609374464684983 * eta2) * xi2) * xi
    )

    gamma3 = (
        1.3081615607036106
        - 0.005537729694807678 * eta
        + (-0.06782917938621007
           - 0.6689834970767117 * eta
           + 3.403147966134083 * eta2
           + (-0.05296577374411866
              - 0.9923793203111362 * eta
              + 4.820681208409587 * eta2) * xi
           + (-0.006134139870393713
              - 0.38429253308696365 * eta
              + 1.7561754421985984 * eta2) * xi2) * xi
    )

    return gamma1, gamma2, gamma3


@jit
def compute_rho_coefficients(
    eta: Array,
    eta2: Array,
    xi: Array,
) -> tuple[Array, Array, Array]:
    """
    Compute rho coefficients for the intermediate amplitude.

    These coefficients appear in arXiv:1508.07253 eq. (30), with numerical
    values from Table 5. They parameterize the intermediate amplitude region.

    Parameters
    ----------
    eta : Array
        Symmetric mass ratio.
    eta2 : Array
        Symmetric mass ratio squared.
    xi : Array
        Spin parameter, defined as xi = -1 + chiPN.

    Returns
    -------
    rho1 : Array
        First rho coefficient.
    rho2 : Array
        Second rho coefficient.
    rho3 : Array
        Third rho coefficient.
    """
    xi2 = xi * xi

    rho1 = (
        3931.8979897196696
        - 17395.758706812805 * eta
        + (3132.375545898835
           + 343965.86092361377 * eta
           - 1.2162565819981997e6 * eta2
           + (-70698.00600428853
              + 1.383907177859705e6 * eta
              - 3.9662761890979446e6 * eta2) * xi
           + (-60017.52423652596
              + 803515.1181825735 * eta
              - 2.091710365941658e6 * eta2) * xi2) * xi
    )

    rho2 = (
        -40105.47653771657
        + 112253.0169706701 * eta
        + (23561.696065836168
           - 3.476180699403351e6 * eta
           + 1.137593670849482e7 * eta2
           + (754313.1127166454
              - 1.308476044625268e7 * eta
              + 3.6444584853928134e7 * eta2) * xi
           + (596226.612472288
              - 7.4277901143564405e6 * eta
              + 1.8928977514040343e7 * eta2) * xi2) * xi
    )

    rho3 = (
        83208.35471266537
        - 191237.7264145924 * eta
        + (-210916.2454782992
           + 8.71797508352568e6 * eta
           - 2.6914942420669552e7 * eta2
           + (-1.9889806527362722e6
              + 3.0888029960154563e7 * eta
              - 8.390870279256162e7 * eta2) * xi
           + (-1.4535031953446497e6
              + 1.7063528990822166e7 * eta
              - 4.2748659731120914e7 * eta2) * xi2) * xi
    )

    return rho1, rho2, rho3


@jit
def compute_v1(
    f: Array,
    Acoeffs: Array,
) -> Array:
    """
    Compute v1, the inspiral amplitude model evaluated at frequency f.

    This evaluates the inspiral amplitude using the pre-computed Acoeffs array.

    Parameters
    ----------
    f : Array
        Frequency at which to evaluate the inspiral amplitude.
    Acoeffs : Array
        Array of amplitude coefficients with shape (..., AMP_NUM_COEFFS).

    Returns
    -------
    v1 : Array
        Inspiral amplitude at frequency f.
    """
    f_2_3 = f ** (2. / 3.)
    f_4_3 = f_2_3 * f_2_3
    f_5_3 = f_4_3 * f ** (1. / 3.)
    f_7_3 = f_4_3 * f
    f_8_3 = f_7_3 * f ** (1. / 3.)

    v1 = (
        1.
        + f_2_3 * Acoeffs[..., AMP_TWO_THIRDS]
        + f_4_3 * Acoeffs[..., AMP_FOUR_THIRDS]
        + f_5_3 * Acoeffs[..., AMP_FIVE_THIRDS]
        + f_7_3 * Acoeffs[..., AMP_SEVEN_THIRDS]
        + f_8_3 * Acoeffs[..., AMP_EIGHT_THIRDS]
        + f * (Acoeffs[..., AMP_ONE]
               + f * Acoeffs[..., AMP_TWO]
               + f * f * Acoeffs[..., AMP_THREE])
    )

    return v1


@jit
def compute_v2(
    eta: Array,
    eta2: Array,
    xi: Array,
) -> Array:
    """
    Compute v2, the amplitude at the intermediate collocation point.

    This is the fitted value of the amplitude at f2, from the collocation
    points in the intermediate region.

    Parameters
    ----------
    eta : Array
        Symmetric mass ratio.
    eta2 : Array
        Symmetric mass ratio squared.
    xi : Array
        Spin parameter, defined as xi = -1 + chiPN.

    Returns
    -------
    v2 : Array
        Amplitude at the intermediate collocation point.
    """
    xi2 = xi * xi

    v2 = (
        0.8149838730507785
        + 2.5747553517454658 * eta
        + (1.1610198035496786
           - 2.3627771785551537 * eta
           + 6.771038707057573 * eta2
           + (0.7570782938606834
              - 2.7256896890432474 * eta
              + 7.1140380397149965 * eta2) * xi
           + (0.1766934149293479
              - 0.7978690983168183 * eta
              + 2.1162391502005153 * eta2) * xi2) * xi
    )

    return v2


@jit
def compute_v3(
    f: Array,
    fring: Array,
    fdamp: Array,
    gamma1: Array,
    gamma2: Array,
    gamma3: Array,
) -> Array:
    """
    Compute v3, the merger-ringdown amplitude model evaluated at frequency f.

    This evaluates the merger-ringdown amplitude from arXiv:1508.07253 eq. (19).

    Parameters
    ----------
    f : Array
        Frequency at which to evaluate (typically f3Interm = fpeak).
    fring : Array
        Ringdown frequency (typically fringlm[1] for the 22 mode).
    fdamp : Array
        Damping frequency (typically fdamplm[1] for the 22 mode).
    gamma1 : Array
        First gamma coefficient.
    gamma2 : Array
        Second gamma coefficient.
    gamma3 : Array
        Third gamma coefficient.

    Returns
    -------
    v3 : Array
        Merger-ringdown amplitude at frequency f.
    """
    df = f - fring
    fdamp_gamma3 = fdamp * gamma3

    v3 = (
        jnp.exp(-df * gamma2 / fdamp_gamma3)
        * (fdamp_gamma3 * gamma1)
        / (df * df + fdamp_gamma3 * fdamp_gamma3)
    )

    return v3


@jit
def compute_d1(
    f: Array,
    eta: Array,
    eta2: Array,
    chi1: Array,
    chi2: Array,
    chi12: Array,
    chi22: Array,
    Seta: Array,
    SetaPlus1: Array,
    rho1: Array,
    rho2: Array,
    rho3: Array,
) -> Array:
    """
    Compute d1, the derivative of the inspiral amplitude model at frequency f.

    This is the derivative of the inspiral amplitude with respect to frequency,
    used for matching conditions at the inspiral-intermediate boundary.

    Parameters
    ----------
    f : Array
        Frequency at which to evaluate (typically f1Interm = AMP_fJoin_INS).
    eta : Array
        Symmetric mass ratio.
    eta2 : Array
        Symmetric mass ratio squared.
    chi1 : Array
        Spin of the primary (z-component).
    chi2 : Array
        Spin of the secondary (z-component).
    chi12 : Array
        chi1 squared.
    chi22 : Array
        chi2 squared.
    Seta : Array
        sqrt(1 - 4*eta), related to mass difference.
    SetaPlus1 : Array
        1 + Seta.
    rho1, rho2, rho3 : Array
        Rho coefficients from compute_rho_coefficients.

    Returns
    -------
    d1 : Array
        Derivative of the inspiral amplitude at frequency f.
    """
    pi = jnp.pi
    pi_2_3 = pi ** (2. / 3.)
    pi_4_3 = pi ** (4. / 3.)
    pi_5_3 = pi ** (5. / 3.)
    pi2 = pi * pi

    f_1_3 = f ** (1. / 3.)
    f_2_3 = f_1_3 * f_1_3
    f_4_3 = f_2_3 * f_2_3
    f_5_3 = f_4_3 * f_1_3

    eta3 = eta2 * eta

    # Term 1: (-969 + 1804*eta) * pi^(2/3) / (1008 * f^(1/3))
    term1 = ((-969. + 1804. * eta) * pi_2_3) / (1008. * f_1_3)

    # Term 2: spin contribution at 1PN
    term2 = ((chi1 * (81. * SetaPlus1 - 44. * eta)
              + chi2 * (81. - 81. * Seta - 44. * eta)) * pi) / 48.

    # Term 3: f^(1/3) * pi^(4/3) term
    term3 = ((-27312085.
              - 10287648. * chi22
              - 10287648. * chi12 * SetaPlus1
              + 10287648. * chi22 * Seta
              + 24. * (-1975055. + 857304. * chi12 - 994896. * chi1 * chi2 + 857304. * chi22) * eta
              + 35371056. * eta2) * f_1_3 * pi_4_3) / 6.096384e6

    # Term 4: f^(2/3) * pi^(5/3) term
    term4 = (5. * f_2_3 * pi_5_3 * (
        chi2 * (-285197. * (-1 + Seta) + 4. * (-91902. + 1579. * Seta) * eta - 35632. * eta2)
        + chi1 * (285197. * SetaPlus1 - 4. * (91902. + 1579. * Seta) * eta - 35632. * eta2)
        + 42840. * (-1 + 4 * eta) * pi)) / 96768.

    # Term 5: f * pi^2 term (note the minus sign in front)
    term5 = -(f * pi2 * (
        -336. * (-3248849057.0 + 2943675504. * chi12 - 3339284256. * chi1 * chi2 + 2943675504. * chi22) * eta2
        - 324322727232. * eta3
        - 7. * (-177520268561. + 107414046432. * chi22 + 107414046432. * chi12 * SetaPlus1
                - 107414046432. * chi22 * Seta
                + 11087290368 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) * pi)
        + 12. * eta * (-545384828789.0 - 176491177632. * chi1 * chi2 + 202603761360. * chi22
                       + 77616. * chi12 * (2610335. + 995766. * Seta)
                       - 77287373856. * chi22 * Seta
                       + 5841690624. * (chi1 + chi2) * pi
                       + 21384760320 * pi2))) / 3.0042980352e10

    # Term 6: rho terms
    term6 = (7.0 / 3.0) * f_4_3 * rho1 + (8.0 / 3.0) * f_5_3 * rho2 + 3. * f * f * rho3

    d1 = term1 + term2 + term3 + term4 + term5 + term6

    return d1


@jit
def compute_d2(
    f: Array,
    fring: Array,
    fdamp: Array,
    gamma1: Array,
    gamma2: Array,
    gamma3: Array,
) -> Array:
    """
    Compute d2, the derivative of the merger-ringdown amplitude at frequency f.

    This is the derivative of the merger-ringdown amplitude (eq. 19 of
    arXiv:1508.07253) with respect to frequency.

    Parameters
    ----------
    f : Array
        Frequency at which to evaluate (typically f3Interm = fpeak).
    fring : Array
        Ringdown frequency (typically fringlm[1] for the 22 mode).
    fdamp : Array
        Damping frequency (typically fdamplm[1] for the 22 mode).
    gamma1 : Array
        First gamma coefficient.
    gamma2 : Array
        Second gamma coefficient.
    gamma3 : Array
        Third gamma coefficient.

    Returns
    -------
    d2 : Array
        Derivative of the merger-ringdown amplitude at frequency f.
    """
    df = f - fring
    fdamp_gamma3 = fdamp * gamma3
    df2 = df * df
    fdamp_gamma3_2 = fdamp_gamma3 * fdamp_gamma3
    denom = df2 + fdamp_gamma3_2

    d2 = (
        ((-2. * fdamp * df * gamma3 * gamma1) / denom - (gamma2 * gamma1))
        / (jnp.exp(df * gamma2 / fdamp_gamma3) * denom)
    )

    return d2


@jit
def compute_delta_coefficients(
    f1Interm: Array,
    f2Interm: Array,
    f3Interm: Array,
    d1: Array,
    d2: Array,
    v1: Array,
    v2: Array,
    v3: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """
    Compute delta coefficients for the intermediate amplitude region.

    These coefficients appear in arXiv:1508.07253 eq. (21) and are used to
    construct the intermediate amplitude as a polynomial interpolation between
    the inspiral and merger-ringdown regions.

    Parameters
    ----------
    f1Interm : Array
        First intermediate frequency (AMP_fJoin_INS).
    f2Interm : Array
        Second intermediate frequency (midpoint).
    f3Interm : Array
        Third intermediate frequency (fpeak).
    d1 : Array
        Derivative of inspiral amplitude at f1Interm.
    d2 : Array
        Derivative of merger-ringdown amplitude at f3Interm.
    v1 : Array
        Inspiral amplitude at f1Interm.
    v2 : Array
        Amplitude at f2Interm (from fit).
    v3 : Array
        Merger-ringdown amplitude at f3Interm.

    Returns
    -------
    delta0, delta1, delta2, delta3, delta4 : tuple[Array, ...]
        Delta coefficients for the intermediate amplitude polynomial.
    """
    # Pre-compute powers of frequencies
    f1 = f1Interm
    f2 = f2Interm
    f3 = f3Interm

    f12 = f1 * f1
    f13 = f1 * f12
    f14 = f1 * f13
    f15 = f1 * f14

    f22 = f2 * f2
    f23 = f2 * f22
    f24 = f2 * f23

    f32 = f3 * f3
    f33 = f3 * f32
    f34 = f3 * f33
    f35 = f3 * f34

    # Pre-compute common denominators
    f1_m_f2 = f1 - f2
    f1_m_f3 = f1 - f3
    f3_m_f2 = f3 - f2

    denom = (f1_m_f2 * f1_m_f2 * f1_m_f3 * f1_m_f3 * f1_m_f3 * f3_m_f2 * f3_m_f2)

    # delta0
    delta0 = -((d2 * f15 * f22 * f3 - 2. * d2 * f14 * f23 * f3 + d2 * f13 * f24 * f3
                - d2 * f15 * f2 * f32 + d2 * f14 * f22 * f32 - d1 * f13 * f23 * f32
                + d2 * f13 * f23 * f32 + d1 * f12 * f24 * f32 - d2 * f12 * f24 * f32
                + d2 * f14 * f2 * f33 + 2. * d1 * f13 * f22 * f33 - 2. * d2 * f13 * f22 * f33
                - d1 * f12 * f23 * f33 + d2 * f12 * f23 * f33 - d1 * f1 * f24 * f33
                - d1 * f13 * f2 * f34 - d1 * f12 * f22 * f34 + 2. * d1 * f1 * f23 * f34
                + d1 * f12 * f2 * f35 - d1 * f1 * f22 * f35
                + 4. * f12 * f23 * f32 * v1 - 3. * f1 * f24 * f32 * v1
                - 8. * f12 * f22 * f33 * v1 + 4. * f1 * f23 * f33 * v1 + f24 * f33 * v1
                + 4. * f12 * f2 * f34 * v1 + f1 * f22 * f34 * v1 - 2. * f23 * f34 * v1
                - 2. * f1 * f2 * f35 * v1 + f22 * f35 * v1
                - f15 * f32 * v2 + 3. * f14 * f33 * v2 - 3. * f13 * f34 * v2 + f12 * f35 * v2
                - f15 * f22 * v3 + 2. * f14 * f23 * v3 - f13 * f24 * v3
                + 2. * f15 * f2 * f3 * v3 - f14 * f22 * f3 * v3 - 4. * f13 * f23 * f3 * v3
                + 3. * f12 * f24 * f3 * v3 - 4. * f14 * f2 * f32 * v3
                + 8. * f13 * f22 * f32 * v3 - 4. * f12 * f23 * f32 * v3) / denom)

    # delta1
    delta1 = -((-(d2 * f15 * f22) + 2. * d2 * f14 * f23 - d2 * f13 * f24
                - d2 * f14 * f22 * f3 + 2. * d1 * f13 * f23 * f3 + 2. * d2 * f13 * f23 * f3
                - 2. * d1 * f12 * f24 * f3 - d2 * f12 * f24 * f3
                + d2 * f15 * f32 - 3. * d1 * f13 * f22 * f32 - d2 * f13 * f22 * f32
                + 2. * d1 * f12 * f23 * f32 - 2. * d2 * f12 * f23 * f32
                + d1 * f1 * f24 * f32 + 2. * d2 * f1 * f24 * f32
                - d2 * f14 * f33 + d1 * f12 * f22 * f33 + 3. * d2 * f12 * f22 * f33
                - 2. * d1 * f1 * f23 * f33 - 2. * d2 * f1 * f23 * f33 + d1 * f24 * f33
                + d1 * f13 * f34 + d1 * f1 * f22 * f34 - 2. * d1 * f23 * f34
                - d1 * f12 * f35 + d1 * f22 * f35
                - 8. * f12 * f23 * f3 * v1 + 6. * f1 * f24 * f3 * v1
                + 12. * f12 * f22 * f32 * v1 - 8. * f1 * f23 * f32 * v1
                - 4. * f12 * f34 * v1 + 2. * f1 * f35 * v1
                + 2. * f15 * f3 * v2 - 4. * f14 * f32 * v2 + 4. * f12 * f34 * v2 - 2. * f1 * f35 * v2
                - 2. * f15 * f3 * v3 + 8. * f12 * f23 * f3 * v3 - 6. * f1 * f24 * f3 * v3
                + 4. * f14 * f32 * v3 - 12. * f12 * f22 * f32 * v3 + 8. * f1 * f23 * f32 * v3) / denom)

    # delta2
    delta2 = -((d2 * f15 * f2 - d1 * f13 * f23 - 3. * d2 * f13 * f23
                + d1 * f12 * f24 + 2. * d2 * f12 * f24
                - d2 * f15 * f3 + d2 * f14 * f2 * f3 - d1 * f12 * f23 * f3 + d2 * f12 * f23 * f3
                + d1 * f1 * f24 * f3 - d2 * f1 * f24 * f3
                - d2 * f14 * f32 + 3. * d1 * f13 * f2 * f32 + d2 * f13 * f2 * f32
                - d1 * f1 * f23 * f32 + d2 * f1 * f23 * f32 - 2. * d1 * f24 * f32 - d2 * f24 * f32
                - 2. * d1 * f13 * f33 + 2. * d2 * f13 * f33 - d1 * f12 * f2 * f33
                - 3. * d2 * f12 * f2 * f33 + 3. * d1 * f23 * f33 + d2 * f23 * f33
                + d1 * f12 * f34 - d1 * f1 * f2 * f34
                + d1 * f1 * f35 - d1 * f2 * f35
                + 4. * f12 * f23 * v1 - 3. * f1 * f24 * v1
                + 4. * f1 * f23 * f3 * v1 - 3. * f24 * f3 * v1
                - 12. * f12 * f2 * f32 * v1 + 4. * f23 * f32 * v1
                + 8. * f12 * f33 * v1 - f1 * f34 * v1 - f35 * v1
                - f15 * v2 - f14 * f3 * v2 + 8. * f13 * f32 * v2
                - 8. * f12 * f33 * v2 + f1 * f34 * v2 + f35 * v2
                + f15 * v3 - 4. * f12 * f23 * v3 + 3. * f1 * f24 * v3
                + f14 * f3 * v3 - 4. * f1 * f23 * f3 * v3 + 3. * f24 * f3 * v3
                - 8. * f13 * f32 * v3 + 12. * f12 * f2 * f32 * v3 - 4. * f23 * f32 * v3) / denom)

    # delta3
    delta3 = -((-2. * d2 * f14 * f2 + d1 * f13 * f22 + 3. * d2 * f13 * f22
                - d1 * f1 * f24 - d2 * f1 * f24
                + 2. * d2 * f14 * f3 - 2. * d1 * f13 * f2 * f3 - 2. * d2 * f13 * f2 * f3
                + d1 * f12 * f22 * f3 - d2 * f12 * f22 * f3 + d1 * f24 * f3 + d2 * f24 * f3
                + d1 * f13 * f32 - d2 * f13 * f32 - 2. * d1 * f12 * f2 * f32
                + 2. * d2 * f12 * f2 * f32 + d1 * f1 * f22 * f32 - d2 * f1 * f22 * f32
                + d1 * f12 * f33 - d2 * f12 * f33 + 2. * d1 * f1 * f2 * f33
                + 2. * d2 * f1 * f2 * f33 - 3. * d1 * f22 * f33 - d2 * f22 * f33
                - 2. * d1 * f1 * f34 + 2. * d1 * f2 * f34
                - 4. * f12 * f22 * v1 + 2. * f24 * v1
                + 8. * f12 * f2 * f3 * v1 - 4. * f1 * f22 * f3 * v1
                - 4. * f12 * f32 * v1 + 8. * f1 * f2 * f32 * v1 - 4. * f22 * f32 * v1
                - 4. * f1 * f33 * v1 + 2. * f34 * v1
                + 2. * f14 * v2 - 4. * f13 * f3 * v2 + 4. * f1 * f33 * v2 - 2. * f34 * v2
                - 2. * f14 * v3 + 4. * f12 * f22 * v3 - 2. * f24 * v3
                + 4. * f13 * f3 * v3 - 8. * f12 * f2 * f3 * v3 + 4. * f1 * f22 * f3 * v3
                + 4. * f12 * f32 * v3 - 8. * f1 * f2 * f32 * v3 + 4. * f22 * f32 * v3) / denom)

    # delta4
    delta4 = -((d2 * f13 * f2 - d1 * f12 * f22 - 2. * d2 * f12 * f22
                + d1 * f1 * f23 + d2 * f1 * f23
                - d2 * f13 * f3 + 2. * d1 * f12 * f2 * f3 + d2 * f12 * f2 * f3
                - d1 * f1 * f22 * f3 + d2 * f1 * f22 * f3 - d1 * f23 * f3 - d2 * f23 * f3
                - d1 * f12 * f32 + d2 * f12 * f32 - d1 * f1 * f2 * f32
                - 2. * d2 * f1 * f2 * f32 + 2. * d1 * f22 * f32 + d2 * f22 * f32
                + d1 * f1 * f33 - d1 * f2 * f33
                + 3. * f1 * f22 * v1 - 2. * f23 * v1
                - 6. * f1 * f2 * f3 * v1 + 3. * f22 * f3 * v1
                + 3. * f1 * f32 * v1 - f33 * v1
                - f13 * v2 + 3. * f12 * f3 * v2 - 3. * f1 * f32 * v2 + f33 * v2
                + f13 * v3 - 3. * f1 * f22 * v3 + 2. * f23 * v3
                - 3. * f12 * f3 * v3 + 6. * f1 * f2 * f3 * v3 - 3. * f22 * f3 * v3) / denom)

    return delta0, delta1, delta2, delta3, delta4


@jit
def compute_full_phase(
    f: Array,
    PhiInspcoeffs: Array,
    eta: Array,
    beta1: Array,
    beta2: Array,
    beta3: Array,
    C1Int: Array,
    C2Int: Array,
    alpha1: Array,
    alpha2: Array,
    alpha3: Array,
    alpha4: Array,
    alpha5: Array,
    fring22: Array,
    fdamp22: Array,
    fMRDJoinPh: Array,
    PHI_fJoin_INS: Array,
    fcutPar: Array,
    C1MRDuse: Array,
    C2MRDuse: Array,
    RhoUse: Array,
    TauUse: Array,
) -> Array:
    """
    Compute the full IMRPhenomX phase across all frequency regimes.

    This function evaluates the gravitational wave phase as a piecewise function:
    - Inspiral phase for f < PHI_fJoin_INS
    - Intermediate phase for PHI_fJoin_INS <= f < fMRDJoinPh
    - Merger-ringdown phase for fMRDJoinPh <= f < fcutPar
    - Zero for f >= fcutPar

    Parameters
    ----------
    f : Array
        Frequency array at which to evaluate the phase.
    PhiInspcoeffs : Array
        Array of inspiral phase coefficients with shape (..., PHI_NUM_COEFFS).
    eta : Array
        Symmetric mass ratio.
    beta1, beta2, beta3 : Array
        Intermediate phase coefficients.
    C1Int, C2Int : Array
        Integration constants for intermediate phase continuity.
    alpha1, alpha2, alpha3, alpha4, alpha5 : Array
        Merger-ringdown phase coefficients.
    fring22 : Array
        Ringdown frequency for the (2,2) mode.
    fdamp22 : Array
        Damping frequency for the (2,2) mode.
    fMRDJoinPh : Array
        Frequency at which the intermediate phase joins the merger-ringdown phase.
    PHI_fJoin_INS : Array
        Frequency at which the inspiral phase joins the intermediate phase.
    fcutPar : Array
        Cutoff frequency above which the phase is set to zero.
    C1MRDuse : Array
        Integration constant for merger-ringdown phase (offset).
    C2MRDuse : Array
        Integration constant for merger-ringdown phase (slope).
    RhoUse : Array
        Ratio fring22/fringlm for mode scaling.
    TauUse : Array
        Ratio fdamplm/fdamp22 for mode scaling.

    Returns
    -------
    phase : Array
        Full phase evaluated at frequency f.
    """
    # Inspiral phase (f < PHI_fJoin_INS)
    phi_inspiral = compute_PhiInsJoin(f, PhiInspcoeffs, eta)

    # Intermediate phase (PHI_fJoin_INS <= f < fMRDJoinPh)
    f3 = f * f * f
    phi_intermediate = (
        (beta1 * f - beta3 / (3. * f3) + beta2 * jnp.log(f)) / eta
        + C1Int + C2Int * f
    )

    # Merger-ringdown phase (fMRDJoinPh <= f < fcutPar)
    phi_mrd = (
        (-alpha2 / f
            + (4. / 3.) * alpha3 * f ** (3. / 4.)
            + alpha1 * f
            + alpha4 * RhoUse * jnp.arctan((f - alpha5 * fring22) / (fdamp22 * RhoUse * TauUse))
        ) / eta
        + C1MRDuse + C2MRDuse * f
    )

    # Combine using nested jnp.where for frequency regime selection
    return jnp.where(
        f < PHI_fJoin_INS,
        phi_inspiral,
        jnp.where(
            f < fMRDJoinPh,
            phi_intermediate,
            jnp.where(f < fcutPar, phi_mrd, 0.)
        )
    )


@jit
def compute_PhisAllModes(
    fgrid: Array,
    PhiInspcoeffs: Array,
    eta: Array,
    beta1: Array,
    beta2: Array,
    beta3: Array,
    C1Int: Array,
    C2Int: Array,
    alpha1: Array,
    alpha2: Array,
    alpha3: Array,
    alpha4: Array,
    alpha5: Array,
    fring22: Array,
    fdamp22: Array,
    fMRDJoinPh: Array,
    PHI_fJoin_INS: Array,
    fcutPar: Array,
    C1MRDHM: Array,
    C2MRDHM: Array,
    Rholm: Array,
    Taulm: Array,
    Map_ai: Array,
    Map_bi: Array,
    Map_amPhi: Array,
    Map_bmPhi: Array,
    Map_arPhi: Array,
    Map_brPhi: Array,
    Map_fiPhi: Array,
    Map_fr: Array,
    PhDBconst: Array,
    PhDCconst: Array,
    PhDBAterm: Array,
    tmpphaseC: Array,
) -> Array:
    """
    Compute the full phase for all higher-order modes across all frequency regimes.

    This function computes the gravitational wave phase for multiple (l,m) modes
    using the PhenomHM frequency mapping. It evaluates the phase in three regimes
    (inspiral, intermediate, merger-ringdown) and applies the appropriate frequency
    scaling for each mode.

    Parameters
    ----------
    fgrid : Array
        Frequency grid at which to evaluate the phases, shape (..., n_freqs, n_modes).
    PhiInspcoeffs : Array
        Array of inspiral phase coefficients with shape (..., PHI_NUM_COEFFS).
    eta : Array
        Symmetric mass ratio.
    beta1, beta2, beta3 : Array
        Intermediate phase coefficients.
    C1Int, C2Int : Array
        Integration constants for intermediate phase continuity.
    alpha1, alpha2, alpha3, alpha4, alpha5 : Array
        Merger-ringdown phase coefficients.
    fring22 : Array
        Ringdown frequency for the (2,2) mode.
    fdamp22 : Array
        Damping frequency for the (2,2) mode.
    fMRDJoinPh : Array
        Frequency at which the intermediate phase joins the merger-ringdown phase.
    PHI_fJoin_INS : Array
        Frequency at which the inspiral phase joins the intermediate phase.
    fcutPar : Array
        Cutoff frequency above which the phase is set to zero.
    C1MRDHM, C2MRDHM : Array
        Integration constants for merger-ringdown phase for each mode.
    Rholm, Taulm : Array
        Mode-dependent scaling factors for ringdown frequency and damping time.
    Map_ai, Map_bi : Array
        Frequency mapping coefficients for the inspiral regime.
    Map_amPhi, Map_bmPhi : Array
        Frequency mapping coefficients for the intermediate regime.
    Map_arPhi, Map_brPhi : Array
        Frequency mapping coefficients for the merger-ringdown regime.
    Map_fiPhi : Array
        Frequency boundary between inspiral and intermediate regimes.
    Map_fr : Array
        Frequency boundary between intermediate and merger-ringdown regimes.
    PhDBconst, PhDCconst : Array
        Phase continuity constants at regime boundaries.
    PhDBAterm, tmpphaseC : Array
        Additional phase correction terms for continuity.

    Returns
    -------
    PhisAllModes : Array
        Full phase for all modes evaluated at the frequency grid.
    """
    # Compute phases for each frequency regime
    phase_inspiral = compute_full_phase(
        (fgrid * Map_ai + Map_bi), PhiInspcoeffs, eta,
        beta1, beta2, beta3, C1Int, C2Int,
        alpha1, alpha2, alpha3, alpha4, alpha5,
        fring22, fdamp22, fMRDJoinPh,
        PHI_fJoin_INS, fcutPar,
        C1MRDHM, C2MRDHM, Rholm, Taulm
    ) / Map_ai

    phase_intermediate = compute_full_phase(
        (fgrid * Map_amPhi + Map_bmPhi), PhiInspcoeffs, eta,
        beta1, beta2, beta3, C1Int, C2Int,
        alpha1, alpha2, alpha3, alpha4, alpha5,
        fring22, fdamp22, fMRDJoinPh,
        PHI_fJoin_INS, fcutPar,
        C1MRDHM, C2MRDHM, Rholm, Taulm
    ) / Map_amPhi

    phase_mrd = compute_full_phase(
        (fgrid * Map_arPhi + Map_brPhi), PhiInspcoeffs, eta,
        beta1, beta2, beta3, C1Int, C2Int,
        alpha1, alpha2, alpha3, alpha4, alpha5,
        fring22, fdamp22, fMRDJoinPh,
        PHI_fJoin_INS, fcutPar,
        C1MRDHM, C2MRDHM, Rholm, Taulm
    ) / Map_arPhi

    # Combine using nested jnp.where for frequency regime selection
    return jnp.where(
        fgrid < Map_fiPhi,
        phase_inspiral,
        jnp.where(
            fgrid < Map_fr,
            -PhDBconst + PhDBAterm + phase_intermediate,
            -PhDCconst + tmpphaseC + phase_mrd
        )
    )


@jit
def compute_temp_phase_coefficients(
    PhiInspcoeffs: Array,
    eta: Array,
    beta1: Array,
    beta2: Array,
    beta3: Array,
    C1Int: Array,
    C2Int: Array,
    alpha1: Array,
    alpha2: Array,
    alpha3: Array,
    alpha4: Array,
    alpha5: Array,
    fring22: Array,
    fdamp22: Array,
    fMRDJoinPh: Array,
    PHI_fJoin_INS: Array,
    fcutPar: Array,
    C1MRDHM: Array,
    C2MRDHM: Array,
    Rholm: Array,
    Taulm: Array,
    Map_ai: Array,
    Map_bi: Array,
    Map_amPhi: Array,
    Map_bmPhi: Array,
    Map_arPhi: Array,
    Map_brPhi: Array,
    Map_fiPhi: Array,
    Map_fr: Array,
) -> tuple[Array, Array, Array, Array]:
    """
    Compute temporary phase coefficients for mode continuity corrections.

    These coefficients ensure phase continuity across the frequency regime
    boundaries when mapping from the (2,2) mode to higher-order modes using
    the PhenomHM frequency scaling.

    Parameters
    ----------
    PhiInspcoeffs : Array
        Array of inspiral phase coefficients with shape (..., PHI_NUM_COEFFS).
    eta : Array
        Symmetric mass ratio.
    beta1, beta2, beta3 : Array
        Intermediate phase coefficients.
    C1Int, C2Int : Array
        Integration constants for intermediate phase continuity.
    alpha1, alpha2, alpha3, alpha4, alpha5 : Array
        Merger-ringdown phase coefficients.
    fring22 : Array
        Ringdown frequency for the (2,2) mode.
    fdamp22 : Array
        Damping frequency for the (2,2) mode.
    fMRDJoinPh : Array
        Frequency at which the intermediate phase joins the merger-ringdown phase.
    PHI_fJoin_INS : Array
        Frequency at which the inspiral phase joins the intermediate phase.
    fcutPar : Array
        Cutoff frequency above which the phase is set to zero.
    C1MRDHM, C2MRDHM : Array
        Integration constants for merger-ringdown phase for each mode.
    Rholm, Taulm : Array
        Mode-dependent scaling factors for ringdown frequency and damping time.
    Map_ai, Map_bi : Array
        Frequency mapping coefficients for the inspiral regime.
    Map_amPhi, Map_bmPhi : Array
        Frequency mapping coefficients for the intermediate regime.
    Map_arPhi, Map_brPhi : Array
        Frequency mapping coefficients for the merger-ringdown regime.
    Map_fiPhi : Array
        Frequency boundary between inspiral and intermediate regimes.
    Map_fr : Array
        Frequency boundary between intermediate and merger-ringdown regimes.

    Returns
    -------
    PhDBconst : Array
        Phase constant at the inspiral-intermediate boundary (scaled by Map_amPhi).
    PhDCconst : Array
        Phase constant at the intermediate-ringdown boundary (scaled by Map_arPhi).
    PhDBAterm : Array
        Phase term from inspiral regime at the boundary (scaled by Map_ai).
    tmpphaseC : Array
        Combined phase correction for ringdown regime continuity.
    """
    # Phase at inspiral-intermediate boundary (Map_amPhi scaling)
    tmpMf_B = Map_amPhi * Map_fiPhi + Map_bmPhi
    PhDBconst = compute_full_phase(
        tmpMf_B.T, PhiInspcoeffs, eta,
        beta1, beta2, beta3, C1Int, C2Int,
        alpha1, alpha2, alpha3, alpha4, alpha5,
        fring22, fdamp22, fMRDJoinPh,
        PHI_fJoin_INS, fcutPar,
        C1MRDHM, C2MRDHM, Rholm, Taulm
    ) / Map_amPhi.T

    # Phase at intermediate-ringdown boundary (Map_arPhi scaling)
    tmpMf_C = Map_arPhi * Map_fr + Map_brPhi
    PhDCconst = compute_full_phase(
        tmpMf_C.T, PhiInspcoeffs, eta,
        beta1, beta2, beta3, C1Int, C2Int,
        alpha1, alpha2, alpha3, alpha4, alpha5,
        fring22, fdamp22, fMRDJoinPh,
        PHI_fJoin_INS, fcutPar,
        C1MRDHM, C2MRDHM, Rholm, Taulm
    ) / Map_arPhi.T

    # Phase from inspiral regime at boundary (Map_ai scaling)
    tmpMf_BA = Map_ai * Map_fiPhi + Map_bi
    PhDBAterm = (compute_full_phase(
        tmpMf_BA.T, PhiInspcoeffs, eta,
        beta1, beta2, beta3, C1Int, C2Int,
        alpha1, alpha2, alpha3, alpha4, alpha5,
        fring22, fdamp22, fMRDJoinPh,
        PHI_fJoin_INS, fcutPar,
        C1MRDHM, C2MRDHM, Rholm, Taulm
    ).T / Map_ai).T

    # Combined correction for ringdown regime continuity
    tmpMf_amfr = Map_amPhi * Map_fr + Map_bmPhi
    tmpphaseC = -PhDBconst + PhDBAterm + compute_full_phase(
        tmpMf_amfr.T, PhiInspcoeffs, eta,
        beta1, beta2, beta3, C1Int, C2Int,
        alpha1, alpha2, alpha3, alpha4, alpha5,
        fring22, fdamp22, fMRDJoinPh,
        PHI_fJoin_INS, fcutPar,
        C1MRDHM, C2MRDHM, Rholm, Taulm
    ) / Map_amPhi.T

    return PhDBconst, PhDCconst, PhDBAterm, tmpphaseC


@jit
def compute_TF2_coefficients(eta, eta2, Seta, chi_s, chi_a, chi1, chi2,
                              chi1dotchi2, chi12, chi22,
                              m1ByM, m2ByM, QuadMon1, QuadMon2):
    """
    Compute TF2 (TaylorF2) phase coefficients as a JAX array.

    This function computes the post-Newtonian phase coefficients used in the
    inspiral phase of the IMRPhenomXPHM waveform model. Returns an array
    instead of a dictionary for JIT compatibility.

    Parameters
    ----------
    eta : array_like
        Symmetric mass ratio
    eta2 : array_like
        eta squared
    Seta : array_like
        sqrt(1 - 4*eta), related to mass difference
    chi_s : array_like
        Symmetric spin combination (chi1 + chi2)/2
    chi_a : array_like
        Antisymmetric spin combination (chi1 - chi2)/2
    chi1, chi2 : array_like
        Individual spin components (z-components)
    chi1dotchi2 : array_like
        Product chi1 * chi2
    chi12, chi22 : array_like
        Squared spins chi1^2, chi2^2
    m1ByM, m2ByM : array_like
        Mass fractions m1/M and m2/M
    QuadMon1, QuadMon2 : array_like
        Quadrupole moment parameters

    Returns
    -------
    TF2coeffs : jnp.ndarray
        Array of shape (..., 10) containing the TF2 coefficients:
        [zero, one, two, three, four, five, five_log, six, six_log, seven]
    """
    # Coefficient 0: 'zero'
    coeff_zero = jnp.ones_like(eta)

    # Coefficient 1: 'one'
    coeff_one = jnp.zeros_like(eta)

    # Coefficient 2: 'two'
    coeff_two = 3715./756. + (55.*eta)/9.

    # Coefficient 3: 'three'
    coeff_three = -16.*jnp.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s

    # Coefficient 4: 'four' - 2PN with quadrupole moments
    coeff_four = (5.*(3058.673/7.056 + 5429./7.*eta + 617.*eta2)/72.
                  + 247./4.8*eta*chi1dotchi2 - 721./4.8*eta*chi1dotchi2
                  + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12
                  + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22
                  + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12
                  + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22)

    # Common part for coefficients 5 and 5_log
    TF2_5coeff_tmp = ((732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s
                      + (732985./2268. + 140.*eta/9.)*Seta*chi_a)

    # Coefficient 5: 'five'
    coeff_five = 38645.*jnp.pi/756. - 65.*jnp.pi*eta/9. - TF2_5coeff_tmp

    # Coefficient 6: 'five_log'
    coeff_five_log = (38645.*jnp.pi/756. - 65.*jnp.pi*eta/9. - TF2_5coeff_tmp)*3.

    # Coefficient 7: 'six' - 3PN with quadrupole moments
    coeff_six = (11583.231236531/4.694215680 - 640./3.*jnp.pi*jnp.pi
                 - 684.8/2.1*jnp.euler_gamma
                 + eta*(-15737.765635/3.048192 + 225.5/1.2*jnp.pi*jnp.pi)
                 + eta2*76.055/1.728 - eta2*eta*127.825/1.296
                 - jnp.log(4.)*684.8/2.1
                 + jnp.pi*chi1*m1ByM*(1490./3. + m1ByM*260.)
                 + jnp.pi*chi2*m2ByM*(1490./3. + m2ByM*260.)
                 + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2
                 + (4703.5/8.4 + 2935./6.*m1ByM - 120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12
                 + (-4108.25/6.72 - 108.5/1.2*m1ByM + 125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12
                 + (4703.5/8.4 + 2935./6.*m2ByM - 120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22
                 + (-4108.25/6.72 - 108.5/1.2*m2ByM + 125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22)

    # Coefficient 8: 'six_log'
    coeff_six_log = -6848./21. * jnp.ones_like(eta)

    # Coefficient 9: 'seven'
    coeff_seven = (77096675.*jnp.pi/254016. + 378515.*jnp.pi*eta/1512.
                   - 74045.*jnp.pi*eta2/756.
                   + (-25150083775./3048192. + 10566655595.*eta/762048.
                      - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s
                   + Seta*((-25150083775./3048192. + 26804935.*eta/6048.
                           - 1985.*eta2/48.)*chi_a))

    # Remove the part that was not available when IMRPhenomD was tuned
    coeff_six = coeff_six - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2
                             + ((4703.5/8.4 + 2935./6.*m1ByM - 120.*m1ByM*m1ByM)
                                + (-4108.25/6.72 - 108.5/1.2*m1ByM + 125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12
                             + ((4703.5/8.4 + 2935./6.*m2ByM - 120.*m2ByM*m2ByM)
                                + (-4108.25/6.72 - 108.5/1.2*m2ByM + 125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)

    # Stack all coefficients into an array
    # Shape: (..., 10) where ... is the batch dimensions from eta
    TF2coeffs = jnp.stack([coeff_zero, coeff_one, coeff_two, coeff_three,
                           coeff_four, coeff_five, coeff_five_log,
                           coeff_six, coeff_six_log, coeff_seven], axis=-1)

    return TF2coeffs

@jit
def compute_PhiInsp_coefficients(TF2coeffs, TF2OverallAmpl, sigma1, sigma2, sigma3, sigma4):
    """
    Compute inspiral phase coefficients as a JAX array.

    This function computes the inspiral phase coefficients from TF2 coefficients
    and sigma calibration parameters. Returns an array instead of a dictionary
    for JIT compatibility.

    Parameters
    ----------
    TF2coeffs : jnp.ndarray
        Array of TF2 coefficients from compute_TF2_coefficients()
    TF2OverallAmpl : array_like
        Overall amplitude factor 3/(128*eta)
    sigma1, sigma2, sigma3, sigma4 : array_like
        Calibration coefficients from phenomenological fits

    Returns
    -------
    PhiInspcoeffs : jnp.ndarray
        Array of shape (..., 14) containing the inspiral phase coefficients:
        [initial_phasing, two_thirds, third, third_log, log, min_third,
         min_two_thirds, min_one, min_four_thirds, min_five_thirds,
         one, four_thirds, five_thirds, two]
    """
    # Coefficient 0: 'initial_phasing'
    initial_phasing = TF2coeffs[..., TF2_FIVE] * TF2OverallAmpl - (jnp.pi / 4)

    # Coefficient 1: 'two_thirds'
    two_thirds = TF2coeffs[..., TF2_SEVEN] * TF2OverallAmpl * (jnp.pi ** (2. / 3.))

    # Coefficient 2: 'third'
    third = TF2coeffs[..., TF2_SIX] * TF2OverallAmpl * (jnp.pi ** (1. / 3.))

    # Coefficient 3: 'third_log'
    third_log = TF2coeffs[..., TF2_SIX_LOG] * TF2OverallAmpl * (jnp.pi ** (1. / 3.))

    # Coefficient 4: 'log'
    log_coeff = TF2coeffs[..., TF2_FIVE_LOG] * TF2OverallAmpl

    # Coefficient 5: 'min_third'
    min_third = TF2coeffs[..., TF2_FOUR] * TF2OverallAmpl * (jnp.pi ** (-1. / 3.))

    # Coefficient 6: 'min_two_thirds'
    min_two_thirds = TF2coeffs[..., TF2_THREE] * TF2OverallAmpl * (jnp.pi ** (-2. / 3.))

    # Coefficient 7: 'min_one'
    min_one = TF2coeffs[..., TF2_TWO] * TF2OverallAmpl / jnp.pi

    # Coefficient 8: 'min_four_thirds'
    min_four_thirds = TF2coeffs[..., TF2_ONE] * TF2OverallAmpl * (jnp.pi ** (-4. / 3.))

    # Coefficient 9: 'min_five_thirds'
    min_five_thirds = TF2coeffs[..., TF2_ZERO] * TF2OverallAmpl * (jnp.pi ** (-5. / 3.))

    # Coefficient 10: 'one'
    one = sigma1

    # Coefficient 11: 'four_thirds'
    four_thirds = sigma2 * 0.75

    # Coefficient 12: 'five_thirds'
    five_thirds = sigma3 * 0.6

    # Coefficient 13: 'two'
    two = sigma4 * 0.5

    # Stack all coefficients into an array
    # Shape: (..., 14) where ... is the batch dimensions
    PhiInspcoeffs = jnp.stack([initial_phasing, two_thirds, third, third_log,
                                log_coeff, min_third, min_two_thirds, min_one,
                                min_four_thirds, min_five_thirds, one,
                                four_thirds, five_thirds, two], axis=-1)

    return PhiInspcoeffs

@jit
def compute_Acoeffs(eta, eta2, Seta, SetaPlus1, chi1, chi2, chi12, chi22, rho1, rho2, rho3):
    """
    Compute the amplitude coefficients as a JAX array (JIT-compatible).

    Parameters
    ----------
    eta : jnp.ndarray
        Symmetric mass ratio
    eta2 : jnp.ndarray
        eta squared
    Seta : jnp.ndarray
        sqrt(1 - 4*eta)
    SetaPlus1 : jnp.ndarray
        1 + Seta
    chi1, chi2 : jnp.ndarray
        Component spins
    chi12, chi22 : jnp.ndarray
        chi1^2 and chi2^2
    rho1, rho2, rho3 : jnp.ndarray
        Higher order amplitude coefficients

    Returns
    -------
    Acoeffs : jnp.ndarray
        Array of shape (..., 8) containing amplitude coefficients
    """
    # Coefficient 0: 'two_thirds'
    two_thirds = ((-969. + 1804.*eta)*(jnp.pi**(2./3.)))/672.

    # Coefficient 1: 'one'
    one = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*jnp.pi)/48.

    # Coefficient 2: 'four_thirds'
    four_thirds = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta
                    + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta
                    + 35371056*eta2) * (jnp.pi**(4./3.)))/8.128512e6

    # Coefficient 3: 'five_thirds'
    five_thirds = ((jnp.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2)
                   + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2)
                   + 42840.*(-1.0 + 4.*eta)*jnp.pi)) / 32256.

    # Coefficient 4: 'two'
    two = - ((jnp.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2
             - 324322727232.*eta2*eta
             - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta
                   + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*jnp.pi)
             + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22
                        + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta
                        + 5841690624.*(chi1 + chi2)*jnp.pi + 21384760320.*jnp.pi*jnp.pi)))/6.0085960704e10

    # Coefficient 5: 'seven_thirds'
    seven_thirds = rho1

    # Coefficient 6: 'eight_thirds'
    eight_thirds = rho2

    # Coefficient 7: 'three'
    three = rho3

    # Stack all coefficients into an array
    # Shape: (..., 8) where ... is the batch dimensions
    Acoeffs = jnp.stack([two_thirds, one, four_thirds, five_thirds,
                         two, seven_thirds, eight_thirds, three], axis=-1)

    return Acoeffs

@jit
def compute_alpha_coefficients(eta, eta2, xi):
    """
    Compute the alpha coefficients appearing in arXiv:1508.07253 eq. (14) as a JAX array.

    These coefficients derive from a fit, with numerical values from arXiv:1508.07253 Tab. 5.

    Parameters
    ----------
    eta : jnp.ndarray
        Symmetric mass ratio
    eta2 : jnp.ndarray
        eta squared
    xi : jnp.ndarray
        -1 + chiPN (PN spin parameter)

    Returns
    -------
    alpha_coeffs : jnp.ndarray
        Array of shape (..., 5) containing alpha1 through alpha5
    """
    # alpha1
    alpha1 = (43.31514709695348 + 638.6332679188081*eta
              + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2
                 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi
                 + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi)

    # alpha2
    alpha2 = (-0.07020209449091723 - 0.16269798450687084*eta
              + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2
                 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi
                 + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi)

    # alpha3
    alpha3 = (9.5988072383479 - 397.05438595557433*eta
              + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2
                 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi
                 + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi)

    # alpha4
    alpha4 = (-0.02989487384493607 + 1.4022106448583738*eta
              + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2
                 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi
                 + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi)

    # alpha5
    alpha5 = (0.9974408278363099 - 0.007884449714907203*eta
              + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2
                 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi
                 + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi)

    # Stack all coefficients into an array
    # Shape: (..., 5) where ... is the batch dimensions
    alpha_coeffs = jnp.stack([alpha1, alpha2, alpha3, alpha4, alpha5], axis=-1)

    return alpha_coeffs

@jit
def compute_beta_coefficients(eta, eta2, xi):
    """
    Compute the beta coefficients appearing in arXiv:1508.07253 eq. (16) as a JAX array.

    These coefficients derive from a fit, with numerical values from arXiv:1508.07253 Tab. 5.

    Parameters
    ----------
    eta : jnp.ndarray
        Symmetric mass ratio
    eta2 : jnp.ndarray
        eta squared
    xi : jnp.ndarray
        -1 + chiPN (PN spin parameter)

    Returns
    -------
    beta_coeffs : jnp.ndarray
        Array of shape (..., 3) containing beta1 through beta3
    """
    # beta1
    beta1 = (97.89747327985583 - 42.659730877489224*eta
             + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2
                + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi
                + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi)

    # beta2
    beta2 = (-3.282701958759534 - 9.051384468245866*eta
             + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2
                + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi
                + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi)

    # beta3
    beta3 = (-0.000025156429818799565 + 0.000019750256942201327*eta
             + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2
                + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi
                + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi)

    # Stack all coefficients into an array
    # Shape: (..., 3) where ... is the batch dimensions
    beta_coeffs = jnp.stack([beta1, beta2, beta3], axis=-1)

    return beta_coeffs

@jit
def compute_sigma_coefficients(eta, eta2, xi):
    """
    Compute the sigma coefficients appearing in arXiv:1508.07253 eq. (28) as a JAX array.

    These coefficients derive from a fit, with numerical values from arXiv:1508.07253 Tab. 5.

    Parameters
    ----------
    eta : jnp.ndarray
        Symmetric mass ratio
    eta2 : jnp.ndarray
        eta squared
    xi : jnp.ndarray
        -1 + chiPN (PN spin parameter)

    Returns
    -------
    sigma_coeffs : jnp.ndarray
        Array of shape (..., 4) containing sigma1 through sigma4
    """
    # sigma1
    sigma1 = (2096.551999295543 + 1463.7493168261553*eta
              + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2
                 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi
                 + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi)

    # sigma2
    sigma2 = (-10114.056472621156 - 44631.01109458185*eta
              + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2
                 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi
                 + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi)

    # sigma3
    sigma3 = (22933.658273436497 + 230960.00814979506*eta
              + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2
                 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi
                 + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi)

    # sigma4
    sigma4 = (-14621.71522218357 - 377812.8579387104*eta
              + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2
                 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi
                 + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi)

    # Stack all coefficients into an array
    # Shape: (..., 4) where ... is the batch dimensions
    sigma_coeffs = jnp.stack([sigma1, sigma2, sigma3, sigma4], axis=-1)

    return sigma_coeffs


@jit
def compute_full_amplitude(
    infreqs, Overallamp, amp0, Acoeffs, fpeak,
    delta0, delta1, delta2, delta3, delta4,
    fringlm_22, fdamplm_22, gamma1, gamma2, gamma3,
    AMP_fJoin_INS, fcutPar
):
    inspiral = compute_v1(infreqs, Acoeffs)
    intermediate = delta0 + infreqs * delta1 + infreqs**2 * (delta2 + infreqs * delta3 + infreqs**2 * delta4)
    merger_ringdown = compute_v3(infreqs, fringlm_22, fdamplm_22, gamma1, gamma2, gamma3)
    
    return Overallamp * amp0 * (infreqs ** (-7./6.)) * jnp.where(
        infreqs < AMP_fJoin_INS,
        inspiral,
        jnp.where(infreqs < fpeak, intermediate,
                  jnp.where(infreqs < fcutPar, merger_ringdown, 0.)))


@jit
def hphc(frequency_array, chirp_mass, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, iota, luminosity_distance, initial_phase, reference_frequency):
    """
    Compute the plus and cross polarisations of the GW as a function of frequency, given the events parameters, avoiding for loops over the modes.
    
    :param array frequency_array: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
    :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
    :return: Plus and cross polarisations of the GW for the chosen events evaluated on the frequency grid.
    :rtype: tuple(array, array)
    
    """
    fcutPar = 0.2
    complShiftm = jnp.array([0., jnp.pi*0.5, 0., -jnp.pi*0.5, jnp.pi, jnp.pi*0.5, 0.])
    # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
    AMP_fJoin_INS = 0.014
    # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
    PHI_fJoin_INS = 0.018
    

    # This function retuns directly the full plus and cross polarisations, avoiding for loops over the modes
    total_mass = chirp_mass/(eta**(3./5.))
    mass_ratio = symmetric_mass_ratio_to_mass_ratio(eta)
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)

    eta2 = eta*eta # These can speed up a bit, we call them multiple times
    chi1, chi2 = chi1z, chi2z

    QuadMon1, QuadMon2 = jnp.ones(total_mass.shape), jnp.ones(total_mass.shape)
    
    chi12, chi22 = chi1*chi1, chi2*chi2
    chi1dotchi2  = chi1*chi2
    # This is needed to stabilize JAX derivatives
    Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    SetaPlus1 = 1.0 + Seta
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    chi1dotchi2    = chi1*chi2

    # These are mass_1/total_mass and mass_2/total_mass
    m1ByM = 0.5 * (1.0 + Seta)
    m2ByM = 0.5 * (1.0 - Seta)

    # We work in dimensionless frequency M*f, not f
    fgrid = total_mass*MTSUN_SI*frequency_array
    # This is MfRef, needed to recover LAL, which sets fRef to f_min if fRef=0
    #reference_frequency  = jnp.amin(fgrid, axis=0)

    reference_frequency = total_mass*MTSUN_SI*reference_frequency
    # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
    chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
    xi = - 1.0 + chiPN
    # Compute final spin, radiated energy and mass
    aeff = _finalspin(eta, chi1, chi2)
    Erad = _radiatednrg(eta, chi1, chi2)
    finMass = 1. - Erad

    # Compute the real and imag parts of the complex ringdown frequency for the (l,m) mode as in LALSimIMRPhenomHM.c line 189
    # These are all fits of the different modes. We directly exploit the fact that the relevant HM in this WF are 6
    #modes = jnp.array([21,22,32,33,43,44]) #
    modes = jnp.array([21, 22, 32, 33, 44])
    
    ells = jnp.floor(modes/10).astype(jnp.int32)
    mms = modes - ells*10
    # Domain mapping for dimnesionless BH spin
    
    chip = XLALSimPhenomUtilsChiP(mass_1, mass_2, 
                                    chi1x, chi1y, 
                                    chi2x, chi2y)

    fringlm, fdamplm = compute_fring_and_fdamp(aeff = aeff,
                                                finMass = finMass,
                                                chip = chip,
                                                m1ByM = m1ByM,
                                                modes = modes)
    
    #print(f"values of fringlm {fringlm} fring {fring} and fring_phenomD {fring_phenomD}")
    #fringlm[1] = fringlm[1]
    #fdamplm[1] = fdamplm[1]
    # Compute sigma coefficients as a JAX array (JIT-compatible)
    sigma_coeffs = compute_sigma_coefficients(eta, eta2, xi)
    sigma1 = sigma_coeffs[..., SIGMA_1]
    sigma2 = sigma_coeffs[..., SIGMA_2]
    sigma3 = sigma_coeffs[..., SIGMA_3]
    sigma4 = sigma_coeffs[..., SIGMA_4]
    
    # Compute beta coefficients as a JAX array (JIT-compatible)
    beta_coeffs = compute_beta_coefficients(eta, eta2, xi)
    beta1 = beta_coeffs[..., BETA_1]
    beta2 = beta_coeffs[..., BETA_2]
    beta3 = beta_coeffs[..., BETA_3]
    
    # Compute alpha coefficients as a JAX array (JIT-compatible)
    alpha_coeffs = compute_alpha_coefficients(eta, eta2, xi)
    alpha1 = alpha_coeffs[..., ALPHA_1]
    alpha2 = alpha_coeffs[..., ALPHA_2]
    alpha3 = alpha_coeffs[..., ALPHA_3]
    alpha4 = alpha_coeffs[..., ALPHA_4]
    alpha5 = alpha_coeffs[..., ALPHA_5]
    
    # Compute the TF2 phase coefficients as a JAX array (JIT-compatible)
    TF2coeffs = compute_TF2_coefficients(
        eta, eta2, Seta, chi_s, chi_a, chi1, chi2,
        chi1dotchi2, chi12, chi22, m1ByM, m2ByM, QuadMon1, QuadMon2
    )
    TF2OverallAmpl = 3./(128. * eta)
    # Compute inspiral phase coefficients as a JAX array (JIT-compatible)
    PhiInspcoeffs = compute_PhiInsp_coefficients(
        TF2coeffs, TF2OverallAmpl, sigma1, sigma2, sigma3, sigma4
    )
    
    #Now compute the coefficients to align the three parts
    
    fInsJoinPh = PHI_fJoin_INS
    fMRDJoinPh = 0.5*fringlm[1]
    
    # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
    # Equations to solve for to get C(1) continuous join
    # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
    # Joining at fInsJoin
    # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
    # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
    # This is the first derivative wrt f of the inspiral phase computed at fInsJoin
    DPhiIns = compute_DPhiIns(fInsJoinPh, TF2coeffs, TF2OverallAmpl, sigma1, sigma2, sigma3, sigma4, eta)
    # This is the first derivative of the Intermediate phase computed at fInsJoin
    DPhiInt = (beta1 + beta3/(fInsJoinPh**4) + beta2/fInsJoinPh)/eta
    
    C2Int = DPhiIns - DPhiInt
    
    # This is the inspiral phase computed at fInsJoin
    PhiInsJoin = compute_PhiInsJoin(fInsJoinPh, PhiInspcoeffs, eta)
    # This is the Intermediate phase computed at fInsJoin
    PhiIntJoin = beta1*fInsJoinPh - beta3/(3.*fInsJoinPh*fInsJoinPh*fInsJoinPh) + beta2*jnp.log(fInsJoinPh)
    
    C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoinPh
    
    # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
    PhiIntTempVal  = (beta1*fMRDJoinPh - beta3/(3.*fMRDJoinPh*fMRDJoinPh*fMRDJoinPh) + beta2*jnp.log(fMRDJoinPh))/eta + C1Int + C2Int*fMRDJoinPh
    DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoinPh**4) + beta2/fMRDJoinPh)/eta
    DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoinPh*fMRDJoinPh) + alpha3/(fMRDJoinPh**(1./4.)) + alpha4/(fdamplm[1]*(1. + (fMRDJoinPh - alpha5*fringlm[1])*(fMRDJoinPh - alpha5*fringlm[1])/(fdamplm[1]*fdamplm[1]))))/eta
    PhiMRJoinTemp  = -(alpha2/fMRDJoinPh) + (4.0/3.0) * (alpha3 * (fMRDJoinPh**(3./4.))) + alpha1 * fMRDJoinPh + alpha4 * jnp.arctan((fMRDJoinPh - alpha5 * fringlm[1])/fdamplm[1])
    
    C2MRD = DPhiIntTempVal - DPhiMRDVal
    C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoinPh

    # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
    gamma1, gamma2, gamma3 = compute_gamma_coefficients(eta, eta2, xi)
    # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
    fpeak = jnp.where(gamma2 >= 1.0, jnp.fabs(fringlm[1] - (fdamplm[1]*gamma3)/gamma2), jnp.fabs(fringlm[1] + (fdamplm[1]*(-1.0 + jnp.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2))


    # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
    rho1, rho2, rho3 = compute_rho_coefficients(eta, eta2, xi)
    # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
    f1Interm = AMP_fJoin_INS
    f3Interm = fpeak
    dfInterm = 0.5*(f3Interm - f1Interm)
    f2Interm = f1Interm + dfInterm
    # Compute inspiral amplitude coefficients as a JAX array (JIT-compatible)
    amp0 = jnp.sqrt(2.0*eta/3.0)*(jnp.pi**(-1./6.))
    Acoeffs = compute_Acoeffs(eta, eta2, Seta, SetaPlus1, chi1, chi2, chi12, chi22, rho1, rho2, rho3)
    # v1 is the inspiral model evaluated at f1Interm
    v1 = compute_v1(f1Interm, Acoeffs)
    # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
    v2 = compute_v2(eta, eta2, xi)
    # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
    v3 = compute_v3(f3Interm, fringlm[1], fdamplm[1], gamma1, gamma2, gamma3)
    
    # d1 is the derivative of the inspiral model evaluated at f1
    d1 = compute_d1(f1Interm, eta, eta2, chi1, chi2, chi12, chi22, Seta, SetaPlus1, rho1, rho2, rho3)
    # d2 is the derivative of the merger-ringdown model evaluated at f3
    d2 = compute_d2(f3Interm, fringlm[1], fdamplm[1], gamma1, gamma2, gamma3)
    # Compute the delta coefficients for the intermediate amplitude
    delta0, delta1, delta2, delta3, delta4 = compute_delta_coefficients(
        f1Interm, f2Interm, f3Interm, d1, d2, v1, v2, v3
    )
    
    # Defined as in LALSimulation - LALSimIMRPhenomUtils.c line 70. Final units are correctly Hz^-1
    # there is a 2 * sqrt(5/(64*pi)) missing w.r.t the standard coefficient, which comes from the (2,2) shperical harmonic

    Overallamp = total_mass * GMsun_over_c2_Gpc * total_mass * MTSUN_SI / luminosity_distance
    
    # Time shift so that peak amplitude is approximately at t=0
    # Use PhenomD-style fring/fdamp/fpeak to match LALSim's IMRPhenomDComputet0
    
    
    t0 = DPhiMRD(fpeak, alpha1, alpha2, alpha3, alpha4, alpha5, fringlm[1], eta, fdamplm[1], 1, 1)
    #t0 = (alpha1 + alpha2/(fpeak_phenomD*fpeak_phenomD) + alpha3/(fpeak_phenomD**(1./4.)) + alpha4/(fdamp_phenomD*(1. + (fpeak_phenomD - alpha5*fring_phenomD)*(fpeak_phenomD - alpha5*fring_phenomD)/(fdamp_phenomD*fdamp_phenomD))))/eta

    phiRef = compute_full_phase(reference_frequency, PhiInspcoeffs, eta,
                                beta1, beta2, beta3, C1Int, C2Int,
                                alpha1, alpha2, alpha3, alpha4, alpha5,
                                fringlm[1], fdamplm[1], fMRDJoinPh,
                                PHI_fJoin_INS, fcutPar,
                                C1MRD, C2MRD, 1, 1)

    #phiRef = completePhase(reference_frequency, C1MRD, C2MRD, 1., 1.) # Matches exactly with lalsimulation
    phi0   = 0.5*phiRef #+ kwargs['Phicoal']
    #FIXME Need to swtich on kwargs['Phicoal'] at some point
    
    # Now compute all the modes, they are 6, we parallelize
    
    Rholm, Taulm = (fringlm[1]/fringlm.T), (fdamplm.T/fdamplm[1])
    # Rholm and Taulm only figure in the MRD part, the rest of the coefficients is the same, recompute only this
    DPhiMRDVal    = (alpha1 + alpha2/(fMRDJoinPh*fMRDJoinPh) + alpha3/(fMRDJoinPh**(1./4.)) + alpha4/(fdamplm[1]*Taulm*(1. + (fMRDJoinPh - alpha5*fringlm[1])*(fMRDJoinPh - alpha5*fringlm[1])/(fdamplm[1]*Taulm*Rholm*fdamplm[1]*Taulm*Rholm))))/eta
    PhiMRJoinTemp = -(alpha2/fMRDJoinPh) + (4.0/3.0) * (alpha3 * (fMRDJoinPh**(3./4.))) + alpha1 * fMRDJoinPh + alpha4 * Rholm* jnp.arctan((fMRDJoinPh - alpha5 * fringlm[1])/(fdamplm[1]*Rholm*Taulm))
    C2MRDHM = DPhiIntTempVal - DPhiMRDVal
    C1MRDHM = (PhiIntTempVal - PhiMRJoinTemp/eta - C2MRDHM*fMRDJoinPh).T
    Rholm, Taulm, DPhiMRDVal, PhiMRJoinTemp, C2MRDHM = Rholm.T, Taulm.T, DPhiMRDVal.T, PhiMRJoinTemp.T, C2MRDHM.T
    
    # Scale input frequencies according to PhenomHM model
    # Compute mapping coefficinets
    Map_flPhi = PHI_fJoin_INS
    Map_fiPhi = Map_flPhi / Rholm
    Map_flAmp = AMP_fJoin_INS
    Map_fiAmp = Map_flAmp / Rholm
    Map_fr = fringlm
    
    Map_ai, Map_bi = 2./mms, 0.

    Map_TrdAmp = Map_fr - fringlm + jnp.expand_dims(fringlm[1], len(fringlm[1].shape))
    Map_TiAmp  = 2. * Map_fiAmp / mms
    Map_amAmp  = (Map_TrdAmp - Map_TiAmp) / (Map_fr - Map_fiAmp)
    Map_bmAmp  = Map_TiAmp - Map_fiAmp * Map_amAmp

    Map_TrdPhi = Map_fr * Rholm
    Map_TiPhi  = 2. * Map_fiPhi / mms
    Map_amPhi  = (Map_TrdPhi - Map_TiPhi) / (Map_fr - Map_fiPhi)
    Map_bmPhi  = Map_TiPhi - Map_fiPhi * Map_amPhi

    Map_arAmp, Map_brAmp = 1., - Map_fr + jnp.expand_dims(fringlm[1], len(fringlm[1].shape))
    Map_arPhi, Map_brPhi = Rholm, 0.
    
    # Now scale as f -> f*a+b for each regime
    fgrid = jnp.expand_dims(fgrid, len(fgrid.shape))# Need a new axis to do all the 6 calculations together

    fgridScaled = jnp.where(fgrid < Map_fiAmp, fgrid*Map_ai + Map_bi, jnp.where(fgrid < Map_fr, fgrid*Map_amAmp + Map_bmAmp, fgrid*Map_arAmp + Map_brAmp))
    # Map the ampliude's range
    # We divide by the leading order l=m=2 behavior, and then scale in the expected PN behavior for the multipole of interest.




    beta_term1  = OnePointFiveSpinPN(fgrid, chi_s, chi_a, mms, modes, eta, Seta)
    beta_term2  = OnePointFiveSpinPN(2.*fgrid/mms, chi_s, chi_a, mms, modes, eta, Seta)
    HMamp_term1 = OnePointFiveSpinPN(fgridScaled, chi_s, chi_a, mms, modes, eta, Seta)
    fgridScaled = jnp.moveaxis(fgridScaled, len(fgridScaled.shape)-1, len(fgridScaled.shape) - 2)
    #fgridScaled = fgridScaled.transpose(0,2,1)
    HMamp_term2 = jnp.pi * jnp.sqrt(eta * 2. / 3.) * ((jnp.pi*fgridScaled)**(-7./6.))

    # The (3,3) and (4,3) modes vanish if eta=0.25 (equal mass case) and the (2,1) mode vanishes if both eta=0.25 and chi1z=chi2z
    # This results in NaNs having 0/0, correct for this using jnp.nan_to_num()
    full_amplitude = compute_full_amplitude(fgridScaled, 
                                            Overallamp, 
                                            amp0, 
                                            Acoeffs, 
                                            fpeak,
                                            delta0, delta1, delta2, delta3, delta4,
                                            fringlm[1], fdamplm[1], gamma1, gamma2, gamma3,
                                            AMP_fJoin_INS, fcutPar)
    
    AmplsAllModes = jnp.nan_to_num(full_amplitude * (beta_term1 / beta_term2) * HMamp_term1 / HMamp_term2)
    
    AmplsAllModes = jnp.moveaxis(AmplsAllModes, len(AmplsAllModes.shape)-1, len(AmplsAllModes.shape) - 2)
    
    #AmplsAllModes = AmplsAllModes.transpose(0,2,1)
    C1MRDHM, C2MRDHM, Rholm, Taulm = C1MRDHM.T, C2MRDHM.T, Rholm.T, Taulm.T

    # Compute temporary phase coefficients for mode continuity
    PhDBconst, PhDCconst, PhDBAterm, tmpphaseC = compute_temp_phase_coefficients(
        PhiInspcoeffs, eta,
        beta1, beta2, beta3, C1Int, C2Int,
        alpha1, alpha2, alpha3, alpha4, alpha5,
        fringlm[1], fdamplm[1], fMRDJoinPh,
        PHI_fJoin_INS, fcutPar,
        C1MRDHM, C2MRDHM, Rholm, Taulm,
        Map_ai, Map_bi, Map_amPhi, Map_bmPhi,
        Map_arPhi, Map_brPhi, Map_fiPhi, Map_fr
    )
    PhDBconst, PhDCconst, PhDBAterm, tmpphaseC = PhDBconst.T, PhDCconst.T, PhDBAterm.T, tmpphaseC.T

    # Compute phases for all modes
    PhisAllModes = compute_PhisAllModes(
        fgrid, PhiInspcoeffs, eta,
        beta1, beta2, beta3, C1Int, C2Int,
        alpha1, alpha2, alpha3, alpha4, alpha5,
        fringlm[1], fdamplm[1], fMRDJoinPh,
        PHI_fJoin_INS, fcutPar,
        C1MRDHM, C2MRDHM, Rholm, Taulm,
        Map_ai, Map_bi, Map_amPhi, Map_bmPhi,
        Map_arPhi, Map_brPhi, Map_fiPhi, Map_fr,
        PhDBconst, PhDCconst, PhDBAterm, tmpphaseC
    )

    #FIXME
    # Save PhisAllModes to dat file for debugging (frequency + modes as columns)
    #freqs_flat = fgrid.flatten()
    #n_modes = PhisAllModes.shape[1] if len(PhisAllModes.shape) > 1 else 1
    #phases_2d = PhisAllModes.reshape(-1, n_modes)  # shape: (nfreqs, n_modes)
    #save_data = np.column_stack([freqs_flat, phases_2d])
    #np.savetxt('PhisAllModes_ripple.dat', save_data, header='f 21 22 32 33 43')


    PhisAllModes = PhisAllModes - jnp.expand_dims(t0, len(t0.shape))*(fgrid - jnp.expand_dims(reference_frequency, len(reference_frequency.shape))) - mms*jnp.expand_dims(phi0, len(phi0.shape)) + complShiftm[mms]

    #print(f"ripple debug t0 value {t0}")
    #print(f"ripple debug phi0 {phi0}")



    modes = jnp.expand_dims(modes, len(modes.shape))
    Y, Ymstar = SpinWeighted_SphericalHarmonic(iota, modes)
    Y, Ymstar = Y.T, jnp.conj(Ymstar).T

    #hp = jnp.sum(AmplsAllModes*jnp.exp(-1j*PhisAllModes)*(0.5*(Y + ((-1)**ells)*Ymstar)), axis=-1)
    #hc = -jnp.sum(AmplsAllModes*jnp.exp(-1j*PhisAllModes)*(-1j* 0.5 * (Y - ((-1)**ells)* Ymstar)), axis=-1)

    hlm = AmplsAllModes * jnp.exp(-1j*PhisAllModes) * jnp.power(-1, ells)

    return hlm




class WaveFormModel(ABC):
    """
    Abstract class to compute waveforms
    
    :param str objType: The kind of system the wf model is made for, can be ``'BBH'``, ``'BNS'`` or ``'NSBH'``.
    :param float fcutPar: The cut frequency factor of the waveform. This can either be given in :math:`\\rm Hz`, as for :py:class:`gwfast.waveforms.TaylorF2_RestrictedPN`, or as an adimensional frequency (Mf), as for the IMR models.
    :param bool, optional is_newtonian: Boolean specifying if the waveform is a simple Newtonian inspiral.
    :param bool, optional is_tidal: Boolean specifying if the waveform includes tidal effects.
    :param bool, optional is_HigherModes: Boolean specifying if the waveform includes the contribution of sub-dominant (higher-order) modes.
    :param bool, optional is_chi1chi2: Boolean specifying if, in the aligned spins only case, the individual spins are used in place of the ``'chiS'`` and ``'chiA'`` combinations.
    :param bool, optional is_Precessing: Boolean specifying if the waveform includes spin-precession effects.
    :param bool, optional is_LAL: Boolean specifying if the waveform comes from the ``LAL`` library.
    :param bool, optional is_prec_ang: Boolean specifying if, in the precessing spin case, the angular variables of the spins are used, namely ``'theta_JN'``, ``'chi1'``, ``'chi2'``, ``'tilt1'``, ``'tilt2'``, ``'phiJL'``, ``'phi12'``.
    :param bool, optional is_eccentric: Boolean specifying if the waveform includes orbital eccentricity.
    :param bool, optional is_holomorphic: Boolean specifying if the waveform function is holomorphic (needed for derivatives handling).
    :param bool, optional apply_fcut: Boolean specifying if the waveform has to be cut at the chosen maximum frequency specified by ``fcutPar`` (as in ``LAL``) or not.
    
    """
    
    def __init__(self, objType, fcutPar, is_newtonian=False, is_tidal=False, is_HigherModes=False, is_chi1chi2=True, is_Precessing=False, is_LAL=False, is_prec_ang=False, is_eccentric=False, is_holomorphic=False, apply_fcut=True):
        """
        Constructor method
        """
        # The kind of system the wf model is made for, can be 'BBH', 'BNS' or 'NSBH'
        self.objType = objType 
        # The cut frequency factor of the waveform, in Hz, to be divided by Mtot (in units of Msun). The method fcut can be redefined, as e.g. in the IMRPhenomD implementation, and fcutPar can be passed as an adimensional frequency (Mf)
        fcutPar = fcutPar
        
        # Dictionary containing the order in which the parameters will appear in the Fisher matrix
        self.ParNums = {'chirp_mass':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chiS':9,  'chiA':10}
        """
        Dictionary containing the number of the rows/columns in which the parameters will appear in the Fisher matrix.
        
        :type: dict(int)
        """
        self.is_newtonian=is_newtonian
        self.is_tidal=is_tidal
        self.is_HigherModes = is_HigherModes
        self.nParams = 11
        self.is_chi1chi2 = is_chi1chi2
        self.is_Precessing = is_Precessing
        self.is_LAL = is_LAL
        self.is_eccentric=is_eccentric
        self.is_holomorphic=is_holomorphic
        self.apply_fcut = apply_fcut
        
        if is_newtonian:
            # In the Newtonian case eta and the spins are not included in the Fisher, since they do not enter the signal
            self.ParNums = {'chirp_mass':0, 'dL':1, 'theta':2, 'phi':3, 'iota':4, 'psi':5, 'tcoal':6, 'Phicoal':7}
            self.nParams = 8
        if (is_Precessing) and (is_tidal):
            if not is_eccentric:
                self.ParNums = {'chirp_mass':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10, 'chi1x':11, 'chi2x':12, 'chi1y':13, 'chi2y':14, 'LambdaTilde':15, 'deltaLambda':16}
                self.nParams = 17
            else:
                self.ParNums = {'chirp_mass':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10, 'chi1x':11, 'chi2x':12, 'chi1y':13, 'chi2y':14, 'LambdaTilde':15, 'deltaLambda':16, 'ecc':17}
                self.nParams = 18
        elif (is_tidal) and (not is_Precessing):
            # Note that the Fisher is computed for LabdaTilde and deltaLambda, but the waveforms accept as input only Lambda1 and Lambda2
            self.ParNums['LambdaTilde']=11
            self.ParNums['deltaLambda']=12
            if not is_eccentric:
                self.nParams = 13
            else:
                self.ParNums['ecc']=13
                self.nParams = 14
        elif (not is_tidal) and (is_Precessing):
            if not is_eccentric:
                self.ParNums = {'chirp_mass':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10, 'chi1x':11, 'chi2x':12, 'chi1y':13, 'chi2y':14}
                self.nParams = 15
            else:
                self.ParNums = {'chirp_mass':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10, 'chi1x':11, 'chi2x':12, 'chi1y':13, 'chi2y':14, 'ecc':15}
                self.nParams = 16
        elif (not is_tidal) and (not is_Precessing) and (is_eccentric):
            self.ParNums['ecc']=11
            self.nParams = 12
        if (not is_Precessing) and (is_chi1chi2):
            self.ParNums['chi1z'] = self.ParNums['chiS']
            self.ParNums['chi2z'] = self.ParNums['chiA']
            self.ParNums.pop('chiS')
            self.ParNums.pop('chiA')
        if (is_Precessing) and (is_prec_ang):
            self.ParNums['chi1']  = self.ParNums['chi1z']
            self.ParNums['chi2']  = self.ParNums['chi2z']
            self.ParNums['tilt1'] = self.ParNums['chi1x']
            self.ParNums['tilt2'] = self.ParNums['chi2x']
            self.ParNums['phiJL'] = self.ParNums['chi1y']
            self.ParNums['phi12'] = self.ParNums['chi2y']
            self.ParNums['theta_JN'] = self.ParNums['iota']
            
            self.ParNums.pop('chi1z')
            self.ParNums.pop('chi2z')
            self.ParNums.pop('chi1x')
            self.ParNums.pop('chi2x')
            self.ParNums.pop('chi1y')
            self.ParNums.pop('chi2y')
            self.ParNums.pop('iota')
        
        self.ParNums = dict(sorted(self.ParNums.items(), key=lambda item: item[1]))
    @abstractmethod    
    def Phi(self, f, **kwargs):
        """
        Compute the phase of the GW as a function of frequency, given the events parameters.

        We compute here only the GW phase, not the full phase of the signal, which also includes the reference phase and the time of coalescence.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: GW phase for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        pass
    
    @abstractmethod
    def Ampl(self, f, **kwargs):
        """
        Compute the amplitude of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the amplitude of, as in :py:data:`events`.
        :return: GW amplitude for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        pass
        
    def tau_star(self, f, **kwargs):
        # The relation among the time to coalescence (in seconds) and the frequency (in Hz). We use as default 
        # the expression in M. Maggiore - Gravitational Waves Vol. 1 eq. (4.21), valid in Newtonian and restricted PN approximation
        """
        Compute the time to coalescence (in seconds) as a function of frequency (in :math:`\\rm Hz`), given the events parameters.
        
        :param array f: Frequency grid on which the time to coalescence will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
        :return: time to coalescence for the chosen events evaluated on the frequency grid, in seconds.
        :rtype: array
        
        """
        return 2.18567 * ((1.21/kwargs['chirp_mass'])**(5./3.)) * ((100/f)**(8./3.))
    
    def fcut(self, **kwargs):
        """
        Compute the cut frequency of the waveform as a function of the events parameters, in :math:`\\rm Hz`.
        
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the cut frequency of, as in :py:data:`events`.
        :return: Cut frequency of the waveform for the chosen events, in :math:`\\rm Hz`.
        :rtype: array
        
        """
        return fcutPar/(kwargs['chirp_mass']/(kwargs['eta']**(3./5.)))


class IMRPhenomXPHM(WaveFormModel):
    """
    IMRPhenomHM waveform model.
    
    Relevant references:
        [1] `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_
        
        [2] `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>`_
        
        [3] `arXiv:1708.00404 <https://arxiv.org/abs/1708.00404>`_
        
        [4] `arXiv:1909.10010 <https://arxiv.org/abs/1909.10010>`_
    
    :param float, optional fRef: Reference frequency of the waveform, in :math:`\\rm Hz`. If not provided, the minimum of the frequency grid will be used.
    :param kwargs: Optional arguments to be passed to the parent class :py:class:`WaveFormModel`, such as ``is_chi1chi2``.
        
    """
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253, arXiv:1708.00404, arXiv:1909.10010
    def __init__(self, reference_frequency=None, **kwargs):
        """
        Constructor method
        """
        # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        AMP_fJoin_INS = 0.014
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        PHI_fJoin_INS = 0.018
        
        
        #super().__init__('BBH', fcutPar, is_HigherModes=True, **kwargs)
        
        # List of phase shifts: the index is the azimuthal number m

        
    def Phi(self, f, **kwargs):
        return None
    

    def Ampl(self, f, **kwargs):
        return None
        
    def generate_waveform_struct(self, m1, m2, chi1z, chi2z,
                                 distance, inclination, phi0,  
                                 duration, minimum_frequency, 
                                 maximum_frequency, 
                                 reference_frequency):
        # distance input is in Gpc. Need to convert it to meters
        lalParams = {}
        m1_SI = m1*MSUN
        m2_SI = m2*MSUN
        deltaF = 1/duration
        distance *= 3.08567758128e25


        pWF = IMRPhenomXSetWaveformVariables(m1_SI,
                                             m2_SI,
                                             chi1z, 
                                             chi2z,
                                             deltaF,
                                             reference_frequency, 
                                             phi0,
                                             minimum_frequency, 
                                             maximum_frequency,
                                             distance,
                                             inclination,
                                             lalParams,
                                             debug = False)


        return pWF
    

    def twistup(self, Mf, mass_1, mass_2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, phiRef_In, inclination, reference_frequency, hlm, pWF):
        "Copy of lalsimulation IMRPhenomXPHMTwistUp"
        "Function to twist up hlms"

        # Check if we are using multibanding for angles. 
        # Default in lalsimulation is True but I will force it to False

        # Check PrecVersion
        # Available options 101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321, 330
        # I will use 223 which is default in lalsimulation

        # Modes 21, 22, 32, 33, 43, 44 in that order

        bigM = 1
        eta = mass_1*mass_2/jnp.power(mass_1+mass_2, 2)
        eta2 = jnp.power(eta, 2)
        chi1L = chi1z
        chi2L = chi2z
        total_mass = mass_1 + mass_2

        mass_1_fraction = mass_1 / total_mass
        mass_2_fraction = mass_2 / total_mass

        delta = mass_1_fraction - mass_2_fraction
        
        orbital_angular_momentum = pPrec.flag_222_223_twoPN_non_spinning_orbitan_angular_momentum(
        eta, eta2, chi1L, chi2L, delta, jnp.power(jnp.pi, 2))
        Msec = (mass_1 + mass_2) * MTSUN
        piM = jnp.pi * Msec
        v_ref = jnp.cbrt(piM * reference_frequency)
        LRef = bigM * bigM * pPrec.XLALSimIMRPhenomXLPNAnsatz(v_ref, eta / v_ref, orbital_angular_momentum[0], orbital_angular_momentum[1], orbital_angular_momentum[2], orbital_angular_momentum[3], orbital_angular_momentum[4], orbital_angular_momentum[5], orbital_angular_momentum[6], orbital_angular_momentum[7], orbital_angular_momentum[8], orbital_angular_momentum[9]) 
        
        theta_JN, Nz_Jf, Nx_Jf, phiJ_Sf, kappa = pPrec.compute_thetaJN_and_kappa(mass_1_fraction, mass_2_fraction, 
                                                            chi1x, chi1y, chi1z, 
                                                            chi2x, chi2y, chi2z, 
                                                            LRef, phiRef_In, inclination)
        
        zeta_polarisations = pPrec.compute_zeta_polarization(mass_1_fraction, mass_2_fraction, chi1x, chi1y, chi1z,
                                                             chi2x, chi2y, chi2z, LRef, phiRef_In, inclination, Nz_Jf, Nx_Jf, kappa)


        def compute_twist_for_mode(mode_idx):
            # mode_idx: 0->21, 1->22, 2->32, 3->33, 4->43, 5->44
            emms = jnp.array([1, 2, 2, 3, 3, 4])

            emm = emms[mode_idx]
            
            #print("what is the value of offset:", alpha_offset_emm, epsilon_offset_emm)
            alpha, epsilon, cos_beta = pPrec.compute_evolved_spin_using_msa(Mf, mass_1, mass_2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, emm, reference_frequency, kappa, phiJ_Sf)
            cBetah, sBetah = IMRPhenomXWignerdCoefficients_cosbeta(cos_beta)

            cexp_i_alpha = jnp.exp(1j * alpha)

            beta_powers = BetaPowers.from_half_angle_trig(cBetah, sBetah)

            # Select the appropriate twist function based on mode_idx
            # Order: 21, 22, 32, 33, 43, 44
            hp_twist, hc_twist = jax.lax.switch(
                mode_idx,
                [
                    lambda: twist_21(cexp_i_alpha, theta_JN, beta_powers),
                    lambda: twist_22(cexp_i_alpha, theta_JN, beta_powers),
                    lambda: twist_32(cexp_i_alpha, theta_JN, beta_powers),
                    lambda: twist_33(cexp_i_alpha, theta_JN, beta_powers),
                    #lambda: twist_43(cexp_i_alpha, pPrec.theta_JN, beta_powers),
                    lambda: twist_44(cexp_i_alpha, theta_JN, beta_powers),
                ]
            )

            return hp_twist, hc_twist, epsilon*emm

        mode_indices = jnp.arange(5)  # 0 to 5 for modes 21, 22, 32, 33, 43, 44
        hp_twist_all_modes, hc_twist_all_modes, epsilon_all_modes = jax.vmap(
            compute_twist_for_mode
        )(mode_indices)


        _hp = jnp.sum(hlm * hp_twist_all_modes.T * jnp.exp(-1j * epsilon_all_modes.T) / 2, axis=1)
        _hc = jnp.sum(hlm * hc_twist_all_modes.T * jnp.exp(-1j * epsilon_all_modes.T) / 2, axis=1)

        hp, hc = apply_polarization_rotation(zeta_polarisations, _hp, _hc)
        
        return hp, hc
    
    def generate_xphm(self, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, distance, inclination, phi0, duration, minimum_frequency, maximum_frequency, reference_frequency):

        pWF = self.generate_waveform_struct(m1, m2, chi1z, chi2z,
                                 distance, inclination, phi0,  
                                 duration, minimum_frequency, 
                                 maximum_frequency, 
                                 reference_frequency)
        
        
        frequency_array = jnp.arange(minimum_frequency, maximum_frequency, 1/duration)
        Mf = XLALSimIMRPhenomXUtilsHztoMf(frequency_array, m1+m2)

        chirp_mass = component_masses_to_chirp_mass(m1, m2)
        eta = m1 * m2 / jnp.power(m1+m2, 2)

        hlm = hphc(frequency_array,
                         chirp_mass = chirp_mass,
                         eta = eta,
                         luminosity_distance=distance,
                         iota = inclination,
                         initial_phase = phi0,
                         chi1x = chi1x,
                         chi1y = chi1y,
                         chi1z = chi1z,
                         chi2x = chi2x,
                         chi2y = chi2y,
                         chi2z = chi2z, 
                         reference_frequency= reference_frequency)
        
        eta = m1*m2/jnp.power(m1+m2, 2)

        hp, hc = self.twistup(Mf, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, phi0, inclination, reference_frequency, hlm, pWF)


        #zeta_polarization = pPrec.zeta_polarization

        #hp, hc = apply_polarization_rotation(zeta_polarization, _hp, _hc)
       


        
        return  hp, hc
        




def GPSt_to_LMST(t_GPS, lat, long):
    """
    Compute the Local Mean Sidereal Time (LMST) in units of fraction of day, from GPS time and location (given as latitude and longitude in degrees)
    
    :param array or float t_GPS: GPS time(s) to convert, in seconds.
    :param float lat: Latitude of the chosen location, in :math:`\\rm deg`.
    :param float long: Longitude of the chosen location, in :math:`\\rm deg`.
    
    :return: Local Mean Sidereal Time(s).
    :rtype: array or float
    
    """
    from astropy.coordinates import EarthLocation
    import astropy.time as aspyt
    import astropy.units as u
    # Uncomment the next two lines in case of troubles with IERS
    #import astropy
    #astropy.utils.iers.conf.iers_degraded_accuracy='ignore'
    loc = EarthLocation(lat=lat*u.deg, lon=long*u.deg)
    t = aspyt.Time(t_GPS, format='gps', location=(loc))
    LMST = t.sidereal_time('mean').value
    return jnp.array(LMST/24.)



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
    cBetah: float
    cBetah2: float
    cBetah3: float
    cBetah4: float
    cBetah5: float
    cBetah6: float
    cBetah7: float
    cBetah8: float
    sBetah: float
    sBetah2: float
    sBetah3: float
    sBetah4: float
    sBetah5: float
    sBetah6: float
    sBetah7: float
    sBetah8: float

    @classmethod
    def from_half_angle_trig(cls, cBetah: float, sBetah: float):
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


    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha

    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha

    cexp_im_alpha_l2 = jnp.stack([cexp_m2i_alpha, cexp_mi_alpha, jnp.ones_like(cexp_i_alpha), cexp_i_alpha, cexp_2i_alpha], axis=0)

    Y2m2 = compute_sminus2_l2(theta_JN, m=-2)
    Y2m1 = compute_sminus2_l2(theta_JN, m=-1)
    Y20 = compute_sminus2_l2(theta_JN, m=0)
    Y21 = compute_sminus2_l2(theta_JN, m=1)
    Y22 = compute_sminus2_l2(theta_JN, m=2)
    Y2mA = jnp.array([Y2m2, Y2m1, Y20, Y21, Y22])



    # Wigner-d coefficients
    # d^2_{-2,2}, d^2_{-1,2}, d^2_{0,2}, d^2_{1,2}, d^2_{2,2}

    d22 = jnp.array([
        beta_powers.sBetah4,
        2.0 * beta_powers.cBetah * beta_powers.sBetah3,
        jnp.sqrt(6) * beta_powers.sBetah2 * beta_powers.cBetah2,
        2.0 * beta_powers.cBetah3 * beta_powers.sBetah,
        beta_powers.cBetah4
    ])

    # Exploit symmetry d^2_{-m,-2} = (-1)^m d^2_{-m,2}. See eq. A2 of Precessing paper
    # d^2_{-2,-2}, d^2_{-1,-2}, d^2_{0,-2}, d^2_{1,-2}, d^2_{2,-2}
    d2m2 = jnp.array([d22[4], -d22[3], d22[2], -d22[1], d22[0]])



    for m in range(-2, 2+1):
        
        A2m2emm = cexp_im_alpha_l2[-m+2] * d2m2[m+2] * Y2mA[m+2]
        #print(f"m {m} and A2m2emm {A2m2emm[0]}")
        A22emmstar = cexp_im_alpha_l2[m+2] * d22[m+2] * jnp.conj(Y2mA[m+2])
        hp_sum += (A2m2emm + A22emmstar)
        hc_sum += 1j*(A2m2emm - A22emmstar) 

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
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha

    cexp_im_alpha_l2 = jnp.stack([cexp_m2i_alpha, cexp_mi_alpha, jnp.ones_like(cexp_i_alpha), cexp_i_alpha, cexp_2i_alpha], axis=0)

    Y2m2 = compute_sminus2_l2(theta_JN, m=-2)
    Y2m1 = compute_sminus2_l2(theta_JN, m=-1)
    Y20 = compute_sminus2_l2(theta_JN, m=0)
    Y21 = compute_sminus2_l2(theta_JN, m=1)
    Y22 = compute_sminus2_l2(theta_JN, m=2)
    Y2mA = jnp.array([Y2m2, Y2m1, Y20, Y21, Y22])

    # Wigner-d coefficients for m'=1
    # d^2_{-2,1}, d^2_{-1,1}, d^2_{0,1}, d^2_{1,1}, d^2_{2,1}
    d21 = jnp.array([
        2.0 * beta_powers.cBetah * beta_powers.sBetah3,
        3.0 * beta_powers.cBetah2 * beta_powers.sBetah2 - beta_powers.sBetah4,
        jnp.sqrt(6) * (beta_powers.cBetah3 * beta_powers.sBetah - beta_powers.cBetah * beta_powers.sBetah3),
        beta_powers.cBetah2 * (beta_powers.cBetah2 - 3.0 * beta_powers.sBetah2),
        -2.0 * beta_powers.cBetah3 * beta_powers.sBetah
    ])

    # Exploit symmetry d^2_{-m,-1} = -(-1)^m d^2_{m,1}. See eq. A2 of Precessing paper.
    # d^2_{-2,-1}, d^2_{-1,-1}, d^2_{0,-1}, d^2_{1,-1}, d^2_{2,-1}
    d2m1 = jnp.array([-d21[4], d21[3], -d21[2], d21[1], -d21[0]])

    for m in range(-2, 2+1):
        # Transfer functions, see eqs. 3.5-3.7 in Precessing paper.
        A2m1emm = cexp_im_alpha_l2[-m+2] * d2m1[m+2] * Y2mA[m+2]
        A21emmstar = cexp_im_alpha_l2[m+2] * d21[m+2] * jnp.conj(Y2mA[m+2])
        hp_sum += (A2m1emm + A21emmstar)
        hc_sum += 1j * (A2m1emm - A21emmstar)

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
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha

    cexp_im_alpha_l3 = jnp.stack([cexp_m3i_alpha, cexp_m2i_alpha, cexp_mi_alpha, jnp.ones_like(cexp_i_alpha), cexp_i_alpha, cexp_2i_alpha, cexp_3i_alpha], axis=0)

    Y3m3 = compute_sminus2_l3(theta=theta_JN, m=-3)
    Y3m2 = compute_sminus2_l3(theta=theta_JN, m=-2)
    Y3m1 = compute_sminus2_l3(theta=theta_JN, m=-1)
    Y30 = compute_sminus2_l3(theta=theta_JN, m=0)
    Y31 = compute_sminus2_l3(theta=theta_JN, m=1)
    Y32 = compute_sminus2_l3(theta=theta_JN, m=2)
    Y33 = compute_sminus2_l3(theta=theta_JN, m=3)
    Y3mA = jnp.array([Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33])

    # Wigner-d coefficients for m'=3
    # d^3_{-3,3}, d^3_{-2,3}, d^3_{-1,3}, d^3_{0,3}, d^3_{1,3}, d^3_{2,3}, d^3_{3,3}
    sqrt6 = jnp.sqrt(6.0)
    sqrt15 = jnp.sqrt(15.0)
    sqrt5 = jnp.sqrt(5.0)

    d33 = jnp.array([
        beta_powers.sBetah6,
        sqrt6 * beta_powers.cBetah * beta_powers.sBetah5,
        sqrt15 * beta_powers.cBetah2 * beta_powers.sBetah4,
        2.0 * sqrt5 * beta_powers.cBetah3 * beta_powers.sBetah3,
        sqrt15 * beta_powers.cBetah4 * beta_powers.sBetah2,
        sqrt6 * beta_powers.cBetah5 * beta_powers.sBetah,
        beta_powers.cBetah6
    ])

    # Exploit symmetry d^3_{-m,-3} = -(-1)^m d^3_{m,3}. See eq. A2 of Precessing paper.
    # d^3_{-3,-3}, d^3_{-2,-3}, d^3_{-1,-3}, d^3_{0,-3}, d^3_{1,-3}, d^3_{2,-3}, d^3_{3,-3}
    d3m3 = jnp.array([d33[6], -d33[5], d33[4], -d33[3], d33[2], -d33[1], d33[0]])

    for m in range(-3, 3+1):
        # Transfer functions
        A3m3emm = cexp_im_alpha_l3[-m+3] * d3m3[m+3] * Y3mA[m+3]
        A33emmstar = cexp_im_alpha_l3[m+3] * d33[m+3] * jnp.conj(Y3mA[m+3])
        hp_sum += (A3m3emm - A33emmstar)
        hc_sum += 1j * (A3m3emm + A33emmstar)

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
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha

    cexp_im_alpha_l3 = jnp.stack([cexp_m3i_alpha, cexp_m2i_alpha, cexp_mi_alpha, jnp.ones_like(cexp_i_alpha), cexp_i_alpha, cexp_2i_alpha, cexp_3i_alpha], axis=0)

    Y3m3 = compute_sminus2_l3(theta=theta_JN, m=-3)
    Y3m2 = compute_sminus2_l3(theta=theta_JN, m=-2)
    Y3m1 = compute_sminus2_l3(theta=theta_JN, m=-1)
    Y30 = compute_sminus2_l3(theta=theta_JN, m=0)
    Y31 = compute_sminus2_l3(theta=theta_JN, m=1)
    Y32 = compute_sminus2_l3(theta=theta_JN, m=2)
    Y33 = compute_sminus2_l3(theta=theta_JN, m=3)
    Y3mA = jnp.array([Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33])

    # Wigner-d coefficients for m'=2
    # d^3_{-3,2}, d^3_{-2,2}, d^3_{-1,2}, d^3_{0,2}, d^3_{1,2}, d^3_{2,2}, d^3_{3,2}
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

    d32 = jnp.array([
        sqrt6 * cBetah * sBetah5,
        sBetah4 * (5.0 * cBetah2 - sBetah2),
        sqrt10 * sBetah3 * (2.0 * cBetah3 - cBetah * sBetah2),
        sqrt30 * cBetah2 * (cBetah2 - sBetah2) * sBetah2,
        sqrt10 * cBetah3 * (cBetah2 * sBetah - 2.0 * sBetah3),
        cBetah4 * (cBetah2 - 5.0 * sBetah2),
        -1.0 * sqrt6 * cBetah5 * sBetah
    ])

    # Exploit symmetry d^3_{-m,-2} = (-1)^m d^3_{m,2}. See eq. A2 of Precessing paper.
    # d^3_{-3,-2}, d^3_{-2,-2}, d^3_{-1,-2}, d^3_{0,-2}, d^3_{1,-2}, d^3_{2,-2}, d^3_{3,-2}
    d3m2 = jnp.array([-d32[6], d32[5], -d32[4], d32[3], -d32[2], d32[1], -d32[0]])

    for m in range(-3, 3+1):
        # Transfer functions, see eqs. 3.5-3.7 in Precessing paper.
        A3m2emm = cexp_im_alpha_l3[-m+3] * d3m2[m+3] * Y3mA[m+3]
        A32emmstar = cexp_im_alpha_l3[m+3] * d32[m+3] * jnp.conj(Y3mA[m+3])
        hp_sum += (A3m2emm - A32emmstar)
        hc_sum += 1j * (A3m2emm + A32emmstar)

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
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_4i_alpha = cexp_i_alpha * cexp_3i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha
    cexp_m4i_alpha = cexp_mi_alpha * cexp_m3i_alpha

    cexp_im_alpha_l4 = jnp.stack([cexp_m4i_alpha, cexp_m3i_alpha, cexp_m2i_alpha, cexp_mi_alpha, jnp.ones_like(cexp_i_alpha), cexp_i_alpha, cexp_2i_alpha, cexp_3i_alpha, cexp_4i_alpha], axis=0)

    Y4m4 = compute_sminus2_l4(theta=theta_JN, m=-4)
    Y4m3 = compute_sminus2_l4(theta=theta_JN, m=-3)
    Y4m2 = compute_sminus2_l4(theta=theta_JN, m=-2)
    Y4m1 = compute_sminus2_l4(theta=theta_JN, m=-1)
    Y40 = compute_sminus2_l4(theta=theta_JN, m=0)
    Y41 = compute_sminus2_l4(theta=theta_JN, m=1)
    Y42 = compute_sminus2_l4(theta=theta_JN, m=2)
    Y43 = compute_sminus2_l4(theta=theta_JN, m=3)
    Y44 = compute_sminus2_l4(theta=theta_JN, m=4)
    Y4mA = jnp.array([Y4m4, Y4m3, Y4m2, Y4m1, Y40, Y41, Y42, Y43, Y44])

    # Wigner-d coefficients for m'=4
    # d^4_{-4,4}, d^4_{-3,4}, d^4_{-2,4}, d^4_{-1,4}, d^4_{0,4}, d^4_{1,4}, d^4_{2,4}, d^4_{3,4}, d^4_{4,4}
    sqrt2 = jnp.sqrt(2.0)
    sqrt7 = jnp.sqrt(7.0)
    sqrt14 = jnp.sqrt(14.0)
    sqrt70 = jnp.sqrt(70.0)

    d44 = jnp.array([
        beta_powers.sBetah8,
        2.0 * sqrt2 * beta_powers.cBetah * beta_powers.sBetah7,
        2.0 * sqrt7 * beta_powers.cBetah2 * beta_powers.sBetah6,
        2.0 * sqrt14 * beta_powers.cBetah3 * beta_powers.sBetah5,
        sqrt70 * beta_powers.cBetah4 * beta_powers.sBetah4,
        2.0 * sqrt14 * beta_powers.cBetah5 * beta_powers.sBetah3,
        2.0 * sqrt7 * beta_powers.cBetah6 * beta_powers.sBetah2,
        2.0 * sqrt2 * beta_powers.cBetah7 * beta_powers.sBetah,
        beta_powers.cBetah8
    ])

    # Exploit symmetry d^4_{-m,-4} = (-1)^m d^4_{m,4}. See eq. A2 of Precessing paper.
    # d^4_{-4,-4}, d^4_{-3,-4}, d^4_{-2,-4}, d^4_{-1,-4}, d^4_{0,-4}, d^4_{1,-4}, d^4_{2,-4}, d^4_{3,-4}, d^4_{4,-4}
    d4m4 = jnp.array([d44[8], -d44[7], d44[6], -d44[5], d44[4], -d44[3], d44[2], -d44[1], d44[0]])

    for m in range(-4, 4+1):
        # Transfer functions, see eqs. 3.5-3.7 in Precessing paper.
        A4m4emm = cexp_im_alpha_l4[-m+4] * d4m4[m+4] * Y4mA[m+4]
        A44emmstar = cexp_im_alpha_l4[m+4] * d44[m+4] * jnp.conj(Y4mA[m+4])
        hp_sum += (A4m4emm + A44emmstar)
        hc_sum += 1j * (A4m4emm - A44emmstar)

    return hp_sum, hc_sum


def twist_43(cexp_i_alpha, theta_JN, beta_powers):
    """
    Compute the twisting contributions for l=4, m'=3 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (4,3) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum J and line of sight N
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    hp_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)
    hc_sum = jnp.zeros_like(cexp_i_alpha, dtype=cexp_i_alpha.dtype)

    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_4i_alpha = cexp_i_alpha * cexp_3i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha
    cexp_m4i_alpha = cexp_mi_alpha * cexp_m3i_alpha

    cexp_im_alpha_l4 = jnp.stack([cexp_m4i_alpha, cexp_m3i_alpha, cexp_m2i_alpha, cexp_mi_alpha, jnp.ones_like(cexp_i_alpha), cexp_i_alpha, cexp_2i_alpha, cexp_3i_alpha, cexp_4i_alpha], axis=0)

    # Compute Y4m spherical harmonics directly
    Y4m4 = compute_sminus2_l4(theta=theta_JN, m=-4)
    Y4m3 = compute_sminus2_l4(theta=theta_JN, m=-3)
    Y4m2 = compute_sminus2_l4(theta=theta_JN, m=-2)
    Y4m1 = compute_sminus2_l4(theta=theta_JN, m=-1)
    Y40 = compute_sminus2_l4(theta=theta_JN, m=0)
    Y41 = compute_sminus2_l4(theta=theta_JN, m=1)
    Y42 = compute_sminus2_l4(theta=theta_JN, m=2)
    Y43 = compute_sminus2_l4(theta=theta_JN, m=3)
    Y44 = compute_sminus2_l4(theta=theta_JN, m=4)
    Y4mA = jnp.array([Y4m4, Y4m3, Y4m2, Y4m1, Y40, Y41, Y42, Y43, Y44])

    # Wigner-d coefficients for m'=3
    # d^4_{-4,3}, d^4_{-3,3}, d^4_{-2,3}, d^4_{-1,3}, d^4_{0,3}, d^4_{1,3}, d^4_{2,3}, d^4_{3,3}, d^4_{4,3}
    sqrt2 = jnp.sqrt(2.0)
    sqrt7 = jnp.sqrt(7.0)
    sqrt14 = jnp.sqrt(14.0)
    sqrt35_over_2 = 5.916079783099616  # 2*sqrt(35/4) = sqrt(35)

    cBetah = beta_powers.cBetah
    cBetah2 = beta_powers.cBetah2
    cBetah3 = beta_powers.cBetah3
    cBetah4 = beta_powers.cBetah4
    cBetah5 = beta_powers.cBetah5
    cBetah6 = beta_powers.cBetah6
    cBetah7 = beta_powers.cBetah7
    cBetah8 = beta_powers.cBetah8
    sBetah = beta_powers.sBetah
    sBetah2 = beta_powers.sBetah2
    sBetah3 = beta_powers.sBetah3
    sBetah4 = beta_powers.sBetah4
    sBetah5 = beta_powers.sBetah5
    sBetah6 = beta_powers.sBetah6
    sBetah7 = beta_powers.sBetah7
    sBetah8 = beta_powers.sBetah8

    d43 = jnp.array([
        2.0 * sqrt2 * cBetah * sBetah7,
        7.0 * cBetah2 * sBetah6 - sBetah8,
        sqrt14 * (3.0 * cBetah3 * sBetah5 - cBetah * sBetah7),
        sqrt7 * (5.0 * cBetah4 * sBetah4 - 3.0 * cBetah2 * sBetah6),
        2.0 * sqrt35_over_2 * (cBetah5 * sBetah3 - cBetah3 * sBetah5),
        sqrt7 * (3.0 * cBetah6 * sBetah2 - 5.0 * cBetah4 * sBetah4),
        sqrt14 * (cBetah7 * sBetah - 3.0 * cBetah5 * sBetah3),
        cBetah8 - 7.0 * cBetah6 * sBetah2,
        -2.0 * sqrt2 * cBetah7 * sBetah
    ])

    # Exploit symmetry d^4_{-m,-3} = -(-1)^m d^4_{m,3}. See eq. A2 of Precessing paper.
    # d^4_{-4,-3}, d^4_{-3,-3}, d^4_{-2,-3}, d^4_{-1,-3}, d^4_{0,-3}, d^4_{1,-3}, d^4_{2,-3}, d^4_{3,-3}, d^4_{4,-3}
    d4m3 = jnp.array([-d43[8], d43[7], -d43[6], d43[5], -d43[4], d43[3], -d43[2], d43[1], -d43[0]])

    for m in range(-4, 4+1):
        # Transfer functions, see eqs. 3.5-3.7 in Precessing paper.
        A4m3emm = cexp_im_alpha_l4[-m+4] * d4m3[m+4] * Y4mA[m+4]
        A43emmstar = cexp_im_alpha_l4[m+4] * d43[m+4] * jnp.conj(Y4mA[m+4])
        hp_sum += (A4m3emm + A43emmstar)
        hc_sum += 1j * (A4m3emm - A43emmstar)

    return hp_sum, hc_sum

@jit
def apply_polarization_rotation(zeta_polarization, _hp, _hc):
    """Apply polarization rotation to waveform components.
    
    Parameters
    ----------
    zeta_polarization : float
        Polarization angle.
    _hp : array_like
        Plus polarization component (unrotated).
    _hc : array_like
        Cross polarization component (unrotated).
    
    Returns
    -------
    hp : array_like
        Rotated plus polarization.
    hc : array_like
        Rotated cross polarization.
    """
    cosPolFac = jnp.cos(2.0 * zeta_polarization)
    sinPolFac = jnp.sin(2.0 * zeta_polarization)
    
    hp = cosPolFac * _hp + sinPolFac * _hc
    hc = cosPolFac * _hc - sinPolFac * _hp
    
    return hp, hc



@jit
def IMRPhenomXWignerdCoefficients_cosbeta(cos_beta):
    """
    Compute cos(beta/2) and sin(beta/2) from cos(beta).
    
    Uses half-angle formulas:
    - cos(beta/2) = sqrt((1 + cos(beta)) / 2)
    - sin(beta/2) = sqrt((1 - cos(beta)) / 2)
    
    Parameters
    ----------
    cos_beta : float or array
        cos(beta)
    
    Returns
    -------
    cos_beta_half : float or array
        cos(beta/2), always non-negative
    sin_beta_half : float or array
        sin(beta/2), always non-negative
    """
    # Note that the results here are indeed always non-negative
    cos_beta_half = jnp.sqrt(jnp.abs(1.0 + cos_beta) / 2.0)  # cos(beta/2)
    sin_beta_half = jnp.sqrt(jnp.abs(1.0 - cos_beta) / 2.0)  # sin(beta/2)
    
    return cos_beta_half, sin_beta_half



@jit
def component_masses_to_chirp_mass(mass_1, mass_2):
    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2


@jit
def XLALSimIMRPhenomXUtilsHztoMf(fHz: float, Mtot_Msun: float) -> float:
    """
    Convert frequency from Hz to geometric units (Mf).

    Parameters
    ----------
    fHz : float
        Frequency in Hz
    Mtot_Msun : float
        Total mass in solar masses

    Returns
    -------
    float
        Geometric frequency Mf
    """
    # Mtot in seconds = Mtot_Msun * MTSUN_SI
    return fHz * Mtot_Msun * MTSUN_SI


@jit
def XLALSimIMRPhenomXUtilsMftoHz(Mf: float, Mtot_Msun: float) -> float:
    """
    Convert frequency from geometric units (Mf) to Hz.

    This function converts dimensionless geometric frequency Mf to physical
    frequency in Hz using the total mass of the binary system.

    Parameters
    ----------
    Mf : float
        Dimensionless geometric frequency (Mf = f * M * G / c^3)
    Mtot_Msun : float
        Total mass of the binary system in solar masses

    Returns
    -------
    float
        Frequency in Hz

    Notes
    -----
    The conversion formula is:
        f_Hz = Mf / (Mtot_Msun * MTSUN_SI)

    where MTSUN_SI is the solar mass expressed in seconds (~4.925e-06 s).
    """
    # Mtot in seconds = Mtot_Msun * MTSUN_SI
    return Mf / (Mtot_Msun * MTSUN_SI)

@jit
def chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio):

    total_mass = chirp_mass_and_mass_ratio_to_total_mass(chirp_mass=chirp_mass,
                                                         mass_ratio=mass_ratio)
    mass_1, mass_2 = (
        total_mass_and_mass_ratio_to_component_masses(
            total_mass=total_mass, mass_ratio=mass_ratio)
    )
    return mass_1, mass_2

@jit
def chirp_mass_and_mass_ratio_to_total_mass(chirp_mass, mass_ratio):
    """
    Convert chirp mass and mass ratio of a binary to its total mass.

    Parameters
    ==========
    chirp_mass: float
        Chirp mass of the binary
    mass_ratio: float
        Mass ratio (mass_2/mass_1) of the binary

    Returns
    =======
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object
    """


    return chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio ** 0.6

@jit
def total_mass_and_mass_ratio_to_component_masses(mass_ratio, total_mass):
    """
    Convert total mass and mass ratio of a binary to its component masses.

    Parameters
    ==========
    mass_ratio: float
        Mass ratio (mass_2/mass_1) of the binary
    total_mass: float
        Total mass of the binary

    Returns
    =======
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object
    """

    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return mass_1, mass_2

@jit
def symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio):
    """
    Convert the symmetric mass ratio to the normal mass ratio.

    Parameters
    ==========
    symmetric_mass_ratio: float
        Symmetric mass ratio of the binary

    Returns
    =======
    mass_ratio: float
        Mass ratio of the binary
    """

    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return temp - (temp ** 2 - 1) ** 0.5





def Get_alpha_epsilon_offset(
    mprime: int,                      # Second index of the non-precessing mode (l, mprime)
    alpha_offset_1,
    epsilon_offset_1,
    alpha_offset,
    epsilon_offset,
    alpha_offset_3,
    epsilon_offset_3,
    alpha_offset_4,
    epsilon_offset_4
):
    """
    Get offset alpha and epsilon angles at reference frequency.
    The angles are evaluated at frequency 2*pi*MfRef/mprime so the offset depends on mprime.

    Returns:
        alpha_offset_mprime: Offset alpha angle at reference frequency
        epsilon_offset_mprime: Offset epsilon angle at reference frequency
    """

    # Use jax.lax.switch for the case statement
    def case_1():
        return alpha_offset_1, epsilon_offset_1

    def case_2():
        return alpha_offset, epsilon_offset  # Already used in XP code, no _2 suffix

    def case_3():
        return alpha_offset_3, epsilon_offset_3

    def case_4():
        return alpha_offset_4, epsilon_offset_4

    # Use jax.lax.switch with mprime-1 as index (since switch uses 0-based indexing)
    alpha_offset_mprime, epsilon_offset_mprime = jax.lax.switch(
        mprime - 1,
        [case_1, case_2, case_3, case_4]
    )

    return alpha_offset_mprime, epsilon_offset_mprime
