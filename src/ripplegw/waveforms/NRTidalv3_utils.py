"""by Robin Chan"""

import jax
import jax.numpy as jnp
from ..constants import gt, m_per_Mpc, PI, TWO_PI, MRSUN, C
from ..typing import Array
from ripplegw import Mc_eta_to_ms, lambda_tildes_to_lambdas
from .IMRPhenom_tidal_utils import get_quadparam_octparam, get_kappa
from .IMRPhenomD_NRTidalv2 import get_tidal_amplitude


"""
Corrections from NRTidalv3: (2311.07456)
---------------------------

- Phase
- Amplitude -> same as NRTidalv2, reuse
- Spin effects
- Spin precession effects

Uses merger frequency as stopping point -> tapering
|_> For (much) later: look into RLO frequency
"""


"""
Compute the merger frequency of a BNS system for NRTidalv3 (https://arxiv.org/pdf/2311.07456.pdf).
This uses a new fit from Gonzalez, et. al (2022) Eq. (23) of https://arxiv.org/abs/2210.16366.
"""
def _get_merger_frequency(theta: Array):
    """
    Computes the merger frequency in Hz of the given system. This is defined in equation (41) in 2311.07456 and the lal source code.
    Pretty much literally copied from LAL at the moment (XLALSimNRTunedTidesMergerFrequency_v3)

    Args:
        theta (Array): Intrinsic parameters with order (m1, m2, chi1, chi2, lambda1, lambda2)
        kappa (float, optional): Tidal parameter kappa. Defaults to None, so that it is computed from the given parameters theta.

    Returns:
        float: The merger frequency in Hz.
    """
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    M = m1 + m2
    m1_s = m1 * gt
    m2_s = m2 * gt
    q = m1_s / m2_s

    Xa = q/(1.0+q)
    Xb = 1.0 - Xa

    nu = Xa*Xb

    Xa_2 = Xa*Xa
    Xa_3 = Xa_2*Xa
    Xb_2 = Xb*Xb
    Xb_3 = Xb_2*Xb

    kappa2eff = 3.0*nu*(Xa_3*lambda1 + Xb_3*lambda2)

    a_0 = 0.22

    a_1M = 0.80
    a_1S = 0.25
    b_1S = -1.99 # This was not written in 2311.07456, but can be found in Table 2 of 2210.16366.

    a_1T = 0.0485
    a_2T = 0.00000586
    a_3T = 0.10
    a_4T = 0.000186

    b_1T = 1.80
    b_2T = 599.99
    b_3T = 7.80
    b_4T = 84.76

    Xval = 1.0 - 4.0*nu

    p_1S = a_1S*(1.0 + b_1S*Xval)
    p_1T = a_1T*(1.0 + b_1T*Xval)
    p_2T = a_2T*(1.0 + b_2T*Xval)
    p_3T = a_3T*(1.0 + b_3T*Xval)
    p_4T = a_4T*(1.0 + b_4T*Xval)

    kappa2eff2 = kappa2eff*kappa2eff

    Sval = Xa_2*chi1 + Xb_2*chi2

    QM = 1.0 + a_1M*Xval
    QS = 1.0 + p_1S*Sval

    num = 1.0 + p_1T*kappa2eff + p_2T*kappa2eff2
    den = 1.0 + p_3T*kappa2eff + p_4T*kappa2eff2
    QT = num / den

    # dimensionless angular frequency of merger
    Qfit = a_0*QM*QS*QT

    Momega_merger = nu*Qfit*TWO_PI


    # convert from angular frequency to frequency (divide by 2*pi) and then convert from dimensionless frequency to Hz (divide by mtot * LAL_MTSUN_SI)
    fHz_merger = Momega_merger / TWO_PI / (M * MRSUN / C)

    return fHz_merger


#####################################################################################################################################
###################################################    TIDAL PHASE CORRECTIONS    ###################################################
#####################################################################################################################################


def general_planck_taper(x, y1, y2):
    return jnp.where(
        x < y1,
        0,
        jnp.where(
            x > y2,
            1,
            1.0 / (jnp.exp((y2 - y1) / (x - y1) + (y2 - y1) / (x - y2)) + 1.0),
        ),
    )


"""
Tidal phase correction for NRTidalv3, Eq. (27,30), from Abac, et. al. (2023) (https://arxiv.org/pdf/2311.07456.pdf)
and is a function of x = angular_orb_freq^(2./3.)
"""
def get_tidal_phase(
    M_omega: Array, # Dimensionless angular GW frequency
    NRTidalv3_coeffs: Array, # NRTidalv3 coefficients
    PN_coeffs: Array # 7.5 PN coefficients to be used as constraints
):
    """
    SimNRTunedTidesFDTidalPhase_v3
    """
  
    s1 = NRTidalv3_coeffs[0]
    s2 = NRTidalv3_coeffs[1]

    exps2s3 = NRTidalv3_coeffs[3]

    s2Mom = -s2*M_omega*2.0
    exps2Mom = jnp.cosh(s2Mom) + jnp.sinh(s2Mom)

    # expression for the effective love number enhancement factor, see Eq. (27) of https://arxiv.org/pdf/2311.07456.pdf.*/
    dynk2bar = 1.0 + ((s1) - 1)*(1.0/(1.0 + exps2Mom*exps2s3)) - ((s1-1.0)/(1.0 + exps2s3)) - 2.0*M_omega*((s1) - 1)*s2*exps2s3/((1.0 + exps2s3)*(1.0 + exps2s3))
    
    PN_x = M_omega**(2.0/3.0) # redundant computation?
    PN_x_2 = PN_x * PN_x
    PN_x_3 = PN_x * PN_x_2
    PN_x_3over2 = PN_x * jnp.sqrt(PN_x)
    PN_x_5over2 = PN_x_3over2 * PN_x

    kappaA = NRTidalv3_coeffs[4]
    kappaB = NRTidalv3_coeffs[5]

    dynkappaA = kappaA*dynk2bar
    dynkappaB = kappaB*dynk2bar

    # Pade Coefficients
    n_5over2A = NRTidalv3_coeffs[6]
    n_3A = NRTidalv3_coeffs[7]
    d_1A = NRTidalv3_coeffs[8]

    n_5over2B = NRTidalv3_coeffs[9]
    n_3B = NRTidalv3_coeffs[10]
    d_1B = NRTidalv3_coeffs[11]

    # 7.5PN Coefficients
    c_NewtA = PN_coeffs[0]
    c_NewtB = PN_coeffs[5]

    # Pade Coefficients constrained with PN 
    n_1A = NRTidalv3_coeffs[12]
    n_3over2A = NRTidalv3_coeffs[13]
    n_2A = NRTidalv3_coeffs[14]
    d_3over2A = NRTidalv3_coeffs[15]

    n_1B = NRTidalv3_coeffs[16]
    n_3over2B = NRTidalv3_coeffs[17]
    n_2B = NRTidalv3_coeffs[18]
    d_3over2B = NRTidalv3_coeffs[19]

    factorA = -c_NewtA*PN_x_5over2*dynkappaA
    factorB = -c_NewtB*PN_x_5over2*dynkappaB

    # Pade approximant, see Eq. (32) of https://arxiv.org/pdf/2311.07456.pdf. */
    numA = 1.0 + (n_1A*PN_x) + (n_3over2A*PN_x_3over2) + (n_2A*PN_x_2) + (n_5over2A*PN_x_5over2) + (n_3A*PN_x_3)
    denA = 1.0 + (d_1A*PN_x) + (d_3over2A*PN_x_3over2) # + (d_2A*PN_x_2)

    numB = 1.0 + (n_1B*PN_x) + (n_3over2B*PN_x_3over2) + (n_2B*PN_x_2) + (n_5over2B*PN_x_5over2) + (n_3B*PN_x_3)
    denB = 1.0 + (d_1B*PN_x) + (d_3over2B*PN_x_3over2) # + (d_2B*PN_x_2)

    ratioA = numA/denA
    ratioB = numB/denB

    tidal_phaseA = factorA*ratioA
    tidal_phaseB = factorB*ratioB

    tidal_phase = tidal_phaseA + tidal_phaseB

    return tidal_phase


def get_tidalphasePN_coeffs(theta_intrinsic: Array):
    """
    Coefficients or the PN tidal phase correction, at 7.5PN, to connect with NRTidalv3 Phase post-merger, see Eq. (45) of https://arxiv.org/pdf/2311.07456.pdf
    XLALSimNRTunedTidesSetFDTidalPhase_PN_Coeffs
    """

    # PN_coeffs = jnp.zeros(10)

    m1, m2, _, _, _, _ = theta_intrinsic
    M = m1 + m2
    Xa = m1 / M
    Xb = m2 / M

    # Powers of Xa and Xb
    Xa_2 = Xa*Xa
    Xa_3 = Xa_2*Xa
    Xa_4 = Xa_3*Xa
    Xa_5 = Xa_4*Xa

    Xb_2 = Xb*Xb
    Xb_3 = Xb_2*Xb
    Xb_4 = Xb_3*Xb
    Xb_5 = Xb_4*Xb

    den_a = 11.0*Xa-12.0
    den_b = 11.0*Xb-12.0

    # 7.5PN Coefficients
    PN_coeffs0 = -3.0*den_a/(16*Xa*Xb_2) #(3.0)*(12.0 -11.0*Xa)/(16*Xa*Xb_2)
    PN_coeffs1 = (-1300.0*Xa_3 + 11430.0*Xa_2 + 4595.0*Xa -15895.0)/(672.0*den_a) #-5.0*(260.0*Xa_3 - 2286.0*Xa_2 - 919.0*Xa + 3179.0)/(672.0*(11.0*Xa-12.0))
    PN_coeffs2 = -PI
    PN_coeffs3 = (22861440.0*Xa_5 - 102135600.0*Xa_4 + 791891100.0*Xa_3 + 874828080.0*Xa_2 + 216234195.0*Xa -1939869350.0)/(27433728.0*den_a)# (5.0*(4572288.0*Xa_5 - 20427120.0*Xa_4 + 158378220.0*Xa_3 +174965616.0*Xa_2 + 43246839.0*Xa -387973870.0))/(27433728.0*(11.0*Xa - 12.0))
    PN_coeffs4 = -PI*(10520.0*Xa_3 -7598.0*Xa_2 +22415.0*Xa - 27719.0)/(672.0*den_a)

    PN_coeffs5 = -3.0*den_b/(16*Xb*Xa_2) #(3.0)*(12.0 -11.0*Xb)/(16*Xb*Xa_2)
    PN_coeffs6 = (-1300.0*Xb_3 + 11430.0*Xb_2 + 4595.0*Xb -15895.0)/(672.0*den_b) #-5.0*(260.0*Xb_3 - 2286.0*Xb_2 - 919.0*Xb + 3179.0)/(672.0*(11.0*Xb-12.0))
    PN_coeffs7 = -PI
    PN_coeffs8 = (22861440.0*Xb_5 - 102135600.0*Xb_4 + 791891100.0*Xb_3 + 874828080.0*Xb_2 + 216234195.0*Xb -1939869350.0)/(27433728.0*den_b) #(5.0*(4572288.0*Xb_5 - 20427120.0*Xb_4 + 158378220.0*Xb_3 +174965616.0*Xb_2 + 43246839.0*Xb -387973870.0))/(27433728.0*(11.0*Xb - 12.0))
    PN_coeffs9 = -PI*(10520.0*Xb_3 -7598.0*Xb_2 +22415.0*Xb - 27719.0)/(672.0*den_b)


    PN_coeffs = jnp.array([PN_coeffs0, PN_coeffs1, PN_coeffs2, PN_coeffs3, PN_coeffs4,
                       PN_coeffs5, PN_coeffs6, PN_coeffs7, PN_coeffs8, PN_coeffs9])

    return PN_coeffs


def get_NRTidalv3_coefficients(
    theta_intrinsic: Array,
    PN_coeffs, # 7.5 PN coefficients to be used for constraints
):
    """
    Set the NRTidalv3 effective love number and phase coefficients in an array for use here and in the IMRPhenomX*_NRTidalv3 implementation
    XLALSimNRTunedTidesSetFDTidalPhase_v3_Coeffs
    """

    NRTidalv3_coeffs = jnp.zeros(20)

    m1, m2, _, _, lambda1, lambda2 = theta_intrinsic
    M = m1 + m2
    Xa = m1 / M
    Xb = m2 / M
    q = Xa/Xb

    # Coefficients for the effective enhancement factor:
    s10 =   1.273000423 #s10
    s11 =   3.64169971e-03 #s11
    s12 =   1.76144380e-03 #s12

    s20 =   2.78793291e+01 #s20
    s21 =   1.18175396e-02 #s21
    s22 =   -5.39996790e-03 #s22

    s30 =   1.42449682e-01 #s30
    s31 =   -1.70505852e-05 #s31
    s32 =   3.38040594e-05 #s32

    kappa2T = get_kappa(theta_intrinsic) # WHY DO THE PAPERS HAVE A PREFACTOR OF 2/13 BUT CODE (BOTH RIPPLE AND LAL) 3/13????

    NRTidalv3_coeffs = NRTidalv3_coeffs.at[0].set(s10 + s11*kappa2T + s12*q*kappa2T)
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[1].set(s20 + s21*kappa2T + s22*q*kappa2T)
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[2].set(s30 + s31*kappa2T + s32*q*kappa2T)

    s2s3 = NRTidalv3_coeffs[1]*NRTidalv3_coeffs[2]
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[3].set(jnp.cosh(s2s3) + jnp.sinh(s2s3))

    NRTidalv3_coeffs = NRTidalv3_coeffs.at[4].set(3.0*Xb*Xa*Xa*Xa*Xa*lambda1)  # kappaA
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[5].set(3.0*Xa*Xb*Xb*Xb*Xb*lambda2)  # kappaB

    # Exponent parameters:
    alpha =   -8.08155404e-03 #alpha
    beta =  -1.13695919e+00 #beta

    kappaA_alpha = jnp.pow(NRTidalv3_coeffs[4] + 1, alpha) #kappaA_alpha
    kappaB_alpha = jnp.pow(NRTidalv3_coeffs[5] + 1, alpha) #kappaB_alpha

    Xa_beta = jnp.pow(Xa, beta) #Xa_beta
    Xb_beta = jnp.pow(Xb, beta) #Xb_beta

    # Pade approximant coefficients:
    n_5over20 =  -9.40654388e+02 #n_5over20
    n_5over21 =  6.26517157e+02 #n_5over21
    n_5over22 =  5.53629706e+02 #n_5over22
    n_5over23 =  8.84823087e+01 #n_5over23

    n_30 =  4.05483848e+02 #n_30
    n_31 =  -4.25525054e+02 #n_31
    n_32 = -1.92004957e+02 #n_32
    n_33 =  -5.10967553e+01 #n_33

    d_10 =  3.80343306e+00 #d_10
    d_11 =  -2.52026996e+01 #d_11
    d_12 =  -3.08054443e+00 #d_12

    NRTidalv3_coeffs = NRTidalv3_coeffs.at[6].set(n_5over20 + n_5over21*Xa + n_5over22*kappaA_alpha + n_5over23*Xa_beta)
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[7].set(n_30 + n_31*Xa + n_32*kappaA_alpha + n_33*Xa_beta)
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[8].set(d_10 + d_11*Xa + d_12*Xa_beta)

    NRTidalv3_coeffs = NRTidalv3_coeffs.at[9].set(n_5over20 + n_5over21*Xb + n_5over22*kappaB_alpha + n_5over23*Xb_beta)
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[10].set(n_30 + n_31*Xb + n_32*kappaB_alpha + n_33*Xb_beta)
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[11].set(d_10 + d_11*Xb + d_12*Xb_beta)

    # 7.5PN Coefficients 
    c_1A, c_3over2A, c_2A, c_5over2A = PN_coeffs[1:5]
    c_1B, c_3over2B, c_2B, c_5over2B = PN_coeffs[6:10]

    inv_c1_A = 1.0 / c_1A
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[12].set(c_1A + NRTidalv3_coeffs[8])
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[13].set(((c_1A*c_3over2A) - c_5over2A - (c_3over2A)*NRTidalv3_coeffs[8] + NRTidalv3_coeffs[6]) * inv_c1_A)
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[14].set(c_2A + c_1A*NRTidalv3_coeffs[8])
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[15].set(-(c_5over2A + c_3over2A*NRTidalv3_coeffs[8] - NRTidalv3_coeffs[6]) * inv_c1_A)

    inv_c1_B = 1.0 / c_1B
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[16].set(c_1B + NRTidalv3_coeffs[11])
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[17].set(((c_1B*c_3over2B) - c_5over2B - (c_3over2B)*NRTidalv3_coeffs[11] + NRTidalv3_coeffs[9]) * inv_c1_B)
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[18].set(c_2B + c_1B*NRTidalv3_coeffs[11])
    NRTidalv3_coeffs = NRTidalv3_coeffs.at[19].set(-(c_5over2B + c_3over2B*NRTidalv3_coeffs[11] - NRTidalv3_coeffs[9]) * inv_c1_B)

    return NRTidalv3_coeffs


"""
PN tidal phase correction, at 7.5PN, to connect with NRTidalv3 Phase post-merger,
see Eq. (22) and (45) of https://arxiv.org/pdf/2311.07456.pdf
and is a function of x = angular_orb_freq^(2./3.)
"""
def get_tidal_phase_PN(
               M_omega, # Dimensionless angular GW frequency
               Xa, #< Mass of companion 1 divided by total mass 
               lambda1, #< dimensionless tidal deformability of companion 1
               lambda2, #< dimensionless tidal deformability of companion 2
               PN_coeffs #< 7.5 PN coefficients
):
    """
    SimNRTunedTidesFDTidalPhase_PN
    """
    Xb = 1.0 - Xa

    PN_x = M_omega **(2.0/3.0)              # pow(M_omega, 2.0/3.0)
    PN_x_2 = PN_x * PN_x                         # pow(PN_x, 2)
    PN_x_3over2 = PN_x * jnp.sqrt(PN_x)              # pow(PN_x, 3.0/2.0)
    PN_x_5over2 = PN_x_3over2 * PN_x      # pow(PN_x, 5.0/2.0)

    kappaA = 3.0*Xb*Xa*Xa*Xa*Xa*lambda1
    kappaB = 3.0*Xa*Xb*Xb*Xb*Xb*lambda2

    # 7.5PN Coefficients 
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

    factorA = -c_NewtA*PN_x_5over2*kappaA
    factorB = -c_NewtB*PN_x_5over2*kappaB

    tidal_phasePNA = factorA*(1.0 + (c_1A*PN_x) + (c_3over2A*PN_x_3over2) + (c_2A*PN_x_2) + (c_5over2A*PN_x_5over2))
    tidal_phasePNB = factorB*(1.0 + (c_1B*PN_x) + (c_3over2B*PN_x_3over2) + (c_2B*PN_x_2) + (c_5over2B*PN_x_5over2))

    tidal_phasePN = tidal_phasePNA + tidal_phasePNB

    return tidal_phasePN


######################################################################################################################################
#########################################################    SPIN EFFECTS    #########################################################
######################################################################################################################################


def get_spin_phase_correction_v3(x: Array, theta: Array) -> Array:
    """
    Get the higher order spin corrections with updated unversal relation, as detailed in Section III A of 2311.07456

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)

    Returns:
        Array: Higher order spin corrections to the phase
    """

    # Compute auxiliary quantities
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
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

    # Get phase contributions
    SS_2 = -50.0 * quadparam1 * chi1_sq * X1sq - 50.0 * quadparam2 * chi2_sq * X2sq

    SS_3 = (
        5.0
        / 84.0
        * (9407.0 + 8218.0 * X1 - 2016.0 * X1sq)
        * quadparam1
        * X1sq
        * chi1_sq
        + 5.0
        / 84.0
        * (9407.0 + 8218.0 * X2 - 2016.0 * X2sq)
        * quadparam2
        * X2sq
        * chi2_sq
    )

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
    psi_SS = prefac * (
        SS_2 * x ** (-1.0 / 2.0) + SS_3 * x ** (1.0 / 2.0) + (SS_3p5 + SSS_3p5) * x
    )

    return psi_SS
        

#######################################################################################################################################
####################################################    SPIN PRECESSION EFFECTS    ####################################################
#######################################################################################################################################


# TODO, better to implement straight into XP(HM)?

