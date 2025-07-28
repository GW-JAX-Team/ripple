import jax
import jax.numpy as jnp
from ripple import Mc_eta_to_ms

from typing import Tuple
from ..constants import gt, MSUN
import numpy as np
from .IMRPhenomD import Phase as PhDPhase
from .IMRPhenomD import Amp as PhDAmp
from .IMRPhenomD_utils import (
    get_coeffs,
    get_transition_frequencies,
    EradRational0815,
    FinalSpin0815_s,
)
from ..typing import Array
from .IMRPhenomD_QNMdata import QNMData_a, QNMData_fRD, QNMData_fdamp


# helper functions for LALtoPhenomP:
def ROTATEZ(angle, x, y, z):
    tmp_x = x * jnp.cos(angle) - y * jnp.sin(angle)
    tmp_y = x * jnp.sin(angle) + y * jnp.cos(angle)
    return tmp_x, tmp_y, z


def ROTATEY(angle, x, y, z):
    tmp_x = x * jnp.cos(angle) + z * jnp.sin(angle)
    tmp_z = -x * jnp.sin(angle) + z * jnp.cos(angle)
    return tmp_x, y, tmp_z


def FinalSpin0815(eta, chi1, chi2):
    Seta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1 * m1
    m2s = m2 * m2
    s = m1s * chi1 + m2s * chi2
    return FinalSpin0815_s(eta, s)


def convert_spins(
    m1: float,
    m2: float,
    f_ref: float,
    phiRef: float,
    incl: float,
    s1x: float,
    s1y: float,
    s1z: float,
    s2x: float,
    s2y: float,
    s2z: float,
) -> Tuple[float, float, float, float, float, float, float]:
    # m1 = m1_SI / MSUN  # Masses in solar masses
    # m2 = m2_SI / MSUN
    M = m1 + m2
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    eta = m1 * m2 / (M * M)  # Symmetric mass-ratio

    # From the components in the source frame, we can easily determine
    # chi1_l, chi2_l, chip and phi_aligned, which we need to return.
    # We also compute the spherical angles of J,
    # which we need to transform to the J frame

    # Aligned spins
    chi1_l = s1z  # Dimensionless aligned spin on BH 1
    chi2_l = s2z  # Dimensionless aligned spin on BH 2

    # Magnitude of the spin projections in the orbital plane
    S1_perp = m1_2 * jnp.sqrt(s1x**2 + s1y**2)
    S2_perp = m2_2 * jnp.sqrt(s2x**2 + s2y**2)

    A1 = 2 + (3 * m2) / (2 * m1)
    A2 = 2 + (3 * m1) / (2 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp
    num = jnp.maximum(ASp1, ASp2)
    den = A2 * m2_2  # warning: this assumes m2 > m1
    chip = num / den

    m_sec = M * gt
    piM = jnp.pi * m_sec
    v_ref = (piM * f_ref) ** (1 / 3)
    L0 = M * M * L2PNR(v_ref, eta)
    J0x_sf = m1_2 * s1x + m2_2 * s2x
    J0y_sf = m1_2 * s1y + m2_2 * s2y
    J0z_sf = L0 + m1_2 * s1z + m2_2 * s2z
    J0 = jnp.sqrt(J0x_sf * J0x_sf + J0y_sf * J0y_sf + J0z_sf * J0z_sf)

    thetaJ_sf = jnp.arccos(J0z_sf / J0)

    phiJ_sf = jnp.arctan2(J0y_sf, J0x_sf)

    phi_aligned = -phiJ_sf

    # First we determine kappa
    # in the source frame, the components of N are given in Eq (35c) of T1500606-v6
    Nx_sf = jnp.sin(incl) * jnp.cos(jnp.pi / 2.0 - phiRef)
    Ny_sf = jnp.sin(incl) * jnp.sin(jnp.pi / 2.0 - phiRef)
    Nz_sf = jnp.cos(incl)

    tmp_x = Nx_sf
    tmp_y = Ny_sf
    tmp_z = Nz_sf

    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)

    kappa = -jnp.arctan2(tmp_y, tmp_x)

    # Then we determine alpha0, by rotating LN
    tmp_x, tmp_y, tmp_z = 0, 0, 1
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    alpha0 = jnp.arctan2(tmp_y, tmp_x)

    # Finally we determine thetaJ, by rotating N
    tmp_x, tmp_y, tmp_z = Nx_sf, Ny_sf, Nz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)
    Nx_Jf, Nz_Jf = tmp_x, tmp_z
    thetaJN = jnp.arccos(Nz_Jf)

    # Finally, we need to redefine the polarizations:
    # PhenomP's polarizations are defined following Arun et al (arXiv:0810.5336)
    # i.e. projecting the metric onto the P,Q,N triad defined with P=NxJ/|NxJ| (see (2.6) in there).
    # By contrast, the triad X,Y,N used in LAL
    # ("waveframe" in the nomenclature of T1500606-v6)
    # is defined in e.g. eq (35) of this document
    # (via its components in the source frame; note we use the defautl Omega=Pi/2).
    # Both triads differ from each other by a rotation around N by an angle \zeta
    # and we need to rotate the polarizations accordingly by 2\zeta

    Xx_sf = -jnp.cos(incl) * jnp.sin(phiRef)
    Xy_sf = -jnp.cos(incl) * jnp.cos(phiRef)
    Xz_sf = jnp.sin(incl)
    tmp_x, tmp_y, tmp_z = Xx_sf, Xy_sf, Xz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    # Now the tmp_a are the components of X in the J frame
    # We need the polar angle of that vector in the P,Q basis of Arun et al
    # P = NxJ/|NxJ| and since we put N in the (pos x)z half plane of the J frame
    PArunx_Jf = 0.0
    PAruny_Jf = -1.0
    PArunz_Jf = 0.0

    # Q = NxP
    QArunx_Jf = Nz_Jf
    QAruny_Jf = 0.0
    QArunz_Jf = -Nx_Jf

    # Calculate the dot products XdotPArun and XdotQArun
    XdotPArun = tmp_x * PArunx_Jf + tmp_y * PAruny_Jf + tmp_z * PArunz_Jf
    XdotQArun = tmp_x * QArunx_Jf + tmp_y * QAruny_Jf + tmp_z * QArunz_Jf

    zeta_polariz = jnp.arctan2(XdotQArun, XdotPArun)
    return chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz


def SpinWeightedY(theta, phi, s, l, m):
    "copied from SphericalHarmonics.c in LAL"
    if s == -2:
        if l == 2:
            if m == -2:
                fac = (
                    jnp.sqrt(5.0 / (64.0 * jnp.pi))
                    * (1.0 - jnp.cos(theta))
                    * (1.0 - jnp.cos(theta))
                )
            elif m == -1:
                fac = (
                    jnp.sqrt(5.0 / (16.0 * jnp.pi))
                    * jnp.sin(theta)
                    * (1.0 - jnp.cos(theta))
                )
            elif m == 0:
                fac = jnp.sqrt(15.0 / (32.0 * jnp.pi)) * jnp.sin(theta) * jnp.sin(theta)
            elif m == 1:
                fac = (
                    jnp.sqrt(5.0 / (16.0 * jnp.pi))
                    * jnp.sin(theta)
                    * (1.0 + jnp.cos(theta))
                )
            elif m == 2:
                fac = (
                    jnp.sqrt(5.0 / (64.0 * jnp.pi))
                    * (1.0 + jnp.cos(theta))
                    * (1.0 + jnp.cos(theta))
                )
            else:
                raise ValueError(f"Invalid mode s={s}, l={l}, m={m} - require |m| <= l")
    return fac * np.exp(1j * m * phi)


def L2PNR(v: float, eta: float) -> float:
    eta2 = eta**2
    x = v**2
    x2 = x**2
    return (
        eta
        * (
            1.0
            + (1.5 + eta / 6.0) * x
            + (3.375 - (19.0 * eta) / 8.0 - eta2 / 24.0) * x2
        )
    ) / x**0.5


def WignerdCoefficients(v: float, SL: float, eta: float, Sp: float):
    # We define the shorthand s := Sp / (L + SL)
    #Sp: perpendicular, SL: parallel
    L = L2PNR(v, eta)    #### what is this function?
    s = Sp / (L + SL)
    s2 = s**2
    cos_beta = 1.0 / (1.0 + s2) ** 0.5
    cos_beta_half = ((1.0 + cos_beta) / 2.0) ** 0.5  # cos(beta/2)
    sin_beta_half = ((1.0 - cos_beta) / 2.0) ** 0.5  # sin(beta/2)

    return cos_beta_half, sin_beta_half


def ComputeNNLOanglecoeffs(q, chil, chip):
    ##### -> IMRPhenomXP.pdf for coefficients, same as IMRPhenomPV
    m2 = q / (1.0 + q)
    m1 = 1.0 / (1.0 + q)
    dm = m1 - m2
    mtot = 1.0
    eta = m1 * m2  # mtot = 1
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    mtot2 = mtot * mtot
    mtot4 = mtot2 * mtot2
    mtot6 = mtot4 * mtot2
    mtot8 = mtot6 * mtot2
    chil2 = chil * chil
    chip2 = chip * chip
    chip4 = chip2 * chip2
    dm2 = dm * dm
    dm3 = dm2 * dm
    m2_2 = m2 * m2
    m2_3 = m2_2 * m2
    m2_4 = m2_3 * m2
    m2_5 = m2_4 * m2
    m2_6 = m2_5 * m2
    m2_7 = m2_6 * m2
    m2_8 = m2_7 * m2

    angcoeffs = {}
    angcoeffs["alphacoeff1"] = -0.18229166666666666 - (5 * dm) / (64.0 * m2)

    angcoeffs["alphacoeff2"] = (-15 * dm * m2 * chil) / (128.0 * mtot2 * eta) - (
        35 * m2_2 * chil
    ) / (128.0 * mtot2 * eta)

    angcoeffs["alphacoeff3"] = (
        -1.7952473958333333
        - (4555 * dm) / (7168.0 * m2)
        - (15 * chip2 * dm * m2_3) / (128.0 * mtot4 * eta2)
        - (35 * chip2 * m2_4) / (128.0 * mtot4 * eta2)
        - (515 * eta) / 384.0
        - (15 * dm2 * eta) / (256.0 * m2_2)
        - (175 * dm * eta) / (256.0 * m2)
    )

    angcoeffs["alphacoeff4"] = (
        -(35 * jnp.pi) / 48.0
        - (5 * dm * jnp.pi) / (16.0 * m2)
        + (5 * dm2 * chil) / (16.0 * mtot2)
        + (5 * dm * m2 * chil) / (3.0 * mtot2)
        + (2545 * m2_2 * chil) / (1152.0 * mtot2)
        - (5 * chip2 * dm * m2_5 * chil) / (128.0 * mtot6 * eta3)
        - (35 * chip2 * m2_6 * chil) / (384.0 * mtot6 * eta3)
        + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
        + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
    )

    angcoeffs["alphacoeff5"] = (
        4.318908476114694
        + (27895885 * dm) / (2.1676032e7 * m2)
        - (15 * chip4 * dm * m2_7) / (512.0 * mtot8 * eta4)
        - (35 * chip4 * m2_8) / (512.0 * mtot8 * eta4)
        - (485 * chip2 * dm * m2_3) / (14336.0 * mtot4 * eta2)
        + (475 * chip2 * m2_4) / (6144.0 * mtot4 * eta2)
        + (15 * chip2 * dm2 * m2_2) / (256.0 * mtot4 * eta)
        + (145 * chip2 * dm * m2_3) / (512.0 * mtot4 * eta)
        + (575 * chip2 * m2_4) / (1536.0 * mtot4 * eta)
        + (39695 * eta) / 86016.0
        + (1615 * dm2 * eta) / (28672.0 * m2_2)
        - (265 * dm * eta) / (14336.0 * m2)
        + (955 * eta2) / 576.0
        + (15 * dm3 * eta2) / (1024.0 * m2_3)
        + (35 * dm2 * eta2) / (256.0 * m2_2)
        + (2725 * dm * eta2) / (3072.0 * m2)
        - (15 * dm * m2 * jnp.pi * chil) / (16.0 * mtot2 * eta)
        - (35 * m2_2 * jnp.pi * chil) / (16.0 * mtot2 * eta)
        + (15 * chip2 * dm * m2_7 * chil2) / (128.0 * mtot8 * eta4)
        + (35 * chip2 * m2_8 * chil2) / (128.0 * mtot8 * eta4)
        + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
        + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
        + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
    )

    angcoeffs["epsiloncoeff1"] = -0.18229166666666666 - (5 * dm) / (64.0 * m2)
    angcoeffs["epsiloncoeff2"] = (-15 * dm * m2 * chil) / (128.0 * mtot2 * eta) - (
        35 * m2_2 * chil
    ) / (128.0 * mtot2 * eta)
    angcoeffs["epsiloncoeff3"] = (
        -1.7952473958333333
        - (4555 * dm) / (7168.0 * m2)
        - (515 * eta) / 384.0
        - (15 * dm2 * eta) / (256.0 * m2_2)
        - (175 * dm * eta) / (256.0 * m2)
    )
    angcoeffs["epsiloncoeff4"] = (
        -(35 * jnp.pi) / 48.0
        - (5 * dm * jnp.pi) / (16.0 * m2)
        + (5 * dm2 * chil) / (16.0 * mtot2)
        + (5 * dm * m2 * chil) / (3.0 * mtot2)
        + (2545 * m2_2 * chil) / (1152.0 * mtot2)
        + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
        + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
    )
    angcoeffs["epsiloncoeff5"] = (
        4.318908476114694
        + (27895885 * dm) / (2.1676032e7 * m2)
        + (39695 * eta) / 86016.0
        + (1615 * dm2 * eta) / (28672.0 * m2_2)
        - (265 * dm * eta) / (14336.0 * m2)
        + (955 * eta2) / 576.0
        + (15 * dm3 * eta2) / (1024.0 * m2_3)
        + (35 * dm2 * eta2) / (256.0 * m2_2)
        + (2725 * dm * eta2) / (3072.0 * m2)
        - (15 * dm * m2 * jnp.pi * chil) / (16.0 * mtot2 * eta)
        - (35 * m2_2 * jnp.pi * chil) / (16.0 * mtot2 * eta)
        + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
        + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
        + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
    )
    return angcoeffs


def FinalSpin_inplane(m1, m2, chi1_l, chi2_l, chip):
    M = m1 + m2
    eta = m1 * m2 / (M * M)
    # Here I assume m1 > m2, the convention used in phenomD
    # (not the convention of internal phenomP)
    q_factor = m1 / M
    af_parallel = FinalSpin0815(eta, chi1_l, chi2_l)
    Sperp = chip * q_factor * q_factor
    af = jnp.copysign(1.0, af_parallel) * jnp.sqrt(
        Sperp * Sperp + af_parallel * af_parallel
    )
    return af


def phP_get_fRD_fdamp(m1, m2, chi1_l, chi2_l, chip):
    # m1 > m2 should hold here
    finspin = FinalSpin_inplane(m1, m2, chi1_l, chi2_l, chip)
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s**2.0)
    Erad = EradRational0815(eta_s, chi1_l, chi2_l)
    fRD = jnp.interp(finspin, QNMData_a, QNMData_fRD) / (1.0 - Erad)
    fdamp = jnp.interp(finspin, QNMData_a, QNMData_fdamp) / (1.0 - Erad)

    return fRD / M_s, fdamp / M_s


def phP_get_transition_frequencies(
    theta: jnp.array,
    gamma2: float,
    gamma3: float,
    chip: float,
) -> Tuple[float, float, float, float, float, float]:
    # m1 > m2 should hold here

    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    f_RD, f_damp = phP_get_fRD_fdamp(m1, m2, chi1, chi2, chip)

    # Phase transition frequencies
    f1 = 0.018 / (M * gt)
    f2 = 0.5 * f_RD

    # Amplitude transition frequencies
    f3 = 0.014 / (M * gt)
    f4_gammaneg_gtr_1 = lambda f_RD_, f_damp_, gamma3_, gamma2_: jnp.abs(
        f_RD_ + (-f_damp_ * gamma3_) / gamma2_
    )
    f4_gammaneg_less_1 = lambda f_RD_, f_damp_, gamma3_, gamma2_: jnp.abs(
        f_RD_ + (f_damp_ * (-1 + jnp.sqrt(1 - (gamma2_) ** 2.0)) * gamma3_) / gamma2_
    )
    f4 = jax.lax.cond(
        gamma2 >= 1,
        f4_gammaneg_gtr_1,
        f4_gammaneg_less_1,
        f_RD,
        f_damp,
        gamma3,
        gamma2,
    )
    return f1, f2, f3, f4, f_RD, f_damp

def IMRPhenomX_Return_phi_zeta_costhetaL_MSA(
    v,  ## velocity
    pWF,  ## IMRPhenomX waveform struct
    pPrec  ## IMRPhenomX precession struct  ### LR: casting??
    ):  ## has to output a jnp array
    
    vout = jnp.array([0,0,0])
    L_norm = pWF["eta"] / v
    
    J_norm = IMRPhenomX_JNorm_MSA(L_norm,pPrec)  
    
    ## J_norm = jax.lax.cond(pPrec["useMSA"], )
    
    L_norm3PN = IMRPhenomX_L_norm_3PN_of_v(v, v*v, L_norm, pPrec)   ## for 223
    
    J_norm3PN = IMRPhenomX_JNorm_MSA(L_norm3PN,pPrec)  
    
    vRoots    = IMRPhenomX_Return_Roots_MSA(L_norm,J_norm,pPrec)  ## return jnp.array
    
    pPrec["S32"]  = vRoots[0]
    pPrec["Smi2"] = vRoots[1]
    pPrec["Spl2"] = vRoots[2]
    
    pPrec["Spl2mSmi2"]   = pPrec["Spl2"] - pPrec["Smi2"]
    pPrec["Spl2pSmi2"]   = pPrec["Spl2"] + pPrec["Smi2"]
    pPrec["Spl"]         = jnp.sqrt(pPrec["Spl2"])
    pPrec["Smi"]           = jnp.sqrt(pPrec["Smi2"])
    
    SNorm = IMRPhenomX_Return_SNorm_MSA(v,pPrec)  
    pPrec["S_norm"]      = SNorm
    pPrec["S_norm_2"]    = SNorm * SNorm
    
    ''' Get phiz_0_MSA and zeta_0_MSA '''
    vMSA = jax.lax.cond((jnp.fabs(pPrec["Smi2"] - pPrec["Spl2"]) > 1.e-5), 
                        IMRPhenomX_Return_MSA_Corrections_MSA,  ## return 3D jnp.array
                        lambda v, L_norm, J_norm, pPrec: jnp.array([0,0,0]),  ## ugly but okay??
                        v, L_norm, J_norm, pPrec)   
    
    phiz_MSA     = vMSA[0]
    zeta_MSA     = vMSA[1]
 
    phiz         = IMRPhenomX_Return_phiz_MSA(v,J_norm,pPrec)  ## (DONE)
    zeta         = IMRPhenomX_Return_zeta_MSA(v,pPrec)  ## (DONE)
    cos_theta_L      = IMRPhenomX_costhetaLJ(L_norm3PN,J_norm3PN,SNorm)  ## (DONE)
 
    vout[0] = phiz + phiz_MSA
    vout[1] = zeta + zeta_MSA
    vout[2] = cos_theta_L
        
    return vout

def WignerdCoefficients_cosbeta(
    cos_beta        ## cos(beta) 
    ):
    ''' Note that the results here are indeed always non-negative '''
    cos_beta_half = + jnp.sqrt( jnp.fabs(1.0 + cos_beta) / 2.0 )
    sin_beta_half = + jnp.sqrt( jnp.fabs(1.0 - cos_beta) / 2.0 )
 
    return cos_beta_half, sin_beta_half

def IMRPhenomX_costhetaLJ(
    L_norm: float, 
    J_norm: float, 
    S_norm: float
    ) -> float:
    costhetaLJ = 0.5 * (J_norm**2 + L_norm**2 - S_norm**2) / L_norm * J_norm

    # Clamp the value to the interval [-1.0, 1.0]
    costhetaLJ = jnp.clip(costhetaLJ, -1.0, 1.0)

    return costhetaLJ

def IMRPhenomX_Return_zeta_MSA(
    v: float, 
    pPrec
    ) -> float:
    invv = 1.0 / v
    invv2 = invv * invv
    invv3 = invv * invv2
    v2 = v * v
    logv = jnp.log(v)

    # Compute zeta using precession coefficients
    zeta_out = pPrec["eta"] * (
        pPrec["Omegazeta0_coeff"] * invv3 +
        pPrec["Omegazeta1_coeff"] * invv2 +
        pPrec["Omegazeta2_coeff"] * invv +
        pPrec["Omegazeta3_coeff"] * logv +
        pPrec["Omegazeta4_coeff"] * v +
        pPrec["Omegazeta5_coeff"] * v2
    ) + pPrec["zeta_0"]

    # Replace NaNs with 0 using jnp.nan_to_num
    zeta_out = jnp.nan_to_num(zeta_out, nan=0.0)

    return zeta_out


def IMRPhenomX_Return_phiz_MSA(
    v: float, 
    JNorm: float, 
    pPrec
    ) -> float:
    
    invv = 1.0 / v
    invv2 = invv * invv
    LNewt = pPrec["eta"] / v

    c1 = pPrec["c1"]
    c12 = c1 * c1

    SAv2 = pPrec["SAv2"]
    SAv = pPrec["SAv"]
    invSAv = pPrec["invSAv"]
    invSAv2 = pPrec["invSAv2"]

    # These are log functions defined in Eq. D27 and D28 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    log1 = jnp.log(jnp.abs(c1 + JNorm * pPrec["eta"] + pPrec["eta"] * LNewt))
    log2 = jnp.log(jnp.abs(c1 + JNorm * SAv * v + SAv2 * v))

    # Eq. D22-D27 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    phiz_0_coeff = (JNorm * pPrec["inveta4"]) * (
        0.5 * c12 - (c1 * pPrec["eta2"] * invv) / 6.0 - (SAv2 * pPrec["eta2"]) / 3.0 - (pPrec["eta4"] * invv2) / 3.0
    ) - (0.5 * c1 * pPrec["inveta"]) * (
        c12 * pPrec["inveta4"] - SAv2 * pPrec["inveta2"]
    ) * log1

    phiz_1_coeff = (
        -0.5 * JNorm * pPrec["inveta2"] * (c1 + pPrec["eta"] * LNewt)
        + 0.5 * pPrec["inveta3"] * (c12 - pPrec["eta2"] * SAv2) * log1
    )

    phiz_2_coeff = -JNorm + SAv * log2 - c1 * log1 * pPrec["inveta"]

    phiz_3_coeff = JNorm * v - pPrec["eta"] * log1 + c1 * log2 * invSAv

    phiz_4_coeff = (
        0.5 * JNorm * invSAv2 * v * (c1 + v * SAv2)
        - 0.5 * invSAv2 * invSAv * (c12 - pPrec["eta2"] * SAv2) * log2
    )

    phiz_5_coeff = (
        -JNorm * v * (
            0.5 * c12 * invSAv2 * invSAv2
            - c1 * v * invSAv2 / 6.0
            - v * v / 3.0
            - pPrec["eta2"] * invSAv2 / 3.0
        )
        + 0.5 * c1 * invSAv2 * invSAv2 * invSAv * (c12 - pPrec["eta2"] * SAv2) * log2
    )

    # Eq. 66 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
 
    # \phi_{z,-1} = \sum^5_{n=0} <\Omega_z>^(n) \phi_z^(n) + \phi_{z,-1}^0
 
    # Note that the <\Omega_z>^(n) are given by pPrec->Omegazn_coeff's as in Eqs. D15-D20
    phiz_out = (
        phiz_0_coeff * pPrec["Omegaz0_coeff"]
        + phiz_1_coeff * pPrec["Omegaz1_coeff"]
        + phiz_2_coeff * pPrec["Omegaz2_coeff"]
        + phiz_3_coeff * pPrec["Omegaz3_coeff"]
        + phiz_4_coeff * pPrec["Omegaz4_coeff"]
        + phiz_5_coeff * pPrec["Omegaz5_coeff"]
        + pPrec["phiz_0"]
    )

    # Ensure no NaN (replace with 0.0 if NaN)
    phiz_out = jnp.nan_to_num(phiz_out, nan=0.0)

    return phiz_out


def IMRPhenomX_Return_MSA_Corrections_MSA(
    v, 
    LNorm, 
    JNorm, 
    pPrec
    ):
    
    v2 = v * v

    # Sets c0, c2 and c4 in pPrec as per Eq. B6-B8 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    c_vec = IMRPhenomX_Return_Constants_c_MSA(v, JNorm, pPrec)
    # Sets d0, d2 and d4 in pPrec as per Eq. B9-B11 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    d_vec = IMRPhenomX_Return_Constants_d_MSA(LNorm, JNorm, pPrec)  

    c0, c2, c4 = c_vec
    d0, d2, d4 = d_vec

    two_d0 = 2.0 * d0
    
    # Eq. B20 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    sd = jnp.sqrt(jnp.abs(d2 * d2 - 4.0 * d0 * d4))

    # Eq. F20-21 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    A_theta_L = 0.5 * ((JNorm / LNorm) + (LNorm / JNorm) - (pPrec["Spl2"] / (JNorm * LNorm)))
    B_theta_L = 0.5 * pPrec["Spl2mSmi2"] / (JNorm * LNorm)

    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd

    nc = nc_num / nc_denom
    nd = nc_denom / two_d0

    sqrt_nc = jnp.sqrt(jnp.abs(nc))
    sqrt_nd = jnp.sqrt(jnp.abs(nd))

    psi = IMRPhenomX_Return_Psi_MSA(v, v2, pPrec) + pPrec["psi0"]  
    psi_dot = IMRPhenomX_Return_Psi_dot_MSA(v, pPrec) 

    tan_psi = jnp.tan(psi)
    atan_psi = jnp.arctan(tan_psi)

    C1 = -0.5 * (c0 / d0 - 2.0 * (c0 + c2 + c4) / nc_num)
    C2num = (c0 * (-2.0 * d0 * d4 + d2 * d2 + d2 * d4) -
             c2 * d0 * (d2 + 2.0 * d4) +
             c4 * d0 * (two_d0 + d2))
    C2den = 2.0 * d0 * sd * (d0 + d2 + d4)
    C2 = C2num / C2den

    Cphi = C1 + C2
    Dphi = C1 - C2

    def compute_Cphi_term():
        
        return jnp.abs((
            (c4 * d0 * ((2 * d0 + d2) + sd) -
                c2 * d0 * ((d2 + 2.0 * d4) - sd) -
                c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - 
                sd))) / C2den) * (sqrt_nc / (nc - 1.0)) * (atan_psi - jnp.arctan(sqrt_nc * tan_psi))) / psi_dot
        
    def compute_Dphi_term():
            return jnp.abs((
                (-c4 * d0 * ((2 * d0 + d2) - sd) +
                 c2 * d0 * ((d2 + 2.0 * d4) + sd) -
                 c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd))) / C2den
            ) * (sqrt_nd / (nd - 1.0)) * (atan_psi - jnp.arctan(sqrt_nd * tan_psi))) / psi_dot

    phiz_0_MSA_Cphi_term = jnp.where(nc == 1.0, 0.0, compute_Cphi_term())
    phiz_0_MSA_Dphi_term = jnp.where(nd == 1.0, 0.0, compute_Dphi_term())

    vMSA_x = phiz_0_MSA_Cphi_term + phiz_0_MSA_Dphi_term

    #####  restart from here
    vMSA_y = A_theta_L * vMSA_x + 2.0 * B_theta_L * d0 * (
                phiz_0_MSA_Cphi_term / (sd - d2) - phiz_0_MSA_Dphi_term / (sd + d2))

    vMSA_x = jnp.where(jnp.isnan(vMSA_x), 0.0, vMSA_x)
    vMSA_y = jnp.where(jnp.isnan(vMSA_y), 0.0, vMSA_y)

    return jnp.array([vMSA_x, vMSA_y, 0.0])

def IMRPhenomX_JNorm_MSA(LNorm, pPrec):
    JNorm2 = (LNorm * LNorm + 2.0 * LNorm * pPrec['c1_over_eta'] + pPrec['SAv2'])
    return jnp.sqrt(JNorm2)

def IMRPhenomX_Return_SNorm_MSA():
    ## TODO: implement elliptic Jacobi functions
    return
    
    
def IMRPhenomX_L_norm_3PN_of_v(v: float, v2: float, L_norm: float, pPrec) -> float:
    cL = pPrec['constants_L']  # shorthand
    return L_norm * 1.0 + v2 * (
        cL[0] + v * cL[1] + v2 * (
            cL[2] + v * cL[3] + v2 * cL[4]
        )
    )
    
def IMRPhenomX_Return_Roots_MSA(LNorm, JNorm, pPrec):
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec)  
    B, C, D = vBCD[0], vBCD[1], vBCD[2]

    B2 = B * B
    B3 = B2 * B
    BC = B * C

    p = C - B2 / 3.0
    qc = (2.0 / 27.0) * B3 - BC / 3.0 + D

    sqrtarg = jnp.sqrt(-p / 3.0)
    acosarg = 1.5 * qc / (p * sqrtarg)
    acosarg = jnp.clip(acosarg, -1.0, 1.0)

    theta = jnp.arccos(acosarg) / 3.0
    cos_theta = jnp.cos(theta)

    invalid_case = (
        jnp.isnan(theta) |
        jnp.isnan(sqrtarg) |
        (pPrec.dotS1Ln == 1.0) |
        (pPrec.dotS2Ln == 1.0) |
        (pPrec.dotS1Ln == -1.0) |
        (pPrec.dotS2Ln == -1.0) |
        (pPrec.S1_norm_2 == 0.0) |
        (pPrec.S2_norm_2 == 0.0)
    )

    def roots_when_valid():
        tmp1 = 2.0 * sqrtarg * jnp.cos(theta - 4.0 * jnp.pi / 3.0) - B / 3.0
        tmp2 = 2.0 * sqrtarg * jnp.cos(theta - 2.0 * jnp.pi / 3.0) - B / 3.0
        tmp3 = 2.0 * sqrtarg * cos_theta - B / 3.0

        tmp4 = jnp.maximum(jnp.maximum(tmp1, tmp2), tmp3)
        tmp5 = jnp.minimum(jnp.minimum(tmp1, tmp2), tmp3)

        tmp6 = jnp.where(
            (tmp4 - tmp3 > 0.0) & (tmp5 - tmp3 < 0.0),
            tmp3,
            jnp.where((tmp4 - tmp1 > 0.0) & (tmp5 - tmp1 < 0.0), tmp1, tmp2)
        )

        S32 = tmp5
        Smi2 = jnp.abs(tmp6)
        Spl2 = jnp.abs(tmp4)
        return S32, Smi2, Spl2

    def roots_when_invalid():
        Smi2 = pPrec['S_0_norm_2']
        Spl2 = Smi2 + 1e-9
        S32 = 0.0
        return S32, Smi2, Spl2

    S32, Smi2, Spl2 = jax.lax.cond(
        invalid_case,
        roots_when_invalid,
        roots_when_valid
    )

    return jnp.array([S32, Smi2, Spl2])

def IMRPhenomX_Return_Constants_c_MSA(v, JNorm, pPrec):
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v6 = v3 * v3
    JNorm2 = JNorm * JNorm
    Seff = pPrec['Seff']


    x = JNorm * (
        0.75 * (1.0 - Seff * v) * v2 * (
            pPrec['eta3']
            + 4.0 * pPrec['eta3'] * Seff * v
            - 2.0 * pPrec['eta'] * (
                JNorm2 - pPrec['Spl2'] + 2.0 * (pPrec['S1_norm_2'] - pPrec['S2_norm_2']) * pPrec['delta_qq']
            ) * v2
            - 4.0 * pPrec['eta'] * Seff * (JNorm2 - pPrec['Spl2']) * v3
            + (JNorm2 - pPrec['Spl2']) ** 2 * v4 * pPrec['inveta']
        )
    )

    y = JNorm * (
        -1.5 * pPrec['eta'] * (pPrec['Spl2'] - pPrec['Smi2'])
        * (1.0 + 2.0 * Seff * v - (JNorm2 - pPrec['Spl2']) * v2 * pPrec['inveta2'])
        * (1.0 - Seff * v) * v4
    )

    z = JNorm * (
        0.75 * pPrec['inveta'] * (pPrec['Spl2'] - pPrec['Smi2']) ** 2
        * (1.0 - Seff * v) * v6
    )

    return jnp.array([x, y, z])

def IMRPhenomX_Return_Constants_d_MSA(LNorm, JNorm, pPrec):
    LNorm2 = LNorm * LNorm
    JNorm2 = JNorm * JNorm

    x = - (JNorm2 - (LNorm + pPrec['Spl'])) ** 2 * (JNorm2 - (LNorm - pPrec['Spl'])) ** 2

    y = -2.0 * (pPrec['Spl2'] - pPrec['Smi2']) * (JNorm2 + LNorm2 - pPrec['Spl2'])

    z = -(pPrec['Spl2'] - pPrec['Smi2']) ** 2

    return jnp.array([x, y, z])

def IMRPhenomX_Return_Psi_MSA(v, v2, pPrec):
    return -0.75 * pPrec['g0'] * pPrec['delta_qq'] * (1.0 + pPrec['psi1'] * v + pPrec['psi2'] * v2) / (v2 * v)

def IMRPhenomX_Return_Psi_dot_MSA(v, pPrec):
    v2 = v * v

    A_coeff = -1.5 * v2 * v2 * v2 * (1.0 - v * pPrec['Seff']) * pPrec['sqrt_inveta']
    psi_dot = 0.5 * A_coeff * jnp.sqrt(pPrec['Spl2'] - pPrec['S32'])

    return psi_dot

def IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec):
    JNorm2 = JNorm * JNorm
    LNorm2 = LNorm * LNorm

    S1Norm2 = pPrec['S1_norm_2']
    S2Norm2 = pPrec['S2_norm_2']
    q       = pPrec['qq']
    eta     = pPrec['eta']
    delta   = pPrec['delta_qq']
    deltaSq = delta * delta
    Seff    = pPrec['Seff']

    J2mL2   = JNorm2 - LNorm2
    J2mL2Sq = J2mL2 * J2mL2

    # B coefficient (Eq. B2)
    B_coeff = ((LNorm2 + S1Norm2) * q +
               2.0 * LNorm * Seff -
               2.0 * JNorm2 -
               S1Norm2 - S2Norm2 +
               (LNorm2 + S2Norm2) / q)

    # C coefficient (Eq. B3)
    C_coeff = (J2mL2Sq -
               2.0 * LNorm * Seff * J2mL2 -
               2.0 * ((1.0 - q) / q) * LNorm2 * (S1Norm2 - q * S2Norm2) +
               4.0 * eta * LNorm2 * Seff * Seff -
               2.0 * delta * (S1Norm2 - S2Norm2) * Seff * LNorm +
               2.0 * ((1.0 - q) / q) * (q * S1Norm2 - S2Norm2) * JNorm2)

    # D coefficient (Eq. B4)
    D_coeff = (((1.0 - q) / q) * (S2Norm2 - q * S1Norm2) * J2mL2Sq +
               deltaSq * (S1Norm2 - S2Norm2)**2 * LNorm2 / eta +
               2.0 * delta * LNorm * Seff * (S1Norm2 - S2Norm2) * J2mL2)

    return jnp.array([B_coeff, C_coeff, D_coeff])

def IMRPhenomXGetAndSetPrecessionVariables(pWF, m1_SI, m2_SI,
                                            chi1x, chi1y, chi1z,
                                            chi2x, chi2y, chi2z,
                                            lalParams):
    
    pPrec['ExpansionOrder'] = XLALSimInspiralWaveformParamsLookupPhenomXPExpansionOrder(lalParams)
    
    Mtot_SI = m1_SI + m2_SI  
    # Normalize masses
    m1 = m1_SI / Mtot_SI
    m2 = m2_SI / Mtot_SI
    M = m1 + m2
    #pWF['M'] = m1 + m2  ### pWF needs to be a dict??

    # Mass ratio and symmetric mass ratio
    q   = m1 / m2
    eta = pWF[1]
    
    ## TODO: compute delta?
    ## TODO: compute chieff?
    ## TODO: compute twopiGM, piGM?

    # Spin inputs
    for i, (x, y, z) in enumerate([(chi1x, chi1y, chi1z), (chi2x, chi2y, chi2z)], start=1):
        chi_norm = jnp.sqrt(x*x + y*y + z*z)
        pPrec[f'chi{i}x'] = x
        pPrec[f'chi{i}y'] = y
        pPrec[f'chi{i}z'] = z
        pPrec[f'chi{i}_norm'] = chi_norm

    # Dimensionful spins
    pPrec['S1x'] = chi1x * m1_2
    pPrec['S1y'] = chi1y * m1_2
    pPrec['S1z'] = chi1z * m1_2
    pPrec['S1_norm'] = jnp.abs(pPrec['chi1_norm']) * m1_2
    pPrec['S2x'] = chi2x * m2_2
    pPrec['S2y'] = chi2y * m2_2
    pPrec['S2z'] = chi2z * m2_2
    pPrec['S2_norm'] = jnp.abs(pPrec['chi2_norm']) * m2_2

    # In-plane magnitudes
    pPrec['chi1_perp'] = jnp.sqrt(chi1x*chi1x + chi1y*chi1y)
    pPrec['chi2_perp'] = jnp.sqrt(chi2x*chi2x + chi2y*chi2y)
    pPrec['S1_perp'] = m1_2 * pPrec['chi1_perp']
    pPrec['S2_perp'] = m2_2 * pPrec['chi2_perp']
    STot_x = pPrec['S1x'] + pPrec['S2x']
    STot_y = pPrec['S1y'] + pPrec['S2y']
    pPrec['STot_perp'] = jnp.sqrt(STot_x**2 + STot_y**2)
    pPrec['chiTot_perp'] = pPrec['STot_perp'] * (M**2) / m1_2
    # pWF['chiTot_perp'] = pPrec['chiTot_perp']  ### pWF needs to be a dict??

    return 0  # Success
