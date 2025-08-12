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

def IMRPhenomX_Initialize_MSA_System(pWF, pPrec, ExpansionOrder):
    eta  = pPrec['eta']
    eta2 = pPrec['eta2']
    eta3 = pPrec['eta3']
    eta4 = pPrec['eta4']

    m1 = pWF['m1']
    m2 = pWF['m2']

    domegadt_constants_NS = jnp.array([
        96. / 5., -1486. / 35., -264. / 5., 384. * jnp.pi / 5., 34103. / 945., 
        13661. / 105., 944. / 15., jnp.pi * (-4159. / 35.), jnp.pi * (-2268. / 5.),
        (16447322263. / 7276500. + jnp.pi**2 * 512. / 5. - jnp.log(2.) * 109568. / 175. - jnp.euler_gamma * 54784. / 175.),
        (-56198689. / 11340. + jnp.pi**2 * 902. / 5.),
        1623. / 140., -1121. / 27., -54784. / 525., -jnp.pi * 883. / 42.,
        jnp.pi * 71735. / 63., jnp.pi * 73196. / 63.
    ])

    domegadt_constants_SO = jnp.array([
        -904. / 5., -120., -62638. / 105., 4636. / 5., -6472. / 35., 3372. / 5.,
        -jnp.pi * 720., -jnp.pi * 2416. / 5., -208520. / 63., 796069. / 105.,
        -100019. / 45., -1195759. / 945., 514046. / 105., -8709. / 5.,
        -jnp.pi * 307708. / 105., jnp.pi * 44011. / 7., -jnp.pi * 7992. / 7.,
        jnp.pi * 151449. / 35.
    ])

    domegadt_constants_SS = jnp.array([
        -494. / 5., -1442. / 5., -233. / 5., -719. / 5.
    ])

    L_csts_nonspin = jnp.array([
        3. / 2., 1. / 6., 27. / 8., -19. / 8., 1. / 24., 135. / 16.,
        -6889. / 144. + 41. / 24. * jnp.pi**2, 31. / 24., 7. / 1296.
    ])

    L_csts_spinorbit = jnp.array([
        -14. / 6., -3. / 2., -11. / 2., 133. / 72., -33. / 8., 7. / 4.
    ])

    # Flip q convention: Chatziioannou uses q < 1 (m1 > m2), IMRPhenomX uses q > 1
    q = m2 / m1           # q < 1, m1 > m2
    invq = 1.0 / q

    pPrec['qq'] = q
    pPrec['invqq'] = invq

    # Reduced mass
    mu = (m1 * m2) / (m1 + m2)
    
    pPrec['delta_qq']  = (1.0 - pPrec['qq']) / (1.0 + pPrec['qq'])
    pPrec['delta2_qq'] = pPrec['delta_qq'] ** 2
    pPrec['delta3_qq'] = pPrec['delta_qq'] * pPrec['delta2_qq']
    pPrec['delta4_qq'] = pPrec['delta_qq'] * pPrec['delta3_qq']

    # Initialize vectors
    S1v = jnp.array([0.0, 0.0, 0.0])
    S2v = jnp.array([0.0, 0.0, 0.0])
    Lhat = jnp.array([0.0, 0.0, 1.0])

    # Set fixed Lhat variables
    pPrec['Lhat_cos_theta'] = 1.0
    pPrec['Lhat_phi'] = 0.0
    pPrec['Lhat_theta'] = 0.0

    # Dimensionful spin vectors (eta = m1 * m2, q = m2 / m1)
    S1v[0] = pPrec['chi1x'] * eta / q
    S1v[1] = pPrec['chi1y'] * eta / q
    S1v[2] = pPrec['chi1z'] * eta / q

    S2v[0] = pPrec['chi2x'] * eta * q
    S2v[1] = pPrec['chi2y'] * eta * q
    S2v[2] = pPrec['chi2z'] * eta * q

    # Norms of spin vectors
    S1_0_norm = jnp.linalg.norm(S1v)
    S2_0_norm = jnp.linalg.norm(S2v)

    # Store initial spin vectors
    pPrec['S1_0'] = S1v
    pPrec['S2_0'] = S2v

    # Reference velocity v and v^2
    pPrec['v_0'] = jnp.cbrt(pPrec['piGM'] * pWF['fRef'])
    pPrec['v_0_2'] = pPrec['v_0'] ** 2

    # Reference orbital angular momentum
    L_0 = Lhat * eta / pPrec['v_0']
    pPrec['L_0'] = L_0
    
    dotS1L = jnp.dot(S1v, Lhat)
    dotS2L = jnp.dot(S2v, Lhat)
    dotS1S2 = jnp.dot(S1v, S2v)
    dotS1Ln = dotS1L / S1_0_norm
    dotS2Ln = dotS2L / S2_0_norm

    # Store results in the precession structure
    pPrec['dotS1L'] = dotS1L
    pPrec['dotS2L'] = dotS2L
    pPrec['dotS1S2'] = dotS1S2
    pPrec['dotS1Ln'] = dotS1Ln
    pPrec['dotS2Ln'] = dotS2Ln
    
    # --- PN Coefficients for orbital angular momentum ---
    pPrec['constants_L'] = jnp.zeros(5)
    pPrec['constants_L'] = pPrec['constants_L'].at[0].set(
        L_csts_nonspin[0] + eta * L_csts_nonspin[1]
    )

    pPrec['constants_L'] = pPrec['constants_L'].at[1].set(
        IMRPhenomX_Get_PN_beta(L_csts_spinorbit[0], L_csts_spinorbit[1], pPrec)  ## (TODO)
    )

    pPrec['constants_L'] = pPrec['constants_L'].at[2].set(
        L_csts_nonspin[2]
        + eta * L_csts_nonspin[3]
        + eta ** 2 * L_csts_nonspin[4]
    )

    pPrec['constants_L'] = pPrec['constants_L'].at[3].set(
        IMRPhenomX_Get_PN_beta(
            L_csts_spinorbit[2] + L_csts_spinorbit[3] * eta,
            L_csts_spinorbit[4] + L_csts_spinorbit[5] * eta,
            pPrec
        )
    )

    pPrec['constants_L'] = pPrec['constants_L'].at[4].set(
        L_csts_nonspin[5]
        + L_csts_nonspin[6] * eta
        + L_csts_nonspin[7] * eta ** 2
        + L_csts_nonspin[8] * eta ** 3
    )

    # Effective total spin 
    Seff = (1.0 + q) * pPrec['dotS1L'] + (1.0 + 1.0 / q) * pPrec['dotS2L']
    pPrec['Seff'] = Seff

    # Initial total spin, S0 = S1 + S2
    S_0 = S1v + S2v  
    pPrec['S_0'] = S_0

    # Initial total angular momentum J0 = L + S1 + S2 
    pPrec['J_0'] = L_0 + S_0 

    pPrec['S_0_norm'] = jnp.linalg.norm(S_0)

    pPrec['L_0_norm'] = jnp.linalg.norm(pPrec['L_0'])
    pPrec['J_0_norm'] = jnp.linalg.norm(pPrec['J_0'])
    
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(pPrec['L_0_norm'], pPrec['J_0_norm'], pPrec)
    
    vRoots = IMRPhenomX_Return_Roots_MSA(pPrec['L_0_norm'], pPrec['J_0_norm'], pPrec)
    
    pPrec['Spl2'] = vRoots[2]
    pPrec['Smi2'] = vRoots[1]
    pPrec['S32']  = vRoots[0]

    pPrec['Spl2pSmi2'] = pPrec['Spl2'] + pPrec['Smi2']
    pPrec['Spl2mSmi2'] = pPrec['Spl2'] - pPrec['Smi2']

    pPrec['Spl'] = jnp.sqrt(pPrec['Spl2'])
    pPrec['Smi'] = jnp.sqrt(pPrec['Smi2'])

    # Eq. 45 of PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['SAv2'] = 0.5 * pPrec['Spl2pSmi2']
    pPrec['SAv']  = jnp.sqrt(pPrec['SAv2'])
    pPrec['invSAv2'] = 1.0 / pPrec['SAv2']
    pPrec['invSAv']  = 1.0 / pPrec['SAv']

    # Eq. 41 of PRD, 95, 104004, (2017), arXiv:1703.03967
    c_1 = 0.5 * (pPrec['J_0_norm']**2 - pPrec['L_0_norm']**2 - pPrec['SAv2']) / (pPrec['L_0_norm'] * eta)
    pPrec['c1'] = c_1
    pPrec['c12'] = c_1 ** 2
    pPrec['c1_over_eta'] = c_1 / eta

    # Average spin couplings over one precession cycle: A9 - A14 of arXiv:1703.03967
    omqsq = (1.0 - q)**2 + 1e-16
    omq2  = (1.0 - q**2) + 1e-16

    pPrec['S1L_pav'] = (c_1 * (1.0 + q) - q * eta * Seff) / (eta * omq2)
    pPrec['S2L_pav'] = -q * (c_1 * (1.0 + q) - eta * Seff) / (eta * omq2)
    pPrec['S1S2_pav'] = 0.5 * pPrec['SAv2'] - 0.5 * (pPrec['S1_norm_2'] + pPrec['S2_norm_2'])
    pPrec['S1Lsq_pav'] = pPrec['S1L_pav'] ** 2 + (pPrec['Spl2mSmi2'] ** 2 * pPrec['v_0_2']) / (32.0 * eta2 * omqsq)
    pPrec['S2Lsq_pav'] = pPrec['S2L_pav'] ** 2 + (q**2 * (pPrec['Spl2mSmi2'] ** 2) * pPrec['v_0_2']) / (32.0 * eta2 * omqsq)
    pPrec['S1LS2L_pav'] = pPrec['S1L_pav'] * pPrec['S2L_pav'] - q * pPrec['Spl2mSmi2'] * pPrec['v_0_2'] / (32.0 * eta2 * omqsq)

    # beta coefficients
    pPrec['beta3'] = ((113.0/12.0) + (25.0/4.0)*(m2/m1)) * pPrec['S1L_pav'] + ((113.0/12.0) 
                        + (25.0/4.0)*(m1/m2)) * pPrec['S2L_pav']

    pPrec['beta5'] = (((31319.0/1008.0) - (1159.0/24.0)*eta) + (m2/m1)*((809.0/84.0) 
                        - (281.0/8.0)*eta)) * pPrec['S1L_pav'] + (((31319.0/1008.0) - (1159.0/24.0)*eta) + (m1/m2)*((809.0/84.0) - (281.0/8.0)*eta)) * pPrec['S2L_pav']

    pPrec['beta6'] = jnp.pi * (
        ((75.0/2.0) + (151.0/6.0)*(m2/m1)) * pPrec['S1L_pav'] +
        ((75.0/2.0) + (151.0/6.0)*(m1/m2)) * pPrec['S2L_pav']
    )

    beta7_common = (130325.0/756.0) - (796069.0/2016.0)*eta + (100019.0/864.0)*eta2
    beta7_S = (1195759.0/18144.0 - 257023.0/1008.0 * eta + 2903.0/32.0 * eta2) 

    pPrec['beta7'] = beta7_common + (m2/m1) * beta7_S * pPrec['S1L_pav'] + beta7_common + (m1/m2) * beta7_S * pPrec['S2L_pav']

    pPrec['sigma4'] = (1.0 / mu) * ((247.0/48.0) * pPrec['S1S2_pav'] - (721.0/48.0) * pPrec['S1L_pav'] * pPrec['S2L_pav']) + (1.0 / (m1**2)) * ((233.0/96.0) * pPrec['S1_norm_2'] 
                        - (719.0/96.0) * pPrec['S1Lsq_pav']) + (1.0 / (m2**2)) * ((233.0/96.0) * pPrec['S2_norm_2'] - (719.0/96.0) * pPrec['S2Lsq_pav'])

    # PN coefficients
    pPrec['a0'] = 96.0 * eta / 5.0
    '''
    pPrec['a2'] = (-(743.0/336.0) - (11.0/4.0)*eta) * pPrec['a0']
    pPrec['a3'] = (4.0 * jnp.pi - pPrec['beta3']) * pPrec['a0']
    pPrec['a4'] = ((34103.0/18144.0) + (13661.0/2016.0)*eta + (59.0/18.0)*eta2 - pPrec['sigma4']) * pPrec['a0']
    pPrec['a5'] = (-(4159.0/672.0)*jnp.pi - (189.0/8.0)*jnp.pi*eta - pPrec['beta5']) * pPrec['a0']
    '''
    pPrec['a6'] = ((16447322263.0/139708800.0) + (16.0/3.0)*jnp.pi**2 - (856.0/105.0)*jnp.log(16.0) - (1712.0/105.0)*jnp.euler_gamma 
            - pPrec['beta6'] + eta*(451.0/48.0)*jnp.pi**2 - (56198689.0/217728.0) + eta2*(541.0/896.0) - eta3*(5605.0/2592.0)) * pPrec['a0']
    pPrec['a7'] = (-(4415.0/4032.0)*jnp.pi + (358675.0/6048.0)*jnp.pi*eta + (91495.0/1512.0)*jnp.pi*eta2 - pPrec['beta7']) * pPrec['a0']
    
    # Default behaviour of IMRPhenomXP (223: MSA with fallback to NNLO)
    pPrec['a0'] = eta * domegadt_constants_NS[0]
    pPrec['a2'] = eta * (domegadt_constants_NS[1] + eta * domegadt_constants_NS[2])
    pPrec['a3'] = eta * (domegadt_constants_NS[3] +
        IMRPhenomX_Get_PN_beta(domegadt_constants_SO[0],domegadt_constants_SO[1],pPrec))
    pPrec['a4'] = eta * (domegadt_constants_NS[4] +eta * (domegadt_constants_NS[5] + eta * domegadt_constants_NS[6]) +
        IMRPhenomX_Get_PN_sigma(domegadt_constants_SS[0],domegadt_constants_SS[1],pPrec) +
        IMRPhenomX_Get_PN_tau(domegadt_constants_SS[2],domegadt_constants_SS[3],pPrec))
    pPrec['a5'] = eta * (domegadt_constants_NS[7] +eta * domegadt_constants_NS[8] +
        IMRPhenomX_Get_PN_beta(
            domegadt_constants_SO[2] + eta * domegadt_constants_SO[3],
            domegadt_constants_SO[4] + eta * domegadt_constants_SO[5],
            pPrec
        )
    )
    
    pPrec['a0_2'] = pPrec['a0'] * pPrec['a0']
    pPrec['a0_3'] = pPrec['a0_2'] * pPrec['a0']
    pPrec['a2_2'] = pPrec['a2'] * pPrec['a2']

    # g-coefficients from Appendix A of Chatziioannou et al, PRD, 95, 104004, (2017), arXiv:1703.03967.
    pPrec['g0'] = 1.0 / pPrec['a0']
    pPrec['g2'] = -pPrec['a2'] / pPrec['a0_2']
    pPrec['g3'] = -pPrec['a3'] / pPrec['a0_2']
    pPrec['g4'] = -(pPrec['a4'] * pPrec['a0'] - pPrec['a2_2']) / pPrec['a0_3']
    pPrec['g5'] = -(pPrec['a5'] * pPrec['a0'] - 2.0 * pPrec['a3'] * pPrec['a2']) / pPrec['a0_3']

    delta = pPrec['delta_qq']
    delta2 = delta * delta
    delta3 = delta * delta2
    delta4 = delta * delta3

    # Phase coefficients: Eq. 51 and Appendix C of arXiv:1703.03967
    pPrec['psi0'] = 0.0
    pPrec['psi1'] = 3.0 * (2.0 * eta2 * Seff - c_1) / (eta * delta2)
    pPrec['psi2'] = 0.0

    # Precompute useful quantities
    c_1_over_nu = pPrec['c1_over_eta']
    c_1_over_nu_2 = c_1_over_nu * c_1_over_nu
    one_p_q_sq = (1.0 + q)**2
    Seff_2 = Seff * Seff
    q_2 = q * q
    one_m_q_sq = (1.0 - q)**2
    one_m_q2_2 = (1.0 - q_2)**2
    one_m_q_4 = one_m_q_sq * one_m_q_sq
    
    Del1 = 4.0 * c_1_over_nu_2 * one_p_q_sq
    Del2 = 8.0 * c_1_over_nu * q * (1.0 + q) * Seff
    Del3 = 4.0 * (one_m_q2_2 * pPrec['S1_norm_2'] - q_2 * Seff_2)
    Del4 = 4.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
    Del5 = 8.0 * c_1_over_nu * q_2 * (1.0 + q) * Seff
    Del6 = 4.0 * (one_m_q2_2 * pPrec['S2_norm_2'] - q_2 * Seff_2)

    pPrec['Delta'] = jnp.sqrt(jnp.abs((Del1 - Del2 - Del3) * (Del4 - Del5 - Del6)))
    
    u1 = 3.0 * pPrec['g2'] / pPrec['g0']
    u2 = 0.75 * one_p_q_sq / one_m_q_4
    u3 = -20.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
    u4 = 2.0 * one_m_q2_2 * (q * (2.0 + q) * pPrec['S1_norm_2']
        + (1.0 + 2.0 * q) * pPrec['S2_norm_2'] - 2.0 * q * pPrec['SAv2'])
    u5 = 4.0 * q_2 * (7.0 + 6.0 * q + 7.0 * q_2) * c_1_over_nu * Seff
    u6 = 2.0 * q_2 * (3.0 + 4.0 * q + 3.0 * q_2) * Seff_2
    u7 = q * pPrec['Delta']

    # (Eq. C2 of 1703.03967)
    pPrec['psi2'] = u1 + u2 * (u3 + u4 + u5 - u6 + u7)
    
    # Eq. D1 - D5  of 1703.03967
    Rm = pPrec['Spl2'] - pPrec['Smi2']
    Rm_2 = Rm * Rm
    cp = pPrec['Spl2'] * eta2 - pPrec['c12']
    cm = pPrec['Smi2'] * eta2 - pPrec['c12']
    cpcm = jnp.abs(cp * cm)
    sqrt_cpcm = jnp.sqrt(cpcm)
    a1dD = 0.5 + 0.75 / eta
    a2dD = -0.75 * Seff / eta

    # Eq. E3- E4 of 1703.03967
    D2RmSq = (cp - sqrt_cpcm) / eta2
    D4RmSq = -0.5 * Rm * sqrt_cpcm / eta2 - cp / eta4 * (sqrt_cpcm - cp)

    S0m = pPrec['S1_norm_2'] - pPrec['S2_norm_2']

    aw = -3.0 * (1. + q) / q * (2. * (1. + q) * eta2 * Seff * c_1 - (1. + q) * pPrec['c12'] + (1. - q) * eta2 * S0m)
    cw = 3.0 / 32.0 / eta * Rm_2
    dw = 4.0 * cp - 4.0 * D2RmSq * eta2
    hw = -2.0 * (2.0 * D2RmSq - Rm) * c_1
    fw = Rm * D2RmSq - D4RmSq - 0.25 * Rm_2

    adD = aw / dw
    hdD = hw / dw
    cdD = cw / dw
    fdD = fw / dw

    gw = 3. / 16. / eta2 / eta * Rm_2 * (c_1 - eta2 * Seff)
    gdD = gw / dw

    # Powers of coefficients
    hdD_2 = hdD * hdD
    adDfdD = adD * fdD
    adDfdDhdD = adDfdD * hdD
    adDhdD_2 = adD * hdD_2

    # Eq. D10-D15 in PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['Omegaz0'] = a1dD + adD    
    pPrec['Omegaz1'] = a2dD - adD*Seff - adD*hdD
    pPrec['Omegaz2'] = adD*hdD*Seff + cdD - adD*fdD + adD*hdD_2
    pPrec['Omegaz3'] = (adDfdD - cdD - adDhdD_2)*(Seff + hdD) + adDfdDhdD
    pPrec['Omegaz4'] = (cdD + adDhdD_2 - 2.0*adDfdD)*(hdD*Seff + hdD_2 - fdD) - adD*fdD*fdD
    pPrec['Omegaz5'] = (cdD - adDfdD + adDhdD_2) * fdD * (Seff + 2.0*hdD) - (cdD + adDhdD_2 - 2.0*adDfdD) * hdD_2 * (Seff + hdD) - adDfdD*fdD*hdD
    
    # Condition for MSA fallback to NNLO
    condition = jnp.abs(pPrec['Omegaz5']) > 1000.0
    pPrec['MSA_ERROR'] = jnp.where(condition, 1, 0)
    
    g0 = pPrec['g0']

    # Eq. 65 coefficients (D16 - D21 of PRD, 95, 104004, (2017), arXiv:1703.03967)
    pPrec['Omegaz0_coeff'] = 3.0 * g0 * pPrec['Omegaz0']
    pPrec['Omegaz1_coeff'] = 3.0 * g0 * pPrec['Omegaz1']
    pPrec['Omegaz2_coeff'] = 3.0 * (g0 * pPrec['Omegaz2'] + pPrec['g2'] * pPrec['Omegaz0'])
    pPrec['Omegaz3_coeff'] = 3.0 * (g0 * pPrec['Omegaz3'] + pPrec['g2'] * pPrec['Omegaz1'] + pPrec['g3'] * pPrec['Omegaz0'])
    pPrec['Omegaz4_coeff'] = 3.0 * (g0 * pPrec['Omegaz4'] + pPrec['g2'] * pPrec['Omegaz2'] + pPrec['g3'] * pPrec['Omegaz1'] + pPrec['g4'] * pPrec['Omegaz0'])
    pPrec['Omegaz5_coeff'] = 3.0 * (g0 * pPrec['Omegaz5'] + pPrec['g2'] * pPrec['Omegaz3'] + pPrec['g3'] * pPrec['Omegaz2'] + pPrec['g4'] * pPrec['Omegaz1'] + pPrec['g5'] * pPrec['Omegaz0'])

    # zeta coefficients (Appendix E of PRD, 95, 104004, (2017), arXiv:1703.03967)
    c1oveta2 = c_1 / eta2
    pPrec['Omegazeta0'] = pPrec['Omegaz0']
    pPrec['Omegazeta1'] = pPrec['Omegaz1'] + pPrec['Omegaz0'] * c1oveta2
    pPrec['Omegazeta2'] = pPrec['Omegaz2'] + pPrec['Omegaz1'] * c1oveta2
    pPrec['Omegazeta3'] = pPrec['Omegaz3'] + pPrec['Omegaz2'] * c1oveta2 + gdD
    pPrec['Omegazeta4'] = pPrec['Omegaz4'] + pPrec['Omegaz3'] * c1oveta2 - gdD * Seff - gdD * hdD
    pPrec['Omegazeta5'] = pPrec['Omegaz5'] + pPrec['Omegaz4'] * c1oveta2 + gdD * hdD * Seff + gdD * (hdD_2 - fdD)

    pPrec['Omegazeta0_coeff'] = -pPrec['g0'] * pPrec['Omegazeta0']
    pPrec['Omegazeta1_coeff'] = -1.5 * pPrec['g0'] * pPrec['Omegazeta1']
    pPrec['Omegazeta2_coeff'] = -3.0 * (pPrec['g0'] * pPrec['Omegazeta2'] + pPrec['g2'] * pPrec['Omegazeta0'])
    pPrec['Omegazeta3_coeff'] = 3.0 * (pPrec['g0'] * pPrec['Omegazeta3'] + pPrec['g2'] * pPrec['Omegazeta1'] + pPrec['g3'] * pPrec['Omegazeta0'])
    pPrec['Omegazeta4_coeff'] = 3.0 * (pPrec['g0'] * pPrec['Omegazeta4'] + pPrec['g2'] * pPrec['Omegazeta2'] + pPrec['g3'] * pPrec['Omegazeta1'] + pPrec['g4'] * pPrec['Omegazeta0'])
    pPrec['Omegazeta5_coeff'] = 1.5 * (pPrec['g0'] * pPrec['Omegazeta5'] + pPrec['g2'] * pPrec['Omegazeta3'] + pPrec['g3'] * pPrec['Omegazeta2'] + pPrec['g4'] * pPrec['Omegazeta1'] + pPrec['g5'] * pPrec['Omegazeta0'])
    
    ## Here we're only considering the default setting, where the expansion order for MSA correction is 5
    ## switch to choose expansion order not yet implemented (TODO)
    pPrec['Omegaz5_coeff']    = 0.0
    pPrec['Omegazeta5_coeff'] = 0.0
    

    condition_equal = jnp.abs(pPrec['Smi2'] - pPrec['Spl2']) < 1.0e-5

    def branch_equal(pPrec):
        return 0.0

    def branch_not_equal(pPrec):
        mm_val = jnp.sqrt((pPrec['Smi2'] - pPrec['Spl2']) 
                          (pPrec['S32'] - pPrec['Spl2']))
        tmpB_val = ((pPrec['S_0_norm'] * pPrec['S_0_norm']) - pPrec['Spl2']) / (pPrec['Smi2'] - pPrec['Spl2'])

        vol_elem = jnp.dot(jnp.cross(L_0, S1v),S2v)
        vol_sign_val = jnp.sign(vol_elem)

        psi_v0_val = IMRPhenomX_psiofv(pPrec['v_0'], pPrec['v_0_2'], 0.0,
                                       pPrec['psi1'], pPrec['psi2'], pPrec)

        # Clamp tmpB in conditions
        cond_case1 = jnp.logical_and(tmpB_val > 1.0,
                                     (tmpB_val - 1.0) < 1.0e-5)
        cond_case2 = jnp.logical_and(tmpB_val < 0.0,
                                     tmpB_val > -1.0e-5)

        def case1():
            return gsl_sf_ellint_F( ##(TODO)
                jnp.arcsin(vol_sign_val * jnp.sqrt(1.0)),
                mm_val
            ) - psi_v0_val

        def case2():
            return gsl_sf_ellint_F(
                jnp.arcsin(vol_sign_val * jnp.sqrt(0.0)),
                mm_val
            ) - psi_v0_val

        def case3():
            return gsl_sf_ellint_F(
                jnp.arcsin(vol_sign_val * jnp.sqrt(tmpB_val)),
                mm_val
            ) - psi_v0_val

        psi0_val = jnp.select(
            [cond_case1, cond_case2, jnp.logical_not(
                jnp.logical_or(tmpB_val > 1.0, tmpB_val < 0.0))],
            [case1(), case2(), case3()]
        )

        return psi0_val

    pPrec['psi0'] = jnp.where(condition_equal,
                     branch_equal(pPrec),
                     branch_not_equal(pPrec))
    
    vMSA = jnp.where(condition_equal, jnp.array([0.,0.,0.]), 
                     IMRPhenomX_Return_MSA_Corrections_MSA(pPrec['v_0'],pPrec['L_0_norm'],pPrec['J_0_norm'],pPrec))
    
    phiz_0        = IMRPhenomX_Return_phiz_MSA(pPrec['v_0'],pPrec['J_0_norm'],pPrec)
    
    zeta_0        = IMRPhenomX_Return_zeta_MSA(pPrec['v_0'],pPrec)
    
    pPrec['phiz_0']    = - phiz_0 - vMSA[0]
    pPrec['zeta_0']    = - zeta_0 - vMSA[1]
    
    return pPrec