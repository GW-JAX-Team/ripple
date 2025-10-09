import jax
import jax.numpy as jnp
from ripple import Mc_eta_to_ms

from ..constants import gt, MSUN
import numpy as np
from .IMRPhenomXAS import Phase as PhDPhase
from .IMRPhenomXAS import Amp as PhDAmp
from .IMRPhenomXAS import gen_IMRPhenomXAS
from .IMRPhenomX_utils import PhenomX_amp_coeff_table, PhenomX_phase_coeff_table

from ..typing import Array
from .IMRPhenomXP_utils import *   
from .IMRPhenomX_utils import *


def PhenomXPCoreTwistUp22(
    Mf,  ## Frequency in geometric units (on LAL says Hz?)
    hAS,  ## Underlying aligned-spin IMRPhenomXAS strain
    pWF,  ## IMRPhenomX Waveform Struct (TODO)
    pPrec  ## IMRPhenomXP Precession Struct (TODO)  
):

    omega = jnp.pi * Mf
    logomega = jnp.log(omega)
    omega_cbrt = (omega) ** (1 / 3)
    omega_cbrt2 = omega_cbrt * omega_cbrt
    
    v = omega_cbrt
    
    vangles = jnp.array([0,0,0])
    
    ## Euler Angles from Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    vangles  = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(v,pWF,pPrec)
    alpha    = vangles[0] - pPrec["alpha_offset"]
    epsilon  = vangles[1] - pPrec["epsilon_offset"]
    cos_beta = vangles[2]
     

    # print("alpha, epsilon: ", alpha, epsilon)
    cBetah, sBetah = WignerdCoefficients_cosbeta(cos_beta)

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah2 * cBetah
    cBetah4 = cBetah3 * cBetah
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah2 * sBetah
    sBetah4 = sBetah3 * sBetah

    # d2 = jnp.array(
    #     [
    #         sBetah4,
    #         2 * cBetah * sBetah3,
    #         jnp.sqrt(6) * sBetah2 * cBetah2,
    #         2 * cBetah3 * sBetah,
    #         cBetah4,
    #     ]
    # )  ## LR in PhenomP.py we don't compute this, but in X.c and P.c yes
    ##  same for dm2
    
    # Y2m are the spherical harmonics with s=-2, l=2, m=-2,-1,0,1,2
    Y2mA = jnp.array([pPrec['Y2m2'],pPrec['Y2m1'],pPrec['Y20'],pPrec['Y21'],pPrec['Y22']])  # need to pass Y2m in a 5-component list
    hp_sum = 0
    hc_sum = 0

    cexp_i_alpha = jnp.exp(1j * alpha)
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    A2m2emm = (
        cexp_2i_alpha * cBetah4 * Y2mA[0]
        - cexp_i_alpha * 2 * cBetah3 * sBetah * Y2mA[1]
        + 1 * jnp.sqrt(6) * sBetah2 * cBetah2 * Y2mA[2]
        - cexp_mi_alpha * 2 * cBetah * sBetah3 * Y2mA[3]
        + cexp_m2i_alpha * sBetah4 * Y2mA[4]
    )
    A22emmstar = (
        cexp_m2i_alpha * sBetah4 * jnp.conjugate(Y2mA[0])
        + cexp_mi_alpha * 2 * cBetah * sBetah3 * jnp.conjugate(Y2mA[1])
        + 1 * jnp.sqrt(6) * sBetah2 * cBetah2 * jnp.conjugate(Y2mA[2])
        + cexp_i_alpha * 2 * cBetah3 * sBetah * jnp.conjugate(Y2mA[3])
        + cexp_2i_alpha * cBetah4 * jnp.conjugate(Y2mA[4])
    )
    hp_sum = A2m2emm + A22emmstar * pPrec['PolarizationSymmetry']
    hc_sum = 1j * (A2m2emm - A22emmstar * pPrec['PolarizationSymmetry'])
    eps_phase_hP = jnp.exp(-2j * epsilon) * hAS / 2.0

    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum

    return hp, hc


def gen_IMRPhenomXP_hphc(f: Array, 
                          params: Array,  
                          prec_params,    
                          f_ref: float):
    """
    Returns:
    --------
      hp (array): Strain of the plus polarization
      hc (array): Strain of the cross polarization
    """
    iota = params[7]
    h0 = gen_IMRPhenomXAS(f, params, f_ref)

    hp, hc = PhenomXPCoreTwistUp22(f, h0, params, prec_params)
    
    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))  
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc