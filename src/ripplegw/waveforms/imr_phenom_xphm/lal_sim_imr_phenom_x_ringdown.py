"""LAL's IMRPhenomX_ringdown.c JAX implementation."""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.typing import Array

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXPhaseCoefficientsDataClass,
    IMRPhenomXUsefulPowersDataClass,
    IMRPhenomXWaveformDataClass,
)

def imr_phenom_x_ringdown_phase_22_v4(eta: float, s: float, dchi: float, delta: float, rd_phase_flag: int) -> float:
    """IMRPhenomX_Ringdown_Phase_22_v4."""
    # /*
    #     Effective Spin Used: STotR.
    # */

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s

    checkify.check(
        rd_phase_flag == 105,
        "Error in IMRPhenomX_Ringdown_Phase_22_v4: "
        "IMRPhenomXRingdownPhaseVersion is not valid. Recommended flag is 105.",
    )

    # /* Canonical, 5 coefficients */
    no_spin = (
        -85.86062966719405
        - 4616.740713893726 * eta
        - 4925.756920247186 * eta2
        + 7732.064464348168 * eta3
        + 12828.269960300782 * eta4
        - 39783.51698102803 * eta5
    ) / (1.0 + 50.206318806624004 * eta)

    eq_spin = (
        s
        * (
            33.335857451144356
            - 36.49019206094966 * s
            + eta3 * (1497.3545918387515 - 101.72731770500685 * s) * s
            - 3.835967351280833 * s2
            + 2.302712009652155 * s3
            + eta2
            * (
                93.64156367505917
                - 18.184492163348665 * s
                + 423.48863373726243 * s2
                - 104.36120236420928 * s3
                - 719.8775484010988 * s4
            )
            + 1.6533417657003922 * s4
            + eta
            * (
                -69.19412903018717
                + 26.580344399838758 * s
                - 15.399770764623746 * s2
                + 31.231253209893488 * s3
                + 97.69027029734173 * s4
            )
            + eta4
            * (
                1075.8686153198323
                - 3443.0233614187396 * s
                - 4253.974688619423 * s2
                - 608.2901586790335 * s3
                + 5064.173605639933 * s4
            )
        )
    ) / (-1.3705601055555852 + 1.0 * s)

    uneq_spin = dchi * delta * eta * (22.363215261437862 + 156.08206945239374 * eta)

    return no_spin + eq_spin + uneq_spin


def imr_phenom_x_ringdown_phase_22_d12(eta: float, s: float, dchi: float, delta: float, rd_phase_flag: int) -> float:

    # /*
    #     Effective Spin Used: STotR.
    # */

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s

    checkify.check(
        rd_phase_flag == 105,
        "Error in IMRPhenomX_Ringdown_Phase_22_d12: "
        "IMRPhenomXRingdownPhaseVersion is not valid. Recommended flag is 105.",
    )

    no_spin = (eta * (0.7207992174994245 - 1.237332073800276 * eta + 6.086871214811216 * eta2)) / (
        0.006851189888541745 + 0.06099184229137391 * eta - 0.15500218299268662 * eta2 + 1.0 * eta3
    )

    eq_spin = (
        (
            0.06519048552628343
            - 25.25397971063995 * eta
            - 308.62513664956975 * eta4
            + 58.59408241189781 * eta2
            + 160.14971486043524 * eta3
        )
        * s
        + eta
        * (-5.215945111216946 + 153.95945758807616 * eta - 693.0504179144295 * eta2 + 835.1725103648205 * eta3)
        * s2
        + (0.20035146870472367 - 0.28745205203100666 * eta - 47.56042058800358 * eta4) * s3
        + eta * (5.7756520242745735 - 43.97332874253772 * eta + 338.7263666984089 * eta3) * s4
        + (-0.2697933899920511 + 4.917070939324979 * eta - 22.384949087140086 * eta4 - 11.61488280763592 * eta2) * s5
    ) / (1.0 - 0.6628745847248266 * s)

    uneq_spin = -23.504907495268824 * dchi * delta * eta2

    return no_spin + eq_spin + uneq_spin


def imr_phenom_x_ringdown_phase_22_d24(eta: float, s: float, dchi: float, delta: float, rd_phase_flag: int) -> float:

    # /*
    #     Effective Spin Used: STotR.
    # */

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    s2 = s * s
    s3 = s2 * s

    checkify.check(
        rd_phase_flag == 105,
        "Error in IMRPhenomX_Ringdown_Phase_22_d24: "
        "IMRPhenomXRingdownPhaseVersion is not valid. Recommended flag is 105.",
    )

    no_spin = (eta * (-9.460253118496386 + 9.429314399633007 * eta + 64.69109972468395 * eta2)) / (
        -0.0670554310666559 - 0.09987544893382533 * eta + 1.0 * eta2
    )

    eq_spin = (
        17.36495157980372 * eta * s
        + eta3 * s * (930.3458437154668 + 808.457330742532 * s)
        + eta4 * s * (-774.3633787391745 - 2177.554979351284 * s - 1031.846477275069 * s2)
        + eta2 * s * (-191.00932194869588 - 62.997389062600035 * s + 64.42947340363101 * s2)
        + 0.04497628581617564 * s3
    ) / (1.0 - 0.7267610313751913 * s)

    uneq_spin = dchi * delta * (-36.66374091965371 + 91.60477826830407 * eta) * eta2

    return no_spin + eq_spin + uneq_spin


def imr_phenom_x_ringdown_phase_22_d34(eta: float, s: float, dchi: float, delta: float, rd_phase_flag: int) -> float:

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta5 = eta3 * eta2

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s

    checkify.check(
        rd_phase_flag == 105,
        "Error in IMRPhenomX_Ringdown_Phase_22_d34: "
        "IMRPhenomXRingdownPhaseVersion is not valid. Recommended flag is 105.",
    )

    no_spin = (eta * (-8.506898502692536 + 13.936621412517798 * eta)) / (-0.40919671232073945 + 1.0 * eta)

    eq_spin = (
        eta * (1.7280582989361533 * s + 18.41570325463385 * s3 - 13.743271480938104 * s4)
        + eta2 * (73.8367329022058 * s - 95.57802408341716 * s3 + 215.78111099820157 * s4)
        + 0.046849371468156265 * s2
        + eta3 * s * (-27.976989112929353 + 6.404060932334562 * s - 633.1966645925428 * s3 + 109.04824706217418 * s2)
    ) / (1.0 - 0.6862449113932192 * s)

    uneq_spin = 641.8965762829259 * dchi * delta * eta5

    return no_spin + eq_spin + uneq_spin


def imr_phenom_x_ringdown_phase_22_d54(eta: float, s: float, dchi: float, delta: float, rd_phase_flag: int) -> float:

    eta2 = eta * eta
    eta3 = eta2 * eta

    checkify.check(
        rd_phase_flag == 105,
        "Error in IMRPhenomX_Ringdown_Phase_22_d34: "
        "IMRPhenomXRingdownPhaseVersion is not valid. Recommended flag is 105.",
    )

    no_spin = (eta * (7.05731400277692 + 22.455288821807095 * eta + 119.43820622871043 * eta2)) / (
        0.26026709603623255 + 1.0 * eta
    )

    eq_spin = (
        eta2 * (134.88158268621922 - 56.05992404859163 * s) * s
        + eta * s * (-7.9407123129681425 + 9.486783128047414 * s)
        + eta3 * s * (-316.26970506215554 + 90.31815139272628 * s)
    ) / (1.0 - 0.7162058321905909 * s)

    uneq_spin = 43.82713604567481 * dchi * delta * eta3

    return no_spin + eq_spin + uneq_spin


def imr_phenom_x_ringdown_phase_22_ansatz(
        ff: float,
        powers_of_f: IMRPhenomXUsefulPowersDataClass,
        p_wf: IMRPhenomXWaveformDataClass,
        p_phase: IMRPhenomXPhaseCoefficientsDataClass,
) -> float | Array:
    """
    Phenomenological ringdown phase derivative ansatz:
    a_0 + a_1 f^(-1) + a_2 f^(-2) + a_3 f^(-3) + a_4 f^(-4) + ( aRD ) / ( (f_damp^2 + (f - f_ring)^2 ) )
    where a_5 = - dphase0 * aRD
    The canonical ringdown ansatz used here sets a_3 = 0.
	See Eq. 7.11 of arXiv:2001.11412.
    """
  
    rd_phase_flag = p_wf.imr_phenom_x_ringdown_phase_version

    # //invf    = powers_of_f->m_one
    invf2   = powers_of_f.m_two
    invf4   = powers_of_f.m_four
    invf1o3 = powers_of_f.m_one_third

    frd     = p_wf.f_ring
    fda     = p_wf.f_damp

    # // c0 = a0, c1 = a1, c2 = a2, c3 = a4 are the polynomial Coefficients
    # // c4 = a_L = -(dphase0 * a_RD) is the Lorentzian coefficient.

    checkify.check(
        rd_phase_flag == 105,
        "Error in IMRPhenomX_Ringdown_Phase_22_Ansatz: IMRPhenomXRingdownPhaseVersion is not valid.",
    )

    return p_phase.c0 + p_phase.c1*invf1o3 + p_phase.c2*invf2 + p_phase.c4*invf4 + ( p_phase.c_l / (fda*fda + (ff - frd)*(ff - frd)) ) 

    
def imr_phenom_x_ringdown_phase_22_ansatz_int(
        ff: float,
        powers_of_f: IMRPhenomXUsefulPowersDataClass,
        p_wf: IMRPhenomXWaveformDataClass,
        p_phase: IMRPhenomXPhaseCoefficientsDataClass,
) -> float | Array:
    """
    Phenomenological ringdown phase ansatz (i.e. integral of phase derivative ansatz). See. Eq. 7.11 of arxiv:2001.11412.
    """
    rd_phase_flag = p_wf.imr_phenom_x_ringdown_phase_version

    invf     = powers_of_f.m_one
    invf3    = powers_of_f.m_three
    # //logf     = powers_of_f->log
    f2o3     = powers_of_f.two_thirds

    frd      = p_wf.f_ring
    fda      = p_wf.f_damp

    c0       = p_phase.c0
    c1       = p_phase.c1
    c2       = p_phase.c2
    c4ov3    = p_phase.c4ov3
    c_lovfda  = p_phase.c_lovfda

    checkify.check(
        rd_phase_flag == 105,
        "Error in IMRPhenomX_Ringdown_Phase_22_AnsatzInt: IMRPhenomXRingdownPhaseVersion is not valid.",
    )

    return c0*ff + 1.5*c1*f2o3 - c2*invf - c4ov3*invf3 + (c_lovfda * jnp.arctan( (ff - frd )/fda ) )
