import jax.numpy as jnp

def IMRPhenomX_Ringdown_Amp_22_gamma2(eta: float, S: float, dchi: float, delta: float, RDAmpFlag: int) -> float:
    """
    Phenomenological Ringdown Amplitude Coefficient, gamma_2 (lambda in arXiv:2001.11412).

    See Section VI.C of arXiv:2001.11412.
    Effective Spin Used: STotR.

    Args:
        eta: Symmetric mass ratio
        S: Effective spin parameter (STotR)
        dchi: Spin difference parameter
        delta: Mass asymmetry parameter
        RDAmpFlag: Ringdown amplitude version flag

    Returns:
        gamma2 coefficient
    """
    S2 = S * S
    eta2 = eta * eta

    if RDAmpFlag == 103:
        noSpin = (
            (0.8312293675316895 + 7.480371544268765*eta - 18.256121237800397*eta2)
            / (1. + 10.915453595496611*eta - 30.578409433912874*eta2)
        )

        eqSpin = (
            (S*(0.5869408584532747 + eta*(-0.1467158405070222 - 2.8489481072076472*S)
                + 0.031852563636196894*S + eta2*(0.25295441250444334 + 4.6849496672664594*S)))
            / (3.8775263105069953 - 3.41755361841226*S + 1.*S2)
        )

        uneqSpin = -0.00548054788508203*dchi*delta*eta

        return noSpin + eqSpin + uneqSpin
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Amp_22_gamma2: RDAmpFlag={RDAmpFlag} is not valid. "
            f"Recommended flag is 103."
        )

def IMRPhenomX_Ringdown_Amp_22_gamma3(eta: float, S: float, dchi: float, delta: float, RDAmpFlag: int) -> float:
    """
    Phenomenological Ringdown Amplitude Coefficient, gamma_3 (sigma in arXiv:2001.11412).

    See Section VI.C of arXiv:2001.11412.
    Effective Spin: STotR.

    Args:
        eta: Symmetric mass ratio
        S: Effective spin parameter (STotR)
        dchi: Spin difference parameter
        delta: Mass asymmetry parameter
        RDAmpFlag: Ringdown amplitude version flag

    Returns:
        gamma3 coefficient
    """
    eta2 = eta * eta
    eta3 = eta2 * eta

    if RDAmpFlag == 103:
        noSpin = (
            (1.3666000000000007 - 4.091333144596439*eta + 2.109081209912545*eta2 - 4.222259944408823*eta3)
            / (1. - 2.7440263888207594*eta)
        )

        eqSpin = (
            (0.07179105336478316 + eta2*(2.331724812782498 - 0.6330998412809531*S)
             + eta*(-0.8752427297525086 + 0.4168560229353532*S) - 0.05633734476062242*S)*S
        )

        uneqSpin = 0. * delta * dchi  # Explicitly zero in the C code

        return noSpin + eqSpin + uneqSpin
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Amp_22_gamma3: RDAmpFlag={RDAmpFlag} is not valid. "
            f"Recommended flag is 103."
        )



def IMRPhenomX_Ringdown_Amp_22_v1(eta: float, S: float, dchi: float, delta: float, RDAmpFlag: int) -> float:
    """
    Collocation point for ringdown amplitude evaluated at F1 = f_peak.

    Effective Spin: STotR.

    Args:
        eta: Symmetric mass ratio
        S: Effective spin parameter (STotR)
        dchi: Spin difference parameter
        delta: Mass asymmetry parameter
        RDAmpFlag: Ringdown amplitude version flag

    Returns:
        v1 collocation point value
    """
    eta2 = eta * eta
    eta3 = eta2 * eta

    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S

    if RDAmpFlag == 103:
        noSpin = (
            (0.03689164742964719 + 25.417967754401182*eta + 162.52904393600332*eta2)
            / (1. + 61.19874463331437*eta - 29.628854485544874*eta2)
        )

        eqSpin = (
            (S*(-0.14352506969368556 + 0.026356911108320547*S + 0.19967405175523437*S2
                - 0.05292913111731128*S3
                + eta3*(-48.31945248941757 - 3.751501972663298*S + 81.9290740950083*S2
                       + 30.491948143930266*S3 - 132.77982622925845*S4)
                + eta*(-4.805034453745424 + 1.11147906765112*S + 6.176053843938542*S2
                      - 0.2874540719094058*S3 - 8.990840289951514*S4)
                - 0.18147275151697131*S4
                + eta2*(27.675454081988036 - 2.398327419614959*S - 47.99096500250743*S2
                       - 5.104257870393138*S3 + 72.08174136362386*S4)))
            / (-1.4160870461211452 + 1.*S)
        )

        uneqSpin = -0.04426571511345366*dchi*delta*eta2

        return noSpin + eqSpin + uneqSpin
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Amp_22_v1: RDAmpFlag={RDAmpFlag} is not valid. "
            f"Recommended flag is 103."
        )



def IMRPhenomX_Ringdown_Amp_22_PeakFrequency(gamma2: float, gamma3: float, fRING: float, fDAMP: float, RDAmpFlag: int) -> float:
    """
    Frequency of amplitude peak (f_peak).

    See Eq. 20 of 1508.07253 or Eq. 5.14 in arXiv:2001.11412.
    Abs[fring + fdamp * gamma3 * (Sqrt[1 - gamma2^2] - 1)/gamma2]

    Args:
        gamma2: Ringdown amplitude coefficient gamma2
        gamma3: Ringdown amplitude coefficient gamma3
        fRING: Ring-down frequency
        fDAMP: Damping frequency
        RDAmpFlag: Ringdown amplitude version flag

    Returns:
        Peak frequency f_peak
    """
    

    if RDAmpFlag == 103:
        # If gamma2 > 1, then the square root term can become imaginary. Set this term to zero.
        if gamma2 <= 1.0:
            fpeak = jnp.abs(fRING + fDAMP * gamma3 * (jnp.sqrt(1.0 - gamma2 * gamma2) - 1.0) / gamma2)
        else:
            fpeak = jnp.abs(fRING + fDAMP*(-1.0)*gamma3/gamma2)
        return fpeak
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Amp_22_PeakFrequency: RDAmpFlag={RDAmpFlag} is not valid. "
            f"Recommended flag is 103."
        )


def IMRPhenomX_Ringdown_Amp_22_Ansatz(ff: float, pWF: dict, pAmp: dict) -> float:
    """
    Phenomenological Ringdown Amplitude Ansatz (deformed Lorentzian).

    See Eq. 6.17 of arXiv:2001.11412.

    Args:
        ff: Frequency
        pWF: Waveform parameter structure
        pAmp: Amplitude coefficient structure

    Returns:
        Ringdown amplitude value
    """
    

    RDAmpFlag = pWF.get('IMRPhenomXRingdownAmpVersion', 103)

    if RDAmpFlag == 103:
        gammaR = pAmp['gammaR']      # gamma2 / (gamma3 * fDAMP)
        gammaD13 = pAmp['gammaD13']  # fDAMP * gamma1 * gamma3
        gammaD2 = pAmp['gammaD2']    # (fDAMP * gamma3)^2

        dfr = ff - pWF['fRING']

        # Switch to only doubles in the expression
        ampRD = jnp.exp(- dfr * gammaR) * gammaD13 / (dfr*dfr + gammaD2)

        return ampRD
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Amp_22_Ansatz: RDAmpFlag={RDAmpFlag} is not valid. "
            f"Recommended flag is 103."
        )

def IMRPhenomX_Ringdown_Amp_22_DAnsatz(ff: float, pWF: dict, pAmp: dict) -> float:
    """
    Derivative (with respect to f) of Phenomenological Ringdown Amplitude Ansatz.

    See Eq. 6.17 of arXiv:2001.11412.

    Args:
        ff: Frequency
        pWF: Waveform parameter structure
        pAmp: Amplitude coefficient structure

    Returns:
        Derivative of ringdown amplitude
    """
    

    RDAmpFlag = pWF.get('IMRPhenomXRingdownAmpVersion', 103)

    if RDAmpFlag == 103:
        g1 = pAmp['gamma1']
        g2 = pAmp['gamma2']
        g3 = pAmp['gamma3']

        frd = pWF['fRING']
        fda = pWF['fDAMP']
        dfr = ff - frd
        dfd = fda * g3

        prefactor = - jnp.exp(- g2 * dfr / dfd) * g1
        numerator = (dfr*dfr*g2 + 2.0*fda*dfr*g3 + fda*fda*g2*g3*g3)
        denominator = (dfr*dfr + dfd*dfd) * (dfr*dfr + dfd*dfd)
        DampRD = prefactor * numerator / denominator

        return DampRD
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Amp_22_DAnsatz: RDAmpFlag={RDAmpFlag} is not valid. "
            f"Recommended flag is 103."
        )


# ============================================================================
# Phase Functions
# ============================================================================

def IMRPhenomX_Ringdown_Phase_22_v4(eta: float, S: float, dchi: float, delta: float, RDPhaseFlag: int) -> float:
    """
    Collocation point for ringdown phase evaluated at f_ring = f_4.

    See Section VII.C of arXiv:2001.11412.
    Effective Spin Used: STotR.

    Args:
        eta: Symmetric mass ratio
        S: Effective spin parameter (STotR)
        dchi: Spin difference parameter
        delta: Mass asymmetry parameter
        RDPhaseFlag: Ringdown phase version flag

    Returns:
        v4 phase collocation point value
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta

    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S

    if RDPhaseFlag == 105:
        noSpin = (
            (-85.86062966719405 - 4616.740713893726*eta - 4925.756920247186*eta2
             + 7732.064464348168*eta3 + 12828.269960300782*eta4 - 39783.51698102803*eta5)
            / (1. + 50.206318806624004*eta)
        )

        eqSpin = (
            (S*(33.335857451144356 - 36.49019206094966*S
                + eta3*(1497.3545918387515 - 101.72731770500685*S)*S - 3.835967351280833*S2
                + 2.302712009652155*S3
                + eta2*(93.64156367505917 - 18.184492163348665*S + 423.48863373726243*S2
                       - 104.36120236420928*S3 - 719.8775484010988*S4) + 1.6533417657003922*S4
                + eta*(-69.19412903018717 + 26.580344399838758*S - 15.399770764623746*S2
                      + 31.231253209893488*S3 + 97.69027029734173*S4)
                + eta4*(1075.8686153198323 - 3443.0233614187396*S - 4253.974688619423*S2
                       - 608.2901586790335*S3 + 5064.173605639933*S4)))
            / (-1.3705601055555852 + 1.*S)
        )

        uneqSpin = dchi*delta*eta*(22.363215261437862 + 156.08206945239374*eta)

        return noSpin + eqSpin + uneqSpin
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Phase_22_v4: RDPhaseFlag={RDPhaseFlag} is not valid. "
            f"Recommended flag is 105."
        )


def IMRPhenomX_Ringdown_Phase_22_d12(eta: float, S: float, dchi: float, delta: float, RDPhaseFlag: int) -> float:
    """
    Difference between collocation points 1 and 2 (d12 = v1 - v2).

    See Section VII.C of arXiv:2001.11412.
    Effective Spin Used: STotR.

    Args:
        eta: Symmetric mass ratio
        S: Effective spin parameter (STotR)
        dchi: Spin difference parameter
        delta: Mass asymmetry parameter
        RDPhaseFlag: Ringdown phase version flag

    Returns:
        d12 phase collocation difference
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    S5 = S4 * S

    if RDPhaseFlag == 105:
        noSpin = (
            (eta*(0.7207992174994245 - 1.237332073800276*eta + 6.086871214811216*eta2))
            / (0.006851189888541745 + 0.06099184229137391*eta - 0.15500218299268662*eta2 + 1.*eta3)
        )

        eqSpin = (
            ((0.06519048552628343 - 25.25397971063995*eta - 308.62513664956975*eta4
              + 58.59408241189781*eta2 + 160.14971486043524*eta3)*S
             + eta*(-5.215945111216946 + 153.95945758807616*eta - 693.0504179144295*eta2
                   + 835.1725103648205*eta3)*S2
             + (0.20035146870472367 - 0.28745205203100666*eta - 47.56042058800358*eta4)*S3
             + eta*(5.7756520242745735 - 43.97332874253772*eta + 338.7263666984089*eta3)*S4
             + (-0.2697933899920511 + 4.917070939324979*eta - 22.384949087140086*eta4
               - 11.61488280763592*eta2)*S5)
            / (1. - 0.6628745847248266*S)
        )

        uneqSpin = -23.504907495268824*dchi*delta*eta2

        return noSpin + eqSpin + uneqSpin
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Phase_22_d12: RDPhaseFlag={RDPhaseFlag} is not valid. "
            f"Recommended flag is 105."
        )


def IMRPhenomX_Ringdown_Phase_22_d24(eta: float, S: float, dchi: float, delta: float, RDPhaseFlag: int) -> float:
    """
    Difference between collocation points 2 and 4 (d24 = v2 - v4).

    See Section VII.C of arXiv:2001.11412.
    Effective Spin Used: STotR.

    Args:
        eta: Symmetric mass ratio
        S: Effective spin parameter (STotR)
        dchi: Spin difference parameter
        delta: Mass asymmetry parameter
        RDPhaseFlag: Ringdown phase version flag

    Returns:
        d24 phase collocation difference
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    S2 = S * S
    S3 = S2 * S

    if RDPhaseFlag == 105:
        noSpin = (
            (eta*(-9.460253118496386 + 9.429314399633007*eta + 64.69109972468395*eta2))
            / (-0.0670554310666559 - 0.09987544893382533*eta + 1.*eta2)
        )

        eqSpin = (
            (17.36495157980372*eta*S + eta3*S*(930.3458437154668 + 808.457330742532*S)
             + eta4*S*(-774.3633787391745 - 2177.554979351284*S - 1031.846477275069*S2)
             + eta2*S*(-191.00932194869588 - 62.997389062600035*S + 64.42947340363101*S2)
             + 0.04497628581617564*S3)
            / (1. - 0.7267610313751913*S)
        )

        uneqSpin = dchi*delta*(-36.66374091965371 + 91.60477826830407*eta)*eta2

        return noSpin + eqSpin + uneqSpin
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Phase_22_d24: RDPhaseFlag={RDPhaseFlag} is not valid. "
            f"Recommended flag is 105."
        )


def IMRPhenomX_Ringdown_Phase_22_d34(eta: float, S: float, dchi: float, delta: float, RDPhaseFlag: int) -> float:
    """
    Difference between collocation points 3 and 4 (d34 = v3 - v4).

    See Section VII.C of arXiv:2001.11412.
    Effective Spin Used: STotR.

    Args:
        eta: Symmetric mass ratio
        S: Effective spin parameter (STotR)
        dchi: Spin difference parameter
        delta: Mass asymmetry parameter
        RDPhaseFlag: Ringdown phase version flag

    Returns:
        d34 phase collocation difference
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta5 = eta3 * eta2

    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S

    if RDPhaseFlag == 105:
        noSpin = (
            (eta*(-8.506898502692536 + 13.936621412517798*eta))
            / (-0.40919671232073945 + 1.*eta)
        )

        eqSpin = (
            (eta*(1.7280582989361533*S + 18.41570325463385*S3 - 13.743271480938104*S4)
             + eta2*(73.8367329022058*S - 95.57802408341716*S3 + 215.78111099820157*S4)
             + 0.046849371468156265*S2
             + eta3*S*(-27.976989112929353 + 6.404060932334562*S - 633.1966645925428*S3
                      + 109.04824706217418*S2))
            / (1. - 0.6862449113932192*S)
        )

        uneqSpin = 641.8965762829259*dchi*delta*eta5

        return noSpin + eqSpin + uneqSpin
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Phase_22_d34: RDPhaseFlag={RDPhaseFlag} is not valid. "
            f"Recommended flag is 105."
        )


def IMRPhenomX_Ringdown_Phase_22_d54(eta: float, S: float, dchi: float, delta: float, RDPhaseFlag: int) -> float:
    """
    Difference between collocation points 5 and 4 (d54 = v5 - v4).

    See Section VII.C of arXiv:2001.11412.
    Effective Spin Used: STotR.

    Args:
        eta: Symmetric mass ratio
        S: Effective spin parameter (STotR)
        dchi: Spin difference parameter
        delta: Mass asymmetry parameter
        RDPhaseFlag: Ringdown phase version flag

    Returns:
        d54 phase collocation difference
    """
    eta2 = eta * eta
    eta3 = eta2 * eta

    if RDPhaseFlag == 105:
        noSpin = (
            (eta*(7.05731400277692 + 22.455288821807095*eta + 119.43820622871043*eta2))
            / (0.26026709603623255 + 1.*eta)
        )

        eqSpin = (
            (eta2*(134.88158268621922 - 56.05992404859163*S)*S
             + eta*S*(-7.9407123129681425 + 9.486783128047414*S)
             + eta3*S*(-316.26970506215554 + 90.31815139272628*S))
            / (1. - 0.7162058321905909*S)
        )

        uneqSpin = 43.82713604567481*dchi*delta*eta3

        return noSpin + eqSpin + uneqSpin
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Phase_22_d54: RDPhaseFlag={RDPhaseFlag} is not valid. "
            f"Recommended flag is 105."
        )


def IMRPhenomX_Ringdown_Phase_22_Ansatz(ff: float, powers_of_f: 'IMRPhenomX_UsefulPowers', pWF: dict, pPhase: dict) -> float:
    """
    Phenomenological ringdown phase derivative ansatz.

    a_0 + a_1 f^(-1) + a_2 f^(-2) + a_3 f^(-3) + a_4 f^(-4) + ( aRD ) / ( (f_damp^2 + (f - f_ring)^2 ) )

    where a_5 = - dphase0 * aRD

    The canonical ringdown ansatz used here sets a_3 = 0.

    See Eq. 7.11 of arXiv:2001.11412.

    Args:
        ff: Frequency
        powers_of_f: Pre-computed powers of frequency
        pWF: Waveform parameter structure
        pPhase: Phase coefficient structure

    Returns:
        Phase derivative value
    """
    RDPhaseFlag = pWF.get('IMRPhenomXRingdownPhaseVersion', 105)

    if RDPhaseFlag == 105:
        invf2 = powers_of_f.m_two
        invf4 = powers_of_f.m_four
        invf1o3 = powers_of_f.m_one_third

        frd = pWF['fRING']
        fda = pWF['fDAMP']

        # c0 = a0, c1 = a1, c2 = a2, c3 = a4 are the polynomial Coefficients
        # c4 = a_L = -(dphase0 * a_RD) is the Lorentzian coefficient.
        phaseRD = (
            pPhase['c0'] + pPhase['c1']*invf1o3 + pPhase['c2']*invf2
            + pPhase['c4']*invf4 + (pPhase['cL'] / (fda*fda + (ff - frd)*(ff - frd)))
        )

        return phaseRD
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Phase_22_Ansatz: RDPhaseFlag={RDPhaseFlag} is not valid. "
            f"Recommended flag is 105."
        )


def IMRPhenomX_Ringdown_Phase_22_AnsatzInt(ff: float, powers_of_f: 'IMRPhenomX_UsefulPowers', pWF: dict, pPhase: dict) -> float:
    """
    Phenomenological ringdown phase ansatz (i.e. integral of phase derivative ansatz).

    See Eq. 7.11 of arXiv:2001.11412.

    Args:
        ff: Frequency
        powers_of_f: Pre-computed powers of frequency
        pWF: Waveform parameter structure
        pPhase: Phase coefficient structure

    Returns:
        Integrated phase value
    """
    

    RDPhaseFlag = pWF.get('IMRPhenomXRingdownPhaseVersion', 105)

    if RDPhaseFlag == 105:
        invf = powers_of_f.m_one
        invf3 = powers_of_f.m_three
        f2o3 = powers_of_f.two_thirds

        frd = pWF['fRING']
        fda = pWF['fDAMP']

        c0 = pPhase['c0']
        c1 = pPhase['c1']
        c2 = pPhase['c2']
        c4ov3 = pPhase['c4ov3']
        cLovfda = pPhase['cLovfda']

        phaseRDInt = (
            c0*ff + 1.5*c1*f2o3 - c2*invf - c4ov3*invf3
            + (cLovfda * jnp.arctan((ff - frd) / fda))
        )

        return phaseRDInt
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Ringdown_Phase_22_AnsatzInt: RDPhaseFlag={RDPhaseFlag} is not valid. "
            f"Recommended flag is 105."
        )
