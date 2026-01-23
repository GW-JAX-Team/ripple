import jax.numpy as jnp

def IMRPhenomX_Intermediate_Amp_22_v2(eta: float, STotR: float, dchi: float, delta: float, version: int) -> float:
    """
    Intermediate amplitude v2 coefficient for IMRPhenomX waveform model.

    Translated from LALSimIMRPhenomX_intermediate.c

    Parameters
    ----------
    eta : float
        Symmetric mass ratio
    STotR : float
        Total aligned spin parameter
    dchi : float
        Spin difference parameter
    delta : float
        Mass difference parameter
    version : int
        Intermediate amplitude flag (should be 105 or 1043)

    Returns
    -------
    float
        Intermediate amplitude value
    """
    eta2 = eta * eta
    eta3 = eta2 * eta

    S2 = STotR * STotR

    # Compute components for valid flags (105 or 1043)
    noSpin = (2.2436523786378983 + 2162.4749081764216*eta + 24460.158604784723*eta2 - 12112.140570900956*eta3) / \
             (1. + 120.78623282522702*eta + 416.4179522274108*eta2)

    eqSpin = (STotR * (6.727511603827924 + eta2*(414.1400701039126 - 234.3754066885935*STotR) - 5.399284768639545*STotR +
                       eta*(-186.87972530996245 + 128.7402290554767*STotR))) / \
             (3.24359204029217 - 3.975650468231452*STotR + 1.*S2)

    uneqSpin = dchi * delta * (-59.52510939953099 + 13.12679437100751*eta) * eta2

    return noSpin + eqSpin + uneqSpin

def IMRPhenomX_Intermediate_Amp_22_v3(eta: float, STotR: float, dchi: float, delta: float, version: int) -> float:
    """
    Intermediate amplitude v3 coefficient for IMRPhenomX waveform model.

    Translated from LALSimIMRPhenomX_intermediate.c

    Parameters
    ----------
    eta : float
        Symmetric mass ratio
    STotR : float
        Total aligned spin parameter
    dchi : float
        Spin difference parameter
    delta : float
        Mass difference parameter
    version : int
        Intermediate amplitude flag (should be 105 or 1043)

    Returns
    -------
    float
        Intermediate amplitude value
    """
    eta2 = eta * eta
    eta3 = eta2 * eta

    S2 = STotR * STotR

    # Compute components for valid flags (105 or 1043)
    noSpin = (1.195392410912163 + 1677.2558976605421*eta + 24838.37133975971*eta2 - 17277.938868280915*eta3) / \
             (1. + 144.78606839716073*eta + 428.8155916011666*eta2)

    eqSpin = (STotR * (-2.1413952025647793 + 0.5719137940424858*STotR +
                       eta*(46.61350006858767 + 0.40917927503842105*STotR - 11.526500209146906*S2) +
                       1.1833965566688387*S2 +
                       eta2*(-84.82318288272965 - 34.90591158988979*STotR + 19.494962340530186*S2))) / \
             (-1.4786392693666195 + 1.*STotR)

    uneqSpin = dchi * delta * (-333.7662575986524 + 532.2475589084717*eta) * eta3

    return noSpin + eqSpin + uneqSpin

def IMRPhenomX_Intermediate_Amp_22_vA(eta: float, STotR: float, dchi: float, delta: float, version: int) -> float:
    """
    Intermediate amplitude vA coefficient for IMRPhenomX waveform model.

    Translated from LALSimIMRPhenomX_intermediate.c

    Parameters
    ----------
    eta : float
        Symmetric mass ratio
    STotR : float
        Total aligned spin parameter
    dchi : float
        Spin difference parameter
    delta : float
        Mass difference parameter
    version : int
        Intermediate amplitude flag (should be 104)

    Returns
    -------
    float
        Intermediate amplitude value
    """
    eta2 = eta * eta
    eta3 = eta2 * eta

    S2 = STotR * STotR

    # Compute components for valid flag (104)
    noSpin = (1.4873184918202145 + 1974.6112656679577*eta + 27563.641024162127*eta2 - 19837.908020966777*eta3) / \
             (1. + 143.29004876335128*eta + 458.4097306093354*eta2)

    eqSpin = (STotR * (27.952730865904343 + eta*(-365.55631765202895 - 260.3494489873286*STotR) +
                       3.2646808851249016*STotR + 3011.446602208493*eta2*STotR - 19.38970173389662*S2 +
                       eta3*(1612.2681322644232 - 6962.675551371755*STotR + 1486.4658089990298*S2))) / \
             (12.647425554323242 - 10.540154508599963*STotR + 1.*S2)

    uneqSpin = dchi * delta * (-0.016404056649860943 - 296.473359655246*eta) * eta2

    return noSpin + eqSpin + uneqSpin




def IMRPhenomX_Intermediate_Amp_22_delta0(d1: float, d4: float, V1: float, V2: float, V3: float, V4: float,
                                          F1: float, F2: float, F3: float, F4: float, version: int) -> float:
    """
    Solve for intermediate amplitude delta0 coefficient.

    Solves system of equations for 5th order polynomial ansatz.
    Section VI.B, Eq. 6.5 of arXiv:2001.11412.

    Translated from LALSimIMRPhenomX_intermediate.c

    Parameters
    ----------
    d1, d4 : float
        Derivative constraints
    V1, V2, V3, V4 : float
        Amplitude values at collocation points
    F1, F2, F3, F4 : float
        Frequency collocation points
    version : int
        Intermediate amplitude flag (104, 105, or 1043)

    Returns
    -------
    float
        delta0 coefficient
    """
    f1, f2, f3, f4 = F1, F2, F3, F4
    v1, v2, v3, v4 = V1, V2, V3, V4

    if version == 1043:  # no left derivative; 1043 is used in IMRPhenomXHM
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f45 = f44 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf42 = f1mf4 * f1mf4
        f3mf42 = f3mf4 * f3mf4
        f2mf42 = f2mf4 * f2mf4

        retVal = (-(d4*f1*f1mf2*f1mf3*f1mf4*f2*f2mf3*f2mf4*f3*f3mf4*f4) - f1*f1mf3*f1mf42*f3*f3mf42*f42*v2 +
        f24*(-(f1*f1mf42*f42*v3) + f33*(f42*v1 + f12*v4 - 2*f1*f4*v4) + f3*f4*(f43*v1 + 2*f13*v4 - 3*f12*f4*v4) - f32*(2*f43*v1 + f13*v4 - 3*f1*f42*v4)) +
        f2*f4*(f12*f1mf42*f43*v3 - f34*(f43*v1 + 2*f13*v4 - 3*f12*f4*v4) - f32*f4*(f44*v1 + 3*f14*v4 - 4*f13*f4*v4) + 2*f33*(f44*v1 + f14*v4 - 2*f12*f42*v4)) +
        f22*(-(f1*f1mf42*(2*f1 + f4)*f43*v3) + f3*f42*(f44*v1 + 3*f14*v4 - 4*f13*f4*v4) + f34*(2*f43*v1 + f13*v4 - 3*f1*f42*v4) - f33*(3*f44*v1 + f14*v4 - 4*f1*f43*v4)) +
        f23*(f1*f1mf42*(f1 + 2*f4)*f42*v3 - f34*(f42*v1 + f12*v4 - 2*f1*f4*v4) + f32*(3*f44*v1 + f14*v4 - 4*f1*f43*v4) - 2*f3*(f45*v1 + f14*f4*v4 - 2*f12*f43*v4)))/(f1mf2*f1mf3*f1mf42*f2mf3*f2mf42*f3mf42)

    elif version == 104:
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = ((-(d4*f12*f1mf22*f1mf4*f2*f2mf4*f4) + d1*f1*f1mf2*f1mf4*f2*f2mf42*f42 + f42*(f2*f2mf42*(-4*f12 + 3*f1*f2 + 2*f1*f4 - f2*f4)*v1 + f12*f1mf43*v2) +
     f12*f1mf22*f2*(f1*f2 - 2*f1*f4 - 3*f2*f4 + 4*f42)*v4)/(f1mf22*f1mf43*f2mf42))

    elif version == 105:
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1
        f15 = f14 * f1
        f16 = f15 * f1
        f17 = f16 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2
        f25 = f24 * f2
        f26 = f25 * f2
        f27 = f26 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3
        f35 = f34 * f3
        f36 = f35 * f3
        f37 = f36 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f45 = f44 * f4
        f46 = f45 * f4
        f47 = f46 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf32 = f1mf3 * f1mf3
        f1mf42 = f1mf4 * f1mf4
        f2mf32 = f2mf3 * f2mf3
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = (
          (-(d4*f12*f1mf22*f1mf32*f1mf4*f2*f2mf3*f2mf4*f3*f3mf4*f4) - d1*f1*f1mf2*f1mf3*f1mf4*f2*f2mf3*f2mf42*f3*f3mf42*f42 + 5*f13*f24*f33*f42*v1 - 4*f12*f25*f33*f42*v1 - 5*f13*f23*f34*f42*v1 +
           3*f1*f25*f34*f42*v1 + 4*f12*f23*f35*f42*v1 - 3*f1*f24*f35*f42*v1 - 10*f13*f24*f32*f43*v1 + 8*f12*f25*f32*f43*v1 + 5*f12*f24*f33*f43*v1 - 4*f1*f25*f33*f43*v1 + 10*f13*f22*f34*f43*v1 -
           5*f12*f23*f34*f43*v1 - f25*f34*f43*v1 - 8*f12*f22*f35*f43*v1 + 4*f1*f23*f35*f43*v1 + f24*f35*f43*v1 + 5*f13*f24*f3*f44*v1 - 4*f12*f25*f3*f44*v1 + 15*f13*f23*f32*f44*v1 -
           10*f12*f24*f32*f44*v1 - f1*f25*f32*f44*v1 - 15*f13*f22*f33*f44*v1 + 5*f1*f24*f33*f44*v1 + 2*f25*f33*f44*v1 - 5*f13*f2*f34*f44*v1 + 10*f12*f22*f34*f44*v1 - 5*f1*f23*f34*f44*v1 +
           4*f12*f2*f35*f44*v1 + f1*f22*f35*f44*v1 - 2*f23*f35*f44*v1 - 10*f13*f23*f3*f45*v1 + 5*f12*f24*f3*f45*v1 + 2*f1*f25*f3*f45*v1 - f12*f23*f32*f45*v1 + 2*f1*f24*f32*f45*v1 - f25*f32*f45*v1 +
           10*f13*f2*f33*f45*v1 + f12*f22*f33*f45*v1 - 3*f24*f33*f45*v1 - 5*f12*f2*f34*f45*v1 - 2*f1*f22*f34*f45*v1 + 3*f23*f34*f45*v1 - 2*f1*f2*f35*f45*v1 + f22*f35*f45*v1 + 5*f13*f22*f3*f46*v1 +
           2*f12*f23*f3*f46*v1 - 4*f1*f24*f3*f46*v1 - 5*f13*f2*f32*f46*v1 - f1*f23*f32*f46*v1 + 2*f24*f32*f46*v1 - 2*f12*f2*f33*f46*v1 + f1*f22*f33*f46*v1 + 4*f1*f2*f34*f46*v1 - 2*f22*f34*f46*v1 -
           3*f12*f22*f3*f47*v1 + 2*f1*f23*f3*f47*v1 + 3*f12*f2*f32*f47*v1 - f23*f32*f47*v1 - 2*f1*f2*f33*f47*v1 + f22*f33*f47*v1 - f17*f33*f42*v2 + 2*f16*f34*f42*v2 - f15*f35*f42*v2 + 2*f17*f32*f43*v2 -
           f16*f33*f43*v2 - 4*f15*f34*f43*v2 + 3*f14*f35*f43*v2 - f17*f3*f44*v2 - 4*f16*f32*f44*v2 + 8*f15*f33*f44*v2 - 3*f13*f35*f44*v2 + 3*f16*f3*f45*v2 - 8*f14*f33*f45*v2 + 4*f13*f34*f45*v2 +
           f12*f35*f45*v2 - 3*f15*f3*f46*v2 + 4*f14*f32*f46*v2 + f13*f33*f46*v2 - 2*f12*f34*f46*v2 + f14*f3*f47*v2 - 2*f13*f32*f47*v2 + f12*f33*f47*v2 + f17*f23*f42*v3 - 2*f16*f24*f42*v3 +
           f15*f25*f42*v3 - 2*f17*f22*f43*v3 + f16*f23*f43*v3 + 4*f15*f24*f43*v3 - 3*f14*f25*f43*v3 + f17*f2*f44*v3 + 4*f16*f22*f44*v3 - 8*f15*f23*f44*v3 + 3*f13*f25*f44*v3 - 3*f16*f2*f45*v3 +
           8*f14*f23*f45*v3 - 4*f13*f24*f45*v3 - f12*f25*f45*v3 + 3*f15*f2*f46*v3 - 4*f14*f22*f46*v3 - f13*f23*f46*v3 + 2*f12*f24*f46*v3 - f14*f2*f47*v3 + 2*f13*f22*f47*v3 - f12*f23*f47*v3 +
           f12*f1mf22*f1mf32*f2*f2mf3*f3*(f4*(-3*f2*f3 + 4*(f2 + f3)*f4 - 5*f42) + f1*(f2*f3 - 2*(f2 + f3)*f4 + 3*f42))*v4)/(f1mf22*f1mf32*f1mf43*f2mf3*f2mf42*f3mf42)
        )

    else:
        raise ValueError(f"Error in IMRPhenomX_Intermediate_Amp_22_delta0: version {version} is not valid. Recommended flag is 104.")

    return retVal

def IMRPhenomX_Intermediate_Amp_22_delta1(d1: float, d4: float, V1: float, V2: float, V3: float, V4: float,
                                          F1: float, F2: float, F3: float, F4: float, version: int) -> float:
    """
    Solve for intermediate amplitude delta1 coefficient.

    Solves system of equations for 5th order polynomial ansatz.
    Section VI.B, Eq. 6.5 of arXiv:2001.11412.

    Translated from LALSimIMRPhenomX_intermediate.c

    Parameters
    ----------
    d1, d4 : float
        Derivative constraints
    V1, V2, V3, V4 : float
        Amplitude values at collocation points
    F1, F2, F3, F4 : float
        Frequency collocation points
    version : int
        Intermediate amplitude flag (104, 105, or 1043)

    Returns
    -------
    float
        delta1 coefficient
    """
    f1, f2, f3, f4 = F1, F2, F3, F4
    v1, v2, v3, v4 = V1, V2, V3, V4

    if version == 1043:  # no left derivative; 1043 is used in IMRPhenomXHM
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4

        v1mv2 = v1 - v2
        v2mv3 = v2 - v3
        v2mv4 = v2 - v4
        v1mv3 = v1 - v3
        v1mv4 = v1 - v4
        v3mv4 = v3 - v4

        retVal = (d4*f1mf4*f2mf4*f3mf4*(f1*f2*f3 + f2*f3*f4 + f1*(f2 + f3)*f4) + (f4*(f12*f1mf42*f43*v2mv3 + f34*(f43*v1mv2 +
          3*f12*f4*v2mv4 + 2*f13*(-v2 + v4)) + f32*f4*(f44*v1mv2 + 4*f13*f4*v2mv4 + 3*f14*(-v2 + v4)) + 2*f33*(f44*(-v1 + v2)
          + f14*v2mv4 + 2*f12*f42*(-v2 + v4)) + 2*f23*(f44*v1mv3 + f34*v1mv4 + 2*f12*f42*v3mv4 + 2*f32*f42*(-v1 + v4) + f14*(-v3 + v4))
          + f24*(3*f32*f4*v1mv4 + f43*(-v1 + v3) + 2*f13*v3mv4 + 2*f33*(-v1 + v4) + 3*f12*f4*(-v3 + v4)) + f22*f4*(4*f33*f4*v1mv4 + f44*(-v1 + v3)
          + 3*f14*v3mv4 + 3*f34*(-v1 + v4) + 4*f13*f4*(-v3 + v4))))/(f1mf2*f1mf3*f2mf3))/(f1mf42*f2mf42*f3mf42)

    elif version == 104:
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4

        f1mf2 = f1 - f2
        f1mf4 = f1 - f4
        f2mf4 = f2 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = ((d4*f1*f1mf22*f1mf4*f2mf4*(2*f2*f4 + f1*(f2 + f4)) + f4*(-(d1*f1mf2*f1mf4*f2mf42*(2*f1*f2 + (f1 + f2)*f4)) -
          2*f1*(f44*(v1 - v2) + 3*f24*(v1 - v4) + f14*(v2 - v4) + 4*f23*f4*(-v1 + v4)
          + 2*f13*f4*(-v2 + v4) + f1*(2*f43*(-v1 + v2) + 6*f22*f4*(v1 - v4) + 4*f23*(-v1 + v4)))))/(f1mf22*f1mf43*f2mf42))

    elif version == 105:
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1
        f15 = f14 * f1
        f16 = f15 * f1
        f17 = f16 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2
        f25 = f24 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3
        f35 = f34 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f45 = f44 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf32 = f1mf3 * f1mf3
        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = (
          (d4*f1*f1mf22*f1mf32*f1mf4*f2mf3*f2mf4*f3mf4*(f1*f2*f3 + 2*f2*f3*f4 + f1*(f2 + f3)*f4) +
          f4*(d1*f1mf2*f1mf3*f1mf4*f2mf3*f2mf42*f3mf42*(2*f1*f2*f3 + f2*f3*f4 + f1*(f2 + f3)*f4) +
          f1*(f16*(f43*(v2 - v3) + 2*f33*(v2 - v4) + 3*f22*f4*(v3 - v4) + 3*f32*f4*(-v2 + v4) + 2*f23*(-v3 + v4)) +
          f13*f4*(f45*(-v2 + v3) + 5*f34*f4*(v2 - v4) + 4*f25*(v3 - v4) + 4*f35*(-v2 + v4) + 5*f24*f4*(-v3 + v4)) +
          f14*(3*f45*(v2 - v3) + 2*f35*(v2 - v4) + 5*f34*f4*(v2 - v4) + 10*f23*f42*(v3 - v4) + 10*f33*f42*(-v2 + v4) + 2*f25*(-v3 + v4) + 5*f24*f4*(-v3 + v4)) +
          f15*(3*f44*(-v2 + v3) + 2*f33*f4*(v2 - v4) + 5*f32*f42*(v2 - v4) + 4*f24*(v3 - v4) + 4*f34*(-v2 + v4) + 2*f23*f4*(-v3 + v4) + 5*f22*f42*(-v3 + v4)) -
          5*f12*(-(f32*f3mf42*f43*(v1 - v2)) + 2*f23*(f44*(-v1 + v3) + 2*f32*f42*(v1 - v4) + f34*(-v1 + v4)) + f24*(f43*(v1 - v3) + 2*f33*(v1 - v4) + 3*f32*f4*(-v1 + v4)) +
          f22*f4*(f44*(v1 - v3) + 3*f34*(v1 - v4) + 4*f33*f4*(-v1 + v4))) +
          f1*(-(f32*f3mf42*(4*f3 + 3*f4)*f43*(v1 - v2)) + 2*f23*(f45*(-v1 + v3) + 5*f34*f4*(v1 - v4) + 4*f35*(-v1 + v4)) + 4*f25*(f43*(v1 - v3) + 2*f33*(v1 - v4) + 3*f32*f4*(-v1 + v4)) -
          5*f24*f4*(f43*(v1 - v3) + 2*f33*(v1 - v4) + 3*f32*f4*(-v1 + v4)) + 3*f22*f4*(f45*(v1 - v3) + 4*f35*(v1 - v4) + 5*f34*f4*(-v1 + v4))) -
          2*(-(f33*f3mf42*f44*(v1 - v2)) + f24*(2*f45*(-v1 + v3) + 5*f33*f42*(v1 - v4) + 3*f35*(-v1 + v4)) + f25*(f44*(v1 - v3) + 3*f34*(v1 - v4) + 4*f33*f4*(-v1 + v4)) +
          f23*f4*(f45*(v1 - v3) + 4*f35*(v1 - v4) + 5*f34*f4*(-v1 + v4))))))/(f1mf22*f1mf32*f1mf43*f2mf3*f2mf42*f3mf42)
        )

    else:
        raise ValueError(f"Error in IMRPhenomX_Intermediate_Amp_22_delta1: version {version} is not valid. Recommended flag is 104.")

    return retVal

def IMRPhenomX_Intermediate_Amp_22_delta2(d1: float, d4: float, V1: float, V2: float, V3: float, V4: float,
                                          F1: float, F2: float, F3: float, F4: float, version: int) -> float:
    """
    Solve for intermediate amplitude delta2 coefficient.

    Translated from LALSimIMRPhenomX_intermediate.c
    """
    f1, f2, f3, f4 = F1, F2, F3, F4
    v1, v2, v3, v4 = V1, V2, V3, V4

    if version == 1043:  # no left derivative: v1, v2, v3, v4, d4; 1043 is used in IMRPhenomXHM
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f46 = f44 * f42

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4

        v1mv2 = v1 - v2
        v2mv3 = v2 - v3
        v2mv4 = v2 - v4
        v1mv3 = v1 - v3
        v1mv4 = v1 - v4
        v3mv4 = v3 - v4

        retVal = (-(d4*f1mf2*f1mf3*f1mf4*f2mf3*f2mf4*f3mf4*(f3*f4 + f2*(f3 + f4) + f1*(f2 + f3 + f4))) - 2*f34*f43*v1 + 3*f33*f44*v1 - f3*f46*v1 - f14*f33*v2 + f13*f34*v2 + 3*f14*f3*f42*v2 - 3*f1*f34*f42*v2 -
        2*f14*f43*v2 - 4*f13*f3*f43*v2 + 4*f1*f33*f43*v2 + 2*f34*f43*v2 + 3*f13*f44*v2 - 3*f33*f44*v2 - f1*f46*v2 + f3*f46*v2 + 2*f14*f43*v3 - 3*f13*f44*v3 + f1*f46*v3 +
        f2*f42*(f44*v1mv3 + 3*f34*v1mv4 - 4*f33*f4*v1mv4 - 3*f14*v3mv4 + 4*f13*f4*v3mv4) + f24*(2*f43*v1mv3 + f33*v1mv4 - 3*f3*f42*v1mv4 - f13*v3mv4 + 3*f1*f42*v3mv4) +
        f23*(-3*f44*v1mv3 - f34*v1mv4 + 4*f3*f43*v1mv4 + f14*v3mv4 - 4*f1*f43*v3mv4) + f14*f33*v4 - f13*f34*v4 - 3*f14*f3*f42*v4 + 3*f1*f34*f42*v4 + 4*f13*f3*f43*v4 - 4*f1*f33*f43*v4)/\
        (f1mf2*f1mf3*f1mf42*f2mf3*f2mf42*f3mf42)

    elif version == 104:
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1
        f15 = f14 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f45 = f44 * f4

        f1mf2 = f1 - f2
        f1mf4 = f1 - f4
        f2mf4 = f2 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = ((-(d4*f1mf22*f1mf4*f2mf4*(f12 + f2*f4 + 2*f1*(f2 + f4))) + d1*f1mf2*f1mf4*f2mf42*(f1*f2 + 2*(f1 + f2)*f4 + f42)
        - 4*f12*f23*v1 + 3*f1*f24*v1 - 4*f1*f23*f4*v1 + 3*f24*f4*v1 + 12*f12*f2*f42*v1 -
       4*f23*f42*v1 - 8*f12*f43*v1 + f1*f44*v1 + f45*v1 + f15*v2 + f14*f4*v2 - 8*f13*f42*v2 + 8*f12*f43*v2 - f1*f44*v2 - f45*v2 -
       f1mf22*(f13 + f2*(3*f2 - 4*f4)*f4 + f12*(2*f2 + f4) + f1*(3*f2 - 4*f4)*(f2 + 2*f4))*v4)/(f1mf22*f1mf43*f2mf42))

    elif version == 105:
        # This case has extremely long expressions - implementing as in C
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1
        f15 = f14 * f1
        f16 = f15 * f1
        f17 = f16 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2
        f25 = f24 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3
        f35 = f34 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f45 = f44 * f4
        f46 = f45 * f4
        f47 = f46 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf32 = f1mf3 * f1mf3
        f1mf42 = f1mf4 * f1mf4
        f2mf32 = f2mf3 * f2mf3
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = (
          (-(d4*f1mf22*f1mf32*f1mf4*f2mf3*f2mf4*f3mf4*(f2*f3*f4 + f12*(f2 + f3 + f4) + 2*f1*(f2*f3 + (f2 + f3)*f4))) -
          d1*f1mf2*f1mf3*f1mf4*f2mf3*f2mf42*f3mf42*(f1*f2*f3 + 2*(f2*f3 + f1*(f2 + f3))*f4 + (f1 + f2 + f3)*f42) + 5*f13*f24*f33*v1 - 4*f12*f25*f33*v1 - 5*f13*f23*f34*v1 + 3*f1*f25*f34*v1 +
          4*f12*f23*f35*v1 - 3*f1*f24*f35*v1 + 5*f12*f24*f33*f4*v1 - 4*f1*f25*f33*f4*v1 - 5*f12*f23*f34*f4*v1 + 3*f25*f34*f4*v1 + 4*f1*f23*f35*f4*v1 - 3*f24*f35*f4*v1 - 15*f13*f24*f3*f42*v1 +
          12*f12*f25*f3*f42*v1 + 5*f1*f24*f33*f42*v1 - 4*f25*f33*f42*v1 + 15*f13*f2*f34*f42*v1 - 5*f1*f23*f34*f42*v1 - 12*f12*f2*f35*f42*v1 + 4*f23*f35*f42*v1 + 10*f13*f24*f43*v1 - 8*f12*f25*f43*v1 +
          20*f13*f23*f3*f43*v1 - 15*f12*f24*f3*f43*v1 - 20*f13*f2*f33*f43*v1 + 5*f24*f33*f43*v1 - 10*f13*f34*f43*v1 + 15*f12*f2*f34*f43*v1 - 5*f23*f34*f43*v1 + 8*f12*f35*f43*v1 - 15*f13*f23*f44*v1 +
          10*f12*f24*f44*v1 + f1*f25*f44*v1 + 15*f13*f33*f44*v1 - 10*f12*f34*f44*v1 - f1*f35*f44*v1 + f12*f23*f45*v1 - 2*f1*f24*f45*v1 + f25*f45*v1 - f12*f33*f45*v1 + 2*f1*f34*f45*v1 - f35*f45*v1 +
          5*f13*f2*f46*v1 + f1*f23*f46*v1 - 2*f24*f46*v1 - 5*f13*f3*f46*v1 - f1*f33*f46*v1 + 2*f34*f46*v1 - 3*f12*f2*f47*v1 + f23*f47*v1 + 3*f12*f3*f47*v1 - f33*f47*v1 - f17*f33*v2 + 2*f16*f34*v2 -
          f15*f35*v2 - f16*f33*f4*v2 + 2*f15*f34*f4*v2 - f14*f35*f4*v2 + 3*f17*f3*f42*v2 - f15*f33*f42*v2 - 10*f14*f34*f42*v2 + 8*f13*f35*f42*v2 - 2*f17*f43*v2 - 5*f16*f3*f43*v2 + 15*f14*f33*f43*v2 -
          8*f12*f35*f43*v2 + 4*f16*f44*v2 - 15*f13*f33*f44*v2 + 10*f12*f34*f44*v2 + f1*f35*f44*v2 + f12*f33*f45*v2 - 2*f1*f34*f45*v2 + f35*f45*v2 - 4*f14*f46*v2 + 5*f13*f3*f46*v2 + f1*f33*f46*v2 -
          2*f34*f46*v2 + 2*f13*f47*v2 - 3*f12*f3*f47*v2 + f33*f47*v2 + f17*f23*v3 - 2*f16*f24*v3 + f15*f25*v3 + f16*f23*f4*v3 - 2*f15*f24*f4*v3 + f14*f25*f4*v3 - 3*f17*f2*f42*v3 + f15*f23*f42*v3 +
          10*f14*f24*f42*v3 - 8*f13*f25*f42*v3 + 2*f17*f43*v3 + 5*f16*f2*f43*v3 - 15*f14*f23*f43*v3 + 8*f12*f25*f43*v3 - 4*f16*f44*v3 + 15*f13*f23*f44*v3 - 10*f12*f24*f44*v3 - f1*f25*f44*v3 -
          f12*f23*f45*v3 + 2*f1*f24*f45*v3 - f25*f45*v3 + 4*f14*f46*v3 - 5*f13*f2*f46*v3 - f1*f23*f46*v3 + 2*f24*f46*v3 - 2*f13*f47*v3 + 3*f12*f2*f47*v3 - f23*f47*v3 -
          f1mf22*f1mf32*f2mf3*(f13*(f22 + f2*f3 + f32 - 3*f42) + f2*f3*f4*(3*f2*f3 - 4*(f2 + f3)*f4 + 5*f42) + f1*(f2*f3 + 2*(f2 + f3)*f4)*(3*f2*f3 - 4*(f2 + f3)*f4 + 5*f42) +
          f12*(2*f2*f3*(f2 + f3) + (f22 + f2*f3 + f32)*f4 - 6*(f2 + f3)*f42 + 5*f43))*v4)/(f1mf22*f1mf32*f1mf43*f2mf3*f2mf42*f3mf42)
        )

    else:
        raise ValueError(f"Error in IMRPhenomX_Intermediate_Amp_22_delta2: version {version} is not valid. Recommended flag is 104.")

    return retVal

def IMRPhenomX_Intermediate_Amp_22_delta3(d1: float, d4: float, V1: float, V2: float, V3: float, V4: float,
                                          F1: float, F2: float, F3: float, F4: float, version: int) -> float:
    """
    Solve for intermediate amplitude delta3 coefficient.

    Translated from LALSimIMRPhenomX_intermediate.c
    """
    f1, f2, f3, f4 = F1, F2, F3, F4
    v1, v2, v3, v4 = V1, V2, V3, V4

    if version == 1043:  # no left derivative: v1, v2, v3, v4, d4; 1043 is used in IMRPhenomXHM
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f32 = f3 * f3
        f34 = f32 * f32

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f45 = f44 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4

        v1mv2 = v1 - v2
        v2mv3 = v2 - v3
        v2mv4 = v2 - v4
        v1mv3 = v1 - v3
        v1mv4 = v1 - v4
        v3mv4 = v3 - v4

        retVal = (d4*f1mf2*f1mf3*f1mf4*f2mf3*f2mf4*f3mf4*(f1 + f2 + f3 + f4) + f34*f42*v1 - 3*f32*f44*v1 + 2*f3*f45*v1 + f14*f32*v2 - f12*f34*v2 - 2*f14*f3*f4*v2 + 2*f1*f34*f4*v2 + f14*f42*v2 - f34*f42*v2 +
        4*f12*f3*f43*v2 - 4*f1*f32*f43*v2 - 3*f12*f44*v2 + 3*f32*f44*v2 + 2*f1*f45*v2 - 2*f3*f45*v2 - f14*f42*v3 + 3*f12*f44*v3 - 2*f1*f45*v3 +
        f24*(-(f42*v1mv3) - f32*v1mv4 + 2*f3*f4*v1mv4 + f12*v3mv4 - 2*f1*f4*v3mv4) - 2*f2*f4*(f44*v1mv3 + f34*v1mv4 - 2*f32*f42*v1mv4 - f14*v3mv4 + 2*f12*f42*v3mv4) +
        f22*(3*f44*v1mv3 + f34*v1mv4 - 4*f3*f43*v1mv4 - f14*v3mv4 + 4*f1*f43*v3mv4) - f14*f32*v4 + f12*f34*v4 + 2*f14*f3*f4*v4 - 2*f1*f34*f4*v4 - 4*f12*f3*f43*v4 + 4*f1*f32*f43*v4)/\
        (f1mf2*f1mf3*f1mf42*f2mf3*f2mf42*f3mf42)

    elif version == 104:
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4

        f1mf2 = f1 - f2
        f1mf4 = f1 - f4
        f2mf4 = f2 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = ((d4*f1mf22*f1mf4*f2mf4*(2*f1 + f2 + f4) - d1*f1mf2*f1mf4*f2mf42*(f1 + f2 + 2*f4)
                  + 2*(f44*(-v1 + v2) + 2*f12*f2mf42*(v1 - v4) + 2*f22*f42*(v1 - v4)
                  + 2*f13*f4*(v2 - v4) + f24*(-v1 + v4) + f14*(-v2 + v4) + 2*f1*f4*(f42*(v1 - v2) + f22*(v1 - v4) + 2*f2*f4*(-v1 + v4)))) / (f1mf22*f1mf43*f2mf42))

    elif version == 105:
        # Version 105 has extremely long expression
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1
        f15 = f14 * f1
        f16 = f15 * f1
        f17 = f16 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2
        f25 = f24 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3
        f35 = f34 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f45 = f44 * f4
        f46 = f45 * f4
        f47 = f46 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf32 = f1mf3 * f1mf3
        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = (
          (d4*f1mf22*f1mf32*f1mf4*f2mf3*f2mf4*f3mf4*(f12 + f2*f3 + (f2 + f3)*f4 + 2*f1*(f2 + f3 + f4)) + d1*f1mf2*f1mf3*f1mf4*f2mf3*f2mf42*f3mf42*(f1*f2 + f1*f3 + f2*f3 + 2*(f1 + f2 + f3)*f4 + f42) -
          5*f13*f24*f32*v1 + 4*f12*f25*f32*v1 + 5*f13*f22*f34*v1 - 2*f25*f34*v1 - 4*f12*f22*f35*v1 + 2*f24*f35*v1 + 10*f13*f24*f3*f4*v1 - 8*f12*f25*f3*f4*v1 - 5*f12*f24*f32*f4*v1 + 4*f1*f25*f32*f4*v1 -
          10*f13*f2*f34*f4*v1 + 5*f12*f22*f34*f4*v1 + 8*f12*f2*f35*f4*v1 - 4*f1*f22*f35*f4*v1 - 5*f13*f24*f42*v1 + 4*f12*f25*f42*v1 + 10*f12*f24*f3*f42*v1 - 8*f1*f25*f3*f42*v1 - 5*f1*f24*f32*f42*v1 +
          4*f25*f32*f42*v1 + 5*f13*f34*f42*v1 - 10*f12*f2*f34*f42*v1 + 5*f1*f22*f34*f42*v1 - 4*f12*f35*f42*v1 + 8*f1*f2*f35*f42*v1 - 4*f22*f35*f42*v1 - 5*f12*f24*f43*v1 + 4*f1*f25*f43*v1 -
          20*f13*f22*f3*f43*v1 + 10*f1*f24*f3*f43*v1 + 20*f13*f2*f32*f43*v1 - 5*f24*f32*f43*v1 + 5*f12*f34*f43*v1 - 10*f1*f2*f34*f43*v1 + 5*f22*f34*f43*v1 - 4*f1*f35*f43*v1 + 15*f13*f22*f44*v1 -
          5*f1*f24*f44*v1 - 2*f25*f44*v1 - 15*f13*f32*f44*v1 + 5*f1*f34*f44*v1 + 2*f35*f44*v1 - 10*f13*f2*f45*v1 - f12*f22*f45*v1 + 3*f24*f45*v1 + 10*f13*f3*f45*v1 + f12*f32*f45*v1 - 3*f34*f45*v1 +
          2*f12*f2*f46*v1 - f1*f22*f46*v1 - 2*f12*f3*f46*v1 + f1*f32*f46*v1 + 2*f1*f2*f47*v1 - f22*f47*v1 - 2*f1*f3*f47*v1 + f32*f47*v1 + f17*f32*v2 - 3*f15*f34*v2 + 2*f14*f35*v2 - 2*f17*f3*f4*v2 +
          f16*f32*f4*v2 + 5*f14*f34*f4*v2 - 4*f13*f35*f4*v2 + f17*f42*v2 - 2*f16*f3*f42*v2 + f15*f32*f42*v2 + f16*f43*v2 + 10*f15*f3*f43*v2 - 15*f14*f32*f43*v2 + 4*f1*f35*f43*v2 - 8*f15*f44*v2 +
          15*f13*f32*f44*v2 - 5*f1*f34*f44*v2 - 2*f35*f44*v2 + 8*f14*f45*v2 - 10*f13*f3*f45*v2 - f12*f32*f45*v2 + 3*f34*f45*v2 - f13*f46*v2 + 2*f12*f3*f46*v2 - f1*f32*f46*v2 - f12*f47*v2 +
          2*f1*f3*f47*v2 - f32*f47*v2 - f17*f22*v3 + 3*f15*f24*v3 - 2*f14*f25*v3 + 2*f17*f2*f4*v3 - f16*f22*f4*v3 - 5*f14*f24*f4*v3 + 4*f13*f25*f4*v3 - f17*f42*v3 + 2*f16*f2*f42*v3 - f15*f22*f42*v3 -
          f16*f43*v3 - 10*f15*f2*f43*v3 + 15*f14*f22*f43*v3 - 4*f1*f25*f43*v3 + 8*f15*f44*v3 - 15*f13*f22*f44*v3 + 5*f1*f24*f44*v3 + 2*f25*f44*v3 - 8*f14*f45*v3 + 10*f13*f2*f45*v3 + f12*f22*f45*v3 -
          3*f24*f45*v3 + f13*f46*v3 - 2*f12*f2*f46*v3 + f1*f22*f46*v3 + f12*f47*v3 - 2*f1*f2*f47*v3 + f22*f47*v3 +
          f1mf22*f1mf32*f2mf3*(2*f22*f32 + f13*(f2 + f3 - 2*f4) + f12*(f2 + f3 - 2*f4)*(2*(f2 + f3) + f4) - 4*(f22 + f2*f3 + f32)*f42 + 5*(f2 + f3)*f43 +
          f1*(4*f2*f3*(f2 + f3) - 4*(f22 + f2*f3 + f32)*f4 - 3*(f2 + f3)*f42 + 10*f43))*v4)/(f1mf22*f1mf32*f1mf43*f2mf3*f2mf42*f3mf42)
        )

    else:
        raise ValueError(f"Error in IMRPhenomX_Intermediate_Amp_22_delta3: version {version} is not valid. Recommended flag is 104.")

    return retVal

def IMRPhenomX_Intermediate_Amp_22_delta4(d1: float, d4: float, V1: float, V2: float, V3: float, V4: float,
                                          F1: float, F2: float, F3: float, F4: float, version: int) -> float:
    """
    Solve for intermediate amplitude delta4 coefficient.

    Translated from LALSimIMRPhenomX_intermediate.c
    """
    f1, f2, f3, f4 = F1, F2, F3, F4
    v1, v2, v3, v4 = V1, V2, V3, V4

    if version == 1043:  # no left derivative: v1, v2, v3, v4, d4; 1043 is used in IMRPhenomXHM
        f12 = f1 * f1
        f13 = f12 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f32 = f3 * f3
        f33 = f32 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4

        v1mv2 = v1 - v2
        v2mv3 = v2 - v3
        v2mv4 = v2 - v4
        v1mv3 = v1 - v3
        v1mv4 = v1 - v4
        v3mv4 = v3 - v4

        retVal = (-(d4*f1mf2*f1mf3*f1mf4*f2mf3*f2mf4*f3mf4) - f33*f42*v1 + 2*f32*f43*v1 - f3*f44*v1 - f13*f32*v2 + f12*f33*v2 + 2*f13*f3*f4*v2 - 2*f1*f33*f4*v2 - f13*f42*v2 - 3*f12*f3*f42*v2 + 3*f1*f32*f42*v2 +
        f33*f42*v2 + 2*f12*f43*v2 - 2*f32*f43*v2 - f1*f44*v2 + f3*f44*v2 + f13*f42*v3 - 2*f12*f43*v3 + f1*f44*v3 + f23*(f42*v1mv3 + f32*v1mv4 - 2*f3*f4*v1mv4 - f12*v3mv4 + 2*f1*f4*v3mv4) +
        f2*f4*(f43*v1mv3 + 2*f33*v1mv4 - 3*f32*f4*v1mv4 - 2*f13*v3mv4 + 3*f12*f4*v3mv4) + f22*(-2*f43*v1mv3 - f33*v1mv4 + 3*f3*f42*v1mv4 + f13*v3mv4 - 3*f1*f42*v3mv4) + f13*f32*v4 - f12*f33*v4 -
        2*f13*f3*f4*v4 + 2*f1*f33*f4*v4 + 3*f12*f3*f42*v4 - 3*f1*f32*f42*v4)/(f1mf2*f1mf3*f1mf42*f2mf3*f2mf42*f3mf42)

    elif version == 104:
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4

        f1mf2 = f1 - f2
        f1mf4 = f1 - f4
        f2mf4 = f2 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = ((-(d4*f1mf22*f1mf4*f2mf4) + d1*f1mf2*f1mf4*f2mf42 - 3*f1*f22*v1 + 2*f23*v1 + 6*f1*f2*f4*v1 - 3*f22*f4*v1
                - 3*f1*f42*v1 + f43*v1 + f13*v2 - 3*f12*f4*v2 + 3*f1*f42*v2 - f43*v2 - f1mf22*(f1 + 2*f2 - 3*f4)*v4)/(f1mf22*f1mf43*f2mf42))

    elif version == 105:
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1
        f15 = f14 * f1
        f16 = f15 * f1
        f17 = f16 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2
        f25 = f24 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3
        f35 = f34 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f45 = f44 * f4
        f46 = f45 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf32 = f1mf3 * f1mf3
        f1mf42 = f1mf4 * f1mf4
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = (
          (-(d4*f1mf22*f1mf32*f1mf4*f2mf3*f2mf4*f3mf4*(2*f1 + f2 + f3 + f4)) - d1*f1mf2*f1mf3*f1mf4*f2mf3*f2mf42*f3mf42*(f1 + f2 + f3 + 2*f4) + 5*f13*f23*f32*v1 - 3*f1*f25*f32*v1 - 5*f13*f22*f33*v1 +
          2*f25*f33*v1 + 3*f1*f22*f35*v1 - 2*f23*f35*v1 - 10*f13*f23*f3*f4*v1 + 6*f1*f25*f3*f4*v1 + 5*f12*f23*f32*f4*v1 - 3*f25*f32*f4*v1 + 10*f13*f2*f33*f4*v1 - 5*f12*f22*f33*f4*v1 -
          6*f1*f2*f35*f4*v1 + 3*f22*f35*f4*v1 + 5*f13*f23*f42*v1 - 3*f1*f25*f42*v1 + 15*f13*f22*f3*f42*v1 - 10*f12*f23*f3*f42*v1 - 15*f13*f2*f32*f42*v1 + 5*f1*f23*f32*f42*v1 - 5*f13*f33*f42*v1 +
          10*f12*f2*f33*f42*v1 - 5*f1*f22*f33*f42*v1 + 3*f1*f35*f42*v1 - 10*f13*f22*f43*v1 + 5*f12*f23*f43*v1 + f25*f43*v1 + 15*f12*f22*f3*f43*v1 - 10*f1*f23*f3*f43*v1 + 10*f13*f32*f43*v1 -
          15*f12*f2*f32*f43*v1 + 5*f23*f32*f43*v1 - 5*f12*f33*f43*v1 + 10*f1*f2*f33*f43*v1 - 5*f22*f33*f43*v1 - f35*f43*v1 + 5*f13*f2*f44*v1 - 10*f12*f22*f44*v1 + 5*f1*f23*f44*v1 - 5*f13*f3*f44*v1 +
          10*f12*f32*f44*v1 - 5*f1*f33*f44*v1 + 5*f12*f2*f45*v1 + 2*f1*f22*f45*v1 - 3*f23*f45*v1 - 5*f12*f3*f45*v1 - 2*f1*f32*f45*v1 + 3*f33*f45*v1 - 4*f1*f2*f46*v1 + 2*f22*f46*v1 + 4*f1*f3*f46*v1 -
          2*f32*f46*v1 - 2*f16*f32*v2 + 3*f15*f33*v2 - f13*f35*v2 + 4*f16*f3*f4*v2 - 2*f15*f32*f4*v2 - 5*f14*f33*f4*v2 + 3*f12*f35*f4*v2 - 2*f16*f42*v2 - 5*f15*f3*f42*v2 + 10*f14*f32*f42*v2 -
          3*f1*f35*f42*v2 + 4*f15*f43*v2 - 5*f14*f3*f43*v2 + f35*f43*v2 + 5*f13*f3*f44*v2 - 10*f12*f32*f44*v2 + 5*f1*f33*f44*v2 - 4*f13*f45*v2 + 5*f12*f3*f45*v2 + 2*f1*f32*f45*v2 - 3*f33*f45*v2 +
          2*f12*f46*v2 - 4*f1*f3*f46*v2 + 2*f32*f46*v2 + 2*f16*f22*v3 - 3*f15*f23*v3 + f13*f25*v3 - 4*f16*f2*f4*v3 + 2*f15*f22*f4*v3 + 5*f14*f23*f4*v3 - 3*f12*f25*f4*v3 + 2*f16*f42*v3 +
          5*f15*f2*f42*v3 - 10*f14*f22*f42*v3 + 3*f1*f25*f42*v3 - 4*f15*f43*v3 + 5*f14*f2*f43*v3 - f25*f43*v3 - 5*f13*f2*f44*v3 + 10*f12*f22*f44*v3 - 5*f1*f23*f44*v3 + 4*f13*f45*v3 - 5*f12*f2*f45*v3 -
          2*f1*f22*f45*v3 + 3*f23*f45*v3 - 2*f12*f46*v3 + 4*f1*f2*f46*v3 - 2*f22*f46*v3 -
          f1mf22*f1mf32*f2mf3*(2*f2*f3*(f2 + f3) + 2*f12*(f2 + f3 - 2*f4) - 3*(f22 + f2*f3 + f32)*f4 + f1*(f22 + 5*f2*f3 + f32 - 6*(f2 + f3)*f4 + 5*f42) + 5*f43)*v4)/
          (f1mf22*f1mf32*f1mf43*f2mf3*f2mf42*f3mf42)
        )

    else:
        raise ValueError(f"Error in IMRPhenomX_Intermediate_Amp_22_delta4: version {version} is not valid. Recommended flag is 104.")

    return retVal

def IMRPhenomX_Intermediate_Amp_22_delta5(d1: float, d4: float, V1: float, V2: float, V3: float, V4: float,
                                          F1: float, F2: float, F3: float, F4: float, version: int) -> float:
    """
    Solve for intermediate amplitude delta5 coefficient.

    Translated from LALSimIMRPhenomX_intermediate.c
    """
    f1, f2, f3, f4 = F1, F2, F3, F4
    v1, v2, v3, v4 = V1, V2, V3, V4

    if version == 1043:  # no left derivative: v1, v2, v3, v4, d4; 1043 is used in IMRPhenomXHM
        retVal = 0.0

    elif version == 104:
        retVal = 0.0

    elif version == 105:
        f12 = f1 * f1
        f13 = f12 * f1
        f14 = f13 * f1
        f15 = f14 * f1
        f16 = f15 * f1
        f17 = f16 * f1

        f22 = f2 * f2
        f23 = f22 * f2
        f24 = f23 * f2
        f25 = f24 * f2

        f32 = f3 * f3
        f33 = f32 * f3
        f34 = f33 * f3
        f35 = f34 * f3

        f42 = f4 * f4
        f43 = f42 * f4
        f44 = f43 * f4
        f45 = f44 * f4
        f46 = f45 * f4
        f47 = f46 * f4

        f1mf2 = f1 - f2
        f1mf3 = f1 - f3
        f1mf4 = f1 - f4
        f2mf3 = f2 - f3
        f2mf4 = f2 - f4
        f3mf4 = f3 - f4

        f1mf22 = f1mf2 * f1mf2
        f1mf32 = f1mf3 * f1mf3
        f1mf42 = f1mf4 * f1mf4
        f2mf32 = f2mf3 * f2mf3
        f2mf42 = f2mf4 * f2mf4
        f3mf42 = f3mf4 * f3mf4
        f1mf43 = f1mf4 * f1mf4 * f1mf4

        retVal = (
          (d4*f1mf22*f1mf32*f1mf4*f2mf3*f2mf4*f3mf4 + d1*f1mf2*f1mf3*f1mf4*f2mf3*f2mf42*f3mf42 - 4*f12*f23*f32*v1 + 3*f1*f24*f32*v1 + 4*f12*f22*f33*v1 - 2*f24*f33*v1 - 3*f1*f22*f34*v1 + 2*f23*f34*v1 +
            8*f12*f23*f3*f4*v1 - 6*f1*f24*f3*f4*v1 - 4*f1*f23*f32*f4*v1 + 3*f24*f32*f4*v1 - 8*f12*f2*f33*f4*v1 + 4*f1*f22*f33*f4*v1 + 6*f1*f2*f34*f4*v1 - 3*f22*f34*f4*v1 - 4*f12*f23*f42*v1 +
            3*f1*f24*f42*v1 - 12*f12*f22*f3*f42*v1 + 8*f1*f23*f3*f42*v1 + 12*f12*f2*f32*f42*v1 - 4*f23*f32*f42*v1 + 4*f12*f33*f42*v1 - 8*f1*f2*f33*f42*v1 + 4*f22*f33*f42*v1 - 3*f1*f34*f42*v1 +
            8*f12*f22*f43*v1 - 4*f1*f23*f43*v1 - f24*f43*v1 - 8*f12*f32*f43*v1 + 4*f1*f33*f43*v1 + f34*f43*v1 - 4*f12*f2*f44*v1 - f1*f22*f44*v1 + 2*f23*f44*v1 + 4*f12*f3*f44*v1 + f1*f32*f44*v1 -
            2*f33*f44*v1 + 2*f1*f2*f45*v1 - f22*f45*v1 - 2*f1*f3*f45*v1 + f32*f45*v1 + f15*f32*v2 - 2*f14*f33*v2 + f13*f34*v2 - 2*f15*f3*f4*v2 + f14*f32*f4*v2 + 4*f13*f33*f4*v2 - 3*f12*f34*f4*v2 +
            f15*f42*v2 + 4*f14*f3*f42*v2 - 8*f13*f32*f42*v2 + 3*f1*f34*f42*v2 - 3*f14*f43*v2 + 8*f12*f32*f43*v2 - 4*f1*f33*f43*v2 - f34*f43*v2 + 3*f13*f44*v2 - 4*f12*f3*f44*v2 - f1*f32*f44*v2 +
            2*f33*f44*v2 - f12*f45*v2 + 2*f1*f3*f45*v2 - f32*f45*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f4*v3 - f14*f22*f4*v3 - 4*f13*f23*f4*v3 + 3*f12*f24*f4*v3 - f15*f42*v3 -
            4*f14*f2*f42*v3 + 8*f13*f22*f42*v3 - 3*f1*f24*f42*v3 + 3*f14*f43*v3 - 8*f12*f22*f43*v3 + 4*f1*f23*f43*v3 + f24*f43*v3 - 3*f13*f44*v3 + 4*f12*f2*f44*v3 + f1*f22*f44*v3 - 2*f23*f44*v3 +
            f12*f45*v3 - 2*f1*f2*f45*v3 + f22*f45*v3 + f1mf22*f1mf32*f2mf3*(2*f2*f3 + f1*(f2 + f3 - 2*f4) - 3*(f2 + f3)*f4 + 4*f42)*v4)/(f1mf22*f1mf32*f1mf43*f2mf3*f2mf42*f3mf42)
          )

    else:
        raise ValueError(f"Error in IMRPhenomX_Intermediate_Amp_22_delta5: version {version} is not valid. Recommended flag is 104.")

    return retVal


def IMRPhenomX_Intermediate_Phase_22_Ansatz(
    ff: float,
    powers_of_f: 'IMRPhenomX_UsefulPowers',
    pWF: dict,
    pPhase: dict
) -> float:
    """
    Compute the derivative of the intermediate phase ansatz (dPhi/df).

    This function evaluates the derivative of the intermediate regime phase ansatz
    for the 22 mode. The ansatz includes polynomial terms in inverse powers of
    frequency and a Lorentzian term that provides smooth connection to the ringdown.

    Translated from LALSimIMRPhenomX_intermediate.c

    Parameters
    ----------
    ff : float
        Frequency in geometric units
    powers_of_f : IMRPhenomX_UsefulPowers
        Pre-computed powers of frequency for efficiency. Should have attributes:
            - m_one: f^(-1) = 1/f
            - m_two: f^(-2) = 1/f^2
            - m_three: f^(-3) = 1/f^3
            - m_four: f^(-4) = 1/f^4
    pWF : dict
        Waveform parameter structure containing:
            - IMRPhenomXIntermediatePhaseVersion: Version flag (104 or 105)
            - fRING: Ring-down frequency
            - fDAMP: Damping frequency
    pPhase : dict
        Phase coefficient structure containing:
            - b0: Constant term coefficient
            - b1: Coefficient for 1/f term
            - b2: Coefficient for 1/f^2 term
            - b3: Coefficient for 1/f^3 term (only for version 105)
            - b4: Coefficient for 1/f^4 term
            - cLGR: GR Lorentzian coefficient

    Returns
    -------
    float
        Derivative of intermediate phase (dPhi/df) at frequency ff

    Raises
    ------
    ValueError
        If IMRPhenomXIntermediatePhaseVersion is not valid (104 or 105)

    Notes
    -----
    Version 104 uses 4 coefficients: b0, b1, b2, b4 (canonical, recommended)
    Version 105 uses 5 coefficients: b0, b1, b2, b3, b4 (canonical)

    The Lorentzian term is: (4*cL) / ((4*fda^2) + (ff - frd)^2)
    where cL = -a_RD * dphase0, ensuring GR coefficient variations in ringdown
    decouple from the intermediate regime.
    """
    # Handle powers_of_f as either object or dictionary
    if isinstance(powers_of_f, dict):
        invff1 = powers_of_f['m_one']
        invff2 = powers_of_f['m_two']
        invff3 = powers_of_f['m_three']
        invff4 = powers_of_f['m_four']
    else:
        invff1 = powers_of_f.m_one
        invff2 = powers_of_f.m_two
        invff3 = powers_of_f.m_three
        invff4 = powers_of_f.m_four

    fda = pWF['fDAMP']
    frd = pWF['fRING']
    # We pass the GR Lorentzian term to make sure that variations to the GR
    # coefficients in the ringdown decouple from the intermediate regime
    cL = pPhase['cLGR']

    IntPhaseVersion = pWF['IMRPhenomXIntermediatePhaseVersion']

    if IntPhaseVersion == 104:  # Canonical, 4 coefficients
        # This is the Lorentzian term where cL = - a_{RD} dphase0
        LorentzianTerm = (4.0 * cL) / ((4.0 * fda * fda) + (ff - frd) * (ff - frd))

        # Return a polynomial embedded in the background from the merger
        phaseOut = (pPhase['b0'] + pPhase['b1'] * invff1 + pPhase['b2'] * invff2 +
                    pPhase['b4'] * invff4 + LorentzianTerm)

    elif IntPhaseVersion == 105:  # Canonical, 5 coefficients
        # This is the Lorentzian term where cL = - a_{RD} dphase0
        LorentzianTerm = (4.0 * cL) / ((4.0 * fda * fda) + (ff - frd) * (ff - frd))

        # Return a polynomial embedded in the background from the merger
        phaseOut = (pPhase['b0'] + pPhase['b1'] * invff1 + pPhase['b2'] * invff2 +
                    pPhase['b3'] * invff3 + pPhase['b4'] * invff4 + LorentzianTerm)

    else:
        raise ValueError(
            f"Error in IMRPhenomX_Intermediate_Phase_22_Ansatz: "
            f"IMRPhenomXIntermediatePhaseVersion = {IntPhaseVersion} is not valid. "
            f"Recommended flag is 104."
        )

    return phaseOut


def IMRPhenomX_Intermediate_Phase_22_AnsatzInt(
    f: float,
    powers_of_f: 'IMRPhenomX_UsefulPowers',
    pWF: dict,
    pPhase: dict
) -> float:
    """
    Compute the intermediate phase using the ansatz with Lorentzian term.

    This function evaluates the intermediate regime phase ansatz for the 22 mode.
    The ansatz includes polynomial terms and a Lorentzian (arctangent) term that
    smoothly connects to the ringdown regime.

    Translated from LALSimIMRPhenomX_intermediate.c

    Parameters
    ----------
    f : float
        Frequency in geometric units
    powers_of_f : IMRPhenomX_UsefulPowers
        Pre-computed powers of frequency for efficiency. Should have attributes:
            - m_one: f^(-1) = 1/f
            - m_two: f^(-2) = 1/f^2
            - m_three: f^(-3) = 1/f^3
            - log: log(f)
    pWF : dict
        Waveform parameter structure containing:
            - IMRPhenomXIntermediatePhaseVersion: Version flag (104 or 105)
            - fRING: Ring-down frequency
            - fDAMP: Damping frequency
    pPhase : dict
        Phase coefficient structure containing:
            - b0: Coefficient for f term
            - b1: Coefficient for log(f) term
            - b2: Coefficient for 1/f term
            - b3: Coefficient for 1/f^2 term (only for version 105)
            - b4: Coefficient for 1/f^3 term
            - cLGR: GR Lorentzian coefficient

    Returns
    -------
    float
        Intermediate phase value at frequency f

    Raises
    ------
    ValueError
        If IMRPhenomXIntermediatePhaseVersion is not valid (104 or 105)

    Notes
    -----
    Version 104 uses 4 coefficients: b0, b1, b2, b4 (canonical)
    Version 105 uses 5 coefficients: b0, b1, b2, b3, b4 (canonical)

    The GR Lorentzian term ensures that variations to GR coefficients in the
    ringdown regime decouple from the intermediate regime.
    """
    invff1 = powers_of_f.m_one
    invff2 = powers_of_f.m_two
    invff3 = powers_of_f.m_three
    logfv = powers_of_f.log

    frd = pWF['fRING']
    fda = pWF['fDAMP']

    b0 = pPhase['b0']
    b1 = pPhase['b1']
    b2 = pPhase['b2']
    b3 = pPhase['b3']
    b4 = pPhase['b4']
    # We pass the GR Lorentzian term to make sure that variations to the GR
    # coefficients in the ringdown decouple from the intermediate regime
    cL = pPhase['cLGR']

    IntPhaseVersion = pWF['IMRPhenomXIntermediatePhaseVersion']

    if IntPhaseVersion == 104:  # Canonical, 4 coefficients
        phaseOut = (
            b0 * f
            + b1 * logfv
            - b2 * invff1
            - (b4 * invff3 / 3.0)
            + (2.0 * cL * jnp.arctan((f - frd) / (2.0 * fda))) / fda
        )
    elif IntPhaseVersion == 105:  # Canonical, 5 coefficients
        phaseOut = (
            b0 * f
            + b1 * logfv
            - b2 * invff1
            - b3 * invff2 / 2.0
            - (b4 * invff3 / 3.0)
            + (2.0 * cL * jnp.arctan((f - frd) / (2.0 * fda))) / fda
        )
    else:
        raise ValueError(
            f"Error in IMRPhenomX_Intermediate_Phase_22_AnsatzInt: "
            f"IMRPhenomXIntermediatePhaseVersion={IntPhaseVersion} is not valid. "
            f"Recommended flag is 104."
        )

    return phaseOut

