"""JAX implementation of LALSimIMRPhenomX_PNR_deviations.c"""

from __future__ import annotations

import jax.numpy as jnp

from ripplegw.typing import Array


def xlal_sim_imr_phenom_xcp_mu1_l2m2(theta: float, eta: float, a1: float) -> Array:
    """
    MU1 fit implementation
    Header formatting for MU1 of (l,m)=(2,2) multipole
    Hola, soy un codigo escribido por "4b_document_fits.py". Dat script is geschreven door een mens die niet kan worden genoemd.
    """
    # // // Flatten in eta
    # // eta = ( eta >= 0.09876 ) ? eta : 0.09876
    # // a1  = ( a1  <= 0.8     ) ? a1  : 0.8
    # // a1  = ( a1  >= 0.2     ) ? a1  : 0.2

    # // Preliminaries
    u = jnp.cos(theta)
    u2 = u * u
    u3 = u2 * u
    a12 = a1 * a1
    eta2 = eta * eta
    eta3 = eta2 * eta

    # // Evaluate fit for this parameter
    mu1 = (
        3.10731588e00 * a1
        + -2.98890174e-01
        + 2.83580591e00 * eta
        + -2.69235962e-01 * u
        + 4.58765298e00 * (u * eta)
        + -4.58961732e01 * (eta * a1)
        + -5.27239994e00 * a12
        + 2.35767086e00 * (u * a1)
        + -1.66981400e00 * (u * u)
        + -6.65783904e00 * eta2
        + 2.28592256e02 * (eta2 * a1)
        + -2.85191772e00 * (u * a12)
        + 8.53735433e01 * (eta * a12)
        + -1.69564326e01 * (u * eta2)
        + 3.80240537e01 * (u2 * eta)
        + -5.90965234e01 * (u * eta * a1)
        + -3.25953234e-01 * (u2 * u)
        + -3.71962074e02 * (eta3 * a1)
        + 3.65316048e02 * (u * eta2 * a1)
        + 3.69910553e00 * (u2 * a12)
        + 2.43498635e00 * (u3 * a1)
        + -4.48771815e02 * (eta2 * a12)
        + -3.06134124e01 * (u2 * eta * a1)
        + -2.49773688e02 * (u2 * eta2)
        + 7.77424585e01 * (u * eta * a12)
        + -3.86558965e01 * (u2 * eta * a12)
        + 7.62503928e02 * (eta3 * a12)
        + 5.19092710e02 * (u2 * eta2 * eta)
        + -5.04214664e02 * (u * eta2 * a12)
        + -6.11685145e02 * (u * eta3 * a1)
        + 2.91303571e02 * (u2 * eta2 * a1)
        + -6.71291161e00 * (u3 * eta * a1)
        + -3.34353185e00 * (u3 * a12)
        + 4.49179407e01 * (u3 * eta2 * eta)
        + 1.10210016e02 * (u2 * eta2 * a12)
        + 9.08677395e02 * (u * eta3 * a12)
        + -3.78457492e01 * (u3 * eta2 * a1)
        + -7.50921355e02 * (u2 * eta3 * a1)
        + 1.89942289e01 * (u3 * eta * a12)
    )

    # // Return answer
    return mu1


def xlal_sim_imr_phenom_xcp_mu2_l2m2(theta: float, eta: float, a1: float) -> Array:
    """
    MU2 fit implementation
    Header formatting for MU2 of (l,m)=(2,2) multipole
    Hola, soy un codigo escribido por "4b_document_fits.py". Dat script is geschreven door een mens die niet kan worden genoemd.
    """

    # // // Flatten in eta
    # // eta = ( eta >= 0.09876 ) ? eta : 0.09876
    # // a1  = ( a1  <= 0.8     ) ? a1  : 0.8
    # // a1  = ( a1  >= 0.2     ) ? a1  : 0.2

    # // Preliminaries
    u = jnp.cos(theta)
    u2 = u * u
    u3 = u2 * u
    a12 = a1 * a1
    eta2 = eta * eta
    eta3 = eta2 * eta

    # // Evaluate fit for this parameter
    mu2 = (
        -2.02502540e-01
        + 8.72364229e00 * (eta * a1)
        + -1.58131236e00 * a12
        + -3.87863737e00 * (u * u)
        + -5.05936693e01 * (eta2 * a1)
        + -8.00671915e00 * (u * a12)
        + 9.12534447e00 * (u2 * a1)
        + 5.33069685e00 * (eta * a12)
        + 8.51436158e00 * (eta2 * eta)
        + 5.78604124e01 * (u2 * eta)
        + 6.55310845e00 * (u * eta * a1)
        + -1.94647765e00 * (u2 * u)
        + 4.08691562e01 * (u3 * eta)
        + 7.52332356e01 * (eta3 * a1)
        + -1.06817659e01 * (u2 * a12)
        + -2.87950209e00 * (u3 * a1)
        + -6.43240440e01 * (u2 * eta * a1)
        + -2.49087968e01 * (u * eta2 * eta)
        + -3.16185296e02 * (u2 * eta2)
        + 1.06442533e02 * (u * eta * a12)
        + 1.03449590e02 * (u2 * eta * a12)
        + 6.51115537e02 * (u2 * eta2 * eta)
        + -5.79811830e02 * (u * eta2 * a12)
        + -2.76924732e02 * (u3 * eta2)
        + 2.10132916e01 * (u3 * eta * a1)
        + 2.70715432e00 * (u3 * a12)
        + 6.47099541e02 * (u3 * eta2 * eta)
        + -2.05802266e02 * (u2 * eta2 * a12)
        + 1.03222489e03 * (u * eta3 * a12)
        + -9.28493309e01 * (u3 * eta2 * a1)
        + 2.54661923e02 * (u2 * eta3 * a1)
    )  # reading this is a sign you should take a break

    # // Return answer
    return mu2


def xlal_sim_imr_phenom_xcp_mu3_l2m2(theta: float, eta: float, a1: float) -> Array:
    """
    MU3 fit implementation
    Header formatting for MU3 of (l,m)=(2,2) multipole
    Hola, soy un codigo escribido por "4b_document_fits.py". Dat script is geschreven door een mens die niet kan worden genoemd.
    """

    #     // // Flatten in eta
    # // eta = ( eta >= 0.09876 ) ? eta : 0.09876
    # // a1  = ( a1  <= 0.8     ) ? a1  : 0.8
    # // a1  = ( a1  >= 0.2     ) ? a1  : 0.2

    # // Preliminaries
    u = jnp.cos(theta)
    u2 = u * u
    u3 = u2 * u
    a12 = a1 * a1
    eta2 = eta * eta
    eta3 = eta2 * eta

    # // Evaluate fit for this parameter
    mu3 = (
        6.43824401e-02 * a1
        + -3.17858924e-03
        + 6.54067059e-03 * eta
        + 2.48066282e-02 * u
        + -1.50482854e-01 * (u * eta)
        + -8.51847106e-01 * (eta * a1)
        + -1.14919973e-01 * (u * a1)
        + -1.23872940e-02 * (u * u)
        + 4.35501780e00 * (eta2 * a1)
        + 1.51718799e-01 * (u * a12)
        + -4.01788734e-02 * (u2 * a1)
        + 1.94815013e-01 * (u * eta2)
        + 1.61029465e-01 * (u2 * eta)
        + 5.63051639e-01 * (u * eta * a1)
        + -9.08508144e-03 * (u2 * u)
        + -7.78742176e00 * (eta3 * a1)
        + 7.84548415e-02 * (u3 * a1)
        + 7.87294341e-01 * (u2 * eta * a1)
        + -9.60114345e-01 * (u2 * eta2)
        + -9.10606320e-01 * (u * eta * a12)
        + -1.42496492e-01 * (u2 * eta * a12)
        + 1.67682276e00 * (u2 * eta2 * eta)
        + 1.14108488e00 * (u * eta2 * a12)
        + -1.45121681e00 * (u * eta3 * a1)
        + -3.90246382e00 * (u2 * eta2 * a1)
        + -1.88632810e-01 * (u3 * eta * a1)
        + -1.20970655e-01 * (u3 * a12)
        + 5.44663689e-01 * (u3 * eta2 * eta)
        + 3.60304651e-01 * (u2 * eta2 * a12)
        + -5.29498004e-01 * (u3 * eta2 * a1)
        + 6.67574591e00 * (u2 * eta3 * a1)
        + 5.00625025e-01 * (u3 * eta * a12)
    )

    # // Return answer
    return mu3


def xlal_sim_imr_phenom_xcp_nu4_l2m2(theta: float, eta: float, a1: float) -> Array:
    """
    NU4 fit implementation
    Header formatting for NU4 of (l,m)=(2,2) multipole
    Hola, soy un codigo escribido por "4b_document_fits.py". Dat script is geschreven door een mens die niet kan worden genoemd.
    """
    # //// Flatten in eta
    # // eta = ( eta >= 0.09876 ) ? eta : 0.09876
    # // a1  = ( a1  <= 0.8     ) ? a1  : 0.8
    # // a1  = ( a1  >= 0.2     ) ? a1  : 0.2

    # // Preliminaries
    u = jnp.cos(theta)
    u2 = u * u
    u3 = u2 * u
    a12 = a1 * a1
    eta2 = eta * eta
    eta3 = eta2 * eta

    # // Evaluate fit for this parameter
    nu4 = (
        -3.16028942e-02 * a1
        + 4.49481835e-03
        + -4.67316682e-02 * eta
        + 1.59179379e-02 * u
        + -3.27118716e-01 * (u * eta)
        + 3.77496559e-01 * (eta * a1)
        + 3.25729551e-02 * a12
        + -7.76768528e-02 * (u * a1)
        + -7.33656551e-03 * (u * u)
        + 1.08635205e-01 * eta2
        + -1.32662456e00 * (eta2 * a1)
        + 4.84795811e-02 * (u * a12)
        + 7.64861112e-02 * (u2 * a1)
        + -3.45847253e-01 * (eta * a12)
        + 1.90217409e00 * (u * eta2)
        + 1.74016014e00 * (u * eta * a1)
        + 1.38785202e00 * (eta3 * a1)
        + -1.06852751e01 * (u * eta2 * a1)
        + -8.33709907e-02 * (u2 * a12)
        + -7.93716551e-03 * (u3 * a1)
        + 8.34552086e-01 * (eta2 * a12)
        + -7.44879716e-01 * (u2 * eta * a1)
        + -3.35542027e00 * (u * eta2 * eta)
        + 5.13747474e-01 * (u2 * eta2)
        + -1.42151090e00 * (u * eta * a12)
        + 8.28154534e-01 * (u2 * eta * a12)
        + -1.67075105e00 * (u2 * eta2 * eta)
        + 9.46433414e00 * (u * eta2 * a12)
        + 1.97711659e01 * (u * eta3 * a1)
        + 1.80163775e00 * (u2 * eta2 * a1)
        + 2.22117601e-01 * (u3 * eta2)
        + -9.13086579e-02 * (u3 * eta * a1)
        + 3.41917884e-02 * (u3 * a12)
        + -1.01537065e00 * (u3 * eta2 * eta)
        + -2.00129470e00 * (u2 * eta2 * a12)
        + -1.81767526e01 * (u * eta3 * a12)
        + 5.87741796e-01 * (u3 * eta2 * a1)
        + -1.52767109e-01 * (u3 * eta * a12)
    )

    # // Return answer
    return nu4


def xlal_sim_imr_phenom_xcp_nu5_l2m2(theta: float, eta: float, a1: float) -> Array:
    """
    NU5 fit implementation
    Header formatting for NU5 of (l,m)=(2,2) multipole
        Hola, soy un codigo escribido por "4b_document_fits.py". Dat script is geschreven door een mens die niet kan worden genoemd.
    """

    # // // Flatten in eta
    # // eta = ( eta >= 0.09876 ) ? eta : 0.09876
    # // a1  = ( a1  <= 0.8     ) ? a1  : 0.8
    # // a1  = ( a1  >= 0.2     ) ? a1  : 0.2

    # // Preliminaries
    u = jnp.cos(theta)
    u2 = u * u
    u3 = u2 * u
    a12 = a1 * a1
    eta2 = eta * eta
    eta3 = eta2 * eta

    # // Evaluate fit for this parameter
    nu5 = (
        -8.24852329e-03
        + 1.29541487e-01 * eta
        + -2.68106259e-02 * u
        + 3.96908249e-01 * (u * eta)
        + -4.57402308e-01 * (eta * a1)
        + -6.96270582e-02 * a12
        + 8.87404766e-02 * (u * a1)
        + -3.95509060e-02 * (u * u)
        + -7.66209616e-01 * eta2
        + 3.52527019e00 * (eta2 * a1)
        + -8.52495198e-02 * (u * a12)
        + 1.28777123e-01 * (u2 * a1)
        + 1.34613777e00 * (eta * a12)
        + -1.97688956e00 * (u * eta2)
        + 1.33051669e00 * (eta2 * eta)
        + 6.66123113e-01 * (u2 * eta)
        + -8.83292400e-01 * (u * eta * a1)
        + 4.13508242e-03 * (u2 * u)
        + -6.34717777e00 * (eta3 * a1)
        + 3.39853108e00 * (u * eta2 * a1)
        + -5.33540972e-02 * (u2 * a12)
        + -3.96980662e-02 * (u3 * a1)
        + -8.95748509e00 * (eta2 * a12)
        + -1.67674467e00 * (u2 * eta * a1)
        + 3.19730879e00 * (u * eta2 * eta)
        + -4.21512938e00 * (u2 * eta2)
        + 3.45438117e-01 * (u * eta * a12)
        + 2.93065248e-01 * (u2 * eta * a12)
        + 1.82938901e01 * (eta3 * a12)
        + 9.11903434e00 * (u2 * eta2 * eta)
        + -5.03020035e00 * (u * eta3 * a1)
        + 9.57710907e00 * (u2 * eta2 * a1)
        + -6.00764557e-01 * (u3 * eta2)
        + 4.25544610e-01 * (u3 * eta * a1)
        + 4.02874363e-02 * (u3 * a12)
        + 2.55062055e00 * (u3 * eta2 * eta)
        + -1.31746981e00 * (u3 * eta2 * a1)
        + -2.11785354e01 * (u2 * eta3 * a1)
        + -1.29990564e-01 * (u3 * eta * a12)
    )

    # // Return answer
    return nu5


def xlal_sim_imr_phenom_xcp_nu6_l2m2(theta: float, eta: float, a1: float) -> Array:
    """
    NU6 fit implementation
    Header formatting for NU6 of (l,m)=(2,2) multipole
    Hola, soy un codigo escribido por "4b_document_fits.py". Dat script is geschreven door een mens die niet kan worden genoemd.
    """

    # // // Flatten in eta
    # // eta = ( eta >= 0.09876 ) ? eta : 0.09876
    # // a1  = ( a1  <= 0.8     ) ? a1  : 0.8
    # // a1  = ( a1  >= 0.2     ) ? a1  : 0.2

    # // Preliminaries
    u = jnp.cos(theta)
    u2 = u * u
    u3 = u2 * u
    a12 = a1 * a1
    eta2 = eta * eta
    eta3 = eta2 * eta

    # // Evaluate fit for this parameter
    nu6 = (
        -8.14163374e-02 * a1
        + 1.60753917e-02
        + -3.14932520e-01 * eta
        + 3.69589365e-02 * u
        + -5.43366127e-01 * (u * eta)
        + 1.65256072e00 * (eta * a1)
        + 1.57642062e-01 * a12
        + -3.53871924e-01 * (u * a1)
        + 1.91773389e00 * eta2
        + -1.01369676e01 * (eta2 * a1)
        + 4.49846052e-01 * (u * a12)
        + -3.23664263e-02 * (u2 * a1)
        + -2.77552262e00 * (eta * a12)
        + 3.18104872e00 * (u * eta2)
        + -3.81823248e00 * (eta2 * eta)
        + -4.05623855e-02 * (u2 * eta)
        + 5.78266831e00 * (u * eta * a1)
        + 4.67460809e-02 * (u2 * u)
        + -9.03957684e-01 * (u3 * eta)
        + 1.97689861e01 * (eta3 * a1)
        + -3.29265427e01 * (u * eta2 * a1)
        + 4.68080513e-02 * (u2 * a12)
        + 1.57596950e01 * (eta2 * a12)
        + 4.06667657e-01 * (u2 * eta * a1)
        + -6.37440535e00 * (u * eta2 * eta)
        + -7.23117535e00 * (u * eta * a12)
        + -6.79662526e-01 * (u2 * eta * a12)
        + -2.89973575e01 * (eta3 * a12)
        + 2.92509266e-01 * (u2 * eta2 * eta)
        + 3.96497438e01 * (u * eta2 * a12)
        + 6.16173322e01 * (u * eta3 * a1)
        + 4.62060734e00 * (u3 * eta2)
        + 5.63722799e-01 * (u3 * eta * a1)
        + -7.10443312e-02 * (u3 * a12)
        + -7.10237285e00 * (u3 * eta2 * eta)
        + 1.81706494e00 * (u2 * eta2 * a12)
        + -7.13125243e01 * (u * eta3 * a12)
        + -2.12229389e00 * (u3 * eta2 * a1)
        + -3.62928100e00 * (u2 * eta3 * a1)
        + 2.39385816e-01 * (u3 * eta * a12)
    )

    # // Return answer
    return nu6


def xlal_sim_imr_phenom_xcp_zeta1_l2m2(theta: float, eta: float, a1: float) -> Array:
    """
    ZETA1 fit implementation
    Header formatting for ZETA1 of (l,m)=(2,2) multipole
    Hola, soy un codigo escribido por "4b_document_fits.py". Dat script is geschreven door een mens die niet kan worden genoemd.
    """

    # // // Flatten in eta
    # // eta = ( eta >= 0.09876 ) ? eta : 0.09876
    # // a1  = ( a1  <= 0.8     ) ? a1  : 0.8
    # // a1  = ( a1  >= 0.2     ) ? a1  : 0.2

    # // Preliminaries
    u = jnp.cos(theta)
    u2 = u * u
    u3 = u2 * u
    a12 = a1 * a1
    eta2 = eta * eta
    eta3 = eta2 * eta

    # // Evaluate fit for this parameter
    zeta1 = (
        9.14075603e-05 * a1
        + -6.64596933e-06
        + 7.26570370e-05 * eta
        + -2.65487354e-05 * u
        + 6.13445304e-04 * (u * eta)
        + -1.13031091e-03 * (eta * a1)
        + -7.04782574e-05 * a12
        + 2.78657414e-04 * (u * a1)
        + 2.93046524e-05 * (u * u)
        + -1.62592535e-04 * eta2
        + 4.77274706e-03 * (eta2 * a1)
        + -3.28088748e-04 * (u * a12)
        + -2.40358466e-04 * (u2 * a1)
        + 6.89104675e-04 * (eta * a12)
        + -4.33992656e-03 * (u * eta2)
        + -1.71570550e-04 * (u2 * eta)
        + -5.75897400e-03 * (u * eta * a1)
        + -4.06577752e-05 * (u2 * u)
        + 7.19295439e-04 * (u3 * eta)
        + -7.10577068e-03 * (eta3 * a1)
        + 3.68151682e-02 * (u * eta2 * a1)
        + 1.70628764e-04 * (u2 * a12)
        + 2.97287666e-05 * (u3 * a1)
        + -1.58004432e-03 * (eta2 * a12)
        + 2.69942014e-03 * (u2 * eta * a1)
        + 9.27082591e-03 * (u * eta2 * eta)
        + 6.89186183e-03 * (u * eta * a12)
        + -1.33029865e-03 * (u2 * eta * a12)
        + 1.14545659e-03 * (u2 * eta2 * eta)
        + -4.28423981e-02 * (u * eta2 * a12)
        + -7.28752610e-02 * (u * eta3 * a1)
        + -1.14024958e-02 * (u2 * eta2 * a1)
        + -3.59470348e-03 * (u3 * eta2)
        + -6.18248914e-04 * (u3 * eta * a1)
        + 5.76871905e-03 * (u3 * eta2 * eta)
        + 2.67905006e-03 * (u2 * eta2 * a12)
        + 8.18766554e-02 * (u * eta3 * a12)
        + 1.68471071e-03 * (u3 * eta2 * a1)
        + 1.73091684e-02 * (u2 * eta3 * a1)
        + 8.47582112e-05 * (u3 * eta * a12)
    )

    # // Return answer
    return zeta1


def xlal_sim_imr_phenom_xcp_zeta2_l2m2(theta: float, eta: float, a1: float) -> Array:
    """
    ZETA2 fit implementation
    Header formatting for ZETA2 of (l,m)=(2,2) multipole
    Hola, soy un codigo escribido por "4b_document_fits.py". Dat script is geschreven door een mens die niet kan worden genoemd.
    """

    # // // Flatten in eta
    # // eta = ( eta >= 0.09876 ) ? eta : 0.09876
    # // a1  = ( a1  <= 0.8     ) ? a1  : 0.8
    # // a1  = ( a1  >= 0.2     ) ? a1  : 0.2

    # // Preliminaries
    u = jnp.cos(theta)
    u2 = u * u
    u3 = u2 * u
    a12 = a1 * a1
    eta2 = eta * eta
    eta3 = eta2 * eta

    # // Evaluate fit for this parameter
    zeta2 = (
        -1.93349374e01 * a1
        + 1.21917535e00
        + 5.64084304e00 * u
        + -1.35818573e02 * (u * eta)
        + 2.03450770e02 * (eta * a1)
        + 1.91050318e01 * a12
        + -3.45695270e01 * (u * a1)
        + -4.07600123e00 * (u * u)
        + -7.13247650e01 * eta2
        + -6.59883962e02 * (eta2 * a1)
        + 3.69257896e01 * (u * a12)
        + 4.19166290e01 * (u2 * a1)
        + -1.83835664e02 * (eta * a12)
        + 8.83374450e02 * (u * eta2)
        + 2.33819330e02 * (eta2 * eta)
        + 8.18248325e02 * (u * eta * a1)
        + 3.34325806e00 * (u2 * u)
        + -1.82452234e01 * (u3 * eta)
        + 5.76375582e02 * (eta3 * a1)
        + -5.26052382e03 * (u * eta2 * a1)
        + -3.57570544e01 * (u2 * a12)
        + -1.54037927e01 * (u3 * a1)
        + 4.36853250e02 * (eta2 * a12)
        + -4.35345536e02 * (u2 * eta * a1)
        + -1.69758341e03 * (u * eta2 * eta)
        + 2.99182150e02 * (u2 * eta2)
        + -8.86211351e02 * (u * eta * a12)
        + 3.47168913e02 * (u2 * eta * a12)
        + -9.23263240e02 * (u2 * eta2 * eta)
        + 5.67490957e03 * (u * eta2 * a12)
        + 1.00342283e04 * (u * eta3 * a1)
        + 1.27778712e03 * (u2 * eta2 * a1)
        + 9.20983561e01 * (u3 * eta * a1)
        + 1.53555516e01 * (u3 * a12)
        + -8.06783834e02 * (u2 * eta2 * a12)
        + -1.07373527e04 * (u * eta3 * a12)
        + -3.64839255e01 * (u3 * eta2 * a1)
        + -8.79147687e02 * (u2 * eta3 * a1)
        + -8.26009443e01 * (u3 * eta * a12)
    )

    # // Return answer
    return zeta2


def xlal_sim_imr_phenom_xcp_nu0_l2m2(theta: float, eta: float, a1: float) -> Array:
    """
    NU0 fit implementation
    Header formatting for NU0 of (l,m)=(2,2) multipole
    Hola, soy un codigo escribido por "4b_document_fits.py". Dat script is geschreven door een mens die niet kan worden genoemd.
    */"""

    # // Flatten in eta
    # // eta = ( eta >= 0.09876 ) ? eta : 0.09876
    # // a1  = ( a1  <= 0.8     ) ? a1  : 0.8
    # // a1  = ( a1  >= 0.2     ) ? a1  : 0.2

    # // Preliminaries
    u = jnp.cos(theta)
    u2 = u * u
    u3 = u2 * u
    a12 = a1 * a1
    eta2 = eta * eta
    eta3 = eta2 * eta

    # // Evaluate fit for this parameter
    nu0 = (
        9.69656479e02 * a1
        + -1.40050910e02
        + 1.28604150e03 * eta
        + 1.21045131e02 * u
        + -8.58972371e03 * (eta * a1)
        + -3.80191737e02 * a12
        + -2.62095295e03 * (u * a1)
        + 3.75573898e02 * (u * u)
        + -3.02176869e03 * eta2
        + 1.91665811e04 * (eta2 * a1)
        + 3.76489144e03 * (u * a12)
        + -2.80301545e03 * (u2 * a1)
        + -3.28538542e03 * (u2 * eta)
        + 3.46510614e04 * (u * eta * a1)
        + 3.56880662e02 * (u2 * u)
        + -8.93916723e03 * (u3 * eta)
        + -1.76931761e05 * (u * eta2 * a1)
        + 2.34502372e03 * (u2 * a12)
        + 1.13064791e03 * (u3 * a1)
        + 2.54661712e04 * (eta2 * a12)
        + 2.98001687e04 * (u2 * eta * a1)
        + -1.27661553e04 * (u * eta2 * eta)
        + -5.40526831e04 * (u * eta * a12)
        + -2.41699850e04 * (u2 * eta * a12)
        + -7.73072478e04 * (eta3 * a12)
        + 2.65990668e04 * (u2 * eta2 * eta)
        + 2.78587059e05 * (u * eta2 * a12)
        + 3.35360241e05 * (u * eta3 * a1)
        + -7.36180264e04 * (u2 * eta2 * a1)
        + 4.96525314e04 * (u3 * eta2)
        + -3.75871238e03 * (u3 * eta * a1)
        + -1.09191647e03 * (u3 * a12)
        + -7.11798894e04 * (u3 * eta2 * eta)
        + 5.84802389e04 * (u2 * eta2 * a12)
        + -4.99111316e05 * (u * eta3 * a12)
        + -7.99273344e03 * (u3 * eta2 * a1)
        + 5.11072226e03 * (u3 * eta * a12)
    )

    # // Return answer
    return nu0
