
from .LALSimIMRPhenomX_internals import IMRPhenomXSetWaveformVariables

import jax.numpy as jnp

from .LALSimIMRPhenomX_precession import (IMRPhenomX_Return_phi_zeta_costhetaL_MSA, IMRPhenomXGetAndSetPrecessionVariables, XLALSimIMRPhenomXUtilsHztoMf)

from .LALSimIMRPhenomX_internals import (IMRPhenomXGetPhaseCoefficients, IMRPhenomXGetAmplitudeCoefficients)

from .LALSimIMRPhenomXHM_multiband import (deltaF_mergerBin, 
                                           deltaF_ringdownBin, 
                                           deltaF_MergerRingdown,
                                           XLALSimIMRPhenomXMultibandingGrid)

from .LALSimIMRPhenomXHM_internals import (IMRPhenomXHM_Initialize_QNMs, 
                                           IMRPhenomXHM_SetHMWaveformVariables, 
                                           IMRPhenomXHM_FillAmpFitsArray, 
                                           IMRPhenomXHM_FillPhaseFitsArray,
                                           GetSpheroidalCoefficients, 
                                           IMRPhenomXHM_GetAmplitudeCoefficients,
                                           IMRPhenomXHM_GetPhaseCoefficients)

import jax


def XLALSimIMRPhenomXPHMMultibandingGrid(
    ell: int,                        # First index non-precessing mode
    emmprime: int,                   # Second index non-precessing mode
    pWF,                             # IMRPhenomX Waveform Struct
    lalParams                        # LAL dictionary
):
    """
    Create non-uniform coarse frequency grid for multiband evaluation.
    This function is basically a copy of the first part of IMRPhenomXHMMultiBandOneMode
    and IMRPhenomXHMMultiBandOneModeMixing.

    Returns:
        coarseFreqs: Non-uniform coarse frequency grid (1D array)
        actualnumberofGrids: Number of subgrids used
    """

    # Create non-uniform grid for each mode
    thresholdMB = 0.001#lalParams['PhenomXPHMThresholdMband']

    # Compute the coarse frequency array. It is stored in a list of grids.
    iStart = int(pWF['fMin'] / pWF['deltaF'])

    # Final grid spacing, adimensional (NR) units
    evaldMf = XLALSimIMRPhenomXUtilsHztoMf(pWF['deltaF'], pWF['Mtot'])

    # Variable for the Multibanding criteria. See eq. 2.8-2.9 of arXiv:2001.10897.
    dfpower = 11.0 / 6.0
    pi_m_one_sixth = jnp.power(jnp.pi, -1.0/6)
    dfcoefficient = (8.0 * jnp.sqrt(3.0 / 5.0) * jnp.pi * pi_m_one_sixth *
                     jnp.sqrt(2.0) * jnp.cbrt(2.0) / (jnp.cbrt(emmprime) * emmprime) *
                     jnp.sqrt(thresholdMB * pWF['eta']))

    # Variables for the coarse frequency grid
    Mfmin = XLALSimIMRPhenomXUtilsHztoMf(iStart * pWF['deltaF'], pWF['Mtot'])
    Mfmax = XLALSimIMRPhenomXUtilsHztoMf(pWF['f_max_prime'], pWF['Mtot'])

    dfmerger = 0.0
    dfringdown = 0.0
    lengthallGrids = 20

    # Initialize grid structures (simplified for JAX)
    # In JAX, we'll need to handle this differently than malloc
    # This is a placeholder that would need proper implementation
    allGrids = []  # List of grid dictionaries
    pPhase22 = IMRPhenomXGetPhaseCoefficients(pWF)
    pAmp22 = {}
    print('jax debug 4...22 mode complete ell emm', ell, emmprime)
    if ell == 2 and emmprime == 2:
        
        MfMECO = pWF['fMECO']
        MfLorentzianEnd = pWF['fRING'] + 2 * pWF['fDAMP']

        # Get phase and amplitude coefficients for 22 mode
        #pPhase22 = IMRPhenomXGetPhaseCoefficients(pWF)
        pAmp22 = IMRPhenomXGetAmplitudeCoefficients(pWF, pAmp)

        dfmerger = deltaF_mergerBin(pWF['fDAMP'], pPhase22['cLovfda'] / pWF['eta'], thresholdMB)
        dfringdown = deltaF_ringdownBin(pWF['fDAMP'], pPhase22['cLovfda'] / pWF['eta'],
                                        pAmp22['gamma2'] / (pAmp22['gamma3'] * pWF['fDAMP']), thresholdMB)
    else:
        # Initialize QNMs and populate pWFHM for higher modes
        qnms = IMRPhenomXHM_Initialize_QNMs()
        pWFHM = IMRPhenomXHM_SetHMWaveformVariables(ell, emmprime, pWF, qnms, lalParams)

        # Get phase and amplitude coefficients
        #FIXME These two statements need to be double checked
        

        pAmp = IMRPhenomXHM_FillAmpFitsArray()
        pPhase = IMRPhenomXHM_FillPhaseFitsArray()
        print("jax debug 5 pWFHM['MixingOn']", pWFHM['MixingOn'])
        if pWFHM['MixingOn'] == 1:
            pPhase = GetSpheroidalCoefficients(pPhase, pPhase22, pWFHM, pWF) #What does this function return?
            pAmp22 = IMRPhenomXGetAmplitudeCoefficients(pWF)


        IMRPhenomXHM_GetAmplitudeCoefficients(pAmp, pPhase, pAmp22, pPhase22, pWFHM, pWF)
        IMRPhenomXHM_GetPhaseCoefficients(pAmp, pPhase, pAmp22, pPhase22, pWFHM, pWF, lalParams)
        print('jax debug 7...')
        MfMECO = pWFHM['fMECOlm']
        MfLorentzianEnd = pWFHM['fRING'] + 2 * pWFHM['fDAMP']

        dfmerger, dfringdown = deltaF_MergerRingdown(thresholdMB, pWFHM, pAmp, pPhase)

    # Generate the multibanding grid
    nGridsUsed, allGrids = XLALSimIMRPhenomXMultibandingGrid(
        Mfmin, MfMECO, MfLorentzianEnd, Mfmax, evaldMf,
        dfpower, dfcoefficient, dfmerger, dfringdown
    )

    if allGrids is None:
        return None, -1

    # Number of fine frequencies per coarse interval in every coarse grid
    actualnumberofGrids = 0
    lenCoarseArray = 0

    # Transform the coarse frequency array to 1D array
    # Take only the subgrids needed
    for kk in range(nGridsUsed):
        lenCoarseArray = lenCoarseArray + allGrids[kk]['Length']
        actualnumberofGrids += 1

        if allGrids[kk]['xMax'] + evaldMf >= Mfmax:
            break

    # Add extra points to the coarse grid if the last freq is lower than Mfmax
    while allGrids[actualnumberofGrids - 1]['xMax'] < Mfmax:
        allGrids[actualnumberofGrids - 1]['xMax'] += allGrids[actualnumberofGrids - 1]['deltax']
        allGrids[actualnumberofGrids - 1]['Length'] += 1
        lenCoarseArray += 1

    # Transform coarse frequency array to 1D vector
    coarseFreqs = jnp.zeros(lenCoarseArray)
    lencoarseFreqs = 0

    for kk in range(actualnumberofGrids):
        for ll in range(allGrids[kk]['Length']):
            coarseFreqs = coarseFreqs.at[lencoarseFreqs].set(
                allGrids[kk]['xStart'] + allGrids[kk]['deltax'] * ll
            )
            lencoarseFreqs += 1

    return coarseFreqs, actualnumberofGrids


