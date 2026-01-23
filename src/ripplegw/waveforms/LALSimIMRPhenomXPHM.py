
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



def XLALSimIMRPhenomXPHM(m1_SI, m2_SI,
                         chi1x, chi1y, chi1z,
                         chi2x, chi2y, chi2z,
                         distance, inclination, phiRef,
                         f_min, f_max, deltaF, fRef_In, lalParams):
    
    #Check what f_max_prime is doing. 
    
    pWF = IMRPhenomXSetWaveformVariables(m1_SI, m2_SI, chi1z, chi2z, 
                                         deltaF, fRef_In, phiRef, f_min, f_max, 
                                         distance, inclination, lalParams)
    
    samples_frequencies = jnp.arange(f_min, f_max, deltaF) # Double check 284-285

    pPrec = IMRPhenomXGetAndSetPrecessionVariables(pWF,
                                                   m1_SI,
                                                   m2_SI,
                                                   chi1x,
                                                   chi1y,
                                                   chi1z,
                                                   chi2x,
                                                   chi2y,
                                                   chi2z,
                                                   lalParams,
                                                   debug_flag=False
                                                   )
    
    
    hplus, hcross = IMRPhenomXPHM_hplushcross(samples_frequencies, pWF, pPrec, lalParams)


    ## FIXME Resizing routine ###

    return hplus, hcross


def IMRPhenomXPHM_hplushcross(frequency_array, pWF, pPrec, lalParams):
    # Line 741

    # skip 751 - 754


    L_MAX = 4

    for ell in jnp.arange(2, L_MAX+1, 1):
        print(ell)
        for emmprime in jnp.arange(1, ell+1, 1):
            cond = (pWF['q']==1) & (pWF['chi1L']==pWF['chi2L']) & (emmprime%2 !=0)
            if cond:
                "Skip twisting up"
                continue

            if (ell==3) and (emmprime==2):
                htildelm = IMRPhenomXHMMultiBandOneModeMixing(pWF, ell, emmprime, lalParams)

            else:
                htildelm = IMRPhenomXHMMultiBandOneMode(pWF, ell, emmprime, lalParams)

            # To be implemented
            #/* If the 22 and 32 modes are active, we recycle the 22 mode for the mixing in the 32 and it is passed to IMRPhenomXHMMultiBandOneModeMixing.
            #The 22 mode is always computed first than the 32, we store the 22 mode in the variable htilde22. */


            ########             TWISTING UP         #############
            #Transform modes from the precessing L-frame to inertial J-frame.

            coarseFreqs = XLALSimIMRPhenomXPHMMultibandingGrid(ell, emmprime, pWF, lalParams)
            v = jnp.cbrt(jnp.pi * coarseFreqs * 2  / emmprime)
            vangles = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(v, pWF, pPrec)

            alpha_offset_mprime, epsilon_offset_mprime = Get_alpha_epsilon_offset(emmprime, pPrec)

            valpha = vangles[0] - alpha_offset_mprime
            vepsilon = vangles[1] - epsilon_offset_mprime
            cos_beta = vangles[2]

            #cBetah, sBetah = IMRPhenomXWignerdCoefficients_cosbeta(cos_beta)
            #vbeta = jnp.acos(cBetah)

            # j < len(coarseFreqs)-1 and finecount < istop
            # For loop for interpolatio #TODO

            #Now we have the complex exponentials of the three Euler angles alpha, beta, epsilon evaluated in the fine frequency grid.
            #Next step is do the twisting up with these.

            #/************** TWISTING UP in the fine grid *****************/

            #IMRPhenomXPHMTwistUp(Mf, hlmcoprec, pWF, pPrec, ell, emmprime)

    
    
    hp = None
    hc = None

    return hp, hc


def IMRPhenomXHMMultiBandOneModeMixing(pWF, ell, emmprime, lalParams):
    htildelm = None
    return htildelm

def IMRPhenomXHMMultiBandOneMode(pWF, ell, emmprime, lalParams):
    htildelm = None
    return htildelm


def Get_alpha_epsilon_offset(
    mprime: int,                      # Second index of the non-precessing mode (l, mprime)
    pPrec                             # IMRPhenomXP Precessing structure
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
        return pPrec.alpha_offset_1, pPrec.epsilon_offset_1

    def case_2():
        return pPrec.alpha_offset, pPrec.epsilon_offset  # Already used in XP code, no _2 suffix

    def case_3():
        return pPrec.alpha_offset_3, pPrec.epsilon_offset_3

    def case_4():
        return pPrec.alpha_offset_4, pPrec.epsilon_offset_4

    # Use jax.lax.switch with mprime-1 as index (since switch uses 0-based indexing)
    alpha_offset_mprime, epsilon_offset_mprime = jax.lax.switch(
        mprime - 1,
        [case_1, case_2, case_3, case_4]
    )

    return alpha_offset_mprime, epsilon_offset_mprime


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


