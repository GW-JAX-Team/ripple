import jax.numpy as jnp
from typing import Dict, Tuple, List


def logbase(base: float, x: float) -> float:
    """
    Compute logarithm with arbitrary base.

    Parameters
    ----------
    base : float
        Base of the logarithm
    x : float
        Argument of the logarithm

    Returns
    -------
    float
        log_base(x) = ln(x) / ln(base)
    """
    return jnp.log(x) / jnp.log(base)


def XLALSimIMRPhenomXGridComp(fSTART: float, fEND: float, mydf: float) -> Dict:
    """
    Compute a uniform frequency grid structure.

    Parameters
    ----------
    fSTART : float
        Starting frequency of the uniform bin
    fEND : float
        Ending frequency of the uniform bin
    mydf : float
        Frequency spacing of the bin

    Returns
    -------
    dict
        Grid structure containing:
        - nIntervals: Number of coarse intervals
        - xStart: Starting frequency
        - xEndRequested: Requested ending frequency
        - xMax: Actual maximum frequency
        - deltax: Frequency spacing
        - Length: Total number of points
    """
    Deltaf = fEND - fSTART
    nCoarseIntervals = int(jnp.ceil(Deltaf / mydf))
    maxfreq = fSTART + mydf * nCoarseIntervals

    myGrid = {
        "nIntervals": nCoarseIntervals,
        "xStart": fSTART,
        "xEndRequested": fEND,
        "xMax": maxfreq,
        "deltax": mydf,
        "Length": nCoarseIntervals + 1,
        "intdfRatio": 1,  # Will be set by caller
    }

    return myGrid


def deltaF_MergerRingdown(
    resTest: float, pWFHM: Dict, pAmp: Dict, pPhase: Dict
) -> Tuple[float, float]:
    """
    Compute frequency spacing for merger and ringdown regions.

    This function determines appropriate frequency bin sizes for the merger
    and ringdown regions based on phase and amplitude coefficients.

    Parameters
    ----------
    resTest : float
        Resolution test parameter (absolute error tolerance)
    pWFHM : dict
        HM waveform structure containing:
        - fDAMP: Damping frequency
        - MixingOn: Flag for mode mixing
        - IMRPhenomXHMRingdownPhaseVersion: Version number
    pAmp : dict
        Amplitude coefficients dictionary with RDCoefficient array
    pPhase : dict
        Phase coefficients dictionary with alphaL, alphaL_S, or RDCoefficient

    Returns
    -------
    tuple of (float, float)
        (dfmerger, dfringdown) - Frequency spacings for merger and ringdown

    Raises
    ------
    ValueError
        If computed dfmerger or dfringdown is not positive
    """
    phase_coeff = 0.0

    if pWFHM["MixingOn"] == 0:
        phase_coeff = pPhase["alphaL"]
    else:
        if pWFHM["IMRPhenomXHMRingdownPhaseVersion"] == 122019:
            phase_coeff = pPhase["alphaL_S"]
        elif pWFHM["IMRPhenomXHMRingdownPhaseVersion"] == 122022:
            phase_coeff = pPhase["RDCoefficient"][4]

    amp_coeff = pAmp["RDCoefficient"][1] / (pAmp["RDCoefficient"][2] * pWFHM["fDAMP"])

    dfmerger = deltaF_mergerBin(pWFHM["fDAMP"], phase_coeff, resTest)
    dfringdown = deltaF_ringdownBin(pWFHM["fDAMP"], phase_coeff, amp_coeff, resTest)

    if dfmerger <= 0:
        raise ValueError(f"dfmerger = {dfmerger:.6e}. It must be > 0")
    if dfringdown <= 0:
        raise ValueError(f"dfringdown = {dfringdown:.6e}. It must be > 0")

    return dfmerger, dfringdown


def XLALSimIMRPhenomXMultibandingGrid(
    fstartIn: float,
    fend: float,
    MfLorentzianEnd: float,
    Mfmax: float,
    evaldMf: float,
    dfpower: float,
    dfcoefficient: float,
    dfmerger: float,
    dfringdown: float,
) -> Tuple[List[Dict], int]:
    """
    Build non-equally spaced coarse frequency grid for multibanding.

    This function implements the multibanding strategy described in arXiv:2001.10897,
    creating a series of frequency grids with progressively coarser spacing as
    frequency increases.

    Parameters
    ----------
    fstartIn : float
        Minimum frequency in geometric units
    fend : float
        End of inspiral frequency
    MfLorentzianEnd : float
        Determines the last frequency bin for merger region
    Mfmax : float
        Maximum frequency in geometric units
    evaldMf : float
        Spacing of the uniform fine frequency grid
    dfpower : float
        Decaying frequency power to estimate spacing (typically -1)
    dfcoefficient : float
        Multiplying factor to the estimate of frequency spacing
    dfmerger : float
        Spacing for merger bin
    dfringdown : float
        Spacing for ringdown bin

    Returns
    -------
    tuple of (list, int)
        (allGrids, nGridsUsed) where:
        - allGrids: List of grid dictionaries
        - nGridsUsed: Total number of grids used
    """
    # Initialize list to store all grids (max possible: 1 precompute + many inspiral + 1 merger + 1 RD)
    # We'll allocate generously and trim later
    MAX_GRIDS = 50
    allGrids = [None] * MAX_GRIDS

    # The df power law: df = dfcoefficient * f^dfpower
    # Number of fine freq points between two coarse freq points (eqs. 2.8, 2.9 in arXiv:2001.10897)
    dfRatio = dfcoefficient * jnp.power(fstartIn, dfpower) / evaldMf
    df0original = dfRatio * evaldMf

    # Check if user's df is coarser than multibanding criteria
    if dfRatio < 1.0:
        preComputeFirstGrid = 1
        intdfRatio = 1
        df0 = evaldMf
        fEndGrid0 = jnp.power(evaldMf / dfcoefficient, 1.0 / dfpower)
        fStartInspDerefinement = fEndGrid0 + 2 * df0
    else:
        preComputeFirstGrid = 0
        intdfRatio = int(jnp.floor(dfRatio))
        fStartInspDerefinement = fstartIn

    df0 = evaldMf * intdfRatio

    # Compute number of inspiral subgrids
    if fStartInspDerefinement >= fend:
        fEndInsp = fStartInspDerefinement
        nDerefineInspiralGrids = 0
    else:
        FrequencyFactor = jnp.power(2.0, 1.0 / dfpower)
        origLogFreqFact = logbase(
            FrequencyFactor, fend / fStartInspDerefinement
        )  # eq 2.40
        nDerefineInspiralGrids = int(jnp.ceil(origLogFreqFact))
        fEndInsp = fStartInspDerefinement * jnp.power(
            FrequencyFactor, nDerefineInspiralGrids
        )

    # Adjust transition frequencies for special cases
    if fEndInsp + evaldMf >= MfLorentzianEnd:
        nMergerGrid = 0
        nRingdownGrid = 1 if fEndInsp + evaldMf < Mfmax else 0
    else:
        nMergerGrid = 1
        nRingdownGrid = 0 if MfLorentzianEnd > Mfmax else 1

    # Precompute the first grid if needed
    if preComputeFirstGrid > 0:
        mydf = evaldMf
        coarseGrid = XLALSimIMRPhenomXGridComp(fstartIn, fEndGrid0, mydf)
        allGrids[0] = coarseGrid
        allGrids[0]["intdfRatio"] = 1

        fStartInspDerefinement = coarseGrid["xMax"]
        df0 = 2 * coarseGrid["deltax"]
        df0original = 2 * df0original

    # Loop over inspiral derefinement grids
    if nDerefineInspiralGrids > 0:
        index = 0
        nextfSTART = fStartInspDerefinement
        FrequencyFactor = jnp.power(2.0, 1.0 / dfpower)

        while index < nDerefineInspiralGrids:
            if df0original < evaldMf:
                mydf = evaldMf
                intdfRatio = 1
            else:
                intdfRatio = int(jnp.floor(df0original / evaldMf))
                mydf = evaldMf * intdfRatio

            if index + preComputeFirstGrid == 0:
                fSTART = nextfSTART
            else:
                fSTART = nextfSTART + mydf

            fEND = fSTART * FrequencyFactor

            coarseGrid = XLALSimIMRPhenomXGridComp(fSTART, fEND, mydf)

            df0original = 2 * df0original
            nextfSTART = coarseGrid["xMax"]

            allGrids[index + preComputeFirstGrid] = coarseGrid
            allGrids[index + preComputeFirstGrid]["intdfRatio"] = intdfRatio

            index = index + 1

        fEndInsp = coarseGrid["xMax"]
    else:
        if preComputeFirstGrid > 0:
            fEndInsp = coarseGrid["xMax"]

    # Add merger grid
    if nMergerGrid > 0:
        df0original = dfmerger
        if 2 * coarseGrid["deltax"] < dfmerger:
            df0original = 2 * coarseGrid["deltax"]

        if df0original < evaldMf:
            mydf = evaldMf
            intdfRatio = 1
        else:
            intdfRatio = int(jnp.floor(df0original / evaldMf))
            mydf = evaldMf * intdfRatio

        fSTART = fEndInsp + mydf
        if fEndInsp == fstartIn:
            fSTART = fEndInsp

        mergerIndex = preComputeFirstGrid + nDerefineInspiralGrids

        if fSTART <= MfLorentzianEnd:
            coarseGrid = XLALSimIMRPhenomXGridComp(fSTART, MfLorentzianEnd, mydf)
            df0original = 2 * df0original

            allGrids[mergerIndex] = coarseGrid
            allGrids[mergerIndex]["intdfRatio"] = intdfRatio
        else:
            nMergerGrid = 0

    # Add ringdown grid
    if nRingdownGrid > 0:
        df0original = dfringdown
        if df0original < evaldMf:
            mydf = evaldMf
            intdfRatio = 1
        else:
            intdfRatio = int(jnp.floor(df0original / evaldMf))
            mydf = evaldMf * intdfRatio

        fSTART = coarseGrid["xMax"] + mydf
        if coarseGrid["xMax"] == fstartIn:
            fSTART = fEndInsp

        RDindex = preComputeFirstGrid + nDerefineInspiralGrids + nMergerGrid

        if fSTART <= Mfmax:
            coarseGrid = XLALSimIMRPhenomXGridComp(fSTART, Mfmax, mydf)
            df0original = 2 * df0original

            allGrids[RDindex] = coarseGrid
            allGrids[RDindex]["intdfRatio"] = intdfRatio
        else:
            nRingdownGrid = 0

    nGridsUsed = (
        preComputeFirstGrid + nDerefineInspiralGrids + nMergerGrid + nRingdownGrid
    )

    # Trim the list to actual size
    allGrids = [g for g in allGrids[:nGridsUsed] if g is not None]

    return allGrids, nGridsUsed


def deltaF_mergerBin(fdamp: float, alpha4: float, abserror: float) -> float:
    """
    Right hand side of eq. 2.27 in arXiv:2001.10897.

    Calculates the frequency bin size for the merger region.

    Args:
        fdamp: Damping frequency (float)
        alpha4: Alpha4 coefficient (float)
        abserror: Absolute error tolerance (float)

    Returns:
        float: Frequency bin size for merger
    """
    aux = jnp.sqrt(jnp.sqrt(3.0) * 3.0)
    return 4.0 * fdamp * jnp.sqrt(abserror / jnp.abs(alpha4)) / aux


def deltaF_ringdownBin(
    fdamp: float, alpha4: float, LAMBDA: float, abserror: float
) -> float:
    """
    Correspond to eqs. 2.28 and 2.31 in arXiv:2001.10897.

    Calculates the frequency bin size for the ringdown region based on
    phase and amplitude contributions, returning the minimum.

    Args:
        fdamp: Damping frequency (float)
        alpha4: Alpha4 coefficient (float)
        LAMBDA: Lambda parameter (float)
        abserror: Absolute error tolerance (float)

    Returns:
        float: Frequency bin size for ringdown (minimum of phase and amplitude contributions)
    """
    dfphase = 5.0 * fdamp * jnp.sqrt(abserror * 0.5 / jnp.abs(alpha4))
    dfamp = jnp.sqrt(2.0 * abserror) / jnp.abs(LAMBDA)

    return jnp.where(dfphase <= dfamp, dfphase, dfamp)
