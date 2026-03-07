import jax
import jax.numpy as jnp
from typing import Dict, Callable

from .LALSimIMRPhenomTHM_fits import (
    evaluate_QNMfit_fring21,
    evaluate_QNMfit_fring33,
    evaluate_QNMfit_fring32,
    evaluate_QNMfit_fring44,
)

from .LALSimIMRPhenomTHM_fits import (
    evaluate_QNMfit_fdamp21,
    evaluate_QNMfit_fdamp33,
    evaluate_QNMfit_fdamp32,
    evaluate_QNMfit_fdamp44,
)

from .LALSimIMRPhenomTHM_fits import (
    evaluate_QNMfit_re_l2m2lp2,
    evaluate_QNMfit_im_l2m2lp2,
    evaluate_QNMfit_re_l2m2lp3,
    evaluate_QNMfit_im_l2m2lp3,
    evaluate_QNMfit_re_l3m2lp2,
    evaluate_QNMfit_im_l3m2lp2,
    evaluate_QNMfit_re_l3m2lp3,
    evaluate_QNMfit_im_l3m2lp3,
)


from .LALSimIMRPhenomXHM_ringdown import (
    IMRPhenomXHM_RD_Phase_Ansatz,
    IMRPhenomXHM_RD_Phase_AnsatzInt,
    IMRPhenomXHM_RD_Phase_DerAnsatz,
    IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift,
    IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift,
    IMRPhenomXHM_RD_Phase_22_alpha2,
    IMRPhenomXHM_RD_Phase_22_alphaL,
)

from .LALSimIMRPhenomXHM_ringdown import (
    IMRPhenomXHM_RD_Amp_Ansatz,
    IMRPhenomXHM_RD_Amp_DAnsatz,
    IMRPhenomXHM_RD_Amp_NDAnsatz,
    IMRPhenomXHM_RD_Amp_Coefficients,
)


from .LALSimIMRPhenomX_internals import (
    IMRPhenomXGetPhaseCoefficients,
    IMRPhenomX_dPhase_22,
    IMRPhenomX_Phase_22,
    IMRPhenomX_TimeShift_22,
    IMRPhenomX_Phase_22_ConnectionCoefficients,
)


from .LALSimIMRPhenomXHM_inspiral import (
    IMRPhenomXHM_Insp_Phase_LambdaPN,
    IMRPhenomXHM_Inspiral_Phase_AnsatzInt,
    IMRPhenomXHM_Inspiral_Phase_Ansatz,
)

from .LALSimIMRPhenomXHM_inspiral import (
    IMRPhenomXHM_Inspiral_PNAmp_Ansatz,
    IMRPhenomXHM_Inspiral_Amp_Ansatz,
    IMRPhenomXHM_Inspiral_Amp_NDAnsatz,
    IMRPhenomXHM_Inspiral_Amp_rho1,
    IMRPhenomXHM_Inspiral_Amp_rho2,
    IMRPhenomXHM_Inspiral_Amp_rho3,
    IMRPhenomXHM_Inspiral_Amplitude_Veto,
    IMRPhenomXHM_Get_Inspiral_Amp_Coefficients,
)

from .LALSimIMRPhenomXHM_intermediate import (
    IMRPhenomXHM_Inter_Phase_AnsatzInt,
    IMRPhenomXHM_Inter_Phase_Ansatz,
)

from .LALSimIMRPhenomXHM_intermediate import (
    IMRPhenomXHM_Intermediate_Amp_Coefficients,
    IMRPhenomXHM_Intermediate_Amp_delta0,
    IMRPhenomXHM_Intermediate_Amp_delta1,
    IMRPhenomXHM_Intermediate_Amp_delta2,
    IMRPhenomXHM_Intermediate_Amp_delta3,
    IMRPhenomXHM_Intermediate_Amp_delta4,
    IMRPhenomXHM_Intermediate_Amp_delta5,
)


from .LALSimIMRPhenomXUtilities import IMRPhenomXPsi4ToStrain

from . import LALSimIMRPhenomXHM_inspiral as XHM_inspiral
from . import LALSimIMRPhenomXHM_intermediate as XHM_intermediate

from . import LALSimIMRPhenomXHM_ringdown as XHM_ringdown


def WavyPoints(p1: float, p2: float, p3: float) -> int:
    """
    Check if the three collocation points are wavy (non-monotonic).

    Parameters
    ----------
    p1, p2, p3 : float
        Three point values to check

    Returns
    -------
    int
        1 if points are wavy, 0 otherwise
    """
    if (p1 > p2 and p2 < p3) or (p1 < p2 and p2 > p3):
        return 1
    else:
        return 0


def IMRPhenomXHM_GenerateRingdownFrequency(ell: int, emm: int, wf22: dict) -> float:
    """
    Wrapper function to return ringdown frequency

    Args:
        ell: Spherical harmonic l index (int)
        emm: Spherical harmonic m index (int) - guaranteed to be positive
        wf22: Waveform structure dictionary (dict)

    Returns:
        float: Ringdown frequency
    """
    # emm is guaranteed to be positive
    modeTag = ell * 10 + emm

    # if the tuned coprecessing tuning is activated, use the precessing final spin
    afinal = jnp.where(
        wf22["IMRPhenomXPNRUseTunedCoprec"], wf22["afinal_prec"], wf22["afinal"]
    )

    def case_21():
        return evaluate_QNMfit_fring21(afinal) / wf22["Mfinal"]

    def case_22():
        return wf22["fRING"]

    def case_33():
        return evaluate_QNMfit_fring33(afinal) / wf22["Mfinal"]

    def case_32():
        return evaluate_QNMfit_fring32(afinal) / wf22["Mfinal"]

    def case_44():
        return evaluate_QNMfit_fring44(afinal) / wf22["Mfinal"]

    def default_case():
        # In JAX, we can't raise errors in jit-compiled code
        # Return NaN or handle error upstream
        return jnp.nan

    # Create branches for each case
    fRING = jax.lax.switch(
        jnp.where(
            modeTag == 21,
            0,
            jnp.where(
                modeTag == 22,
                1,
                jnp.where(
                    modeTag == 33,
                    2,
                    jnp.where(modeTag == 32, 3, jnp.where(modeTag == 44, 4, 5)),
                ),
            ),
        ),
        [case_21, case_22, case_33, case_32, case_44, default_case],
    )

    # if the coprecessing tuning is activated, return Effective RD frequency
    fRING = jnp.where(
        jnp.logical_and(
            wf22["IMRPhenomXPNRUseTunedCoprec"], jnp.logical_or(ell != 2, emm != 2)
        ),
        fRING - emm * wf22["fRINGEffShiftDividedByEmm"],
        fRING,
    )

    return fRING


def IMRPhenomXHM_Initialize_QNMs() -> Dict[str, Dict[int, Callable]]:
    """
    Initialize QNM (Quasi-Normal Mode) fit functions for different modes.

    This function creates dictionaries mapping mode indices to their corresponding
    fring and fdamp QNM fit functions. The mode indices are:
        0: (2,1) mode
        1: (3,3) mode
        2: (3,2) mode
        3: (4,4) mode

    Returns:
        dict: Dictionary containing 'fring_lm' and 'fdamp_lm' subdictionaries,
              each mapping mode indices to their respective fit functions.


    """
    # Note: In Python/JAX, we use dictionaries instead of C-style function pointer arrays
    # This is more Pythonic and works well with JAX's functional programming style

    qnms = {
        "fring_lm": {
            0: evaluate_QNMfit_fring21,
            1: evaluate_QNMfit_fring33,
            2: evaluate_QNMfit_fring32,
            3: evaluate_QNMfit_fring44,
        },
        "fdamp_lm": {
            0: evaluate_QNMfit_fdamp21,
            1: evaluate_QNMfit_fdamp33,
            2: evaluate_QNMfit_fdamp32,
            3: evaluate_QNMfit_fdamp44,
        },
    }

    return qnms


def IMRPhenomXHM_Initialize_MixingCoeffs(wf: dict, wf22: dict) -> dict:
    """
    Initialize mixing coefficients for (3,2) mode.

    The mixing coefficients are used to transform the spheroidal-harmonic
    ringdown ansatz back to spherical-harmonic representation.
    Only the (3,2) mode has mixing with other modes.

    Args:
        wf: Waveform dictionary for the higher mode
        wf22: Waveform dictionary for the (2,2) mode

    Returns:
        dict: Updated waveform dictionary with mixing coefficients
    """
    # Initialize mixing coefficients array (4 complex numbers)
    # In C: wf->mixingCoeffs[0] through wf->mixingCoeffs[3]
    # In Python/JAX: store as array of complex numbers

    afinal = wf22["afinal"]

    # Compute complex mixing coefficients
    # mixingCoeffs[0] corresponds to l=2, m=2, lp=2
    a222 = evaluate_QNMfit_re_l2m2lp2(afinal) + 1j * evaluate_QNMfit_im_l2m2lp2(afinal)

    # mixingCoeffs[1] corresponds to l=2, m=2, lp=3
    a223 = evaluate_QNMfit_re_l2m2lp3(afinal) + 1j * evaluate_QNMfit_im_l2m2lp3(afinal)

    # mixingCoeffs[2] corresponds to l=3, m=2, lp=2
    a322 = evaluate_QNMfit_re_l3m2lp2(afinal) + 1j * evaluate_QNMfit_im_l3m2lp2(afinal)

    # mixingCoeffs[3] corresponds to l=3, m=2, lp=3
    a323 = evaluate_QNMfit_re_l3m2lp3(afinal) + 1j * evaluate_QNMfit_im_l3m2lp3(afinal)

    # Adjust conventions so that they match the ones used for the hybrids
    # In the C code: wf->mixingCoeffs[2] = -1.* wf->mixingCoeffs[2]
    #                wf->mixingCoeffs[3] = -1.* wf->mixingCoeffs[3]
    a322 = -1.0 * a322
    a323 = -1.0 * a323

    # Store in waveform dictionary as a list
    wf["mixingCoeffs"] = jnp.array([a222, a223, a322, a323])

    return wf


def IMRPhenomXHM_SetHMWaveformVariables(
    ell: int, emm: int, wf22: dict, qnms: dict, LALParams: dict = None
) -> dict:
    """
    Set higher mode waveform variables for IMRPhenomXHM.

    This function initializes all the parameters needed for generating
    a specific higher mode (ell, emm) waveform based on the 22 mode
    properties and QNM fits.

    Args:
        ell: Spherical harmonic l index
        emm: Spherical harmonic m index (absolute value)
        wf22: Dictionary containing 22 mode waveform parameters
        qnms: Dictionary containing QNM fit functions
        LALParams: Optional dictionary for LAL parameters

    Returns:
        dict: Dictionary containing all waveform variables for the (ell, emm) mode
    """
    # Initialize LALParams if not provided
    if LALParams is None:
        LALParams = {}

    # Initialize waveform dictionary
    wf = {}

    # Read in which mode is being generated
    wf["ell"] = ell
    wf["emm"] = emm
    wf["modeTag"] = ell * 10 + emm  # 21, 33, 32, 44
    wf["ampNorm"] = wf22["ampNorm"]
    wf["fMECOlm"] = wf22["fMECO"] * emm * 0.5
    wf["Ampzero"] = 0  # Ampzero = 1 (true) for odd modes and equal black holes
    wf["Amp0"] = wf22["amp0"]
    wf["useFAmpPN"] = (
        0  # Only true for the 21, this mode has a different inspiral ansatz
    )
    wf["AmpEMR"] = 0  # Only one intermediate region
    wf["InspiralAmpVeto"] = 0
    wf["IntermediateAmpVeto"] = 0
    wf["RingdownAmpVeto"] = 0

    # Mode-specific settings
    modeTag = wf["modeTag"]
    print("jax debug 6...modeTag", wf["modeTag"])

    # Determine modeInt and MixingOn based on modeTag
    def set_mode_21():
        return (
            0,
            0,
            jnp.where(
                jnp.logical_and(wf22["q"] == 1.0, wf22["chi1L"] == wf22["chi2L"]), 1, 0
            ),
        )

    def set_mode_33():
        return (
            1,
            0,
            jnp.where(
                jnp.logical_and(wf22["q"] == 1.0, wf22["chi1L"] == wf22["chi2L"]), 1, 0
            ),
        )

    def set_mode_32():
        return 2, 1, 0

    def set_mode_44():
        return 3, 0, 0

    def default_mode():
        return -1, 0, 0

    modeInt, MixingOn, Ampzero_mode = jax.lax.switch(
        jnp.where(
            modeTag == 21,
            0,
            jnp.where(
                modeTag == 33,
                1,
                jnp.where(modeTag == 32, 2, jnp.where(modeTag == 44, 3, 4)),
            ),
        ),
        [set_mode_21, set_mode_33, set_mode_32, set_mode_44, default_mode],
    )

    modeTag = wf["modeTag"]

    wf["modeInt"] = modeInt
    wf["MixingOn"] = MixingOn
    wf["Ampzero"] = Ampzero_mode
    print("jax debug 6...modeTag", wf["modeTag"], wf["MixingOn"])

    # Version numbers - default to 122019 if not specified in LALParams
    wf["IMRPhenomXHMReleaseVersion"] = LALParams.get(
        "IMRPhenomXHMReleaseVersion", 122019
    )

    wf["IMRPhenomXHMInspiralPhaseVersion"] = LALParams.get(
        "IMRPhenomXHMInspiralPhaseVersion", 122019
    )
    wf["IMRPhenomXHMIntermediatePhaseVersion"] = LALParams.get(
        "IMRPhenomXHMIntermediatePhaseVersion", 122019
    )
    wf["IMRPhenomXHMRingdownPhaseVersion"] = LALParams.get(
        "IMRPhenomXHMRingdownPhaseVersion", 122019
    )

    wf["IMRPhenomXHMInspiralAmpFitsVersion"] = LALParams.get(
        "IMRPhenomXHMInspiralAmpFitsVersion", 122018
    )
    wf["IMRPhenomXHMIntermediateAmpFitsVersion"] = LALParams.get(
        "IMRPhenomXHMIntermediateAmpFitsVersion", 122018
    )
    wf["IMRPhenomXHMRingdownAmpFitsVersion"] = LALParams.get(
        "IMRPhenomXHMRingdownAmpFitsVersion", 122018
    )

    wf["IMRPhenomXHMInspiralAmpFreqsVersion"] = LALParams.get(
        "IMRPhenomXHMInspiralAmpFreqsVersion", 122018
    )
    wf["IMRPhenomXHMIntermediateAmpFreqsVersion"] = LALParams.get(
        "IMRPhenomXHMIntermediateAmpFreqsVersion", 122018
    )
    wf["IMRPhenomXHMRingdownAmpFreqsVersion"] = LALParams.get(
        "IMRPhenomXHMRingdownAmpFreqsVersion", 122018
    )

    wf["IMRPhenomXHMInspiralAmpVersion"] = LALParams.get(
        "IMRPhenomXHMInspiralAmpVersion", 3
    )
    wf["IMRPhenomXHMIntermediateAmpVersion"] = LALParams.get(
        "IMRPhenomXHMIntermediateAmpVersion", 2
    )
    wf["IMRPhenomXHMRingdownAmpVersion"] = LALParams.get(
        "IMRPhenomXHMRingdownAmpVersion", 0
    )

    # Initialize HM tuning parameters (all zeros by default)
    wf["MU1"] = 0.0
    wf["MU2"] = 0.0
    wf["MU3"] = 0.0
    wf["MU4"] = 0.0
    wf["NU0"] = 0.0
    wf["NU4"] = 0.0
    wf["NU5"] = 0.0
    wf["NU6"] = 0.0
    wf["ZETA1"] = 0.0
    wf["ZETA2"] = 0.0

    # Special handling for (3,3) mode tuning
    if modeTag == 33:
        # Set PNR deviation parameter
        wf["PNR_DEV_PARAMETER"] = wf22["delta"] * wf22.get("PNR_DEV_PARAMETER", 0.0)

        # Check if tuned coprecessing model should be used
        IMRPhenomXPNRUseTunedCoprec33 = wf22.get("IMRPhenomXPNRUseTunedCoprec33", False)

        if IMRPhenomXPNRUseTunedCoprec33:
            # These would need to be imported from PhenomXCP fits
            # For now, we'll use LALParams values
            # In full implementation, these should call:
            # XLALSimIMRPhenomXCP_MU1_l3m3, XLALSimIMRPhenomXCP_MU2_l3m3, etc.
            wf["MU1"] = LALParams.get("PhenomXCPMU1l3m3", 0.0)
            wf["MU2"] = LALParams.get("PhenomXCPMU2l3m3", 0.0)
            wf["MU3"] = LALParams.get("PhenomXCPMU3l3m3", 0.0)
            wf["MU4"] = LALParams.get("PhenomXCPMU4l3m3", 0.0)
            wf["NU4"] = LALParams.get("PhenomXCPNU4l3m3", 0.0)
            wf["NU5"] = LALParams.get("PhenomXCPNU5l3m3", 0.0)
            wf["NU6"] = LALParams.get("PhenomXCPNU6l3m3", 0.0)
            wf["ZETA1"] = LALParams.get("PhenomXCPZETA1l3m3", 0.0)
            wf["ZETA2"] = LALParams.get("PhenomXCPZETA2l3m3", 0.0)
        else:
            # Use values from LALParams
            wf["MU1"] = LALParams.get("PhenomXCPMU1l3m3", 0.0)
            wf["MU2"] = LALParams.get("PhenomXCPMU2l3m3", 0.0)
            wf["MU3"] = LALParams.get("PhenomXCPMU3l3m3", 0.0)
            wf["MU4"] = LALParams.get("PhenomXCPMU4l3m3", 0.0)
            wf["NU0"] = LALParams.get("PhenomXCPNU0l3m3", 0.0)
            wf["NU4"] = LALParams.get("PhenomXCPNU4l3m3", 0.0)
            wf["NU5"] = LALParams.get("PhenomXCPNU5l3m3", 0.0)
            wf["NU6"] = LALParams.get("PhenomXCPNU6l3m3", 0.0)
            wf["ZETA1"] = LALParams.get("PhenomXCPZETA1l3m3", 0.0)
            wf["ZETA2"] = LALParams.get("PhenomXCPZETA2l3m3", 0.0)
    else:
        wf["PNR_DEV_PARAMETER"] = 0.0

    # Set phase version variables
    wf["IMRPhenomXHMInspiralPhaseFitsVersion"] = wf["IMRPhenomXHMInspiralPhaseVersion"]
    wf["IMRPhenomXHMIntermediatePhaseFitsVersion"] = wf[
        "IMRPhenomXHMIntermediatePhaseVersion"
    ]
    wf["IMRPhenomXHMRingdownPhaseFitsVersion"] = wf["IMRPhenomXHMRingdownPhaseVersion"]

    wf["IMRPhenomXHMInspiralPhaseFreqsVersion"] = wf["IMRPhenomXHMInspiralPhaseVersion"]
    wf["IMRPhenomXHMIntermediatePhaseFreqsVersion"] = wf[
        "IMRPhenomXHMIntermediatePhaseVersion"
    ]
    wf["IMRPhenomXHMRingdownPhaseFreqsVersion"] = wf["IMRPhenomXHMRingdownPhaseVersion"]

    wf["nCollocPtsRDPhase"] = 0

    # Release-specific settings
    if wf["IMRPhenomXHMReleaseVersion"] == 122019:
        # For q>70 and chi1<0.9 use two intermediate regions
        if wf22["eta"] < 0.013886133703630232 and wf22["chi1L"] <= 0.9:
            wf["AmpEMR"] = 1

        # Mode-specific vetos and settings for 122019
        if modeTag == 21:
            if wf22["q"] < 8.0:
                wf["InspiralAmpVeto"] = 1
                wf["IntermediateAmpVeto"] = 1
            wf["RingdownAmpVeto"] = 1
            if wf22["eta"] >= 0.0237954:  # for EMR (q>40)
                wf["useFAmpPN"] = 1
        elif modeTag == 33:
            if wf22["q"] == 1.0 and wf22["chi1L"] == wf22["chi2L"]:
                wf["Ampzero"] = 1
        elif modeTag == 32:
            wf["RingdownAmpVeto"] = 1
            wf["nCollocPtsRDPhase"] = 4

        wf["nCollocPtsInspAmp"] = wf["IMRPhenomXHMInspiralAmpVersion"]
        wf["nCollocPtsInterAmp"] = wf["IMRPhenomXHMIntermediateAmpVersion"]
        wf["IMRPhenomXHMRingdownPhaseFreqsVersion"] = 122019

    elif wf["IMRPhenomXHMReleaseVersion"] == 122022:
        # Amplitude versions for 122022 release
        wf["IMRPhenomXHMInspiralAmpFreqsVersion"] = 122022
        wf["IMRPhenomXHMIntermediateAmpFreqsVersion"] = 0
        wf["IMRPhenomXHMRingdownAmpFreqsVersion"] = 122022

        wf["IMRPhenomXHMInspiralAmpFitsVersion"] = 122022
        wf["IMRPhenomXHMIntermediateAmpFitsVersion"] = 122022
        wf["IMRPhenomXHMRingdownAmpFitsVersion"] = 122022

        wf["IMRPhenomXHMInspiralAmpVersion"] = 123
        wf["IMRPhenomXHMIntermediateAmpVersion"] = 211112
        wf["IMRPhenomXHMRingdownAmpVersion"] = 2

        # Mode-specific tweaks
        if modeTag == 21:
            wf["IMRPhenomXHMIntermediateAmpVersion"] = 110102
        elif modeTag == 32:
            wf["IMRPhenomXHMRingdownAmpVersion"] = 1
            wf["nCollocPtsRDPhase"] = 4

        # Calculate number of collocation points from version number
        wf["nCollocPtsInspAmp"] = len(str(wf["IMRPhenomXHMInspiralAmpVersion"]))
        wf["nCollocPtsInterAmp"] = len(str(wf["IMRPhenomXHMIntermediateAmpVersion"]))
    else:
        raise ValueError(
            f"IMRPhenomXHMReleaseVersion={wf['IMRPhenomXHMReleaseVersion']} is not valid."
        )

    # Common for all releases
    if modeTag == 32:
        wf["nCollocPtsInterPhase"] = 6
    else:
        wf["nCollocPtsInterPhase"] = 5

    # Limit between comparable and extreme mass ratios for the phase
    wf["etaEMR"] = 0.05

    # Spin parameterizations
    wf["chi_s"] = (wf22["chi1L"] + wf22["chi2L"]) * 0.5
    wf["chi_a"] = (wf22["chi1L"] - wf22["chi2L"]) * 0.5

    # Ringdown and damping frequencies
    afinal = wf22["afinal"]
    # Convert modeInt to Python int for indexing (JAX arrays are not hashable)
    modeInt_idx = int(modeInt)
    wf["fRING"] = qnms["fring_lm"][modeInt_idx](afinal) / wf22["Mfinal"]
    wf["fDAMP"] = qnms["fdamp_lm"][modeInt_idx](afinal) / wf22["Mfinal"]

    # Apply precessing corrections if using tuned coprecessing model
    IMRPhenomXPNRUseTunedCoprec = wf22.get("IMRPhenomXPNRUseTunedCoprec", False)

    if IMRPhenomXPNRUseTunedCoprec:
        afinal_prec = wf22.get("afinal_prec", afinal)
        wf["fRING"] = qnms["fring_lm"][modeInt_idx](afinal_prec) / wf22["Mfinal"]
        wf["fDAMP"] = qnms["fdamp_lm"][modeInt_idx](afinal_prec) / wf22["Mfinal"]

        # Apply effective ringdown frequency shift
        fRINGEffShiftDividedByEmm = wf22.get("fRINGEffShiftDividedByEmm", 0.0)
        wf["fRING"] = wf["fRING"] - emm * fRINGEffShiftDividedByEmm

    # Apply (3,3) mode specific corrections
    IMRPhenomXPNRUseTunedCoprec33 = wf22.get("IMRPhenomXPNRUseTunedCoprec33", False)
    if IMRPhenomXPNRUseTunedCoprec33 and modeTag == 33:
        # Apply PNR CoPrec deviations
        wf["fRING"] = wf["fRING"] - (wf["PNR_DEV_PARAMETER"] * wf["NU5"])
        wf["fDAMP"] = wf["fDAMP"] + (wf["PNR_DEV_PARAMETER"] * wf["NU6"])

        if IMRPhenomXPNRUseTunedCoprec:
            # Transition to EZH effective ringdown frequency
            pnr_window = wf22.get("pnr_window", 1.0)
            fRINGEffShiftDividedByEmm = wf22.get("fRINGEffShiftDividedByEmm", 0.0)
            wf["fRING"] = (
                wf["fRING"] - (1.0 - pnr_window) * emm * fRINGEffShiftDividedByEmm
            )

    # If (l,m)=(3,2), initialize mixing coefficients
    if modeTag == 32:
        wf["MixingOn"] = 1
        wf = IMRPhenomXHM_Initialize_MixingCoeffs(wf, wf22)

    # Phase shift and time shift

    wf["timeshift"] = 0.0
    wf["phaseshift"] = 0.0

    # Current time-alignment of the hybrids
    psi4tostrain = IMRPhenomXPsi4ToStrain(wf22["eta"], wf22["STotR"], wf22["dchi"])
    wf["DeltaT"] = -2.0 * jnp.pi * (500.0 + psi4tostrain)

    wf["fPhaseRDflat"] = 0.0
    wf["fAmpRDfalloff"] = 0.0

    return wf


def IMRPhenomXHM_FillAmpFitsArray() -> dict:
    """
    Fill arrays with function references for amplitude coefficient/collocation point fits.

    This function creates a mapping of fit functions for amplitude coefficients
    and collocation points across three regions (inspiral, intermediate, ringdown)
    and four modes (21, 33, 32, 44).

    Returns:
        dict: Dictionary containing three arrays:
            - 'InspiralAmpFits': List of 12 functions for inspiral amplitude fits
            - 'IntermediateAmpFits': List of 24 functions for intermediate amplitude fits
            - 'RingdownAmpFits': List of 26 functions for ringdown amplitude fits


    """
    pAmp = {}

    # In the full implementation, these would be function references
    pAmp["InspiralAmpFits"] = [None] * 12
    pAmp["IntermediateAmpFits"] = [None] * 24
    pAmp["RingdownAmpFits"] = [None] * 26

    # ******Inspiral Fits for collocation points******
    # Each mode (21, 33, 32, 44) has 3 collocation points at different frequencies

    # Mode 21
    pAmp["InspiralAmpFits"][0] = XHM_inspiral.IMRPhenomXHM_Insp_Amp_21_iv1  # fcutInsp
    pAmp["InspiralAmpFits"][1] = (
        XHM_inspiral.IMRPhenomXHM_Insp_Amp_21_iv2
    )  # fcutInsp*0.75
    pAmp["InspiralAmpFits"][2] = (
        XHM_inspiral.IMRPhenomXHM_Insp_Amp_21_iv3
    )  # fcutInsp*0.5

    # Mode 33
    pAmp["InspiralAmpFits"][3] = XHM_inspiral.IMRPhenomXHM_Insp_Amp_33_iv1  # fcutInsp
    pAmp["InspiralAmpFits"][4] = (
        XHM_inspiral.IMRPhenomXHM_Insp_Amp_33_iv2
    )  # fcutInsp*0.75
    pAmp["InspiralAmpFits"][5] = (
        XHM_inspiral.IMRPhenomXHM_Insp_Amp_33_iv3
    )  # fcutInsp*0.5

    # Mode 32
    pAmp["InspiralAmpFits"][6] = XHM_inspiral.IMRPhenomXHM_Insp_Amp_32_iv1  # fcutInsp
    pAmp["InspiralAmpFits"][7] = (
        XHM_inspiral.IMRPhenomXHM_Insp_Amp_32_iv2
    )  # fcutInsp*0.75
    pAmp["InspiralAmpFits"][8] = (
        XHM_inspiral.IMRPhenomXHM_Insp_Amp_32_iv3
    )  # fcutInsp*0.5

    # Mode 44
    pAmp["InspiralAmpFits"][9] = XHM_inspiral.IMRPhenomXHM_Insp_Amp_44_iv1  # fcutInsp
    pAmp["InspiralAmpFits"][10] = (
        XHM_inspiral.IMRPhenomXHM_Insp_Amp_44_iv2
    )  # fcutInsp*0.75
    pAmp["InspiralAmpFits"][11] = (
        XHM_inspiral.IMRPhenomXHM_Insp_Amp_44_iv3
    )  # fcutInsp*0.5

    # *****Intermediate Fits for EMR collocation points, 2 Intermediate regions*****

    # Mode 21
    pAmp["IntermediateAmpFits"][0] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_21_int1
    )  # fcutInsp + (fcutRD-fcutInsp)/3
    pAmp["IntermediateAmpFits"][1] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_21_int2
    )  # fcutInsp + 2(fcutRD-fcutInsp)/3

    # Mode 33
    pAmp["IntermediateAmpFits"][2] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_33_int1
    )  # fcutInsp + (fcutRD-fcutInsp)/3
    pAmp["IntermediateAmpFits"][3] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_33_int2
    )  # fcutInsp + 2(fcutRD-fcutInsp)/3

    # Mode 32
    pAmp["IntermediateAmpFits"][4] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_32_int1
    )  # fcutInsp + (fcutRD-fcutInsp)/3
    pAmp["IntermediateAmpFits"][5] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_32_int2
    )  # fcutInsp + 2(fcutRD-fcutInsp)/3

    # Mode 44
    pAmp["IntermediateAmpFits"][6] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_44_int1
    )  # fcutInsp + (fcutRD-fcutInsp)/3
    pAmp["IntermediateAmpFits"][7] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_44_int2
    )  # fcutInsp + 2(fcutRD-fcutInsp)/3

    # Additional intermediate fits for EMR
    # Mode 21
    pAmp["IntermediateAmpFits"][8] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_21_int0
    )  # fcutInsp + (fInt1 - fcutInsp)/3
    pAmp["IntermediateAmpFits"][9] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_21_dint0
    )  # fcutInsp + (fInt1 - fcutInsp)/3

    # Mode 33
    pAmp["IntermediateAmpFits"][10] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_33_int0
    )  # fcutInsp + (fInt1 - fcutInsp)/3
    pAmp["IntermediateAmpFits"][11] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_33_dint0
    )  # fcutInsp + (fInt1 - fcutInsp)/3

    # Mode 32
    pAmp["IntermediateAmpFits"][12] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_32_int0
    )  # fcutInsp + (fInt1 - fcutInsp)/3
    pAmp["IntermediateAmpFits"][13] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_32_dint0
    )  # fcutInsp + (fInt1 - fcutInsp)/3

    # Mode 44
    pAmp["IntermediateAmpFits"][14] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_44_int0
    )  # fcutInsp + (fInt1 - fcutInsp)/3
    pAmp["IntermediateAmpFits"][15] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Amp_44_dint0
    )  # fcutInsp + (fInt1 - fcutInsp)/3

    # Additional intermediate fits
    # Mode 21
    pAmp["IntermediateAmpFits"][16] = XHM_intermediate.IMRPhenomXHM_Inter_Amp_21_int3
    pAmp["IntermediateAmpFits"][17] = XHM_intermediate.IMRPhenomXHM_Inter_Amp_21_int4

    # Mode 33
    pAmp["IntermediateAmpFits"][18] = XHM_intermediate.IMRPhenomXHM_Inter_Amp_33_int3
    pAmp["IntermediateAmpFits"][19] = XHM_intermediate.IMRPhenomXHM_Inter_Amp_33_int4

    # Mode 32
    pAmp["IntermediateAmpFits"][20] = XHM_intermediate.IMRPhenomXHM_Inter_Amp_32_int3
    pAmp["IntermediateAmpFits"][21] = XHM_intermediate.IMRPhenomXHM_Inter_Amp_32_int4

    # Mode 44
    pAmp["IntermediateAmpFits"][22] = XHM_intermediate.IMRPhenomXHM_Inter_Amp_44_int3
    pAmp["IntermediateAmpFits"][23] = XHM_intermediate.IMRPhenomXHM_Inter_Amp_44_int4

    # ****Ringdown Fits for coefficients****

    # Mode 21
    pAmp["RingdownAmpFits"][0] = XHM_ringdown.IMRPhenomXHM_RD_Amp_21_alambda
    pAmp["RingdownAmpFits"][1] = XHM_ringdown.IMRPhenomXHM_RD_Amp_21_lambda
    pAmp["RingdownAmpFits"][2] = XHM_ringdown.IMRPhenomXHM_RD_Amp_21_sigma

    # Mode 33
    pAmp["RingdownAmpFits"][3] = XHM_ringdown.IMRPhenomXHM_RD_Amp_33_alambda
    pAmp["RingdownAmpFits"][4] = XHM_ringdown.IMRPhenomXHM_RD_Amp_33_lambda
    pAmp["RingdownAmpFits"][5] = (
        XHM_ringdown.IMRPhenomXHM_RD_Amp_33_sigma
    )  # currently constant

    # Mode 32
    pAmp["RingdownAmpFits"][6] = XHM_ringdown.IMRPhenomXHM_RD_Amp_32_alambda
    pAmp["RingdownAmpFits"][7] = XHM_ringdown.IMRPhenomXHM_RD_Amp_32_lambda
    pAmp["RingdownAmpFits"][8] = (
        XHM_ringdown.IMRPhenomXHM_RD_Amp_32_sigma
    )  # currently constant

    # Mode 44
    pAmp["RingdownAmpFits"][9] = XHM_ringdown.IMRPhenomXHM_RD_Amp_44_alambda
    pAmp["RingdownAmpFits"][10] = XHM_ringdown.IMRPhenomXHM_RD_Amp_44_lambda
    pAmp["RingdownAmpFits"][11] = (
        XHM_ringdown.IMRPhenomXHM_RD_Amp_44_sigma
    )  # currently constant

    # ****Ringdown Fits for Collocation Points****

    # Mode 21
    pAmp["RingdownAmpFits"][12] = XHM_ringdown.IMRPhenomXHM_RD_Amp_21_rdcp1
    pAmp["RingdownAmpFits"][13] = XHM_ringdown.IMRPhenomXHM_RD_Amp_21_rdcp2
    pAmp["RingdownAmpFits"][14] = XHM_ringdown.IMRPhenomXHM_RD_Amp_21_rdcp3

    # Mode 33
    pAmp["RingdownAmpFits"][15] = XHM_ringdown.IMRPhenomXHM_RD_Amp_33_rdcp1
    pAmp["RingdownAmpFits"][16] = XHM_ringdown.IMRPhenomXHM_RD_Amp_33_rdcp2
    pAmp["RingdownAmpFits"][17] = XHM_ringdown.IMRPhenomXHM_RD_Amp_33_rdcp3

    # Mode 32
    pAmp["RingdownAmpFits"][18] = XHM_ringdown.IMRPhenomXHM_RD_Amp_32_rdcp1
    pAmp["RingdownAmpFits"][19] = XHM_ringdown.IMRPhenomXHM_RD_Amp_32_rdcp2
    pAmp["RingdownAmpFits"][20] = XHM_ringdown.IMRPhenomXHM_RD_Amp_32_rdcp3

    # Mode 44
    pAmp["RingdownAmpFits"][21] = XHM_ringdown.IMRPhenomXHM_RD_Amp_44_rdcp1
    pAmp["RingdownAmpFits"][22] = XHM_ringdown.IMRPhenomXHM_RD_Amp_44_rdcp2
    pAmp["RingdownAmpFits"][23] = XHM_ringdown.IMRPhenomXHM_RD_Amp_44_rdcp3

    # Mode 32 auxiliary fits
    pAmp["RingdownAmpFits"][24] = XHM_ringdown.IMRPhenomXHM_RD_Amp_32_rdaux1
    pAmp["RingdownAmpFits"][25] = XHM_ringdown.IMRPhenomXHM_RD_Amp_32_rdaux2

    return pAmp


def IMRPhenomXHM_FillPhaseFitsArray() -> dict:
    """
    Fill arrays with function references for phase coefficient/collocation point fits.

    This function creates a mapping of fit functions for phase coefficients
    and collocation points across three regions (inspiral, intermediate, ringdown)
    and four modes (21, 33, 32, 44).

    Returns:
        dict: Dictionary containing three arrays:
            - 'InspiralPhaseFits': List of 4 functions for inspiral phase lambda fits
            - 'IntermediatePhaseFits': List of 24 functions for intermediate phase fits
            - 'RingdownPhaseFits': List of 5 functions for ringdown phase fits (32 mode spheroidal)


    """
    pPhase = {}

    # In the full implementation, these would be function references
    pPhase["InspiralPhaseFits"] = [None] * 4
    pPhase["IntermediatePhaseFits"] = [None] * 24
    pPhase["RingdownPhaseFits"] = [None] * 5

    # ******Inspiral Phase Fits for lambda coefficients******

    # Mode 21
    pPhase["InspiralPhaseFits"][0] = XHM_inspiral.IMRPhenomXHM_Insp_Phase_21_lambda

    # Mode 33
    pPhase["InspiralPhaseFits"][1] = XHM_inspiral.IMRPhenomXHM_Insp_Phase_33_lambda

    # Mode 32
    pPhase["InspiralPhaseFits"][2] = XHM_inspiral.IMRPhenomXHM_Insp_Phase_32_lambda

    # Mode 44
    pPhase["InspiralPhaseFits"][3] = XHM_inspiral.IMRPhenomXHM_Insp_Phase_44_lambda

    # ******Intermediate Phase Fits for collocation points******

    # Mode 21 - 6 collocation points (p1 through p6)
    pPhase["IntermediatePhaseFits"][0] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_21_p1
    pPhase["IntermediatePhaseFits"][1] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_21_p2
    pPhase["IntermediatePhaseFits"][2] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_21_p3
    pPhase["IntermediatePhaseFits"][3] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_21_p4
    pPhase["IntermediatePhaseFits"][4] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_21_p5
    pPhase["IntermediatePhaseFits"][5] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_21_p6

    # Mode 33 - 6 collocation points (p1 through p6)
    pPhase["IntermediatePhaseFits"][6] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_33_p1
    pPhase["IntermediatePhaseFits"][7] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_33_p2
    pPhase["IntermediatePhaseFits"][8] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_33_p3
    pPhase["IntermediatePhaseFits"][9] = XHM_intermediate.IMRPhenomXHM_Inter_Phase_33_p4
    pPhase["IntermediatePhaseFits"][10] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_33_p5
    )
    pPhase["IntermediatePhaseFits"][11] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_33_p6
    )

    # Mode 32 - 6 collocation points (p1 through p6)
    pPhase["IntermediatePhaseFits"][12] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_32_p1
    )
    pPhase["IntermediatePhaseFits"][13] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_32_p2
    )
    pPhase["IntermediatePhaseFits"][14] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_32_p3
    )
    pPhase["IntermediatePhaseFits"][15] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_32_p4
    )
    pPhase["IntermediatePhaseFits"][16] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_32_p5
    )
    pPhase["IntermediatePhaseFits"][17] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_32_p6
    )

    # Mode 44 - 6 collocation points (p1 through p6)
    pPhase["IntermediatePhaseFits"][18] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_44_p1
    )
    pPhase["IntermediatePhaseFits"][19] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_44_p2
    )
    pPhase["IntermediatePhaseFits"][20] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_44_p3
    )
    pPhase["IntermediatePhaseFits"][21] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_44_p4
    )
    pPhase["IntermediatePhaseFits"][22] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_44_p5
    )
    pPhase["IntermediatePhaseFits"][23] = (
        XHM_intermediate.IMRPhenomXHM_Inter_Phase_44_p6
    )

    # ******Ringdown Phase Fits (32 Spheroidal)******

    # Mode 32 - 5 collocation points for spheroidal ringdown phase
    pPhase["RingdownPhaseFits"][0] = XHM_ringdown.IMRPhenomXHM_RD_Phase_32_p1
    pPhase["RingdownPhaseFits"][1] = XHM_ringdown.IMRPhenomXHM_RD_Phase_32_p2
    pPhase["RingdownPhaseFits"][2] = XHM_ringdown.IMRPhenomXHM_RD_Phase_32_p3
    pPhase["RingdownPhaseFits"][3] = XHM_ringdown.IMRPhenomXHM_RD_Phase_32_p4
    pPhase["RingdownPhaseFits"][4] = XHM_ringdown.IMRPhenomXHM_RD_Phase_32_p5

    return pPhase


def compute_powers_of_f(f: float) -> dict:
    """
    Compute useful powers of frequency for phase/amplitude calculations.

    Args:
        f: Frequency value

    Returns:
        Dictionary containing various powers of f
    """
    return {
        "itself": f,
        "one_sixth": f ** (1.0 / 6.0),
        "one_third": f ** (1.0 / 3.0),
        "two_thirds": f ** (2.0 / 3.0),
        "four_thirds": f ** (4.0 / 3.0),
        "five_thirds": f ** (5.0 / 3.0),
        "seven_sixths": f ** (7.0 / 6.0),
        "seven_thirds": f ** (7.0 / 3.0),
        "eight_thirds": f ** (8.0 / 3.0),
        "sqrt": jnp.sqrt(f),
        "two": f**2.0,
        "three": f**3.0,
        "four": f**4.0,
        "five": f**5.0,
        "m_one_sixth": f ** (-1.0 / 6.0),
        "m_one_third": f ** (-1.0 / 3.0),
        "m_two_thirds": f ** (-2.0 / 3.0),
        "m_four_thirds": f ** (-4.0 / 3.0),
        "m_five_thirds": f ** (-5.0 / 3.0),
        "m_seven_sixths": f ** (-7.0 / 6.0),
        "m_sqrt": f ** (-0.5),
        "m_one": 1.0 / f,
        "m_two": f ** (-2.0),
        "m_three": f ** (-3.0),
        "m_four": f ** (-4.0),
        "m_five": f ** (-5.0),
        "log": jnp.log(f),
    }


def GetSpheroidalCoefficients(
    pPhase: dict, pPhase22: dict, pWFHM: dict, pWF22: dict
) -> dict:
    """
    Compute spheroidal ringdown phase coefficients for higher modes.

    This function reconstructs the spheroidal ringdown phase derivative using collocation
    points and solves for the ansatz coefficients. It then adjusts the time and phase
    shifts relative to the (2,2) mode.

    Args:
        pPhase: Phase coefficient dictionary for the higher mode
        pPhase22: Phase coefficient dictionary for the (2,2) mode
        pWFHM: Waveform structure dictionary for the higher mode
        pWF22: Waveform structure dictionary for the (2,2) mode

    Returns:
        dict: Updated pPhase dictionary with spheroidal coefficients
    """

    nCollocationPts_RD_Phase = pWFHM["nCollocPtsRDPhase"]

    # Initialize collocation arrays
    CollocValuesPhaseRingdown = jnp.zeros(nCollocationPts_RD_Phase)
    CollocFreqsPhaseRingdown = jnp.zeros(nCollocationPts_RD_Phase)

    # Initialize GSL-equivalent arrays for JAX linear algebra
    b = jnp.zeros(nCollocationPts_RD_Phase)
    A = jnp.zeros((nCollocationPts_RD_Phase, nCollocationPts_RD_Phase))

    # Get collocation point frequencies
    pPhase = IMRPhenomXHM_Ringdown_CollocPtsFreqs(pPhase, pWFHM, pWF22)

    # Fill in the collocation points for the phase
    if pWFHM["IMRPhenomXHMRingdownPhaseVersion"] == 122019:
        for i in range(nCollocationPts_RD_Phase):
            CollocValuesPhaseRingdown = CollocValuesPhaseRingdown.at[i].set(
                pPhase["RingdownPhaseFits"][i](
                    pWF22, pWFHM["IMRPhenomXHMRingdownPhaseFitsVersion"]
                )
            )
            CollocFreqsPhaseRingdown = CollocFreqsPhaseRingdown.at[i].set(
                pPhase["CollocationPointsFreqsPhaseRD"][i]
            )
            b = b.at[i].set(CollocValuesPhaseRingdown[i])

            ff = CollocFreqsPhaseRingdown[i]
            ffm1 = 1.0 / ff
            ffm2 = ffm1 * ffm1
            lorentzian = pWFHM["fDAMP"] / (
                pWFHM["fDAMP"] ** 2 + (ff - pWFHM["fRING"]) ** 2
            )
            fpowers = jnp.array([1.0, lorentzian, ffm2, ffm2 * ffm2])

            for j in range(nCollocationPts_RD_Phase):
                A = A.at[i, j].set(fpowers[j])
    else:
        for i in range(nCollocationPts_RD_Phase):
            CollocValuesPhaseRingdown = CollocValuesPhaseRingdown.at[i].set(
                pPhase["RingdownPhaseFits"][i](
                    pWF22, pWFHM["IMRPhenomXHMRingdownPhaseFitsVersion"]
                )
            )
            CollocFreqsPhaseRingdown = CollocFreqsPhaseRingdown.at[i].set(
                pPhase["CollocationPointsFreqsPhaseRD"][i]
            )
            b = b.at[i].set(CollocValuesPhaseRingdown[i])

            ff = CollocFreqsPhaseRingdown[i]
            ffm1 = 1.0 / ff
            ffm2 = ffm1 * ffm1
            lorentzian = pWFHM["fDAMP"] / (
                pWFHM["fDAMP"] * pWFHM["fDAMP"]
                + (ff - pWFHM["fRING"]) * (ff - pWFHM["fRING"])
            )
            fpowers = jnp.array([1.0, ffm1, ffm2, ffm2 * ffm2, lorentzian])

            for j in range(nCollocationPts_RD_Phase):
                A = A.at[i, j].set(fpowers[j])

    # Solve the linear system using JAX (instead of GSL LU decomposition)
    x = jnp.linalg.solve(A, b)

    # Extract coefficients based on version
    if pWFHM["IMRPhenomXHMRingdownPhaseVersion"] == 122019:
        # ansatz: alpha0 + (alpha2)/(f^2) + (alpha4)/(f^4) + alphaL*(fdamplm)/((fdamplm)^2 + (f - fRDlm)^2)
        pPhase["alpha0_S"] = x[0]
        pPhase["alphaL_S"] = x[1]
        pPhase["alpha2_S"] = x[2]
        pPhase["alpha4_S"] = x[3]
    else:
        # Store all coefficients
        if "RDCoefficient" not in pPhase:
            pPhase["RDCoefficient"] = jnp.zeros(nCollocationPts_RD_Phase + 3)
        for i in range(nCollocationPts_RD_Phase):
            pPhase["RDCoefficient"] = pPhase["RDCoefficient"].at[i].set(x[i])

    # Adjust the relative time and phase shift of the final phiS wrt IMRPhenomX
    frefRD = pWF22["fRING"] + pWF22["fDAMP"]
    powers_of_FREF = compute_powers_of_f(frefRD)

    # Get time shift from fit
    tshift = IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift(
        pWF22, pWFHM["IMRPhenomXHMRingdownPhaseFitsVersion"]
    )

    # Compute connection coefficients for the 22 mode if not already computed
    if "c0" not in pPhase22:
        pPhase22 = IMRPhenomXGetPhaseCoefficients(pWF22)

    # Compute time shift for 22 mode
    pWFHM["timeshift"] = IMRPhenomX_TimeShift_22(pPhase22, pWF22)

    pPhase["phi0_S"] = 0.0

    # Impose that dphiS(fref) - dphi22(fref) has the value given by our fit
    if pWFHM["IMRPhenomXHMRingdownPhaseVersion"] == 122019:
        dphi22ref = (1.0 / pWF22["eta"]) * IMRPhenomX_dPhase_22(
            frefRD, powers_of_FREF, pPhase22, pWF22
        ) + pWFHM["timeshift"]
        pPhase["alpha0_S"] = (
            pPhase["alpha0_S"]
            + dphi22ref
            + tshift
            - IMRPhenomXHM_RD_Phase_Ansatz(frefRD, powers_of_FREF, pWFHM, pPhase)
        )
    else:
        dphi22ref = (1.0 / pWF22["eta"]) * IMRPhenomX_dPhase_22(
            frefRD, powers_of_FREF, pPhase22, pWF22
        ) + pWFHM["timeshift"]
        dphi32ref = IMRPhenomXHM_RD_Phase_Ansatz(frefRD, powers_of_FREF, pWFHM, pPhase)
        pPhase["RDCoefficient"] = (
            pPhase["RDCoefficient"].at[0].add(dphi22ref + tshift - dphi32ref)
        )

        # Compute additional coefficients for flattening at high frequencies
        exponent = 5  # a + b / f^5
        ff = pWFHM["fRING"] + 2 * pWFHM["fDAMP"]
        powers_of_ff = compute_powers_of_f(ff)

        pPhase["RDCoefficient"] = (
            pPhase["RDCoefficient"]
            .at[nCollocationPts_RD_Phase + 1]
            .set(
                -IMRPhenomXHM_RD_Phase_DerAnsatz(ff, powers_of_ff, pWFHM, pPhase)
                * ff ** (exponent + 1)
                / exponent
            )
        )
        pPhase["RDCoefficient"] = (
            pPhase["RDCoefficient"]
            .at[nCollocationPts_RD_Phase]
            .set(
                IMRPhenomXHM_RD_Phase_Ansatz(ff, powers_of_ff, pWFHM, pPhase)
                - pPhase["RDCoefficient"][nCollocationPts_RD_Phase + 1] / ff**exponent
            )
        )
        pPhase["RDCoefficient"] = (
            pPhase["RDCoefficient"]
            .at[nCollocationPts_RD_Phase + 2]
            .set(
                IMRPhenomXHM_RD_Phase_AnsatzInt(ff, powers_of_ff, pWFHM, pPhase)
                - (
                    pPhase["RDCoefficient"][5] * ff
                    - 0.25 * pPhase["RDCoefficient"][6] * powers_of_ff["m_four"]
                )
            )
        )
        pWFHM["fPhaseRDflat"] = ff

    # Phase-shift of spheroidal ansatz
    frefRD = pWF22["fRING"]
    powers_of_FREF = compute_powers_of_f(frefRD)

    # Compute phi22(fref)
    powers_of_MfRef = compute_powers_of_f(pWF22["MfRef"])
    pWFHM["phiref22"] = (
        -1.0
        / pWF22["eta"]
        * IMRPhenomX_Phase_22(pWF22["MfRef"], powers_of_MfRef, pPhase22, pWF22)
        - pWFHM["timeshift"] * pWF22["MfRef"]
        - pWFHM["phaseshift"]
        + 2.0 * pWF22["phi0"]
        + jnp.pi / 4.0
    )

    phi22ref = (
        1.0
        / pWF22["eta"]
        * IMRPhenomX_Phase_22(frefRD, powers_of_FREF, pPhase22, pWF22)
        + pWFHM["timeshift"] * frefRD
        + pWFHM["phaseshift"]
        + pWFHM["phiref22"]
    )

    # Get phase shift from fit
    phishift = IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift(
        pWF22, pWFHM["IMRPhenomXHMRingdownPhaseFitsVersion"]
    )

    # Adjust the relative phase of our reconstruction
    pPhase["phi0_S"] = (
        phi22ref
        - IMRPhenomXHM_RD_Phase_AnsatzInt(frefRD, powers_of_FREF, pWFHM, pPhase)
        + phishift
    )

    return pPhase


def IMRPhenomXHM_GetAmplitudeCoefficients(
    pAmp: dict, pPhase: dict, pAmp22: dict, pPhase22: dict, pWFHM: dict, pWF22: dict
) -> dict:
    """
    Calculate amplitude coefficients for IMRPhenomXHM higher modes.

    This function computes amplitude coefficients across three regions (inspiral,
    intermediate, and ringdown) for a specific higher mode. It handles mode-specific
    normalizations, Post-Newtonian coefficients, and applies various vetoes for
    regions outside calibration.

    Args:
        pAmp: Amplitude coefficient dictionary for the higher mode
        pPhase: Phase coefficient dictionary for the higher mode
        pAmp22: Amplitude coefficient dictionary for the (2,2) mode
        pPhase22: Phase coefficient dictionary for the (2,2) mode
        pWFHM: Waveform structure dictionary for the higher mode
        pWF22: Waveform structure dictionary for the (2,2) mode

    Returns:
        dict: Updated pAmp dictionary with all amplitude coefficients
    """

    # Initialize amplitude normalization
    pAmp["ampNorm"] = pWF22["ampNorm"]
    pAmp["PNdominant"] = pWF22["ampNorm"] * jnp.power(2.0 / pWFHM["emm"], -7.0 / 6.0)

    # Mode-specific PN dominant term
    modeTag = pWFHM["modeTag"]

    if modeTag == 21:
        if pWF22["q"] == 1:
            pAmp["PNdominantlmpower"] = 2
            pAmp["PNdominantlm"] = (
                jnp.sqrt(2.0)
                / 3.0
                * 1.5
                * pWF22["dchi"]
                * 0.5
                * jnp.power(2 * jnp.pi / pWFHM["emm"], 2.0 / 3.0)
            )
        else:
            pAmp["PNdominantlmpower"] = 1
            pAmp["PNdominantlm"] = (
                jnp.sqrt(2.0)
                / 3.0
                * pWF22["delta"]
                * jnp.power(2 * jnp.pi / pWFHM["emm"], 1.0 / 3.0)
            )
    elif modeTag == 33:
        if pWF22["q"] == 1:
            pAmp["PNdominantlmpower"] = 4
            pAmp["PNdominantlm"] = (
                0.75
                * jnp.sqrt(5.0 / 7.0)
                * pWF22["dchi"]
                * 0.5
                * (65.0 / 24.0 - 28.0 / 3.0 * pWF22["eta"])
                * jnp.power(2 * jnp.pi / pWFHM["emm"], 4.0 / 3.0)
            )
        else:
            pAmp["PNdominantlmpower"] = 1
            pAmp["PNdominantlm"] = (
                0.75
                * jnp.sqrt(5.0 / 7.0)
                * pWF22["delta"]
                * jnp.power(2 * jnp.pi / pWFHM["emm"], 1.0 / 3.0)
            )
    elif modeTag == 32:
        pAmp["PNdominantlmpower"] = 2
        pAmp["PNdominantlm"] = (
            0.75 * jnp.sqrt(5.0 / 7.0) * jnp.power(2 * jnp.pi / pWFHM["emm"], 1.0 / 3.0)
        )
    elif modeTag == 44:
        pAmp["PNdominantlmpower"] = 2
        pAmp["PNdominantlm"] = (
            4.0
            / 9.0
            * jnp.sqrt(10.0 / 7.0)
            * (1 - 3 * pWF22["eta"])
            * jnp.power(2 * jnp.pi / pWFHM["emm"], 1.0 / 3.0)
        )
    else:
        raise ValueError(
            f"Mode {modeTag} not supported. Available modes: 21, 33, 32, 44"
        )

    pAmp["PNdominantlm"] = jnp.abs(pAmp["PNdominantlm"])
    pWFHM["fAmpRDfalloff"] = 0.0
    pAmp["nCoefficientsInter"] = 0

    # Branch based on release version
    if pWFHM["IMRPhenomXHMReleaseVersion"] != 122019:
        # Newer version (122022)
        pAmp["InspRescaleFactor"] = 0
        pAmp["RDRescaleFactor"] = 0
        pAmp["InterRescaleFactor"] = 0

        # Transform IMRPhenomXHMIntermediateAmpVersion to int array
        num = pWFHM["IMRPhenomXHMIntermediateAmpVersion"]
        pAmp["VersionCollocPtsInter"] = jnp.zeros(
            pWFHM["nCollocPtsInterAmp"], dtype=jnp.int32
        )
        for i in range(pWFHM["nCollocPtsInterAmp"]):
            pAmp["VersionCollocPtsInter"] = (
                pAmp["VersionCollocPtsInter"]
                .at[pWFHM["nCollocPtsInterAmp"] - i - 1]
                .set(num % 10)
            )
            num = num // 10

        pAmp["nCoefficientsInter"] = jnp.sum(pAmp["VersionCollocPtsInter"])

        if pAmp["nCoefficientsInter"] > pWFHM["nCollocPtsInterAmp"] + 2:
            raise ValueError(
                f"Inconsistent number of collocation points ({pWFHM['nCollocPtsInterAmp'] + 2}) "
                f"and free parameters ({pAmp['nCoefficientsInter']})"
            )

        pAmp["nCoefficientsRDAux"] = 0
        if pWFHM["MixingOn"]:
            pAmp["nCollocPtsRDAux"] = 2
            pAmp["nCoefficientsRDAux"] = 4
            pAmp["fRDAux"] = pWFHM["fRING"] - pWFHM["fDAMP"]

        # Get cutting frequencies
        pAmp["fAmpMatchIN"] = IMRPhenomXHM_Amplitude_fcutInsp(pWFHM, pWF22)
        pAmp["fAmpMatchIM"] = IMRPhenomXHM_Amplitude_fcutRD(pWFHM, pWF22)
        pWFHM["fAmpRDfalloff"] = pWFHM["fRING"] + 2 * pWFHM["fDAMP"]

        # Get PN amplitude coefficients
        pAmp = IMRPhenomXHM_GetPNAmplitudeCoefficients(pAmp, pWFHM, pWF22)

        # Get coefficients for each region
        pAmp = IMRPhenomXHM_Get_Inspiral_Amp_Coefficients(pAmp, pWFHM, pWF22)
        pAmp = IMRPhenomXHM_RD_Amp_Coefficients(pWF22, pWFHM, pAmp)
        pAmp = IMRPhenomXHM_Intermediate_Amp_Coefficients(
            pAmp, pWFHM, pWF22, pPhase, pAmp22, pPhase22
        )

        # Set rescale factors
        pAmp["InspRescaleFactor"] = 0
        pAmp["RDRescaleFactor"] = 0
        pAmp["InterRescaleFactor"] = 0

    else:
        # Original version (122019)
        pAmp["InspRescaleFactor"] = 1
        pAmp["InterRescaleFactor"] = -1
        pAmp["RDRescaleFactor"] = 1

        # Options for extrapolation outside calibration region
        if (
            (modeTag == 44 or modeTag == 33)
            and pWF22["q"] > 7.0
            and pWF22["chi1L"] > 0.95
        ):
            pAmp["useInspAnsatzRingdown"] = 1
        else:
            pAmp["useInspAnsatzRingdown"] = 0

        pAmp["WavyInsp"] = 0
        pAmp["WavyInt"] = 0
        if modeTag == 21:
            pAmp["WavyInsp"] = 1
            pAmp["WavyInt"] = 1
        if modeTag == 32:
            pAmp["WavyInsp"] = 1

        # Get cutting frequencies
        pAmp["fAmpMatchIN"] = IMRPhenomXHM_Amplitude_fcutInsp(pWFHM, pWF22)
        pAmp["fAmpMatchIM"] = IMRPhenomXHM_Amplitude_fcutRD(pWFHM, pWF22)

        # Compute intermediate collocation point frequencies
        df = pAmp["fAmpMatchIM"] - pAmp["fAmpMatchIN"]
        if "CollocationPointsFreqsAmplitudeInter" not in pAmp:
            pAmp["CollocationPointsFreqsAmplitudeInter"] = jnp.zeros(2)
        pAmp["CollocationPointsFreqsAmplitudeInter"] = (
            pAmp["CollocationPointsFreqsAmplitudeInter"]
            .at[0]
            .set(pAmp["fAmpMatchIN"] + df / 3.0)
        )
        pAmp["CollocationPointsFreqsAmplitudeInter"] = (
            pAmp["CollocationPointsFreqsAmplitudeInter"]
            .at[1]
            .set(pAmp["fAmpMatchIN"] + df * 2.0 / 3.0)
        )

        nCollocPtsInspAmp = pWFHM["nCollocPtsInspAmp"]
        nCollocPtsInterAmp = pWFHM["nCollocPtsInterAmp"]
        modeint = pWFHM["modeInt"]

        # === INSPIRAL REGION ===

        # Get PN amplitude coefficients
        pAmp = IMRPhenomXHM_GetPNAmplitudeCoefficients(pAmp, pWFHM, pWF22)

        # Initialize collocation point frequencies
        if "CollocationPointsFreqsAmplitudeInsp" not in pAmp:
            pAmp["CollocationPointsFreqsAmplitudeInsp"] = jnp.zeros(3)
        pAmp["CollocationPointsFreqsAmplitudeInsp"] = (
            pAmp["CollocationPointsFreqsAmplitudeInsp"]
            .at[0]
            .set(1.0 * pAmp["fAmpMatchIN"])
        )
        pAmp["CollocationPointsFreqsAmplitudeInsp"] = (
            pAmp["CollocationPointsFreqsAmplitudeInsp"]
            .at[1]
            .set(0.75 * pAmp["fAmpMatchIN"])
        )
        pAmp["CollocationPointsFreqsAmplitudeInsp"] = (
            pAmp["CollocationPointsFreqsAmplitudeInsp"]
            .at[2]
            .set(0.5 * pAmp["fAmpMatchIN"])
        )

        fcutInsp = pAmp["fAmpMatchIN"]
        f1 = pAmp["CollocationPointsFreqsAmplitudeInsp"][0]
        f2 = pAmp["CollocationPointsFreqsAmplitudeInsp"][1]
        f3 = pAmp["CollocationPointsFreqsAmplitudeInsp"][2]

        # Compute useful powers
        powers_of_fcutInsp = compute_powers_of_f(fcutInsp)
        powers_of_f1 = compute_powers_of_f(f1)
        powers_of_f2 = compute_powers_of_f(f2)
        powers_of_f3 = compute_powers_of_f(f3)

        # Compute PN ansatz values
        PNf1 = IMRPhenomXHM_Inspiral_PNAmp_Ansatz(powers_of_f1, pWFHM, pAmp)
        PNf2 = IMRPhenomXHM_Inspiral_PNAmp_Ansatz(powers_of_f2, pWFHM, pAmp)
        PNf3 = IMRPhenomXHM_Inspiral_PNAmp_Ansatz(powers_of_f3, pWFHM, pAmp)

        if "PNAmplitudeInsp" not in pAmp:
            pAmp["PNAmplitudeInsp"] = jnp.zeros(3)
        pAmp["PNAmplitudeInsp"] = pAmp["PNAmplitudeInsp"].at[0].set(PNf1)
        pAmp["PNAmplitudeInsp"] = pAmp["PNAmplitudeInsp"].at[1].set(PNf2)
        pAmp["PNAmplitudeInsp"] = pAmp["PNAmplitudeInsp"].at[2].set(PNf3)

        # Get collocation point values from fits
        if "CollocationPointsValuesAmplitudeInsp" not in pAmp:
            pAmp["CollocationPointsValuesAmplitudeInsp"] = jnp.zeros(nCollocPtsInspAmp)
        for i in range(nCollocPtsInspAmp):
            pAmp["CollocationPointsValuesAmplitudeInsp"] = (
                pAmp["CollocationPointsValuesAmplitudeInsp"]
                .at[i]
                .set(
                    jnp.abs(
                        pAmp["InspiralAmpFits"][modeint * nCollocPtsInspAmp + i](
                            pWF22, pWFHM["IMRPhenomXHMInspiralAmpFitsVersion"]
                        )
                    )
                )
            )

        # Pseudo-PN collocation points
        iv1 = pAmp["CollocationPointsValuesAmplitudeInsp"][0] - PNf1
        iv2 = pAmp["CollocationPointsValuesAmplitudeInsp"][1] - PNf2
        iv3 = pAmp["CollocationPointsValuesAmplitudeInsp"][2] - PNf3

        # Apply inspiral veto
        if pWFHM["InspiralAmpVeto"] == 1:
            iv1, iv2, iv3, powers_of_f1, powers_of_f2, powers_of_f3 = (
                IMRPhenomXHM_Inspiral_Amplitude_Veto(
                    iv1, iv2, iv3, powers_of_f1, powers_of_f2, powers_of_f3, pAmp, pWFHM
                )
            )

        # Mode-specific vetoes
        if (
            modeTag == 32
            and pWF22["q"] > 2.5
            and pWF22["chi1L"] < -0.9
            and pWF22["chi2L"] < -0.9
        ):
            pWFHM["IMRPhenomXHMInspiralAmpVersion"] = 0

        if (
            modeTag == 32
            and pWF22["q"] > 2.5
            and pWF22["chi1L"] < -0.6
            and pWF22["chi2L"] > 0.0
            and iv1 != 0
        ):
            pWFHM["IMRPhenomXHMInspiralAmpVersion"] = (
                pWFHM["IMRPhenomXHMInspiralAmpVersion"] - 1
            )
            iv1 = 0.0

        if modeTag == 33 and (
            1.2 > pWF22["q"] > 1.0
            and pWF22["chi1L"] < -0.1
            and pWF22["chi2L"] > 0.0
            and iv1 != 0
        ):
            pWFHM["IMRPhenomXHMInspiralAmpVersion"] = (
                pWFHM["IMRPhenomXHMInspiralAmpVersion"] - 1
            )
            iv1 = 0.0

        # Check for wavy collocation points
        if pWFHM["IMRPhenomXHMInspiralAmpVersion"] == 3 and pAmp["WavyInsp"] == 1:
            if (
                WavyPoints(
                    pAmp["CollocationPointsValuesAmplitudeInsp"][0]
                    * powers_of_f1["m_seven_sixths"],
                    pAmp["CollocationPointsValuesAmplitudeInsp"][1]
                    * powers_of_f2["m_seven_sixths"],
                    pAmp["CollocationPointsValuesAmplitudeInsp"][2]
                    * powers_of_f3["m_seven_sixths"],
                )
                == 1
            ):
                iv2 = 0
                pWFHM["IMRPhenomXHMInspiralAmpVersion"] = (
                    pWFHM["IMRPhenomXHMInspiralAmpVersion"] - 1
                )

        # Rename collocation points
        if iv2 == 0:
            iv2 = iv3
            iv3 = 0.0
            powers_of_f2 = powers_of_f3

        if iv1 == 0:
            iv1 = iv2
            powers_of_f1 = powers_of_f2
            powers_of_fcutInsp = powers_of_f1
            iv2 = iv3
            iv3 = 0.0
            powers_of_f2 = powers_of_f3

        if pWFHM["IMRPhenomXHMInspiralAmpVersion"] == 0:
            powers_of_fcutInsp = compute_powers_of_f(pWFHM["fMECOlm"])

        # Update inspiral cutting frequency
        pAmp["fAmpMatchIN"] = powers_of_fcutInsp["itself"]

        # Get pseudo-PN coefficients
        pAmp["rho1"] = IMRPhenomXHM_Inspiral_Amp_rho1(
            iv1,
            iv2,
            iv3,
            powers_of_fcutInsp,
            powers_of_f1,
            powers_of_f2,
            powers_of_f3,
            pWFHM,
        )
        pAmp["rho2"] = IMRPhenomXHM_Inspiral_Amp_rho2(
            iv1,
            iv2,
            iv3,
            powers_of_fcutInsp,
            powers_of_f1,
            powers_of_f2,
            powers_of_f3,
            pWFHM,
        )
        pAmp["rho3"] = IMRPhenomXHM_Inspiral_Amp_rho3(
            iv1,
            iv2,
            iv3,
            powers_of_fcutInsp,
            powers_of_f1,
            powers_of_f2,
            powers_of_f3,
            pWFHM,
        )

        # Store useful powers for later use
        pAmp["fcutInsp_seven_thirds"] = powers_of_fcutInsp["seven_thirds"]
        pAmp["fcutInsp_eight_thirds"] = powers_of_fcutInsp["eight_thirds"]
        pAmp["fcutInsp_three"] = powers_of_fcutInsp["three"]

        # === RINGDOWN REGION ===

        # Get ringdown coefficients from fits
        if "RDCoefficient" not in pAmp:
            pAmp["RDCoefficient"] = jnp.zeros(4)

        pAmp["RDCoefficient"] = (
            pAmp["RDCoefficient"]
            .at[0]
            .set(
                jnp.abs(
                    pAmp["RingdownAmpFits"][modeint * 3](
                        pWF22, pWFHM["IMRPhenomXHMRingdownAmpFitsVersion"]
                    )
                )
            )
        )
        pAmp["RDCoefficient"] = (
            pAmp["RDCoefficient"]
            .at[1]
            .set(
                pAmp["RingdownAmpFits"][modeint * 3 + 1](
                    pWF22, pWFHM["IMRPhenomXHMRingdownAmpFitsVersion"]
                )
            )
        )
        pAmp["RDCoefficient"] = (
            pAmp["RDCoefficient"]
            .at[2]
            .set(
                pAmp["RingdownAmpFits"][modeint * 3 + 2](
                    pWF22, pWFHM["IMRPhenomXHMRingdownAmpFitsVersion"]
                )
            )
        )
        pAmp["RDCoefficient"] = pAmp["RDCoefficient"].at[3].set(1.0 / 12.0)

        # Apply PNR tuning for (3,3) mode
        if modeTag == 33:
            pAmp["RDCoefficient"] = (
                pAmp["RDCoefficient"]
                .at[0]
                .add(pWFHM["PNR_DEV_PARAMETER"] * pWFHM["MU3"])
            )
            pAmp["RDCoefficient"] = (
                pAmp["RDCoefficient"]
                .at[2]
                .add(pWFHM["PNR_DEV_PARAMETER"] * pWFHM["MU4"])
            )

        # Handle extreme spin cases
        if pAmp["useInspAnsatzRingdown"] == 1:
            powers_of_fRD = compute_powers_of_f(pAmp["fAmpMatchIM"])
            insp_amp = IMRPhenomXHM_Inspiral_Amp_Ansatz(powers_of_fcutInsp, pWFHM, pAmp)
            rd_amp = (
                IMRPhenomXHM_RD_Amp_Ansatz(powers_of_fRD, pWFHM, pAmp)
                / pAmp["RDCoefficient"][0]
            )
            pAmp["RDCoefficient"] = (
                pAmp["RDCoefficient"].at[0].set(0.9 * jnp.abs(insp_amp / rd_amp))
            )

        # === INTERMEDIATE REGION ===

        # Get collocation point values from fits
        if "CollocationPointsValuesAmplitudeInter" not in pAmp:
            pAmp["CollocationPointsValuesAmplitudeInter"] = jnp.zeros(
                nCollocPtsInterAmp
            )
        for i in range(nCollocPtsInterAmp):
            pAmp["CollocationPointsValuesAmplitudeInter"] = (
                pAmp["CollocationPointsValuesAmplitudeInter"]
                .at[i]
                .set(
                    jnp.abs(
                        pAmp["IntermediateAmpFits"][modeint * nCollocPtsInterAmp + i](
                            pWF22, pWFHM["IMRPhenomXHMIntermediateAmpFitsVersion"]
                        )
                    )
                )
            )

        # Set up frequencies
        F1 = powers_of_fcutInsp["itself"]
        F2 = pAmp["CollocationPointsFreqsAmplitudeInter"][0]
        F3 = pAmp["CollocationPointsFreqsAmplitudeInter"][1]
        F4 = pAmp["fAmpMatchIM"]

        # Compute powers
        powers_of_F1 = compute_powers_of_f(F1)
        powers_of_F4 = compute_powers_of_f(F4)

        # Compute boundary values
        inspF1 = IMRPhenomXHM_Inspiral_Amp_Ansatz(powers_of_F1, pWFHM, pAmp)

        if pWFHM["MixingOn"] == 1:
            rdF4 = jnp.abs(
                SpheroidalToSpherical(
                    powers_of_F4, pAmp22, pPhase22, pAmp, pPhase, pWFHM, pWF22
                )
            )
        else:
            rdF4 = IMRPhenomXHM_RD_Amp_Ansatz(powers_of_F4, pWFHM, pAmp)

        # Compute boundary derivatives
        d1 = IMRPhenomXHM_Inspiral_Amp_NDAnsatz(powers_of_F1, pWFHM, pAmp)

        if pWFHM["MixingOn"] == 1:
            d4 = IMRPhenomXHM_RD_Amp_NDAnsatz(
                powers_of_F4, pAmp, pPhase, pWFHM, pAmp22, pPhase22, pWF22
            )
        else:
            d4 = IMRPhenomXHM_RD_Amp_DAnsatz(powers_of_F4, pWFHM, pAmp)

        # Set rescale factors
        pAmp["InspRescaleFactor"] = 0
        pAmp["InterRescaleFactor"] = 0
        pAmp["RDRescaleFactor"] = 0

        # Transform derivatives
        d1 = ((7.0 / 6.0) * jnp.power(F1, 1.0 / 6.0) / inspF1) - (
            jnp.power(F1, 7.0 / 6.0) * d1 / (inspF1 * inspF1)
        )
        d4 = ((7.0 / 6.0) * jnp.power(F4, 1.0 / 6.0) / rdF4) - (
            jnp.power(F4, 7.0 / 6.0) * d4 / (rdF4 * rdF4)
        )

        # Set up collocation point values
        pWFHM["IMRPhenomXHMIntermediateAmpVersion"] = (
            105  # Default: 5th order polynomial
        )

        V1 = powers_of_F1["m_seven_sixths"] * inspF1
        V2 = pAmp["CollocationPointsValuesAmplitudeInter"][0]
        V3 = pAmp["CollocationPointsValuesAmplitudeInter"][1]
        V4 = powers_of_F4["m_seven_sixths"] * rdF4

        # Apply veto for extreme cases
        if pAmp["useInspAnsatzRingdown"] == 1:
            V2 = 1.0
            V3 = 1.0
            pWFHM["IMRPhenomXHMIntermediateAmpVersion"] = 101  # Linear reconstruction

        # Invert for polynomial reconstruction
        V1 = 1.0 / V1
        V2 = 1.0 / V2
        V3 = 1.0 / V3
        V4 = 1.0 / V4

        # Apply NR tuning for (3,3) mode
        if modeTag == 33:
            V2 = V2 + (pWFHM["PNR_DEV_PARAMETER"] * pWFHM["MU1"])
            V3 = V3 + (pWFHM["PNR_DEV_PARAMETER"] * pWFHM["MU2"])

        # Handle EMR cases (two intermediate regions)
        if pWFHM["AmpEMR"] == 1:
            # First intermediate region
            F0 = F1 + (F2 - F1) / 3.0
            pAmp["fAmpMatchInt12"] = F0

            V0 = pAmp["IntermediateAmpFits"][modeint * nCollocPtsInterAmp + 8](
                pWF22, pWFHM["IMRPhenomXHMIntermediateAmpFitsVersion"]
            )
            d0 = pAmp["IntermediateAmpFits"][modeint * nCollocPtsInterAmp + 9](
                pWF22, pWFHM["IMRPhenomXHMIntermediateAmpFitsVersion"]
            )

            F0_seven_sixths = jnp.power(F0, 7.0 / 6.0)
            d0 = ((7.0 / 6.0) / (V0 * F0)) - (d0 / (V0 * V0 * F0_seven_sixths))
            V0 = 1.0 / V0

            # Get coefficients for first intermediate region
            if "alpha0" not in pAmp:
                pAmp["alpha0"] = 0.0
                pAmp["alpha1"] = 0.0
                pAmp["alpha2"] = 0.0
                pAmp["alpha3"] = 0.0
                pAmp["alpha4"] = 0.0

            pAmp["alpha0"] = IMRPhenomXHM_Intermediate_Amp_delta0(
                d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104
            )
            pAmp["alpha1"] = IMRPhenomXHM_Intermediate_Amp_delta1(
                d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104
            )
            pAmp["alpha2"] = IMRPhenomXHM_Intermediate_Amp_delta2(
                d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104
            )
            pAmp["alpha3"] = IMRPhenomXHM_Intermediate_Amp_delta3(
                d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104
            )
            pAmp["alpha4"] = IMRPhenomXHM_Intermediate_Amp_delta4(
                d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104
            )

            # Update values for second intermediate region
            d1 = d0
            V1 = V0
            F1 = F0

        # Get coefficients for main intermediate region
        version = pWFHM["IMRPhenomXHMIntermediateAmpVersion"]

        if "delta0" not in pAmp:
            pAmp["delta0"] = 0.0
            pAmp["delta1"] = 0.0
            pAmp["delta2"] = 0.0
            pAmp["delta3"] = 0.0
            pAmp["delta4"] = 0.0
            pAmp["delta5"] = 0.0

        pAmp["delta0"] = IMRPhenomXHM_Intermediate_Amp_delta0(
            d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version
        )
        pAmp["delta1"] = IMRPhenomXHM_Intermediate_Amp_delta1(
            d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version
        )
        pAmp["delta2"] = IMRPhenomXHM_Intermediate_Amp_delta2(
            d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version
        )
        pAmp["delta3"] = IMRPhenomXHM_Intermediate_Amp_delta3(
            d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version
        )
        pAmp["delta4"] = IMRPhenomXHM_Intermediate_Amp_delta4(
            d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version
        )

        if version == 105:
            pAmp["delta5"] = IMRPhenomXHM_Intermediate_Amp_delta5(
                d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version
            )

        # Final rescale factors
        pAmp["InspRescaleFactor"] = 0
        pAmp["RDRescaleFactor"] = 0
        pAmp["InterRescaleFactor"] = 0

    return pAmp


def IMRPhenomXHM_GetPhaseCoefficients(
    pAmp: dict,
    pPhase: dict,
    pAmp22: dict,
    pPhase22: dict,
    pWFHM: dict,
    pWF22: dict,
    lalParams: dict = None,
) -> dict:
    """
    Compute phase coefficients for IMRPhenomXHM higher modes.

    This function computes all phenomenological phase coefficients for a given higher mode,
    including inspiral (rescaled from 22 mode), intermediate (via collocation), and
    ringdown regions. It handles both spherical and spheroidal mode mixing cases.

    Parameters
    ----------
    pAmp : dict
        Amplitude coefficients structure for the higher mode
    pPhase : dict
        Phase coefficients structure for the higher mode (will be updated)
    pAmp22 : dict
        Amplitude coefficients for the (2,2) mode
    pPhase22 : dict
        Phase coefficients for the (2,2) mode
    pWFHM : dict
        Waveform structure for the higher mode
    pWF22 : dict
        Waveform structure for the (2,2) mode
    lalParams : dict, optional
        Additional LAL parameters (unused)

    Returns
    -------
    dict
        Updated pPhase dictionary with all computed coefficients
    """

    ell = pWFHM["ell"]
    emm = pWFHM["emm"]

    # Pre-initialize all phenomenological coefficients
    # Inspiral - will be computed by rescaling IMRPhenomX inspiral coefficients
    pPhase["phi"] = jnp.zeros(16)
    pPhase["phiL"] = jnp.zeros(16)

    # Intermediate - determined by solving linear system at collocation points
    pPhase["c0"] = 0.0
    pPhase["c1"] = 0.0
    pPhase["c2"] = 0.0
    pPhase["c3"] = 0.0
    pPhase["c4"] = 0.0
    pPhase["cL"] = 0.0

    # Ringdown spherical - obtained by rescaling the 22 ringdown parameters
    pPhase["alpha0"] = 0.0
    pPhase["alpha2"] = 0.0
    pPhase["alphaL"] = 0.0

    # Set number of collocation points used (depends on the mode)
    nCollocationPts_inter = pWFHM["nCollocPtsInterPhase"]
    eta_m1 = 1.0 / pWF22["eta"]
    modeint = pWFHM["modeInt"]
    N_MAX_COEFFICIENTS_PHASE_INTER = 6
    # Initialize frequencies of collocation points in intermediate region
    pPhase = IMRPhenomXHM_Intermediate_CollocPtsFreqs(pPhase, pWFHM, pWF22)

    # For each collocation point, call fit giving the value of the phase derivative
    for i in range(N_MAX_COEFFICIENTS_PHASE_INTER):
        pPhase["CollocationPointsValuesPhaseInter"][i] = pPhase[
            "IntermediatePhaseFits"
        ][modeint * N_MAX_COEFFICIENTS_PHASE_INTER + i](
            pWF22, pWFHM["IMRPhenomXHMIntermediatePhaseFitsVersion"]
        )
        # Time-shift waveform so that modes peak around t=0
        pPhase["CollocationPointsValuesPhaseInter"][i] += pWFHM["DeltaT"]

    fcutRD = pPhase["fPhaseMatchIM"]
    fcutInsp = pPhase["fPhaseMatchIN"]

    # ============ Inspiral: rescale PhenomX and apply PN corrections ============

    # Collect all the PhenomX inspiral coefficients
    phenXnonLog = jnp.array(
        [
            pPhase22["phi_minus2"],
            pPhase22["phi_minus1"],
            pPhase22["phi0"],
            pPhase22["phi1"],
            pPhase22["phi2"],
            pPhase22["phi3"],
            pPhase22["phi4"],
            pPhase22["phi5"],
            pPhase22["phi6"],
            pPhase22["phi7"],
            pPhase22["phi8"],
            pPhase22["phi9"],
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    phenXLog = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            pPhase22["phi5L"],
            pPhase22["phi6L"],
            0.0,
            pPhase22["phi8L"],
            pPhase22["phi9L"],
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    pseudoPN = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            pPhase22["sigma1"],
            pPhase22["sigma2"],
            pPhase22["sigma3"],
            pPhase22["sigma4"],
            pPhase22["sigma5"],
            0.0,
        ]
    )

    # Rescale coefficients by applying phi_lm(f) ~ m/2 * phi_22(2/m * f)
    m_over_2 = emm * 0.5
    two_over_m = 1.0 / m_over_2

    fact = pPhase22["phiNorm"] / pWF22["eta"]

    phenXnonLog = phenXnonLog * fact
    phenXLog = phenXLog * fact
    pseudoPN = pseudoPN * fact

    # Scaling logarithmic terms introduces extra contributions in non-log terms
    for i in range(16):
        power_idx = (10 - i) / 3.0
        pPhase["phi"] = (
            pPhase["phi"]
            .at[i]
            .set(
                (phenXnonLog[i] + pseudoPN[i] - phenXLog[i] * jnp.log(m_over_2))
                * jnp.power(m_over_2, power_idx)
            )
        )
        pPhase["phiL"] = (
            pPhase["phiL"].at[i].set(phenXLog[i] * jnp.power(m_over_2, power_idx))
        )

    # Add PN amplitude correction to orbital phase
    if pWF22["eta"] > 0.01:
        # Use complex PN amplitudes for non-extreme mass ratios

        pPhase["LambdaPN"] = IMRPhenomXHM_Insp_Phase_LambdaPN(
            pWF22["eta"], pWFHM["modeTag"]
        )
    else:
        # Use phenomenological fits for extreme mass ratios
        pPhase["LambdaPN"] = pPhase["InspiralPhaseFits"][pWFHM["modeInt"]](
            pWF22, pWFHM["IMRPhenomXHMInspiralPhaseVersion"]
        )

    pPhase["phi"] = pPhase["phi"].at[10].add(pPhase["LambdaPN"])

    # ============ Intermediate-Ringdown Region ============

    # Choose collocation points according to spin/mass ratio
    cpoints_indices = jnp.array([0, 1, 0, 0, 5], dtype=jnp.int32)

    if (
        (pWF22["eta"] < pWFHM["etaEMR"])
        or (emm == ell and pWF22["STotR"] >= 0.8)
        or (pWFHM["modeTag"] == 33 and pWF22["STotR"] < 0)
    ):
        cpoints_indices = cpoints_indices.at[2].set(3)
        cpoints_indices = cpoints_indices.at[3].set(4)
    elif pWF22["STotR"] >= 0.8 and pWFHM["modeTag"] == 21:
        cpoints_indices = cpoints_indices.at[2].set(2)
        cpoints_indices = cpoints_indices.at[3].set(4)
    else:
        cpoints_indices = cpoints_indices.at[2].set(2)
        cpoints_indices = cpoints_indices.at[3].set(3)

    # Handle mode mixing cases
    if pWFHM["MixingOn"] == 0:
        # Mode-mixing off: spherical harmonics
        # Compute ringdown coefficients by rescaling
        if pWFHM["ell"] == pWFHM["emm"]:
            wlm = 2.0
        else:
            wlm = pWFHM["emm"] / 3.0

        pPhase["alpha2"] = (
            1.0
            / (pWFHM["fRING"] ** 2)
            * wlm
            * IMRPhenomXHM_RD_Phase_22_alpha2(
                pWF22, pWFHM["IMRPhenomXHMRingdownPhaseVersion"]
            )
        )

        pPhase["alphaL"] = eta_m1 * IMRPhenomXHM_RD_Phase_22_alphaL(
            pWF22, pWFHM["IMRPhenomXHMRingdownPhaseVersion"]
        )

        # Compute spherical-harmonic phase and derivative at matching frequency
        powers_of_f = compute_powers_of_f(fcutRD)

        pPhase["phi0RD"] = IMRPhenomXHM_RD_Phase_AnsatzInt(
            fcutRD, powers_of_f, pWFHM, pPhase
        )
        pPhase["dphi0RD"] = IMRPhenomXHM_RD_Phase_Ansatz(
            fcutRD, powers_of_f, pWFHM, pPhase
        )

        # Set up linear system for intermediate region
        A = jnp.zeros((nCollocationPts_inter, nCollocationPts_inter))
        b = jnp.zeros(nCollocationPts_inter)

        for i in range(nCollocationPts_inter):
            ind = cpoints_indices[i]
            b = b.at[i].set(pPhase["CollocationPointsValuesPhaseInter"][ind])
            ff = pPhase["CollocationPointsFreqsPhaseInter"][ind]
            ffm1 = 1.0 / ff
            ffm2 = ffm1 * ffm1
            fpowers = jnp.array(
                [
                    1.0,
                    pWFHM["fDAMP"] / (pWFHM["fDAMP"] ** 2 + (ff - pWFHM["fRING"]) ** 2),
                    ffm1,
                    ffm2,
                    ffm2 * ffm2,
                    ffm1 * ffm2,
                ]
            )
            for j in range(nCollocationPts_inter):
                A = A.at[i, j].set(fpowers[j])

    elif pWFHM["MixingOn"] == 1:
        # Mode-mixing on (32 mode with spheroidal harmonics)
        # Compute derivatives using finite differences
        fstep = 0.0000001
        SphericalWF = jnp.zeros(3, dtype=jnp.complex128)

        for i in range(3):
            FF = fcutRD + (i - 1) * fstep
            powers_of_FF = compute_powers_of_f(FF)
            SphericalWF = SphericalWF.at[i].set(
                SpheroidalToSpherical(
                    powers_of_FF, pAmp22, pPhase22, pAmp, pPhase, pWFHM, pWF22
                )
            )

        phase_args = jnp.array(
            [
                jnp.mod(jnp.angle(SphericalWF[0]), 2.0 * jnp.pi),
                jnp.mod(jnp.angle(SphericalWF[1]), 2.0 * jnp.pi),
                jnp.mod(jnp.angle(SphericalWF[2]), 2.0 * jnp.pi),
            ]
        )

        # Ensure all points belong to same branch
        phase_args = jnp.where(phase_args > 0, phase_args - 2.0 * jnp.pi, phase_args)

        pPhase["phi0RD"] = phase_args[1]
        fstep_m1 = 1.0 / fstep
        pPhase["dphi0RD"] = 0.5 * fstep_m1 * (phase_args[2] - phase_args[0])
        d2phi0RD = (
            fstep_m1 * fstep_m1 * (phase_args[2] - 2.0 * phase_args[1] + phase_args[0])
        )

        # Update collocation points with derivatives
        if pWF22["eta"] > pWFHM["etaEMR"]:
            pPhase["CollocationPointsFreqsPhaseInter"] = (
                pPhase["CollocationPointsFreqsPhaseInter"]
                .at[nCollocationPts_inter - 2]
                .set(fcutRD)
            )
            pPhase["CollocationPointsValuesPhaseInter"] = (
                pPhase["CollocationPointsValuesPhaseInter"]
                .at[nCollocationPts_inter - 2]
                .set(pPhase["dphi0RD"])
            )

        pPhase["CollocationPointsFreqsPhaseInter"] = (
            pPhase["CollocationPointsFreqsPhaseInter"]
            .at[nCollocationPts_inter - 1]
            .set(fcutRD)
        )
        pPhase["CollocationPointsValuesPhaseInter"] = (
            pPhase["CollocationPointsValuesPhaseInter"]
            .at[nCollocationPts_inter - 1]
            .set(d2phi0RD)
        )

        # Set up linear system
        A = jnp.zeros((nCollocationPts_inter, nCollocationPts_inter))
        b = jnp.zeros(nCollocationPts_inter)

        for i in range(nCollocationPts_inter - 1):
            b = b.at[i].set(pPhase["CollocationPointsValuesPhaseInter"][i])
            ff = pPhase["CollocationPointsFreqsPhaseInter"][i]
            ffm1 = 1.0 / ff
            ffm2 = ffm1 * ffm1
            fpowers = jnp.array(
                [
                    1.0,
                    pWFHM["fDAMP"] / (pWFHM["fDAMP"] ** 2 + (ff - pWFHM["fRING"]) ** 2),
                    ffm1,
                    ffm2,
                    ffm2 * ffm2,
                    ffm1 * ffm2,
                ]
            )
            for j in range(nCollocationPts_inter):
                A = A.at[i, j].set(fpowers[j])

        # Last point: second derivative constraint
        cpoint_ind = nCollocationPts_inter - 1
        b = b.at[cpoint_ind].set(
            pPhase["CollocationPointsValuesPhaseInter"][cpoint_ind]
        )
        ff = pPhase["CollocationPointsFreqsPhaseInter"][cpoint_ind]
        ffm1 = 1.0 / ff
        ffm2 = ffm1 * ffm1
        ffm3 = ffm2 * ffm1
        ffm4 = ffm2 * ffm2
        ffm5 = ffm3 * ffm2

        fpowers = jnp.array(
            [
                0.0,
                -2.0
                * pWFHM["fDAMP"]
                * (ff - pWFHM["fRING"])
                / ((pWFHM["fDAMP"] ** 2 + (ff - pWFHM["fRING"]) ** 2) ** 2),
                -ffm2,
                -2.0 * ffm3,
                -4.0 * ffm5,
                -3.0 * ffm4,
            ]
        )
        for j in range(nCollocationPts_inter):
            A = A.at[cpoint_ind, j].set(fpowers[j])

    # Solve linear system A x = b
    x = jnp.linalg.solve(A, b)

    pPhase["c0"] = x[0]
    pPhase["cL"] = x[1]
    pPhase["c1"] = x[2]
    pPhase["c2"] = x[3]
    pPhase["c4"] = x[4]

    # Add PNR deviations for 33 mode
    if pWFHM["modeTag"] == 33:
        pPhase["c0"] += pWFHM["PNR_DEV_PARAMETER"] * pWFHM["NU0"]
        pPhase["cL"] += pWFHM["PNR_DEV_PARAMETER"] * pWFHM["NU4"]
        pPhase["c1"] += pWFHM["PNR_DEV_PARAMETER"] * pWFHM["ZETA2"]
        pPhase["c4"] += pWFHM["PNR_DEV_PARAMETER"] * pWFHM["ZETA1"]

    # 32 mode uses one extra collocation point
    if pWFHM["modeTag"] == 32:
        pPhase["c3"] = x[5]

        # Glue intermediate and ringdown for extreme mass ratios
        if pWF22["eta"] < pWFHM["etaEMR"]:
            powers_of_f = compute_powers_of_f(fcutRD)
            pPhase["c0"] += pPhase["dphi0RD"] - IMRPhenomXHM_Inter_Phase_Ansatz(
                fcutRD, powers_of_f, pWFHM, pPhase
            )

    # Glue inspiral and intermediate regions
    powers_of_f = compute_powers_of_f(fcutInsp)

    pPhase["C1INSP"] = IMRPhenomXHM_Inter_Phase_Ansatz(
        fcutInsp, powers_of_f, pWFHM, pPhase
    ) - IMRPhenomXHM_Inspiral_Phase_Ansatz(fcutInsp, powers_of_f, pPhase)

    pPhase["CINSP"] = (
        -pPhase["C1INSP"] * fcutInsp
        + IMRPhenomXHM_Inter_Phase_AnsatzInt(fcutInsp, powers_of_f, pWFHM, pPhase)
        - IMRPhenomXHM_Inspiral_Phase_AnsatzInt(fcutInsp, powers_of_f, pPhase)
    )

    # Glue ringdown and intermediate regions
    powers_of_f = compute_powers_of_f(fcutRD)
    pPhase["C1RD"] = (
        IMRPhenomXHM_Inter_Phase_Ansatz(fcutRD, powers_of_f, pWFHM, pPhase)
        - pPhase["dphi0RD"]
    )
    pPhase["CRD"] = (
        -pPhase["C1RD"] * fcutRD
        + IMRPhenomXHM_Inter_Phase_AnsatzInt(fcutRD, powers_of_f, pWFHM, pPhase)
        - pPhase["phi0RD"]
    )

    # Align each mode so low-frequency relative phase matches PN
    if pWF22["eta"] > pWFHM["etaEMR"]:
        falign = 0.6 * m_over_2 * pWF22["fMECO"]
    else:
        falign = m_over_2 * pWF22["fMECO"]

    powers_of_falign = compute_powers_of_f(falign)
    powers_of_f = compute_powers_of_f(two_over_m * falign)

    # Compute phase normalization for spherical modes
    if pWFHM["MixingOn"] == 0:
        powers_of_MfRef = compute_powers_of_f(pWF22["MfRef"])
        pPhase22 = IMRPhenomX_Phase_22_ConnectionCoefficients(pWF22, pPhase22)
        pWFHM["timeshift"] = IMRPhenomX_TimeShift_22(pPhase22, pWF22)
        pWFHM["phiref22"] = (
            -1.0
            / pWF22["eta"]
            * IMRPhenomX_Phase_22(pWF22["MfRef"], powers_of_MfRef, pPhase22, pWF22)
            - pWFHM["timeshift"] * pWF22["MfRef"]
            - pWFHM["phaseshift"]
            + 2.0 * pWF22["phi0"]
            + jnp.pi / 4.0
        )

    # Phase normalization from Eq. (4.13)
    deltaphiLM = (
        m_over_2
        * (
            1.0
            / pWF22["eta"]
            * IMRPhenomX_Phase_22(two_over_m * falign, powers_of_f, pPhase22, pWF22)
            + pWFHM["phaseshift"]
            + pWFHM["phiref22"]
        )
        + pWFHM["timeshift"] * falign
        - 3.0 * jnp.pi / 4.0 * (1.0 - m_over_2)
        - (
            IMRPhenomXHM_Inspiral_Phase_AnsatzInt(falign, powers_of_falign, pPhase)
            + pPhase["C1INSP"] * falign
            + pPhase["CINSP"]
        )
    )
    pPhase["deltaphiLM"] = jnp.mod(deltaphiLM, 2.0 * jnp.pi)

    # For 21 mode, account for sign flip in PN amplitude
    if pWFHM["modeTag"] == 21:
        ampsign = IMRPhenomXHM_PN21AmpSign(0.008, pWF22)
        if ampsign > 0:
            pPhase["deltaphiLM"] += jnp.pi

    return pPhase


def IMRPhenomXHM_Ringdown_CollocPtsFreqs(
    pPhase: dict, pWFHM: dict, pWF22: dict
) -> dict:
    """
    Initialize frequencies of collocation points for spheroidal ringdown reconstruction.

    This function initializes the frequencies stored in the pPhase struct, which gets
    updated with new values as the code processes each mode.

    Args:
        pPhase: Phase coefficient dictionary for the higher mode
        pWFHM: Waveform structure dictionary for the higher mode
        pWF22: Waveform structure dictionary for the (2,2) mode

    Returns:
        dict: Updated pPhase dictionary with collocation point frequencies
    """
    # Extract ringdown and damping frequencies for the mode
    fringlm = pWFHM["fRING"]
    fdamplm = pWFHM["fDAMP"]
    fring22 = pWF22["fRING"]

    # Initialize collocation points array if not present
    if "CollocationPointsFreqsPhaseRD" not in pPhase:
        # Maximum size needed based on version 122022 (5 points)
        pPhase["CollocationPointsFreqsPhaseRD"] = jnp.zeros(5)

    # Switch based on version
    version = pWFHM["IMRPhenomXHMRingdownPhaseFreqsVersion"]

    def case_122019():
        """Version 122019: 4 collocation points"""
        freqs = jnp.zeros(5)
        freqs = freqs.at[0].set(fring22)
        freqs = freqs.at[2].set(fringlm - 0.5 * fdamplm)
        freqs = freqs.at[1].set(fringlm - 1.5 * fdamplm)
        freqs = freqs.at[3].set(fringlm + 0.5 * fdamplm)
        return freqs

    def case_122022():
        """Version 122022: 5 collocation points"""
        fdamp22 = pWF22["fDAMP"]
        freqs = jnp.zeros(5)
        freqs = freqs.at[0].set(fring22 - fdamp22)
        freqs = freqs.at[1].set(fring22)
        freqs = freqs.at[2].set((fring22 + fringlm) * 0.5)
        freqs = freqs.at[3].set(fringlm)
        freqs = freqs.at[4].set(fringlm + fdamplm)
        return freqs

    def default_case():
        """Default case: return NaN (error condition)"""
        # In JAX we can't raise errors in jit-compiled code
        # Return array of NaNs to signal error
        return jnp.full(5, jnp.nan)

    # Use lax.switch for conditional branching
    freqs = jax.lax.switch(
        jnp.where(version == 122019, 0, jnp.where(version == 122022, 1, 2)),
        [case_122019, case_122022, default_case],
    )

    pPhase["CollocationPointsFreqsPhaseRD"] = freqs

    return pPhase


def RescaleFactor(powers_of_Mf: dict, pAmp: dict, rescalefactor: int) -> float:
    """
    Compute the rescaling factor for amplitude normalization.

    This function returns different rescaling factors depending on the rescalefactor
    parameter. It is used to convert between different amplitude normalizations:
    - 0: Strain (no rescaling)
    - 1: 22 mode factor
    - 2: lm mode factor (mode-dependent)

    Parameters
    ----------
    powers_of_Mf : dict
        Dictionary of powers of the frequency (Mf)
    pAmp : dict
        Amplitude coefficients structure containing:
        - ampNorm: Amplitude normalization
        - PNdominant: PN dominant term
        - PNdominantlm: Mode-specific PN dominant term
        - PNdominantlmpower: Power for the mode-specific term (1, 2, 3, or 4)
    rescalefactor : int
        Rescale factor version (0, 1, or 2)

    Returns
    -------
    float
        The rescaling factor

    Raises
    ------
    ValueError
        If rescalefactor is not 0, 1, or 2
    """
    if rescalefactor == 0:  # Strain
        factor = 1.0

    elif rescalefactor == 1:  # 22factor
        factor = pAmp["ampNorm"] * powers_of_Mf["m_seven_sixths"]

    elif rescalefactor == 2:  # lmfactor
        PNdominantlmpower = pAmp["PNdominantlmpower"]
        base_factor = (
            pAmp["PNdominant"] * powers_of_Mf["m_seven_sixths"] * pAmp["PNdominantlm"]
        )

        if PNdominantlmpower == 1:
            factor = base_factor * powers_of_Mf["one_third"]
        elif PNdominantlmpower == 2:
            factor = base_factor * powers_of_Mf["two_thirds"]
        elif PNdominantlmpower == 3:
            factor = base_factor * powers_of_Mf["itself"]
        elif PNdominantlmpower == 4:
            factor = base_factor * powers_of_Mf["four_thirds"]
        else:
            raise ValueError(
                f"Error in RescaleFactor: PNdominantlmpower {PNdominantlmpower} is not valid. "
                f"Valid values are 1, 2, 3, or 4."
            )

    else:
        raise ValueError(
            f"Error in RescaleFactor: version {rescalefactor} is not valid. "
            f"Recommended version is 1."
        )

    return factor


def IMRPhenomXHM_Intermediate_CollocPtsFreqs(
    pPhase: dict, pWFHM: dict, pWF22: dict
) -> dict:
    """
    Initialize frequencies of collocation points for intermediate phase reconstruction.

    The frequencies are stored in the pPhase struct, which gets initialized with new
    values as the code processes each mode. This function sets up the collocation points
    for the intermediate region between inspiral and ringdown.

    Parameters
    ----------
    pPhase : dict
        Phase coefficient dictionary (will be updated with collocation frequencies)
    pWFHM : dict
        Waveform structure for the higher mode containing:
        - fRING: Ringdown frequency for the mode
        - fDAMP: Damping frequency for the mode
        - IMRPhenomXHMIntermediatePhaseFreqsVersion: Version flag (122019 or 122022)
        - modeTag: Mode identifier (21, 33, 32, 44)
        - fMECOlm: MECO frequency for the mode
    pWF22 : dict
        Waveform structure for the (2,2) mode containing:
        - fRING: Ringdown frequency for (2,2) mode
        - fDAMP: Damping frequency for (2,2) mode
        - eta: Symmetric mass ratio
        - chi1L: Aligned spin of body 1

    Returns
    -------
    dict
        Updated pPhase dictionary with collocation point frequencies set

    Raises
    ------
    ValueError
        If version is not valid
    """
    import jax.numpy as jnp

    fring = pWFHM["fRING"]
    fdamp = pWFHM["fDAMP"]
    version = pWFHM["IMRPhenomXHMIntermediatePhaseFreqsVersion"]

    if version == 122019 or version == 122022:  # Default version
        fcut = GetfcutInsp(pWF22, pWFHM)
        pPhase["CollocationPointsFreqsPhaseInter"][0] = fcut

        if pWFHM["modeTag"] == 32:
            # Special handling for (3,2) mode with spheroidal harmonics
            fRD22 = pWF22["fRING"]
            fdamp22 = pWF22["fDAMP"]
            fEnd = fRD22 - 0.5 * fdamp22

            pPhase["CollocationPointsFreqsPhaseInter"][1] = (
                jnp.sqrt(3.0) * (fcut - fEnd) + 2.0 * (fcut + fEnd)
            ) / 4.0
            pPhase["CollocationPointsFreqsPhaseInter"][2] = (3.0 * fcut + fEnd) / 4.0
            pPhase["CollocationPointsFreqsPhaseInter"][3] = (fcut + fEnd) / 2.0
            # We use first and second derivative at fEnd, so this frequency is duplicated
            pPhase["CollocationPointsFreqsPhaseInter"][4] = fEnd
            pPhase["CollocationPointsFreqsPhaseInter"][5] = fEnd
            pPhase["fPhaseMatchIM"] = fEnd

            # Correct cutting frequency for EMR with negative spins
            if pWF22["eta"] < 0.01 and pWF22["chi1L"] < 0 and version == 122019:
                pPhase["fPhaseMatchIM"] = pPhase["fPhaseMatchIM"] * (
                    1.2 - 0.25 * pWF22["chi1L"]
                )

        else:
            # For modes 21, 33, 44 (spherical harmonics)
            pPhase["CollocationPointsFreqsPhaseInter"][1] = (
                jnp.sqrt(3.0) * (fcut - fring) + 2.0 * (fcut + fring)
            ) / 4.0
            pPhase["CollocationPointsFreqsPhaseInter"][2] = (3.0 * fcut + fring) / 4.0
            pPhase["CollocationPointsFreqsPhaseInter"][3] = (fcut + fring) / 2.0
            pPhase["CollocationPointsFreqsPhaseInter"][4] = (fcut + 3.0 * fring) / 4.0
            pPhase["CollocationPointsFreqsPhaseInter"][5] = (fcut + 7.0 * fring) / 8.0
            pPhase["fPhaseMatchIM"] = fring - fdamp

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Intermediate_CollocPtsFreqs: version {version} is not valid. "
            f"Version recommended is 122019."
        )

    pPhase["fPhaseMatchIN"] = pWFHM["fMECOlm"]

    return pPhase


def SpheroidalToSpherical(
    powers_of_Mf: dict,
    pAmp22: dict,
    pPhase22: dict,
    pAmplm: dict,
    pPhaselm: dict,
    pWFlm: dict,
    pWF22: dict,
) -> complex:
    """
    Convert spheroidal harmonics to spherical harmonics for mode mixing.

    This function rotates the waveform from a spheroidal harmonic basis to a spherical
    harmonic basis. Currently implemented for the (3,2) mode, which requires mixing
    with the (2,2) mode due to spheroidal effects near the ringdown.

    In principle, this could be generalized to the (4,3) mode, but for now it assumes
    the mode being solved is only the (3,2) mode.

    Parameters
    ----------
    powers_of_Mf : dict
        Dictionary of powers of the frequency (Mf)
    pAmp22 : dict
        Amplitude coefficients for the (2,2) mode
    pPhase22 : dict
        Phase coefficients for the (2,2) mode
    pAmplm : dict
        Amplitude coefficients for the (l,m) mode
    pPhaselm : dict
        Phase coefficients for the (l,m) mode
    pWFlm : dict
        Waveform structure for the (l,m) mode containing:
        - timeshift: Time shift parameter
        - phaseshift: Phase shift parameter
        - phiref22: Reference phase for (2,2) mode
        - IMRPhenomXHMRingdownAmpVersion: Ringdown amplitude version
        - ampNorm: Amplitude normalization
        - mixingCoeffs: Array of mixing coefficients [c_222, c_223, c_322, c_323]
    pWF22 : dict
        Waveform structure for the (2,2) mode containing:
        - fRING: Ringdown frequency
        - fDAMP: Damping frequency
        - eta: Symmetric mass ratio

    Returns
    -------
    complex
        The spherical harmonic waveform (complex amplitude * exp(i*phase))
    """
    import jax.numpy as jnp

    Mf = powers_of_Mf["itself"]

    # Compute the 22 mode using PhenomX functions
    # This gives the 22 mode rescaled with the leading order (because 32 is also rescaled)
    amp22 = XLALSimIMRPhenomXRingdownAmplitude22AnsatzAnalytical(
        Mf,
        pWF22["fRING"],
        pWF22["fDAMP"],
        pAmp22["gamma1"],
        pAmp22["gamma2"],
        pAmp22["gamma3"],
    )

    phi22 = (
        1.0 / pWF22["eta"] * IMRPhenomX_Phase_22(Mf, powers_of_Mf, pPhase22, pWF22)
        + pWFlm["timeshift"] * Mf
        + pWFlm["phaseshift"]
        + pWFlm["phiref22"]
    )

    wf22R = amp22 * jnp.exp(1j * phi22)

    if pWFlm["IMRPhenomXHMRingdownAmpVersion"] != 0:
        wf22R *= pWFlm["ampNorm"] * powers_of_Mf["m_seven_sixths"]

    # Compute 32 mode in spheroidal basis
    amplm = IMRPhenomXHM_RD_Amp_Ansatz(powers_of_Mf, pWFlm, pAmplm)
    philm = IMRPhenomXHM_RD_Phase_AnsatzInt(Mf, powers_of_Mf, pWFlm, pPhaselm)

    # Perform the rotation from spheroidal to spherical basis
    # Using mixing coefficients: c_322 (index 2) and c_323 (index 3)
    sphericalWF_32 = jnp.conj(pWFlm["mixingCoeffs"][2]) * wf22R + jnp.conj(
        pWFlm["mixingCoeffs"][3]
    ) * amplm * jnp.exp(1j * philm)

    return sphericalWF_32


def IMRPhenomXHM_PN21AmpSign(ff: float, wf22: dict) -> int:
    """
    Determine the sign of the Post-Newtonian 21 mode amplitude.

    This function computes the sign of the PN amplitude for the (2,1) mode at a given
    frequency. The sign can flip across the parameter space, and this must be accounted
    for in the phase reconstruction since the model amplitude is positive by construction.

    Parameters
    ----------
    ff : float
        Frequency at which to evaluate the PN amplitude sign
    wf22 : dict
        Waveform structure for the (2,2) mode containing:
        - eta: Symmetric mass ratio
        - chi1L: Aligned spin of body 1
        - chi2L: Aligned spin of body 2

    Returns
    -------
    int
        1 if the PN amplitude is positive or zero, -1 if negative
    """
    import jax.numpy as jnp

    eta = wf22["eta"]
    chi1 = wf22["chi1L"]
    chi2 = wf22["chi2L"]
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    # Compute PN amplitude expression up to relevant order
    output = (
        (-16.0 * delta * eta * ff * jnp.power(jnp.pi, 1.5)) / (3.0 * jnp.sqrt(5.0))
        + (
            4.0
            * jnp.power(2.0, 1.0 / 3.0)
            * (chi1 - chi2 + delta * (chi1 + chi2))
            * eta
            * jnp.power(ff, 4.0 / 3.0)
            * jnp.power(jnp.pi, 11.0 / 6.0)
        )
        / jnp.sqrt(5.0)
        + (
            2.0
            * jnp.power(2.0, 2.0 / 3.0)
            * eta
            * (306.0 * delta - 360.0 * delta * eta)
            * jnp.power(ff, 5.0 / 3.0)
            * jnp.power(jnp.pi, 13.0 / 6.0)
        )
        / (189.0 * jnp.sqrt(5.0))
    )

    if output >= 0:
        return 1
    else:
        return -1


def IMRPhenomXHM_Amplitude_fcutRD(pWFHM: dict, pWF22: dict) -> float:
    """
    Ringdown cutting frequency for the amplitude.

    Returns the end of the intermediate region and the beginning of the ringdown
    for the amplitude of one mode. The cutting frequency depends on the mode and
    the version of the model.

    Parameters
    ----------
    pWFHM : dict
        Waveform structure for the higher mode containing:
        - fRING: Ringdown frequency for the mode
        - fDAMP: Damping frequency for the mode
        - modeTag: Mode identifier (21, 33, 32, 44)
        - IMRPhenomXHMRingdownAmpFreqsVersion: Version flag (122018 or 122022)
        - MixingOn: Flag for mode mixing (0 or 1)
    pWF22 : dict
        Waveform structure for the (2,2) mode containing:
        - eta: Symmetric mass ratio
        - chi1L: Aligned spin of body 1
        - fRING: Ringdown frequency for the (2,2) mode
        - fDAMP: Damping frequency for the (2,2) mode

    Returns
    -------
    float
        The cutting frequency between intermediate and ringdown regions

    Raises
    ------
    ValueError
        If version is not valid
    """
    import jax.numpy as jnp

    fring = pWFHM["fRING"]
    fdamp = pWFHM["fDAMP"]
    version = pWFHM["IMRPhenomXHMRingdownAmpFreqsVersion"]
    eta = pWF22["eta"]
    chi1 = pWF22["chi1L"]

    if version == 122018:  # Default version
        modeTag = pWFHM["modeTag"]

        if modeTag == 21:
            fcut = 0.75 * fring

        elif modeTag == 33:
            fcut = 0.95 * fring

        elif modeTag == 32:
            fRD22 = pWF22["fRING"]
            c = 0.5
            r = 5.0

            if eta < 0.0453515:
                # For extreme mass ratios (q > 20)
                # Smooth step function between fring (negative spins) and fRD22 (positive spins)
                fcut = (fring * jnp.exp(c * r) + fRD22 * jnp.exp(r * chi1)) / (
                    jnp.exp(c * r) + jnp.exp(r * chi1)
                ) - fdamp
            else:
                # For comparable mass ratios
                fcut = fRD22

            # Special case for 6 < q < 45 with high spin
            if 0.02126654064272212 < eta < 0.12244897959183673 and chi1 > 0.95:
                fcut = fring - 2.0 * fdamp

        elif modeTag == 44:
            fcut = 0.9 * fring

        else:
            raise ValueError(
                f"Error in IMRPhenomXHM_Amplitude_fcutRD: modeTag {modeTag} is not valid. "
                f"Valid modes are 21, 33, 32, 44."
            )

    elif version == 122022:
        if pWFHM["MixingOn"] == 1:
            fcut = pWF22["fRING"] - 0.5 * pWF22["fDAMP"]  # v8
        else:
            fcut = fring - fdamp  # v2

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Amplitude_fcutRD: version {version} is not valid. "
            f"Valid versions are 122018 or 122022."
        )

    return fcut


def IMRPhenomXHM_Amplitude_fcutInsp(pWFHM: dict, pWF22: dict) -> float:
    """
    Inspiral cutting frequency for the amplitude.

    Returns the end frequency of the inspiral region and the beginning of the
    intermediate region for the amplitude of one mode.

    Parameters
    ----------
    pWFHM : dict
        Waveform structure for the higher mode containing:
        - IMRPhenomXHMInspiralAmpFreqsVersion: Version flag (122018 or 122022)
        - fMECOlm: MECO frequency for the mode
        - emm: Azimuthal quantum number m
        - modeTag: Mode identifier (21, 33, 32, 44)
        - fRING: Ringdown frequency for the mode
    pWF22 : dict
        Waveform structure for the (2,2) mode containing:
        - eta: Symmetric mass ratio
        - chi1L: Aligned spin of body 1
        - chiEff: Effective aligned spin
        - fISCO: ISCO frequency
        - fRING: Ringdown frequency for the (2,2) mode
        - q: Mass ratio

    Returns
    -------
    float
        The cutting frequency between inspiral and intermediate regions

    Raises
    ------
    ValueError
        If version is not valid
    """
    import jax.numpy as jnp

    version = pWFHM["IMRPhenomXHMInspiralAmpFreqsVersion"]
    fMECO = pWFHM["fMECOlm"]
    emm = float(pWFHM["emm"])
    eta = pWF22["eta"]
    chi1 = pWF22["chi1L"]

    # Cutting frequency for extreme mass ratios (fit to geometrical structure)
    fcutEMR = (
        1.25
        * emm
        * (
            (
                0.011671068725758493
                - 0.0000858396080377194 * chi1
                + 0.000316707064291237 * chi1**2
            )
            * (0.8447212540381764 + 6.2873167352395125 * eta)
        )
        / (1.2857082764038923 - 0.9977728883419751 * chi1)
    )

    if version == 122018:  # Default version
        fring = pWFHM["fRING"]
        chieff = pWF22["chiEff"]
        fISCO = pWF22["fISCO"] * emm * 0.5
        modeTag = pWFHM["modeTag"]

        if modeTag == 21:
            if eta < 0.023795359904818562:  # EMR (q > 40)
                fcut = fcutEMR
            else:  # Comparable mass ratios
                fcut = fMECO + (
                    0.75 - 0.235 * chieff - 5.0 / 6.0 * chieff * chieff
                ) * jnp.abs(fISCO - fMECO)

        elif modeTag == 33:
            if eta < 0.04535147392290249:  # EMR (q > 20)
                fcut = fcutEMR
            else:  # Comparable mass ratios
                fcut = fMECO + (0.75 - 0.235 * chieff - 5.0 / 6.0 * chieff) * jnp.abs(
                    fISCO - fMECO
                )

        elif modeTag == 32:
            if eta < 0.04535147392290249:  # EMR (q > 20)
                fcut = fcutEMR
            else:  # Comparable mass ratios
                fcut = fMECO + (0.75 - 0.235 * jnp.abs(chieff)) * jnp.abs(fISCO - fMECO)
                fcut = fcut * fring / pWF22["fRING"]

        elif modeTag == 44:
            if eta < 0.04535147392290249:  # EMR (q > 20)
                fcut = fcutEMR
            else:  # Comparable mass ratios
                fcut = fMECO + (0.75 - 0.235 * chieff) * jnp.abs(fISCO - fMECO)

        else:
            raise ValueError(
                f"Error in IMRPhenomXHM_Amplitude_fcutInsp: modeTag {modeTag} is not valid. "
                f"Valid modes are 21, 33, 32, 44."
            )

    elif version == 122022:
        if pWF22["q"] < 20.0:
            fcut = fMECO
        else:
            transition_eta = 0.0192234  # q = 50
            sharpness = 0.004
            funcs = 0.5 + 0.5 * jnp.tanh((eta - transition_eta) / sharpness)
            fcut = funcs * fMECO + (1.0 - funcs) * fcutEMR

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Amplitude_fcutInsp: version {version} is not valid. "
            f"Valid versions are 122018 or 122022."
        )

    return fcut


def Get21PNAmplitudeCoefficients(pAmp: dict, pWF22: dict) -> None:
    """
    Post-Newtonian Inspiral Ansatz Coefficients for the 21 mode.

    The 21 ansatz in Fourier Domain is built multiplying the Time-domain
    Post-Newtonian series up to 3PN by the phasing factor given by the
    Stationary-Phase-Approximation.

    This function fills the pAmp dictionary with time-domain PN coefficients
    and phasing factor expansion coefficients for the (2,1) mode.

    Parameters
    ----------
    pAmp : dict
        Amplitude coefficients dictionary to be filled with:
        - PNTDfactor: Overall time-domain factor
        - x05, x1, x15, x2, x25, x3: Time-domain PN coefficients
        - xdot5, xdot6, xdot65, xdot7, xdot75, xdot8, xdot8Log, xdot85: Phasing coefficients
        - log2pi_two_thirds: Log of (2π)^(2/3)
    pWF22 : dict
        Waveform structure for the (2,2) mode containing:
        - m1, m2: Component masses
        - chi1L, chi2L: Aligned spins
        - delta: Mass difference parameter
        - eta: Symmetric mass ratio

    Returns
    -------
    None
        The function modifies pAmp in place
    """
    import jax.numpy as jnp

    m1 = pWF22["m1"]
    m2 = pWF22["m2"]
    m12 = m1 * m1
    m22 = m2 * m2
    m13 = m12 * m1
    m23 = m22 * m2
    m14 = m13 * m1
    m24 = m23 * m2
    m15 = m14 * m1
    m25 = m24 * m2
    m16 = m15 * m1

    chi1 = pWF22["chi1L"]
    chi2 = pWF22["chi2L"]
    chiS = (chi1 + chi2) * 0.5
    chiA = (chi1 - chi2) * 0.5
    delta = pWF22["delta"]
    eta = pWF22["eta"]
    chi12 = chi1 * chi1
    chi22 = chi2 * chi2
    Sc = m12 * chi1 + m22 * chi2
    Sigmac = m2 * chi2 - m1 * chi1

    # Powers of 2
    powers_of_2 = jnp.power(
        2.0,
        jnp.array(
            [1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0, 7.0 / 3.0, 8.0 / 3.0]
        ),
    )
    pow_2_one_third = powers_of_2[0]
    pow_2_two_thirds = powers_of_2[1]
    pow_2_itself = powers_of_2[2]
    pow_2_four_thirds = powers_of_2[3]
    pow_2_five_thirds = powers_of_2[4]
    pow_2_two = powers_of_2[5]
    pow_2_seven_thirds = powers_of_2[6]
    pow_2_eight_thirds = powers_of_2[7]

    logof2 = jnp.log(2.0)
    log4 = 1.3862943611198906
    EulerGamma = 0.5772156649015329

    # Powers of pi
    pi_sqrt = jnp.sqrt(jnp.pi)
    pi_one_third = jnp.power(jnp.pi, 1.0 / 3.0)
    pi_two_thirds = jnp.power(jnp.pi, 2.0 / 3.0)
    pi_itself = jnp.pi
    pi_four_thirds = jnp.power(jnp.pi, 4.0 / 3.0)
    pi_five_thirds = jnp.power(jnp.pi, 5.0 / 3.0)
    pi_two = jnp.pi * jnp.pi
    pi_seven_thirds = jnp.power(jnp.pi, 7.0 / 3.0)
    pi_eight_thirds = jnp.power(jnp.pi, 8.0 / 3.0)

    # Complex coefficients of the Time-Domain Post-Newtonian expansion
    factor = (
        8.0 * pWF22["eta"] * pi_two_thirds * pow_2_two_thirds * pi_sqrt / jnp.sqrt(5.0)
    )
    pAmp["PNTDfactor"] = factor
    pAmp["x05"] = 1j * delta / 3.0 * pow_2_one_third * pi_one_third
    pAmp["x1"] = -1j * 0.5 * (chiA + chiS * delta) * pow_2_two_thirds * pi_two_thirds
    pAmp["x15"] = (
        1j * delta * (-17.0 / 28.0 + 5.0 * eta / 7.0) / 3.0 * pow_2_itself * pi_itself
    )
    pAmp["x2"] = (
        (
            1j * (-43.0 / 21.0 * delta * Sc + (-79.0 + 139.0 * eta) / 42.0 * Sigmac)
            + 1j / 3.0 * delta * (pi_itself + 1j * (-0.5 - 2.0 * logof2))
        )
        * pow_2_four_thirds
        * pi_four_thirds
    )
    pAmp["x25"] = (
        1j
        * delta
        * ((-43.0 - 509.0 * eta) / 126.0 + 79.0 * eta * eta / 168.0)
        / 3.0
        * pow_2_five_thirds
        * pi_five_thirds
    )
    pAmp["x3"] = (
        1j
        * delta
        * (
            (-17.0 + 6.0 * eta) / 28.0 * pi_itself
            + 1j
            * (
                17.0 / 56.0
                + eta * (-353.0 / 28.0 - 3.0 * logof2 / 7.0)
                + 17.0 * logof2 / 14.0
            )
        )
        / 3.0
        * pow_2_two
        * pi_two
    )

    # Coefficients of the phasing factor expansion (equation E4 of arXiv:2001.10914)
    pAmp["xdot5"] = (
        -(
            m1
            * m2
            * (-838252800.0 * m1 * m2 - 419126400.0 * m12 - 419126400.0 * m22)
            / 3.274425e7
        )
        * pow_2_five_thirds
        * pow_2_five_thirds
        * pi_five_thirds
        * pi_five_thirds
    )

    pAmp["xdot6"] = (
        -(
            m1
            * m2
            * (
                1152597600.0 * m2 * m13
                + 926818200.0 * m22
                + 2494800.0 * m1 * m2 * (743.0 + 462.0 * m22)
                + 1247400.0 * m12 * (743.0 + 1848.0 * m22)
            )
            / 3.274425e7
        )
        * jnp.power(pow_2_two, 2)
        * jnp.power(pi_two, 2)
    )

    pAmp["xdot65"] = (
        -(
            m1
            * m2
            * (
                -34927200.0
                * m1
                * m2
                * (-(m2 * (75.0 * chi1 + 376.0 * chi2 * m2)) + 96.0 * pi_itself)
                - 34927200.0
                * (
                    -(m2 * (75.0 * chi2 + 188.0 * (chi1 + chi2) * m2))
                    + 48.0 * pi_itself
                )
                * m12
                - 2619540000.0 * chi1 * m13
                + 13132627200.0 * chi1 * m2 * m13
                + 6566313600.0 * chi1 * m14
                - 34927200.0
                * (chi2 * (75.0 - 188.0 * m2) * m2 + 48.0 * pi_itself)
                * m22
            )
            / 3.274425e7
        )
        * pow_2_eight_thirds
        * pow_2_five_thirds
        * pi_eight_thirds
        * pi_five_thirds
    )

    pAmp["xdot7"] = (
        -(
            m1
            * m2
            * (
                207900.0
                * m2
                * (-13661.0 - 19908.0 * chi1 * chi2 + 10206.0 * chi12 + 10206.0 * chi22)
                * m13
                - 23100.0 * (34103.0 + 91854.0 * chi22) * m22
                - 1373803200.0 * m14 * m22
                + 23100.0
                * m1
                * m2
                * (
                    -2.0 * (34103.0 + 45927.0 * chi12 + 45927.0 * chi22)
                    + 9.0
                    * (
                        -13661.0
                        - 19908.0 * chi1 * chi2
                        + 10206.0 * chi12
                        + 10206.0 * chi22
                    )
                    * m22
                )
                - 2747606400.0 * m13 * m23
                - 23100.0
                * m12
                * (
                    34103.0
                    + 91854.0 * chi12
                    - 18.0
                    * (
                        -13661.0
                        - 19908.0 * chi1 * chi2
                        + 10206.0 * chi12
                        + 10206.0 * chi22
                    )
                    * m22
                    + 59472.0 * m24
                )
            )
            / 3.274425e7
        )
        * pow_2_seven_thirds
        * pow_2_seven_thirds
        * pi_seven_thirds
        * pi_seven_thirds
    )

    pAmp["xdot75"] = (
        -(
            m1
            * m2
            * (
                -4036586400.0 * chi1 * m13
                + 5821200.0 * m2 * (5861.0 * chi1 + 1701.0 * pi_itself) * m13
                + 17059026600.0 * chi1 * m14
                + 14721814800.0 * chi1 * m2 * m14
                - 34962127200.0 * chi1 * m2 * m15
                + 207900.0
                * (2.0 * chi2 * m2 * (-9708.0 + 41027.0 * m2) + 12477.0 * pi_itself)
                * m22
                - 14721814800.0 * chi2 * m13 * m22
                - 69924254400.0 * chi1 * m14 * m22
                - 34962127200.0 * (chi1 + chi2) * m13 * m23
                + 207900.0
                * m12
                * (
                    3.0 * pi_itself * (4159.0 + 31752.0 * m22)
                    + 2.0
                    * m2
                    * (
                        9708.0 * chi2
                        + 41027.0 * (chi1 + chi2) * m2
                        - 35406.0 * chi1 * m22
                        - 168168.0 * chi2 * m23
                    )
                )
                + 415800.0
                * m1
                * m2
                * (
                    9708.0 * chi1 * m2
                    + 82054.0 * chi2 * m22
                    + 3.0 * pi_itself * (4159.0 + 7938.0 * m22)
                    + 35406.0 * chi2 * m23
                    - 84084.0 * chi2 * m24
                )
            )
            / 3.274425e7
        )
        * pow_2_eight_thirds
        * pow_2_seven_thirds
        * pi_eight_thirds
        * pi_seven_thirds
    )

    pAmp["xdot8"] = (
        -(
            m1
            * m2
            * (
                -10548014400.0 * chi1 * pi_itself * m13
                - 63392868000.0 * chi1 * chi2 * m2 * m14
                + 34927200.0 * chi1 * (-375.0 * chi1 + 752.0 * pi_itself) * m14
                + 63392868000.0 * chi12 * m15
                - 153213984000.0 * m2 * chi12 * m15
                - 76606992000.0 * chi12 * m16
                - 63392868000.0 * chi1 * (chi1 - chi2) * m13 * m22
                - 51975.0
                * (
                    4869.0
                    + 2711352.0 * chi1 * chi2
                    + 1702428.0 * chi12
                    + 228508.0 * chi22
                )
                * m14
                * m22
                - 103950.0
                * (
                    4869.0
                    + 2711352.0 * chi1 * chi2
                    + 228508.0 * chi12
                    + 228508.0 * chi22
                )
                * m13
                * m23
                + 906328500.0 * m15 * m23
                + 1812657000.0 * m14 * m24
                + 906328500.0 * m13 * m25
                + 1925.0
                * m2
                * m13
                * (
                    56198689.0
                    + 13635864.0 * chi1 * chi2
                    + 27288576.0 * chi1 * pi_itself
                    + 30746952.0 * chi12
                    + 3617892.0 * chi22
                    - 2045736.0 * pi_two
                )
                - 3.0
                * m22
                * (
                    16447322263.0
                    - 2277918720.0 * EulerGamma
                    - 23284800.0 * chi2 * m2 * (-151.0 + 376.0 * m2) * pi_itself
                    - 2277918720.0 * log4
                    + 2321480700.0 * chi22
                    + 4365900000.0 * chi22 * m22
                    - 21130956000.0 * chi22 * m23
                    + 25535664000.0 * chi22 * m24
                    + 745113600.0 * pi_two
                )
                + m12
                * (
                    6833756160.0 * EulerGamma
                    + 10548014400.0 * chi2 * m2 * pi_itself
                    + 63392868000.0 * (chi1 - chi2) * chi2 * m23
                    - 51975.0
                    * (
                        4869.0
                        + 2711352.0 * chi1 * chi2
                        + 228508.0 * chi12
                        + 1702428.0 * chi22
                    )
                    * m24
                    - 3850.0
                    * m22
                    * (
                        -56198689.0
                        + 13580136.0 * chi1 * chi2
                        - 6822144.0 * (chi1 + chi2) * pi_itself
                        - 6976422.0 * chi12
                        - 6976422.0 * chi22
                        + 2045736.0 * pi_two
                    )
                    - 3.0
                    * (
                        16447322263.0
                        - 2277918720.0 * log4
                        + 2321480700.0 * chi12
                        + 745113600.0 * pi_two
                    )
                )
                + m1
                * m2
                * (
                    13667512320.0 * EulerGamma
                    + 10548014400.0 * chi1 * m2 * pi_itself
                    - 63392868000.0 * chi1 * chi2 * m23
                    - 153213984000.0 * chi22 * m24
                    - 1925.0
                    * m22
                    * (
                        -56198689.0
                        - 13635864.0 * chi1 * chi2
                        - 27288576.0 * chi2 * pi_itself
                        - 3617892.0 * chi12
                        - 30746952.0 * chi22
                        + 2045736.0 * pi_two
                    )
                    - 6.0
                    * (
                        16447322263.0
                        - 2277918720.0 * log4
                        + 1160740350.0 * chi12
                        + 1160740350.0 * chi22
                        + 745113600.0 * pi_two
                    )
                )
            )
            / 3.274425e7
        )
        * pow_2_eight_thirds
        * pow_2_eight_thirds
        * pi_eight_thirds
        * pi_eight_thirds
    )

    pAmp["xdot8Log"] = (
        -(m1 * m2 * 3416878080.0)
        / 3.274425e7
        * pow_2_eight_thirds
        * pow_2_eight_thirds
        * pi_eight_thirds
        * pi_eight_thirds
    )

    pAmp["xdot85"] = (
        -(
            m1
            * m2
            * (
                -14891068500.0 * chi1 * m13
                + 1925.0
                * m2
                * (97151928.0 * chi1 + 6613488.0 * chi2 - 12912300.0 * pi_itself)
                * m13
                + 87143248500.0 * chi1 * m14
                + 33313480200.0 * chi1 * m2 * m14
                - 198816225300.0 * chi1 * m2 * m15
                + 57750.0
                * (2.0 * chi2 * m2 * (-128927.0 + 754487.0 * m2) + 7947.0 * pi_itself)
                * m22
                - 33313480200.0 * chi2 * m13 * m22
                - 138600.0
                * (3399633.0 * chi1 + 530712.0 * chi2 + 182990.0 * pi_itself)
                * m14
                * m22
                - 35665037100.0 * chi1 * m15 * m22
                + 84184254000.0 * chi1 * m16 * m22
                - 23100.0
                * m1
                * m2
                * (
                    15.0 * pi_itself * (-2649.0 + 71735.0 * m22)
                    + m2
                    * (
                        -(chi1 * (644635.0 + 551124.0 * m2))
                        + chi2 * m2 * (-8095994.0 - 1442142.0 * m2 + 8606763.0 * m22)
                    )
                )
                - 69300.0
                * (4991769.0 * (chi1 + chi2) + 731960.0 * pi_itself)
                * m13
                * m23
                + 35665037100.0 * chi2 * m14 * m23
                + 170726094000.0 * chi1 * m15 * m23
                + 2357586000.0 * chi2 * m15 * m23
                + 35665037100.0 * chi1 * m13 * m24
                + 88899426000.0 * (chi1 + chi2) * m14 * m24
                + 9702000.0 * (243.0 * chi1 + 17597.0 * chi2) * m13 * m25
                - 11550.0
                * m12
                * (
                    15.0 * pi_itself * (-2649.0 + 286940.0 * m22 + 146392.0 * m24)
                    + 2.0
                    * m2
                    * (
                        -644635.0 * chi2
                        - 4874683.0 * (chi1 + chi2) * m2
                        + 1442142.0 * chi1 * m22
                        + 54.0 * (58968.0 * chi1 + 377737.0 * chi2) * m23
                        + 1543941.0 * chi2 * m24
                        - 3644340.0 * chi2 * m25
                    )
                )
            )
            / 3.274425e7
        )
        * pow_2_eight_thirds
        * pow_2_eight_thirds
        * pow_2_one_third
        * pi_eight_thirds
        * pi_eight_thirds
        * pi_one_third
    )

    pAmp["log2pi_two_thirds"] = jnp.log(pi_two_thirds * pow_2_two_thirds)


def IMRPhenomXHM_GetPNAmplitudeCoefficients(
    pAmp: dict, pWFHM: dict, pWF22: dict
) -> None:
    """
    Fill pAmp with coefficients of power series in frequency for the
    Fourier Domain Post-Newtonian Inspiral Ansatz.

    The ansatz in Fourier Domain is built by multiplying the Time-domain
    Post-Newtonian series up to 3PN by the phasing factor given by the
    Stationary-Phase-Approximation, and then re-expanding in powers of f up to 3PN.

    The 21 mode by default does not use the power series because it breaks down
    before the end of the inspiral, but a corresponding power series is available.

    The coefficients correspond to those in equations E10-E14 in arXiv:2001.10914.

    Parameters
    ----------
    pAmp : dict
        Amplitude coefficients dictionary to be filled with:
        - PNglobalfactor: Global prefactor for the mode
        - pnInitial, pnOneThird, pnTwoThirds, pnThreeThirds, pnFourThirds,
          pnFiveThirds, pnSixThirds: PN frequency series coefficients
    pWFHM : dict
        Waveform structure for the higher mode containing:
        - emm: Mode m value (1, 2, 3, or 4)
        - modeTag: Mode identifier (21, 33, 32, 44)
        - modeInt: Mode integer index (0, 1, 2, 3)
        - chi_a: Antisymmetric spin combination
        - chi_s: Symmetric spin combination
        - useFAmpPN: Flag for 21 mode (1 = use alternative form, 0 = power series)
    pWF22 : dict
        Waveform structure for the (2,2) mode containing:
        - eta: Symmetric mass ratio
        - delta: Mass difference parameter

    Returns
    -------
    None
        The function modifies pAmp in place

    Raises
    ------
    ValueError
        If modeTag is not valid (21, 33, 32, 44)
    """
    import jax.numpy as jnp

    chiA = pWFHM["chi_a"]
    chiS = pWFHM["chi_s"]
    eta = pWF22["eta"]
    delta = pWF22["delta"]
    PI = jnp.pi

    # Global factors of each PN h_lm
    prefactors = jnp.array(
        [
            jnp.sqrt(2.0) / 3.0,  # 21 mode
            0.75 * jnp.sqrt(5.0 / 7.0),  # 33 mode
            jnp.sqrt(5.0 / 7.0) / 3.0,  # 32 mode
            4.0 * jnp.sqrt(2.0) / 9.0 * jnp.sqrt(5.0 / 7.0),  # 44 mode
        ]
    )

    # Compensate for rescaling data with the leading order of the 22
    pAmp["PNglobalfactor"] = (
        jnp.power(2.0 / pWFHM["emm"], -7.0 / 6.0) * prefactors[pWFHM["modeInt"]]
    )

    modeTag = pWFHM["modeTag"]

    if modeTag == 21:
        if pWFHM["useFAmpPN"] == 1:
            # Use the more accurate non-reexpanded form
            Get21PNAmplitudeCoefficients(pAmp, pWF22)
            pAmp["pnInitial"] = 0.0
            pAmp["pnOneThird"] = 0.0
            pAmp["pnTwoThirds"] = 0.0
            pAmp["pnThreeThirds"] = 0.0
            pAmp["pnFourThirds"] = 0.0
            pAmp["pnFiveThirds"] = 0.0
            pAmp["pnSixThirds"] = 0.0
        else:
            # Use power series expansion
            pow_2_one_third = jnp.power(2.0, 1.0 / 3.0)
            pow_2_two_thirds = jnp.power(2.0, 2.0 / 3.0)
            pow_2_itself = 2.0
            pow_2_four_thirds = jnp.power(2.0, 4.0 / 3.0)
            pow_2_five_thirds = jnp.power(2.0, 5.0 / 3.0)
            pow_2_two = 4.0

            pi_one_third = jnp.power(PI, 1.0 / 3.0)
            pi_two_thirds = jnp.power(PI, 2.0 / 3.0)
            pi_itself = PI
            pi_four_thirds = jnp.power(PI, 4.0 / 3.0)
            pi_five_thirds = jnp.power(PI, 5.0 / 3.0)
            pi_two = PI * PI

            pAmp["pnInitial"] = 0.0
            pAmp["pnOneThird"] = delta * pi_one_third * pow_2_one_third
            pAmp["pnTwoThirds"] = (
                -3.0 * (chiA + chiS * delta) / 2.0 * pi_two_thirds * pow_2_two_thirds
            )
            pAmp["pnThreeThirds"] = (
                (335.0 * delta + 1404.0 * delta * eta)
                / 672.0
                * pi_itself
                * pow_2_itself
            )
            pAmp["pnFourThirds"] = (
                (
                    3427.0 * chiA
                    - 1j * 672.0 * delta
                    + 3427.0 * chiS * delta
                    - 8404.0 * chiA * eta
                    - 3860.0 * chiS * delta * eta
                    - 1344.0 * delta * PI
                    - 1j * 672.0 * delta * jnp.log(16.0)
                )
                / 1344.0
                * pi_four_thirds
                * pow_2_four_thirds
            )
            pAmp["pnFiveThirds"] = (
                (
                    -155965824.0 * chiA * chiS
                    - 964357.0 * delta
                    + 432843264.0 * chiA * chiS * eta
                    - 23670792.0 * delta * eta
                    + 24385536.0 * chiA * PI
                    + 24385536.0 * chiS * delta * PI
                    - 77982912.0 * delta * chiA * chiA
                    + 81285120.0 * delta * eta * chiA * chiA
                    - 77982912.0 * delta * chiS * chiS
                    + 39626496.0 * delta * eta * chiS * chiS
                    + 21535920.0 * delta * eta * eta
                )
                / 8.128512e6
                * pi_five_thirds
                * pow_2_five_thirds
            )
            pAmp["pnSixThirds"] = (
                (
                    143063173.0 * chiA
                    - 1j * 1350720.0 * delta
                    + 143063173.0 * chiS * delta
                    - 546199608.0 * chiA * eta
                    - 1j * 72043776.0 * delta * eta
                    - 169191096.0 * chiS * delta * eta
                    - 9898560.0 * delta * PI
                    + 20176128.0 * delta * eta * PI
                    - 1j * 5402880.0 * delta * jnp.log(2.0)
                    - 1j * 17224704.0 * delta * eta * jnp.log(2.0)
                    + 61725888.0 * chiS * delta * chiA * chiA
                    - 81285120.0 * chiS * delta * eta * chiA * chiA
                    + 20575296.0 * jnp.power(chiA, 3)
                    - 81285120.0 * eta * jnp.power(chiA, 3)
                    + 61725888.0 * chiA * chiS * chiS
                    - 165618432.0 * chiA * eta * chiS * chiS
                    + 20575296.0 * delta * jnp.power(chiS, 3)
                    - 1016064.0 * delta * eta * chiS * chiS * chiS
                    + 128873808.0 * chiA * eta * eta
                    - 3859632.0 * chiS * delta * eta * eta
                )
                / 5.419008e6
                * pi_two
                * pow_2_two
            )

    elif modeTag == 33:
        pow_2d3_one_third = jnp.power(2.0 / 3.0, 1.0 / 3.0)
        pow_2d3_itself = 2.0 / 3.0
        pow_2d3_four_thirds = jnp.power(2.0 / 3.0, 4.0 / 3.0)
        pow_2d3_five_thirds = jnp.power(2.0 / 3.0, 5.0 / 3.0)
        pow_2d3_two = jnp.power(2.0 / 3.0, 2.0)

        pi_one_third = jnp.power(PI, 1.0 / 3.0)
        pi_itself = PI
        pi_four_thirds = jnp.power(PI, 4.0 / 3.0)
        pi_five_thirds = jnp.power(PI, 5.0 / 3.0)
        pi_two = PI * PI

        pAmp["pnInitial"] = 0.0
        pAmp["pnOneThird"] = delta * pi_one_third * pow_2d3_one_third
        pAmp["pnTwoThirds"] = 0.0
        pAmp["pnThreeThirds"] = (
            (-1945.0 * delta + 2268.0 * delta * eta)
            / 672.0
            * pi_itself
            * pow_2d3_itself
        )
        pAmp["pnFourThirds"] = (
            (
                325.0 * chiA
                - 1j * 504.0 * delta
                + 325.0 * chiS * delta
                - 1120.0 * chiA * eta
                - 80.0 * chiS * delta * eta
                + 120.0 * delta * PI
                + 1j * 720.0 * delta * jnp.log(1.5)
            )
            / 120.0
            * pi_four_thirds
            * pow_2d3_four_thirds
        )
        pAmp["pnFiveThirds"] = (
            (
                -2263282560.0 * chiA * chiS
                - 1077664867.0 * delta
                + 9053130240.0 * chiA * chiS * eta
                - 5926068792.0 * delta * eta
                - 1131641280.0 * delta * chiA * chiA
                + 4470681600.0 * delta * eta * chiA * chiA
                - 1131641280.0 * delta * chiS * chiS
                + 55883520.0 * delta * eta * chiS * chiS
                + 2966264784.0 * delta * eta * eta
            )
            / 4.4706816e8
            * pi_five_thirds
            * pow_2d3_five_thirds
        )
        pAmp["pnSixThirds"] = (
            (
                22007835.0 * chiA
                + 1j * 26467560.0 * delta
                + 22007835.0 * chiS * delta
                - 80190540.0 * chiA * eta
                - 1j * 98774368.0 * delta * eta
                - 31722300.0 * chiS * delta * eta
                - 9193500.0 * delta * PI
                + 17826480.0 * delta * eta * PI
                - 1j * 37810800.0 * delta * jnp.log(1.5)
                + 1j * 37558080.0 * delta * eta * jnp.log(1.5)
                - 12428640.0 * chiA * eta * eta
                - 6078240.0 * chiS * delta * eta * eta
            )
            / 2.17728e6
            * pi_two
            * pow_2d3_two
        )

    elif modeTag == 32:
        pi_two_thirds = jnp.power(PI, 2.0 / 3.0)
        pi_itself = PI
        pi_four_thirds = jnp.power(PI, 4.0 / 3.0)
        pi_five_thirds = jnp.power(PI, 5.0 / 3.0)
        pi_two = PI * PI

        pAmp["pnInitial"] = 0.0
        pAmp["pnOneThird"] = 0.0
        pAmp["pnTwoThirds"] = (-1.0 + 3.0 * eta) * pi_two_thirds
        pAmp["pnThreeThirds"] = -4.0 * chiS * eta * pi_itself
        pAmp["pnFourThirds"] = (
            (10471.0 - 61625.0 * eta + 82460.0 * eta * eta) / 10080.0 * pi_four_thirds
        )
        pAmp["pnFiveThirds"] = (
            (
                1j * 2520.0
                - 3955.0 * chiS
                - 3955.0 * chiA * delta
                - 1j * 11088.0 * eta
                + 10810.0 * chiS * eta
                + 11865.0 * chiA * delta * eta
                - 12600.0 * chiS * eta * eta
            )
            / 840.0
            * pi_five_thirds
        )
        pAmp["pnSixThirds"] = (
            (
                824173699.0
                + 2263282560.0 * chiA * chiS * delta
                - 26069649.0 * eta
                - 15209631360.0 * chiA * chiS * delta * eta
                + 3576545280.0 * chiS * eta * PI
                + 1131641280.0 * chiA * chiA
                - 7865605440.0 * eta * chiA * chiA
                + 1131641280.0 * chiS * chiS
                - 11870591040.0 * eta * chiS * chiS
                - 13202119896.0 * eta * eta
                + 13412044800.0 * chiA * chiA * eta * eta
                + 5830513920.0 * chiS * chiS * eta * eta
                + 5907445488.0 * jnp.power(eta, 3)
            )
            / 4.4706816e8
            * pi_two
        )

    elif modeTag == 44:
        pow_0p5_two_thirds = jnp.power(0.5, 2.0 / 3.0)
        pow_0p5_four_thirds = jnp.power(0.5, 4.0 / 3.0)
        pow_0p5_five_thirds = jnp.power(0.5, 5.0 / 3.0)
        pow_0p5_two = 0.25

        pi_two_thirds = jnp.power(PI, 2.0 / 3.0)
        pi_four_thirds = jnp.power(PI, 4.0 / 3.0)
        pi_five_thirds = jnp.power(PI, 5.0 / 3.0)
        pi_two = PI * PI

        pAmp["pnInitial"] = 0.0
        pAmp["pnOneThird"] = 0.0
        pAmp["pnTwoThirds"] = (1.0 - 3.0 * eta) * pi_two_thirds * pow_0p5_two_thirds
        pAmp["pnThreeThirds"] = 0.0
        pAmp["pnFourThirds"] = (
            (-158383.0 + 641105.0 * eta - 446460.0 * eta * eta)
            / 36960.0
            * pi_four_thirds
            * pow_0p5_four_thirds
        )
        pAmp["pnFiveThirds"] = (
            (
                1j * -1008.0
                + 565.0 * chiS
                + 565.0 * chiA * delta
                + 1j * 3579.0 * eta
                - 2075.0 * chiS * eta
                - 1695.0 * chiA * delta * eta
                + 240.0 * PI
                - 720.0 * eta * PI
                + 1j * 960.0 * jnp.log(2.0)
                - 1j * 2880.0 * eta * jnp.log(2.0)
                + 1140.0 * chiS * eta * eta
            )
            / 120.0
            * pi_five_thirds
            * pow_0p5_five_thirds
        )
        pAmp["pnSixThirds"] = (
            (
                7888301437.0
                - 147113366400.0 * chiA * chiS * delta
                - 745140957231.0 * eta
                + 441340099200.0 * chiA * chiS * delta * eta
                - 73556683200.0 * chiA * chiA
                + 511264353600.0 * eta * chiA * chiA
                - 73556683200.0 * chiS * chiS
                + 224302478400.0 * eta * chiS * chiS
                + 2271682065240.0 * eta * eta
                - 871782912000.0 * chiA * chiA * eta * eta
                - 10897286400.0 * chiS * chiS * eta * eta
                - 805075876080.0 * jnp.power(eta, 3)
            )
            / 2.90594304e10
            * pi_two
            * pow_0p5_two
        )

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_GetPNAmplitudeCoefficients: modeTag {modeTag} is not valid. "
            f"Valid modes are 21, 33, 32, 44."
        )
