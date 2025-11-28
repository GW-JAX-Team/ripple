"""Dataclass to hold parameters for IMRPhenomXPHM waveform generation."""

from __future__ import annotations

import dataclasses

from ripplegw.waveforms.imr_phenom_xphm.dataclass_utils import _register_dataclass

from ripplegw.typing import Array


@_register_dataclass
@dataclasses.dataclass(frozen=True) 
class IMRPhenomXPHMParameterDataClass:  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold parameters for IMRPhenomXPHM waveform generation."""

    ins_phase_version: int = 104
    int_phase_version: int = 105
    rd_phase_version: int = 105

    ins_amp_version: int = 103
    int_amp_version: int = 104
    rd_amp_version: int = 103

    pnr_use_tuned_coprec: int = 0
    pnr_use_tuned_coprec33: int = 0

    phenom_x_only_return_phase: int = 0
    pnr_force_xhm_alignment: int = 0
    
    mode_array: Array | None = None
    # Tidal parameters
    # Enable or disable tidal effects
    phen_x_tidal: int = 0

    # Tidal deformabilities
    # Lambda1 and Lambda2 for the two compact objects
    lambda1: float = 0.0
    lambda2: float = 0.0

    d_quad_mon1: float = 0.0
    d_quad_mon2: float = 0.0

    # Precession parameters
    precession_version: int = 300
    expansion_order: int = 5
    pnr_use_tuned_angles: int = 0
    pnr_interp_tolerance: float = 0.01
    antisymmetric_waveform: int = 0
    mband_version: int = 0


###### These are here to easily find some of the standard parameter values used in LAL ######

# /* IMRPhenomX Parameters */
# DEFINE_INSERT_FUNC(PhenomXInspiralPhaseVersion, INT4, "InsPhaseVersion", 104)
# DEFINE_INSERT_FUNC(PhenomXInspiralAmpVersion, INT4, "InsAmpVersion", 103)
# DEFINE_INSERT_FUNC(PhenomXIntermediatePhaseVersion, INT4, "IntPhaseVersion", 105)
# DEFINE_INSERT_FUNC(PhenomXIntermediateAmpVersion, INT4, "IntAmpVersion", 104)
# DEFINE_INSERT_FUNC(PhenomXRingdownPhaseVersion, INT4, "RDPhaseVersion", 105)
# DEFINE_INSERT_FUNC(PhenomXRingdownAmpVersion, INT4, "RDAmpVersion", 103)
# DEFINE_INSERT_FUNC(PhenomXPrecVersion, INT4, "PrecVersion", 300)
# DEFINE_INSERT_FUNC(PhenomXReturnCoPrec, INT4, "ReturnCoPrec", 0)
# DEFINE_INSERT_FUNC(PhenomXPExpansionOrder, INT4, "ExpansionOrder", 5)
# DEFINE_INSERT_FUNC(PhenomXPConvention, INT4, "Convention", 1)
# DEFINE_INSERT_FUNC(PhenomXPFinalSpinMod, INT4, "FinalSpinMod", 4)
# DEFINE_INSERT_FUNC(PhenomXPTransPrecessionMethod, INT4, "TransPrecessionMethod", 1)
# DEFINE_INSERT_FUNC(PhenomXPSpinTaylorVersion, String, "SpinTaylorVersion", NULL)
# DEFINE_INSERT_FUNC(PhenomXPSpinTaylorCoarseFactor, INT4, "SpinTaylorCoarseFactor",10);

# /* IMRPhenomXAS_NRTidalvX */
# DEFINE_INSERT_FUNC(PhenomXTidalFlag, INT4, "PhenXTidal", 0)

# /* IMRPhenomXHM Parameters */
# DEFINE_INSERT_FUNC(PhenomXHMReleaseVersion, INT4, "PhenomXHMReleaseVersion", 122022)
# DEFINE_INSERT_FUNC(PhenomXHMInspiralPhaseVersion, INT4, "InsPhaseHMVersion", 122019)
# DEFINE_INSERT_FUNC(PhenomXHMIntermediatePhaseVersion, INT4, "IntPhaseHMVersion", 122019)
# DEFINE_INSERT_FUNC(PhenomXHMRingdownPhaseVersion, INT4, "RDPhaseHMVersion", 122019)
# DEFINE_INSERT_FUNC(PhenomXHMInspiralAmpVersion, INT4, "InsAmpHMVersion", 3)
# DEFINE_INSERT_FUNC(PhenomXHMIntermediateAmpVersion, INT4, "IntAmpHMVersion", 2)
# DEFINE_INSERT_FUNC(PhenomXHMRingdownAmpVersion, INT4, "RDAmpHMVersion", 0)
# DEFINE_INSERT_FUNC(PhenomXHMInspiralAmpFitsVersion, INT4, "InsAmpFitsVersion", 122018)
# DEFINE_INSERT_FUNC(PhenomXHMIntermediateAmpFitsVersion, INT4, "IntAmpFitsVersion", 122018)
# DEFINE_INSERT_FUNC(PhenomXHMRingdownAmpFitsVersion, INT4, "RDAmpFitsVersion", 122018)
# DEFINE_INSERT_FUNC(PhenomXHMInspiralAmpFreqsVersion, INT4, "InsAmpFreqsVersion", 122018)
# DEFINE_INSERT_FUNC(PhenomXHMIntermediateAmpFreqsVersion, INT4, "IntAmpFreqsVersion", 122018)
# DEFINE_INSERT_FUNC(PhenomXHMRingdownAmpFreqsVersion, INT4, "RDAmpFreqsVersion", 122018)
# DEFINE_INSERT_FUNC(PhenomXHMPhaseRef21, REAL8, "PhaseRef21", 0.)
# DEFINE_INSERT_FUNC(PhenomXHMThresholdMband, REAL8, "ThresholdMband", 0.001)
# DEFINE_INSERT_FUNC(PhenomXHMAmpInterpolMB, INT4, "AmpInterpol", 1)

# /* IMRPhenomXPHM Parameters */
# DEFINE_INSERT_FUNC(PhenomXPHMMBandVersion, INT4, "MBandPrecVersion", 0)
# DEFINE_INSERT_FUNC(PhenomXPHMThresholdMband, REAL8, "PrecThresholdMband", 0.001)
# DEFINE_INSERT_FUNC(PhenomXPHMUseModes, INT4, "UseModes", 0)
# DEFINE_INSERT_FUNC(PhenomXPHMModesL0Frame, INT4, "ModesL0Frame", 0)
# DEFINE_INSERT_FUNC(PhenomXPHMPrecModes, INT4, "PrecModes", 0)
# DEFINE_INSERT_FUNC(PhenomXPHMTwistPhenomHM, INT4, "TwistPhenomHM", 0)

# /* IMRPhenomX_PNR Parameters */
# DEFINE_INSERT_FUNC(PhenomXPNRUseTunedAngles, INT4, "PNRUseTunedAngles", 0)
# DEFINE_INSERT_FUNC(PhenomXPNRUseTunedCoprec, INT4, "PNRUseTunedCoprec", 0)
# DEFINE_INSERT_FUNC(PhenomXPNRUseTunedCoprec33, INT4, "PNRUseTunedCoprec33", 0)
# // Option to only be used when actively tuning PNR Coprec relative to XHM wherein the non-precessing final spin is used
# DEFINE_INSERT_FUNC(PhenomXPNRUseInputCoprecDeviations, INT4, "PNRUseInputCoprecDeviations", 0)
# // Dev option for forcing 22 phase derivative inspiral values to align with XHM at a low ref frequency
# DEFINE_INSERT_FUNC(PhenomXPNRForceXHMAlignment, INT4, "PNRForceXHMAlignment", 0)
# /* Toggle output of XAS phase for debugging purposes */
# DEFINE_INSERT_FUNC(PhenomXOnlyReturnPhase, INT4, "PhenomXOnlyReturnPhase", 0)
# DEFINE_INSERT_FUNC(PhenomXPNRInterpTolerance, REAL8, "PNRInterpTolerance", 0.01)
# /* IMRPhenomX_PNR_Asymmetry Parameters */
# DEFINE_INSERT_FUNC(PhenomXAntisymmetricWaveform, INT4, "AntisymmetricWaveform", 0)