"""Dataclass to hold parameters for IMRPhenomXPHM waveform generation."""

from __future__ import annotations

import dataclasses

from ripplegw.waveforms.imr_phenom_xphm.dataclass_utils import _register_dataclass


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

    # Tidal parameters
    # Enable or disable tidal effects
    phen_x_tidal: int = 0

    # Tidal deformabilities
    # Lambda1 and Lambda2 for the two compact objects
    lambda1: float = 0.0
    lambda2: float = 0.0

    d_quad_mon1: float = 0.0
    d_quad_mon2: float = 0.0
