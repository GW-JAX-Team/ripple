"""Dataclass for storing QNM fit parameters for IMRPhenomXPHM waveform model."""

from __future__ import annotations

import dataclasses

from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.dataclass_utils import _register_dataclass

N_HIGHERMODES_IMPLEMENTED = 4


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class QNMFits:
    """Dataclass for storing QNM fit parameters."""

    f_ring_lm: Array
    """Ringdown frequency for mode (l,m)."""
    f_damp_lm: Array
    """Damping frequency for mode (l,m)."""
