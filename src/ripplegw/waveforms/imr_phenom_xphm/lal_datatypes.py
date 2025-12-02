""" "LAL datatypes for IMRPhenomXPHM waveform model."""

from __future__ import annotations

import dataclasses

from ripplegw.waveforms.imr_phenom_xphm.dataclass_utils import _register_dataclass


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class LIGOTimeGPS:
    """Dataclass representing LIGO GPS time."""

    gps_seconds: int
    """Seconds since 0h UTC 6 Jan 1980."""
    gps_nano_seconds: int
    """Residual nanoseconds."""


LIGO_TIME_GPS_ZERO = LIGOTimeGPS(gps_seconds=0, gps_nano_seconds=0)
