"""Dataclass to hold internal parameters for IMRPhenomX waveform generation."""

from __future__ import annotations

import dataclasses

from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.dataclass_utils import _register_dataclass


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class PhenomXPInspiralArrays:
    """Dataclass to hold inspiral arrays for IMRPhenomX waveform generation."""

    v_pn: Array
    """Velocity parameter array."""
    s1x_pn: Array
    """Spin component 1 x array."""
    s1y_pn: Array
    """Spin component 1 y array."""
    s1z_pn: Array
    """Spin component 1 z array."""
    s2x_pn: Array
    """Spin component 2 x array."""
    s2y_pn: Array
    """Spin component 2 y array."""
    s2z_pn: Array
    """Spin component 2 z array."""
    ln_hat_x_pn: Array
    """Logarithm of unit vector x component array."""
    ln_hat_y_pn: Array
    """Logarithm of unit vector y component array."""
    ln_hat_z_pn: Array
    """Logarithm of unit vector z component array."""
