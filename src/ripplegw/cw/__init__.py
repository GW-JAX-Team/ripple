"""Continuous-wave (pulsar) signal generation in JAX.

This subpackage ports LALPulsar's pulsar signal generators to differentiable
JAX, compatible with ripple's ``{"p", "c"}`` polarization convention.

Currently implemented:

* :class:`ExactPulsarSignal` / :func:`exact_pulsar_polarizations` — the exact
  (geometric) model of ``XLALSimulateExactPulsarSignal`` for an isolated
  pulsar, returning plus/cross polarizations.

Supporting modules: :mod:`ephemeris` (JPL ephemeris reader), :mod:`barycenter`
(detector SSB position + Roemer delay), :mod:`time_utils` (leap seconds,
sidereal time) and :mod:`detectors` (detector locations).
"""

from .detectors import Detector, DETECTORS, get_detector
from .ephemeris import Ephemeris, load_ephemeris, read_ephemeris_file
from .pulsar_signal import (
    exact_pulsar_polarizations,
    generate_binary_pulsar_polarizations,
    generate_pulsar_polarizations,
)
from .waveform import BinaryPulsarSignal, ExactPulsarSignal, PulsarSignal

__all__ = [
    "Detector",
    "DETECTORS",
    "get_detector",
    "Ephemeris",
    "load_ephemeris",
    "read_ephemeris_file",
    "exact_pulsar_polarizations",
    "generate_pulsar_polarizations",
    "generate_binary_pulsar_polarizations",
    "ExactPulsarSignal",
    "PulsarSignal",
    "BinaryPulsarSignal",
]
