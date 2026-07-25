from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ripplegw")
except PackageNotFoundError:
    __version__ = "unknown"

from ripplegw.interfaces import (
    AmplitudePhaseWaveform,
    DistanceScaledWaveform,
    FrequencyDomainWaveform,
    TimeDomainWaveform,
    Waveform,
)
from ripplegw.registry import (
    WAVEFORM_REGISTRY,
    get_waveform_metadata,
    list_waveforms,
    register,
    waveform,
)

# Importing the waveforms package auto-imports every in-tree family module,
# each of which self-registers via @register. Adding a family needs no edit
# here — just a new self-registering module under ripplegw.waveforms.
from ripplegw import waveforms as _waveforms  # noqa: F401

# The entire public API. Concrete waveform classes (IMRPhenomD, TaylorF2, ...)
# are intentionally NOT exposed here — the single entry point is
# ripplegw.waveform(name, **config); use ripplegw.list_waveforms() to discover
# names.
__all__ = [
    "__version__",
    "Waveform",
    "FrequencyDomainWaveform",
    "TimeDomainWaveform",
    "AmplitudePhaseWaveform",
    "DistanceScaledWaveform",
    "waveform",
    "list_waveforms",
    "get_waveform_metadata",
    "register",
    "WAVEFORM_REGISTRY",
]
