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
    load_plugins,
    register,
    waveform,
)

# Importing the waveforms package auto-imports every in-tree family module,
# each of which self-registers via @register. Then discover any externally
# installed families exposed through the "ripplegw.waveforms" entry-point group.
# Adding a family (in-tree module or plugin package) needs no edit here.
from ripplegw import waveforms as _waveforms  # noqa: F401

load_plugins()

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
