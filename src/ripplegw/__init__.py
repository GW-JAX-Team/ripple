from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ripplegw")
except PackageNotFoundError:
    __version__ = "unknown"

from ripplegw.interfaces import (
    Waveform,
    TaylorF2,
    IMRPhenomD,
    IMRPhenomD_NRTidalv2,
    IMRPhenomHM,
    IMRPhenomPv2,
    IMRPhenomXAS,
    IMRPhenomXAS_NRTidalv3,
    IMRPhenomXHM,
    IMRPhenomXP,
    IMRPhenomXPHM,
    SineGaussian,
    waveform_preset,
)
from ripplegw.registry import (
    WAVEFORM_REGISTRY,
    list_waveforms,
    register,
    waveform,
)

__all__ = [
    "__version__",
    "Waveform",
    "TaylorF2",
    "IMRPhenomD",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomHM",
    "IMRPhenomPv2",
    "IMRPhenomXAS",
    "IMRPhenomXAS_NRTidalv3",
    "IMRPhenomXHM",
    "IMRPhenomXP",
    "IMRPhenomXPHM",
    "SineGaussian",
    "waveform_preset",
    # top-level registry API
    "WAVEFORM_REGISTRY",
    "waveform",
    "list_waveforms",
    "register",
]
