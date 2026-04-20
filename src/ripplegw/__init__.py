from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ripplegw")
except PackageNotFoundError:
    __version__ = "unknown"

from .interfaces import (
    IMRPhenomD,
    IMRPhenomPv2,
    TaylorF2,
    IMRPhenomD_NRTidalv2,
    IMRPhenomXAS,
    IMRPhenomXAS_NRTidalv3,
    IMRPhenomXPHM,
    IMRPhenomXHM,
    SineGaussian,
    waveform_preset,
)

__all__ = [
    "__version__",
    "IMRPhenomD",
    "IMRPhenomPv2",
    "TaylorF2",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomXAS",
    "IMRPhenomXAS_NRTidalv3",
    "IMRPhenomXPHM",
    "IMRPhenomXHM",
    "SineGaussian",
    "waveform_preset",
]
