from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ripplegw")
except PackageNotFoundError:
    __version__ = "unknown"

from .interfaces import (
    TaylorF2,
    IMRPhenomD,
    IMRPhenomD_NRTidalv2,
    IMRPhenomPv2,
    IMRPhenomXAS,
    IMRPhenomXAS_NRTidalv3,
    IMRPhenomXHM,
    IMRPhenomXP,
    IMRPhenomXPHM,
    IMRPhenomHM,
    SineGaussian,
    waveform_preset,
)

__all__ = [
    "__version__",
    "TaylorF2",
    "IMRPhenomD",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomPv2",
    "IMRPhenomXAS",
    "IMRPhenomXAS_NRTidalv3",
    "IMRPhenomXHM",
    "IMRPhenomXP",
    "IMRPhenomXPHM",
    "IMRPhenomHM",
    "SineGaussian",
    "waveform_preset",
]
