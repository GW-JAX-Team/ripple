from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ripplegw")
except PackageNotFoundError:
    __version__ = "unknown"

from ripplegw.interfaces import Waveform, waveform_preset

# Importing each family module registers its Waveform subclass in the global
# registry (via @register) and re-exports the class at the top level for
# backward compatibility.
from ripplegw.waveforms.TaylorF2 import TaylorF2
from ripplegw.waveforms.IMRPhenomD import IMRPhenomD
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import IMRPhenomD_NRTidalv2
from ripplegw.waveforms.IMRPhenomHM import IMRPhenomHM
from ripplegw.waveforms.IMRPhenomPv2 import IMRPhenomPv2
from ripplegw.waveforms.IMRPhenomXAS import IMRPhenomXAS
from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import IMRPhenomXAS_NRTidalv3
from ripplegw.waveforms.IMRPhenomXHM import IMRPhenomXHM
from ripplegw.waveforms.IMRPhenomXP import IMRPhenomXP
from ripplegw.waveforms.IMRPhenomXPHM import IMRPhenomXPHM
from ripplegw.waveforms.SineGaussian import SineGaussian
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
