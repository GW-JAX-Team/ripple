from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ripplegw")
except PackageNotFoundError:
    __version__ = "unknown"

from ripplegw.interfaces import Waveform, waveform_preset
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

# Stable, family-agnostic public API. Concrete waveform classes are not listed
# statically — they are resolved by name from the registry via ``__getattr__``
# and appended to ``__all__`` below, so no family is hard-coded at the top level.
__all__ = [
    "__version__",
    "Waveform",
    "waveform",
    "list_waveforms",
    "get_waveform_metadata",
    "register",
    "waveform_preset",
    "WAVEFORM_REGISTRY",
]

# Expose registered model names for ``from ripplegw import *`` and ``dir()``.
__all__ += sorted(WAVEFORM_REGISTRY)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name):
    """Resolve registered waveform classes by name (e.g.
    ``from ripplegw import IMRPhenomD``) without hard-coding any family."""
    cls = WAVEFORM_REGISTRY.get(name)
    if cls is not None:
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
