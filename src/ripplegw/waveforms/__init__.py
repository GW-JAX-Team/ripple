"""Built-in waveform families.

Importing this package imports every family module below it (recursing into
subpackages), so each ``Waveform`` subclass decorated with ``@register``
self-registers into the top-level registry. Adding a new in-tree family means
adding a module or a subpackage here — never editing this file.
"""

import importlib
import pkgutil


def _reraise(name: str) -> None:
    """pkgutil.walk_packages silently drops a subpackage on import failure
    unless given an onerror handler; re-raise so a broken family is loud."""
    raise


for _info in pkgutil.walk_packages(__path__, f"{__name__}.", onerror=_reraise):
    if not any(part.startswith("_") for part in _info.name.split(".")):
        importlib.import_module(_info.name)

del importlib, pkgutil
