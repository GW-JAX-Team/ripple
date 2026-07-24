"""Built-in waveform families.

Every non-private submodule here is imported automatically at package import so
that any :class:`~ripplegw.interfaces.Waveform` subclass decorated with
``@register`` self-registers into the top-level registry. Adding a new in-tree
family therefore needs only a new ``ripplegw/waveforms/<Family>.py`` module —
no edits to this file, to ``ripplegw/__init__.py``, or to any central list.
"""

import importlib
import pkgutil

for _module_info in pkgutil.iter_modules(__path__):
    if not _module_info.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module_info.name}")

del importlib, pkgutil
