"""Top-level waveform registry.

A single, family-agnostic entry point for constructing waveforms by name. Each
``Waveform`` subclass registers itself with ``register`` — in-tree families are
imported for their registration side-effect; external packages self-register
via the ``"ripplegw.waveforms"`` entry-point group. Users then construct any
model through ``waveform`` without importing the implementing module:

    import ripplegw
    wf = ripplegw.waveform("IMRPhenomXAS", f_ref=20.0)
    ripplegw.list_waveforms(domain="FD")

Adding a family means adding a self-registering module (or installing a plugin
package) — never editing this file.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ripplegw.interfaces import Waveform

__all__ = [
    "WAVEFORM_REGISTRY",
    "get_waveform_metadata",
    "list_waveforms",
    "load_plugins",
    "register",
    "waveform",
]

WAVEFORM_REGISTRY: dict[str, type["Waveform"]] = {}
"""Global name -> ``Waveform`` subclass registry, populated at import by
``register`` (in-tree families) and ``load_plugins`` (external ones)."""


def register(name: Optional[str] = None, *, override: bool = False, **metadata):
    """Class decorator that adds a ``Waveform`` subclass to the registry.

    Keyword arguments are stored on the class as ``waveform_metadata``; that is
    what ``list_waveforms`` filters on.

    Args:
        name: Registry key users pass to ``waveform()``. Defaults to the class name.
        override: Allow replacing a name that is already registered.
        **metadata: Descriptive tags, e.g. ``is_tidal=True``.

    Returns:
        The decorator, which returns the class unchanged.

    Raises:
        ValueError: If ``name`` is already registered and ``override`` is False.
        TypeError: If the decorated class is not a ``Waveform`` subclass.
    """

    def deco(cls):
        _check_is_waveform(cls)
        key = name or cls.__name__
        if key in WAVEFORM_REGISTRY and not override:
            raise ValueError(
                f"Waveform {key!r} is already registered to "
                f"{WAVEFORM_REGISTRY[key]!r}; pass override=True to replace it."
            )
        # A dedicated dict (not plain attributes) so metadata can't shadow real
        # API members. Each class gets its own copy, inheriting the base's.
        inherited = dict(getattr(cls, "waveform_metadata", {}) or {})
        inherited.update(metadata)
        cls.waveform_metadata = inherited
        WAVEFORM_REGISTRY[key] = cls
        return cls

    return deco


def _check_is_waveform(cls) -> None:
    """Raise ``TypeError`` unless ``cls`` is a ``Waveform`` subclass.

    Imports ``Waveform`` lazily to avoid a cycle with ``ripplegw.interfaces``.
    """
    from ripplegw.interfaces import Waveform

    if not (isinstance(cls, type) and issubclass(cls, Waveform)):
        raise TypeError(
            f"Cannot register {cls!r}: waveforms must be subclasses of "
            f"ripplegw.interfaces.Waveform."
        )


def waveform(name: str, /, **config) -> "Waveform":
    """Construct a registered waveform by name.

    Args:
        name: A registered model name (see ``list_waveforms``).
        **config: Constructor configuration forwarded to the model
            (e.g. ``f_ref=20.0`` for CBC models).

    Returns:
        A configured instance, callable as ``wf(axis, params)``.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    load_plugins()
    try:
        cls = WAVEFORM_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown waveform {name!r}. Available: {sorted(WAVEFORM_REGISTRY)}"
        ) from None
    return cls(**config)


def list_waveforms(**filters) -> list[str]:
    """List registered waveform names, optionally filtered by metadata.

    Args:
        **filters: Constraints on ``waveform_metadata``, e.g. ``domain="FD"`` or
            ``is_precessing=True``. A model matches only if its metadata defines
            every requested key with the requested value. Unknown/typo'd keys
            simply match nothing; use ``get_waveform_metadata`` to inspect
            available keys.

    Returns:
        Sorted matching names.
    """
    load_plugins()
    return sorted(
        nm
        for nm, cls in WAVEFORM_REGISTRY.items()
        if all(
            getattr(cls, "waveform_metadata", {}).get(key) == val
            for key, val in filters.items()
        )
    )


def get_waveform_metadata(name: str) -> dict[str, Any]:
    """Return a copy of the descriptive metadata for a registered waveform.

    Args:
        name: A registered model name.

    Returns:
        The model's metadata, e.g. ``{"domain": "FD", ...}``.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    load_plugins()
    try:
        cls = WAVEFORM_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown waveform {name!r}. Available: {sorted(WAVEFORM_REGISTRY)}"
        ) from None
    return dict(getattr(cls, "waveform_metadata", {}))


_PLUGINS_LOADED = False


def load_plugins() -> None:
    """Discover external waveform families via entry points (idempotent).

    Any installed package exposing the ``"ripplegw.waveforms"`` entry-point
    group contributes ``name -> Waveform subclass`` mappings, so a ``pip
    install`` makes the model available through ``waveform`` with no edits to
    ripple. In-tree registrations take precedence over plugins of the same
    name. A misbehaving plugin (fails to load, or is not a ``Waveform``
    subclass) is skipped with a warning rather than aborting discovery.
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    for ep in entry_points(group="ripplegw.waveforms"):
        if ep.name in WAVEFORM_REGISTRY:
            continue  # in-tree registration wins over a same-named plugin
        try:
            cls = ep.load()
            _check_is_waveform(cls)
        except Exception as exc:  # noqa: BLE001 - isolate one bad plugin
            warnings.warn(
                f"Skipping waveform plugin {ep.name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        WAVEFORM_REGISTRY[ep.name] = cls
    _PLUGINS_LOADED = True
