"""Top-level waveform registry.

A single, family-agnostic entry point for constructing waveforms by name. Each
:class:`~ripplegw.interfaces.Waveform` subclass registers itself with
:func:`register` (in-tree families are imported for their registration
side-effect; external packages self-register via the ``"ripplegw.waveforms"``
entry-point group). Users then construct any model through :func:`waveform`
without importing the implementing module:

    >>> import ripplegw
    >>> wf = ripplegw.waveform("IMRPhenomXAS", f_ref=20.0)
    >>> ripplegw.list_waveforms(domain="FD")          # doctest: +SKIP

This keeps the top level decoupled from any specific waveform family: adding a
family means adding a self-registering module (or installing a plugin package),
never editing this file.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ripplegw.interfaces import Waveform

__all__ = [
    "WAVEFORM_REGISTRY",
    "register",
    "waveform",
    "list_waveforms",
    "load_plugins",
]

#: Global name -> ``Waveform`` subclass registry. Populated once at import via
#: :func:`register` (in-tree families) and :func:`load_plugins` (external ones).
#: Treat as read-only after import; never mutate inside a JAX-transformed path.
WAVEFORM_REGISTRY: dict[str, type["Waveform"]] = {}


def register(name: str | None = None, *, override: bool = False, **metadata):
    """Class decorator registering a ``Waveform`` subclass under ``name``.

    Any keyword ``metadata`` (e.g. ``domain="FD"``, ``is_tidal=False``,
    ``is_precessing=False``) is attached to the class as attributes, so family
    properties live declaratively at the definition site — a single source of
    truth for both :func:`list_waveforms` filtering and the test suite.

    Args:
        name (str | None): Registry key. Defaults to the class ``__name__``.
        override (bool): If False (default), re-registering an existing name
            raises; pass True to replace intentionally.
        **metadata: Class attributes to set (``domain``, ``is_tidal``, ...).

    Returns:
        The decorator, which returns the class unchanged (besides metadata).
    """

    def deco(cls):
        key = name or cls.__name__
        if key in WAVEFORM_REGISTRY and not override:
            raise ValueError(
                f"Waveform {key!r} is already registered to "
                f"{WAVEFORM_REGISTRY[key]!r}; pass override=True to replace it."
            )
        for attr, value in metadata.items():
            setattr(cls, attr, value)
        WAVEFORM_REGISTRY[key] = cls
        return cls

    return deco


def waveform(name: str, /, **config) -> "Waveform":
    """Construct a registered waveform by name.

    Args:
        name (str): A registered model name (see :func:`list_waveforms`).
        **config: Constructor configuration forwarded to the model
            (e.g. ``f_ref=20.0`` for CBC models).

    Returns:
        Waveform: A configured instance, callable as ``wf(axis, params)``.

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
    """List registered waveform names, optionally filtered by class metadata.

    Args:
        **filters: Metadata attribute constraints, e.g. ``domain="FD"`` or
            ``is_precessing=True``. A model matches only if it defines every
            requested attribute with the requested value.

    Returns:
        list[str]: Sorted matching names.
    """
    load_plugins()
    return sorted(
        nm
        for nm, cls in WAVEFORM_REGISTRY.items()
        if all(getattr(cls, key, None) == val for key, val in filters.items())
    )


_PLUGINS_LOADED = False


def load_plugins() -> None:
    """Discover external waveform families via entry points (idempotent).

    Any installed package exposing the ``"ripplegw.waveforms"`` entry-point
    group contributes ``name -> Waveform subclass`` mappings, so a
    ``pip install`` makes the model available through :func:`waveform` with no
    edits to ripple. In-tree registrations take precedence over plugins of the
    same name.
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    for ep in entry_points(group="ripplegw.waveforms"):
        WAVEFORM_REGISTRY.setdefault(ep.name, ep.load())
