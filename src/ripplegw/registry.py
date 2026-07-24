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

import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ripplegw.interfaces import Waveform

__all__ = [
    "WAVEFORM_REGISTRY",
    "register",
    "waveform",
    "list_waveforms",
    "get_waveform_metadata",
    "load_plugins",
]

#: Global name -> ``Waveform`` subclass registry. Populated once at import via
#: :func:`register` (in-tree families) and :func:`load_plugins` (external ones).
#: Treat as read-only after import; never mutate inside a JAX-transformed path.
WAVEFORM_REGISTRY: dict[str, type["Waveform"]] = {}


def register(name: str | None = None, *, override: bool = False, **metadata):
    """Class decorator registering a ``Waveform`` subclass under ``name``.

    Any keyword ``metadata`` (e.g. ``domain="FD"``, ``is_tidal=False``,
    ``is_precessing=False``) is recorded in the class's ``waveform_metadata``
    dict — a single declarative source of truth for :func:`list_waveforms`
    filtering and :func:`get_waveform_metadata`. Metadata is stored in this
    dedicated container rather than as free attributes, so it can never shadow
    real API members (``parameter_names``, ``__call__``, future base fields).
    Each class gets its own dict, inheriting any metadata declared on a base.

    Args:
        name (str | None): Registry key. Defaults to the class ``__name__``.
        override (bool): If False (default), re-registering an existing name
            raises; pass True to replace intentionally.
        **metadata: Descriptive tags (``domain``, ``is_tidal``, ...). Arbitrary
            keys are allowed so new families can add their own.

    Returns:
        The decorator, which returns the class unchanged.
    """

    def deco(cls):
        _check_is_waveform(cls)
        key = name or cls.__name__
        if key in WAVEFORM_REGISTRY and not override:
            raise ValueError(
                f"Waveform {key!r} is already registered to "
                f"{WAVEFORM_REGISTRY[key]!r}; pass override=True to replace it."
            )
        inherited = dict(getattr(cls, "waveform_metadata", {}) or {})
        inherited.update(metadata)
        cls.waveform_metadata = inherited
        WAVEFORM_REGISTRY[key] = cls
        return cls

    return deco


def _check_is_waveform(cls) -> None:
    """Validate that ``cls`` is a ``Waveform`` subclass (imported lazily to
    avoid an import cycle with :mod:`ripplegw.interfaces`)."""
    from ripplegw.interfaces import Waveform

    if not (isinstance(cls, type) and issubclass(cls, Waveform)):
        raise TypeError(
            f"Cannot register {cls!r}: waveforms must be subclasses of "
            f"ripplegw.interfaces.Waveform."
        )


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
    """List registered waveform names, optionally filtered by metadata.

    Args:
        **filters: Constraints on ``waveform_metadata``, e.g. ``domain="FD"`` or
            ``is_precessing=True``. A model matches only if its metadata defines
            every requested key with the requested value. Unknown/typo'd keys
            simply match nothing; use :func:`get_waveform_metadata` to inspect
            available keys.

    Returns:
        list[str]: Sorted matching names.
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
        name (str): A registered model name.

    Returns:
        dict[str, Any]: The model's metadata (e.g. ``{"domain": "FD", ...}``).

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
    group contributes ``name -> Waveform subclass`` mappings (the entry-point
    value must load to a :class:`~ripplegw.interfaces.Waveform` subclass), so a
    ``pip install`` makes the model available through :func:`waveform` with no
    edits to ripple. In-tree registrations take precedence over plugins of the
    same name.

    A misbehaving plugin (fails to load, or is not a ``Waveform`` subclass) is
    skipped with a warning rather than aborting discovery, so one bad plugin
    cannot hide the others. Discovery is marked complete only after a full pass.
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
