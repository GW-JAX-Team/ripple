"""Abstract base class for ripple waveform models.

The concrete waveform families live in their own modules under
``ripplegw.waveforms`` (and other subpackages) and register themselves via
``@register`` from :mod:`ripplegw.registry`. Construct any model through
:func:`ripplegw.waveform`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping

from ripplegw.registry import WAVEFORM_REGISTRY


class Waveform(ABC):
    """Abstract base class for gravitational waveform models.

    The evaluation contract is deliberately generic so the top-level API can
    host *any* waveform family, not just compact-binary coalescences:

    * ``__call__(axis, params)`` takes an evaluation ``axis`` — a frequency or
      time grid, or any array/structure a model needs — and a ``params`` mapping
      whose values may be scalars, arrays, or structured objects.
    * It returns a ``dict`` keyed by model-defined component/polarization labels.
      The CBC and burst models here use ``"p"`` (plus) and ``"c"`` (cross), but
      that is only their convention; a model may return any keys (e.g. extra
      non-tensor polarizations or spin-weighted modes).

    Anything a model needs beyond the per-evaluation ``params`` — a reference
    frequency, a detector, an ephemeris/data handle, a frequency grid — is fixed
    at construction (``__init__``), keeping ``__call__`` a pure function of
    ``(axis, params)`` that is friendly to ``jit``/``vmap``/``grad``.

    Metadata convention:
        Concrete models attach descriptive metadata (e.g. ``domain="FD"``, and
        family-specific tags like ``is_tidal`` / ``is_precessing`` for CBC) via
        the :func:`ripplegw.register` decorator. It is stored in the
        :attr:`waveform_metadata` dict (never written over real attributes) and
        drives generic discovery/filtering through
        :func:`ripplegw.list_waveforms`. All metadata keys are optional; the
        top-level API assumes none of them.
    """

    #: Per-class descriptive metadata populated by :func:`ripplegw.register`.
    #: Read by :func:`ripplegw.list_waveforms` and
    #: :func:`ripplegw.get_waveform_metadata`.
    waveform_metadata: ClassVar[dict[str, Any]] = {}

    def __init__(self):
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Ordered tuple of parameter names required by this waveform model.

        Returns:
            tuple[str, ...]: Parameter names in the order they are consumed,
                matching the keys expected in the ``params`` mapping passed to
                ``__call__``.
        """
        raise NotImplementedError(
            "Waveform.parameter_names must be implemented by subclasses"
        )

    @abstractmethod
    def __call__(self, axis: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate the waveform.

        Args:
            axis: Evaluation grid or points — a frequency array, a time array,
                or any structure the model consumes.
            params (Mapping[str, Any]): Source parameters. Values may be
                scalars, arrays, or structured objects.

        Returns:
            dict[str, Any]: Output keyed by model-defined component/polarization
                labels. The CBC/burst models use ``"p"`` (plus) and ``"c"``
                (cross) — frequency-domain models return complex arrays,
                time-domain models return real arrays — but the set of keys is
                model-specific.
        """
        raise NotImplementedError("Waveform.__call__ must be implemented by subclasses")


#: Mapping from model name strings to ``Waveform`` subclasses.
#:
#: Alias of :data:`ripplegw.registry.WAVEFORM_REGISTRY`, populated by every
#: model via ``@register`` at import. Retained for backward compatibility;
#: prefer :func:`ripplegw.waveform` / :func:`ripplegw.list_waveforms`.
waveform_preset: dict[str, type[Waveform]] = WAVEFORM_REGISTRY


def __getattr__(name: str) -> Any:
    """Backward-compatible access to concrete model classes.

    The waveform classes moved to their family modules under
    ``ripplegw.waveforms``, but legacy code may still do
    ``from ripplegw.interfaces import IMRPhenomD``. Resolve such names lazily
    from the registry (populated when the ``ripplegw`` package is imported),
    which keeps back-compat without re-coupling this module to any family.
    """
    cls = WAVEFORM_REGISTRY.get(name)
    if cls is not None:
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
