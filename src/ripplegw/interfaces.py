"""Abstract base class for ripple waveform models.

The concrete waveform families live in their own modules under
``ripplegw.waveforms`` (and other subpackages) and register themselves via
``@register`` from :mod:`ripplegw.registry`. Construct any model through
:func:`ripplegw.waveform`.
"""

from abc import ABC, abstractmethod

from jaxtyping import Array, Complex, Float

from ripplegw.registry import WAVEFORM_REGISTRY


class Waveform(ABC):
    """Abstract base class for gravitational waveform models.

    Subclasses implement the frequency- (or time-) domain waveform and expose it
    via ``__call__``, returning a dictionary keyed by polarization. The compact-
    binary and burst models here use ``"p"`` (plus) and ``"c"`` (cross), but the
    contract is deliberately generic: a model may return any set of polarization
    keys (e.g. additional non-tensor modes) as long as ``__call__`` takes an
    ``axis`` (frequency *or* time grid) and a ``params`` dict.

    Metadata convention:
        Concrete models declare descriptive class attributes so they can be
        discovered and filtered generically via
        :func:`ripplegw.list_waveforms`. ``domain`` (``"FD"`` / ``"TD"``) is
        universal; families may attach any further tags (e.g. ``is_tidal``,
        ``is_precessing`` for CBC) — these are optional and not assumed by the
        top-level API. The :func:`ripplegw.register` decorator sets them.
    """

    #: ``"FD"`` (frequency domain) or ``"TD"`` (time domain); ``None`` if unset.
    #: Universal metadata read by :func:`ripplegw.list_waveforms`.
    domain: str | None = None

    def __init__(self):
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Ordered tuple of parameter names required by this waveform model.

        Returns:
            tuple[str, ...]: Parameter names in the order they are consumed,
                matching the keys expected in the ``params`` dict passed to
                ``__call__``.
        """
        raise NotImplementedError(
            "Waveform.parameter_names must be implemented by subclasses"
        )

    @abstractmethod
    def __call__(
        self, axis: Float[Array, " n"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n"] | Complex[Array, " n"]]:
        """Evaluate the waveform.

        Args:
            axis (Float[Array, " n"]): Frequency or time grid.
            params (dict[str, Float]): Source parameter dictionary.

        Returns:
            dict[str, Float[Array, " n"] | Complex[Array, " n"]]: Dictionary
                with keys ``"p"`` (plus polarization) and ``"c"`` (cross
                polarization). Frequency-domain waveforms return complex arrays;
                time-domain waveforms return real arrays.
        """
        raise NotImplementedError("Waveform.__call__ must be implemented by subclasses")


#: Mapping from model name strings to ``Waveform`` subclasses.
#:
#: Alias of :data:`ripplegw.registry.WAVEFORM_REGISTRY`, populated by every
#: model via ``@register`` at import. Retained for backward compatibility;
#: prefer :func:`ripplegw.waveform` / :func:`ripplegw.list_waveforms`.
waveform_preset: dict[str, type[Waveform]] = WAVEFORM_REGISTRY
