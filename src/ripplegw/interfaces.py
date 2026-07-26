"""Base classes for ripple waveform models.

Concrete waveform families live in their own modules under ``ripplegw.waveforms``
and register themselves with ``@register`` from ``ripplegw.registry``. Construct
any model through ``ripplegw.waveform(name, **config)``.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ripplegw.typing import FloatLike

StrainDict = dict[str, Float[Array, " n"] | Complex[Array, " n"]]


class Waveform(ABC):
    """Base class for all waveform models.

    A model is configured once at construction (reference frequency, detector,
    and so on) and then called as ``wf(axis, params)``: ``axis`` is the
    evaluation grid, ``params`` maps parameter names to values, and the result
    is a dict keyed by polarization, e.g. ``{"p": ..., "c": ...}``.

    Subclass ``FrequencyDomainWaveform`` or ``TimeDomainWaveform`` rather than
    this class directly, and decorate the model with ``@register`` so users can
    reach it via ``ripplegw.waveform("YourModel")``.

    Attributes:
        waveform_metadata: Descriptive metadata set by ``@register`` (e.g.
            ``is_tidal``); read by ``ripplegw.list_waveforms`` and
            ``ripplegw.get_waveform_metadata``.
    """

    waveform_metadata: ClassVar[dict[str, Any]] = {}

    def __init__(self):
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter names required by this model, in the order ``__call__`` expects them."""
        raise NotImplementedError(
            "Waveform.parameter_names must be implemented by subclasses"
        )

    @abstractmethod
    def __call__(
        self, axis: Float[Array, " n"], /, params: Mapping[str, FloatLike]
    ) -> StrainDict:
        """Evaluate the waveform.

        Args:
            axis: Evaluation grid (a frequency or time array).
            params: Source parameters, keyed by name.

        Returns:
            A dict keyed by polarization/component, e.g. ``{"p": ..., "c": ...}``.
        """
        raise NotImplementedError("Waveform.__call__ must be implemented by subclasses")


class FrequencyDomainWaveform(Waveform):
    """A ``Waveform`` whose ``axis`` is a frequency array."""

    waveform_metadata: ClassVar[dict[str, Any]] = {"domain": "FD"}


class TimeDomainWaveform(Waveform):
    """A ``Waveform`` whose ``axis`` is a time array."""

    waveform_metadata: ClassVar[dict[str, Any]] = {"domain": "TD"}


class AmplitudePhaseWaveform(FrequencyDomainWaveform):
    """A frequency-domain model that reports amplitude and phase separately.

    ``amplitude(f, params) * exp(1j * phase(f, params))`` reproduces the
    pre-polarization strain ``h0`` this model builds internally — see
    ``strain``. ``__call__`` then applies the inclination-dependent plus/cross
    prefactors on top of ``h0``, so ``amplitude``/``phase`` never need to know
    about polarization.
    """

    @abstractmethod
    def amplitude(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> Float[Array, " n_freq"]:
        """Amplitude of ``h0``, as a function of frequency. Includes distance scaling."""
        raise NotImplementedError

    @abstractmethod
    def phase(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> Float[Array, " n_freq"]:
        """Phase of ``h0``, as a function of frequency (the exponent in ``exp(1j * phase)``)."""
        raise NotImplementedError

    def strain(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> Complex[Array, " n_freq"]:
        """Pre-polarization strain: ``amplitude(f, params) * exp(1j * phase(f, params))``.

        Computed as ``amp * cos(phase) + 1j * (amp * sin(phase))``, not via
        ``jnp.exp(1j * phase)`` -- XLA's complex-exponential lowering costs
        roughly 2x the transcendental ops of the equivalent explicit cos/sin
        split (measured), and this identity is exact for real ``phase``.
        """
        amp = self.amplitude(frequency, params)
        phase = self.phase(frequency, params)
        return amp * jnp.cos(phase) + 1j * (amp * jnp.sin(phase))


class DistanceScaledWaveform(Waveform):
    """Mixin for waveforms whose ``params`` includes a distance ``d_L``.

    ``at_unit_distance(axis, params) == __call__(axis, {**params, "d_L": 1.0})``
    exactly, by construction. ``__call__(axis, params) ==
    at_unit_distance(axis, params) / params["d_L"]`` holds only to
    floating-point precision, since this re-evaluates the model rather than
    factoring distance out of a cached result.
    """

    def at_unit_distance(
        self, axis: Float[Array, " n"], params: Mapping[str, FloatLike]
    ) -> StrainDict:
        """Evaluate at ``d_L = 1`` Mpc; any ``d_L`` already in ``params`` is ignored."""
        return self(axis, {**params, "d_L": 1.0})
