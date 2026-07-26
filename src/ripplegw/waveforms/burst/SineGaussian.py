from collections.abc import Mapping

import jax.numpy as jnp
from jax.lax import complex
from jaxtyping import Array, Float

from ripplegw.constants import PI
from ripplegw.interfaces import TimeDomainWaveform
from ripplegw.registry import register
from ripplegw.typing import FloatLike


def semi_major_minor_from_e(e: FloatLike) -> tuple[FloatLike, FloatLike]:
    """
    Calculate the semi-major and semi-minor axes of an ellipse given the
    eccentricity of the ellipse.

    Args:
        e: Eccentricity of the ellipse
    Returns:
        Semi-major (a) and semi-minor (b) axes of the ellipse
    """
    a = 1.0 / jnp.sqrt(2.0 - (e * e))
    b = a * jnp.sqrt(1.0 - (e * e))
    return a, b


def gen_SineGaussian_hphc(
    t: Float[Array, " n_time"],
    theta: Float[Array, "5"],
) -> tuple[Float[Array, " n_time"], Float[Array, " n_time"]]:
    """
    Generate lalinference implementation of a sine-Gaussian waveform in JAX.
    See
    git.ligo.org/lscsoft/lalsuite/-/blob/master/lalinference/lib/LALInferenceBurstRoutines.c#L381
    for details on parameter definitions.

    Args:
        t (Array): Time grid (centered at t=0) on which to evaluate the waveform.
            Create it using ``jax.numpy.arange(-duration/2, duration/2, 1/fs)``
            where ``duration`` is the duration of the waveform (in seconds) and ``fs``
            is the sample rate at which the waveform is evaluated.
        theta (Array): Array of waveform parameters [quality, frequency, hrss, phase, eccentricity]:

            - quality: Quality factor of the sine-Gaussian waveform.
            - frequency: Central frequency of the sine-Gaussian waveform.
            - hrss: Hrss of the sine-Gaussian waveform.
            - phase: Phase of the sine-Gaussian waveform.
            - eccentricity: Eccentricity of the sine-Gaussian waveform.
              Controls the relative amplitudes of the hplus and hcross polarizations.

    Returns:
        tuple[Array, Array]: JAX Arrays of plus and cross polarizations (in that order).
    """

    quality, frequency, hrss, phase, eccentricity = theta

    pi = jnp.array([PI])

    # calculate relative hplus / hcross amplitudes based on eccentricity
    # as well as normalization factors
    a, b = semi_major_minor_from_e(eccentricity)
    norm_prefactor = quality / (4.0 * frequency * jnp.sqrt(pi))
    cosine_norm = norm_prefactor * (1.0 + jnp.exp(-quality * quality))
    sine_norm = norm_prefactor * (1.0 - jnp.exp(-quality * quality))

    cos_phase, sin_phase = jnp.cos(phase), jnp.sin(phase)

    h0_plus = (
        hrss * a / jnp.sqrt(cosine_norm * (cos_phase**2) + sine_norm * (sin_phase**2))
    )
    h0_cross = (
        hrss * b / jnp.sqrt(cosine_norm * (sin_phase**2) + sine_norm * (cos_phase**2))
    )

    # cast the phase to a complex number
    phi = 2 * pi * frequency * t
    complex_phase = complex(jnp.zeros_like(phi), (phi - phase))

    # calculate the waveform and apply a tukey
    # window to taper the waveform
    fac = jnp.exp(phi**2 / (-2.0 * quality**2) + complex_phase)

    cross = fac.imag * h0_cross
    plus = fac.real * h0_plus

    return plus, cross


@register("SineGaussian", is_tidal=False, is_precessing=False)
class SineGaussian(TimeDomainWaveform):
    """Sine-Gaussian time-domain burst waveform."""

    def __init__(self) -> None:
        pass

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("Q", "f_0", "hrss", "phase", "e")

    def __call__(
        self, t: Float[Array, " n_time"], params: Mapping[str, FloatLike]
    ) -> dict[str, Float[Array, " n_time"]]:
        """Evaluate the SineGaussian waveform.

        Args:
            t (Float[Array, " n_time"]): Time grid centered at t=0. Create using
                ``jnp.arange(-duration/2, duration/2, 1/fs)``.
            params: Source parameters with keys ``Q``
                (quality factor), ``f_0`` (central frequency in Hz), ``hrss``,
                ``phase``, ``e`` (eccentricity).

        Returns:
            dict[str, Float[Array, " n_time"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        theta = jnp.array(
            [
                params["Q"],
                params["f_0"],
                params["hrss"],
                params["phase"],
                params["e"],
            ]
        )
        hp, hc = gen_SineGaussian_hphc(t, theta)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return "SineGaussian()"
