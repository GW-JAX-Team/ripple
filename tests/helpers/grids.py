"""Evaluation-grid construction for waveform tests.

Frequency grids mirror the ``rfftfreq`` band a real analysis would use; time
grids are centred at zero, matching the convention every ``TD`` model in
ripple expects (see ``SineGaussian``).
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import ripplegw


def freq_grid(f_l: float, f_u: float, f_sampling: float, T: float) -> jnp.ndarray:
    """Frequency grid strictly inside ``(f_l, f_u)``, from an rfftfreq band of duration ``T``."""
    delta_t = 1 / f_sampling
    tlen = round(T / delta_t)
    freqs = np.fft.rfftfreq(tlen, delta_t)
    return jnp.array(freqs[(freqs > f_l) & (freqs < f_u)])


def time_grid(duration: float, f_sampling: float) -> jnp.ndarray:
    """Time grid centred at zero, matching ripple's time-domain axis convention."""
    return jnp.arange(-duration / 2, duration / 2, 1 / f_sampling)


def grid_for(name: str, *, small: bool = False) -> jnp.ndarray:
    """Evaluation axis appropriate for a registered waveform's domain.

    ``small=True`` returns a ~500-point grid, cheap enough for eager (non-jit)
    edge-case calls; otherwise a production-sized grid.
    """
    domain = ripplegw.get_waveform_metadata(name)["domain"]
    if domain == "FD":
        if small:
            return jnp.linspace(20.0, 512.0, 500)
        return freq_grid(20.0, 1024.0, 2048.0, 2.0)
    if domain == "TD":
        if small:
            return jnp.linspace(-0.5, 0.5, 512)
        return time_grid(1.0, 4096.0)
    raise ValueError(f"Unknown domain {domain!r} for waveform {name!r}")


@dataclass(frozen=True)
class Grid:
    """A frequency-domain evaluation axis plus the metadata a reference
    backend needs to regenerate the same band independently.

    ``axis`` is what gets passed to ``wf(axis, params)``; ``f_l``/``f_u``/``df``
    describe the analysis band it was masked from, since a reference backend
    (e.g. LAL) needs the *unmasked* generation parameters, not just the axis.
    """

    axis: jnp.ndarray
    f_l: float
    f_u: float
    df: float
    f_ref: float


def accuracy_grid(
    f_l: float, f_u: float, f_sampling: float, T: float, f_ref: float
) -> Grid:
    """``Grid`` for the cross-validation campaign, built the same way ``freq_grid`` is."""
    axis = freq_grid(f_l, f_u, f_sampling, T)
    return Grid(axis=axis, f_l=f_l, f_u=f_u, df=1.0 / T, f_ref=f_ref)
