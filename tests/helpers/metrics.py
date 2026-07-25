"""Noise-weighted comparison metrics used by the cross-validation campaign."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.integrate import trapezoid


def get_nyquist_mask(frequencies: jnp.ndarray, n_bins: int = 2) -> jnp.ndarray:
    """Mask that zeros the last ``n_bins`` frequency bins near Nyquist.

    LAL's behavior at the Nyquist frequency boundary is inconsistent -- it
    sometimes zeros 1 bin, sometimes 2 bins depending on the waveform
    parameters. Applying the same mask to both sides gives a fair comparison.
    """
    n_freqs = len(frequencies)
    return jnp.where(jnp.arange(n_freqs) < n_freqs - n_bins, 1.0, 0.0)


def noise_weighted_inner_product(
    h1: jnp.ndarray, h2: jnp.ndarray, psd: jnp.ndarray, frequencies: jnp.ndarray
) -> float:
    """Noise-weighted inner product ``(h1|h2)``, real part only.

    Use ``inner_product_complex`` for phase-sensitive operations.
    """
    integrand = jnp.conj(h1) * h2 / psd
    return 4 * trapezoid(integrand, x=frequencies, axis=-1).real


def inner_product_complex(
    h1: jnp.ndarray, h2: jnp.ndarray, psd: jnp.ndarray, frequencies: jnp.ndarray
) -> complex:
    """Complex noise-weighted inner product, retaining phase information."""
    integrand = jnp.conj(h1) * h2 / psd
    return 4 * trapezoid(integrand, x=frequencies, axis=-1)


def overlap(
    h1: jnp.ndarray, h2: jnp.ndarray, psd: jnp.ndarray, frequencies: jnp.ndarray
) -> float:
    """Noise-weighted overlap ``Re(<h1|h2>) / sqrt(<h1|h1> * <h2|h2>)``.

    No maximization over time or phase is performed.
    """
    h1_sq = noise_weighted_inner_product(h1, h1, psd, frequencies)
    h2_sq = noise_weighted_inner_product(h2, h2, psd, frequencies)
    h1_h2 = inner_product_complex(h1, h2, psd, frequencies)
    return h1_h2.real / jnp.sqrt(h1_sq * h2_sq)


def overlap_loss(
    h1: jnp.ndarray, h2: jnp.ndarray, psd: jnp.ndarray, frequencies: jnp.ndarray
) -> float:
    """``1 - overlap``, with improved numerical precision for near-unity overlaps.

    Uses the numerically stable identity::

        1 - C/sqrt(A*B) = (A*B - C**2) / (sqrt(A*B) * (sqrt(A*B) + C))

    where ``A = <h1|h1>``, ``B = <h2|h2>``, ``C = Re(<h1|h2>)``. No
    maximization over time or phase is performed.
    """
    h1_sq = noise_weighted_inner_product(h1, h1, psd, frequencies)
    h2_sq = noise_weighted_inner_product(h2, h2, psd, frequencies)
    h1_h2 = inner_product_complex(h1, h2, psd, frequencies).real
    denom = jnp.sqrt(h1_sq * h2_sq)
    loss = (h1_sq * h2_sq - h1_h2**2) / (denom * (denom + h1_h2))
    return jnp.clip(loss, 0.0)


def inner_product_phase(
    h1: jnp.ndarray, h2: jnp.ndarray, psd: jnp.ndarray, frequencies: jnp.ndarray
) -> float:
    """Phase angle of the noise-weighted inner product ``<h1|h2>``.

    For correctly phased waveforms this should be near 0 rad. A constant
    global phase offset phi (e.g. a spurious +pi in phi_ref) gives
    ``arg(<h1|h2>) = phi`` regardless of ``tc`` or amplitude scaling --
    the overlap test alone is blind to this, since ``|<e^{i phi} h1|h2>|^2``
    is identical for any real ``phi``.
    """
    h1_h2 = inner_product_complex(h1, h2, psd, frequencies)
    return jnp.angle(h1_h2)
