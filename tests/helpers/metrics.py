"""Frequency- and time-domain comparison metrics used by cross-validation."""

import jax.numpy as jnp
import numpy as np
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


def time_domain_overlap_loss(h1, h2) -> float:
    """White, normalized mismatch between two real sampled time series.

    This is ``1 - <h1, h2> / (||h1|| ||h2||)`` with the ordinary sampled
    time-domain inner product. It is appropriate when both series use the same,
    uniformly sampled time grid: the common ``delta_t`` factor cancels. It performs
    no FFT, whitening, or maximization over time or phase.

    The result measures shape and phase agreement, not absolute amplitude agreement:
    multiplying either input by a positive constant leaves it unchanged. Pair it with
    :func:`relative_norm_error` when a validation must also catch a global amplitude
    scale regression.

    Half the squared distance between normalized vectors is algebraically identical
    to the expression above, but remains well-defined for anti-correlated series and
    avoids subtracting nearly equal ``A*B`` and ``C**2`` terms.
    """
    h1 = _as_real_time_series(h1, "h1")
    h2 = _as_real_time_series(h2, "h2")
    _require_matching_shapes(h1, h2)

    norm1 = float(np.linalg.norm(h1))
    norm2 = float(np.linalg.norm(h2))
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("normalized time-domain mismatch is undefined for zero strain")

    difference = h1 / norm1 - h2 / norm2
    loss = 0.5 * float(difference @ difference)
    # Rounding can put an exactly identical/anti-correlated pair a few ulps outside
    # the mathematical [0, 2] range.
    return float(np.clip(loss, 0.0, 2.0))


def relative_norm_error(h1, h2) -> float:
    """Relative difference between two sampled time-series norms.

    This detects a global amplitude-scale regression without conflating it with a
    phase-only error between equal-norm signals. The result is
    ``abs(||h1|| / ||h2|| - 1)``; ``h2`` is normally the reference strain.
    """
    h1 = _as_real_time_series(h1, "h1")
    h2 = _as_real_time_series(h2, "h2")
    _require_matching_shapes(h1, h2)

    norm1 = float(np.linalg.norm(h1))
    norm2 = float(np.linalg.norm(h2))
    if norm2 == 0.0:
        raise ValueError("relative norm error is undefined for zero reference strain")
    return abs(norm1 / norm2 - 1.0)


def _as_real_time_series(values, name: str) -> np.ndarray:
    """Validate one real, finite, non-empty one-dimensional time series."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional time series")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_matching_shapes(h1: np.ndarray, h2: np.ndarray) -> None:
    if h1.shape != h2.shape:
        raise ValueError(
            f"time series must have identical shapes, got {h1.shape} and {h2.shape}"
        )
