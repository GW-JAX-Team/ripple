"""Chirp time: the time from a given frequency to merger."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ripplegw.constants import MTSUN, PI
from ripplegw.typing import FloatLike


def time_to_merger(
    frequency: Float[Array, " *n_freq"] | FloatLike,
    M_c: FloatLike,
    eta: FloatLike,
    s1_z: FloatLike = 0.0,
    s2_z: FloatLike = 0.0,
    mode: tuple[int, int] = (2, 2),
) -> Float[Array, " *n_freq"]:
    r"""Time from emission at ``frequency`` to merger, at 2PN with aligned spins, for one $(\ell, m)$ mode.

    Works under ``jax.jit`` (with ``mode`` static), ``jax.vmap`` and ``jax.grad``.
    It is an inspiral approximation and loses accuracy close to merger.

    Example:
        ```python
        f = jnp.geomspace(20.0, 200.0, 100)
        tau_22 = time_to_merger(f, M_c, eta, s1_z, s2_z)  # shape (100,)
        tau_33 = time_to_merger(f, M_c, eta, s1_z, s2_z, mode=(3, 3))  # shape (100,)
        ```

    Args:
        frequency (Array): Frequency of the mode in Hz, of any shape. Must be positive.
        M_c (FloatLike): Detector-frame chirp mass in solar masses.
        eta (FloatLike): Symmetric mass ratio, $0 < \eta \leq 1/4$.
        s1_z (FloatLike, optional): Aligned dimensionless spin of the heavier component. Defaults to 0.
        s2_z (FloatLike, optional): Aligned dimensionless spin of the lighter component. Defaults to 0.
        mode (tuple[int, int], optional): The ``(l, m)`` mode ``frequency`` refers to, with ``l >= 2`` and ``0 < |m| <= l``.
            Defaults to ``(2, 2)``.

    Returns:
        Array: Time to merger in seconds, with the shape of ``frequency``.

    Raises:
        ValueError: If ``mode`` is not a tuple of two ints with ``l >= 2`` and ``0 < |m| <= l``.
    """
    if not (
        isinstance(mode, tuple)
        and len(mode) == 2
        and all(isinstance(x, int) for x in mode)
    ):
        raise ValueError(f"mode must be a tuple (l, m) of two ints; got {mode!r}.")
    ell, m = mode
    if ell < 2 or m == 0 or abs(m) > ell:
        raise ValueError(f"mode needs l >= 2 and 0 < |m| <= l; got {mode!r}.")
    f_22 = 2.0 * jnp.asarray(frequency) / abs(m)
    return _time_to_merger_2PN(f_22, M_c, eta, s1_z, s2_z)


def _times_delta(c: FloatLike, eta: FloatLike) -> FloatLike:
    """``c * sqrt(1 - 4 eta)``, with finite derivatives at ``eta = 1/4`` when ``c == 0``."""
    one_minus_4eta = 1.0 - 4.0 * eta
    guard = (one_minus_4eta <= 0.0) & (c == 0)
    root = jnp.sqrt(jnp.where(guard, 1.0, one_minus_4eta))
    return jnp.where(guard, 0.0, c * root)


def _time_to_merger_2PN(
    f_22: Float[Array, " *shape"],
    M_c: FloatLike,
    eta: FloatLike,
    s1_z: FloatLike,
    s2_z: FloatLike,
) -> Float[Array, " *shape"]:
    """2PN time to merger of the dominant mode at frequency ``f_22``."""
    M_s = M_c * eta ** (-3.0 / 5.0) * MTSUN

    x = jnp.cbrt(PI * M_s * f_22)
    x2 = jnp.power(x, 2)
    x3 = jnp.power(x, 3)
    x4 = jnp.power(x, 4)
    beta_sym = (56.5 - 38.0 * eta) * (s1_z + s2_z) / 12.0
    # Twice the delta-free part of sigma.
    sigma_sym_2 = (
        79.0 * eta * s1_z * s2_z + 81.0 / 4.0 * (1.0 - 2.0 * eta) * (s1_z**2 + s2_z**2)
    ) / 4.0
    # Spin-orbit and spin-spin terms proportional to delta = sqrt(1 - 4 eta).
    delta_coefficient = (
        8.0 / 5.0 * 113.0 / 12.0 * (s1_z - s2_z) / 2.0 * x3
        - 2.0 * 81.0 / 16.0 * (s1_z**2 - s2_z**2) / 2.0 * x4
    )

    tau_newtonian = 5.0 * M_s / (256.0 * eta * jnp.power(x, 8))
    return tau_newtonian * (
        1.0
        + x2 * (743.0 / 84.0 + 11.0 * eta) / 3.0
        - x3 * 8.0 / 5.0 * (4.0 * PI - beta_sym)
        + x4
        * (
            3058673.0 / 508032.0
            + 5429.0 / 504.0 * eta
            + 617.0 / 72.0 * eta**2
            - sigma_sym_2
        )
        + _times_delta(delta_coefficient, eta)
    )
