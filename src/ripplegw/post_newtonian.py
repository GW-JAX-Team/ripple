r"""
Post-Newtonian relations for compact binary inspirals.

These are closed-form approximations that are useful outside a full waveform
model, for example to map a frequency to the time at which the signal sweeps
through it.
"""

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Integer

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.typing import FloatLike
from ripplegw.waveforms.cbc.Taylor.TaylorF2 import get_PNPhasing_F2

MAX_PN_2 = 7  # Twice the maximum PN order

# A bare two-int sequence is accepted at runtime and read as one (l, m) pair.
Mode = (
    int
    | tuple[int, int]
    | Sequence[int]
    | Sequence[tuple[int, int]]
    | Integer[Array, "n_modes 2"]
)


def time_to_merger(
    frequency: Float[Array, " *n_freq"] | FloatLike,
    M_c: FloatLike,
    eta: FloatLike,
    s1_z: FloatLike = 0.0,
    s2_z: FloatLike = 0.0,
    mode: Mode = 2,
) -> Float[Array, " *n_out"]:
    r"""Time from emission of ``frequency`` to merger, at 2PN, for any $(\ell, m)$ mode.

    Under the stationary-phase approximation a harmonic with azimuthal number
    $m$ is emitted at frequency $f$ when the orbital frequency is $f / |m|$,
    so its time to merger equals that of the GW frequency of the dominant mode
    $f_{22} = 2 f / |m|$; the polar index $\ell$ does not enter. For the
    dominant mode,

    $$
    \tau(f_{22}) = \frac{5}{256}\,\frac{M}{\eta \, x^4}\,
        \Big[1 + \tfrac{4}{3}\big(\tfrac{743}{336} + \tfrac{11}{4}\eta\big)x
        - \tfrac{8}{5}(4\pi - \beta)\,x^{3/2}
        + 2\big(\tfrac{3058673}{1016064} + \tfrac{5429}{1008}\eta
        + \tfrac{617}{144}\eta^2 - \sigma\big)x^2\Big],
    $$

    with $x = (\pi M f_{22})^{2/3}$, where $M = \mathcal{M}\,\eta^{-3/5}$
    is the total mass (not the chirp mass), and, for spins aligned with the
    orbital angular momentum,

    $$
    \beta = \frac{1}{12}\sum_i \big(113\,\hat m_i^2 + 75\,\eta\big)\chi_i,
    \qquad
    \sigma = \frac{79}{8}\,\eta\,\chi_1\chi_2
        + \frac{81}{16}\sum_i \hat m_i^2 \chi_i^2,
    $$

    where $\hat m_i = m_i / M$. The spin-orbit term $\beta$ and the
    spin-spin interaction in $\sigma$ follow Poisson & Will (1995),
    [arXiv:gr-qc/9502040](https://arxiv.org/abs/gr-qc/9502040), Eq. (3.3).
    The self-spin terms in $\sigma$ include the quadrupole-monopole
    coupling for black holes (spin-induced quadrupole parameter equal to 1),
    so that, for black holes or non-spinning bodies, $\tau$ is the
    stationary-phase time of the TaylorF2 phasing truncated at 2PN. For a
    spinning neutron star TaylorF2 uses a larger quadrupole parameter; the
    difference in $\tau$ is small for realistic neutron-star spins.

    The non-spinning coefficients, and the spin terms for equal masses with
    equal spins, coincide with the 2PN chirp time of LALSimulation's
    reduced-spin TaylorF2 model, ``XLALSimInspiralTaylorF2ReducedSpinChirpTime``
    (Ajith 2011, [arXiv:1107.1267](https://arxiv.org/abs/1107.1267)); for a
    non-spinning $1.4 + 1.4\,M_\odot$ binary both give $\tau = 1016.163\,$s
    from 10 Hz, the familiar ~17 minutes. The mapping $f_{22} = 2f/|m|$ is
    the post-Newtonian frequency scaling used for the inspiral of every mode
    in IMRPhenomHM (London et al. 2018,
    [arXiv:1708.00404](https://arxiv.org/abs/1708.00404)).

    The expression is an inspiral approximation: it diverges at low frequency
    and loses accuracy near merger, where it can turn negative.

    Only ``mode`` is validated. Physical inputs are not checked, so that the
    function stays cheap under ``jax.jit``: a zero frequency gives ``inf``, a
    negative frequency ``nan``, and $\eta > 1/4$ gives ``nan`` for unequal
    spins (and a finite analytic continuation for equal spins). At $\eta = 1/4$ the value and its gradients are finite
    for equal spins, including non-spinning binaries. For unequal spins the
    derivative with respect to $\eta$ diverges as $\eta \to 1/4$, because
    the component masses depend on $\sqrt{1 - 4\eta}$, and ``jax.grad``
    returns $\pm\infty$ there.

    Example:
        ```python
        f = jnp.geomspace(20.0, 200.0, 100)
        tau_22 = time_to_merger(f, M_c, eta, s1_z, s2_z)  # shape (100,)
        tau_33 = time_to_merger(f, M_c, eta, s1_z, s2_z, mode=(3, 3))  # shape (100,)
        tau_hm = time_to_merger(
            f, M_c, eta, s1_z, s2_z, mode=((2, 2), (2, 1), (3, 3), (4, 4))
        )  # shape (4, 100)
        ```

    Args:
        frequency (Array): Frequency of the mode in Hz, of any shape. Must be
            positive.
        M_c (FloatLike): Detector-frame chirp mass in solar masses.
        eta (FloatLike): Symmetric mass ratio, $0 < \eta \leq 1/4$.
        s1_z (FloatLike, optional): Aligned dimensionless spin of the primary
            (heavier) component. Defaults to 0.
        s2_z (FloatLike, optional): Aligned dimensionless spin of the
            secondary component. Defaults to 0.
        mode (optional): The mode(s) that ``frequency`` refers to, given as an
            azimuthal number ``m``, a single ``(l, m)`` pair, or a sequence or
            ``(n_modes, 2)`` integer array of ``(l, m)`` pairs (the layout of
            a waveform's mode array). Only ``|m|`` affects the result. A
            bare two-element sequence of integers, such as ``[3, 2]``, is one
            ``(l, m)`` pair, not two azimuthal numbers. ``mode`` must be
            concrete, not traced; to pass it as a static argument of
            ``jax.jit`` it must also be hashable, so use tuples rather than
            lists. Defaults to 2.

    Returns:
        Array: Time to merger in seconds. For a single mode it has the shape
            of ``frequency``; for several modes it has shape
            ``(n_modes, *frequency.shape)``, in the order given.

    Raises:
        ValueError: If a mode has ``m = 0``, for which there is no
            frequency-to-time mapping, if an ``(l, m)`` pair has ``l < 2`` or
            ``|m| > l``, or if ``mode`` is not one of the accepted forms.
    """
    abs_m, single = _azimuthal_numbers(mode)
    f = jnp.asarray(frequency)
    f_22 = 2.0 * f / abs_m.reshape((-1,) + (1,) * f.ndim)
    tau = _time_to_merger_2PN(f_22, M_c, eta, s1_z, s2_z)
    return tau[0] if single else tau


def _azimuthal_numbers(mode: Mode) -> tuple[np.ndarray, bool]:
    """Validate ``mode`` and return ``(|m| per mode, whether a single mode was given)``."""
    try:
        modes = np.asarray(mode)
    except Exception as err:  # e.g. a traced JAX array
        raise ValueError(
            "mode must be concrete: an int m, an (l, m) pair, or a sequence of "
            "(l, m) pairs."
        ) from err
    if modes.size == 0 or not np.issubdtype(modes.dtype, np.integer):
        raise ValueError(
            f"mode must be an int m, an (l, m) pair, or a sequence of (l, m) "
            f"pairs of integers; got {mode!r}."
        )

    if modes.ndim == 0:
        m = modes.reshape(1)
        ell = None
        single = True
    elif modes.shape == (2,):
        ell, m = modes[:1], modes[1:]
        single = True
    elif modes.ndim == 2 and modes.shape[1] == 2:
        ell, m = modes[:, 0], modes[:, 1]
        single = False
    else:
        raise ValueError(
            f"mode must be an int m, an (l, m) pair, or a sequence of (l, m) "
            f"pairs; got an array of shape {modes.shape}."
        )

    if np.any(m == 0):
        raise ValueError(
            "mode must have a non-zero azimuthal number: an m = 0 mode has no "
            "frequency-to-time mapping."
        )
    if ell is not None and (np.any(ell < 2) or np.any(np.abs(m) > ell)):
        raise ValueError(
            f"each (l, m) pair needs l >= 2 and |m| <= l; got {modes.tolist()}."
        )
    return np.abs(m).astype(float), single


def _times_delta(c: FloatLike, eta: FloatLike) -> FloatLike:
    """``c * sqrt(1 - 4 eta)``, made differentiable where the root is singular.

    The root has an infinite derivative at ``eta = 1/4``; multiplied by a zero
    coefficient, automatic differentiation would turn the finite limit into
    ``nan``. The guard fires only there, at ``eta >= 1/4`` with ``c == 0``,
    where the product and its derivatives are exactly zero. Everywhere else
    the expression is differentiated as written, so derivatives through ``c``
    are kept (for example with respect to equal spins below ``eta = 1/4``) and
    the derivative is infinite at ``eta = 1/4`` when ``c`` is non-zero.
    """
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

    # With m1_hat, m2_hat = (1 +/- delta) / 2 and delta**2 = 1 - 4 eta,
    #   sum_i m_hat_i**2 a_i = (1 - 2 eta) (a_1 + a_2) / 2 + delta (a_1 - a_2) / 2,
    # so delta enters only through the spin asymmetries. beta and sigma are
    # split into a delta-free part and a part linear in delta; the latter is
    # collected into one coefficient so that its derivative is guarded once.
    x = jnp.cbrt(PI * M_s * f_22)
    x2 = jnp.power(x, 2)
    x3 = jnp.power(x, 3)
    x4 = jnp.power(x, 4)
    beta_sym = (56.5 - 38.0 * eta) * (s1_z + s2_z) / 12.0
    # twice of sigma_sym
    sigma_sym_2 = (
        79.0 * eta * s1_z * s2_z + 81.0 / 4.0 * (1.0 - 2.0 * eta) * (s1_z**2 + s2_z**2)
    ) / 4.0
    # delta * d(beta)/d(delta) = delta * 113 / 12 * (s1 - s2) / 2, and likewise
    # for sigma with 81 / 16 * (s1**2 - s2**2) / 2.
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


PHASE_TO_TIME = {
    "0PN": 1.0,
    "1PN": 0.0,
    "2PN": 0.8,  # 4 / 5
    "3PN": 0.64,  # 16 / 25
    "4PN": 0.4,  # 2 / 5
    "5PN": -0.32,  # - 8 / 25
    "6PN": -0.4,  # - 2 / 5
    "7PN": -1.6,  # - 8 / 5
    "8PN": 1.6,  # 8 / 5
    "9PN": 1.6,  # 8 / 5
    "6lPN": -0.8,  # - 4 / 5
    "7lPN": -3.2,  # - 16 / 5
    "8lPN": 1.6,  # 8 / 5
    "9lPN": 6.4,  # 32 / 5
    "8llPN": 2.4,  # 12 / 5
}


def time_to_merger_4p5PN(
    f_22: Float[Array, " *shape"],
    M_c: FloatLike,
    eta: FloatLike,
    s1_z: FloatLike,
    s2_z: FloatLike,
    max_pn_2: Integer = MAX_PN_2,
) -> Float[Array, " *shape"]:
    r"""Compute the time to merger up to 4.5 PN order.
    Based on the relation between the phase and time.
    Assuming the phase can be expanded in a post-Newtonian series as:
    $$
    \Phi(v) = \phi_c - \frac{1}{16\eta} v^{-5} \left[
        1 + \sum_{k = 1}^{9} c_k\, v^k + \sum_{k = 5}^{9} d_k\,\ln(v)v^k
    \right]
    $$
    where $ v = (\pi M f)^{1/3}$ is the PN expansion parameter, and
    each term is known as the $k/2$-PN order term.

    The time to merger can then be obtained by noting that:
    $$
    \begin{align}
    \frac{\mathrm{d} t}{\mathrm{d} v}
    &= \frac{\mathrm{d} t}{\mathrm{d} \Phi} \,
        \frac{\mathrm{d} \Phi}{\mathrm{d} v} \\
    &= \frac{M}{v^3}\, \frac{\mathrm{d} \Phi}{\mathrm{d} v} \ .
    \end{align}
    $$
    Therefore, one has
    $$
    t(v) = t_\mathrm{c} + M \,\left(
        \frac{\Phi(v)}{v^3} + 3 \int \frac{\Phi(v)}{v^4} \, \mathrm{d} v
    \right) \ ,
    $$
    which can be again expressed as the following PN series:
    $$
    t(v) = t_\mathrm{c} - \frac{5\,M}{256\eta} v^{-8} \left[
        1 + \sum_{k = 2}^{9} C_k\, v^k + \sum_{k = 5}^{9} D_k\,\ln(v)v^k
        + E_8\,\ln^2(v)v^8
    \right]
    $$
    where $C_k$ and $D_k$ are some linear combinations of the $c_k$ and $d_k$
    from above. In fact, most $D_k$ are simply proportional to $d_k$, except
    $D_8$. The last term is simply $E_8 = (12 / 5) d_8$.
    """
    m1, m2 = Mc_eta_to_ms(jnp.stack([jnp.asarray(M_c), jnp.asarray(eta)]))
    Mtot = m1 + m2
    c_k, d_k = get_PNPhasing_F2(m1, m2, s1_z, s2_z, 0.0, 0.0)

    v = jnp.cbrt(np.pi * Mtot * MTSUN * f_22)
    tau_Newt = -(5 / 256) * (Mtot / eta) / jnp.power(v, 8)

    linear_term = 0
    log_term = 0

    # Start from the highest order (for precision)
    if max_pn_2 >= 9:
        # k == 9
        lab, llab = "9PN", "9lPN"
        vk = jnp.power(v, 9)
        _c_k = c_k.get(lab, 0)
        _d_k = d_k.get(lab, 0)
        linear_term += PHASE_TO_TIME[lab] * (4 * _c_k - 3 * _d_k) * vk
        log_term += PHASE_TO_TIME[llab] * _d_k * vk * jnp.log(v)

    if max_pn_2 >= 8:
        # k == 8
        lab, llab, lllab = "8PN", "8lPN", "8llPN"
        vk = jnp.power(v, 8)
        _c_k = c_k.get(lab, 0)
        _d_k = d_k.get(lab, 0)
        linear_term += PHASE_TO_TIME[lab] * _c_k * vk
        log_term += PHASE_TO_TIME[llab] * (3 * _c_k + _d_k) * vk * jnp.log(v)
        log_term += PHASE_TO_TIME[lllab] * _d_k * vk * jnp.log(v) ** 2

    # k from 6 to 7
    for k in reversed(range(6, 8)):
        lab, llab = f"{k}PN", f"{k}lPN"
        vk = jnp.power(v, k)
        _c_k = c_k.get(lab, 0)
        _d_k = d_k.get(lab, 0)
        linear_term += PHASE_TO_TIME[lab] * (2 * _c_k + 3 * _d_k) * vk
        log_term += PHASE_TO_TIME[llab] * _d_k * vk * jnp.log(v)

    # k == 5
    lab = "5PN"
    linear_term += PHASE_TO_TIME[lab] * d_k.get(lab, 0) * jnp.power(v, 5)

    # k from 2 to 4
    for k in reversed(range(2, 5)):
        lab = f"{k}PN"
        _c_k = c_k.get(lab, 0)
        linear_term += PHASE_TO_TIME[lab] * _c_k * jnp.power(v, k)

    # This is to correct for the extra scaling from `get_PNPhasing_F2``
    pfaN = 3.0 / (128.0 * eta)  # prefactor

    return tau_Newt / pfaN * (1 + linear_term + log_term) * MTSUN
