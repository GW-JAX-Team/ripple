"""Exact continuous-wave pulsar signal: plus/cross polarizations in JAX.

This is the JAX port of the *source* model underlying
``XLALSimulateExactPulsarSignal`` (lalpulsar/lib/SimulatePulsarSignal.c),
returning the two polarizations

    h_+(t) = A_+ cos Ψ(t),   h_x(t) = A_x sin Ψ(t),

with the barycentered wave phase

    Ψ(t) = φ₀ + 2π Σ_{k≥0}  f^{(k)} / (k+1)!  · τ(t)^{k+1},
    τ(t) = [t + n·rDetector(t)] − t_ref^SSB.

LAL builds the *detector strain* ``h(t) = F₊(t) h₊(t) + F_×(t) h_×(t)`` from
these polarizations and the antenna-pattern functions; here we return the
polarizations only (matching ripple's ``{"p", "c"}`` convention). The
antenna-pattern projection — which also fixes the polarization angle ψ and the
arm-opening factor sin ζ — is left to the caller.

Like LAL's exact reference, the Einstein and Shapiro relativistic timing terms
are neglected (the delay is the purely geometric ``n·rDetector``), and the wave
phase is evaluated in REAL8 GPS time.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constants import PI, TWO_PI
from .barycenter import emission_delay, roemer_delay
from .earth import earth_state

__all__ = [
    "exact_pulsar_polarizations",
    "generate_pulsar_polarizations",
    "generate_binary_pulsar_polarizations",
]


def _phase_taylor(
    tau: Float[Array, " n"], f0: Float, fkdot: Float[Array, " k"] | tuple
) -> Float[Array, " n"]:
    """Wave phase ``2π Σ f^{(s)}/(s+1)! · τ^{s+1}`` (excluding φ₀).

    Args:
        tau (Float[Array, " n"]): SSB time relative to the reference epoch (s).
        f0 (Float): Wave frequency at the reference epoch (Hz).
        fkdot (Float[Array, " k"]): Spindown terms ``(f1, f2, ...)`` in
            Hz/s, Hz/s², … (may be empty).

    Returns:
        Float[Array, " n"]: Phase contribution in radians.
    """
    # coefficients f^{(s)} for s = 0, 1, ... : [f0, f1, f2, ...]
    coeffs = jnp.concatenate(
        [
            jnp.atleast_1d(jnp.asarray(f0, dtype=tau.dtype)),
            jnp.asarray(fkdot, dtype=tau.dtype).ravel(),
        ]
    )
    phase = jnp.zeros_like(tau)
    fact = 1.0  # will hold (s+1)!
    tau_pow = tau  # τ^{s+1}
    for s in range(coeffs.shape[0]):
        fact *= s + 1  # (s+1)!
        phase = phase + (coeffs[s] / fact) * tau_pow
        tau_pow = tau_pow * tau
    return TWO_PI * phase


def exact_pulsar_polarizations(
    dt_rel: Float[Array, " n"],
    start_gps: int,
    alpha: Float,
    delta: Float,
    f0: Float,
    phi0: Float,
    aplus: Float,
    across: Float,
    det_location_m: tuple[float, float, float],
    eph_gps0: float,
    eph_dt: float,
    eph_pos: Float[Array, "m 3"],
    eph_vel: Float[Array, "m 3"],
    eph_acc: Float[Array, "m 3"],
    fkdot: Float[Array, " k"] | tuple = (),
    ref_time_ssb: float | None = None,
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Plus/cross polarizations of an isolated pulsar, evaluated at the detector.

    Args:
        dt_rel (Float[Array, " n"]): Sample times as seconds since
            ``start_gps`` (small numbers, to preserve float64 precision).
        start_gps (int): Integer GPS second of ``dt_rel == 0`` (the start of
            the time series / detector arrival time of the first sample).
        alpha (Float): Source right ascension ``α`` (radians).
        delta (Float): Source declination ``δ`` (radians).
        f0 (Float): Wave frequency at the reference epoch (Hz).
        phi0 (Float): Initial phase ``φ₀`` at the reference epoch (radians).
        aplus (Float): Plus polarization amplitude ``A₊``.
        across (Float): Cross polarization amplitude ``A_×``.
        det_location_m (tuple): Detector geocentric location (metres).
        eph_gps0, eph_dt, eph_pos, eph_vel, eph_acc: Earth ephemeris table.
        fkdot (Float[Array, " k"] | tuple): Spindown terms ``(f1, f2, ...)``
            (Hz/s, …). Up to any order; defaults to none.
        ref_time_ssb (float | None): SSB reference time for the spin parameters
            (seconds). If ``None``, uses the SSB time of the first sample,
            matching LAL's default ``refTime == startTime → SSB``.

    Returns:
        tuple: ``(h_plus, h_cross)``, each ``Float[Array, " n"]``.
    """
    dt_rel = jnp.asarray(dt_rel)
    floor_dt = jnp.floor(dt_rel)
    gps_int = start_gps + floor_dt
    gps_frac = dt_rel - floor_dt

    eph = (eph_gps0, eph_dt, eph_pos, eph_vel, eph_acc)

    dT = roemer_delay(gps_int, gps_frac, alpha, delta, det_location_m, *eph)
    # Elapsed time since start_gps, kept exact by subtracting the integer
    # second before adding the fraction (avoids cancelling two ~1e9 numbers).
    t_elapsed = (gps_int - start_gps) + gps_frac

    if ref_time_ssb is None:
        # startTimeSSB = t0 + n·rDetector(t0), with t0 = start_gps (frac 0).
        dT0 = roemer_delay(
            jnp.atleast_1d(jnp.asarray(float(start_gps))),
            jnp.zeros(1, dtype=dT.dtype),
            alpha,
            delta,
            det_location_m,
            *eph,
        )[0]
        # τ = SSB_geom(t) − startTimeSSB = (t − t0) + (dT(t) − dT(t0)).
        tau = t_elapsed + (dT - dT0)
    else:
        # τ = SSB_geom(t) − t_ref = (t − t0) + (t0 − t_ref) + dT(t).
        tau = (t_elapsed + (float(start_gps) - ref_time_ssb)) + dT

    phase = phi0 + _phase_taylor(tau, f0, fkdot)
    h_plus = aplus * jnp.cos(phase)
    h_cross = across * jnp.sin(phase)
    return h_plus, h_cross


def generate_pulsar_polarizations(
    dt_rel: Float[Array, " n"],
    start_gps: int,
    alpha: Float,
    delta: Float,
    f0: Float,
    phi0: Float,
    aplus: Float,
    across: Float,
    det_location_m: tuple[float, float, float],
    earth_gps0: float,
    earth_dt: float,
    earth_pos: Float[Array, "m 3"],
    earth_vel: Float[Array, "m 3"],
    earth_acc: Float[Array, "m 3"],
    sun_gps0: float,
    sun_dt: float,
    sun_pos: Float[Array, "k 3"],
    sun_vel: Float[Array, "k 3"],
    sun_acc: Float[Array, "k 3"],
    fkdot: Float[Array, " j"] | tuple = (),
    ref_time_ssb: float | None = None,
    f_heterodyne: float = 0.0,
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Plus/cross polarizations of an isolated pulsar (full ``XLALGeneratePulsarSignal``).

    Unlike :func:`exact_pulsar_polarizations`, the time delay is the *full*
    barycentering delay (Roemer + Earth-rotation with precession/nutation +
    Einstein - Shapiro), and an optional heterodyne phase
    ``2π f_het (t - t_het)`` is removed (with ``t_het`` = ``start_gps``, the
    heterodyne epoch used by LAL).

    Args:
        dt_rel (Float[Array, " n"]): Sample times in seconds since ``start_gps``.
        start_gps (int): Integer GPS second of ``dt_rel == 0``; also the
            heterodyne epoch ``t_het``.
        alpha, delta (Float): Source sky position (radians).
        f0 (Float): Wave frequency at the reference epoch (Hz).
        phi0 (Float): Initial phase at the reference epoch (radians).
        aplus, across (Float): Polarization amplitudes.
        det_location_m (tuple): Detector geocentric location (metres).
        earth_*: Earth ephemeris table.
        sun_*: Sun ephemeris table (for the Shapiro delay).
        fkdot (Float[Array, " j"] | tuple): Spindown terms ``(f1, f2, ...)``.
        ref_time_ssb (float | None): SSB reference epoch for the spins. If
            ``None``, uses the *full* SSB time of the first sample
            (``start_gps + emit.deltaT(start_gps)``), matching LAL's default.
        f_heterodyne (float): Heterodyne frequency (Hz). ``0`` disables it.

    Returns:
        tuple: ``(h_plus, h_cross)`` heterodyned polarizations.
    """
    dt_rel = jnp.asarray(dt_rel)
    floor_dt = jnp.floor(dt_rel)
    gps_int = start_gps + floor_dt
    gps_frac = dt_rel - floor_dt
    t_elapsed = (gps_int - start_gps) + gps_frac

    earth_args = (earth_gps0, earth_dt, earth_pos, earth_vel, earth_acc)
    sun_args = (sun_gps0, sun_dt, sun_pos, sun_vel, sun_acc)

    es = earth_state(gps_int, gps_frac, *earth_args, *sun_args)
    delta_t = emission_delay(es, alpha, delta, det_location_m)

    if ref_time_ssb is None:
        es0 = earth_state(
            jnp.atleast_1d(jnp.asarray(float(start_gps))),
            jnp.zeros(1, dtype=delta_t.dtype),
            *earth_args,
            *sun_args,
        )
        delta_t0 = emission_delay(es0, alpha, delta, det_location_m)[0]
        # τ = t_e(t) − t_e(t0) = (t − t0) + (deltaT(t) − deltaT(t0)).
        tau = t_elapsed + (delta_t - delta_t0)
    else:
        # τ = t_e(t) − t_ref = (t − t0) + (t0 − t_ref) + deltaT(t).
        tau = (t_elapsed + (float(start_gps) - ref_time_ssb)) + delta_t

    phase = phi0 + _phase_taylor(tau, f0, fkdot)
    phase = phase - TWO_PI * f_heterodyne * t_elapsed
    h_plus = aplus * jnp.cos(phase)
    h_cross = across * jnp.sin(phase)
    return h_plus, h_cross


def _solve_kepler(x, a, b, n_iter=64):
    """Solve ``E + a·sin E + b·(cos E - 1) = x`` for E ∈ [0, 2π).

    Newton iteration with LAL's step clamping (|ΔE| ≤ π, E ∈ [0, 2π]). A fixed
    iteration count keeps the solve jit-/grad-friendly; for the physical regime
    ``v_p < 1`` it converges in well under 64 steps.
    """
    e = x  # mean anomaly is a good initial guess
    for _ in range(n_iter):
        sine = jnp.sin(e)
        cose = jnp.cos(e) - 1.0
        dx = e + a * sine + b * cose - x
        de = dx / (1.0 + a * cose + a - b * sine)
        de = jnp.clip(de, -PI, PI)
        e = e - de
        e = jnp.clip(e, 0.0, TWO_PI)
    return e


def _binary_source_phase(
    tau_minus_tp, spin_off, f0, phi0, fkdot, asini, ecc, period, argp
):
    """Unheterodyned source phase ``φ(τ)`` for an elliptic-orbit pulsar.

    Port of the per-sample core of ``XLALGenerateSpinOrbitCW`` (elliptic case):
    map the SSB arrival time to the pulsar emission time via Kepler's equation,
    then apply the spindown phase polynomial.

    Args:
        tau_minus_tp: SSB arrival time minus periapsis time ``τ - t_p`` (s).
        spin_off: ``t_p - t_ref`` (s), offset of periapsis from the spin epoch.
        f0, phi0, fkdot: Wave frequency, initial phase, spindowns.
        asini, ecc, period, argp: Orbital elements (light-s, -, s, rad).

    Returns:
        Phase ``φ`` in radians (including ``φ0``).
    """
    one_minus_ecc = 1.0 - ecc
    one_plus_ecc = 1.0 + ecc
    angular_speed = (TWO_PI / period) * jnp.sqrt(one_plus_ecc / one_minus_ecc**3)
    r_peri_norm = asini * one_minus_ecc
    vp = r_peri_norm * angular_speed
    v_dot_avg = angular_speed * jnp.sqrt(one_minus_ecc**3 / one_plus_ecc)
    a_kep = vp * one_minus_ecc * jnp.cos(argp) + one_minus_ecc - 1.0
    b_kep = vp * jnp.sqrt(one_minus_ecc / one_plus_ecc) * jnp.sin(argp)

    x_full = v_dot_avg * (tau_minus_tp - r_peri_norm * jnp.sin(argp))
    n_orb = jnp.floor(x_full / TWO_PI)
    x = x_full - TWO_PI * n_orb

    big_e = _solve_kepler(x, a_kep, b_kep)
    t_emit = (big_e + TWO_PI * n_orb - ecc * jnp.sin(big_e)) / v_dot_avg + spin_off
    return phi0 + _phase_taylor(t_emit, f0, fkdot)


def generate_binary_pulsar_polarizations(
    dt_rel: Float[Array, " n"],
    start_gps: int,
    alpha: Float,
    delta: Float,
    f0: Float,
    phi0: Float,
    aplus: Float,
    across: Float,
    asini: Float,
    ecc: Float,
    period: Float,
    argp: Float,
    tp_ssb: float,
    det_location_m: tuple[float, float, float],
    earth_gps0: float,
    earth_dt: float,
    earth_pos: Float[Array, "m 3"],
    earth_vel: Float[Array, "m 3"],
    earth_acc: Float[Array, "m 3"],
    sun_gps0: float,
    sun_dt: float,
    sun_pos: Float[Array, "p 3"],
    sun_vel: Float[Array, "p 3"],
    sun_acc: Float[Array, "p 3"],
    fkdot: Float[Array, " j"] | tuple = (),
    ref_time_ssb: float | None = None,
    f_heterodyne: float = 0.0,
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Plus/cross polarizations of a pulsar in a binary orbit.

    Extends :func:`generate_pulsar_polarizations` with the orbital light-travel
    modulation of ``XLALGenerateSpinOrbitCW`` (elliptic orbit): the SSB arrival
    time is mapped through Kepler's equation to the pulsar emission time before
    the spindown phase is evaluated.

    Args:
        dt_rel, start_gps, alpha, delta, f0, phi0, aplus, across: As in
            :func:`generate_pulsar_polarizations`.
        asini (Float): Projected semi-major axis ``a sin i`` (light-seconds).
        ecc (Float): Orbital eccentricity.
        period (Float): Orbital period (seconds).
        argp (Float): Argument of periapsis ``ω`` (radians).
        tp_ssb (float): Time of periapsis passage in the SSB frame (GPS seconds).
        det_location_m, earth_*, sun_*: Detector and ephemerides.
        fkdot (Float[Array, " j"] | tuple): Spindown terms.
        ref_time_ssb (float | None): SSB spin reference epoch (default: full SSB
            time of the first sample).
        f_heterodyne (float): Heterodyne frequency (Hz).

    Returns:
        tuple: ``(h_plus, h_cross)``.
    """
    dt_rel = jnp.asarray(dt_rel)
    floor_dt = jnp.floor(dt_rel)
    gps_int = start_gps + floor_dt
    gps_frac = dt_rel - floor_dt
    t_elapsed = (gps_int - start_gps) + gps_frac

    earth_args = (earth_gps0, earth_dt, earth_pos, earth_vel, earth_acc)
    sun_args = (sun_gps0, sun_dt, sun_pos, sun_vel, sun_acc)
    es = earth_state(gps_int, gps_frac, *earth_args, *sun_args)
    delta_t = emission_delay(es, alpha, delta, det_location_m)

    # SSB arrival time minus periapsis: kept precise by subtracting integers
    # before adding the (possibly large) start->periapsis gap.
    tau_minus_tp = (t_elapsed + delta_t) + (float(start_gps) - tp_ssb)

    if ref_time_ssb is None:
        # t_ref = startTimeSSB = start_gps + emit.deltaT(start_gps). Keep
        # deltaT0 as a traced scalar (do NOT cast to float) so the function
        # stays jit/grad-compatible when the sky position is traced.
        es0 = earth_state(
            jnp.atleast_1d(jnp.asarray(float(start_gps))),
            jnp.zeros(1, dtype=delta_t.dtype),
            *earth_args,
            *sun_args,
        )
        delta_t0 = emission_delay(es0, alpha, delta, det_location_m)[0]
        spin_off = (tp_ssb - float(start_gps)) - delta_t0  # t_p - t_ref
    else:
        spin_off = tp_ssb - ref_time_ssb

    phase = _binary_source_phase(
        tau_minus_tp, spin_off, f0, phi0, fkdot, asini, ecc, period, argp
    )
    phase = phase - TWO_PI * f_heterodyne * t_elapsed
    h_plus = aplus * jnp.cos(phase)
    h_cross = across * jnp.sin(phase)
    return h_plus, h_cross
