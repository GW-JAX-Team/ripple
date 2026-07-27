"""Unit tests for the pulsar-signal polarization generators.

The exact-signal tests use a *synthetic static* Earth ephemeris (zero
velocity and acceleration) together with a detector placed essentially at
the geocentre, so the geometric delay ``dT`` is constant. With the default
reference time the barycentered time then reduces to the plain elapsed time
``τ = t``, giving a closed-form phase against which the implementation can
be checked without any LAL dependency.

The full/binary-signal tests avoid any LAL/ephemeris-data dependency too:
they use a small synthetic ephemeris and check structural identities
(heterodyne relation, binary->isolated reduction, Kepler residual) plus
jit/grad.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from ripplegw.waveforms.cw.pulsar_signal import (
    _solve_kepler,
    exact_pulsar_polarizations,
    generate_binary_pulsar_polarizations,
    generate_pulsar_polarizations,
)

START_GPS = 1_000_000_000

# --- exact pulsar signal -----------------------------------------------------


def _static_ephemeris(n=8, dt=7200.0):
    """A constant-position Earth ephemeris (vel = acc = 0)."""
    gps0 = float(START_GPS - dt)  # so START_GPS lands inside the table
    pos = jnp.tile(jnp.asarray([500.0, 0.0, 0.0]), (n, 1))
    vel = jnp.zeros((n, 3))
    acc = jnp.zeros((n, 3))
    return gps0, dt, pos, vel, acc


# detector ~at geocentre: diurnal term ~1e-14 s, negligible for these tests
TINY_DET = (1e-6, 1e-6, 1e-6)


def _call(t_rel, *, f0, phi0, aplus, across, alpha=0.7, delta=0.3, fkdot=()):
    gps0, dt, pos, vel, acc = _static_ephemeris()
    return exact_pulsar_polarizations(
        t_rel,
        START_GPS,
        alpha,
        delta,
        f0,
        phi0,
        aplus,
        across,
        TINY_DET,
        gps0,
        dt,
        pos,
        vel,
        acc,
        fkdot=fkdot,
    )


def test_phase_linear_monochromatic():
    """With f1=0 and a static Earth, the phase is φ0 + 2π f0 t exactly."""
    f0, phi0, aplus, across = 10.0, 1.1, 1.0, 0.64
    t = jnp.linspace(0.0, 1000.0, 257, dtype=jnp.float64)
    hp, hc = _call(t, f0=f0, phi0=phi0, aplus=aplus, across=across)
    psi = phi0 + 2.0 * np.pi * f0 * np.asarray(t)
    np.testing.assert_allclose(np.asarray(hp), aplus * np.cos(psi), atol=1e-9)
    np.testing.assert_allclose(np.asarray(hc), across * np.sin(psi), atol=1e-9)


def test_phase_with_spindown():
    """Spindown adds the expected Taylor terms to the phase."""
    f0, f1, f2, phi0 = 12.3, -1.1e-9, 2.0e-18, 0.5
    t = jnp.linspace(0.0, 5000.0, 333, dtype=jnp.float64)
    hp, _ = _call(t, f0=f0, phi0=phi0, aplus=1.0, across=1.0, fkdot=(f1, f2))
    tt = np.asarray(t)
    psi = phi0 + 2 * np.pi * (f0 * tt + 0.5 * f1 * tt**2 + (1 / 6) * f2 * tt**3)
    np.testing.assert_allclose(np.asarray(hp), np.cos(psi), atol=1e-8)


def test_initial_phase_and_amplitudes():
    """At t=0 the polarizations reduce to the initial phase and amplitudes."""
    hp, hc = _call(jnp.zeros(1), f0=7.0, phi0=0.9, aplus=2.0, across=1.5)
    assert float(hp[0]) == pytest.approx(2.0 * np.cos(0.9), abs=1e-9)
    assert float(hc[0]) == pytest.approx(1.5 * np.sin(0.9), abs=1e-9)


def test_polarization_ellipse_invariant():
    """(hp/A+)^2 + (hc/Ax)^2 = 1 holds for any phase (uses a moving Earth)."""
    n, dt = 8, 7200.0
    gps0 = float(START_GPS - dt)
    pos = jnp.tile(jnp.asarray([490.0, 30.0, -10.0]), (n, 1))
    # nonzero, slowly varying velocity -> genuine barycentered phase
    vel = jnp.tile(jnp.asarray([1e-4, -2e-5, 3e-5]), (n, 1))
    acc = jnp.zeros((n, 3))
    t = jnp.linspace(0.0, 3000.0, 200, dtype=jnp.float64)
    aplus, across = 1.3, 0.7
    hp, hc = exact_pulsar_polarizations(
        t,
        START_GPS,
        1.0,
        -0.3,
        50.0,
        0.2,
        aplus,
        across,
        (1e-3, 2e-3, 1e-3),
        gps0,
        dt,
        pos,
        vel,
        acc,
    )
    inv = (np.asarray(hp) / aplus) ** 2 + (np.asarray(hc) / across) ** 2
    np.testing.assert_allclose(inv, 1.0, atol=1e-9)


def test_exact_jit_and_grad():
    """The exact signal is jit-compilable and differentiable in its parameters."""
    t = jnp.linspace(0.0, 1000.0, 128, dtype=jnp.float64)

    @jax.jit
    def hp_sum(f0, alpha):
        hp, _ = _call(t, f0=f0, phi0=0.3, aplus=1.0, across=1.0, alpha=alpha)
        return jnp.sum(hp)

    val = hp_sum(13.0, 0.7)
    assert jnp.isfinite(val)
    gf0, galpha = jax.grad(hp_sum, argnums=(0, 1))(13.0, 0.7)
    assert jnp.isfinite(gf0) and jnp.isfinite(galpha)
    assert abs(float(gf0)) > 0.0  # frequency genuinely affects the waveform


# --- full & binary pulsar signal ---------------------------------------------


def _ephem(n=12, dt=7200.0, pos=(500.0, 30.0, -10.0)):
    """Constant-position synthetic ephemeris (vel = acc = 0)."""
    gps0 = float(START_GPS - dt)
    p = jnp.tile(jnp.asarray(pos), (n, 1))
    z = jnp.zeros((n, 3))
    return gps0, dt, p, z, z


# Earth and Sun synthetic tables (distinct positions so the Earth-Sun vector
# used by the Shapiro term is well defined and nonzero).
_E = _ephem(pos=(500.0, 30.0, -10.0))
_S = _ephem(pos=(2.0, 1.0, 0.5))


def _full(t, *, f_het=0.0, fkdot=(), **p):
    return generate_pulsar_polarizations(
        t,
        START_GPS,
        p["alpha"],
        p["delta"],
        p["f0"],
        p["phi0"],
        p["aplus"],
        p["across"],
        TINY_DET,
        *_E,
        *_S,
        fkdot=fkdot,
        f_heterodyne=f_het,
    )


def test_kepler_residual_zero():
    """The Kepler solver drives the residual to ~machine precision."""
    x = jnp.linspace(0.0, 2 * np.pi, 50)
    for a, b in [(0.05, -0.03), (0.2, 0.1), (-0.15, 0.07)]:
        e = _solve_kepler(x, a, b)
        resid = e + a * jnp.sin(e) + b * (jnp.cos(e) - 1.0) - x
        assert float(jnp.max(jnp.abs(resid))) < 1e-12


def test_heterodyne_is_phase_rotation():
    """Heterodyning rotates the complex polarization by exp(-i 2π f_het t)."""
    t = jnp.linspace(0.0, 1000.0, 257, dtype=jnp.float64)
    base = {
        "alpha": 0.7,
        "delta": 0.3,
        "f0": 20.0,
        "phi0": 0.4,
        "aplus": 1.3,
        "across": 0.7,
    }
    fh = 19.5
    hp0, hc0 = _full(t, f_het=0.0, **base)
    hph, hch = _full(t, f_het=fh, **base)
    z0 = np.asarray(hp0) / base["aplus"] + 1j * np.asarray(hc0) / base["across"]
    zh = np.asarray(hph) / base["aplus"] + 1j * np.asarray(hch) / base["across"]
    expected = z0 * np.exp(-1j * 2 * np.pi * fh * np.asarray(t))
    np.testing.assert_allclose(zh, expected, atol=1e-9)


def test_binary_reduces_to_isolated():
    """As a sin i → 0 the binary signal matches the isolated full signal."""
    t = jnp.linspace(0.0, 2000.0, 401, dtype=jnp.float64)
    common = {
        "alpha": 1.0,
        "delta": -0.3,
        "f0": 30.0,
        "phi0": 0.2,
        "aplus": 1.0,
        "across": 0.6,
    }
    ref = float(START_GPS)
    hp_iso, hc_iso = generate_pulsar_polarizations(
        t,
        START_GPS,
        common["alpha"],
        common["delta"],
        common["f0"],
        common["phi0"],
        common["aplus"],
        common["across"],
        TINY_DET,
        *_E,
        *_S,
        ref_time_ssb=ref,
    )
    hp_bin, hc_bin = generate_binary_pulsar_polarizations(
        t,
        START_GPS,
        common["alpha"],
        common["delta"],
        common["f0"],
        common["phi0"],
        common["aplus"],
        common["across"],
        1e-9,
        0.0,
        5.0 * 3600,
        0.9,
        ref + 1234.0,
        TINY_DET,
        *_E,
        *_S,
        ref_time_ssb=ref,
    )
    # asini ~ 0 -> orbital delay negligible -> phases coincide
    np.testing.assert_allclose(np.asarray(hp_bin), np.asarray(hp_iso), atol=1e-6)
    np.testing.assert_allclose(np.asarray(hc_bin), np.asarray(hc_iso), atol=1e-6)


def test_binary_orbit_changes_signal():
    """A real orbit (asini ~ 1 light-s) measurably modulates the phase."""
    t = jnp.linspace(0.0, 2000.0, 401, dtype=jnp.float64)
    ref = float(START_GPS)
    args = (START_GPS, 1.0, -0.3, 30.0, 0.2, 1.0, 0.6)
    orb = {"ref_time_ssb": ref}
    hp0, _ = generate_binary_pulsar_polarizations(
        t, *args, 1e-9, 0.1, 5.0 * 3600, 0.9, ref + 1234.0, TINY_DET, *_E, *_S, **orb
    )
    hp1, _ = generate_binary_pulsar_polarizations(
        t, *args, 1.2, 0.1, 5.0 * 3600, 0.9, ref + 1234.0, TINY_DET, *_E, *_S, **orb
    )
    assert np.max(np.abs(np.asarray(hp1) - np.asarray(hp0))) > 1e-2


def test_full_jit_and_grad():
    """The full and binary generators are jit-able and differentiable."""
    t = jnp.linspace(0.0, 1000.0, 128, dtype=jnp.float64)

    @jax.jit
    def loss_full(f0):
        hp, _ = _full(t, f0=f0, alpha=0.7, delta=0.3, phi0=0.4, aplus=1.0, across=0.6)
        return jnp.sum(hp)

    assert jnp.isfinite(jax.grad(loss_full)(25.0))

    # Use the DEFAULT ref_time_ssb=None path and a traced sky position: this
    # exercises the internal startTimeSSB computation under jit/grad (a float()
    # cast there would raise ConcretizationTypeError).
    @jax.jit
    def loss_bin(asini, alpha):
        hp, _ = generate_binary_pulsar_polarizations(
            t,
            START_GPS,
            alpha,
            -0.3,
            30.0,
            0.2,
            1.0,
            0.6,
            asini,
            0.15,
            5.0 * 3600,
            0.9,
            float(START_GPS) + 1234.0,
            TINY_DET,
            *_E,
            *_S,
        )
        return jnp.sum(hp)

    g_asini, g_alpha = jax.grad(loss_bin, argnums=(0, 1))(1.2, 1.0)
    assert jnp.isfinite(g_asini) and abs(float(g_asini)) > 0.0
    assert jnp.isfinite(g_alpha)
