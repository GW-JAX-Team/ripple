"""Unit tests for the exact pulsar-signal polarizations.

These tests use a *synthetic static* Earth ephemeris (zero velocity and
acceleration) together with a detector placed essentially at the geocentre, so
the geometric delay ``dT`` is constant. With the default reference time the
barycentered time then reduces to the plain elapsed time ``τ = t``, giving a
closed-form phase against which the implementation can be checked without any
LAL dependency.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripplegw.cw.barycenter import source_unit_vector
from ripplegw.cw.pulsar_signal import exact_pulsar_polarizations

START_GPS = 1_000_000_000


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
        t_rel, START_GPS, alpha, delta, f0, phi0, aplus, across, TINY_DET,
        gps0, dt, pos, vel, acc, fkdot=fkdot,
    )


def test_source_unit_vector():
    """The source unit vector matches the analytic expression and is a unit."""
    alpha, delta = 1.1, -0.4
    n = np.asarray(source_unit_vector(alpha, delta))
    expected = np.array(
        [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)]
    )
    np.testing.assert_allclose(n, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-14)


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
        t, START_GPS, 1.0, -0.3, 50.0, 0.2, aplus, across, (1e-3, 2e-3, 1e-3),
        gps0, dt, pos, vel, acc,
    )
    inv = (np.asarray(hp) / aplus) ** 2 + (np.asarray(hc) / across) ** 2
    np.testing.assert_allclose(inv, 1.0, atol=1e-9)


def test_jit_and_grad():
    """The signal is jit-compilable and differentiable in its parameters."""
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
