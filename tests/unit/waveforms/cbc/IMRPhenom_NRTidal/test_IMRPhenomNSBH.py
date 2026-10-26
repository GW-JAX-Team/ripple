"""IMRPhenomNSBH helpers: the tidal root ``xi_tide`` and the Kerr ISCO radius near zero spin."""

import importlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

nsbh = importlib.import_module("ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomNSBH")

# Corners and interior of the supported domain: q in [1, 100], BH spin near
# +-1, and NS compactness from very soft to very stiff (mu = q C).
CORNERS = list(
    itertools.product(
        (1.0, 2.0, 10.0, 100.0),
        (-0.999, -0.5, 0.0, 0.5, 0.999),
        (0.05, 0.15, 0.3),
    )
)


def _reference_xi(q, a, mu):
    """Square of the largest positive real root of LAL's degree-10 polynomial."""
    # Descending coefficients of z^10 - 3 mu z^8 + 2 a mu^1.5 z^7 - 3 q z^4
    # + 6 q mu z^2 - 3 q mu^2 a^2 (Eq. 8 of 1509.00512).
    coeffs = [1.0, 0.0, -3.0 * mu, 2.0 * a * mu**1.5, 0.0, 0.0, -3.0 * q, 0.0]
    coeffs += [6.0 * q * mu, 0.0, -3.0 * q * mu**2 * a**2]
    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) <= 1e-7 * np.abs(roots)].real
    return np.max(real[real > 0.0]) ** 2


@pytest.mark.parametrize(("q", "a", "C"), CORNERS)
def test_xi_tide_matches_polynomial_roots(q, a, C):
    # Pins the bisection: too few halvings leave the bracket wider than rtol.
    expected = _reference_xi(q, a, q * C)
    actual = float(jax.jit(nsbh.xi_tide)(q, a, q * C))
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=1e-12)


@pytest.mark.parametrize(("q", "a", "C"), CORNERS[::7])
def test_xi_tide_gradient_matches_finite_differences(q, a, C):
    # Pins the implicit-function step, the only path for derivatives through the
    # stopped bisection; compare with central differences of the reference.
    # The tolerance is the differences' own error (up to ~1e-6) with margin.
    theta = np.array([q, a, q * C])
    actual = np.asarray(jax.grad(lambda t: nsbh.xi_tide(*t))(theta))
    expected = np.empty(3)
    for i in range(3):
        step = 1e-6 * max(abs(theta[i]), 1.0)
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        expected[i] = (_reference_xi(*up) - _reference_xi(*down)) / (2.0 * step)
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=1e-5)


def test_kerr_isco_radius_at_zero_spin():
    # r = 6 - (4 sqrt(6) / 3) a + O(a^2); the closed form gives 0/0 for d/da here.
    value, grad = jax.value_and_grad(nsbh.kerr_isco_radius)(0.0)
    np.testing.assert_allclose(value, 6.0, atol=0.0, rtol=1e-15)
    np.testing.assert_allclose(grad, -4.0 * np.sqrt(6.0) / 3.0, atol=0.0, rtol=1e-15)


@pytest.mark.parametrize("a", [1e-2, -1e-2])
def test_kerr_isco_radius_branches_agree_at_switch(a):
    # Series just inside |a| < 1e-2 against the closed form just outside.
    fn = jax.value_and_grad(nsbh.kerr_isco_radius)
    inside, outside = fn(np.nextafter(a, 0.0)), fn(a)
    np.testing.assert_allclose(inside[0], outside[0], atol=0.0, rtol=1e-14)
    np.testing.assert_allclose(inside[1], outside[1], atol=0.0, rtol=1e-12)


@pytest.mark.parametrize(("a", "expected"), [(1.0, 1.0), (-1.0, 9.0), (0.5, 4.2330025)])
def test_kerr_isco_radius_closed_form(a, expected):
    np.testing.assert_allclose(nsbh.kerr_isco_radius(a), expected, rtol=1e-7)


def test_waveform_gradient_finite_at_zero_bh_spin():
    # Mc, eta, chi_BH = 0, chi_NS, lambda_2, d_L, t_c, phase_c
    theta = jnp.array([3.971, 0.06693, 0.0, -0.01965, 1321.7, 100.0, 0.0, 0.3])
    f = jnp.geomspace(10.0, 4000.0, 128)

    def loss(t):
        return jnp.sum(jnp.abs(nsbh.gen_IMRPhenomNSBH(f, t, 20.0)) ** 2)

    assert np.all(np.isfinite(jax.grad(loss)(theta)))
    assert np.all(np.isfinite(jax.jacfwd(loss)(theta)))
