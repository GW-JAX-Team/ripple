"""Tests for ripplegw.tools.chirp_time."""

import re

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.tools import time_to_merger
from ripplegw.utils.tidal import get_quadparam_octparam
from ripplegw.waveforms.cbc.Taylor.TaylorF2 import get_PNPhasing_F2

# (m1, m2, s1_z, s2_z): equal-mass, aligned, anti-aligned and single-spin systems.
SYSTEMS = [
    (1.4, 1.4, 0.0, 0.0),
    (36.0, 29.0, 0.6, -0.4),
    (30.0, 25.0, -0.9, 0.9),
    (8.0, 1.4, 0.8, 0.0),
    (1.6, 1.2, 0.05, 0.02),
]


def _taylorf2_time_to_merger(f, m1, m2, s1_z, s2_z):
    """-(1/2pi) dPsi/df of the 2PN-truncated TaylorF2 phasing."""
    coeffs, _ = get_PNPhasing_F2(m1, m2, s1_z, s2_z, 0.0, 0.0)
    M_s = (m1 + m2) * MTSUN

    def phasing(freq):
        v = jnp.cbrt(PI * M_s * freq)
        return sum(coeffs[f"{k}PN"] * v ** (k - 5) for k in range(5))

    return -jax.vmap(jax.grad(phasing))(f) / (2 * PI)


# lalsimulation 6.2.1 SimInspiralTaylorF2ReducedSpinChirpTime(O=4): m1, m2, chi1 = chi2, f (Hz), tau (s).
LAL_REFERENCE = [
    (1.4, 1.4, 0.0, 10.0, 1016.1634806096841),
    (36.0, 29.0, 0.0, 10.0, 5.3542960525536145),
    (8.0, 1.4, 0.0, 20.0, 41.95444944096782),
    (10.0, 10.0, 0.5, 20.0, 6.270753478332759),
    (1.4, 1.4, -0.3, 30.0, 54.41252991775242),
]


@pytest.mark.parametrize("m1,m2,chi,f,tau_lal", LAL_REFERENCE)
def test_matches_lalsimulation_reference_values(m1, m2, chi, f, tau_lal):
    if chi != 0.0:
        assert m1 == m2, "the reduced-spin model is exact only for equal masses"
    M_c, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    tau = time_to_merger(f, M_c, eta, chi, chi)
    assert float(tau) == pytest.approx(tau_lal, rel=1e-12, abs=0.0)


def test_matches_lalsimulation_live():
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    rng = np.random.default_rng(0)
    for _ in range(50):
        m1, m2 = np.sort(rng.uniform(1.0, 60.0, 2))[::-1]
        chi = rng.uniform(-0.99, 0.99)
        if rng.uniform() < 0.5:
            # Unequal masses: only the non-spinning case is exact.
            chi = 0.0
        else:
            m2 = m1
        f = rng.uniform(5.0, 0.01 / (PI * (m1 + m2) * MTSUN))
        eta = m1 * m2 / (m1 + m2) ** 2
        tau_lal = lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(
            f, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, chi * (1 - 76 * eta / 113), 4
        )
        M_c, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
        tau = time_to_merger(f, M_c, eta, chi, chi)
        assert float(tau) == pytest.approx(tau_lal, rel=1e-11, abs=0.0)


def test_point_particle_quadrupole_parameter_is_one():
    # time_to_merger assumes zero tidal deformability means a black hole in TaylorF2.
    assert float(get_quadparam_octparam(0.0)[0]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("m1,m2,s1_z,s2_z", SYSTEMS)
def test_matches_taylorf2_stationary_phase_time(m1, m2, s1_z, s2_z):
    f = jnp.geomspace(5.0, 0.02 / (PI * (m1 + m2) * MTSUN), 50)
    M_c, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    tau = time_to_merger(f, M_c, eta, s1_z, s2_z)
    expected = _taylorf2_time_to_merger(f, m1, m2, s1_z, s2_z)
    np.testing.assert_allclose(tau, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("m1,m2", [(36.0, 29.0), (8.0, 1.4), (1.4, 1.4)])
@pytest.mark.parametrize("spin", [0.5, -0.5])
def test_each_spin_changes_the_result(m1, m2, spin):
    # Guards against one spin being dropped.
    f = jnp.array([20.0])
    M_c, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    non_spinning = time_to_merger(f, M_c, eta)
    primary_only = time_to_merger(f, M_c, eta, spin, 0.0)
    secondary_only = time_to_merger(f, M_c, eta, 0.0, spin)
    assert not np.allclose(primary_only, non_spinning, rtol=1e-8, atol=0.0)
    assert not np.allclose(secondary_only, non_spinning, rtol=1e-8, atol=0.0)


def test_newtonian_limit_at_low_frequency():
    M_c, eta = 1.2, 0.2
    f = jnp.array([1e-4])
    M_c_s = M_c * MTSUN
    newtonian = 5 / 256 * M_c_s * (PI * M_c_s * f) ** (-8 / 3)
    np.testing.assert_allclose(
        time_to_merger(f, M_c, eta, 0.5, 0.5), newtonian, rtol=1e-4, atol=0.0
    )


def test_known_value_binary_neutron_star():
    # Newtonian ~17 min from 10 Hz; the PN corrections add +1.37%.
    M_c, eta = ms_to_Mc_eta(jnp.array([1.4, 1.4]))
    tau = float(time_to_merger(10.0, M_c, eta))
    M_c_s = float(M_c) * MTSUN
    newtonian = 5 / 256 * M_c_s * (PI * M_c_s * 10.0) ** (-8 / 3)
    assert newtonian == pytest.approx(1003.0, rel=2e-3)
    assert tau / newtonian == pytest.approx(1.0137, abs=1e-4)


@pytest.mark.parametrize("mode", [(2, 1), (3, 3), (4, 4), (2, -2), (3, -3)])
def test_mode_maps_to_dominant_mode_frequency(mode):
    M_c, eta = ms_to_Mc_eta(jnp.array([8.0, 1.4]))
    f = jnp.geomspace(10.0, 200.0, 20)
    np.testing.assert_allclose(
        time_to_merger(f, M_c, eta, 0.7, 0.1, mode=mode),
        time_to_merger(2 * f / abs(mode[1]), M_c, eta, 0.7, 0.1),
        rtol=1e-14,
        atol=0.0,
    )


def test_higher_mode_reaches_frequency_earlier_in_time_to_merger():
    # At a fixed frequency the (3, 3) mode is emitted further from merger.
    M_c, eta = ms_to_Mc_eta(jnp.array([30.0, 10.0]))
    f = jnp.array([30.0])
    assert time_to_merger(f, M_c, eta, mode=(3, 3)) > time_to_merger(f, M_c, eta)


@pytest.mark.parametrize("ell,m", [(2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (3, -3)])
def test_ell_does_not_enter(ell, m):
    M_c, eta = ms_to_Mc_eta(jnp.array([8.0, 1.4]))
    f = jnp.geomspace(10.0, 200.0, 20)
    np.testing.assert_array_equal(
        time_to_merger(f, M_c, eta, 0.7, 0.1, mode=(ell, m)),
        time_to_merger(f, M_c, eta, 0.7, 0.1, mode=(4, m)),
    )


@pytest.mark.parametrize(
    "mode,match",
    [
        (2, "tuple (l, m) of two ints"),
        ([2, 2], "tuple (l, m) of two ints"),
        ((2, 2, 2), "tuple (l, m) of two ints"),
        ((2.0, 2), "tuple (l, m) of two ints"),
        (((2, 2), (3, 3)), "tuple (l, m) of two ints"),
        ((2, 0), "l >= 2 and 0 < |m| <= l"),
        ((2, 3), "l >= 2 and 0 < |m| <= l"),
        ((1, 1), "l >= 2 and 0 < |m| <= l"),
    ],
)
def test_invalid_mode_raises(mode, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        time_to_merger(jnp.array([20.0]), 1.2, 0.24, mode=mode)


def test_traced_mode_raises():
    with pytest.raises(ValueError, match="tuple"):
        jax.jit(lambda m: time_to_merger(20.0, 1.2, 0.24, mode=m))(jnp.array([3, 3]))


def _d_tau_d_eta(eta, s1_z, s2_z):
    return jax.grad(lambda e: time_to_merger(20.0, 1.2, e, s1_z, s2_z))(eta)


@pytest.mark.parametrize("spin", [0.0, 0.3, -0.7])
def test_equal_mass_gradient_is_finite_for_equal_spins(spin):
    # For equal spins sqrt(1 - 4 eta) drops out, so the gradient at eta = 1/4 is finite.
    eta, h = 0.25, 1e-7
    grad = _d_tau_d_eta(eta, spin, spin)
    assert np.isfinite(grad)
    assert float(grad) == pytest.approx(
        float(_d_tau_d_eta(eta - 1e-10, spin, spin)), rel=1e-6
    )
    tau = lambda e: float(time_to_merger(20.0, 1.2, e, spin, spin))
    assert float(grad) == pytest.approx((tau(eta) - tau(eta - h)) / h, rel=1e-5)


@pytest.mark.parametrize(
    "s1_z,s2_z", [(0.3, -0.2), (0.0, 0.3), (-0.3, 0.0), (0.4, -0.4)]
)
def test_equal_mass_gradient_diverges_for_unequal_spins(s1_z, s2_z):
    assert np.isfinite(time_to_merger(20.0, 1.2, 0.25, s1_z, s2_z))
    assert np.isinf(_d_tau_d_eta(0.25, s1_z, s2_z))
    # Approaching 1/4 the derivative grows like 1/sqrt(1 - 4 eta).
    near = abs(float(_d_tau_d_eta(0.25 - 1e-8, s1_z, s2_z)))
    far = abs(float(_d_tau_d_eta(0.25 - 1e-4, s1_z, s2_z)))
    assert near > 30 * far


@pytest.mark.parametrize("eta", [0.1, 0.2, 0.24, 0.2499])
@pytest.mark.parametrize(
    "s1_z,s2_z", [(0.0, 0.0), (0.3, 0.3), (-0.6, -0.6), (0.3, -0.2), (0.4, -0.4)]
)
def test_gradient_matches_finite_differences(eta, s1_z, s2_z):
    # Includes equal and zero spins, where a too-wide guard would drop spin derivatives.
    f = 20.0
    params = jnp.array([1.2, eta, s1_z, s2_z])
    tau = lambda p: time_to_merger(f, p[0], p[1], p[2], p[3])
    grad = np.asarray(jax.grad(tau)(params))
    # The eta step shrinks near 1/4, where sqrt(1 - 4 eta) curves sharply.
    steps = [1e-6, min(1e-6, 1e-4 * (1 - 4 * eta)), 1e-6, 1e-6]
    for i, h in enumerate(steps):
        e = np.zeros(4)
        e[i] = h
        fd = (float(tau(params + e)) - float(tau(params - e))) / (2 * h)
        assert grad[i] == pytest.approx(fd, rel=1e-5, abs=1e-12), i


def test_jit_vmap_and_grad():
    f = jnp.geomspace(10.0, 100.0, 8)

    jitted = jax.jit(time_to_merger, static_argnames="mode")
    np.testing.assert_allclose(
        jitted(f, 1.2, 0.24, 0.1, 0.2, mode=(3, 3)),
        time_to_merger(f, 1.2, 0.24, 0.1, 0.2, mode=(3, 3)),
        rtol=1e-14,
        atol=0.0,
    )

    batched = jax.vmap(time_to_merger, in_axes=(None, 0, None))(
        f, jnp.array([1.0, 1.2, 1.4]), 0.24
    )
    assert batched.shape == (3, 8)

    # Heavier chirp mass sweeps faster, so the time to merger decreases.
    d_tau = jax.grad(lambda mc: time_to_merger(f, mc, 0.24).sum())(1.2)
    assert np.isfinite(d_tau) and d_tau < 0
