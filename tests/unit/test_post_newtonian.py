"""Tests for ripplegw.post_newtonian.time_to_merger."""

import re

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.post_newtonian import time_to_merger
from ripplegw.utils.tidal import get_quadparam_octparam
from ripplegw.waveforms.cbc.Taylor.TaylorF2 import get_PNPhasing_F2

# (m1, m2, s1_z, s2_z): equal-mass non-spinning, aligned and anti-aligned
# spins, and an asymmetric single-spin NSBH-like system.
SYSTEMS = [
    (1.4, 1.4, 0.0, 0.0),
    (36.0, 29.0, 0.6, -0.4),
    (30.0, 25.0, -0.9, 0.9),
    (8.0, 1.4, 0.8, 0.0),
    (1.6, 1.2, 0.05, 0.02),
]


def _taylorf2_time_to_merger(f, m1, m2, s1_z, s2_z):
    """Stationary-phase time, -(1/2pi) dPsi/df, of the 2PN-truncated TaylorF2 phasing.

    The phasing coefficients come from the LAL port used by the TaylorF2
    waveform, so this is an independent path to the closed-form expression.
    """
    coeffs, _ = get_PNPhasing_F2(m1, m2, s1_z, s2_z, 0.0, 0.0)
    M_s = (m1 + m2) * MTSUN

    def phasing(freq):
        v = jnp.cbrt(PI * M_s * freq)
        return sum(coeffs[f"{k}PN"] * v ** (k - 5) for k in range(5))

    return -jax.vmap(jax.grad(phasing))(f) / (2 * PI)


# External reference: lalsimulation 6.2.1 (lalsuite 7.7.1),
# SimInspiralTaylorF2ReducedSpinChirpTime(f, m1, m2, chi_red, O=4), i.e. the
# 2PN chirp time of the reduced-spin TaylorF2 model, with
# chi_red = chi * (1 - 76 eta / 113) for equal spins. The reduced-spin model is
# exact for non-spinning binaries and for equal masses with equal spins, so
# these values test every 2PN term, including the self-spin part of sigma.
# Columns: m1, m2 (Msun), chi1 = chi2, frequency (Hz), tau (s).
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
    # The comparison below relies on zero tidal deformability meaning a black
    # hole in the TaylorF2 phasing, which is what time_to_merger assumes.
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
    # Each spin must enter on its own: guards against one spin being dropped
    # while a comparison with both spins set still differs from zero spin.
    f = jnp.array([20.0])
    M_c, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    non_spinning = time_to_merger(f, M_c, eta)
    primary_only = time_to_merger(f, M_c, eta, spin, 0.0)
    secondary_only = time_to_merger(f, M_c, eta, 0.0, spin)
    assert not np.allclose(primary_only, non_spinning, rtol=1e-8, atol=0.0)
    assert not np.allclose(secondary_only, non_spinning, rtol=1e-8, atol=0.0)


def test_bare_int_pair_is_one_ell_m_mode():
    f = jnp.geomspace(10.0, 100.0, 5)
    tau = time_to_merger(f, 1.2, 0.24, mode=[3, 2])
    assert tau.shape == f.shape
    np.testing.assert_array_equal(tau, time_to_merger(f, 1.2, 0.24, mode=2))


def test_newtonian_limit_at_low_frequency():
    M_c, eta = 1.2, 0.2
    f = jnp.array([1e-4])
    M_c_s = M_c * MTSUN
    newtonian = 5 / 256 * M_c_s * (PI * M_c_s * f) ** (-8 / 3)
    np.testing.assert_allclose(
        time_to_merger(f, M_c, eta, 0.5, 0.5), newtonian, rtol=1e-4, atol=0.0
    )


def test_known_value_binary_neutron_star():
    # 1.4 + 1.4 Msun non-spinning from 10 Hz: the Newtonian chirp time is the
    # familiar ~17 minutes. With x = (pi M f)^(2/3) = 5.72e-3 the corrections
    # are +2.21% (1PN), -0.87% (1.5PN) and +0.03% (2PN), +1.37% in total.
    M_c, eta = ms_to_Mc_eta(jnp.array([1.4, 1.4]))
    tau = float(time_to_merger(10.0, M_c, eta))
    M_c_s = float(M_c) * MTSUN
    newtonian = 5 / 256 * M_c_s * (PI * M_c_s * 10.0) ** (-8 / 3)
    assert newtonian == pytest.approx(1003.0, rel=2e-3)
    assert tau / newtonian == pytest.approx(1.0137, abs=1e-4)


@pytest.mark.parametrize("mode", [1, 3, 4, -2, -3])
def test_mode_maps_to_dominant_mode_frequency(mode):
    M_c, eta = ms_to_Mc_eta(jnp.array([8.0, 1.4]))
    f = jnp.geomspace(10.0, 200.0, 20)
    np.testing.assert_allclose(
        time_to_merger(f, M_c, eta, 0.7, 0.1, mode=mode),
        time_to_merger(2 * f / abs(mode), M_c, eta, 0.7, 0.1),
        rtol=1e-14,
        atol=0.0,
    )


def test_higher_mode_reaches_frequency_earlier_in_time_to_merger():
    # At a fixed frequency the (3, 3) mode is emitted at a lower orbital
    # frequency than the (2, 2) mode, so further from merger.
    M_c, eta = ms_to_Mc_eta(jnp.array([30.0, 10.0]))
    f = jnp.array([30.0])
    assert time_to_merger(f, M_c, eta, mode=3) > time_to_merger(f, M_c, eta, mode=2)


@pytest.mark.parametrize("ell,m", [(2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (3, -3)])
def test_ell_m_pair_uses_only_m(ell, m):
    M_c, eta = ms_to_Mc_eta(jnp.array([8.0, 1.4]))
    f = jnp.geomspace(10.0, 200.0, 20)
    np.testing.assert_array_equal(
        time_to_merger(f, M_c, eta, 0.7, 0.1, mode=(ell, m)),
        time_to_merger(f, M_c, eta, 0.7, 0.1, mode=m),
    )


@pytest.mark.parametrize(
    "modes",
    [
        [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)],
        ((3, 3),),
        np.array([[2, 1], [2, 2], [3, 2], [3, 3], [4, 3], [4, 4]], dtype=np.int32),
        jnp.array([[2, 1], [2, 2], [3, 2], [3, 3], [4, 3], [4, 4]], dtype=jnp.int32),
    ],
)
def test_several_modes_stack_along_leading_axis(modes):
    M_c, eta = ms_to_Mc_eta(jnp.array([30.0, 10.0]))
    f = jnp.geomspace(10.0, 100.0, 12).reshape(3, 4)
    tau = time_to_merger(f, M_c, eta, 0.4, -0.2, mode=modes)
    pairs = np.asarray(modes).tolist()
    assert tau.shape == (len(pairs), 3, 4)
    for i, (ell, m) in enumerate(pairs):
        np.testing.assert_array_equal(
            tau[i], time_to_merger(f, M_c, eta, 0.4, -0.2, mode=(ell, m))
        )


def test_scalar_frequency_with_several_modes():
    tau = time_to_merger(20.0, 1.2, 0.24, mode=[(2, 2), (3, 3)])
    assert tau.shape == (2,)
    assert tau[1] > tau[0]


@pytest.mark.parametrize(
    "mode,match",
    [
        (0, "non-zero azimuthal"),
        ((2, 0), "non-zero azimuthal"),
        ([(2, 2), (3, 0)], "non-zero azimuthal"),
        ((2, 3), "l >= 2 and |m| <= l"),
        ((1, 1), "l >= 2 and |m| <= l"),
        ([(2, 2), (3, 4)], "l >= 2 and |m| <= l"),
        (2.0, "integers"),
        ([], "integers"),
        ((2, 2, 2), "shape"),
        ([[(2, 2)]], "shape"),
    ],
)
def test_invalid_mode_raises(mode, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        time_to_merger(jnp.array([20.0]), 1.2, 0.24, mode=mode)


def test_traced_mode_raises():
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(lambda m: time_to_merger(20.0, 1.2, 0.24, mode=m))(jnp.array([3, 3]))


def _d_tau_d_eta(eta, s1_z, s2_z):
    return jax.grad(lambda e: time_to_merger(20.0, 1.2, e, s1_z, s2_z))(eta)


@pytest.mark.parametrize("spin", [0.0, 0.3, -0.7])
def test_equal_mass_gradient_is_finite_for_equal_spins(spin):
    # delta = sqrt(1 - 4 eta) has an infinite derivative at eta = 1/4, but for
    # equal spins it drops out, so the limit is finite: compare with the
    # gradient just below 1/4 and with a one-sided finite difference.
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
    # Every parameter, including equal (and zero) spins, where a guard that
    # is too wide would silently drop part of the spin derivative.
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
        jitted(f, 1.2, 0.24, 0.1, 0.2, mode=3),
        time_to_merger(f, 1.2, 0.24, 0.1, 0.2, mode=3),
        rtol=1e-14,
        atol=0.0,
    )
    multi = ((2, 2), (3, 3))
    np.testing.assert_allclose(
        jax.jit(time_to_merger, static_argnames="mode")(f, 1.2, 0.24, mode=multi),
        time_to_merger(f, 1.2, 0.24, mode=multi),
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
