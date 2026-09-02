"""Integration tests for continuous-wave (pulsar) waveform generation.

Covers the three registered ``source_type="cw"`` models (``ExactPulsarSignal``,
``PulsarSignal``, ``BinaryPulsarSignal``): the dict-param call interface,
output validity, ``parameter_names`` / ``repr``, and ``jit`` / ``grad`` /
``vmap`` compatibility.

Unlike the CBC approximants these take a *time* axis (seconds relative to a
GPS start epoch) and an ephemeris fixed at construction; the observing site's
location is a per-call ``params`` entry (``site_x``/``site_y``/``site_z``),
not constructor state, precisely so a single model instance can be
``jax.vmap``ed over multiple sites. These tests do NOT compare against
LALSuite -- that is done in cross_validation/. A tiny synthetic ephemeris
(shared with ``tests/helpers/config.py``'s ``default_config``, which every
other registry-driven integration test relies on for zero-arg-construction
call sites) is used so this runs in CI with no LAL/ephemeris-data dependency.

``default_config`` (and so ``tests/integration/test_output_format.py`` /
``test_jax.py``, which parametrize off the registry generically) always
builds the ``n_spindowns=0`` configuration -- that already exercises the
output format, ``jit``, ``vmap``, and ``grad`` behaviour once per class. What's dedicated here
instead: the ``n_spindowns=1`` and binary-orbital-parameter configurations
those generic tests never construct, ``f_heterodyne`` (exercised nowhere
else), the exact ``parameter_names`` ordering / ``repr`` string (the generic
tests only check uniqueness and that the class name appears), and
``jax.vmap`` over genuinely different observing sites in one call (the
generic ``test_vmap_over_params`` broadcasts one identical value to every
batch slot, so it never exercises this).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import ripplegw
from tests.helpers.config import H1_LOCATION, synthetic_ephemeris_files

START_GPS = 1_000_000_000

# Real LALDetectors.h vertex locations (metres), used only as realistic,
# well-separated test sites -- ripple has no concept of "H1"/"L1"/"V1" itself.
L1_LOCATION = (-7.42760447238e04, -5.49628371971e06, 3.22425701744e06)
V1_LOCATION = (4.54637409900e06, 8.42989697626e05, 4.37857696241e06)


def _site_params(location):
    """``{"site_x", "site_y", "site_z"}`` from a geocentric location tuple."""
    x, y, z = location
    return {"site_x": x, "site_y": y, "site_z": z}


@pytest.fixture(scope="module")
def ephemeris_files():
    """``(earth, sun)`` synthetic ephemeris paths spanning the observation."""
    return synthetic_ephemeris_files()


@pytest.fixture(scope="module")
def time_grid():
    """Detector-frame sample times (s) relative to START_GPS: 1024 pts @ 64 Hz."""
    fs, duration = 64.0, 16.0
    return jnp.arange(0.0, duration, 1.0 / fs)


@pytest.fixture(scope="module")
def isolated_params():
    """Isolated-pulsar params with one spindown (``f1``), at ``H1_LOCATION``."""
    return {
        "alpha": 1.3,
        "delta": -0.5,
        "f0": 12.3,
        "phi0": 1.1,
        "aplus": 1.0,
        "across": 0.64,
        "f1": -1.1e-9,
        **_site_params(H1_LOCATION),
    }


@pytest.fixture(scope="module")
def binary_params():
    """Binary-pulsar params (no spindown) with orbital elements, at ``H1_LOCATION``."""
    return {
        "alpha": 1.3,
        "delta": -0.5,
        "f0": 12.3,
        "phi0": 1.1,
        "aplus": 1.0,
        "across": 0.64,
        "asini": 1.44,
        "ecc": 0.18,
        "period": 6.3 * 3600.0,
        "argp": 1.05,
        "tp_ssb": float(START_GPS) + 1234.0,
        **_site_params(H1_LOCATION),
    }


def assert_cw_valid(out, t):
    """Assert CW dict output ``{"p": h+, "c": hx}`` is finite, real, right shape."""
    hp, hc = out["p"], out["c"]
    assert hp.shape == t.shape, f"h+ shape {hp.shape} != t shape {t.shape}"
    assert hc.shape == t.shape, f"hx shape {hc.shape} != t shape {t.shape}"
    assert jnp.all(jnp.isfinite(hp)), "h+ contains NaN or Inf"
    assert jnp.all(jnp.isfinite(hc)), "hx contains NaN or Inf"
    assert not jnp.iscomplexobj(hp), "h+ should be real-valued"
    assert not jnp.iscomplexobj(hc), "hx should be real-valued"


def _batch(params, b):
    """Expand a scalar-valued param dict to a batched dict of 1-D arrays."""
    return {k: jnp.full(b, float(v)) for k, v in params.items()}


def _assert_vmap_over_distinct_sites(model, time_grid, params):
    """``jax.vmap`` over genuinely different sites must match an explicit
    per-site Python loop (catches a coordinate mix-up that "finite and
    differs between sites" alone would miss), and outputs must actually
    differ between sites (not a silent broadcast)."""
    sites = jnp.array([H1_LOCATION, L1_LOCATION, V1_LOCATION])

    def call_site(site):
        p = {**params, "site_x": site[0], "site_y": site[1], "site_z": site[2]}
        return model(time_grid, p)

    vmapped = jax.vmap(call_site)(sites)
    looped_p = jnp.stack([call_site(s)["p"] for s in sites])
    looped_c = jnp.stack([call_site(s)["c"] for s in sites])

    assert jnp.all(jnp.isfinite(vmapped["p"]))
    assert jnp.all(jnp.isfinite(vmapped["c"]))
    np.testing.assert_allclose(vmapped["p"], looped_p, atol=1e-12)
    np.testing.assert_allclose(vmapped["c"], looped_c, atol=1e-12)
    assert not jnp.allclose(vmapped["p"][0], vmapped["p"][1])
    assert not jnp.allclose(vmapped["p"][0], vmapped["p"][2])


ISOLATED_NAMES = (
    "alpha",
    "delta",
    "f0",
    "phi0",
    "aplus",
    "across",
    "site_x",
    "site_y",
    "site_z",
    "f1",
)


class TestExactPulsarSignal:
    @pytest.fixture(scope="class")
    @classmethod
    def model(cls, ephemeris_files):
        """Earth-only exact model with one spindown (Sun ephemeris unused)."""
        earth, _ = ephemeris_files
        return ripplegw.waveform(
            "ExactPulsarSignal",
            earth_ephemeris_file=earth,
            start_gps=START_GPS,
            n_spindowns=1,
        )

    def test_basic(self, model, time_grid, isolated_params):
        assert_cw_valid(model(time_grid, isolated_params), time_grid)

    def test_jit(self, model, time_grid, isolated_params):
        out = jax.jit(model)(time_grid, isolated_params)
        assert_cw_valid(out, time_grid)

    def test_vmap(self, model, time_grid, isolated_params):
        b = 4
        out = jax.vmap(lambda p: model(time_grid, p))(_batch(isolated_params, b))
        assert out["p"].shape == (b, time_grid.shape[0])
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_vmap_over_distinct_sites(self, model, time_grid, isolated_params):
        _assert_vmap_over_distinct_sites(model, time_grid, isolated_params)

    def test_grad(self, model, time_grid, isolated_params):
        def loss(f0):
            return jnp.sum(model(time_grid, {**isolated_params, "f0": f0})["p"])

        g = jax.grad(loss)(isolated_params["f0"])
        assert jnp.isfinite(g)
        assert abs(float(g)) > 0.0  # frequency genuinely affects the waveform

    def test_parameter_names(self, model):
        assert model.parameter_names == ISOLATED_NAMES

    def test_repr(self, model):
        assert repr(model) == (
            f"ExactPulsarSignal(start_gps={START_GPS}, n_spindowns=1)"
        )


class TestPulsarSignal:
    @pytest.fixture(scope="class")
    @classmethod
    def model(cls, ephemeris_files):
        """Full (isolated) model with one spindown; needs Earth + Sun."""
        earth, sun = ephemeris_files
        return ripplegw.waveform(
            "PulsarSignal",
            earth_ephemeris_file=earth,
            sun_ephemeris_file=sun,
            start_gps=START_GPS,
            n_spindowns=1,
        )

    def test_basic(self, model, time_grid, isolated_params):
        assert_cw_valid(model(time_grid, isolated_params), time_grid)

    def test_jit(self, model, time_grid, isolated_params):
        out = jax.jit(model)(time_grid, isolated_params)
        assert_cw_valid(out, time_grid)

    def test_vmap(self, model, time_grid, isolated_params):
        b = 4
        out = jax.vmap(lambda p: model(time_grid, p))(_batch(isolated_params, b))
        assert out["p"].shape == (b, time_grid.shape[0])
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_vmap_over_distinct_sites(self, model, time_grid, isolated_params):
        _assert_vmap_over_distinct_sites(model, time_grid, isolated_params)

    def test_grad(self, model, time_grid, isolated_params):
        def loss(f0):
            return jnp.sum(model(time_grid, {**isolated_params, "f0": f0})["p"])

        g = jax.grad(loss)(isolated_params["f0"])
        assert jnp.isfinite(g)
        assert abs(float(g)) > 0.0

    def test_heterodyne(self, ephemeris_files, time_grid, isolated_params):
        """A nonzero heterodyne frequency still yields a valid (real) signal."""
        earth, sun = ephemeris_files
        model = ripplegw.waveform(
            "PulsarSignal",
            earth_ephemeris_file=earth,
            sun_ephemeris_file=sun,
            start_gps=START_GPS,
            n_spindowns=1,
            f_heterodyne=11.0,
        )
        assert_cw_valid(model(time_grid, isolated_params), time_grid)

    def test_parameter_names(self, model):
        assert model.parameter_names == ISOLATED_NAMES

    def test_repr(self, model):
        assert repr(model) == (
            f"PulsarSignal(start_gps={START_GPS}, n_spindowns=1, f_heterodyne=0.0)"
        )


class TestBinaryPulsarSignal:
    @pytest.fixture(scope="class")
    @classmethod
    def model(cls, ephemeris_files):
        """Binary model (no spindown); needs Earth + Sun."""
        earth, sun = ephemeris_files
        return ripplegw.waveform(
            "BinaryPulsarSignal",
            earth_ephemeris_file=earth,
            sun_ephemeris_file=sun,
            start_gps=START_GPS,
        )

    def test_basic(self, model, time_grid, binary_params):
        assert_cw_valid(model(time_grid, binary_params), time_grid)

    def test_jit(self, model, time_grid, binary_params):
        out = jax.jit(model)(time_grid, binary_params)
        assert_cw_valid(out, time_grid)

    def test_vmap(self, model, time_grid, binary_params):
        b = 4
        out = jax.vmap(lambda p: model(time_grid, p))(_batch(binary_params, b))
        assert out["p"].shape == (b, time_grid.shape[0])
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_vmap_over_distinct_sites(self, model, time_grid, binary_params):
        _assert_vmap_over_distinct_sites(model, time_grid, binary_params)

    def test_grad(self, model, time_grid, binary_params):
        """Gradient through the orbital modulation (w.r.t. asini)."""

        def loss(asini):
            return jnp.sum(model(time_grid, {**binary_params, "asini": asini})["p"])

        g = jax.grad(loss)(binary_params["asini"])
        assert jnp.isfinite(g)
        assert abs(float(g)) > 0.0  # the orbit genuinely modulates the phase

    def test_parameter_names(self, model):
        assert model.parameter_names == (
            "alpha",
            "delta",
            "f0",
            "phi0",
            "aplus",
            "across",
            "asini",
            "ecc",
            "period",
            "argp",
            "tp_ssb",
            "site_x",
            "site_y",
            "site_z",
        )

    def test_repr(self, model):
        assert repr(model) == (
            f"BinaryPulsarSignal(start_gps={START_GPS}, n_spindowns=0)"
        )


def test_cw_classes_registered_under_cw_source_type():
    """CW models are constructed via the registry, tagged ``source_type="cw"``."""
    assert sorted(ripplegw.list_waveforms(source_type="cw")) == [
        "BinaryPulsarSignal",
        "ExactPulsarSignal",
        "PulsarSignal",
    ]
