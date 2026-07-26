"""Integration tests for continuous-wave (pulsar) waveform generation.

Covers the three registered ``source_type="cw"`` models (``ExactPulsarSignal``,
``PulsarSignal``, ``BinaryPulsarSignal``): the dict-param call interface,
output validity, ``parameter_names`` / ``repr``, and ``jit`` / ``grad`` /
``vmap`` compatibility.

Unlike the CBC approximants these take a *time* axis (seconds relative to a
GPS start epoch) and a detector + ephemeris fixed at construction. These tests
do NOT compare against LALSuite -- that is done in cross_validation/. A tiny
synthetic ephemeris (shared with ``tests/helpers/config.py``'s
``default_config``, which every other registry-driven integration test relies
on for zero-arg-construction call sites) is used so this runs in CI with no
LAL/ephemeris-data dependency.

``default_config`` (and so ``tests/integration/test_output_format.py`` /
``test_transforms.py``, which parametrize off the registry generically) always
builds the ``n_spindowns=0`` configuration -- that already exercises the
output-contract/jit/vmap/grad properties once per class. What's dedicated here
instead: the ``n_spindowns=1`` and binary-orbital-parameter configurations
those generic tests never construct, ``f_heterodyne`` (exercised nowhere
else), and the exact ``parameter_names`` ordering / ``repr`` string (the
generic tests only check uniqueness and that the class name appears).
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import ripplegw
from tests.helpers.config import synthetic_ephemeris_files

START_GPS = 1_000_000_000


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
    """Isolated-pulsar params with one spindown (``f1``)."""
    return {
        "alpha": 1.3,
        "delta": -0.5,
        "f0": 12.3,
        "phi0": 1.1,
        "aplus": 1.0,
        "across": 0.64,
        "f1": -1.1e-9,
    }


@pytest.fixture(scope="module")
def binary_params():
    """Binary-pulsar params (no spindown) with orbital elements."""
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


ISOLATED_NAMES = ("alpha", "delta", "f0", "phi0", "aplus", "across", "f1")


class TestExactPulsarSignal:
    @pytest.fixture(scope="class")
    @classmethod
    def model(cls, ephemeris_files):
        """Earth-only exact model with one spindown (Sun ephemeris unused)."""
        earth, _ = ephemeris_files
        return ripplegw.waveform(
            "ExactPulsarSignal",
            detector="H1",
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
            f"ExactPulsarSignal(detector='H1', start_gps={START_GPS}, n_spindowns=1)"
        )


class TestPulsarSignal:
    @pytest.fixture(scope="class")
    @classmethod
    def model(cls, ephemeris_files):
        """Full (isolated) model with one spindown; needs Earth + Sun."""
        earth, sun = ephemeris_files
        return ripplegw.waveform(
            "PulsarSignal",
            detector="H1",
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
            detector="H1",
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
            f"PulsarSignal(detector='H1', start_gps={START_GPS}, "
            f"n_spindowns=1, f_heterodyne=0.0)"
        )


class TestBinaryPulsarSignal:
    @pytest.fixture(scope="class")
    @classmethod
    def model(cls, ephemeris_files):
        """Binary model (no spindown); needs Earth + Sun."""
        earth, sun = ephemeris_files
        return ripplegw.waveform(
            "BinaryPulsarSignal",
            detector="H1",
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
        )

    def test_repr(self, model):
        assert repr(model) == (
            f"BinaryPulsarSignal(detector='H1', start_gps={START_GPS}, n_spindowns=0)"
        )


def test_cw_classes_registered_under_cw_source_type():
    """CW models are constructed via the registry, tagged ``source_type="cw"``."""
    assert sorted(ripplegw.list_waveforms(source_type="cw")) == [
        "BinaryPulsarSignal",
        "ExactPulsarSignal",
        "PulsarSignal",
    ]
