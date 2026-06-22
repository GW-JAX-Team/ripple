"""Integration tests for continuous-wave (pulsar) waveform generation.

Covers the callable ``Waveform`` classes exported from ``ripplegw.cw``
(``ExactPulsarSignal``, ``PulsarSignal``, ``BinaryPulsarSignal``): the
dict-param call interface, output validity, ``parameter_names`` / ``repr``, and
``jit`` / ``grad`` / ``vmap`` compatibility.

Unlike the CBC approximants these take a *time* axis (seconds relative to a GPS
start epoch) and a detector + ephemeris fixed at construction, and they are not
part of ``waveform_preset``.  These tests do NOT compare against LALSuite — that
is done in cross_validation/.  A tiny synthetic ephemeris (written to a tmp
file) is used so they run in CI with no LAL/ephemeris-data dependency.
"""

import jax
import jax.numpy as jnp
import pytest

from ripplegw import waveform_preset
from ripplegw.cw import BinaryPulsarSignal, ExactPulsarSignal, PulsarSignal

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

START_GPS = 1_000_000_000


# ============================================================================
# Synthetic ephemeris (no LAL / data dependency)
# ============================================================================


def _write_ephemeris(path, pos, vel, acc, *, gps0, dt=7200.0, n=4):
    """Write a tiny synthetic LALPulsar-format ephemeris covering the span.

    Constant pos/vel/acc rows on a uniform GPS grid — enough for the barycenter
    interpolation to produce a finite, well-defined signal without real data.
    Layout matches ``XLALReadEphemerisFile``: a ``gpsYr dt nEntries`` header
    (the first token is ignored by the reader) then ``nEntries`` rows of
    ``gps  pos(3)  vel(3)  acc(3)``.
    """
    lines = [f"{gps0} {dt} {n}"]
    for i in range(n):
        gps = gps0 + i * dt
        row = [gps, *pos, *vel, *acc]
        lines.append(" ".join(repr(float(v)) for v in row))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def ephemeris_files(tmp_path_factory):
    """``(earth, sun)`` synthetic ephemeris paths spanning the observation."""
    d = tmp_path_factory.mktemp("cw_ephem")
    gps0 = START_GPS - 7200.0  # so START_GPS lands inside the table
    earth = _write_ephemeris(
        d / "earth-synth.dat",
        pos=(490.0, 30.0, -10.0),  # ~1 AU in light-seconds
        vel=(1e-4, -2e-5, 3e-5),  # ~Earth orbital v/c
        acc=(0.0, 0.0, 0.0),
        gps0=gps0,
    )
    sun = _write_ephemeris(
        d / "sun-synth.dat",
        pos=(2.0, 1.0, 0.5),  # distinct from Earth → Earth-Sun vector well defined
        vel=(0.0, 0.0, 0.0),
        acc=(0.0, 0.0, 0.0),
        gps0=gps0,
    )
    return earth, sun


# ============================================================================
# Fixtures — grids and params
# ============================================================================


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


# ============================================================================
# Helpers
# ============================================================================


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


# ============================================================================
# Tests per class
# ============================================================================


class TestExactPulsarSignal:
    @pytest.fixture(scope="class")
    @classmethod
    def model(cls, ephemeris_files):
        """Earth-only exact model with one spindown (Sun ephemeris unused)."""
        earth, _ = ephemeris_files
        return ExactPulsarSignal("H1", earth, start_gps=START_GPS, n_spindowns=1)

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
        return PulsarSignal("H1", earth, sun, start_gps=START_GPS, n_spindowns=1)

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
        model = PulsarSignal(
            "H1", earth, sun, start_gps=START_GPS, n_spindowns=1, f_heterodyne=11.0
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
        return BinaryPulsarSignal("H1", earth, sun, start_gps=START_GPS)

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


def test_cw_classes_not_in_waveform_preset():
    """CW models are exposed via ``ripplegw.cw``, not the CBC ``waveform_preset``."""
    for name in ("ExactPulsarSignal", "PulsarSignal", "BinaryPulsarSignal"):
        assert name not in waveform_preset
