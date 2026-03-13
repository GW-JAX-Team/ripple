"""Integration tests for waveform generation.

Each test class covers one approximant and tests both the low-level
``gen_<approximant>_hphc`` function (array params) and the top-level
callable class exported from ``ripplegw`` (dict params).

These tests do NOT compare against LALSuite - that's done in cross_validation/.
"""

import jax
import jax.numpy as jnp
import pytest

from ripplegw import (
    IMRPhenomD,
    IMRPhenomD_NRTidalv2,
    IMRPhenomPv2,
    IMRPhenomXAS,
    IMRPhenomXAS_NRTidalv3,
    SineGaussian,
    TaylorF2,
    waveform_preset,
)
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Fixtures — array params (for low-level gen function tests)
# ============================================================================


@pytest.fixture
def bbh_aligned_params():
    """Array params for aligned-spin BBH: [Mc, eta, chi1, chi2, d, tc, phic, iota]."""
    m1, m2 = 30.0, 25.0
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return jnp.array([Mc, eta, 0.5, -0.3, 400.0, 0.0, 0.5, 0.8])


@pytest.fixture
def bns_tidal_params():
    """Array params for BNS (lambda-tilde convention): [Mc, eta, chi1, chi2, lt, dlt, d, tc, phic, iota]."""
    m1, m2 = 1.4, 1.3
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
        jnp.array([500.0, 400.0, m1, m2])
    )
    return jnp.array([Mc, eta, 0.05, -0.02, lambda_tilde, delta_lambda_tilde, 100.0, 0.0, 0.5, 0.8])


@pytest.fixture
def bbh_precessing_params():
    """Array params for IMRPhenomPv2: [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, d, tc, phic, iota]."""
    m1, m2 = 30.0, 25.0
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return jnp.array([Mc, eta, 0.1, 0.2, 0.3, -0.1, 0.15, -0.2, 400.0, 0.0, 0.5, 0.8])


@pytest.fixture
def bbh_xphm_params():
    """Scalar params for IMRPhenomXPHM generate_xphm positional call."""
    return 50.0, 30.0, 0.2, 0.1, -0.3, -0.1, 0.3, 0.1, 500.0, 0.8, 0.5


@pytest.fixture
def sinegaussian_params():
    """Array params for SineGaussian: [quality, frequency, hrss, phase, eccentricity]."""
    return jnp.array([10.0, 100.0, 1e-21, 0.5, 0.3])


# ============================================================================
# Fixtures — dict params (for top-level approximant class tests)
# ============================================================================


@pytest.fixture
def bbh_aligned_dict():
    """Dict params for aligned-spin BBH approximant classes."""
    m1, m2 = 30.0, 25.0
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {"M_c": float(Mc), "eta": float(eta), "s1_z": 0.5, "s2_z": -0.3,
            "d_L": 400.0, "phase_c": 0.5, "iota": 0.8}


@pytest.fixture
def bbh_precessing_dict():
    """Dict params for IMRPhenomPv2 approximant class."""
    m1, m2 = 30.0, 25.0
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {"M_c": float(Mc), "eta": float(eta),
            "s1_x": 0.1, "s1_y": 0.2, "s1_z": 0.3,
            "s2_x": -0.1, "s2_y": 0.15, "s2_z": -0.2,
            "d_L": 400.0, "phase_c": 0.5, "iota": 0.8}


@pytest.fixture
def bns_tidal_dict():
    """Dict params for tidal approximant classes (lambda_1/lambda_2)."""
    m1, m2 = 1.4, 1.3
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {"M_c": float(Mc), "eta": float(eta), "s1_z": 0.05, "s2_z": -0.02,
            "lambda_1": 500.0, "lambda_2": 400.0, "d_L": 100.0, "phase_c": 0.5, "iota": 0.8}


@pytest.fixture
def bns_tidal_tilde_dict():
    """Dict params for tidal approximant classes (lambda_tilde/delta_lambda_tilde)."""
    m1, m2 = 1.4, 1.3
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([500.0, 400.0, m1, m2]))
    return {"M_c": float(Mc), "eta": float(eta), "s1_z": 0.05, "s2_z": -0.02,
            "lambda_tilde": float(lt), "delta_lambda_tilde": float(dlt),
            "d_L": 100.0, "phase_c": 0.5, "iota": 0.8}


@pytest.fixture
def sinegaussian_dict():
    """Dict params for SineGaussian approximant class."""
    return {"Q": 10.0, "f_0": 100.0, "hrss": 1e-21, "phase": 0.5, "e": 0.3}


# ============================================================================
# Fixtures — grids
# ============================================================================


@pytest.fixture
def test_time_grid():
    fs_sampling, duration = 4096.0, 1.0
    return jnp.arange(-duration / 2, duration / 2, 1 / fs_sampling)


@pytest.fixture
def test_freq_grid():
    f_l, f_u, f_sampling, T = 20.0, 1024.0, 2048.0, 16.0
    delta_t = 1 / f_sampling
    freqs = jnp.fft.rfftfreq(int(round(T / delta_t)), delta_t)
    return freqs[(freqs > f_l) & (freqs < f_u)]


# ============================================================================
# Helpers
# ============================================================================


def assert_fd_valid(hp, hc, fs):
    """Assert frequency-domain (hp, hc) are finite and complex."""
    assert hp.shape == fs.shape, f"hp shape {hp.shape} != fs shape {fs.shape}"
    assert hc.shape == fs.shape, f"hc shape {hc.shape} != fs shape {fs.shape}"
    assert jnp.all(jnp.isfinite(hp)), "hp contains NaN or Inf"
    assert jnp.all(jnp.isfinite(hc)), "hc contains NaN or Inf"
    assert jnp.iscomplexobj(hp), "hp is not complex-valued"
    assert jnp.iscomplexobj(hc), "hc is not complex-valued"


def assert_td_valid(hp, hc, t):
    """Assert time-domain (hp, hc) are finite and real."""
    assert hp.shape == t.shape, f"hp shape {hp.shape} != t shape {t.shape}"
    assert hc.shape == t.shape, f"hc shape {hc.shape} != t shape {t.shape}"
    assert jnp.all(jnp.isfinite(hp)), "hp contains NaN or Inf"
    assert jnp.all(jnp.isfinite(hc)), "hc contains NaN or Inf"
    assert not jnp.iscomplexobj(hp), "hp should be real-valued"
    assert not jnp.iscomplexobj(hc), "hc should be real-valued"


def assert_approx_fd_valid(output, fs):
    """Assert approximant dict output {"p": hp, "c": hc} is finite and complex."""
    assert_fd_valid(output["p"], output["c"], fs)


def assert_approx_td_valid(output, t):
    """Assert approximant dict output {"p": hp, "c": hc} is finite and real."""
    assert_td_valid(output["p"], output["c"], t)


def batch_dict(params, batch_size):
    """Expand a scalar-valued param dict to a batched dict of 1-D arrays."""
    return {k: jnp.full(batch_size, float(v)) for k, v in params.items()}


# ============================================================================
# Tests per approximant
# ============================================================================


class TestIMRPhenomD:
    # --- low-level gen function ---
    def test_gen_basic(self, test_freq_grid, bbh_aligned_params):
        from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
        hp, hc = gen_IMRPhenomD_hphc(test_freq_grid, bbh_aligned_params, 20.0)
        assert_fd_valid(hp, hc, test_freq_grid)

    def test_gen_jit(self, test_freq_grid, bbh_aligned_params):
        from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
        fs, f_ref = test_freq_grid, 20.0
        hp, hc = jax.jit(lambda t: gen_IMRPhenomD_hphc(fs, t, f_ref))(bbh_aligned_params)
        assert_fd_valid(hp, hc, fs)

    def test_gen_vmap(self, test_freq_grid, bbh_aligned_params):
        from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
        fs, f_ref, batch_size = test_freq_grid, 20.0, 5
        batch = jnp.tile(bbh_aligned_params, (batch_size, 1))
        hp_b, hc_b = jax.vmap(lambda t: gen_IMRPhenomD_hphc(fs, t, f_ref))(batch)
        assert hp_b.shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(hp_b))

    def test_gen_grad(self, test_freq_grid, bbh_aligned_params):
        from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
        fs, f_ref = test_freq_grid, 20.0

        def loss(theta):
            hp, _ = gen_IMRPhenomD_hphc(fs, theta, f_ref)
            return jnp.sum(jnp.abs(hp) ** 2)

        grad = jax.grad(loss)(bbh_aligned_params)
        assert grad.shape == bbh_aligned_params.shape
        assert jnp.all(jnp.isfinite(grad))

    # --- top-level approximant class ---
    def test_basic(self, test_freq_grid, bbh_aligned_dict):
        output = IMRPhenomD(f_ref=20.0)(test_freq_grid, bbh_aligned_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_jit(self, test_freq_grid, bbh_aligned_dict):
        model = IMRPhenomD(f_ref=20.0)
        output = jax.jit(model)(test_freq_grid, bbh_aligned_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_vmap(self, test_freq_grid, bbh_aligned_dict):
        model = IMRPhenomD(f_ref=20.0)
        fs, batch_size = test_freq_grid, 4
        out = jax.vmap(lambda p: model(fs, p))(batch_dict(bbh_aligned_dict, batch_size))
        assert out["p"].shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_repr(self):
        assert repr(IMRPhenomD(f_ref=20.0)) == "IMRPhenomD(f_ref=20.0)"

    def test_in_waveform_preset(self):
        assert "IMRPhenomD" in waveform_preset
        assert isinstance(waveform_preset["IMRPhenomD"](f_ref=20.0), IMRPhenomD)


class TestIMRPhenomXAS:
    # --- low-level gen function ---
    def test_gen_basic(self, test_freq_grid, bbh_aligned_params):
        from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
        hp, hc = gen_IMRPhenomXAS_hphc(test_freq_grid, bbh_aligned_params, 20.0)
        assert_fd_valid(hp, hc, test_freq_grid)

    def test_gen_jit(self, test_freq_grid, bbh_aligned_params):
        from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
        fs, f_ref = test_freq_grid, 20.0
        hp, hc = jax.jit(lambda t: gen_IMRPhenomXAS_hphc(fs, t, f_ref))(bbh_aligned_params)
        assert_fd_valid(hp, hc, fs)

    def test_gen_vmap(self, test_freq_grid, bbh_aligned_params):
        from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
        fs, f_ref, batch_size = test_freq_grid, 20.0, 5
        batch = jnp.tile(bbh_aligned_params, (batch_size, 1))
        hp_b, hc_b = jax.vmap(lambda t: gen_IMRPhenomXAS_hphc(fs, t, f_ref))(batch)
        assert hp_b.shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(hp_b))

    # --- top-level approximant class ---
    def test_basic(self, test_freq_grid, bbh_aligned_dict):
        output = IMRPhenomXAS(f_ref=20.0)(test_freq_grid, bbh_aligned_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_jit(self, test_freq_grid, bbh_aligned_dict):
        model = IMRPhenomXAS(f_ref=20.0)
        output = jax.jit(model)(test_freq_grid, bbh_aligned_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_vmap(self, test_freq_grid, bbh_aligned_dict):
        model = IMRPhenomXAS(f_ref=20.0)
        fs, batch_size = test_freq_grid, 4
        out = jax.vmap(lambda p: model(fs, p))(batch_dict(bbh_aligned_dict, batch_size))
        assert out["p"].shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_repr(self):
        assert repr(IMRPhenomXAS(f_ref=20.0)) == "IMRPhenomXAS(f_ref=20.0)"

    def test_in_waveform_preset(self):
        assert "IMRPhenomXAS" in waveform_preset
        assert isinstance(waveform_preset["IMRPhenomXAS"](f_ref=20.0), IMRPhenomXAS)


class TestIMRPhenomPv2:
    # --- low-level gen function ---
    def test_gen_basic(self, test_freq_grid, bbh_precessing_params):
        from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
        hp, hc = gen_IMRPhenomPv2_hphc(test_freq_grid, bbh_precessing_params, 20.0)
        assert_fd_valid(hp, hc, test_freq_grid)

    def test_gen_jit(self, test_freq_grid, bbh_precessing_params):
        from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
        fs, f_ref = test_freq_grid, 20.0
        hp, hc = jax.jit(lambda t: gen_IMRPhenomPv2_hphc(fs, t, f_ref))(bbh_precessing_params)
        assert_fd_valid(hp, hc, fs)

    def test_gen_vmap(self, test_freq_grid, bbh_precessing_params):
        from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
        fs, f_ref, batch_size = test_freq_grid, 20.0, 5
        batch = jnp.tile(bbh_precessing_params, (batch_size, 1))
        hp_b, hc_b = jax.vmap(lambda t: gen_IMRPhenomPv2_hphc(fs, t, f_ref))(batch)
        assert hp_b.shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(hp_b))

    # --- top-level approximant class ---
    def test_basic(self, test_freq_grid, bbh_precessing_dict):
        output = IMRPhenomPv2(f_ref=20.0)(test_freq_grid, bbh_precessing_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_jit(self, test_freq_grid, bbh_precessing_dict):
        model = IMRPhenomPv2(f_ref=20.0)
        output = jax.jit(model)(test_freq_grid, bbh_precessing_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_vmap(self, test_freq_grid, bbh_precessing_dict):
        model = IMRPhenomPv2(f_ref=20.0)
        fs, batch_size = test_freq_grid, 4
        out = jax.vmap(lambda p: model(fs, p))(batch_dict(bbh_precessing_dict, batch_size))
        assert out["p"].shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_repr(self):
        assert repr(IMRPhenomPv2(f_ref=20.0)) == "IMRPhenomPv2(f_ref=20.0)"

    def test_in_waveform_preset(self):
        assert "IMRPhenomPv2" in waveform_preset
        assert isinstance(waveform_preset["IMRPhenomPv2"](f_ref=20.0), IMRPhenomPv2)


class TestIMRPhenomD_NRTidalv2:
    # --- low-level gen function ---
    def test_gen_basic(self, test_freq_grid, bns_tidal_params):
        from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc
        hp, hc = gen_IMRPhenomD_NRTidalv2_hphc(test_freq_grid, bns_tidal_params, 20.0)
        assert_fd_valid(hp, hc, test_freq_grid)

    def test_gen_jit(self, test_freq_grid, bns_tidal_params):
        from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc
        fs, f_ref = test_freq_grid, 20.0
        hp, hc = jax.jit(lambda t: gen_IMRPhenomD_NRTidalv2_hphc(fs, t, f_ref))(bns_tidal_params)
        assert_fd_valid(hp, hc, fs)

    def test_gen_vmap(self, test_freq_grid, bns_tidal_params):
        from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc
        fs, f_ref, batch_size = test_freq_grid, 20.0, 5
        batch = jnp.tile(bns_tidal_params, (batch_size, 1))
        hp_b, hc_b = jax.vmap(lambda t: gen_IMRPhenomD_NRTidalv2_hphc(fs, t, f_ref))(batch)
        assert hp_b.shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(hp_b))

    # --- top-level approximant class ---
    def test_basic_lambda(self, test_freq_grid, bns_tidal_dict):
        output = IMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=False)(test_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_basic_lambda_tildes(self, test_freq_grid, bns_tidal_tilde_dict):
        output = IMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=True)(test_freq_grid, bns_tidal_tilde_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_jit(self, test_freq_grid, bns_tidal_dict):
        model = IMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=False)
        output = jax.jit(model)(test_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_vmap(self, test_freq_grid, bns_tidal_dict):
        model = IMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=False)
        fs, batch_size = test_freq_grid, 4
        out = jax.vmap(lambda p: model(fs, p))(batch_dict(bns_tidal_dict, batch_size))
        assert out["p"].shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_in_waveform_preset(self):
        assert "IMRPhenomD_NRTidalv2" in waveform_preset
        assert isinstance(waveform_preset["IMRPhenomD_NRTidalv2"](f_ref=20.0), IMRPhenomD_NRTidalv2)


class TestIMRPhenomXAS_NRTidalv3:
    # --- low-level gen function ---
    def test_gen_basic(self, test_freq_grid, bns_tidal_params):
        from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
        hp, hc = gen_IMRPhenomXAS_NRTidalv3_hphc(test_freq_grid, bns_tidal_params, 20.0)
        assert_fd_valid(hp, hc, test_freq_grid)

    def test_gen_jit(self, test_freq_grid, bns_tidal_params):
        from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
        fs, f_ref = test_freq_grid, 20.0
        hp, hc = jax.jit(lambda t: gen_IMRPhenomXAS_NRTidalv3_hphc(fs, t, f_ref))(bns_tidal_params)
        assert_fd_valid(hp, hc, fs)

    def test_gen_vmap(self, test_freq_grid, bns_tidal_params):
        from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
        fs, f_ref, batch_size = test_freq_grid, 20.0, 5
        batch = jnp.tile(bns_tidal_params, (batch_size, 1))
        hp_b, hc_b = jax.vmap(lambda t: gen_IMRPhenomXAS_NRTidalv3_hphc(fs, t, f_ref))(batch)
        assert hp_b.shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(hp_b))

    # --- top-level approximant class ---
    def test_basic_lambda(self, test_freq_grid, bns_tidal_dict):
        output = IMRPhenomXAS_NRTidalv3(f_ref=20.0, use_lambda_tildes=False)(test_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_basic_lambda_tildes(self, test_freq_grid, bns_tidal_tilde_dict):
        output = IMRPhenomXAS_NRTidalv3(f_ref=20.0, use_lambda_tildes=True)(test_freq_grid, bns_tidal_tilde_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_jit(self, test_freq_grid, bns_tidal_dict):
        model = IMRPhenomXAS_NRTidalv3(f_ref=20.0, use_lambda_tildes=False)
        output = jax.jit(model)(test_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_vmap(self, test_freq_grid, bns_tidal_dict):
        model = IMRPhenomXAS_NRTidalv3(f_ref=20.0, use_lambda_tildes=False)
        fs, batch_size = test_freq_grid, 4
        out = jax.vmap(lambda p: model(fs, p))(batch_dict(bns_tidal_dict, batch_size))
        assert out["p"].shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_in_waveform_preset(self):
        assert "IMRPhenomXAS_NRTidalv3" in waveform_preset
        assert isinstance(waveform_preset["IMRPhenomXAS_NRTidalv3"](f_ref=20.0), IMRPhenomXAS_NRTidalv3)


class TestTaylorF2:
    # --- low-level gen function ---
    def test_gen_basic(self, test_freq_grid, bns_tidal_params):
        from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc
        hp, hc = gen_TaylorF2_hphc(test_freq_grid, bns_tidal_params, 20.0)
        assert_fd_valid(hp, hc, test_freq_grid)

    def test_gen_jit(self, test_freq_grid, bns_tidal_params):
        from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc
        fs, f_ref = test_freq_grid, 20.0
        hp, hc = jax.jit(lambda t: gen_TaylorF2_hphc(fs, t, f_ref))(bns_tidal_params)
        assert_fd_valid(hp, hc, fs)

    def test_gen_vmap(self, test_freq_grid, bns_tidal_params):
        from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc
        fs, f_ref, batch_size = test_freq_grid, 20.0, 5
        batch = jnp.tile(bns_tidal_params, (batch_size, 1))
        hp_b, hc_b = jax.vmap(lambda t: gen_TaylorF2_hphc(fs, t, f_ref))(batch)
        assert hp_b.shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(hp_b))

    # --- top-level approximant class ---
    def test_basic_lambda(self, test_freq_grid, bns_tidal_dict):
        output = TaylorF2(f_ref=20.0, use_lambda_tildes=False)(test_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_basic_lambda_tildes(self, test_freq_grid, bns_tidal_tilde_dict):
        output = TaylorF2(f_ref=20.0, use_lambda_tildes=True)(test_freq_grid, bns_tidal_tilde_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_jit(self, test_freq_grid, bns_tidal_dict):
        model = TaylorF2(f_ref=20.0, use_lambda_tildes=False)
        output = jax.jit(model)(test_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_vmap(self, test_freq_grid, bns_tidal_dict):
        model = TaylorF2(f_ref=20.0, use_lambda_tildes=False)
        fs, batch_size = test_freq_grid, 4
        out = jax.vmap(lambda p: model(fs, p))(batch_dict(bns_tidal_dict, batch_size))
        assert out["p"].shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_repr(self):
        assert repr(TaylorF2(f_ref=20.0)) == "TaylorF2(f_ref=20.0)"

    def test_in_waveform_preset(self):
        assert "TaylorF2" in waveform_preset
        assert isinstance(waveform_preset["TaylorF2"](f_ref=20.0), TaylorF2)


class TestIMRPhenomXPHM:
    # --- low-level gen function ---
    def test_gen_basic(self, test_freq_grid, bbh_xphm_params):
        from ripplegw.waveforms.IMRPhenomXPHM import generate_xphm
        fs, f_ref = test_freq_grid, 20.0
        m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, d, iota, phi0 = bbh_xphm_params
        hp, hc = generate_xphm(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, d, iota, phi0, fs, f_ref)
        assert_fd_valid(hp, hc, fs)

    def test_gen_jit(self, test_freq_grid, bbh_xphm_params):
        from ripplegw.waveforms.IMRPhenomXPHM import generate_xphm
        fs, f_ref = test_freq_grid, 20.0
        m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, d, iota, phi0 = bbh_xphm_params
        fn = jax.jit(lambda *args: generate_xphm(*args, fs, f_ref))
        hp, hc = fn(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, d, iota, phi0)
        assert_fd_valid(hp, hc, fs)

    def test_gen_vmap(self, test_freq_grid, bbh_xphm_params):
        from ripplegw.waveforms.IMRPhenomXPHM import generate_xphm
        fs, f_ref, batch_size = test_freq_grid, 20.0, 3
        m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, d, iota, phi0 = bbh_xphm_params
        batched = jax.vmap(generate_xphm, in_axes=(0,)*11 + (None, None))
        hp_b, hc_b = batched(
            jnp.full(batch_size, m1), jnp.full(batch_size, m2),
            jnp.full(batch_size, s1x), jnp.full(batch_size, s1y), jnp.full(batch_size, s1z),
            jnp.full(batch_size, s2x), jnp.full(batch_size, s2y), jnp.full(batch_size, s2z),
            jnp.full(batch_size, d), jnp.full(batch_size, iota), jnp.full(batch_size, phi0),
            fs, f_ref,
        )
        assert hp_b.shape == (batch_size, len(fs))
        assert jnp.all(jnp.isfinite(hp_b))


class TestSineGaussian:
    # --- low-level gen function ---
    def test_gen_basic(self, test_time_grid, sinegaussian_params):
        from ripplegw.waveforms.SineGaussian import gen_SineGaussian_hphc
        hp, hc = gen_SineGaussian_hphc(test_time_grid, sinegaussian_params)
        assert_td_valid(hp, hc, test_time_grid)

    def test_gen_jit(self, test_time_grid, sinegaussian_params):
        from ripplegw.waveforms.SineGaussian import gen_SineGaussian_hphc
        t = test_time_grid
        hp, hc = jax.jit(lambda theta: gen_SineGaussian_hphc(t, theta))(sinegaussian_params)
        assert_td_valid(hp, hc, t)

    def test_gen_vmap(self, test_time_grid, sinegaussian_params):
        from ripplegw.waveforms.SineGaussian import gen_SineGaussian_hphc
        t, batch_size = test_time_grid, 5
        batch = jnp.tile(sinegaussian_params, (batch_size, 1))
        hp_b, hc_b = jax.vmap(lambda theta: gen_SineGaussian_hphc(t, theta))(batch)
        assert hp_b.shape == (batch_size, len(t))
        assert jnp.all(jnp.isfinite(hp_b))
        assert jnp.all(jnp.isreal(hp_b))

    # --- top-level approximant class ---
    def test_basic(self, test_time_grid, sinegaussian_dict):
        output = SineGaussian()(test_time_grid, sinegaussian_dict)
        assert_approx_td_valid(output, test_time_grid)

    def test_jit(self, test_time_grid, sinegaussian_dict):
        model = SineGaussian()
        output = jax.jit(model)(test_time_grid, sinegaussian_dict)
        assert_approx_td_valid(output, test_time_grid)

    def test_vmap(self, test_time_grid, sinegaussian_dict):
        model = SineGaussian()
        t, batch_size = test_time_grid, 4
        out = jax.vmap(lambda p: model(t, p))(batch_dict(sinegaussian_dict, batch_size))
        assert out["p"].shape == (batch_size, len(t))
        assert jnp.all(jnp.isfinite(out["p"]))

    def test_repr(self):
        assert repr(SineGaussian()) == "SineGaussian()"

    def test_in_waveform_preset(self):
        assert "SineGaussian" in waveform_preset
        assert isinstance(waveform_preset["SineGaussian"](), SineGaussian)


class TestWaveformPreset:
    def test_all_keys_present(self):
        expected = {
            "IMRPhenomD", "IMRPhenomPv2", "TaylorF2",
            "IMRPhenomD_NRTidalv2", "IMRPhenomXAS",
            "IMRPhenomXAS_NRTidalv3", "SineGaussian",
        }
        assert expected == set(waveform_preset.keys())

    def test_all_instantiable(self):
        for name, cls in waveform_preset.items():
            instance = cls() if name == "SineGaussian" else cls(f_ref=20.0)
            assert callable(instance), f"{name} instance is not callable"
