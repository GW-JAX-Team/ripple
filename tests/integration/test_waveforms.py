"""Integration tests for waveform generation.

Each test class covers the top-level callable class exported from ``ripplegw``
(dict params) and edge cases in the physical parameter space.

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
    IMRPhenomXPHM,
    SineGaussian,
    TaylorF2,
    waveform_preset,
)
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


# ============================================================================
# Fixtures — dict params (for top-level approximant class tests)
# ============================================================================


@pytest.fixture(scope="module")
def bbh_aligned_dict():
    """Dict params for aligned-spin BBH approximant classes."""
    m1, m2 = 30.0, 25.0
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {
        "M_c": float(Mc),
        "eta": float(eta),
        "s1_z": 0.5,
        "s2_z": -0.3,
        "d_L": 400.0,
        "phase_c": 0.5,
        "iota": 0.8,
    }


@pytest.fixture(scope="module")
def bbh_precessing_dict():
    """Dict params for IMRPhenomPv2 approximant class."""
    m1, m2 = 30.0, 25.0
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {
        "M_c": float(Mc),
        "eta": float(eta),
        "s1_x": 0.1,
        "s1_y": 0.2,
        "s1_z": 0.3,
        "s2_x": -0.1,
        "s2_y": 0.15,
        "s2_z": -0.2,
        "d_L": 400.0,
        "phase_c": 0.5,
        "iota": 0.8,
    }


@pytest.fixture(scope="module")
def bns_tidal_dict():
    """Dict params for tidal approximant classes (lambda_1/lambda_2)."""
    m1, m2 = 1.4, 1.3
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {
        "M_c": float(Mc),
        "eta": float(eta),
        "s1_z": 0.05,
        "s2_z": -0.02,
        "lambda_1": 500.0,
        "lambda_2": 400.0,
        "d_L": 100.0,
        "phase_c": 0.5,
        "iota": 0.8,
    }


@pytest.fixture(scope="module")
def bns_tidal_tilde_dict():
    """Dict params for tidal approximant classes (lambda_tilde/delta_lambda_tilde)."""
    m1, m2 = 1.4, 1.3
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([500.0, 400.0, m1, m2]))
    return {
        "M_c": float(Mc),
        "eta": float(eta),
        "s1_z": 0.05,
        "s2_z": -0.02,
        "lambda_tilde": float(lt),
        "delta_lambda_tilde": float(dlt),
        "d_L": 100.0,
        "phase_c": 0.5,
        "iota": 0.8,
    }


@pytest.fixture(scope="module")
def sinegaussian_dict():
    """Dict params for SineGaussian approximant class."""
    return {"Q": 10.0, "f_0": 100.0, "hrss": 1e-21, "phase": 0.5, "e": 0.3}


@pytest.fixture(scope="module")
def bbh_xphm_dict():
    """Dict params for IMRPhenomXPHM approximant class."""
    m1, m2 = 30.0, 25.0
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {
        "M_c": float(Mc),
        "eta": float(eta),
        "s1_x": 0.1,
        "s1_y": 0.2,
        "s1_z": 0.3,
        "s2_x": -0.1,
        "s2_y": 0.15,
        "s2_z": -0.2,
        "d_L": 400.0,
        "phase_c": 0.5,
        "iota": 0.8,
    }


# ============================================================================
# Fixtures — grids
# ============================================================================


@pytest.fixture(scope="module")
def test_time_grid():
    fs_sampling, duration = 4096.0, 1.0
    return jnp.arange(-duration / 2, duration / 2, 1 / fs_sampling)


@pytest.fixture(scope="module")
def test_freq_grid():
    f_l, f_u, f_sampling, T = 20.0, 1024.0, 2048.0, 2.0
    delta_t = 1 / f_sampling
    freqs = jnp.fft.rfftfreq(int(round(T / delta_t)), delta_t)
    return freqs[(freqs > f_l) & (freqs < f_u)]


@pytest.fixture(scope="module")
def edge_freq_grid():
    """Small frequency grid for edge-case tests (~500 pts, eager calls ~0.2s)."""
    return jnp.linspace(20.0, 512.0, 500)


@pytest.fixture(scope="module")
def edge_time_grid():
    """Small time grid for SineGaussian edge-case tests."""
    return jnp.linspace(-0.5, 0.5, 512)


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


def _bbh_dict(m1, m2, s1_z=0.0, s2_z=0.0, d_L=400.0, phase_c=0.0, iota=0.0):
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {
        "M_c": float(Mc),
        "eta": float(eta),
        "s1_z": s1_z,
        "s2_z": s2_z,
        "d_L": d_L,
        "phase_c": phase_c,
        "iota": iota,
    }


def _bns_dict(
    m1,
    m2,
    s1_z=0.0,
    s2_z=0.0,
    lambda_1=500.0,
    lambda_2=400.0,
    d_L=100.0,
    phase_c=0.0,
    iota=0.0,
):
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {
        "M_c": float(Mc),
        "eta": float(eta),
        "s1_z": s1_z,
        "s2_z": s2_z,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "d_L": d_L,
        "phase_c": phase_c,
        "iota": iota,
    }


def _pv2_dict(
    m1,
    m2,
    s1_x=0.0,
    s1_y=0.0,
    s1_z=0.0,
    s2_x=0.0,
    s2_y=0.0,
    s2_z=0.0,
    d_L=400.0,
    phase_c=0.0,
    iota=0.0,
):
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {
        "M_c": float(Mc),
        "eta": float(eta),
        "s1_x": s1_x,
        "s1_y": s1_y,
        "s1_z": s1_z,
        "s2_x": s2_x,
        "s2_y": s2_y,
        "s2_z": s2_z,
        "d_L": d_L,
        "phase_c": phase_c,
        "iota": iota,
    }


def _xphm_dict(
    m1,
    m2,
    s1_x=0.0,
    s1_y=0.0,
    s1_z=0.0,
    s2_x=0.0,
    s2_y=0.0,
    s2_z=0.0,
    d_L=400.0,
    phase_c=0.0,
    iota=0.0,
):
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return {
        "M_c": float(Mc),
        "eta": float(eta),
        "s1_x": s1_x,
        "s1_y": s1_y,
        "s1_z": s1_z,
        "s2_x": s2_x,
        "s2_y": s2_y,
        "s2_z": s2_z,
        "d_L": d_L,
        "phase_c": phase_c,
        "iota": iota,
    }


class TestIMRPhenomD:
    # --- top-level approximant class ---
    def test_basic(self, edge_freq_grid, bbh_aligned_dict):
        output = IMRPhenomD(f_ref=20.0)(edge_freq_grid, bbh_aligned_dict)
        assert_approx_fd_valid(output, edge_freq_grid)

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

    # --- edge cases ---
    def test_equal_mass(self, edge_freq_grid):
        """eta = 0.25: equal-mass binary."""
        params = _bbh_dict(30.0, 30.0)
        assert params["eta"] == pytest.approx(0.25)
        assert_approx_fd_valid(
            IMRPhenomD(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_spins(self, edge_freq_grid):
        """chi1 = chi2 = 0: Schwarzschild limit."""
        params = _bbh_dict(30.0, 25.0, s1_z=0.0, s2_z=0.0)
        assert_approx_fd_valid(
            IMRPhenomD(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_max_aligned_spin(self, edge_freq_grid):
        """chi1 = +0.99, chi2 = -0.99: near-maximal spins."""
        params = _bbh_dict(30.0, 25.0, s1_z=0.99, s2_z=-0.99)
        assert_approx_fd_valid(
            IMRPhenomD(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )


class TestIMRPhenomXAS:
    # --- top-level approximant class ---
    def test_basic(self, edge_freq_grid, bbh_aligned_dict):
        output = IMRPhenomXAS(f_ref=20.0)(edge_freq_grid, bbh_aligned_dict)
        assert_approx_fd_valid(output, edge_freq_grid)

    def test_jit(self, test_freq_grid, bbh_aligned_dict):
        model = IMRPhenomXAS(f_ref=20.0)
        output = jax.jit(model)(test_freq_grid, bbh_aligned_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_repr(self):
        assert repr(IMRPhenomXAS(f_ref=20.0)) == "IMRPhenomXAS(f_ref=20.0)"

    def test_in_waveform_preset(self):
        assert "IMRPhenomXAS" in waveform_preset
        assert isinstance(waveform_preset["IMRPhenomXAS"](f_ref=20.0), IMRPhenomXAS)

    # --- edge cases ---
    def test_equal_mass(self, edge_freq_grid):
        """eta = 0.25: equal-mass binary."""
        params = _bbh_dict(30.0, 30.0)
        assert params["eta"] == pytest.approx(0.25)
        assert_approx_fd_valid(
            IMRPhenomXAS(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_spins(self, edge_freq_grid):
        """chi1 = chi2 = 0: Schwarzschild limit."""
        params = _bbh_dict(30.0, 25.0, s1_z=0.0, s2_z=0.0)
        assert_approx_fd_valid(
            IMRPhenomXAS(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_max_aligned_spin(self, edge_freq_grid):
        """chi1 = +0.99, chi2 = -0.99: near-maximal spins."""
        params = _bbh_dict(30.0, 25.0, s1_z=0.99, s2_z=-0.99)
        assert_approx_fd_valid(
            IMRPhenomXAS(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )


class TestIMRPhenomPv2:
    # --- top-level approximant class ---
    def test_basic(self, edge_freq_grid, bbh_precessing_dict):
        output = IMRPhenomPv2(f_ref=20.0)(edge_freq_grid, bbh_precessing_dict)
        assert_approx_fd_valid(output, edge_freq_grid)

    def test_jit(self, test_freq_grid, bbh_precessing_dict):
        model = IMRPhenomPv2(f_ref=20.0)
        output = jax.jit(model)(test_freq_grid, bbh_precessing_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_repr(self):
        assert repr(IMRPhenomPv2(f_ref=20.0)) == "IMRPhenomPv2(f_ref=20.0)"

    def test_in_waveform_preset(self):
        assert "IMRPhenomPv2" in waveform_preset
        assert isinstance(waveform_preset["IMRPhenomPv2"](f_ref=20.0), IMRPhenomPv2)

    # --- edge cases ---
    def test_equal_mass(self, edge_freq_grid):
        """eta = 0.25: equal-mass precessing binary."""
        params = _pv2_dict(
            30.0, 30.0, s1_x=0.1, s1_y=0.2, s1_z=0.3, s2_x=-0.1, s2_y=0.15, s2_z=-0.2
        )
        assert params["eta"] == pytest.approx(0.25)
        assert_approx_fd_valid(
            IMRPhenomPv2(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_spins(self, edge_freq_grid):
        """All spin components zero — non-spinning limit."""
        params = _pv2_dict(30.0, 25.0)
        assert_approx_fd_valid(
            IMRPhenomPv2(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_aligned_spins_only(self, edge_freq_grid):
        """In-plane spins zero — reduces to aligned-spin limit."""
        params = _pv2_dict(30.0, 25.0, s1_z=0.5, s2_z=-0.3)
        assert_approx_fd_valid(
            IMRPhenomPv2(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )


class TestIMRPhenomD_NRTidalv2:
    # --- top-level approximant class ---
    def test_basic_lambda(self, edge_freq_grid, bns_tidal_dict):
        output = IMRPhenomD_NRTidalv2(f_ref=20.0)(edge_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, edge_freq_grid)

    def test_basic_lambda_tildes(self, test_freq_grid, bns_tidal_tilde_dict):
        model = IMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=True)
        assert_approx_fd_valid(
            model(test_freq_grid, bns_tidal_tilde_dict), test_freq_grid
        )

    def test_jit(self, test_freq_grid, bns_tidal_dict):
        model = IMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=False)
        output = jax.jit(model)(test_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_in_waveform_preset(self):
        assert "IMRPhenomD_NRTidalv2" in waveform_preset
        assert isinstance(
            waveform_preset["IMRPhenomD_NRTidalv2"](f_ref=20.0), IMRPhenomD_NRTidalv2
        )

    # --- edge cases ---
    def test_equal_mass(self, edge_freq_grid):
        """eta = 0.25: equal-mass BNS."""
        params = _bns_dict(1.4, 1.4, lambda_1=500.0, lambda_2=500.0)
        assert params["eta"] == pytest.approx(0.25)
        assert_approx_fd_valid(
            IMRPhenomD_NRTidalv2(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_tidal_deformability(self, edge_freq_grid):
        """lambda_1 = lambda_2 = 0: BH-like tidal correction."""
        params = _bns_dict(1.4, 1.3, lambda_1=0.0, lambda_2=0.0)
        assert_approx_fd_valid(
            IMRPhenomD_NRTidalv2(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_spins(self, edge_freq_grid):
        """Non-spinning BNS."""
        params = _bns_dict(1.4, 1.3, s1_z=0.0, s2_z=0.0)
        assert_approx_fd_valid(
            IMRPhenomD_NRTidalv2(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )


class TestIMRPhenomXAS_NRTidalv3:
    # --- top-level approximant class ---
    def test_basic_lambda(self, edge_freq_grid, bns_tidal_dict):
        output = IMRPhenomXAS_NRTidalv3(f_ref=20.0)(edge_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, edge_freq_grid)

    def test_basic_lambda_tildes(self, test_freq_grid, bns_tidal_tilde_dict):
        model = IMRPhenomXAS_NRTidalv3(f_ref=20.0, use_lambda_tildes=True)
        assert_approx_fd_valid(
            model(test_freq_grid, bns_tidal_tilde_dict), test_freq_grid
        )

    def test_jit(self, test_freq_grid, bns_tidal_dict):
        model = IMRPhenomXAS_NRTidalv3(f_ref=20.0, use_lambda_tildes=False)
        output = jax.jit(model)(test_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_in_waveform_preset(self):
        assert "IMRPhenomXAS_NRTidalv3" in waveform_preset
        assert isinstance(
            waveform_preset["IMRPhenomXAS_NRTidalv3"](f_ref=20.0),
            IMRPhenomXAS_NRTidalv3,
        )

    # --- edge cases ---
    def test_equal_mass(self, edge_freq_grid):
        """eta = 0.25: equal-mass BNS."""
        params = _bns_dict(1.4, 1.4, lambda_1=500.0, lambda_2=500.0)
        assert params["eta"] == pytest.approx(0.25)
        assert_approx_fd_valid(
            IMRPhenomXAS_NRTidalv3(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_tidal_deformability(self, edge_freq_grid):
        """lambda_1 = lambda_2 = 0: BH-like tidal correction."""
        params = _bns_dict(1.4, 1.3, lambda_1=0.0, lambda_2=0.0)
        assert_approx_fd_valid(
            IMRPhenomXAS_NRTidalv3(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_spins(self, edge_freq_grid):
        """Non-spinning BNS."""
        params = _bns_dict(1.4, 1.3, s1_z=0.0, s2_z=0.0)
        assert_approx_fd_valid(
            IMRPhenomXAS_NRTidalv3(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )


class TestTaylorF2:
    # --- top-level approximant class ---
    def test_basic_lambda(self, edge_freq_grid, bns_tidal_dict):
        output = TaylorF2(f_ref=20.0)(edge_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, edge_freq_grid)

    def test_basic_lambda_tildes(self, test_freq_grid, bns_tidal_tilde_dict):
        model = TaylorF2(f_ref=20.0, use_lambda_tildes=True)
        assert_approx_fd_valid(
            model(test_freq_grid, bns_tidal_tilde_dict), test_freq_grid
        )

    def test_jit(self, test_freq_grid, bns_tidal_dict):
        model = TaylorF2(f_ref=20.0, use_lambda_tildes=False)
        output = jax.jit(model)(test_freq_grid, bns_tidal_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_repr(self):
        assert repr(TaylorF2(f_ref=20.0)) == "TaylorF2(f_ref=20.0)"

    def test_in_waveform_preset(self):
        assert "TaylorF2" in waveform_preset
        assert isinstance(waveform_preset["TaylorF2"](f_ref=20.0), TaylorF2)

    # --- edge cases ---
    def test_equal_mass(self, edge_freq_grid):
        """eta = 0.25: equal-mass BNS."""
        params = _bns_dict(1.4, 1.4, lambda_1=500.0, lambda_2=500.0)
        assert params["eta"] == pytest.approx(0.25)
        assert_approx_fd_valid(
            TaylorF2(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_tidal_deformability(self, edge_freq_grid):
        """lambda_1 = lambda_2 = 0: BH-like tidal correction."""
        params = _bns_dict(1.4, 1.3, lambda_1=0.0, lambda_2=0.0)
        assert_approx_fd_valid(
            TaylorF2(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_spins(self, edge_freq_grid):
        """Non-spinning BNS."""
        params = _bns_dict(1.4, 1.3, s1_z=0.0, s2_z=0.0)
        assert_approx_fd_valid(
            TaylorF2(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )


class TestIMRPhenomXPHM:
    def test_basic(self, edge_freq_grid, bbh_xphm_dict):
        output = IMRPhenomXPHM(f_ref=20.0)(edge_freq_grid, bbh_xphm_dict)
        assert_approx_fd_valid(output, edge_freq_grid)

    def test_jit(self, test_freq_grid, bbh_xphm_dict):
        model = IMRPhenomXPHM(f_ref=20.0)
        output = jax.jit(model)(test_freq_grid, bbh_xphm_dict)
        assert_approx_fd_valid(output, test_freq_grid)

    def test_repr(self):
        assert repr(IMRPhenomXPHM(f_ref=20.0)) == "IMRPhenomXPHM(f_ref=20.0)"

    def test_in_waveform_preset(self):
        assert "IMRPhenomXPHM" in waveform_preset
        assert isinstance(waveform_preset["IMRPhenomXPHM"](f_ref=20.0), IMRPhenomXPHM)

    def test_equal_mass(self, edge_freq_grid):
        """m_1 = m_2 = 30 Msun — equal-mass limit with precessing spins."""
        params = _xphm_dict(
            30.0, 30.0, s1_x=0.1, s1_y=0.2, s1_z=0.3, s2_x=-0.1, s2_y=0.15, s2_z=-0.2
        )
        assert_approx_fd_valid(
            IMRPhenomXPHM(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_zero_spins(self, edge_freq_grid):
        """All spin components zero — non-spinning limit."""
        params = _xphm_dict(30.0, 25.0)
        assert_approx_fd_valid(
            IMRPhenomXPHM(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_aligned_spins_only(self, edge_freq_grid):
        """In-plane spins zero — reduces to aligned-spin limit."""
        params = _xphm_dict(30.0, 25.0, s1_z=0.5, s2_z=-0.3)
        assert_approx_fd_valid(
            IMRPhenomXPHM(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )

    def test_fully_precessing(self, edge_freq_grid):
        """Large in-plane spin components — strong precession regime."""
        params = _xphm_dict(
            30.0, 25.0, s1_x=0.5, s1_y=0.5, s1_z=0.1, s2_x=-0.4, s2_y=0.3, s2_z=-0.1
        )
        assert_approx_fd_valid(
            IMRPhenomXPHM(f_ref=20.0)(edge_freq_grid, params), edge_freq_grid
        )


class TestSineGaussian:
    # --- top-level approximant class ---
    def test_basic(self, test_time_grid, sinegaussian_dict):
        output = SineGaussian()(test_time_grid, sinegaussian_dict)
        assert_approx_td_valid(output, test_time_grid)

    def test_jit(self, test_time_grid, sinegaussian_dict):
        model = SineGaussian()
        output = jax.jit(model)(test_time_grid, sinegaussian_dict)
        assert_approx_td_valid(output, test_time_grid)

    def test_repr(self):
        assert repr(SineGaussian()) == "SineGaussian()"

    def test_in_waveform_preset(self):
        assert "SineGaussian" in waveform_preset
        assert isinstance(waveform_preset["SineGaussian"](), SineGaussian)

    # --- edge cases ---
    def test_low_quality_factor(self, edge_time_grid):
        """Q = 2: broad-band burst near the minimum valid Q."""
        params = {"Q": 2.0, "f_0": 100.0, "hrss": 1e-21, "phase": 0.0, "e": 0.0}
        assert_approx_td_valid(SineGaussian()(edge_time_grid, params), edge_time_grid)

    def test_circular_polarisation(self, edge_time_grid):
        """e = 0: purely circular polarisation."""
        params = {"Q": 10.0, "f_0": 100.0, "hrss": 1e-21, "phase": 0.5, "e": 0.0}
        assert_approx_td_valid(SineGaussian()(edge_time_grid, params), edge_time_grid)

    def test_linear_polarisation(self, edge_time_grid):
        """e = 1: purely linear polarisation (sqrt(1 - e^2) -> 0)."""
        params = {"Q": 10.0, "f_0": 100.0, "hrss": 1e-21, "phase": 0.5, "e": 1.0}
        assert_approx_td_valid(SineGaussian()(edge_time_grid, params), edge_time_grid)


class TestWaveformPreset:
    def test_all_keys_present(self):
        expected = {
            "IMRPhenomD",
            "IMRPhenomPv2",
            "TaylorF2",
            "IMRPhenomD_NRTidalv2",
            "IMRPhenomXAS",
            "IMRPhenomXAS_NRTidalv3",
            "IMRPhenomXPHM",
            "IMRPhenomHM",
            "SineGaussian",
        }
        assert expected == set(waveform_preset.keys())

    def test_all_instantiable(self):
        for name, cls in waveform_preset.items():
            instance = cls() if name == "SineGaussian" else cls(f_ref=20.0)
            assert callable(instance), f"{name} instance is not callable"
