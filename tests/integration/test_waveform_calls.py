"""Integration tests for top-level waveform calling interface.

These tests verify that all supported waveform models can be generated
successfully, produce valid outputs (finite, correct shape, complex-valued),
and work with JAX transformations (JIT, vmap, grad).

These tests do NOT compare against LALSuite - that's done in cross_validation/.
"""

import jax
import jax.numpy as jnp
import pytest

from ripplegw import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.constants import PI

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Test parameters for each waveform type
# ============================================================================


@pytest.fixture
def bbh_aligned_params():
    """Fixed parameter set for aligned-spin BBH waveforms."""
    m1, m2 = 30.0, 25.0
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    chi1, chi2 = 0.5, -0.3
    dist_mpc = 400.0
    tc = 0.0
    phic = 0.5
    inclination = 0.8
    return jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])


@pytest.fixture
def bns_tidal_params():
    """Fixed parameter set for BNS waveforms with tidal effects."""
    m1, m2 = 1.4, 1.3
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    chi1, chi2 = 0.05, -0.02
    lambda1, lambda2 = 500.0, 400.0
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
        jnp.array([lambda1, lambda2, m1, m2])
    )
    dist_mpc = 100.0
    tc = 0.0
    phic = 0.5
    inclination = 0.8
    return jnp.array(
        [Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde, dist_mpc, tc, phic, inclination]
    )


@pytest.fixture
def bbh_precessing_params():
    """Fixed parameter set for precessing BBH waveforms (IMRPhenomPv2)."""
    m1, m2 = 30.0, 25.0
    # Spin parameters: s1x, s1y, s1z, s2x, s2y, s2z
    s1x, s1y, s1z = 0.1, 0.2, 0.3
    s2x, s2y, s2z = -0.1, 0.15, -0.2
    dist_mpc = 400.0
    tc = 0.0
    phic = 0.5
    inclination = 0.8
    # IMRPhenomPv2 expects: [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phiRef, incl]
    return jnp.array([m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phic, inclination])


@pytest.fixture
def sinegaussian_params():
    """Fixed parameter set for SineGaussian burst waveforms."""
    quality = 10.0
    frequency = 100.0
    hrss = 1e-21
    phase = 0.5
    eccentricity = 0.3
    return jnp.array([quality, frequency, hrss, phase, eccentricity])


@pytest.fixture
def test_time_grid():
    """Time grid for time-domain waveforms (SineGaussian)."""
    fs_sampling = 4096.0  # Sample rate
    duration = 1.0  # Duration in seconds
    t = jnp.arange(-duration / 2, duration / 2, 1 / fs_sampling)
    return t


@pytest.fixture
def test_freq_grid():
    """Frequency grid for testing."""
    f_l = 20.0
    f_u = 1024.0
    f_sampling = 2048.0
    T = 16.0
    delta_t = 1 / f_sampling
    tlen = int(round(T / delta_t))
    freqs = jnp.fft.rfftfreq(tlen, delta_t)
    fs = freqs[(freqs > f_l) & (freqs < f_u)]
    return fs


# ============================================================================
# Helper functions
# ============================================================================


def assert_waveform_valid(hp, hc, fs):
    """Assert that a waveform output is valid.

    Checks:
    - Output shape matches frequency grid
    - All values are finite (no NaN/Inf)
    - Output is complex-valued
    """
    assert hp.shape == fs.shape, f"hp shape {hp.shape} != fs shape {fs.shape}"
    assert hc.shape == fs.shape, f"hc shape {hc.shape} != fs shape {fs.shape}"
    assert jnp.all(jnp.isfinite(hp)), "hp contains NaN or Inf"
    assert jnp.all(jnp.isfinite(hc)), "hc contains NaN or Inf"
    assert jnp.iscomplexobj(hp), "hp is not complex-valued"
    assert jnp.iscomplexobj(hc), "hc is not complex-valued"


# ============================================================================
# Test IMRPhenomD (aligned-spin BBH)
# ============================================================================


def test_imrphenomd_basic(test_freq_grid, bbh_aligned_params):
    """Test IMRPhenomD waveform generation."""
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc

    fs = test_freq_grid
    f_ref = 20.0
    hp, hc = gen_IMRPhenomD_hphc(fs, bbh_aligned_params, f_ref)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenomd_jit(test_freq_grid, bbh_aligned_params):
    """Test that IMRPhenomD works with JIT compilation."""
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc

    fs = test_freq_grid
    f_ref = 20.0

    @jax.jit
    def waveform_jitted(theta):
        return gen_IMRPhenomD_hphc(fs, theta, f_ref)

    hp, hc = waveform_jitted(bbh_aligned_params)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenomd_vmap(test_freq_grid, bbh_aligned_params):
    """Test that IMRPhenomD works with vmap."""
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc

    fs = test_freq_grid
    f_ref = 20.0

    # Create a batch of parameters
    batch_size = 5
    theta_batch = jnp.tile(bbh_aligned_params, (batch_size, 1))

    waveform_vmapped = jax.vmap(lambda theta: gen_IMRPhenomD_hphc(fs, theta, f_ref))
    hp_batch, hc_batch = waveform_vmapped(theta_batch)

    assert hp_batch.shape == (batch_size, len(fs))
    assert hc_batch.shape == (batch_size, len(fs))
    assert jnp.all(jnp.isfinite(hp_batch))
    assert jnp.all(jnp.isfinite(hc_batch))


def test_imrphenomd_grad(test_freq_grid, bbh_aligned_params):
    """Test that IMRPhenomD is differentiable."""
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc

    fs = test_freq_grid
    f_ref = 20.0

    def waveform_real(theta):
        hp, _ = gen_IMRPhenomD_hphc(fs, theta, f_ref)
        return jnp.sum(jnp.abs(hp) ** 2)

    # Just check that gradient computation doesn't error
    grad_fn = jax.grad(waveform_real)
    grad = grad_fn(bbh_aligned_params)
    assert grad.shape == bbh_aligned_params.shape
    assert jnp.all(jnp.isfinite(grad))


# ============================================================================
# Test IMRPhenomXAS (aligned-spin BBH, newer)
# ============================================================================


def test_imrphenomxas_basic(test_freq_grid, bbh_aligned_params):
    """Test IMRPhenomXAS waveform generation."""
    from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc

    fs = test_freq_grid
    f_ref = 20.0
    hp, hc = gen_IMRPhenomXAS_hphc(fs, bbh_aligned_params, f_ref)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenomxas_jit(test_freq_grid, bbh_aligned_params):
    """Test that IMRPhenomXAS works with JIT compilation."""
    from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc

    fs = test_freq_grid
    f_ref = 20.0

    @jax.jit
    def waveform_jitted(theta):
        return gen_IMRPhenomXAS_hphc(fs, theta, f_ref)

    hp, hc = waveform_jitted(bbh_aligned_params)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenomxas_vmap(test_freq_grid, bbh_aligned_params):
    """Test that IMRPhenomXAS works with vmap."""
    from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc

    fs = test_freq_grid
    f_ref = 20.0

    batch_size = 5
    theta_batch = jnp.tile(bbh_aligned_params, (batch_size, 1))

    waveform_vmapped = jax.vmap(lambda theta: gen_IMRPhenomXAS_hphc(fs, theta, f_ref))
    hp_batch, hc_batch = waveform_vmapped(theta_batch)

    assert hp_batch.shape == (batch_size, len(fs))
    assert hc_batch.shape == (batch_size, len(fs))
    assert jnp.all(jnp.isfinite(hp_batch))
    assert jnp.all(jnp.isfinite(hc_batch))


# ============================================================================
# Test IMRPhenomPv2 (precessing BBH)
# ============================================================================


def test_imrphenompv2_basic(test_freq_grid, bbh_precessing_params):
    """Test IMRPhenomPv2 waveform generation."""
    from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2

    fs = test_freq_grid
    f_ref = 20.0
    hp, hc = gen_IMRPhenomPv2(fs, bbh_precessing_params, f_ref)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenompv2_jit(test_freq_grid, bbh_precessing_params):
    """Test that IMRPhenomPv2 works with JIT compilation."""
    from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2

    fs = test_freq_grid
    f_ref = 20.0

    @jax.jit
    def waveform_jitted(theta):
        return gen_IMRPhenomPv2(fs, theta, f_ref)

    hp, hc = waveform_jitted(bbh_precessing_params)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenompv2_vmap(test_freq_grid, bbh_precessing_params):
    """Test that IMRPhenomPv2 works with vmap."""
    from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2

    fs = test_freq_grid
    f_ref = 20.0

    batch_size = 5
    theta_batch = jnp.tile(bbh_precessing_params, (batch_size, 1))

    waveform_vmapped = jax.vmap(lambda theta: gen_IMRPhenomPv2(fs, theta, f_ref))
    hp_batch, hc_batch = waveform_vmapped(theta_batch)

    assert hp_batch.shape == (batch_size, len(fs))
    assert hc_batch.shape == (batch_size, len(fs))
    assert jnp.all(jnp.isfinite(hp_batch))
    assert jnp.all(jnp.isfinite(hc_batch))


# ============================================================================
# Test IMRPhenomD_NRTidalv2 (BNS with tidal)
# ============================================================================


def test_imrphenomd_nrtidalv2_basic(test_freq_grid, bns_tidal_params):
    """Test IMRPhenomD_NRTidalv2 waveform generation."""
    from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc

    fs = test_freq_grid
    f_ref = 20.0
    hp, hc = gen_IMRPhenomD_NRTidalv2_hphc(fs, bns_tidal_params, f_ref)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenomd_nrtidalv2_jit(test_freq_grid, bns_tidal_params):
    """Test that IMRPhenomD_NRTidalv2 works with JIT compilation."""
    from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc

    fs = test_freq_grid
    f_ref = 20.0

    @jax.jit
    def waveform_jitted(theta):
        return gen_IMRPhenomD_NRTidalv2_hphc(fs, theta, f_ref)

    hp, hc = waveform_jitted(bns_tidal_params)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenomd_nrtidalv2_vmap(test_freq_grid, bns_tidal_params):
    """Test that IMRPhenomD_NRTidalv2 works with vmap."""
    from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc

    fs = test_freq_grid
    f_ref = 20.0

    batch_size = 5
    theta_batch = jnp.tile(bns_tidal_params, (batch_size, 1))

    waveform_vmapped = jax.vmap(
        lambda theta: gen_IMRPhenomD_NRTidalv2_hphc(fs, theta, f_ref)
    )
    hp_batch, hc_batch = waveform_vmapped(theta_batch)

    assert hp_batch.shape == (batch_size, len(fs))
    assert hc_batch.shape == (batch_size, len(fs))
    assert jnp.all(jnp.isfinite(hp_batch))
    assert jnp.all(jnp.isfinite(hc_batch))


# ============================================================================
# Test IMRPhenomXAS_NRTidalv3 (BNS with tidal, newer)
# ============================================================================


def test_imrphenomxas_nrtidalv3_basic(test_freq_grid, bns_tidal_params):
    """Test IMRPhenomXAS_NRTidalv3 waveform generation."""
    from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc

    fs = test_freq_grid
    f_ref = 20.0
    hp, hc = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, bns_tidal_params, f_ref)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenomxas_nrtidalv3_jit(test_freq_grid, bns_tidal_params):
    """Test that IMRPhenomXAS_NRTidalv3 works with JIT compilation."""
    from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc

    fs = test_freq_grid
    f_ref = 20.0

    @jax.jit
    def waveform_jitted(theta):
        return gen_IMRPhenomXAS_NRTidalv3_hphc(fs, theta, f_ref)

    hp, hc = waveform_jitted(bns_tidal_params)
    assert_waveform_valid(hp, hc, fs)


def test_imrphenomxas_nrtidalv3_vmap(test_freq_grid, bns_tidal_params):
    """Test that IMRPhenomXAS_NRTidalv3 works with vmap."""
    from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc

    fs = test_freq_grid
    f_ref = 20.0

    batch_size = 5
    theta_batch = jnp.tile(bns_tidal_params, (batch_size, 1))

    waveform_vmapped = jax.vmap(
        lambda theta: gen_IMRPhenomXAS_NRTidalv3_hphc(fs, theta, f_ref)
    )
    hp_batch, hc_batch = waveform_vmapped(theta_batch)

    assert hp_batch.shape == (batch_size, len(fs))
    assert hc_batch.shape == (batch_size, len(fs))
    assert jnp.all(jnp.isfinite(hp_batch))
    assert jnp.all(jnp.isfinite(hc_batch))


# ============================================================================
# Test TaylorF2 (PN inspiral with tidal)
# ============================================================================


def test_taylorf2_basic(test_freq_grid, bns_tidal_params):
    """Test TaylorF2 waveform generation."""
    from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc

    fs = test_freq_grid
    f_ref = 20.0
    hp, hc = gen_TaylorF2_hphc(fs, bns_tidal_params, f_ref)
    assert_waveform_valid(hp, hc, fs)


def test_taylorf2_jit(test_freq_grid, bns_tidal_params):
    """Test that TaylorF2 works with JIT compilation."""
    from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc

    fs = test_freq_grid
    f_ref = 20.0

    @jax.jit
    def waveform_jitted(theta):
        return gen_TaylorF2_hphc(fs, theta, f_ref)

    hp, hc = waveform_jitted(bns_tidal_params)
    assert_waveform_valid(hp, hc, fs)


def test_taylorf2_vmap(test_freq_grid, bns_tidal_params):
    """Test that TaylorF2 works with vmap."""
    from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc

    fs = test_freq_grid
    f_ref = 20.0

    batch_size = 5
    theta_batch = jnp.tile(bns_tidal_params, (batch_size, 1))

    waveform_vmapped = jax.vmap(lambda theta: gen_TaylorF2_hphc(fs, theta, f_ref))
    hp_batch, hc_batch = waveform_vmapped(theta_batch)
    
    assert hp_batch.shape == (batch_size, len(fs))
    assert hc_batch.shape == (batch_size, len(fs))
    assert jnp.all(jnp.isfinite(hp_batch))
    assert jnp.all(jnp.isfinite(hc_batch))


# ============================================================================
# SineGaussian tests (time-domain burst)
# ============================================================================


def test_sinegaussian_basic(test_time_grid, sinegaussian_params):
    """Test SineGaussian waveform generation (time-domain)."""
    from ripplegw.waveforms.SineGaussian import gen_SineGaussian_hphc

    t = test_time_grid
    hp, hc = gen_SineGaussian_hphc(t, sinegaussian_params)

    assert hp.shape == t.shape, f"hp shape {hp.shape} != t shape {t.shape}"
    assert hc.shape == t.shape, f"hc shape {hc.shape} != t shape {t.shape}"
    assert jnp.all(jnp.isfinite(hp)), "hp contains NaN or Inf"
    assert jnp.all(jnp.isfinite(hc)), "hc contains NaN or Inf"
    # SineGaussian returns real-valued waveforms (time-domain)
    assert not jnp.iscomplexobj(hp), "hp should be real-valued for time-domain"
    assert not jnp.iscomplexobj(hc), "hc should be real-valued for time-domain"


def test_sinegaussian_jit(test_time_grid, sinegaussian_params):
    """Test that SineGaussian works with JIT compilation."""
    from ripplegw.waveforms.SineGaussian import gen_SineGaussian_hphc

    t = test_time_grid

    @jax.jit
    def waveform_jitted(theta):
        return gen_SineGaussian_hphc(t, theta)

    hp, hc = waveform_jitted(sinegaussian_params)
    assert hp.shape == t.shape
    assert hc.shape == t.shape
    assert jnp.all(jnp.isfinite(hp))
    assert jnp.all(jnp.isfinite(hc))


def test_sinegaussian_vmap(test_time_grid, sinegaussian_params):
    """Test that SineGaussian works with vmap."""
    from ripplegw.waveforms.SineGaussian import gen_SineGaussian_hphc

    t = test_time_grid

    batch_size = 5
    theta_batch = jnp.tile(sinegaussian_params, (batch_size, 1))

    waveform_vmapped = jax.vmap(lambda theta: gen_SineGaussian_hphc(t, theta))
    hp_batch, hc_batch = waveform_vmapped(theta_batch)

    assert hp_batch.shape == (batch_size, len(t))
    assert hc_batch.shape == (batch_size, len(t))
    assert jnp.all(jnp.isfinite(hp_batch))
    assert jnp.all(jnp.isfinite(hc_batch))
    # SineGaussian is time-domain, so output should be real
    assert jnp.all(jnp.isreal(hp_batch))
    assert jnp.all(jnp.isreal(hc_batch))
