"""Physically motivated edge cases, parametrized by registry metadata rather
than by name -- a new tidal or precessing model is covered automatically.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import ripplegw
from ripplegw.conversions import ms_to_Mc_eta
from tests.helpers.config import default_config
from tests.helpers.grids import grid_for
from tests.helpers.params import canonical_params

ALL_WAVEFORMS = ripplegw.list_waveforms()
TIDAL_WAVEFORMS = ripplegw.list_waveforms(is_tidal=True)
PRECESSING_WAVEFORMS = ripplegw.list_waveforms(is_precessing=True)
# Positive tagging (not "every non-precessing model"): a family that never sets
# is_precessing either way (e.g. continuous-wave) has no concept of aligned vs
# precessing spins and must not be silently swept into either bucket.
ALIGNED_WAVEFORMS = ripplegw.list_waveforms(is_precessing=False)

# One instance per registered model, used only to inspect parameter_names for
# parametrization -- cheap (no waveform evaluation). Some families (e.g.
# continuous-wave) have no safe zero-argument default, hence default_config.
_INSTANCES = {n: ripplegw.waveform(n, **default_config(n)) for n in ALL_WAVEFORMS}
WITH_INCLINATION = [n for n in ALL_WAVEFORMS if "iota" in _INSTANCES[n].parameter_names]


def _assert_finite(out):
    assert jnp.all(jnp.isfinite(out["p"]))
    assert jnp.all(jnp.isfinite(out["c"]))


@pytest.mark.parametrize("name", ALIGNED_WAVEFORMS)
def test_equal_mass(name, compiled_model):
    """eta = 0.25 exactly: delta = 0, so odd-l higher modes vanish where present."""
    call, wf = compiled_model(name)
    Mc, eta = ms_to_Mc_eta(jnp.array([30.0, 30.0]))
    assert float(eta) == pytest.approx(0.25)
    params = canonical_params(wf, M_c=float(Mc), eta=float(eta))
    _assert_finite(call(grid_for(name, small=True), params))


@pytest.mark.parametrize("name", ALIGNED_WAVEFORMS)
def test_zero_aligned_spins(name, compiled_model):
    call, wf = compiled_model(name)
    params = canonical_params(wf, s1_z=0.0, s2_z=0.0)
    _assert_finite(call(grid_for(name, small=True), params))


@pytest.mark.parametrize("name", ALIGNED_WAVEFORMS)
def test_near_extremal_aligned_spin(name, compiled_model):
    call, wf = compiled_model(name)
    params = canonical_params(wf, s1_z=0.99, s2_z=-0.99)
    _assert_finite(call(grid_for(name, small=True), params))


@pytest.mark.parametrize("name", WITH_INCLINATION)
def test_face_on(name, compiled_model):
    """iota = 0: face-on, only the m=2 modes contribute where higher modes exist."""
    call, wf = compiled_model(name)
    params = canonical_params(wf, iota=0.0)
    _assert_finite(call(grid_for(name, small=True), params))


@pytest.mark.parametrize("name", WITH_INCLINATION)
def test_edge_on(name, compiled_model):
    """iota = pi/2: edge-on, all modes contribute where higher modes exist."""
    call, wf = compiled_model(name)
    params = canonical_params(wf, iota=jnp.pi / 2)
    _assert_finite(call(grid_for(name, small=True), params))


@pytest.mark.parametrize("name", TIDAL_WAVEFORMS)
def test_zero_tidal_deformability(name, compiled_model):
    """lambda_1 = lambda_2 = 0: BH-like tidal correction."""
    call, wf = compiled_model(name)
    params = canonical_params(wf, lambda_1=0.0, lambda_2=0.0)
    _assert_finite(call(grid_for(name, small=True), params))


@pytest.mark.parametrize("name", PRECESSING_WAVEFORMS)
def test_aligned_spins_only(name, compiled_model):
    """In-plane spins zero: a precessing model reduces to the aligned-spin limit."""
    call, wf = compiled_model(name)
    params = canonical_params(wf, s1_x=0.0, s1_y=0.0, s2_x=0.0, s2_y=0.0)
    _assert_finite(call(grid_for(name, small=True), params))


@pytest.mark.parametrize("name", PRECESSING_WAVEFORMS)
def test_fully_precessing(name, compiled_model):
    """Large in-plane spin components: strong precession regime."""
    call, wf = compiled_model(name)
    params = canonical_params(
        wf, s1_x=0.5, s1_y=0.5, s1_z=0.1, s2_x=-0.4, s2_y=0.3, s2_z=-0.1
    )
    _assert_finite(call(grid_for(name, small=True), params))


@pytest.mark.parametrize("name", ripplegw.list_waveforms(domain="TD"))
def test_burst_shape_parameter_boundaries(name, compiled_model):
    """Boundary values of any burst-style shape parameter this model exposes."""
    call, wf = compiled_model(name)
    axis = grid_for(name, small=True)
    boundary_values = {"e": (0.0, 1.0), "Q": (2.0,)}
    tested = False
    for param, values in boundary_values.items():
        if param not in wf.parameter_names:
            continue
        for value in values:
            tested = True
            _assert_finite(call(axis, canonical_params(wf, **{param: value})))
    if not tested:
        _assert_finite(call(axis, canonical_params(wf)))
