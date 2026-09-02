"""jit / vmap / grad, exercised across every registered waveform."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import ripplegw
from tests.helpers.config import default_config
from tests.helpers.grids import grid_for
from tests.helpers.params import canonical_params

ALL_WAVEFORMS = ripplegw.list_waveforms()

# Precessing / tidal models have the largest graphs and dominate this leg's
# runtime. Their jit and vmap paths are already exercised -- against LAL, not
# just for finiteness -- by the accuracy-smoke run on every main-targeted CI
# build (runner.generate_ripple_batch does jax.jit(jax.vmap(...))), so mark
# those two cases slow: they run on main and skip under -m "not slow".
_ACCURACY_COVERED = set(ripplegw.list_waveforms(is_precessing=True)) | set(
    ripplegw.list_waveforms(is_tidal=True)
)
_JIT_VMAP_WAVEFORMS = [
    pytest.param(name, marks=pytest.mark.slow) if name in _ACCURACY_COVERED else name
    for name in ALL_WAVEFORMS
]


def _batch(params: dict, batch_size: int) -> dict:
    return {k: jnp.full(batch_size, float(v)) for k, v in params.items()}


@pytest.mark.parametrize("name", _JIT_VMAP_WAVEFORMS)
def test_jit_matches_eager(name):
    wf = ripplegw.waveform(name, **default_config(name))
    axis = grid_for(name, small=True)
    params = canonical_params(wf)
    eager = wf(axis, params)
    jitted = jax.jit(wf)(axis, params)
    for key in eager:
        assert jnp.allclose(eager[key], jitted[key])


@pytest.mark.parametrize("name", _JIT_VMAP_WAVEFORMS)
def test_vmap_over_params(name, compiled_model):
    jitted, wf = compiled_model(name)
    axis = grid_for(name, small=True)
    params = _batch(canonical_params(wf), batch_size=4)
    out = jax.vmap(lambda p: jitted(axis, p))(params)
    assert out["p"].shape == (4, *axis.shape)
    assert jnp.all(jnp.isfinite(out["p"]))
    assert jnp.all(jnp.isfinite(out["c"]))


@pytest.mark.parametrize("name", ALL_WAVEFORMS)
def test_grad_is_finite(name, compiled_model):
    """Differentiate a scalar summary of the strain w.r.t. every float parameter."""
    _, wf = compiled_model(name)
    axis = grid_for(name, small=True)
    params = canonical_params(wf)

    def loss(p):
        out = wf(axis, p)
        return jnp.sum(jnp.abs(out["p"]) ** 2 + jnp.abs(out["c"]) ** 2)

    grads = jax.grad(loss)(params)
    for key, g in grads.items():
        assert jnp.isfinite(g), f"grad w.r.t. {key!r} is non-finite for {name}"
