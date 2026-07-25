"""Optional per-model behavior: the amplitude/phase split, and distance scaling.

Parametrized by ``issubclass`` against ``ripplegw.interfaces``, not by name --
any future model that implements one of these interfaces is covered
automatically.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import ripplegw
from ripplegw.interfaces import (
    AmplitudePhaseWaveform,
    DistanceScaledWaveform,
)
from tests.helpers.grids import grid_for
from tests.helpers.params import canonical_params

ALL_WAVEFORMS = ripplegw.list_waveforms()
AMP_PHASE_WAVEFORMS = [
    n
    for n in ALL_WAVEFORMS
    if issubclass(ripplegw.WAVEFORM_REGISTRY[n], AmplitudePhaseWaveform)
]
DISTANCE_SCALED_WAVEFORMS = [
    n
    for n in ALL_WAVEFORMS
    if issubclass(ripplegw.WAVEFORM_REGISTRY[n], DistanceScaledWaveform)
]


@pytest.mark.parametrize("name", AMP_PHASE_WAVEFORMS)
def test_strain_equals_amplitude_times_exp_i_phase(name, compiled_model):
    amplitude, wf = compiled_model(name, method="amplitude")
    phase, _ = compiled_model(name, method="phase")
    strain, _ = compiled_model(name, method="strain")
    axis = grid_for(name, small=True)
    params = canonical_params(wf)
    amp = amplitude(axis, params)
    ph = phase(axis, params)
    assert jnp.all(jnp.isfinite(amp))
    assert jnp.all(jnp.isfinite(ph))
    assert jnp.allclose(strain(axis, params), amp * jnp.exp(1j * ph))


@pytest.mark.parametrize("name", DISTANCE_SCALED_WAVEFORMS)
def test_at_unit_distance_matches_dl_equals_one(name, compiled_model):
    """``at_unit_distance`` must equal the ``d_L=1.0`` call exactly, not just
    to floating-point tolerance -- so this runs both sides eager (uncompiled).
    Two independently jit-compiled functions computing the same math are not
    guaranteed to be bit-identical, so comparing jitted output here would
    test XLA's compilation determinism rather than this guarantee.
    """
    _, wf = compiled_model(name)
    axis = grid_for(name, small=True)
    params = canonical_params(wf, d_L=250.0)
    at_unit = wf.at_unit_distance(axis, params)
    direct = wf(axis, {**params, "d_L": 1.0})
    assert jnp.array_equal(at_unit["p"], direct["p"])
    assert jnp.array_equal(at_unit["c"], direct["c"])


@pytest.mark.parametrize("name", DISTANCE_SCALED_WAVEFORMS)
def test_distance_scaling_is_inverse_linear(name, compiled_model):
    """``__call__(d_L=D) == at_unit_distance(...) / D``, to floating-point precision."""
    at_unit_distance, wf = compiled_model(name, method="at_unit_distance")
    call, _ = compiled_model(name)
    axis = grid_for(name, small=True)
    params = canonical_params(wf, d_L=250.0)
    at_unit = at_unit_distance(axis, params)
    scaled = call(axis, params)
    for key in ("p", "c"):
        assert jnp.allclose(scaled[key], at_unit[key] / 250.0, rtol=1e-6)
