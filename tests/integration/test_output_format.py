"""Every registered waveform: output shape/dtype/keys, repr, registry roundtrip.

Parametrized directly off ``ripplegw.list_waveforms()`` -- a newly registered
model is covered the moment it appears, with no test-file edits.
"""

import inspect

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import ripplegw
from tests.helpers.config import default_config
from tests.helpers.grids import grid_for
from tests.helpers.params import canonical_params

ALL_WAVEFORMS = ripplegw.list_waveforms()


def _accepts(name: str, param: str) -> bool:
    cls = ripplegw.WAVEFORM_REGISTRY[name]
    return param in inspect.signature(cls.__init__).parameters


# TaylorF2 / IMRPhenomD_NRTidalv2 / IMRPhenomXAS_NRTidalv3 expose an alternate
# parameter_names ordering behind this flag; both orderings must be checked.
TILDE_WAVEFORMS = [n for n in ALL_WAVEFORMS if _accepts(n, "use_lambda_tildes")]


def _assert_valid_output(out, axis, domain: str):
    assert set(out) == {"p", "c"}
    for key in ("p", "c"):
        val = out[key]
        assert val.shape == axis.shape, (
            f"{key} shape {val.shape} != axis shape {axis.shape}"
        )
        assert jnp.all(jnp.isfinite(val)), f"{key} contains NaN/Inf"
        if domain == "FD":
            assert jnp.iscomplexobj(val), (
                f"{key} should be complex-valued for an FD model"
            )
        else:
            assert not jnp.iscomplexobj(val), (
                f"{key} should be real-valued for a TD model"
            )


@pytest.mark.parametrize("name", ALL_WAVEFORMS)
def test_output_contract(name, compiled_model):
    call, wf = compiled_model(name)
    axis = grid_for(name, small=True)
    out = call(axis, canonical_params(wf))
    _assert_valid_output(out, axis, wf.waveform_metadata["domain"])


@pytest.mark.parametrize("name", TILDE_WAVEFORMS)
def test_output_contract_lambda_tildes(name, compiled_model):
    call, wf = compiled_model(name, use_lambda_tildes=True)
    axis = grid_for(name, small=True)
    out = call(axis, canonical_params(wf))
    _assert_valid_output(out, axis, wf.waveform_metadata["domain"])


@pytest.mark.parametrize("name", ALL_WAVEFORMS)
def test_parameter_names_are_unique_and_nonempty(name, compiled_model):
    _, wf = compiled_model(name)
    assert wf.parameter_names
    assert len(set(wf.parameter_names)) == len(wf.parameter_names)


@pytest.mark.parametrize("name", ALL_WAVEFORMS)
def test_in_registry(name):
    cls = ripplegw.WAVEFORM_REGISTRY[name]
    assert isinstance(ripplegw.waveform(name, **default_config(name)), cls)


@pytest.mark.parametrize("name", ALL_WAVEFORMS)
def test_repr_names_the_class(name, compiled_model):
    _, wf = compiled_model(name)
    assert type(wf).__name__ in repr(wf)


def test_registry_is_nonempty():
    assert ALL_WAVEFORMS, "no waveforms registered"
