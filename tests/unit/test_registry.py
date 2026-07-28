"""Tests for the generic top-level waveform API (registry + factory).

These exercise the family-agnostic surface — :func:`ripplegw.waveform`,
:func:`ripplegw.list_waveforms`, :func:`ripplegw.register` — and, crucially,
prove the API is future-proof for *any* waveform, not just the built-in Phenom
CBC models: a toy non-CBC waveform with novel parameters and non-``{"p","c"}``
polarizations round-trips through the same top-level entry point.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import ripplegw
import ripplegw.registry as reg
from ripplegw.interfaces import Waveform

BUILTINS = {
    "TaylorF2",
    "IMRPhenomD",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomHM",
    "IMRPhenomPv2",
    "IMRPhenomXAS",
    "IMRPhenomXAS_NRTidalv3",
    "IMRPhenomXHM",
    "IMRPhenomXP",
    "IMRPhenomXPHM",
    "SineGaussian",
}


@pytest.fixture
def registry_sandbox():
    """Snapshot and restore the global registry so tests that register a
    custom waveform do not leak state into other tests."""
    saved = dict(reg.WAVEFORM_REGISTRY)
    try:
        yield reg
    finally:
        reg.WAVEFORM_REGISTRY.clear()
        reg.WAVEFORM_REGISTRY.update(saved)


# --- a toy, deliberately non-CBC waveform used to prove genericity -----------


class _ToyBurst(Waveform):
    """A Gaussian-windowed sinusoid with its own parameters/config."""

    domain = "TD"

    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate

    @property
    def parameter_names(self):
        return ("f0", "width")

    def __call__(self, t, params):
        env = jnp.exp(-(t**2) / (2.0 * params["width"] ** 2))
        phase = 2.0 * jnp.pi * params["f0"] * t
        return {"p": env * jnp.cos(phase), "c": env * jnp.sin(phase)}


class _ScalarBreathing(Waveform):
    """A model whose output uses a non-``{p,c}`` polarization key."""

    domain = "TD"

    @property
    def parameter_names(self):
        return ("amp",)

    def __call__(self, t, params):
        return {"b": params["amp"] * jnp.ones_like(t)}


# --- built-ins / listing -----------------------------------------------------


def test_builtins_registered():
    names = set(ripplegw.list_waveforms())
    assert BUILTINS <= names


def test_list_waveforms_filter_domain():
    fd = set(ripplegw.list_waveforms(domain="FD"))
    assert "IMRPhenomXAS" in fd
    assert "SineGaussian" not in fd  # SineGaussian is time-domain
    assert set(ripplegw.list_waveforms(domain="TD")) >= {"SineGaussian"}


def test_list_waveforms_filter_metadata():
    prec = set(ripplegw.list_waveforms(is_precessing=True))
    assert prec == {"IMRPhenomPv2", "IMRPhenomXP", "IMRPhenomXPHM"}
    assert "IMRPhenomD" in ripplegw.list_waveforms(is_tidal=False)


# --- factory -----------------------------------------------------------------


def test_factory_constructs_and_matches_direct():
    wf = ripplegw.waveform("IMRPhenomD", f_ref=20.0)
    cls = reg.WAVEFORM_REGISTRY["IMRPhenomD"]
    assert isinstance(wf, cls)
    assert wf.f_ref == 20.0

    f = jnp.arange(20.0, 60.0, 1.0)
    params = {
        "M_c": 30.0,
        "eta": 0.24,
        "s1_z": 0.1,
        "s2_z": -0.1,
        "d_L": 400.0,
        "phase_c": 0.0,
        "iota": 0.3,
    }
    got = wf(f, params)
    ref = cls(f_ref=20.0)(f, params)
    assert jnp.array_equal(got["p"], ref["p"])
    assert jnp.array_equal(got["c"], ref["c"])


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown waveform 'Nope'"):
        ripplegw.waveform("Nope")


# --- register decorator ------------------------------------------------------


def test_register_and_use_custom_waveform(registry_sandbox):
    @ripplegw.register("ToyBurst", domain="TD", is_burst=True)
    class ToyBurst(_ToyBurst):
        pass

    assert "ToyBurst" in ripplegw.list_waveforms()
    assert ripplegw.list_waveforms(is_burst=True) == ["ToyBurst"]

    wf = ripplegw.waveform("ToyBurst", sampling_rate=16.0)
    assert wf.sampling_rate == 16.0
    t = jnp.linspace(-1.0, 1.0, 64)
    out = wf(t, {"f0": 5.0, "width": 0.2})
    assert set(out) == {"p", "c"}
    assert out["p"].shape == t.shape
    assert bool(jnp.all(jnp.isfinite(out["p"])))


def test_register_allows_nonstandard_polarizations(registry_sandbox):
    ripplegw.register("ScalarBreathing")(_ScalarBreathing)
    wf = ripplegw.waveform("ScalarBreathing")
    out = wf(jnp.arange(5.0), {"amp": 2.0})
    assert set(out) == {"b"}  # generic: not tied to {"p","c"}


def test_register_duplicate_raises_without_override(registry_sandbox):
    ripplegw.register("Dup")(_ToyBurst)
    with pytest.raises(ValueError, match="already registered"):
        ripplegw.register("Dup")(_ScalarBreathing)


def test_register_override(registry_sandbox):
    ripplegw.register("Dup")(_ToyBurst)
    ripplegw.register("Dup", override=True)(_ScalarBreathing)
    assert reg.WAVEFORM_REGISTRY["Dup"] is _ScalarBreathing


def test_register_rejects_non_waveform(registry_sandbox):
    class NotAWaveform:
        pass

    with pytest.raises(TypeError, match="subclasses of"):
        ripplegw.register("Bad")(NotAWaveform)


# --- metadata --------------------------------------------------------------


def test_get_waveform_metadata():
    md = ripplegw.get_waveform_metadata("IMRPhenomXP")
    assert md["domain"] == "FD"
    assert md["is_precessing"] is True
    # returns a copy; mutating it must not affect the registry
    md["is_precessing"] = False
    assert ripplegw.get_waveform_metadata("IMRPhenomXP")["is_precessing"] is True


def test_get_waveform_metadata_unknown_raises():
    with pytest.raises(ValueError, match="Unknown waveform"):
        ripplegw.get_waveform_metadata("Nope")


def test_metadata_does_not_clobber_real_attributes(registry_sandbox):
    # A metadata key colliding with a real API member must not overwrite it.
    @ripplegw.register("Sneaky", parameter_names="oops", domain="TD")
    class Sneaky(_ToyBurst):
        pass

    wf = ripplegw.waveform("Sneaky")
    assert wf.parameter_names == ("f0", "width")  # real property intact
    assert ripplegw.get_waveform_metadata("Sneaky")["parameter_names"] == "oops"


# --- source_type is inferred from module path, like domain from base class --


@pytest.mark.parametrize("name", sorted(BUILTINS - {"SineGaussian"}))
def test_cbc_source_type_inferred(name):
    assert ripplegw.get_waveform_metadata(name)["source_type"] == "cbc"


def test_burst_source_type_inferred():
    assert ripplegw.get_waveform_metadata("SineGaussian")["source_type"] == "burst"


def test_source_type_absent_outside_ripplegw_waveforms(registry_sandbox):
    # _ToyBurst lives in this test module, not under ripplegw.waveforms.<type> --
    # inference has nothing to key off, so it's simply absent, not an error.
    ripplegw.register("ToyNoSourceType")(_ToyBurst)
    assert "source_type" not in ripplegw.get_waveform_metadata("ToyNoSourceType")


def test_explicit_source_type_overrides_inference(registry_sandbox):
    @ripplegw.register("ToyExplicitSourceType", source_type="CW")
    class ToyExplicit(_ToyBurst):
        pass

    assert (
        ripplegw.get_waveform_metadata("ToyExplicitSourceType")["source_type"] == "CW"
    )


# --- no hard-coded family list at the top level -----------------------------


def test_top_level_unknown_attr_raises():
    with pytest.raises(AttributeError):
        ripplegw.DefinitelyNotAWaveform  # noqa: B018 - attribute access is the point of this test


# --- ABC requirements --------------------------------------------------------


def test_waveform_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        Waveform()  # abstract: parameter_names / __call__ not implemented
