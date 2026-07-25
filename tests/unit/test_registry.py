"""Tests for the generic top-level waveform API (registry + factory).

These exercise the family-agnostic surface — :func:`ripplegw.waveform`,
:func:`ripplegw.list_waveforms`, :func:`ripplegw.register`, and entry-point
plugin discovery — and, crucially, prove the API is future-proof for *any*
waveform, not just the built-in Phenom CBC models: a toy non-CBC waveform with
novel parameters and non-``{"p","c"}`` polarizations round-trips through the
same top-level entry point.
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
    """Snapshot and restore the global registry so tests that register or
    discover plugins do not leak state into other tests."""
    saved = dict(reg.WAVEFORM_REGISTRY)
    saved_flag = reg._PLUGINS_LOADED
    try:
        yield reg
    finally:
        reg.WAVEFORM_REGISTRY.clear()
        reg.WAVEFORM_REGISTRY.update(saved)
        reg._PLUGINS_LOADED = saved_flag


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


# --- entry-point plugin discovery -------------------------------------------


class _FakeEP:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def _patch_entry_points(monkeypatch, eps):
    def fake_entry_points(*, group):
        assert group == "ripplegw.waveforms"
        return eps

    monkeypatch.setattr(reg, "entry_points", fake_entry_points)


def test_plugin_discovered_via_entry_point(registry_sandbox, monkeypatch):
    _patch_entry_points(monkeypatch, [_FakeEP("PluginToy", _ToyBurst)])
    reg._PLUGINS_LOADED = False

    # discovery happens transparently through the public factory
    wf = ripplegw.waveform("PluginToy", sampling_rate=8.0)
    assert isinstance(wf, _ToyBurst)
    assert "PluginToy" in ripplegw.list_waveforms()


def test_intree_registration_wins_over_plugin(registry_sandbox, monkeypatch):
    original_cls = reg.WAVEFORM_REGISTRY["IMRPhenomD"]
    sentinel = _ToyBurst  # a same-named plugin must NOT replace the builtin
    _patch_entry_points(monkeypatch, [_FakeEP("IMRPhenomD", sentinel)])
    reg._PLUGINS_LOADED = False
    reg.load_plugins()
    assert reg.WAVEFORM_REGISTRY["IMRPhenomD"] is original_cls


def test_bad_plugin_is_skipped_not_fatal(registry_sandbox, monkeypatch):
    # One broken plugin must not hide the valid ones, and must not poison
    # discovery for the rest of the process.
    _patch_entry_points(
        monkeypatch,
        [_FakeEP("BadPlugin", object), _FakeEP("GoodPlugin", _ToyBurst)],
    )
    reg._PLUGINS_LOADED = False
    with pytest.warns(RuntimeWarning, match="BadPlugin"):
        reg.load_plugins()
    assert "BadPlugin" not in reg.WAVEFORM_REGISTRY
    assert reg.WAVEFORM_REGISTRY.get("GoodPlugin") is _ToyBurst


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


# --- no hard-coded family list at the top level -----------------------------


def test_top_level_unknown_attr_raises():
    with pytest.raises(AttributeError):
        ripplegw.DefinitelyNotAWaveform  # noqa: B018 - attribute access is the point of this test


# --- ABC contract ------------------------------------------------------------


def test_waveform_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        Waveform()  # abstract: parameter_names / __call__ not implemented
